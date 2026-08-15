"""Interest-based job discovery.

Given a job seeker's free-text interests (and optionally their CV profile), rank
candidate jobs by how well each company's background and job description match,
using the LLM to pick the top ``top_k`` jobs with a one-line reason each.

Two entry points:
- :func:`recommend_jobs`        — match on the interest text only.
- :func:`recommend_jobs_for_cv` — match on the interest text + candidate CV.
"""

import json
from pathlib import Path

from job_seeker.config import settings
from job_seeker.llm import extract_json_with_llm
from job_seeker.models import cv_to_text

__all__ = ["recommend_jobs", "recommend_jobs_for_cv"]


RECOMMEND_PROMPT = """You are a job discovery advisor. A job seeker described what they are interested in, and you are given a list of candidate job postings (each summarized with its company background). Select the top {n} jobs the seeker would be most interested in, best fit first.

Seeker interest:
{interest}

{profile_block}Candidate jobs:
{jobs_block}

Output ONLY a JSON object with exactly this schema. Do NOT add markdown code fences and output raw JSON only.

Schema:
{{
  "recommendations": [
    {{"job_id": null, "company": null, "job_title": null, "salary": null, "location": null, "reason": null}}
  ]
}}

Rules:
- "recommendations": an array of exactly {n} objects, best fit first.
- "job_title": the role/title of the job (from its section/JD when present).
- "salary" and "location": copy from the job summary (use null if absent).
- "reason": ONE sentence explaining why this job fits, referencing the company's background/focus AND the seeker's stated interest.
"""


def _load_raw_jobs() -> list[dict]:
    from job_seeker.vector_db.ingest import load_jobs_dir

    jobs = load_jobs_dir()
    if jobs:
        return jobs
    path = settings.jobsdb_output_file
    if not Path(path).is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _has_qdrant_points(collection: str | None) -> bool:
    try:
        from job_seeker.vector_db.qdrant import collection_info

        info = collection_info(collection)
        return bool(info.get("exists") and info.get("points_count", 0) > 0)
    except Exception:
        return False


def _expand(interest: str) -> str:
    try:
        from job_seeker.query_expansion import expand_query

        return expand_query(interest)
    except Exception:
        return interest


def _keyword_score(text: str, keywords: list[str]) -> int:
    low = text.lower()
    return sum(1 for kw in keywords if kw and kw.lower() in low)


def _job_text(job: dict) -> str:
    parts = [str(job.get("company", ""))]
    for key in ("Responsibilities", "Requirements"):
        val = job.get(key, "") or ""
        parts.append(" ".join(val) if isinstance(val, list) else str(val))
    return " ".join(parts)


def _build_pool(interest: str, pool: int, collection: str | None) -> list[dict]:
    """Return up to ``pool`` normalized candidate job dicts.

    Prefers Qdrant hybrid search when the collection is populated; otherwise
    falls back to a keyword-overlap prefilter over the raw crawled jobs JSON.
    """
    if _has_qdrant_points(collection):
        from job_seeker.vector_db.search import search_jobs_expanded

        jobs, _ = search_jobs_expanded(interest, top_k=pool, collection=collection)
        return jobs

    raw = _load_raw_jobs()
    if not raw:
        return []

    expanded = _expand(interest)
    keywords = [w for w in expanded.lower().split() if len(w) > 2]
    scored = [(_keyword_score(_job_text(j), keywords), j) for j in raw]
    scored.sort(key=lambda x: x[0], reverse=True)

    normalized: list[dict] = []
    for _, job in scored[:pool]:
        resp = job.get("Responsibilities", "") or ""
        req = job.get("Requirements", "") or ""
        if isinstance(resp, list):
            resp = " ".join(resp)
        if isinstance(req, list):
            req = " ".join(req)
        normalized.append(
            {
                "job_id": job.get("job_id"),
                "job_url": job.get("job_url"),
                "company": job.get("company"),
                "salary": job.get("salary"),
                "working_location": job.get("working_location"),
                "matched_section": None,
                "matched_text": f"{resp}\n{req}".strip(),
                "score": 0.0,
            }
        )
    return normalized


def _company_context(job: dict) -> str:
    parts = [f"Company: {job.get('company') or 'N/A'}"]
    title = job.get("matched_section") or job.get("job_title") or "Unknown role"
    parts.append(f"Role/section: {title}")
    if job.get("working_location"):
        parts.append(f"Location: {job.get('working_location')}")
    if job.get("salary"):
        parts.append(f"Salary: {job.get('salary')}")
    text = (job.get("matched_text") or "").strip()
    if text:
        parts.append(f"JD excerpt: {text[:400]}")
    return " | ".join(parts)


def _cv_profile_text(cv: dict) -> str:
    return cv_to_text(cv)


def _lookup_raw_job(job_id: str) -> dict | None:
    """Return the full crawled job dict for ``job_id`` from the raw jobs JSON."""
    target = str(job_id).strip()
    if not target:
        return None
    for job in _load_raw_jobs():
        if str(job.get("job_id", "")).strip() == target:
            return job
    return None


def _enrich_recs(recs: list[dict], jobs: list[dict]) -> list[dict]:
    """Attach the full crawled job data (URL, responsibilities, requirements) to each recommendation.

    The LLM only sees summarized job context and cannot produce the URL or the
    full JD, so they are joined back from the candidate pool by ``job_id``
    (falling back to company + location). Responsibilities and Requirements come
    from the raw crawled jobs JSON.
    """
    by_id = {str(job.get("job_id")): job for job in jobs if job.get("job_id")}
    by_company_location = {}
    for job in jobs:
        key = (
            str(job.get("company") or "").strip().lower(),
            str(job.get("working_location") or job.get("location") or "").strip().lower(),
        )
        by_company_location.setdefault(key, job)

    enriched: list[dict] = []
    for rec in recs:
        job = by_id.get(str(rec.get("job_id"))) if rec.get("job_id") else None
        if job is None:
            key = (
                str(rec.get("company") or "").strip().lower(),
                str(rec.get("location") or "").strip().lower(),
            )
            job = by_company_location.get(key)
        rec = {**rec}
        if job is not None:
            if not rec.get("job_url"):
                rec["job_url"] = job.get("job_url")
            if not rec.get("working_location"):
                rec["working_location"] = job.get("working_location")
        raw = _lookup_raw_job(rec.get("job_id")) if rec.get("job_id") else None
        if raw is None:
            raw = job
        if raw is not None:
            if not rec.get("Responsibilities"):
                rec["Responsibilities"] = raw.get("Responsibilities")
            if not rec.get("Requirements"):
                rec["Requirements"] = raw.get("Requirements")
        enriched.append(rec)
    return enriched


def _run_recommend(
    interest: str,
    profile_text: str,
    pool: int,
    top_k: int,
    collection: str | None,
) -> list[dict]:
    jobs = _build_pool(interest, pool, collection)
    if not jobs:
        return []

    n = min(top_k, len(jobs))
    jobs_block = "\n\n".join(
        f"{i + 1}. {_company_context(j)}" for i, j in enumerate(jobs)
    )
    profile_block = f"Candidate profile:\n{profile_text}\n\n" if profile_text else ""
    prompt = RECOMMEND_PROMPT.format(
        n=n,
        interest=interest,
        profile_block=profile_block,
        jobs_block=jobs_block,
    )
    raw = extract_json_with_llm(prompt)
    if isinstance(raw, dict):
        recs = raw.get("recommendations") or []
    elif isinstance(raw, list):
        recs = raw
    else:
        recs = []
    return _enrich_recs(recs, jobs)


def recommend_jobs(
    interest: str,
    top_k: int = 5,
    pool: int = 15,
    collection: str | None = None,
) -> dict:
    """Recommend the top ``top_k`` jobs matching the seeker's interest text only."""
    recs = _run_recommend(interest, "", pool, top_k, collection)
    return {"interest": interest, "recommendations": recs}


def recommend_jobs_for_cv(
    interest: str,
    cv: dict,
    top_k: int = 5,
    pool: int = 15,
    collection: str | None = None,
) -> dict:
    """Recommend jobs matching both the seeker's interest text and their CV profile."""
    profile = _cv_profile_text(cv) if cv else ""
    recs = _run_recommend(interest, profile, pool, top_k, collection)
    return {"interest": interest, "recommendations": recs}
