"""RAG pipeline: retrieve candidate jobs with hybrid search, then rerank with an LLM."""

import json
from pathlib import Path

from job_seeker.config import settings
from job_seeker.llm import extract_json_with_llm
from job_seeker.models import cv_to_text
from job_seeker.vector_db.search import search_jobs

__all__ = ["RERANK_PROMPT", "load_resume", "rerank_with_llm", "rag_search"]

RERANK_PROMPT = """You are a job search advisor. Below is the candidate profile and a list of retrieved job postings (already pre-ranked by a hybrid dense+BM25 retriever). Rerank the jobs from best to worst fit for the candidate and, for the top {top_n}, give a one-sentence reason why it fits.

Candidate profile:
{candidate}

Jobs:
{jobs}

Output ONLY a JSON array of objects with keys "rank" (integer, 1-based), "job_id", "company", "fit_score" (integer 0-100), and "reason" (one short sentence). Do not add markdown code fences and output raw JSON only.
"""


def load_resume(path: str | Path | None = None) -> dict:
    path = path or settings.cv_json_path
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rerank_with_llm(candidate: dict, jobs: list[dict], top_n: int = 5) -> list[dict]:
    profile = cv_to_text(candidate)
    jobs_block = "\n\n".join(
        f"{i + 1}. [{job['job_id']}] {job['company']} | {job['salary']} | "
        f"{job['working_location']} | Section: {job['matched_section']}\n{job['matched_text']}"
        for i, job in enumerate(jobs)
    )
    prompt = RERANK_PROMPT.format(
        candidate=profile,
        jobs=jobs_block,
        top_n=min(top_n, len(jobs)),
    )
    result = extract_json_with_llm(prompt)
    ranked = result if isinstance(result, list) else result.get("rankings", result.get("results", []))
    return ranked


def rag_search(
    query: str,
    top_k: int = 10,
    rerank: bool = True,
    resume: dict | None = None,
    resume_path: str | Path | None = None,
) -> dict:
    """Retrieve jobs with hybrid search and optionally rerank them with the LLM.

    ``query`` is matched against job postings; ``resume`` provides the candidate
    profile used for LLM reranking.
    """
    jobs = search_jobs(query, top_k=top_k)
    ranked = None
    if rerank:
        candidate = resume or load_resume(resume_path)
        ranked = rerank_with_llm(candidate, jobs, top_n=top_k)
    return {"query": query, "retrieved": jobs, "ranked": ranked}
