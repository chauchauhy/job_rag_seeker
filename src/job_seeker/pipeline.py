"""End-to-end Resume-to-Job matching pipeline.

Loads a structured CV (data/raw/cv.json), ingests crawled job ads into a Qdrant
collection, retrieves the most semantically relevant jobs, and evaluates each
match with the LLM (opencode CLI), producing a formatted console report.

Domain models live in :mod:`job_seeker.models`; this module only orchestrates.
"""

import json
import sys
from pathlib import Path

from job_seeker.config import settings
from job_seeker.llm import extract_json_with_llm
from job_seeker.models import ExtractedCV, JobMatchReport
from job_seeker.resume import pdf_to_markdown
from job_seeker.vector_db.ingest import ingest_jobs
from job_seeker.vector_db.qdrant import close_client, ensure_collection, get_client
from job_seeker.vector_db.search import search_jobs

__all__ = [
    "JOBS_COLLECTION",
    "DEFAULT_CV_PATH",
    "load_cv",
    "parse_pdf_to_text",
    "extract_cv_features",
    "ensure_job_collection",
    "build_profile_query",
    "retrieve_matching_jobs",
    "job_ad_to_text",
    "evaluate_job_match",
    "main",
]

JOBS_COLLECTION = settings.qdrant_collection
DEFAULT_CV_PATH = settings.cv_json_path
DEFAULT_JOBS_PATH = settings.jobsdb_output_file


def load_cv(path: str | Path | None = None) -> ExtractedCV:
    """Load and validate the structured CV from a JSON file."""
    path = Path(path or DEFAULT_CV_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"CV JSON not found: {path}")
    with open(path, encoding="utf-8") as f:
        return ExtractedCV.model_validate(json.load(f))


def parse_pdf_to_text(pdf_path: str | Path) -> str:
    """Extract raw text from a resume PDF via the existing MarkItDown pipeline."""
    return pdf_to_markdown(pdf_path)


def extract_cv_features(cv_text: str) -> ExtractedCV:
    """(Optional) Extract structured CV features from raw resume text via LLM.

    For the standard flow, prefer :func:`load_cv` which reads the already
    extracted data/raw/cv.json.
    """
    from job_seeker.resume import extract_resume

    return ExtractedCV.model_validate(extract_resume(cv_text))


def ensure_job_collection(
    jobs_path: str | Path | None = None,
    collection: str = JOBS_COLLECTION,
) -> str:
    """Create the job collection if missing and ingest crawled ads from JSON.

    Idempotent: skips ingestion when the collection already has points.
    """
    collection = ensure_collection(collection=collection, recreate=False)
    info = get_client().get_collection(collection)
    if info.points_count == 0:
        ingest_jobs(path=jobs_path or DEFAULT_JOBS_PATH, collection=collection)
    return collection


def build_profile_query(cv: ExtractedCV) -> str:
    """Build a compact embedding query from the candidate's skills and roles."""
    skills = ", ".join(cv.Hard_Skills) or "Unknown skills"
    titles = " | ".join(
        job.get("Job Title", "") for job in cv.Work_Experience if job.get("Job Title")
    )
    return f"{skills}. Roles: {titles or 'N/A'}"


def retrieve_matching_jobs(
    cv: ExtractedCV,
    top_k: int = 3,
    collection: str = JOBS_COLLECTION,
) -> list[dict]:
    """Retrieve the top_k most semantically relevant job ads from Qdrant."""
    query = build_profile_query(cv)
    return search_jobs(query, top_k=top_k, collection=collection)


def job_ad_to_text(job: dict) -> str:
    """Flatten a retrieved job dict into a readable JD block for the LLM."""
    parts = [
        f"Job ID: {job.get('job_id', 'N/A')}",
        f"Company: {job.get('company', 'N/A')}",
        f"Location: {job.get('working_location', 'N/A')}",
        f"Salary: {job.get('salary', 'N/A')}",
    ]
    matched = job.get("matched_text")
    if matched:
        parts.append(f"Relevant section ({job.get('matched_section', '')}):\n{matched}")
    return "\n".join(parts)


def evaluate_job_match(cv: ExtractedCV, job_ad: dict) -> JobMatchReport:
    """Ask the LLM to score the CV's fit against a single job ad."""
    profile = cv.to_profile_text()
    jd = job_ad_to_text(job_ad)

    prompt = f"""You are a hiring-match evaluator. Evaluate how well the candidate's CV matches the job advertisement below.

Candidate CV:
{profile}

Job Advertisement:
{jd}

Output ONLY the exact JSON schema below. Do NOT add markdown code fences and output raw JSON only.

Schema:
{{
  "Job_Title": null,
  "Match_Score": null,
  "Evidence": null,
  "Gap_Analysis": null
}}

Rules:
- "Job_Title": string, the job title being evaluated.
- "Match_Score": integer 0-100 indicating overall fit.
- "Evidence": string, specific CV details that support the match (skills, experience, projects).
- "Gap_Analysis": string, required skills/experience the candidate is missing or weak in.
"""

    try:
        raw = extract_json_with_llm(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM evaluation failed for job {job_ad.get('job_id')}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected a JSON object from LLM, got {type(raw).__name__}")

    job_title = raw.get("Job_Title") or str(job_ad.get("matched_section") or job_ad.get("job_id") or "Unknown")
    return JobMatchReport(
        Job_Title=job_title,
        Match_Score=raw.get("Match_Score"),
        Evidence=raw.get("Evidence", ""),
        Gap_Analysis=raw.get("Gap_Analysis", ""),
    )


def _print_report(job: dict, report: JobMatchReport) -> None:
    company = job.get("company") or "Unknown company"
    print("=" * 72)
    print(f"  {report.Job_Title}")
    print(f"  {company} | {job.get('working_location', 'N/A')} | {job.get('salary', 'N/A')}")
    print(f"  Match Score: {report.Match_Score}/100")
    print("-" * 72)
    print(f"  Evidence:\n    {report.Evidence}")
    print(f"  Gap Analysis:\n    {report.Gap_Analysis}")
    print()


def main(
    cv_path: str | Path | None = None,
    jobs_path: str | Path | None = None,
    top_k: int = 3,
    collection: str = JOBS_COLLECTION,
) -> None:
    """Orchestrate the full Resume-to-Job matching pipeline."""
    try:
        cv = load_cv(cv_path)
        print(f"Loaded CV for {cv.Candidate_Name} "
              f"({cv.Total_Years_of_Experience} years, {len(cv.Hard_Skills)} hard skills)")

        ensure_job_collection(jobs_path, collection=collection)
        jobs = retrieve_matching_jobs(cv, top_k=top_k, collection=collection)
        if not jobs:
            print("No matching jobs retrieved.")
            return
        print(f"Retrieved {len(jobs)} candidate jobs from Qdrant\n")

        for job in jobs:
            report = evaluate_job_match(cv, job)
            _print_report(job, report)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise
    finally:
        close_client()


if __name__ == "__main__":
    main()
