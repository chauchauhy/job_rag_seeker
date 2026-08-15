"""Resume processing pipeline: PDF -> markdown -> structured JSON via LLM."""

import json
import time
from collections.abc import Callable
from pathlib import Path

from markitdown import MarkItDown

from job_seeker.config import RAW_DIR, RESULTS_DIR, settings
from job_seeker.llm import extract_json_with_llm
from job_seeker.logging_setup import get_logger

logger = get_logger(__name__)

__all__ = ["EXTRACT_PROMPT", "pdf_to_markdown", "extract_resume", "process_resume", "ensure_dirs"]

EXTRACT_PROMPT = """You are an Expert Extractor Agent. Perform Named Entity Recognition (NER) and detailed information extraction on the resume below, and output a structured JSON object.

Output ONLY the exact JSON schema below. Do NOT add any other keys, do not write markdown code fences (e.g., do not use ```json), do not include any conversational prose, and output raw JSON only.

Schema:
{{
  "Candidate Name": null,
  "Total Years of Experience": null,
  "Hard Skills": [],
  "Soft Skills": [],
  "Work Experience": [
    {{
      "Company": null,
      "Job Title": null,
      "Duration": null,
      "Responsibilities": []
    }}
  ],
  "Education": [
    {{
      "Degree": null,
      "Institution": null,
      "Field of Study": null
    }}
  ]
}}

Rules:
- "Candidate Name": string, the full name of the candidate, or null if it cannot be determined.
- "Total Years of Experience": integer, the calculated or extracted total full years of professional experience, or null if it cannot be determined.
- "Hard Skills": array of strings, specific technical, professional, or tool-based skills.
- "Soft Skills": array of strings, interpersonal, leadership, or communication skills.
- "Work Experience": array of objects. Extract each job role. "Responsibilities" MUST be an array of strings, where each string is a distinct bullet point of their achievements and duties from that role. This is critical for downstream semantic matching.
- "Education": array of objects. Use null if a specific field (Degree, Institution, or Field of Study) is missing.

Resume content:
{resume}
"""


def pdf_to_markdown(pdf_path: str | Path) -> str:
    md = MarkItDown()
    result = md.convert(str(pdf_path), output_format="markdown")
    return result.text_content


def extract_resume(resume_text: str) -> dict:
    prompt = EXTRACT_PROMPT.format(resume=resume_text)
    return extract_json_with_llm(prompt)


def process_resume(
    pdf_path: str | Path | None = None,
    on_progress: Callable[[str, float, str], None] | None = None,
) -> dict:
    """Convert a resume PDF to markdown, extract structured JSON, and persist both.

    Returns the extracted JSON dict.
    """
    pdf_path = pdf_path or settings.resume_pdf_path
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.cv_json_path.parent.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress("convert", 0.15, "Converting PDF to markdown…")
    markdown = pdf_to_markdown(pdf_path)
    logger.info("Converted PDF -> markdown (%d chars) in %.1fs", len(markdown), time.time() - started)
    settings.resume_markdown_output.write_text(markdown, encoding="utf-8")

    if on_progress:
        on_progress("extract", 0.6, "Extracting structured profile with LLM…")
    extracted = extract_resume(markdown)
    logger.info("LLM extracted CV in %.1fs", time.time() - started)
    settings.cv_json_path.write_text(
        json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if on_progress:
        on_progress("done", 1.0, "CV extraction complete")
    return extracted


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
