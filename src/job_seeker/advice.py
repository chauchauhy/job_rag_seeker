"""LLM-powered actionable application advice (cover letter, interview prep, talking points)."""

import json

from job_seeker.llm import extract_json_with_llm
from job_seeker.models import ActionableAdvice, cv_to_text

__all__ = ["ADVICE_PROMPT", "generate_actionable_advice"]


ADVICE_PROMPT = """You are a senior career coach helping a candidate apply for a job they already matched against.

Candidate CV:
{cv}

Job match report:
{job_report}

Produce targeted, practical advice for the application. Output ONLY the exact JSON schema below. Do NOT add markdown code fences and output raw JSON only.

Schema:
{{
  "Tags": [],
  "Cover_Letter_Draft": null,
  "Interview_Prep": [],
  "Suggestions_To_Raise": []
}}

Rules:
- "Tags": 3-5 short tags summarizing the job and the match (e.g. ["Python", "AI", "High Match", "Remote"]). Include the top 1-2 skills, a match-level tag, and 1-2 job traits.
- "Cover_Letter_Draft": a polished, personalized cover letter (3-4 short paragraphs) that addresses the candidate's strengths and tactfully bridges the gap analysis.
- "Interview_Prep": an array of 3-5 objects. Each has "Question" (a tough interview question targeting the candidate's Gap Analysis or weak spots) and "Suggested_Answer" (a strong, honest answer the candidate can practice).
- "Suggestions_To_Raise": an array of 5-8 strings. Concrete things the candidate should raise during the interview: strengths to emphasize, talking points that bridge the gap analysis, and questions to ask the interviewer about the team/tech/company direction.
"""


def generate_actionable_advice(cv_data: dict, job_report: dict) -> ActionableAdvice:
    """Ask the LLM (opencode CLI) to produce tailored application advice.

    Args:
        cv_data: Candidate profile dict (either ``ExtractedCV.model_dump()`` or
            the spaced-key raw resume dict).
        job_report: A ``JobMatchReport``-shaped dict.

    Raises:
        RuntimeError: if the LLM call fails or returns a non-object.
    """
    prompt = ADVICE_PROMPT.format(
        cv=cv_to_text(cv_data),
        job_report=json.dumps(job_report, ensure_ascii=False, indent=2),
    )
    try:
        raw = extract_json_with_llm(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM advice generation failed: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected a JSON object from LLM, got {type(raw).__name__}")

    raw.setdefault("Tags", [])
    raw.setdefault("Cover_Letter_Draft", "")
    raw.setdefault("Interview_Prep", [])
    raw.setdefault("Suggestions_To_Raise", [])
    return ActionableAdvice.model_validate(raw)
