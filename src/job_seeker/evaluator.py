"""RAG Phase 4: Evaluate a JD requirement against retrieved resume chunks."""

from typing import Literal, Optional

from job_seeker.llm import extract_json_with_llm

__all__ = ["EvaluationResult", "EVALUATE_PROMPT", "evaluate_requirement"]

EvaluationResult = dict[
    str,
    str | int | Optional[str],
]

EVALUATE_PROMPT = """You are an expert job-match evaluator. Your task is to assess how well a candidate's resume evidence matches a single job description requirement.

Evaluate the requirement against the retrieved resume chunks and output ONLY the exact JSON schema below. Do NOT add markdown code fences, do not include any conversational prose, and output raw JSON only.

Schema:
{{
  "Requirement_Evaluated": null,
  "Match_Status": null,
  "Match_Score": null,
  "Evidence_Reasoning": null,
  "Gap_Identified": null
}}

Rules:
- "Requirement_Evaluated": string, repeat the requirement text exactly as provided.
- "Match_Status": one of "Strong Match", "Partial Match", or "Missing". Choose based on how well the chunks demonstrate the requirement is satisfied.
- "Match_Score": integer 0-100. 90-100 = Strong Match with clear, direct evidence. 50-89 = Partial Match with some relevant evidence but gaps. 0-49 = Missing or very weak/unrelated evidence.
- "Evidence_Reasoning": string, explain which specific chunks support or contradict this requirement and why.
- "Gap_Identified": string or null. If "Missing" or "Partial Match", describe specifically what is absent or insufficient in the resume chunks.

Requirement to evaluate:
{requirement}

Retrieved resume chunks:
{chunks}
"""


def evaluate_requirement(
    jd_requirement: str,
    retrieved_chunks: list[str],
) -> EvaluationResult:
    """Evaluate a single JD requirement against retrieved resume chunks.

    Args:
        jd_requirement: The specific requirement text extracted from a job description.
        retrieved_chunks: List of text chunks retrieved from the resume (e.g. via hybrid search).

    Returns:
        An EvaluationResult dict with keys:
        - Requirement_Evaluated (str)
        - Match_Status (Literal["Strong Match", "Partial Match", "Missing"])
        - Match_Score (int 0-100)
        - Evidence_Reasoning (str)
        - Gap_Identified (Optional[str])

    Raises:
        ValueError: If jd_requirement is empty or retrieved_chunks is empty.
        RuntimeError: If the LLM call fails or returns unparseable JSON.
    """
    if not jd_requirement or not jd_requirement.strip():
        raise ValueError("jd_requirement cannot be empty")

    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")

    chunks_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = EVALUATE_PROMPT.format(
        requirement=jd_requirement.strip(),
        chunks=chunks_block,
    )

    try:
        result = extract_json_with_llm(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM evaluation failed: {exc}") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"Expected a JSON object from LLM, got {type(result).__name__}")

    for key in ("Requirement_Evaluated", "Match_Status", "Match_Score", "Evidence_Reasoning"):
        if key not in result:
            raise RuntimeError(f"LLM output missing required key: {key}")

    return result


if __name__ == "__main__":
    mock_requirement = "5+ years of experience in Java/J2EE, JavaScript, and React JS"
    mock_chunks = [
        "Strong experience in Java/J2EE development including Spring Boot, Hibernate, and Maven.",
        "Proficient in JavaScript (ES6+), TypeScript, and React JS with Redux for state management.",
        "Worked on building RESTful APIs using Node.js and Express.",
        "Skilled in HTML5, CSS3, and responsive web design.",
    ]

    print("Testing evaluate_requirement with mock data...")
    print(f"Requirement: {mock_requirement}")
    print(f"Chunks ({len(mock_chunks)}):")
    for i, c in enumerate(mock_chunks, 1):
        print(f"  [{i}] {c[:80]}...")
    print()

    result = evaluate_requirement(mock_requirement, mock_chunks)

    print("Evaluation Result:")
    print(f"  Requirement_Evaluated: {result.get('Requirement_Evaluated')}")
    print(f"  Match_Status:          {result.get('Match_Status')}")
    print(f"  Match_Score:           {result.get('Match_Score')}")
    print(f"  Evidence_Reasoning:    {result.get('Evidence_Reasoning')}")
    print(f"  Gap_Identified:        {result.get('Gap_Identified')}")
