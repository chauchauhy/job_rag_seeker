"""Query expansion for lexical (BM25) retrieval using the LLM (opencode CLI)."""

from job_seeker.llm import extract_json_with_llm

__all__ = ["EXPAND_PROMPT", "expand_query"]

EXPAND_PROMPT = """You are a job-search query expansion assistant.

Given the user's job-search query, produce concrete related keywords: skills, technologies, frameworks, and synonyms that would help a BM25 keyword search retrieve matching job postings. Prefer specific, job-posted terms (e.g. real technology names, frameworks, and tools). Keep the expansion to the most useful terms, not a long list.

Query: {query}

Output ONLY a JSON object with exactly these keys:
{{
  "keywords": ["keyword1", "keyword2", ...],
  "expanded_query": "original query terms plus the most useful expansion keywords, space separated"
}}
Do not add markdown code fences and output raw JSON only.
"""


def expand_query(query: str, model: str | None = None) -> str:
    """Ask the LLM to expand a query into more keywords and return the expanded text.

    The expanded text is used as the BM25 (sparse) query so lexical matching
    covers synonyms and related technologies the user didn't type.
    """
    prompt = EXPAND_PROMPT.format(query=query)
    result = extract_json_with_llm(prompt, model=model)
    expanded = result.get("expanded_query") or ""
    if not expanded:
        keywords = result.get("keywords") or []
        expanded = " ".join([query, *keywords])
    return expanded
