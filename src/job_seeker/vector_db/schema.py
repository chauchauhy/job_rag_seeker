"""Document chunking for RAG: split job postings into overlapping retrieval units."""

import re
from dataclasses import dataclass

from job_seeker.config import settings

BULLET_PREFIX = re.compile(r"^\s*[\-\u2022\u25AA\u00B7\u25CF\u25A1\u2013\u2014\uFF3F\uFFF0]*\s*")
SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+")

RAG_SECTIONS = ("Responsibilities", "Requirements")


@dataclass
class JobChunk:
    job_id: str
    section: str
    chunk_index: int
    text: str
    payload: dict


def _clean_line(line: str) -> str:
    return BULLET_PREFIX.sub("", line).strip()


def _split_bullets(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        line = _clean_line(line)
        if line:
            items.append(line)
    return items


def _overlap_tail(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[-n_words:]) if len(words) > n_words else text


def _split_long_bullet(bullet: str, chunk_size: int) -> list[str]:
    if len(bullet) <= chunk_size:
        return [bullet]
    parts = SENTENCE_SPLIT.split(bullet)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + " " + part).strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_bullets(
    bullets: list[str],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    chunks: list[str] = []
    current = ""
    for bullet in bullets:
        for piece in _split_long_bullet(bullet, chunk_size):
            candidate = (current + " " + piece).strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = _overlap_tail(current, overlap)
                candidate = (current + " " + piece).strip()
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_chunks(job: dict) -> list[JobChunk]:
    base = {
        "job_id": str(job.get("job_id", "")),
        "job_url": job.get("job_url", ""),
        "company": job.get("company", ""),
        "salary": job.get("salary", ""),
        "working_location": job.get("working_location", ""),
    }
    chunks: list[JobChunk] = []
    for section in RAG_SECTIONS:
        text = job.get(section) or ""
        if not text.strip():
            continue
        bullets = _split_bullets(text)
        if not bullets:
            continue
        for i, piece in enumerate(chunk_bullets(bullets)):
            chunks.append(
                JobChunk(
                    job_id=base["job_id"],
                    section=section,
                    chunk_index=i,
                    text=piece,
                    payload={**base, "section": section, "chunk_index": i, "text": piece},
                )
            )
    return chunks
