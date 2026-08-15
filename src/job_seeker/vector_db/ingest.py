"""Ingest crawled job listings into Qdrant as hybrid (dense + BM25) points."""

import json
import math
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from qdrant_client import models

from job_seeker.config import settings
from job_seeker.logging_setup import get_logger
from job_seeker.vector_db.embeddings import embed_colbert_docs, embed_sparse
from job_seeker.vector_db.qdrant import ensure_collection, get_client
from job_seeker.vector_db.schema import RAG_SECTIONS, build_chunks

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float, str], None]

UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "10"))
EMBED_BATCH_SIZE = 64
BATCH_DELAY_SECONDS = float(os.getenv("QDRANT_BATCH_DELAY_SECONDS", "0.5"))


def load_jobs(path: str | Path | None = None) -> list[dict]:
    path = path or settings.jobsdb_output_file
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jobs_dir(directory: str | Path | None = None) -> list[dict]:
    """Load and merge all ``*.json`` files under a jobs directory.

    Missing or empty directories yield ``[]`` so callers can degrade
    gracefully. Files are sorted by name for deterministic ordering.
    """
    directory = Path(directory) if directory else settings.jobs_dir
    if not directory.is_dir():
        logger.info("Jobs directory %s not found; using empty job list", directory)
        return []
    jobs: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            jobs.extend(load_jobs(path))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable jobs file %s", path)
            continue
    logger.info("Loaded %d jobs from %d file(s) in %s", len(jobs), len(list(directory.glob("*.json"))), directory)
    return jobs


def validate_jobs_data(data) -> tuple[list[dict], list[str]]:
    """Validate uploaded jobs payload.

    Returns ``(jobs, warnings)``. Raises ``ValueError`` with a friendly
    message for structural problems; warnings flag items that would index
    nothing (no Responsibilities/Requirements text).
    """
    if not isinstance(data, list):
        raise ValueError("File must contain a JSON list of job objects.")
    if not data:
        raise ValueError("File contains no jobs.")
    jobs: list[dict] = []
    warnings: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{i + 1} is not a JSON object.")
        text = " ".join(
            str(item.get(section) or "").strip() for section in RAG_SECTIONS
        ).strip()
        if not text:
            label = item.get("job_id") or item.get("company") or f"#{i + 1}"
            warnings.append(f"{label}: no Responsibilities/Requirements text, skipped")
            continue
        jobs.append(item)
    if not jobs:
        raise ValueError("None of the items have job text to index.")
    if warnings:
        logger.warning("Skipping %d item(s) without job text", len(warnings))
    logger.info("Validated %d jobs (%d skipped)", len(jobs), len(warnings))
    return jobs, warnings


def _point_id(chunk) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{chunk.job_id}:{chunk.section}:{chunk.chunk_index}",
        )
    )


def build_points(
    jobs: list[dict],
    embed_batch_size: int = EMBED_BATCH_SIZE,
    on_progress: ProgressCallback | None = None,
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    total_batches = max(1, math.ceil(len(jobs) / embed_batch_size))
    for batch_index, start in enumerate(range(0, len(jobs), embed_batch_size), 1):
        batch = jobs[start : start + embed_batch_size]
        chunks = [chunk for job in batch for chunk in build_chunks(job)]
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        logger.info("Embedding chunk batch %d/%d (%d chunks)", batch_index, total_batches, len(chunks))
        if on_progress:
            on_progress(
                "embed",
                (batch_index - 1) / total_batches,
                f"Embedding chunk batch {batch_index}/{total_batches}…",
            )
        colbert_vecs = embed_colbert_docs(texts)
        sparse_vecs = embed_sparse(texts)
        for chunk, colbert, sparse in zip(chunks, colbert_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    id=_point_id(chunk),
                    vector={
                        settings.qdrant_dense_vector_name: colbert,
                        settings.qdrant_sparse_vector_name: models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload=chunk.payload,
                )
            )
    logger.info("Built %d points from %d jobs", len(points), len(jobs))
    return points


def _upsert_with_retry(
    client,
    collection: str,
    points,
    retries: int = 3,
    delay: float = 2.0,
) -> None:
    """Upsert a batch, retrying on transient errors (e.g. slow cloud cluster)."""
    for attempt in range(retries):
        try:
            client.upsert(collection_name=collection, points=points)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def ingest_jobs_list(
    jobs: list[dict],
    collection: str | None = None,
    recreate: bool = False,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Chunk, embed, and upsert an in-memory list of job dicts.

    Returns the number of points written. Point ids are deterministic
    (``uuid5(job_id:section:chunk_index)``), so re-ingesting an updated job
    with the same ``job_id`` overwrites its previous points cleanly.
    """
    if not jobs:
        raise ValueError("No jobs to ingest")
    started = time.time()
    collection = ensure_collection(collection=collection, recreate=recreate)
    points = build_points(jobs, on_progress=on_progress)
    client = get_client()
    total = len(points)
    for i in range(0, total, UPSERT_BATCH_SIZE):
        end = min(i + UPSERT_BATCH_SIZE, total)
        logger.info(
            "Upserting points %d-%d of %d into '%s'", i, end, total, collection
        )
        if on_progress:
            on_progress(
                "upsert",
                i / max(1, total),
                f"Upserting points {i}-{end} of {total}…",
            )
        _upsert_with_retry(client, collection, points[i:end])
        time.sleep(BATCH_DELAY_SECONDS)
    logger.info(
        "Ingested %d points into '%s' in %.1fs", total, collection, time.time() - started
    )
    return total


def ingest_jobs(
    path: str | Path | None = None,
    recreate: bool = False,
    collection: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Chunk, embed, and upsert all jobs from a JSON file.

    Returns the number of points written.
    """
    jobs = load_jobs(path)
    if not jobs:
        raise ValueError(f"No jobs found in {path or settings.jobsdb_output_file}")
    return ingest_jobs_list(
        jobs, collection=collection, recreate=recreate, on_progress=on_progress
    )


def ingest_jobs_dir(
    directory: str | Path | None = None,
    recreate: bool = False,
    collection: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Chunk, embed, and upsert all ``*.json`` job files under a directory.

    With ``recreate=True`` the collection is rebuilt from scratch using only
    the files present in the directory. Returns the number of points written.
    """
    jobs = load_jobs_dir(directory)
    if not jobs:
        raise ValueError(f"No jobs found in {directory or settings.jobs_dir}")
    if recreate:
        logger.info("Rebuilding collection '%s' from jobs directory", collection or settings.qdrant_collection)
    return ingest_jobs_list(
        jobs, collection=collection, recreate=recreate, on_progress=on_progress
    )
