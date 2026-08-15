"""Ingest crawled job listings into Qdrant as hybrid (dense + BM25) points."""

import json
import os
import time
import uuid
from pathlib import Path

from qdrant_client import models

from job_seeker.config import settings
from job_seeker.vector_db.embeddings import embed_colbert_docs, embed_sparse
from job_seeker.vector_db.qdrant import ensure_collection, get_client
from job_seeker.vector_db.schema import build_chunks

UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "10"))
EMBED_BATCH_SIZE = 64
BATCH_DELAY_SECONDS = float(os.getenv("QDRANT_BATCH_DELAY_SECONDS", "0.5"))


def load_jobs(path: str | Path | None = None) -> list[dict]:
    path = path or settings.jobsdb_output_file
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _point_id(chunk) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{chunk.job_id}:{chunk.section}:{chunk.chunk_index}",
        )
    )


def build_points(jobs: list[dict], embed_batch_size: int = EMBED_BATCH_SIZE) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for start in range(0, len(jobs), embed_batch_size):
        batch = jobs[start : start + embed_batch_size]
        chunks = [chunk for job in batch for chunk in build_chunks(job)]
        if not chunks:
            continue
        texts = [c.text for c in chunks]
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
) -> int:
    """Chunk, embed, and upsert an in-memory list of job dicts.

    Returns the number of points written. Point ids are deterministic
    (``uuid5(job_id:section:chunk_index)``), so re-ingesting an updated job
    with the same ``job_id`` overwrites its previous points cleanly.
    """
    if not jobs:
        raise ValueError("No jobs to ingest")
    collection = ensure_collection(collection=collection, recreate=recreate)
    points = build_points(jobs)
    client = get_client()
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        _upsert_with_retry(client, collection, points[i : i + UPSERT_BATCH_SIZE])
        time.sleep(BATCH_DELAY_SECONDS)
    return len(points)


def ingest_jobs(
    path: str | Path | None = None,
    recreate: bool = False,
    collection: str | None = None,
) -> int:
    """Chunk, embed, and upsert all jobs from a JSON file.

    Returns the number of points written.
    """
    jobs = load_jobs(path)
    if not jobs:
        raise ValueError(f"No jobs found in {path or settings.jobsdb_output_file}")
    return ingest_jobs_list(jobs, collection=collection, recreate=recreate)
