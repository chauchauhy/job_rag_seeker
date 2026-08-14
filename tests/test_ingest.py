"""Tests for job ingestion into the vector database (vector_db.ingest)."""

import numpy as np
import pytest

from job_seeker.vector_db.ingest import build_points, ingest_jobs_list

JOB = {
    "job_id": "manual_demo",
    "job_url": "",
    "company": "Acme Labs",
    "salary": "45K - 55K",
    "working_location": "Hong Kong",
    "Responsibilities": "- Build REST APIs\n- Write automated tests",
    "Requirements": "- Python\n- SQL",
}


class _FakeSparse:
    def __init__(self):
        self.indices = np.array([1, 2])
        self.values = np.array([0.5, 0.5])


def _fake_colbert(texts):
    return [[[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]] for _ in texts]


def _fake_sparse(texts):
    return [_FakeSparse() for _ in texts]


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr("job_seeker.vector_db.ingest.embed_colbert_docs", _fake_colbert)
    monkeypatch.setattr("job_seeker.vector_db.ingest.embed_sparse", _fake_sparse)


class _FakeClient:
    def __init__(self):
        self.upserted = []

    def upsert(self, collection_name=None, points=None):
        self.upserted.extend(points)


def test_build_points_deterministic_ids(fake_embeddings):
    ids_1 = {p.id for p in build_points([JOB])}
    ids_2 = {p.id for p in build_points([JOB])}
    assert ids_1 == ids_2
    assert ids_1, "expected at least one point per chunk"


def test_ingest_jobs_list_empty_raises():
    with pytest.raises(ValueError):
        ingest_jobs_list([])


def test_ingest_jobs_list_upserts_without_duplication(monkeypatch, fake_embeddings):
    client = _FakeClient()
    monkeypatch.setattr("job_seeker.vector_db.ingest.get_client", lambda: client)
    monkeypatch.setattr(
        "job_seeker.vector_db.ingest.ensure_collection",
        lambda collection=None, recreate=False: collection or "jobs_collection",
    )

    first = ingest_jobs_list([JOB])
    ids_first = {p.id for p in client.upserted}
    assert len(ids_first) == first

    client.upserted.clear()
    ingest_jobs_list([JOB])
    ids_second = {p.id for p in client.upserted}
    assert ids_second == ids_first, "re-ingesting the same job must overwrite the same point ids"
