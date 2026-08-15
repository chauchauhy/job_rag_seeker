"""Tests for job ingestion into the vector database (vector_db.ingest)."""

import json

import numpy as np
import pytest

from job_seeker.vector_db.ingest import (
    build_points,
    ingest_jobs_list,
    load_jobs_dir,
    validate_jobs_data,
)

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


def test_load_jobs_dir_merges_json_files(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps([{"job_id": "a"}]), encoding="utf-8"
    )
    (tmp_path / "b.json").write_text(
        json.dumps([{"job_id": "b"}, {"job_id": "c"}]), encoding="utf-8"
    )
    jobs = load_jobs_dir(tmp_path)
    assert [j["job_id"] for j in jobs] == ["a", "b", "c"]


def test_load_jobs_dir_missing_dir_returns_empty(tmp_path):
    assert load_jobs_dir(tmp_path / "does-not-exist") == []


def test_load_jobs_dir_skips_bad_json(tmp_path):
    (tmp_path / "ok.json").write_text(json.dumps([{"job_id": "a"}]), encoding="utf-8")
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    jobs = load_jobs_dir(tmp_path)
    assert [j["job_id"] for j in jobs] == ["a"]


def test_validate_jobs_data_rejects_non_list():
    with pytest.raises(ValueError, match="JSON list"):
        validate_jobs_data({"job_id": "a"})


def test_validate_jobs_data_rejects_empty_list():
    with pytest.raises(ValueError, match="no jobs"):
        validate_jobs_data([])


def test_validate_jobs_data_rejects_non_dict_item():
    with pytest.raises(ValueError, match="not a JSON object"):
        validate_jobs_data([{"job_id": "a"}, "oops"])


def test_validate_jobs_data_warns_on_empty_jobs():
    jobs, warnings = validate_jobs_data(
        [{"job_id": "empty-job"}, {"job_id": "ok-job", "Responsibilities": "- Python"}]
    )
    assert [j["job_id"] for j in jobs] == ["ok-job"]
    assert any("empty-job" in w for w in warnings)


def test_validate_jobs_data_rejects_all_empty():
    with pytest.raises(ValueError, match="job text"):
        validate_jobs_data([{"job_id": "a"}, {"job_id": "b"}])


def test_build_points_reports_embed_progress(fake_embeddings):
    events = []
    build_points([JOB, JOB], on_progress=lambda stage, frac, msg: events.append((stage, frac, msg)))
    assert events, "expected embed progress events"
    assert all(stage == "embed" for stage, _, _ in events)
    assert all(0.0 <= frac <= 1.0 for _, frac, _ in events)


def test_ingest_jobs_list_reports_upsert_progress(monkeypatch, fake_embeddings):
    client = _FakeClient()
    monkeypatch.setattr("job_seeker.vector_db.ingest.get_client", lambda: client)
    monkeypatch.setattr(
        "job_seeker.vector_db.ingest.ensure_collection",
        lambda collection=None, recreate=False: collection or "jobs_collection",
    )
    events = []
    ingest_jobs_list([JOB], on_progress=lambda stage, frac, msg: events.append((stage, frac)))
    assert ("upsert", 0.0) in events, "expected a final upsert progress event"
    assert any(stage == "embed" for stage, _ in events)
