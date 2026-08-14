"""Tests for job-list persistence helpers (crawler.upsert_jobs)."""

from job_seeker.crawler import upsert_jobs


def test_upsert_jobs_appends_new_by_job_id():
    result = upsert_jobs([], [{"job_id": "manual_a", "job_url": "", "company": "Acme"}])
    assert len(result) == 1


def test_upsert_jobs_updates_existing_by_job_id():
    existing = [{"job_id": "manual_a", "company": "Old"}]
    fresh = [{"job_id": "manual_a", "company": "New"}]
    result = upsert_jobs(existing, fresh)
    assert len(result) == 1
    assert result[0]["company"] == "New"


def test_upsert_jobs_still_dedupes_by_url():
    existing = [{"job_url": "https://example.com/job/42", "company": "Old"}]
    fresh = [{"job_url": "https://example.com/job/42#fragment", "company": "New"}]
    result = upsert_jobs(existing, fresh)
    assert len(result) == 1
    assert result[0]["company"] == "New"


def test_upsert_jobs_keeps_unrelated_jobs():
    existing = [{"job_id": "keep", "job_url": "https://example.com/x"}]
    fresh = [{"job_id": "manual_b", "job_url": "", "company": "Acme"}]
    result = upsert_jobs(existing, fresh)
    assert len(result) == 2
