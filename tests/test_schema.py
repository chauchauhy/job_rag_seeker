"""Tests for job-posting chunking (vector_db.schema)."""

from job_seeker.vector_db.schema import build_chunks, chunk_bullets


def test_chunk_bullets_splits_long_multi_sentence_bullet():
    long_bullet = (
        "First sentence introduces the design goals and the overall approach taken. "
        "Second sentence explains the implementation details and the trade-offs. "
        "Third sentence wraps up with the outcomes and the lessons learned."
    )
    chunks = chunk_bullets([long_bullet], chunk_size=60, overlap=10)
    assert len(chunks) > 1
    assert len(chunks[1]) > 0


def test_chunk_bullets_overlaps_adjacent_chunks():
    long_bullet = (
        "Alpha sentence covers the first topic and then some more details here. "
        "Beta sentence moves to the second topic with extra elaboration."
    )
    chunks = chunk_bullets([long_bullet], chunk_size=50, overlap=5)
    assert len(chunks) > 1
    words_a = chunks[0].split()
    words_b = chunks[1].split()
    assert any(w in words_b for w in words_a[-5:]), "overlap tail should reappear at the start of the next chunk"


def test_chunk_bullets_single_short_bullet():
    chunks = chunk_bullets(["Short bullet"], chunk_size=500, overlap=50)
    assert chunks == ["Short bullet"]


def test_build_chunks_payload():
    job = {
        "job_id": "42",
        "job_url": "https://example.com/job/42",
        "company": "Acme",
        "salary": "HK$30k",
        "working_location": "Hong Kong",
        "Responsibilities": "- Build REST APIs\n- Write automated tests",
        "Requirements": "",
    }
    chunks = build_chunks(job)
    assert chunks, "Responsibilities section should produce chunks"
    assert all(c.section == "Responsibilities" for c in chunks)
    first = chunks[0]
    assert first.job_id == "42"
    assert first.payload["company"] == "Acme"
    assert first.payload["job_url"] == "https://example.com/job/42"
    assert "section" in first.payload


def test_build_chunks_skips_empty_sections():
    job = {
        "job_id": "1",
        "Responsibilities": "",
        "Requirements": "",
    }
    assert build_chunks(job) == []
