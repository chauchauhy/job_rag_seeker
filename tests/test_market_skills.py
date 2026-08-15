"""Tests for the non-LLM market-skill analysis (job_seeker.market_skills)."""

import json

import pytest

from job_seeker import market_skills
from job_seeker.market_skills import (
    candidate_gap,
    extract_skills,
    filter_jobs,
    load_skill_dict,
    normalize_text,
    salary_band,
    top_companies,
)

JOB_1 = {
    "job_id": "1",
    "company": "Acme Labs",
    "working_location": "Hong Kong Island",
    "salary": "$40,000 - $45,830 per month",
    "Requirements": "- Java, JavaScript, Python\n- Node.js backend\n- SQL Server",
}
JOB_2 = {
    "job_id": "2",
    "company": "Beta Co",
    "working_location": "Kowloon",
    "salary": "$60,000 per month",
    "Requirements": "Java/JEE and Spring Boot; MongoDB, Docker, Kubernetes",
}
JOB_3 = {
    "job_id": "3",
    "company": "Acme Labs",
    "working_location": "Hong Kong Island",
    "salary": "",
    "Requirements": "",
}

JOBS = [JOB_1, JOB_2, JOB_3]


def test_normalize_text_lowercases_and_collapses():
    assert normalize_text("  Java,\n  Node.js  ") == "java, node.js"


def test_extract_skills_counts_distinct_jobs_not_mentions():
    rows = extract_skills(JOBS)
    by_name = {row["skill"]: row for row in rows}

    java = by_name["Java"]
    assert java["count"] == 2
    assert java["job_ids"] == ["1", "2"]

    node = by_name["Node.js"]
    assert node["count"] == 1
    assert node["job_ids"] == ["1"]

    spring_boot = by_name["Spring Boot"]
    assert spring_boot["count"] == 1


def test_extract_skills_does_not_match_java_inside_javascript():
    rows = extract_skills([{"job_id": "x", "Requirements": "JavaScript expert"}])
    names = {row["skill"] for row in rows}
    assert "JavaScript" in names
    assert "Java" not in names


def test_extract_skills_skips_empty_requirements():
    rows = extract_skills([{"job_id": "x", "Requirements": ""}])
    assert rows == []


def test_extract_skills_can_scan_other_fields():
    rows = extract_skills(
        [{"job_id": "x", "Requirements": "", "Responsibilities": "React + Redux"}],
        fields=("Responsibilities",),
    )
    names = {row["skill"] for row in rows}
    assert "React" in names
    assert "Redux" in names


def test_extract_skills_sorted_by_count_desc():
    rows = extract_skills(JOBS)
    counts = [row["count"] for row in rows]
    assert counts == sorted(counts, reverse=True)


def test_compound_alias_beats_short_alias_in_same_phrase():
    text = "Spring Boot microservices"
    skill_aliases = {"Spring Boot": ["spring boot"], "Spring": ["spring"]}
    assert market_skills._match_skills(normalize_text(text), skill_aliases) == {
        "Spring Boot"
    }


def test_top_companies_sorted_by_count_desc():
    rows = top_companies(JOBS)
    assert rows[0] == {"company": "Acme Labs", "count": 2}
    assert rows[1] == {"company": "Beta Co", "count": 1}


def test_salary_band_bucketing():
    assert salary_band({"salary": "$40,000 - $45,830 per month"}) == "40K\u201360K"
    assert salary_band({"salary": "$25,000 monthly"}) == "< 40K"
    assert salary_band({"salary": "$80,000"}) == "\u2265 60K"
    assert salary_band({"salary": ""}) == "\u2014"


def test_filter_jobs_by_location_and_salary_band():
    only_hk = filter_jobs(JOBS, locations=["Hong Kong Island"])
    assert [job["job_id"] for job in only_hk] == ["1", "3"]

    mid = filter_jobs(JOBS, salary_bands=["40K\u201360K"])
    assert [job["job_id"] for job in mid] == ["1"]

    combined = filter_jobs(
        JOBS, locations=["Hong Kong Island"], salary_bands=["40K\u201360K"]
    )
    assert [job["job_id"] for job in combined] == ["1"]


def test_filter_jobs_empty_selection_is_noop():
    assert filter_jobs(JOBS) == JOBS


def test_candidate_gap_splits_matched_and_missing():
    counts = extract_skills(JOBS)
    gap = candidate_gap(["Java", "Node.js", "Python"], counts, top_n=9)
    matched = {row["skill"] for row in gap["matched"]}
    missing = {row["skill"] for row in gap["missing"]}
    assert "Java" in matched
    assert "Node.js" in matched
    assert "Spring Boot" in missing
    assert "SQL Server" in missing
    assert not (matched & missing)


def test_load_skill_dict_writes_fallback_when_missing(tmp_path):
    path = tmp_path / "nested" / "skills.json"
    loaded = load_skill_dict(path)
    assert loaded == load_skill_dict(path)
    assert json.loads(path.read_text(encoding="utf-8")) == market_skills.SKILL_DICT_FALLBACK
    assert "Java" in loaded
    assert loaded["Java"][0] == "Java"
