"""Tests for the shared domain models and CV serialization."""

import pytest
from pydantic import ValidationError

from job_seeker.models import ActionableAdvice, ExtractedCV, JobMatchReport, cv_to_text
from job_seeker.pipeline import load_cv

SNAKE_CV = {
    "Candidate_Name": "Ada Lovelace",
    "Total_Years_of_Experience": 5,
    "Hard_Skills": ["Python", "SQL"],
    "Soft_Skills": ["Communication"],
    "Work_Experience": [
        {
            "Company": "Analytical Engines",
            "Job Title": "Analyst Programmer",
            "Duration": "2020 - 2023",
            "Responsibilities": ["Built ETL pipelines", "Optimized queries"],
        }
    ],
}

SPACED_CV = {
    "Candidate Name": "Ada Lovelace",
    "Total Years of Experience": 5,
    "Hard Skills": ["Python", "SQL"],
    "Soft Skills": ["Communication"],
    "Work Experience": [
        {
            "Company": "Analytical Engines",
            "Job Title": "Analyst Programmer",
            "Duration": "2020 - 2023",
            "Responsibilities": ["Built ETL pipelines", "Optimized queries"],
        }
    ],
}


def test_extracted_cv_snake_case_keys():
    cv = ExtractedCV.model_validate(SNAKE_CV)
    assert cv.Candidate_Name == "Ada Lovelace"
    assert cv.Hard_Skills == ["Python", "SQL"]
    assert cv.Work_Experience[0]["Job Title"] == "Analyst Programmer"


def test_extracted_cv_accepts_spaced_aliases():
    cv = ExtractedCV.model_validate(SPACED_CV)
    assert cv.Candidate_Name == "Ada Lovelace"
    assert cv.Total_Years_of_Experience == 5


def test_job_match_report_score_bounds():
    with pytest.raises(ValidationError):
        JobMatchReport(
            Job_Title="x",
            Match_Score=101,
            Evidence="e",
            Gap_Analysis="g",
        )


def test_cv_to_text_snake_case():
    text = cv_to_text(SNAKE_CV)
    assert "Candidate: Ada Lovelace" in text
    assert "Hard Skills: Python, SQL" in text
    assert "- Analyst Programmer @ Analytical Engines: Built ETL pipelines | Optimized queries" in text


def test_cv_to_text_spaced_keys_matches_snake_case():
    assert cv_to_text(SPACED_CV) == cv_to_text(SNAKE_CV)


def test_to_profile_text_round_trip():
    cv = ExtractedCV.model_validate(SNAKE_CV)
    assert cv.to_profile_text() == cv_to_text(SNAKE_CV)


def test_actionable_advice_requires_tags():
    with pytest.raises(ValidationError):
        ActionableAdvice(Tags=[], Cover_Letter_Draft="d")


def test_cv_json_round_trip(tmp_path):
    cv = ExtractedCV.model_validate(SNAKE_CV)
    path = tmp_path / "cv.json"
    path.write_text(cv.model_dump_json(), encoding="utf-8")
    loaded = load_cv(path)
    assert loaded == cv
