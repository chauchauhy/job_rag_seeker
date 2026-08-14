"""Tests for JSON extraction from raw LLM output."""

import pytest

from job_seeker.llm import _extract_json_object, extract_json_with_llm


def test_extract_first_balanced_object():
    raw = '{"a": 1} trailing reasoning here'
    assert _extract_json_object(raw) == '{"a": 1}'


def test_extract_ignores_text_before_brace():
    raw = 'Here you go:\n{"key": "value"}'
    assert _extract_json_object(raw) == '{"key": "value"}'


def test_extract_handles_nested_and_strings():
    raw = '{"a": {"b": 1}, "s": "a } brace"}'
    assert _extract_json_object(raw) == raw


def test_extract_returns_none_without_brace():
    assert _extract_json_object("no json here") is None


def test_extract_picks_first_of_two_objects():
    raw = '{"first": 1} then {"second": 2}'
    assert _extract_json_object(raw) == '{"first": 1}'


def test_extract_json_with_llm_raises_on_garbage(monkeypatch):
    monkeypatch.setattr("job_seeker.llm.run_llm", lambda *a, **k: "not json at all")
    with pytest.raises(ValueError):
        extract_json_with_llm("prompt")
