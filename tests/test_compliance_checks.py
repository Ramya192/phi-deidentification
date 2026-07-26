"""
tests/test_compliance_checks.py

Covers agents/compliance_validation_agent.py's _build_compliance_checks()
and its wiring into compliance_validation_agent()/audit_report_agent() --
the per-category checks list + aggregate compliance_score that
complements the single validation_status PASS/FAIL boolean.

Run with: pytest -v tests/test_compliance_checks.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents.compliance_validation_agent import _build_compliance_checks, compliance_validation_agent
from agents import phi_detection_agent as phi_detection_module


@pytest.fixture
def force_fallback_detector():
    original = phi_detection_module.USE_FALLBACK_ONLY
    phi_detection_module.USE_FALLBACK_ONLY = True
    yield
    phi_detection_module.USE_FALLBACK_ONLY = original


def test_no_detected_spans_is_vacuously_compliant():
    checks, score = _build_compliance_checks([], [])
    assert checks == []
    assert score == 1.0


def test_all_categories_clear():
    detected = [
        {"phi_type": "MRN", "text": "a"},
        {"phi_type": "PERSON", "text": "b"},
    ]
    checks, score = _build_compliance_checks(detected, [])
    assert score == 1.0
    by_category = {c["category"]: c for c in checks}
    assert by_category["MRN"]["passed"] is True
    assert by_category["MRN"]["residual_count"] == 0
    assert by_category["PERSON"]["passed"] is True


def test_one_category_still_has_residual():
    detected = [
        {"phi_type": "MRN", "text": "a"},
        {"phi_type": "PERSON", "text": "b"},
        {"phi_type": "DATE_TIME", "text": "c"},
    ]
    remaining = [
        {"phi_type": "PERSON", "text": "b", "start": 0, "end": 1, "confidence": 0.9},
    ]
    checks, score = _build_compliance_checks(detected, remaining)
    by_category = {c["category"]: c for c in checks}
    assert by_category["PERSON"]["passed"] is False
    assert by_category["PERSON"]["residual_count"] == 1
    assert by_category["MRN"]["passed"] is True
    assert by_category["DATE_TIME"]["passed"] is True
    # 2 of 3 categories clear
    assert score == round(2 / 3, 3)


def test_multiple_residuals_same_category_counted():
    detected = [{"phi_type": "PERSON", "text": "a"}]
    remaining = [
        {"phi_type": "PERSON", "text": "a", "start": 0, "end": 1, "confidence": 0.9},
        {"phi_type": "PERSON", "text": "a", "start": 5, "end": 6, "confidence": 0.9},
    ]
    checks, score = _build_compliance_checks(detected, remaining)
    assert checks[0]["residual_count"] == 2
    assert checks[0]["passed"] is False
    assert score == 0.0


def test_compliance_validation_agent_populates_checks_and_score(force_fallback_detector):
    state = {
        "redacted_text": "Patient [PERSON] was seen on [DATE_TIME] and is doing well.",
        "phi_spans": [
            {"phi_type": "PERSON", "text": "x", "start": 0, "end": 1, "confidence": 0.9, "source_agent": "test"},
            {"phi_type": "DATE_TIME", "text": "y", "start": 0, "end": 1, "confidence": 0.9, "source_agent": "test"},
        ],
        "retry_count": 0,
        "audit_log": [],
    }
    result = compliance_validation_agent(state)
    assert result["validation_status"] == "PASS"
    assert result["compliance_score"] == 1.0
    categories = {c["category"] for c in result["compliance_checks"]}
    assert categories == {"PERSON", "DATE_TIME"}
    assert all(c["passed"] for c in result["compliance_checks"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
