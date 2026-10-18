"""
Tests for Medical Service (free retrieval answering, emergency handling, and missing API credentials).
"""

import pytest
import os
from services.medical_service import MedicalQAService
import config


@pytest.fixture
def medical_service():
    service = MedicalQAService()
    service.retrieval.load_data()
    return service


def test_answer_query_free_mode_no_api_key(medical_service, monkeypatch):
    # Ensure GEMINI_API_KEY is not set
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert not config.is_gemini_configured()

    # Query for diabetes
    response = medical_service.answer_query("What are the symptoms of diabetes?", mode="retrieval_free")

    assert response is not None
    assert response["mode_used"] == "retrieval_free"
    assert "Retrieval-Based Answering" in response["mode_label"]
    assert "diabetes" in response["answer"].lower()
    assert response["has_sufficient_evidence"] is True
    assert len(response["references"]) > 0
    assert response["disclaimer"] == config.MEDICAL_DISCLAIMER
    assert not response["is_emergency"]


def test_emergency_situation_triggers_alert(medical_service):
    query = "I am having severe crushing chest pain, sweating, and difficulty breathing."
    response = medical_service.answer_query(query)

    assert response["is_emergency"] is True
    assert response["emergency_alert"] is not None
    assert "CRITICAL WARNING" in response["emergency_alert"]
    assert any("chest pain" in s for s in response["recognized_entities"]["symptoms"])


def test_unmatched_query_states_no_evidence_truthfully(medical_service):
    # Query completely outside medical scope
    query = "How do I calculate aerodynamic drag on a rocket booster?"
    response = medical_service.answer_query(query)

    assert response["has_sufficient_evidence"] is False
    assert "No relevant medical evidence was found" in response["answer"]


def test_gemini_mode_requested_without_key_falls_back_safely(medical_service, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Even if mode="gemini" is passed, if key is missing, it must not crash and must use free retrieval
    response = medical_service.answer_query("What is asthma?", mode="gemini")

    assert response["mode_used"] == "retrieval_free"
    assert "Retrieval-Based Answering" in response["mode_label"]
    assert "asthma" in response["answer"].lower()
