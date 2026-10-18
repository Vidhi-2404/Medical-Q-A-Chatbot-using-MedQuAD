"""
Tests for Entity Recognition and Emergency Detection Service.
"""

import pytest
from services.entity_service import EntityRecognitionService


@pytest.fixture
def entity_service():
    return EntityRecognitionService()


def test_extract_symptoms(entity_service):
    text = "The patient presents with high fever, persistent cough, and severe fatigue."
    entities = entity_service.extract_entities(text)
    assert "fever" in entities["symptoms"]
    assert "cough" in entities["symptoms"]
    assert "fatigue" in entities["symptoms"]


def test_extract_diseases(entity_service):
    text = "Diagnosis includes type 2 diabetes and hypertension, with a history of asthma."
    entities = entity_service.extract_entities(text)
    assert "type 2 diabetes" in entities["diseases"] or "diabetes" in entities["diseases"]
    assert "hypertension" in entities["diseases"]
    assert "asthma" in entities["diseases"]


def test_extract_medications(entity_service):
    text = "Prescribed metformin 500mg and lisinopril daily, plus aspirin as needed."
    entities = entity_service.extract_entities(text)
    assert "metformin" in entities["medications"]
    assert "lisinopril" in entities["medications"]
    assert "aspirin" in entities["medications"]


def test_extract_treatments(entity_service):
    text = "Recommended chemotherapy, surgical resection, and regular physical therapy."
    entities = entity_service.extract_entities(text)
    assert "chemotherapy" in entities["treatments"]
    assert "surgery" in entities["treatments"] or "surgical resection" in entities["treatments"]
    assert "physical therapy" in entities["treatments"]


def test_emergency_detection_positive(entity_service):
    assert entity_service.check_emergency("I have crushing chest pain and shortness of breath")
    assert entity_service.check_emergency("Sudden facial drooping and slurred speech")
    assert entity_service.check_emergency("Patient is having a severe allergic reaction and throat swelling")


def test_emergency_detection_negative(entity_service):
    assert not entity_service.check_emergency("What are the dietary guidelines for managing mild diabetes?")
    assert not entity_service.check_emergency("What is the history of penicillin?")


def test_empty_input(entity_service):
    entities = entity_service.extract_entities("")
    assert entities == {"symptoms": [], "diseases": [], "medications": [], "treatments": []}
    assert not entity_service.check_emergency("")
