"""
Tests for Retrieval Service (TF-IDF, confidence threshold, and attribution).
"""

import pytest
import pandas as pd
from services.retrieval_service import TFIDFRetriever, RetrievalService


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "id": "1",
            "question": "What are the symptoms of asthma?",
            "answer": "Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing.",
            "focus": "Asthma",
            "qtype": "symptoms",
            "source": "NHLBI",
            "url": "https://nhlbi.nih.gov/asthma"
        },
        {
            "id": "2",
            "question": "How is type 2 diabetes managed?",
            "answer": "Management includes diet, regular exercise, metformin medication, and glucose monitoring.",
            "focus": "Type 2 Diabetes",
            "qtype": "treatment",
            "source": "NIDDK",
            "url": "https://niddk.nih.gov/diabetes"
        }
    ])


def test_tfidf_indexer_and_retrieve(sample_df):
    retriever = TFIDFRetriever()
    success = retriever.build_index(sample_df)
    assert success is True
    assert retriever.is_indexed is True

    # Search asthma
    results = retriever.retrieve("wheezing and shortness of breath asthma", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top["focus"] == "Asthma"
    assert top["source"] == "NHLBI"
    assert top["url"] == "https://nhlbi.nih.gov/asthma"
    assert top["score"] > 0.1


def test_tfidf_irrelevant_query(sample_df):
    retriever = TFIDFRetriever()
    retriever.build_index(sample_df)

    # Completely unrelated query
    results = retriever.retrieve("quantum physics semiconductor theory", top_k=2)
    assert len(results) > 0
    # Score should be very low (near zero)
    assert results[0]["score"] < 0.1


def test_retrieval_service_confidence(sample_df, monkeypatch):
    service = RetrievalService()
    service.tfidf_retriever.build_index(sample_df)

    # Relevant query
    results, confident = service.search("What are asthma symptoms?", top_k=1)
    assert len(results) == 1
    assert confident is True

    # Irrelevant query
    results, confident = service.search("how to repair a car engine transmission", top_k=1)
    assert confident is False
