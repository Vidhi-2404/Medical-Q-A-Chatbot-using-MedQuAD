"""
Tests for App Startup and Environment Compatibility.
Ensures the app runs cleanly without GEMINI_API_KEY.
"""

import os
import pytest
import config


def test_config_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Must not raise an exception
    key = config.get_gemini_api_key()
    assert key is None
    assert config.is_gemini_configured() is False


def test_import_services():
    import config
    import services.entity_service
    import services.retrieval_service
    import services.medical_service
    import scripts.prepare_medquad

    assert config.MEDICAL_DISCLAIMER is not None
    assert services.entity_service.entity_service is not None
    assert services.retrieval_service.retrieval_service is not None
    assert services.medical_service.medical_service is not None
