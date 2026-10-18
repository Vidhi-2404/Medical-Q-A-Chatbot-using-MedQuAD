"""
Configuration module for the Medical Q&A MedQuAD Chatbot.
Handles environment variables, Streamlit secrets, directory paths, and medical safety constants.
"""

import os
from pathlib import Path
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "medquad_processed.csv"
SAMPLE_CSV_PATH = DATA_DIR / "sample_medquad.csv"

# Ensure essential directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_gemini_api_key() -> Optional[str]:
    """
    Safely retrieve the Gemini API key from environment variables or Streamlit secrets.
    Never crashes if the key is missing; returns None instead.
    """
    # 1. Check standard environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key_val = st.secrets["GEMINI_API_KEY"]
            if key_val and str(key_val).strip():
                return str(key_val).strip()
    except Exception:
        pass

    return None


def is_gemini_configured() -> bool:
    """Return True if a valid Gemini API key is found."""
    return get_gemini_api_key() is not None


# Retrieval Settings
DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.12  # Minimum TF-IDF / similarity score to consider relevant
CHROMA_COLLECTION_NAME = "medquad_collection"

# Safety and Medical Disclaimers
MEDICAL_DISCLAIMER = (
    "Disclaimer: This application is an educational and informational tool based on the "
    "National Institutes of Health (NIH) MedQuAD dataset. It is NOT a substitute for "
    "professional medical advice, clinical diagnosis, prescription, or treatment. "
    "Always consult a qualified healthcare provider with questions regarding a medical condition. "
    "Never disregard professional medical advice or delay seeking it because of information found here."
)

EMERGENCY_WARNING = (
    "CRITICAL WARNING: The symptoms described may indicate a medical emergency! "
    "Please call your local emergency service immediately (e.g., 911 in the US/Canada, 999 in the UK, 112 in Europe, 108 in India) "
    "or proceed to the nearest emergency department right away. Do not rely on this application during an emergency."
)

# High-risk / emergency symptoms keywords
EMERGENCY_KEYWORDS = [
    "chest pain", "crushing chest", "heart attack", "difficulty breathing",
    "cannot breathe", "shortness of breath", "severe shortness of breath",
    "stroke", "facial drooping", "arm weakness", "slurred speech",
    "sudden numbness", "anaphylaxis", "severe allergic reaction",
    "throat swelling", "coughing blood", "heavy bleeding", "unconscious",
    "loss of consciousness", "overdose", "poisoning", "suicide", "suicidal",
    "severe head injury", "seizure", "convulsions"
]
