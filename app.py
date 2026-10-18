"""
Medical Q&A Chatbot using MedQuAD
A standalone, evidence-grounded clinical question-answering system.
Runs out-of-the-box on Streamlit Community Cloud with zero API keys required.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

import config
from services.medical_service import medical_service
from services.retrieval_service import retrieval_service
from services.entity_service import entity_service

# Page configuration
st.set_page_config(
    page_title="MedQuAD Medical Q&A",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .emergency-box {
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        color: #991B1B;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .answer-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-symptom { background-color: #FEF08A; color: #854D0E; }
    .badge-disease { background-color: #FECACA; color: #991B1B; }
    .badge-med { background-color: #BFDBFE; color: #1E40AF; }
    .badge-treatment { background-color: #BBF7D0; color: #166534; }
    .mode-indicator {
        font-size: 0.85rem;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .mode-retrieval {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .mode-gemini {
        background-color: #FDF2F8;
        color: #9D174D;
        border: 1px solid #FBCFE8;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None


init_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/US-NIH-Library-Logo.svg/320px-US-NIH-Library-Logo.svg.png", width=140)
    st.title("Settings & Knowledge Base")

    # Mode Selector
    gemini_ready = config.is_gemini_configured()
    if gemini_ready:
        mode_choice = st.radio(
            "Answering Engine Mode:",
            options=["Free Retrieval Mode (Default)", "Gemini AI Mode (Configured)"],
            index=0,
            help="Switch between local offline TF-IDF retrieval and Google Gemini synthesis."
        )
        selected_mode = "gemini" if "Gemini" in mode_choice else "retrieval_free"
    else:
        st.info("💡 **Active Mode: Free Retrieval Mode**\n\nRuns 100% locally with zero external API calls or keys required. Ideal for Streamlit Community Cloud.")
        selected_mode = "retrieval_free"

    # Search parameters
    top_k = st.slider("Number of Evidence References (Top-K):", min_value=1, max_value=5, value=3)

    # Knowledge Base Stats
    st.divider()
    st.subheader("📊 Dataset Statistics")
    stats = retrieval_service.get_dataset_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Q&As", stats["total_records"])
    with col2:
        st.metric("Sources", stats["sources_count"])
    st.caption(f"Unique Medical Topics: **{stats['unique_topics']}**")

    # Quick sample queries
    st.divider()
    st.subheader("💡 Example Queries")
    sample_queries = [
        "What are the symptoms of diabetes?",
        "What are the treatments for breast cancer?",
        "How is hypertension treated?",
        "What is metformin and what is it used for?",
        "What are the signs of a stroke?"
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            st.session_state["query_input"] = sq
            st.rerun()

    # Ingestion / Maintenance Expander
    st.divider()
    with st.expander("🛠️ Knowledge Base Management"):
        st.write("MedQuAD Knowledge Base Status:")
        st.write(f"- Indexed: **{'Yes' if stats['is_indexed'] else 'No'}**")
        if st.button("Reload Knowledge Base Index", use_container_width=True):
            with st.spinner("Rebuilding TF-IDF index..."):
                retrieval_service.load_data()
            st.success("Index reloaded successfully!")
            st.rerun()


# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-header">🩺 Medical Q&A Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evidence-grounded medical question answering powered by the NIH MedQuAD dataset (CC BY 4.0).</div>',
    unsafe_allow_html=True
)

# Mandatory Medical Disclaimer Banner
st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Medical Disclaimer:</strong> {config.MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)

# Question Input Section
query_val = st.text_input(
    "Enter your medical inquiry or health question:",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g., What are the treatments for high blood pressure? What are the symptoms of asthma?",
    key="main_question_box"
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        st.session_state["query_input"] = ""
        st.session_state["last_response"] = None
        st.rerun()

# Execute Query
if ask_button and query_val.strip():
    with st.spinner("Analyzing inquiry and retrieving medical evidence from MedQuAD..."):
        response = medical_service.answer_query(
            query=query_val,
            mode=selected_mode,
            top_k=top_k
        )
        st.session_state["last_response"] = response

# Display Results
if st.session_state.get("last_response"):
    res = st.session_state["last_response"]

    # 1. Emergency Warning Banner if detected
    if res.get("is_emergency"):
        st.markdown(
            f'<div class="emergency-box">🚨 {res["emergency_alert"]}</div>',
            unsafe_allow_html=True
        )

    # 2. Mode Label Indicator
    mode_class = "mode-gemini" if res.get("mode_used") == "gemini" else "mode-retrieval"
    st.markdown(
        f'<div class="mode-indicator {mode_class}">📌 {res.get("mode_label")}</div>',
        unsafe_allow_html=True
    )

    # 3. Main Answer Panel
    st.markdown("### 📋 Evidence-Grounded Answer")
    st.markdown(f'<div class="answer-card">{res["answer"]}</div>', unsafe_allow_html=True)

    # 4. Recognized Clinical Entities
    entities = res.get("recognized_entities", {})
    has_entities = any(entities.values())
    if has_entities:
        st.markdown("### 🧬 Recognized Clinical Concepts")
        ent_cols = st.columns(4)
        with ent_cols[0]:
            st.markdown("**Symptoms**")
            if entities.get("symptoms"):
                for s in entities["symptoms"]:
                    st.markdown(f'<span class="badge badge-symptom">🟡 {s.title()}</span>', unsafe_allow_html=True)
            else:
                st.caption("None detected")
        with ent_cols[1]:
            st.markdown("**Conditions / Diseases**")
            if entities.get("diseases"):
                for d in entities["diseases"]:
                    st.markdown(f'<span class="badge badge-disease">🔴 {d.title()}</span>', unsafe_allow_html=True)
            else:
                st.caption("None detected")
        with ent_cols[2]:
            st.markdown("**Medications**")
            if entities.get("medications"):
                for m in entities["medications"]:
                    st.markdown(f'<span class="badge badge-med">💊 {m.title()}</span>', unsafe_allow_html=True)
            else:
                st.caption("None detected")
        with ent_cols[3]:
            st.markdown("**Treatments**")
            if entities.get("treatments"):
                for t in entities["treatments"]:
                    st.markdown(f'<span class="badge badge-treatment">💉 {t.title()}</span>', unsafe_allow_html=True)
            else:
                st.caption("None detected")

    # 5. Retrieved Evidence and Attribution References
    refs = res.get("references", [])
    if refs:
        st.markdown("### 📚 Retrieved MedQuAD Evidence & Source Attribution")
        for idx, ref in enumerate(refs, 1):
            source_name = ref.get("source", "NIH MedQuAD")
            topic = ref.get("focus", "Medical Condition")
            score = ref.get("relevance_pct", "N/A")
            url = ref.get("url", "")
            with st.expander(f"Reference #{idx}: {topic} — {source_name} (Relevance: {score})", expanded=(idx == 1)):
                st.markdown(f"**Original Question:** {ref.get('question')}")
                st.markdown(f"**Verified Evidence Answer:**\n\n{ref.get('answer')}")
                st.markdown(f"**Question Type:** `{ref.get('qtype', 'general')}` | **Record ID:** `{ref.get('id')}`")
                if url:
                    st.markdown(f"🔗 **Source Link:** [{url}]({url})")

st.divider()
st.caption(
    "Dataset: MedQuAD (Medical Question Answering Dataset) created by Asma Ben Abacha & Dina Demner-Fushman, NIH. "
    "Distributed under Creative Commons Attribution 4.0 International License (CC BY 4.0)."
)
