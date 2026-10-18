# medical-qa-medquad-chatbot 🩺
### Standalone Medical Q&A Chatbot using the NIH MedQuAD Dataset

A production-grade, evidence-grounded clinical question-answering application built with **Streamlit**, **Scikit-Learn (TF-IDF)**, **ChromaDB**, and lightweight NLP entity recognition. 

The chatbot operates **100% out-of-the-box without requiring any API key** on **Streamlit Community Cloud**, using local TF-IDF semantic retrieval directly over verified National Institutes of Health (NIH) literature. It optionally supports Google Gemini for generative answer synthesis when a `GEMINI_API_KEY` is provided.

---

## 📋 Features

- 🆓 **Zero API-Key Dependency:** Designed to run freely on Streamlit Community Cloud without requiring any paid or external API key.
- 🔬 **Evidence-Grounded Retrieval:** Uses a fast, memory-efficient TF-IDF vector retrieval engine with cosine similarity over MedQuAD records.
- 🏷️ **Clinical Entity Recognition:** Lightweight NLP extraction identifying symptoms, diseases, medications, and treatments with visual badges.
- 🚨 **Emergency Condition Detection:** Detects high-risk symptoms (e.g., crushing chest pain, stroke signs, severe dyspnea) and immediately provides life-safety emergency guidance.
- 📚 **Full Attribution & Source Linking:** Every response displays source NIH institutes (NCI, NHLBI, NIDDK, CDC, GARD, MedlinePlus) and direct URLs.
- 🛡️ **Truthfulness & Safety:** If relevant evidence is unavailable, the system states this clearly instead of hallucinating answers.
- 🤖 **Optional Gemini Backend:** If `GEMINI_API_KEY` is configured in Streamlit secrets or `.env`, users can toggle Gemini generative answering.

---

## 🏛️ Dataset & Licensing

- **Dataset:** MedQuAD (Medical Question Answering Dataset) created by Asma Ben Abacha & Dina Demner-Fushman.
- **Repository:** [https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- **Citation:**
  > Asma Ben Abacha and Dina Demner-Fushman. *A Question-Entailment Approach to Question Answering*. BMC Bioinformatics, 2019.

*Note on Dataset Status:* The repository includes a representative curated starter dataset (`data/sample_medquad.csv`, 18 authentic records) so that the application runs immediately without downloading the multi-hundred-megabyte raw XML repository. The full 47,000+ MedQuAD XML files can be parsed and indexed locally using `scripts/prepare_medquad.py`.

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/medical-qa-medquad-chatbot.git
cd medical-qa-medquad-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Ingesting the Full MedQuAD Dataset

To process the full dataset from source:
1. Clone the MedQuAD repository into `data/raw/MedQuAD`:
   ```bash
   git clone https://github.com/abachaa/MedQuAD.git data/raw/MedQuAD
   ```
2. Run the parser script:
   ```bash
   python scripts/prepare_medquad.py --input-dir data/raw/MedQuAD --output-csv data/processed/medquad_processed.csv
   ```
   The parser will report total XML files scanned, valid extracted records, skipped empty records, malformed files, and deduplicated records. Repeatable imports are supported without duplicating records.

To re-generate the bundled sample dataset:
```bash
python scripts/prepare_medquad.py --generate-sample
```

### 3. Run Automated Tests

Execute the 21 automated unit and integration tests:
```bash
python -m pytest -v
```

### 4. Launch the Streamlit App Locally

```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Deploying to Streamlit Community Cloud (No API Key Required)

Deploy this app to the public web in under 3 minutes for free:

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Standalone MedQuAD Medical Q&A Chatbot"
   git branch -M main
   git remote add origin https://github.com/<your-username>/medical-qa-medquad-chatbot.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
   - Click **"New app"**.
   - Select your repository: `medical-qa-medquad-chatbot`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - Click **"Deploy!"**.

3. **No API Key Required:**
   - The app will automatically launch in **Free Retrieval Mode (TF-IDF)** without needing any secrets configured.

4. **(Optional) Enable Gemini Mode on Streamlit Cloud:**
   - In your app's dashboard on Streamlit Community Cloud, click **Settings** -> **Secrets**.
   - Add:
     ```toml
     GEMINI_API_KEY = "your-google-gemini-api-key"
     ```
   - Save. The app will automatically detect the key and enable the Gemini AI Mode toggle in the sidebar.

---

## 🧪 Architecture & Services

| Module | Description |
|---|---|
| `app.py` | Streamlit user interface, disclaimer banners, entity badges, and source attribution cards. |
| `config.py` | Safe API key loader, dataset paths, medical disclaimers, and emergency thresholds. |
| `services/entity_service.py` | Lightweight NLP entity recognition for Symptoms, Diseases, Medications, and Treatments, plus emergency condition detection. |
| `services/retrieval_service.py` | Local TF-IDF search engine with cosine similarity and optional ChromaDB integration. |
| `services/medical_service.py` | Medical inquiry orchestrator coordinating emergency checks, entity extraction, retrieval synthesis, and optional Gemini mode. |
| `scripts/prepare_medquad.py` | Robust XML parser handling missing fields, malformed files, deduplication, and statistics reporting. |

---

## ⚠️ Medical Safety & Disclaimer

> **Important:** This chatbot is an educational and informational demonstration project. It is **NOT** a certified medical diagnostic device or clinician substitute. It does not prescribe medications, calculate dosages, or provide definitive diagnoses. If you or someone you know is experiencing severe symptoms (e.g., chest pain, difficulty breathing, stroke warning signs), please contact local emergency medical services immediately.
