"""
Entity recognition service for clinical and medical concepts.
Extracts Symptoms, Diseases/Conditions, Medications, and Treatments
using a fast, deterministic lightweight rule-and-lexicon baseline,
and detects life-threatening emergency triggers.
"""

import re
from typing import Dict, List, Set
from config import EMERGENCY_KEYWORDS


class EntityRecognitionService:
    """
    Lightweight medical entity extraction service covering:
    - Symptoms
    - Diseases / Conditions
    - Medications / Drugs
    - Treatments / Procedures
    """

    def __init__(self):
        # Comprehensive curated medical lexicons aligned with MedQuAD topics
        self.symptoms_lexicon = {
            "fever", "high temperature", "chills", "cough", "dry cough", "productive cough",
            "chest pain", "crushing chest pain", "shortness of breath", "dyspnea",
            "difficulty breathing", "wheezing", "fatigue", "exhaustion", "weakness",
            "headache", "migraine", "dizziness", "lightheadedness", "vertigo",
            "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain",
            "stomach ache", "bloating", "loss of appetite", "weight loss", "unexplained weight loss",
            "weight gain", "joint pain", "swollen joints", "muscle aches", "myalgia",
            "rash", "skin lesion", "itching", "pruritus", "hives", "bruising",
            "sore throat", "runny nose", "congestion", "sinus pressure", "sneezing",
            "swelling", "edema", "palpitations", "irregular heartbeat", "tachycardia",
            "tremor", "numbness", "tingling", "paresthesia", "blurred vision",
            "insomnia", "excessive thirst", "frequent urination", "blood in urine",
            "hematuria", "jaundice", "yellowing of skin", "confusion", "memory loss"
        }

        self.diseases_lexicon = {
            "diabetes", "type 1 diabetes", "type 2 diabetes", "gestational diabetes",
            "hypertension", "high blood pressure", "hypotension", "low blood pressure",
            "asthma", "chronic obstructive pulmonary disease", "copd", "bronchitis", "emphysema",
            "pneumonia", "covid-19", "influenza", "flu", "tuberculosis",
            "breast cancer", "lung cancer", "prostate cancer", "colorectal cancer", "melanoma",
            "leukemia", "lymphoma", "cancer", "tumor", "carcinoma",
            "coronary artery disease", "heart disease", "heart failure", "angina", "myocardial infarction",
            "stroke", "transient ischemic attack", "tia", "arrhythmia", "atrial fibrillation",
            "arthritis", "osteoarthritis", "rheumatoid arthritis", "gout", "osteoporosis",
            "alzheimer's disease", "dementia", "parkinson's disease", "multiple sclerosis", "epilepsy",
            "depression", "anxiety disorder", "bipolar disorder", "schizophrenia",
            "celiac disease", "crohn's disease", "ulcerative colitis", "irritable bowel syndrome", "ibs",
            "chronic kidney disease", "kidney failure", "kidney stones", "cirrhosis", "hepatitis",
            "hypothyroidism", "hyperthyroidism", "hashimoto's thyroiditis", "graves' disease",
            "anemia", "iron deficiency anemia", "sickle cell disease", "thalassemia",
            "glaucoma", "cataracts", "macular degeneration", "psoriasis", "eczema", "lupus"
        }

        self.medications_lexicon = {
            "metformin", "insulin", "glipizide", "sitagliptin", "empagliflozin",
            "lisinopril", "losartan", "amlodipine", "metoprolol", "atenolol", "hydrochlorothiazide",
            "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
            "aspirin", "clopidogrel", "warfarin", "apixaban", "rivaroxaban", "heparin",
            "ibuprofen", "acetaminophen", "paracetamol", "naproxen", "celecoxib", "tramadol",
            "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "cephalexin", "penicillin",
            "albuterol", "fluticasone", "montelukast", "budesonide", "prednisone", "methylprednisolone",
            "omeprazole", "pantoprazole", "famotidine", "ranitidine", "esomeprazole",
            "sertraline", "fluoxetine", "escitalopram", "duloxetine", "bupropion",
            "levothyroxine", "synthroid", "gabapentin", "pregabalin", "allopurinol",
            "methotrexate", "adalimumab", "etanercept", "cyclophosphamide", "doxorubicin", "paclitaxel"
        }

        self.treatments_lexicon = {
            "chemotherapy", "radiation therapy", "radiotherapy", "immunotherapy",
            "surgery", "surgical resection", "biopsy", "mastectomy", "angioplasty", "bypass surgery",
            "physical therapy", "physiotherapy", "occupational therapy", "speech therapy",
            "dialysis", "hemodialysis", "peritoneal dialysis", "kidney transplant", "organ transplant",
            "vaccination", "immunization", "booster shot", "blood transfusion",
            "dietary modification", "exercise regimen", "weight management", "smoking cessation",
            "cardiac rehabilitation", "cognitive behavioral therapy", "psychotherapy", "counseling",
            "mechanical ventilation", "oxygen therapy", "cpap", "pacemaker implantation", "stent placement"
        }

        # Precompile regex search patterns sorted by length descending (to match multi-word first)
        self._symptoms_pattern = self._build_pattern(self.symptoms_lexicon)
        self._diseases_pattern = self._build_pattern(self.diseases_lexicon)
        self._medications_pattern = self._build_pattern(self.medications_lexicon)
        self._treatments_pattern = self._build_pattern(self.treatments_lexicon)

    @staticmethod
    def _build_pattern(lexicon: Set[str]) -> re.Pattern:
        """Create regex pattern matching whole words/phrases, longest first."""
        sorted_terms = sorted(lexicon, key=lambda s: len(s), reverse=True)
        escaped = [re.escape(term) for term in sorted_terms]
        pattern_str = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern_str, re.IGNORECASE)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract symptoms, diseases, medications, and treatments from the input text.
        Returns a dictionary with list of unique normalized terms found.
        """
        if not text or not text.strip():
            return {
                "symptoms": [],
                "diseases": [],
                "medications": [],
                "treatments": []
            }

        symptoms = sorted(list({m.group(0).lower() for m in self._symptoms_pattern.finditer(text)}))
        diseases = sorted(list({m.group(0).lower() for m in self._diseases_pattern.finditer(text)}))
        medications = sorted(list({m.group(0).lower() for m in self._medications_pattern.finditer(text)}))
        treatments = sorted(list({m.group(0).lower() for m in self._treatments_pattern.finditer(text)}))

        return {
            "symptoms": symptoms,
            "diseases": diseases,
            "medications": medications,
            "treatments": treatments
        }

    def check_emergency(self, text: str) -> bool:
        """
        Check if the text contains high-risk emergency keywords indicating
        an urgent medical situation.
        """
        if not text:
            return False
        text_lower = text.lower()
        for kw in EMERGENCY_KEYWORDS:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                return True
        return False


# Singleton instance for application reuse
entity_service = EntityRecognitionService()
