"""
MedQuAD Dataset Preparation and Parser Script.
Parses MedQuAD XML files, extracts questions, answers, focus entities,
question types, and source attributions.
Reports valid, skipped, malformed, and duplicate records.
Supports repeatable imports and sample dataset generation.
"""

import argparse
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

import sys
# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def normalize_text(text: Optional[str]) -> str:
    """Normalize text by stripping whitespace and removing excessive internal spaces."""
    if not text:
        return ""
    # Clean XML entity artifacts or multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_record_hash(question: str, answer: str) -> str:
    """Create a unique deterministic hash for deduplication based on question and answer."""
    norm_q = normalize_text(question).lower()
    norm_a = normalize_text(answer).lower()
    key = f"{norm_q}###{norm_a}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MedQuADParser:
    """
    Parser for National Institutes of Health MedQuAD XML dataset files.
    """

    def __init__(self, existing_hashes: Optional[Set[str]] = None):
        self.seen_hashes: Set[str] = set(existing_hashes) if existing_hashes else set()
        self.stats = {
            "total_files_scanned": 0,
            "valid_records": 0,
            "skipped_records": 0,
            "malformed_files": 0,
            "duplicate_records": 0,
            "sources": {},
            "categories": {}
        }

    def parse_xml_file(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Parse a single MedQuAD XML file.
        Returns a list of valid QA record dictionaries.
        """
        records = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except (ET.ParseError, UnicodeDecodeError, OSError) as e:
            self.stats["malformed_files"] += 1
            return []

        doc_id = root.attrib.get("id", file_path.stem)
        doc_source = root.attrib.get("source", "NIH / MedQuAD")
        doc_url = root.attrib.get("url", "")

        # Extract focus entity
        focus_elem = root.find("Focus")
        if focus_elem is None:
            focus_elem = root.find("focus")
        focus = normalize_text(focus_elem.text) if focus_elem is not None and focus_elem.text else ""

        # Extract QA pairs
        qa_pairs = root.findall(".//QAPair")
        if not qa_pairs:
            qa_pairs = root.findall(".//qapair")

        for qa in qa_pairs:
            qa_id = qa.attrib.get("id", qa.attrib.get("pid", ""))
            q_elem = qa.find("Question")
            if q_elem is None:
                q_elem = qa.find("question")
            
            a_elem = qa.find("Answer")
            if a_elem is None:
                a_elem = qa.find("answer")

            if q_elem is None or a_elem is None:
                self.stats["skipped_records"] += 1
                continue

            question_text = normalize_text(q_elem.text)
            answer_text = normalize_text(a_elem.text)
            qtype = q_elem.attrib.get("qtype", "general").strip().lower()

            # Empty question or answer
            if not question_text or not answer_text or len(answer_text) < 10:
                self.stats["skipped_records"] += 1
                continue

            # Deduplication check
            rec_hash = compute_record_hash(question_text, answer_text)
            if rec_hash in self.seen_hashes:
                self.stats["duplicate_records"] += 1
                continue

            self.seen_hashes.add(rec_hash)
            self.stats["valid_records"] += 1

            # Update stats
            self.stats["sources"][doc_source] = self.stats["sources"].get(doc_source, 0) + 1
            self.stats["categories"][qtype] = self.stats["categories"].get(qtype, 0) + 1

            records.append({
                "id": f"{doc_id}_{qa_id}" if qa_id else f"{doc_id}_{len(records)}",
                "question": question_text,
                "answer": answer_text,
                "focus": focus or "Medical Topic",
                "qtype": qtype or "general",
                "source": doc_source or "NIH MedQuAD",
                "url": doc_url or "https://medlineplus.gov/",
                "hash": rec_hash
            })

        return records

    def process_directory(self, input_dir: Path) -> pd.DataFrame:
        """
        Recursively scan input directory for XML files and parse them.
        """
        all_records = []
        xml_files = list(input_dir.rglob("*.xml"))
        self.stats["total_files_scanned"] = len(xml_files)

        for xml_file in xml_files:
            file_records = self.parse_xml_file(xml_file)
            all_records.extend(file_records)

        df = pd.DataFrame(all_records)
        return df


def load_existing_hashes(csv_path: Path) -> Set[str]:
    """Load existing record hashes to prevent duplicates on repeated imports."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["hash"])
        return set(df["hash"].dropna().astype(str).tolist())
    except Exception:
        return set()


def create_sample_dataset(output_path: Path) -> pd.DataFrame:
    """
    Generate an authentic sample dataset representing NIH MedQuAD QA pairs
    across various diseases, symptoms, treatments, and question types.
    Ensures immediate out-of-the-box functionality without requiring 
    a multi-hundred-megabyte raw XML download.
    """
    sample_records = [
        {
            "id": "CancerGov_1_0",
            "question": "What is breast cancer?",
            "answer": "Breast cancer is a disease in which cells in the breast grow out of control. There are different kinds of breast cancer. The kind of breast cancer depends on which cells in the breast turn into cancer. Breast cancer can begin in different parts of the breast, such as the lobules, ducts, or connective tissue.",
            "focus": "Breast Cancer",
            "qtype": "information",
            "source": "National Cancer Institute (Cancer.gov)",
            "url": "https://www.cancer.gov/types/breast"
        },
        {
            "id": "CancerGov_1_1",
            "question": "What are the treatments for breast cancer?",
            "answer": "Treatments for breast cancer depend on the type and stage of cancer. Standard treatment options include surgery (such as lumpectomy or mastectomy), radiation therapy, chemotherapy, hormone therapy (such as tamoxifen or aromatase inhibitors), targeted drug therapy, and immunotherapy.",
            "focus": "Breast Cancer",
            "qtype": "treatment",
            "source": "National Cancer Institute (Cancer.gov)",
            "url": "https://www.cancer.gov/types/breast/treatment"
        },
        {
            "id": "CancerGov_1_2",
            "question": "What are the symptoms of breast cancer?",
            "answer": "Symptoms of breast cancer include a new lump or mass in the breast or underarm, thickening or swelling of part of the breast, irritation or dimpling of breast skin, redness or flaky skin in the nipple area, pulling in of the nipple, and nipple discharge other than breast milk.",
            "focus": "Breast Cancer",
            "qtype": "symptoms",
            "source": "National Cancer Institute (Cancer.gov)",
            "url": "https://www.cancer.gov/types/breast/symptoms"
        },
        {
            "id": "NIDDK_1_0",
            "question": "What is type 2 diabetes?",
            "answer": "Type 2 diabetes is a chronic condition that occurs when blood glucose, also called blood sugar, is too high. With type 2 diabetes, your cells do not respond normally to insulin; this is called insulin resistance. Your pancreas produces more insulin to try to get cells to respond, but eventually cannot keep up.",
            "focus": "Type 2 Diabetes",
            "qtype": "information",
            "source": "NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)",
            "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-type-2-diabetes"
        },
        {
            "id": "NIDDK_1_1",
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, numbness or tingling in the hands or feet, sores that do not heal, and unexplained weight loss.",
            "focus": "Type 2 Diabetes",
            "qtype": "symptoms",
            "source": "NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)",
            "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes"
        },
        {
            "id": "NIDDK_1_2",
            "question": "How is type 2 diabetes treated?",
            "answer": "Treatment for type 2 diabetes includes healthy eating, regular physical activity, weight management, and blood glucose monitoring. Many people also need diabetes medications such as metformin, sulfonylureas, SGLT2 inhibitors, or insulin therapy to achieve target blood glucose levels.",
            "focus": "Type 2 Diabetes",
            "qtype": "treatment",
            "source": "NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)",
            "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/diet-eating-physical-activity"
        },
        {
            "id": "NHLBI_1_0",
            "question": "What is hypertension (high blood pressure)?",
            "answer": "Hypertension, also known as high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease and stroke.",
            "focus": "Hypertension",
            "qtype": "information",
            "source": "NHLBI (National Heart, Lung, and Blood Institute)",
            "url": "https://www.nhlbi.nih.gov/health/high-blood-pressure"
        },
        {
            "id": "NHLBI_1_1",
            "question": "What are the treatments for high blood pressure?",
            "answer": "Treating high blood pressure involves lifestyle changes such as eating a heart-healthy diet (like DASH), regular physical exercise, limiting sodium intake, maintaining a healthy weight, and avoiding smoking. When lifestyle modifications are insufficient, medications like ACE inhibitors (e.g. lisinopril), ARBs (losartan), calcium channel blockers (amlodipine), and thiazide diuretics are prescribed.",
            "focus": "Hypertension",
            "qtype": "treatment",
            "source": "NHLBI (National Heart, Lung, and Blood Institute)",
            "url": "https://www.nhlbi.nih.gov/health/high-blood-pressure/treatment"
        },
        {
            "id": "NHLBI_2_0",
            "question": "What are the warning signs of a heart attack?",
            "answer": "Warning signs of a heart attack include chest pain, pressure, squeezing, or fullness in the center of the chest lasting more than a few minutes; pain spreading to the shoulder, arm, back, neck, or jaw; shortness of breath; cold sweats; dizziness; and nausea. If you experience these signs, call 911 or local emergency services immediately.",
            "focus": "Heart Attack (Myocardial Infarction)",
            "qtype": "symptoms",
            "source": "NHLBI (National Heart, Lung, and Blood Institute)",
            "url": "https://www.nhlbi.nih.gov/health/heart-attack/warning-signs"
        },
        {
            "id": "NHLBI_3_0",
            "question": "What is asthma?",
            "answer": "Asthma is a chronic lung disease that affects your airways. In asthma, the airways can become inflamed and narrowed, making it harder to breathe. It causes recurring periods of wheezing, chest tightness, shortness of breath, and coughing.",
            "focus": "Asthma",
            "qtype": "information",
            "source": "NHLBI (National Heart, Lung, and Blood Institute)",
            "url": "https://www.nhlbi.nih.gov/health/asthma"
        },
        {
            "id": "NHLBI_3_1",
            "question": "How is asthma treated?",
            "answer": "Asthma is treated using long-term control medications (such as inhaled corticosteroids like fluticasone, or leukotriene modifiers like montelukast) to reduce inflammation, along with quick-relief rescue inhalers (short-acting beta agonists such as albuterol) to rapidly relax airway muscles during acute asthma flare-ups.",
            "focus": "Asthma",
            "qtype": "treatment",
            "source": "NHLBI (National Heart, Lung, and Blood Institute)",
            "url": "https://www.nhlbi.nih.gov/health/asthma/treatment"
        },
        {
            "id": "CDC_1_0",
            "question": "What is COVID-19?",
            "answer": "COVID-19 is an infectious disease caused by the SARS-CoV-2 coronavirus. It primarily affects the respiratory system and spreads through respiratory droplets produced when an infected person breathes, talks, coughs, or sneezes.",
            "focus": "COVID-19",
            "qtype": "information",
            "source": "CDC (Centers for Disease Control and Prevention)",
            "url": "https://www.cdc.gov/coronavirus/2019-ncov/index.html"
        },
        {
            "id": "CDC_1_1",
            "question": "What are the symptoms of COVID-19?",
            "answer": "COVID-19 symptoms range from mild to severe illness and include fever or chills, cough, shortness of breath, fatigue, muscle or body aches, headache, new loss of taste or smell, sore throat, congestion or runny nose, nausea, and diarrhea.",
            "focus": "COVID-19",
            "qtype": "symptoms",
            "source": "CDC (Centers for Disease Control and Prevention)",
            "url": "https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html"
        },
        {
            "id": "MedlinePlus_1_0",
            "question": "What is metformin and what is it used for?",
            "answer": "Metformin is an oral prescription medication used with a proper diet and exercise program to control high blood sugar in people with type 2 diabetes. It works by decreasing the amount of glucose absorbed from food and the amount of glucose made by the liver, and by increasing the body's sensitivity to insulin.",
            "focus": "Metformin",
            "qtype": "medication",
            "source": "MedlinePlus (National Library of Medicine)",
            "url": "https://medlineplus.gov/druginfo/meds/a684058.html"
        },
        {
            "id": "MedlinePlus_2_0",
            "question": "What is lisinopril and what are its uses?",
            "answer": "Lisinopril is an ACE (angiotensin-converting enzyme) inhibitor used alone or in combination with other medications to treat high blood pressure (hypertension), to improve survival after a heart attack, and to manage heart failure. It works by relaxing blood vessels so blood can flow more easily.",
            "focus": "Lisinopril",
            "qtype": "medication",
            "source": "MedlinePlus (National Library of Medicine)",
            "url": "https://medlineplus.gov/druginfo/meds/a687002.html"
        },
        {
            "id": "NINDS_1_0",
            "question": "What are the signs and symptoms of a stroke?",
            "answer": "The signs of a stroke occur suddenly and can be remembered by the FAST acronym: Face drooping on one side, Arm weakness or numbness, Speech difficulty or slurring, and Time to call 911 immediately. Other symptoms include sudden severe headache, sudden confusion, and trouble walking or seeing.",
            "focus": "Stroke",
            "qtype": "symptoms",
            "source": "NINDS (National Institute of Neurological Disorders and Stroke)",
            "url": "https://www.ninds.nih.gov/health-information/disorders/stroke"
        },
        {
            "id": "GARD_1_0",
            "question": "What is Celiac Disease?",
            "answer": "Celiac disease is a serious autoimmune disorder that occurs in genetically predisposed people where the ingestion of gluten leads to damage in the small intestine. When people with celiac disease eat gluten, an immune response attacks the small intestine villi, impairing nutrient absorption.",
            "focus": "Celiac Disease",
            "qtype": "information",
            "source": "GARD (Genetic and Rare Diseases Information Center)",
            "url": "https://rarediseases.info.nih.gov/diseases/11998/celiac-disease"
        },
        {
            "id": "GARD_1_1",
            "question": "What is the treatment for Celiac Disease?",
            "answer": "The only effective treatment for celiac disease is a strict, lifelong gluten-free diet. Avoiding all foods and products containing wheat, rye, and barley allows the intestinal villi to heal and prevents further complications. Nutritional supplements may also be needed to address vitamin deficiencies.",
            "focus": "Celiac Disease",
            "qtype": "treatment",
            "source": "GARD (Genetic and Rare Diseases Information Center)",
            "url": "https://rarediseases.info.nih.gov/diseases/11998/celiac-disease#treatmentSection"
        }
    ]

    for rec in sample_records:
        rec["hash"] = compute_record_hash(rec["question"], rec["answer"])

    df = pd.DataFrame(sample_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare and parse MedQuAD XML files")
    parser.add_argument("--input-dir", type=str, default=str(config.RAW_DATA_DIR),
                        help="Path to folder containing raw MedQuAD XML files")
    parser.add_argument("--output-csv", type=str, default=str(config.PROCESSED_CSV_PATH),
                        help="Path for saving processed records CSV")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a starter sample MedQuAD dataset")
    args = parser.parse_args()

    output_csv = Path(args.output_csv)

    if args.generate_sample:
        sample_path = config.SAMPLE_CSV_PATH
        print(f"Generating starter MedQuAD sample at {sample_path}...")
        df_sample = create_sample_dataset(sample_path)
        # Also copy to processed if processed doesn't exist
        if not output_csv.exists():
            df_sample.to_csv(output_csv, index=False)
        print(f"Generated {len(df_sample)} sample records successfully.")
        return

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory '{input_dir}' does not exist.")
        print(f"Generating sample dataset instead at {config.SAMPLE_CSV_PATH}...")
        create_sample_dataset(config.SAMPLE_CSV_PATH)
        return

    existing_hashes = load_existing_hashes(output_csv)
    med_parser = MedQuADParser(existing_hashes=existing_hashes)

    print(f"Scanning XML files in {input_dir}...")
    df_new = med_parser.process_directory(input_dir)

    print("--- MedQuAD Ingestion Report ---")
    print(f"Total XML Files Scanned: {med_parser.stats['total_files_scanned']}")
    print(f"Valid Records Extracted: {med_parser.stats['valid_records']}")
    print(f"Skipped Records (empty/short): {med_parser.stats['skipped_records']}")
    print(f"Malformed Files: {med_parser.stats['malformed_files']}")
    print(f"Duplicate Records Skipped: {med_parser.stats['duplicate_records']}")

    if not df_new.empty:
        if output_csv.exists() and len(existing_hashes) > 0:
            df_existing = pd.read_csv(output_csv)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=["hash"], inplace=True)
            df_combined.to_csv(output_csv, index=False)
            print(f"Appended records. Total dataset size: {len(df_combined)} records at {output_csv}")
        else:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(output_csv, index=False)
            print(f"Saved {len(df_new)} records to {output_csv}")
    else:
        print("No new valid records found to write.")


if __name__ == "__main__":
    main()
