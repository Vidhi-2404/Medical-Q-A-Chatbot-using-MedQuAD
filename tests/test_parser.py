"""
Tests for MedQuAD XML parsing, error handling, and deduplication.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd
from scripts.prepare_medquad import MedQuADParser, compute_record_hash


@pytest.fixture
def temp_xml_dir():
    """Create a temporary directory with synthetic MedQuAD XML fixtures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Valid XML
        valid_xml = tmp_path / "valid_doc.xml"
        valid_xml.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <Document id="NCI_001" source="CancerGov" url="https://cancer.gov/lung">
                <Focus>Lung Cancer</Focus>
                <QAPairs>
                    <QAPair id="0">
                        <Question qtype="symptoms">What are the symptoms of lung cancer?</Question>
                        <Answer>Common symptoms include a persistent cough, coughing up blood, and chest pain.</Answer>
                    </QAPair>
                    <QAPair id="1">
                        <Question qtype="treatment">What are the treatments for lung cancer?</Question>
                        <Answer>Treatments include surgery, chemotherapy, radiation therapy, and targeted therapy.</Answer>
                    </QAPair>
                </QAPairs>
            </Document>""",
            encoding="utf-8"
        )

        # 2. XML with missing / empty fields
        missing_fields_xml = tmp_path / "missing_fields.xml"
        missing_fields_xml.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <Document id="NCI_002" source="CancerGov" url="https://cancer.gov/test">
                <Focus>Test Condition</Focus>
                <QAPairs>
                    <QAPair id="0">
                        <Question qtype="information">What is test condition?</Question>
                        <!-- Missing Answer tag -->
                    </QAPair>
                    <QAPair id="1">
                        <Question qtype="information">Short answer question?</Question>
                        <Answer>Too short</Answer>
                    </QAPair>
                </QAPairs>
            </Document>""",
            encoding="utf-8"
        )

        # 3. Malformed XML
        malformed_xml = tmp_path / "malformed.xml"
        malformed_xml.write_text(
            """<Document id="NCI_003">
                <Focus>Broken File
                <QAPairs>
                    <QAPair>
            """,
            encoding="utf-8"
        )

        # 4. Duplicate XML (same Q&A as valid_xml QAPair 0)
        duplicate_xml = tmp_path / "duplicate.xml"
        duplicate_xml.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <Document id="NCI_004" source="CancerGov" url="https://cancer.gov/lung">
                <Focus>Lung Cancer</Focus>
                <QAPairs>
                    <QAPair id="0">
                        <Question qtype="symptoms">What are the symptoms of lung cancer?</Question>
                        <Answer>Common symptoms include a persistent cough, coughing up blood, and chest pain.</Answer>
                    </QAPair>
                </QAPairs>
            </Document>""",
            encoding="utf-8"
        )

        yield tmp_path


def test_parse_valid_xml(temp_xml_dir):
    parser = MedQuADParser()
    valid_file = temp_xml_dir / "valid_doc.xml"
    records = parser.parse_xml_file(valid_file)

    assert len(records) == 2
    assert records[0]["focus"] == "Lung Cancer"
    assert records[0]["source"] == "CancerGov"
    assert records[0]["qtype"] == "symptoms"
    assert "persistent cough" in records[0]["answer"]
    assert parser.stats["valid_records"] == 2


def test_skip_missing_or_short_fields(temp_xml_dir):
    parser = MedQuADParser()
    missing_file = temp_xml_dir / "missing_fields.xml"
    records = parser.parse_xml_file(missing_file)

    assert len(records) == 0
    assert parser.stats["skipped_records"] == 2


def test_handle_malformed_xml(temp_xml_dir):
    parser = MedQuADParser()
    malformed_file = temp_xml_dir / "malformed.xml"
    records = parser.parse_xml_file(malformed_file)

    assert len(records) == 0
    assert parser.stats["malformed_files"] == 1


def test_deduplication_and_repeatable_imports(temp_xml_dir):
    parser = MedQuADParser()
    valid_file = temp_xml_dir / "valid_doc.xml"
    dup_file = temp_xml_dir / "duplicate.xml"

    # First parse valid
    records_1 = parser.parse_xml_file(valid_file)
    assert len(records_1) == 2

    # Second parse duplicate
    records_2 = parser.parse_xml_file(dup_file)
    assert len(records_2) == 0
    assert parser.stats["duplicate_records"] == 1


def test_directory_processing(temp_xml_dir):
    parser = MedQuADParser()
    df = parser.process_directory(temp_xml_dir)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Only the 2 valid unique records
    assert parser.stats["total_files_scanned"] == 4
    assert parser.stats["malformed_files"] == 1
    assert parser.stats["skipped_records"] == 2
    assert parser.stats["duplicate_records"] == 1
