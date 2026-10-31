"""Unit tests for FertiScope Phase 1: Research & Data Foundation."""

import json
from pathlib import Path
import pandas as pd
import pytest

from fertiscope.core.exceptions import FertiScopeError
from fertiscope.data.audit import RawDataAuditor
from fertiscope.data.provenance import (
    PRIMARY_CANDIDATE_PROVENANCE,
    DatasetProvenanceRecord,
    compute_sha256,
)
from fertiscope.data.schema import PatientRecordSchema, validate_dataframe_schema


def test_provenance_record_immutability_and_export(tmp_path: Path):
    """Test that provenance metadata is structured, immutable, and JSON-exportable."""
    dest = tmp_path / "provenance.json"
    PRIMARY_CANDIDATE_PROVENANCE.save_json(dest)

    assert dest.exists()
    with open(dest, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["doi"] == "10.17632/x5s3ky9cpv.1"
    assert data["reported_cohort_size"] == 2542
    assert "451" in data["verification_status"]
    assert len(data["expected_files"]) == 2


def test_sha256_computation(tmp_path: Path):
    """Verify cryptographic SHA-256 computation on local files."""
    dummy_file = tmp_path / "test_sample.txt"
    dummy_file.write_text("fertiscope_reproducibility_test_string", encoding="utf-8")

    checksum = compute_sha256(dummy_file)
    assert isinstance(checksum, str)
    assert len(checksum) == 64

    # Non-existent file should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        compute_sha256(tmp_path / "non_existent_file.bin")


def test_raw_auditor_detects_missing_files(tmp_path: Path):
    """Auditor must identify missing files and report clear diagnostic issues."""
    auditor = RawDataAuditor(raw_data_dir=tmp_path)
    result = auditor.audit_primary_embryo_dataset()

    assert not result.audit_passed
    assert "Embryo dataset_2542.xlsx" in result.files_missing
    assert "Data set splitting.zip" in result.files_missing
    assert len(result.issues_detected) > 0


def test_raw_auditor_with_mock_dataset(tmp_path: Path):
    """Auditor correctly inspects an available tabular Excel workbook."""
    mock_excel = tmp_path / "Embryo dataset_2542.xlsx"
    mock_df = pd.DataFrame(
        {
            "Patient_ID": [1, 2, 3],
            "Age": [30.0, 34.0, 29.0],
            "Clinical_Pregnancy_Outcome": [1, 0, 1],
        }
    )
    mock_df.to_excel(mock_excel, index=False)

    auditor = RawDataAuditor(raw_data_dir=tmp_path)
    result = auditor.audit_primary_embryo_dataset()

    assert "Embryo dataset_2542.xlsx" in result.files_present
    assert result.row_count == 3
    assert result.column_count == 3
    assert result.target_column_detected == "Clinical_Pregnancy_Outcome"
    assert result.target_distribution == {"1": 2, "0": 1}
    assert result.duplicate_rows == 0


def test_schema_validation():
    """Test Pydantic schema validation for patient records and DataFrames."""
    valid_record = PatientRecordSchema(
        patient_id="PT-001",
        maternal_age=32.5,
        bmi=23.4,
        clinical_pregnancy=1,
    )
    assert valid_record.maternal_age == 32.5

    # Out of biological range should fail
    with pytest.raises(Exception):
        PatientRecordSchema(maternal_age=110.0)

    # DataFrame schema validation
    valid_df = pd.DataFrame({"clinical_pregnancy": [0, 1, 1, 0]})
    assert validate_dataframe_schema(valid_df, target_col="clinical_pregnancy") == []

    invalid_df = pd.DataFrame({"clinical_pregnancy": ["unknown", "maybe"]})
    issues = validate_dataframe_schema(invalid_df, target_col="clinical_pregnancy")
    assert len(issues) > 0


def test_candidate_hd8yy6gzc5_provenance_record():
    """Verify candidate hd8yy6gzc5 provenance metadata."""
    from fertiscope.data.provenance import CANDIDATE_HD8YY6GZC5_PROVENANCE

    assert CANDIDATE_HD8YY6GZC5_PROVENANCE.doi == "10.17632/hd8yy6gzc5.1"
    assert CANDIDATE_HD8YY6GZC5_PROVENANCE.reported_cohort_size == 5000
    assert CANDIDATE_HD8YY6GZC5_PROVENANCE.license == "CC BY 4.0"
    assert len(CANDIDATE_HD8YY6GZC5_PROVENANCE.expected_files) == 2
    assert "Comprehensive Dataset Profile of In Vitro Fertiliz.zip" in CANDIDATE_HD8YY6GZC5_PROVENANCE.file_checksums_sha256


def test_candidate_hd8yy6gzc5_auditor():
    """Audit the candidate dataset files in data/raw and verify diagnostic detections."""
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    if not (raw_dir / "Dataset Clean.xlsx").exists():
        pytest.skip("Dataset Clean.xlsx not present in data/raw")

    auditor = RawDataAuditor(raw_data_dir=raw_dir)
    result = auditor.audit_candidate_hd8yy6gzc5_dataset()

    assert "Dataset Clean.xlsx" in result.files_present
    assert "Normalized CSV.csv" in result.files_present
    assert result.row_count == 5000
    assert result.column_count == 21
    assert result.target_distribution == {"0": 2508, "1": 2492}
    assert result.formula_count == 37
    assert any("TARGET LEAKAGE" in w for w in result.leakage_warnings)
    assert not result.audit_passed  # Candidate has unresolved methodological issues


def test_candidate_hd8yy6gzc5_arithmetic_and_quality_metrics():
    """Verify cohort arithmetic, sperm units, AMH outliers, and profile duplicates."""
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    if not (raw_dir / "Dataset Clean.xlsx").exists():
        pytest.skip("Dataset Clean.xlsx not present in data/raw")

    auditor = RawDataAuditor(raw_data_dir=raw_dir)
    result = auditor.audit_candidate_hd8yy6gzc5_dataset()

    # Reconstructed transfer cohort: 3,750 rows
    assert result.transfer_cohort_row_count == 3750
    assert result.transfer_cohort_target_distribution == {"1": 2492, "0": 1258}
    assert result.discrepant_hcg_target_count == 8

    # Sperm motility split: 4,296 decimal vs 704 percentage
    assert result.sperm_motility_decimal_count == 4296
    assert result.sperm_motility_percentage_count == 704

    # AMH maximum
    assert result.max_amh_value == pytest.approx(482.10, rel=1e-3)

    # Recurring profiles and cloned cycles (excluding retrieval date)
    assert result.recurring_baseline_profile_count == 1480
    assert result.exact_duplicate_cycle_count == 773



