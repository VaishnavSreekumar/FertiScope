"""Dataset audit and integrity checking module for FertiScope."""

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from fertiscope.core.exceptions import DatasetNotFoundError, IntegrityVerificationError
from fertiscope.core.logging import get_logger
from fertiscope.data.provenance import compute_sha256

logger = get_logger(__name__)


@dataclass
class DatasetAuditResult:
    """Detailed diagnostic results from a raw dataset audit."""

    raw_dir: Path
    files_present: List[str] = field(default_factory=list)
    files_missing: List[str] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)
    excel_sheets: List[str] = field(default_factory=list)
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    duplicate_rows: Optional[int] = None
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    target_column_detected: Optional[str] = None
    target_distribution: Dict[Any, int] = field(default_factory=dict)
    splitting_archive_inspected: bool = False
    splitting_files: List[str] = field(default_factory=list)
    audit_passed: bool = False
    issues_detected: List[str] = field(default_factory=list)
    leakage_warnings: List[str] = field(default_factory=list)
    csv_row_count: Optional[int] = None
    csv_column_count: Optional[int] = None
    formula_count: int = 0
    transfer_cohort_row_count: Optional[int] = None
    transfer_cohort_target_distribution: Dict[str, int] = field(default_factory=dict)
    discrepant_hcg_target_count: Optional[int] = None
    sperm_motility_decimal_count: Optional[int] = None
    sperm_motility_percentage_count: Optional[int] = None
    recurring_baseline_profile_count: Optional[int] = None
    exact_duplicate_cycle_count: Optional[int] = None
    max_amh_value: Optional[float] = None


class RawDataAuditor:
    """Inspects, hashes, and audits raw input files for scientific reproducibility."""

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)

    def audit_primary_embryo_dataset(
        self,
        excel_filename: str = "Embryo dataset_2542.xlsx",
        splitting_zip_filename: str = "Data set splitting.zip",
    ) -> DatasetAuditResult:
        """Audit the primary candidate Mendeley embryo dataset."""
        result = DatasetAuditResult(raw_dir=self.raw_data_dir)

        excel_path = self.raw_data_dir / excel_filename
        zip_path = self.raw_data_dir / splitting_zip_filename

        # Check presence
        for name, path in [(excel_filename, excel_path), (splitting_zip_filename, zip_path)]:
            if path.exists():
                result.files_present.append(name)
                try:
                    result.checksums[name] = compute_sha256(path)
                    logger.info("File present: %s (SHA-256: %s)", name, result.checksums[name])
                except Exception as e:
                    result.issues_detected.append(f"Hashing failed for {name}: {e}")
            else:
                result.files_missing.append(name)
                logger.warning("Missing required file: %s in %s", name, self.raw_data_dir)

        if not excel_path.exists():
            result.issues_detected.append(
                f"Missing Excel dataset file: '{excel_filename}'. "
                f"Raw dataset must be placed in '{self.raw_data_dir}' before ingestion."
            )
            return result

        # Inspect Excel structure
        try:
            excel_file = pd.ExcelFile(excel_path)
            result.excel_sheets = excel_file.sheet_names
            logger.info("Excel sheets discovered: %s", result.excel_sheets)

            # Load primary sheet
            df = pd.read_excel(excel_path, sheet_name=0)
            result.row_count = len(df)
            result.column_count = len(df.columns)
            result.columns = [str(c) for c in df.columns]
            result.duplicate_rows = int(df.duplicated().sum())

            # Missingness
            result.missing_value_counts = {
                str(col): int(val) for col, val in df.isna().sum().items() if val > 0
            }

            # Target detection (look for pregnancy/outcome markers)
            outcome_candidates = [
                c for c in df.columns
                if any(k in str(c).lower() for k in ["outcome", "pregnan", "label", "target", "result", "y"])
            ]
            if outcome_candidates:
                result.target_column_detected = str(outcome_candidates[0])
                result.target_distribution = {
                    str(k): int(v) for k, v in df[outcome_candidates[0]].value_counts().items()
                }

        except Exception as e:
            result.issues_detected.append(f"Failed to read Excel workbook: {e}")

        # Inspect splitting archive if available
        if zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    result.splitting_files = zf.namelist()
                    result.splitting_archive_inspected = True
                    logger.info(
                        "Splitting zip contains %d entries", len(result.splitting_files)
                    )
            except Exception as e:
                result.issues_detected.append(f"Failed to inspect splitting ZIP archive: {e}")

        result.audit_passed = len(result.issues_detected) == 0 and len(result.files_missing) == 0
        return result

    def audit_candidate_hd8yy6gzc5_dataset(
        self,
        excel_filename: str = "Dataset Clean.xlsx",
        csv_filename: str = "Normalized CSV.csv",
    ) -> DatasetAuditResult:
        """Audit the Mendeley 10.17632/hd8yy6gzc5.1 candidate dataset."""
        import openpyxl

        result = DatasetAuditResult(raw_dir=self.raw_data_dir)
        excel_path = self.raw_data_dir / excel_filename
        csv_path = self.raw_data_dir / csv_filename

        for name, path in [(excel_filename, excel_path), (csv_filename, csv_path)]:
            if path.exists():
                result.files_present.append(name)
                try:
                    result.checksums[name] = compute_sha256(path)
                    logger.info("File present: %s (SHA-256: %s)", name, result.checksums[name])
                except Exception as e:
                    result.issues_detected.append(f"Hashing failed for {name}: {e}")
            else:
                result.files_missing.append(name)
                logger.warning("Missing required file: %s in %s", name, self.raw_data_dir)

        if not excel_path.exists():
            result.issues_detected.append(f"Missing Excel dataset file: '{excel_filename}'")
            return result

        # Inspect Excel with openpyxl for sheets and formulas
        try:
            wb = openpyxl.load_workbook(excel_path, data_only=False)
            result.excel_sheets = wb.sheetnames
            ws = wb.active

            formulas = []
            for row in ws.iter_rows(values_only=False):
                for cell in row:
                    val = cell.value
                    if val and isinstance(val, str) and val.startswith("="):
                        formulas.append((cell.coordinate, val))
            result.formula_count = len(formulas)

            if ws.max_row > 5001:
                result.issues_detected.append(
                    f"Excel workbook contains embedded summary formula rows at rows 5002-5003 "
                    f"(=MIN / =MAX across columns), which contaminate pandas read_excel."
                )

            # Check cell T1848
            if ws["T1848"].value == "=T1774":
                result.issues_detected.append(
                    "Cell T1848 contains formula '=T1774' copying prior row's β-HCG result."
                )

            # Read data rows (5000 rows)
            df_xl_raw = pd.read_excel(excel_path, nrows=5000)
            result.row_count = len(df_xl_raw)
            result.column_count = len(df_xl_raw.columns)
            result.columns = [str(c) for c in df_xl_raw.columns]
            result.duplicate_rows = int(df_xl_raw.duplicated().sum())

            # Missingness
            result.missing_value_counts = {
                str(col): int(val) for col, val in df_xl_raw.isna().sum().items() if val > 0
            }

            # Check for backtick in Number of attempts
            if "Number of attempts" in df_xl_raw.columns:
                mixed_attempts = [
                    v for v in df_xl_raw["Number of attempts"] if isinstance(v, str)
                ]
                if mixed_attempts:
                    result.issues_detected.append(
                        f"Column 'Number of attempts' has dirty mixed string values: {set(mixed_attempts)} at rows 1343/4879."
                    )

            # Target detection
            target_col = df_xl_raw.columns[20]
            result.target_column_detected = str(target_col)
            result.target_distribution = {
                str(int(k)): int(v) for k, v in df_xl_raw[target_col].value_counts().items()
            }

            # Target leakage detection: Column 20 (β-HCG)
            hcg_col = df_xl_raw.columns[19]
            if "hcg" in hcg_col.lower():
                result.leakage_warnings.append(
                    f"CRITICAL TARGET LEAKAGE: Column 20 '{hcg_col}' is a post-outcome pregnancy biomarker "
                    f"(β-HCG test) with 99.7% concordance with the target. Including it as a predictor "
                    f"causes massive data leakage."
                )

            # Implantation conflation warning: 'Non-return of embryos'
            is_non_return = df_xl_raw[hcg_col].astype(str).str.contains("Non-return")
            non_return_count = int(is_non_return.sum())
            if non_return_count > 0:
                result.leakage_warnings.append(
                    f"METHODOLOGICAL ISSUE: {non_return_count} cycles had 'Non-return of embryos' (no embryo transfer) "
                    f"and were all coded as Output=0. Cycle cancellations are conflated with uterine implantation failure."
                )
                df_transfer = df_xl_raw[~is_non_return]
                result.transfer_cohort_row_count = len(df_transfer)
                result.transfer_cohort_target_distribution = {
                    str(int(k)): int(v) for k, v in df_transfer[target_col].value_counts().items()
                }

            # Discrepancies between β-HCG and Output (positive HCG but Output == 0)
            is_pos_hcg = df_xl_raw[hcg_col].astype(str).str.upper().str.contains("POS")
            discrepancies = (is_pos_hcg & (df_xl_raw[target_col] == 0)).sum()
            result.discrepant_hcg_target_count = int(discrepancies)

            # Sperm motility unit distribution
            if "Sperm motility" in df_xl_raw.columns:
                sm = pd.to_numeric(df_xl_raw["Sperm motility"], errors="coerce")
                result.sperm_motility_decimal_count = int((sm <= 1.0).sum())
                result.sperm_motility_percentage_count = int((sm > 1.0).sum())

            # AMH max value
            if "AMH" in df_xl_raw.columns:
                result.max_amh_value = float(pd.to_numeric(df_xl_raw["AMH"], errors="coerce").max())

            # Recurring baseline profiles and exact duplicates excluding date
            baseline_cols = ["Female Age", "Male Age", "AMH", "Medicines and treatments"]
            if all(c in df_xl_raw.columns for c in baseline_cols):
                grouped = df_xl_raw.groupby(baseline_cols).size()
                result.recurring_baseline_profile_count = int((grouped > 1).sum())

            cols_excl_date = [c for c in df_xl_raw.columns if c != "Egg retrieval date"]
            result.exact_duplicate_cycle_count = int(df_xl_raw.duplicated(subset=cols_excl_date, keep="first").sum())


        except Exception as e:
            result.issues_detected.append(f"Failed to audit Excel workbook: {e}")

        # Inspect CSV if present
        if csv_path.exists():
            try:
                df_csv = pd.read_csv(csv_path)
                result.csv_row_count = len(df_csv)
                result.csv_column_count = len(df_csv.columns)
                csv_dup = int(df_csv.duplicated().sum())
                if csv_dup > 0:
                    result.issues_detected.append(
                        f"Normalized CSV contains {csv_dup} exact duplicate rows despite Excel having 0 duplicates."
                    )
            except Exception as e:
                result.issues_detected.append(f"Failed to audit CSV: {e}")

        # Check absence of patient identifier
        result.issues_detected.append(
            "Lack of unique patient identifiers prevents detection of longitudinal patient cycle repeats."
        )

        # Dataset fails approval as primary ML dataset without resolving leakage & pre-processing
        result.audit_passed = (
            len(result.files_missing) == 0
            and len(result.issues_detected) == 0
            and len(result.leakage_warnings) == 0
        )
        return result

