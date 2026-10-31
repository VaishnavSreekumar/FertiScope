"""Schema definitions and validation for IVF-ET tabular clinical variables."""

from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fertiscope.core.exceptions import SchemaValidationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


class PatientRecordSchema(BaseModel):
    """Base schema for an individual IVF-ET clinical record.
    
    Fields are designed to accommodate the standard 36 clinical and embryological
    indicators reported in IVF-ET literature (e.g. maternal age, BMI, AMH, antral
    follicle count, oocytes retrieved, blastocyst morphological grading, semen
    parameters, and clinical pregnancy outcome).
    """

    model_config = ConfigDict(extra="allow")

    patient_id: Optional[str] = Field(default=None, description="De-identified patient/cycle identifier")
    maternal_age: Optional[float] = Field(default=None, ge=18.0, le=55.0, description="Maternal age in years")
    bmi: Optional[float] = Field(default=None, ge=10.0, le=60.0, description="Body Mass Index (kg/m^2)")
    infertility_duration_years: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    baseline_fsh: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Baseline FSH (mIU/mL)")
    baseline_lh: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Baseline LH (mIU/mL)")
    baseline_e2: Optional[float] = Field(default=None, ge=0.0, le=2000.0, description="Baseline Estradiol (pg/mL)")
    amh: Optional[float] = Field(default=None, ge=0.0, le=50.0, description="Anti-Müllerian Hormone (ng/mL)")
    antral_follicle_count: Optional[float] = Field(default=None, ge=0.0, le=80.0)
    oocytes_retrieved: Optional[float] = Field(default=None, ge=0.0, le=80.0)
    fertilized_oocytes: Optional[float] = Field(default=None, ge=0.0, le=80.0)
    blastocysts_available: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    endometrial_thickness_mm: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    clinical_pregnancy: Optional[int] = Field(default=None, ge=0, le=1, description="Binary clinical pregnancy outcome (0=No, 1=Yes)")


def validate_dataframe_schema(df: pd.DataFrame, target_col: Optional[str] = None) -> List[str]:
    """Validate DataFrame against structural requirements and return list of validation warnings/errors."""
    issues: List[str] = []

    if df.empty:
        issues.append("DataFrame is empty (0 rows).")
        return issues

    if target_col:
        if target_col not in df.columns:
            issues.append(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")
        else:
            unique_targets = set(df[target_col].dropna().unique())
            valid_binary_sets = [{0, 1}, {0.0, 1.0}, {"0", "1"}, {"negative", "positive"}, {"no", "yes"}, {"false", "true"}]
            if not any(unique_targets.issubset(s) for s in valid_binary_sets):
                issues.append(f"Target column '{target_col}' has non-standard values: {unique_targets}")

    return issues
