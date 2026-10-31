"""Clean-room model prediction and probability distribution parser.

Parses outputs from vision benchmark architectures (CoatNet, ConvNeXt, Swin, etc.).
Strictly validates probability axioms and explicitly flags file corruption.
No original code is copied or reused.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fertiscope.core.exceptions import CorruptedFileError, DatasetNotFoundError, SchemaValidationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


class ModelPredictionSummary(BaseModel):
    """Metadata and integrity summary for an architecture's predictions."""

    model_name: str
    criterion: str
    num_samples: int
    num_classes: int
    has_valid_probabilities: bool
    is_corrupted: bool
    corruption_reason: Optional[str] = None
    mean_confidence: float = Field(ge=0.0, le=1.0)
    mean_predictive_entropy: float = Field(ge=0.0)
    has_swag_uncertainty: bool = False
    mean_swag_uncertainty: Optional[float] = None


class ModelPredictionParser:
    """Independent parser for benchmark model probability outputs and uncertainty."""

    def __init__(self, models_root: Path | str, strict: bool = False):
        self.models_root = Path(models_root)
        self.strict = strict

    def get_available_models(self) -> List[str]:
        """List all model directory names present under the models root."""
        if not self.models_root.exists():
            raise DatasetNotFoundError(f"Models directory not found: {self.models_root}")
        return sorted(
            [d.name for d in self.models_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )

    def load_criterion_probabilities(
        self,
        model_name: str,
        criterion: str,
    ) -> Tuple[pd.DataFrame, ModelPredictionSummary]:
        """Load probability outputs for a specific model and criterion ('EXP', 'ICM', 'TEQ').

        Validates discrete probability axioms: p_i >= 0 and sum(p_i) == 1.0.
        Explicitly flags and handles file corruption without inventing values.
        """
        crit = criterion.lower()
        if crit in ("te", "teq"):
            target_files = ["teq.csv", "te.csv"]
        else:
            target_files = [f"{crit}.csv"]

        model_dir = self.models_root / model_name
        if not model_dir.exists():
            raise DatasetNotFoundError(f"Model directory does not exist: {model_dir}")

        csv_file = None
        for tf in target_files:
            candidate = model_dir / tf
            if candidate.exists():
                csv_file = candidate
                break

        if csv_file is None:
            raise DatasetNotFoundError(
                f"Prediction file for criterion '{criterion}' not found in {model_dir}"
            )

        # Read first line to detect whether header exists
        with open(csv_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        is_headerless = ".png" in first_line.lower() or ".jpg" in first_line.lower()

        if is_headerless:
            df = pd.read_csv(csv_file, header=None)
            # Drop trailing all-NaN columns from trailing delimiters
            df.dropna(axis=1, how="all", inplace=True)
            df.rename(columns={0: "image_name"}, inplace=True)

            num_cols = df.shape[1] - 1  # Excluding image_name
            # Kromp et al. alternating pattern: prob_0, std_0, prob_1, std_1, ...
            if num_cols % 2 == 0 and num_cols >= 2:
                num_classes = num_cols // 2
                prob_col_indices = [1 + 2 * k for k in range(num_classes)]
                std_col_indices = [2 + 2 * k for k in range(num_classes)]

                new_prob_cols = [f"prob_{k}" for k in range(num_classes)]
                new_std_cols = [f"std_{k}" for k in range(num_classes)]

                rename_dict = {}
                for idx, name in zip(prob_col_indices, new_prob_cols):
                    rename_dict[idx] = name
                for idx, name in zip(std_col_indices, new_std_cols):
                    rename_dict[idx] = name
                df.rename(columns=rename_dict, inplace=True)

                prob_cols = new_prob_cols
                has_swag = True
                uncertainty_cols = new_std_cols
            else:
                prob_cols = [c for c in df.columns if c != "image_name"]
                has_swag = False
                uncertainty_cols = []
        else:
            df = pd.read_csv(csv_file)
            # Standardize image identifier column
            col_candidates = [
                c for c in df.columns if c.lower() in ("image", "image_name", "img", "filename", "unnamed: 0")
            ]
            if col_candidates:
                df.rename(columns={col_candidates[0]: "image_name"}, inplace=True)
            elif df.iloc[:, 0].astype(str).str.contains(r"\.png", regex=True).any():
                df.rename(columns={df.columns[0]: "image_name"}, inplace=True)
            else:
                raise SchemaValidationError(f"Could not identify image identifier column in {csv_file}")

            prob_cols = []
            uncertainty_cols = []
            for col in df.columns:
                if col == "image_name":
                    continue
                col_lower = str(col).lower()
                if "std" in col_lower or "unc" in col_lower or "var" in col_lower:
                    uncertainty_cols.append(col)
                else:
                    prob_cols.append(col)
            has_swag = len(uncertainty_cols) > 0

        # Check for NaN corruption
        num_samples = len(df)
        prob_sub = df[prob_cols]

        is_all_nan = prob_sub.isnull().all().all()
        has_any_nan = prob_sub.isnull().any().any()

        if is_all_nan:
            msg = f"Prediction file {csv_file.name} for model '{model_name}' is 100% corrupted with NaNs."
            logger.warning(msg)
            if self.strict:
                raise CorruptedFileError(msg)
            summary = ModelPredictionSummary(
                model_name=model_name,
                criterion=criterion.upper(),
                num_samples=num_samples,
                num_classes=len(prob_cols),
                has_valid_probabilities=False,
                is_corrupted=True,
                corruption_reason="All probability values are NaN",
                mean_confidence=0.0,
                mean_predictive_entropy=0.0,
                has_swag_uncertainty=False,
                mean_swag_uncertainty=None,
            )
            return df, summary

        if has_any_nan:
            msg = f"Prediction file {csv_file.name} for model '{model_name}' contains unexpected NaNs."
            logger.warning(msg)
            if self.strict:
                raise CorruptedFileError(msg)

        # Validate probability constraints
        prob_matrix = prob_sub.values.astype(float)
        row_sums = np.sum(prob_matrix, axis=1)

        # Check if rows sum approximately to 1.0
        sums_valid = bool(np.allclose(row_sums, 1.0, atol=1e-3))
        values_non_negative = bool((prob_matrix >= -1e-5).all())

        if not (sums_valid and values_non_negative):
            logger.warning(
                f"Model {model_name} criterion {criterion} violates discrete probability axioms (min sum: {row_sums.min():.4f}, max sum: {row_sums.max():.4f})"
            )

        # Compute predictive entropy: H(p) = -sum(p * log2(p))
        clipped_probs = np.clip(prob_matrix, 1e-12, 1.0)
        norm_probs = clipped_probs / np.sum(clipped_probs, axis=1, keepdims=True)
        predictive_entropy = -np.sum(norm_probs * np.log2(norm_probs), axis=1)
        max_confidences = np.max(norm_probs, axis=1)

        df["predictive_entropy"] = predictive_entropy
        df["confidence"] = max_confidences

        mean_swag = float(df[uncertainty_cols].values.mean()) if has_swag else None

        summary = ModelPredictionSummary(
            model_name=model_name,
            criterion=criterion.upper(),
            num_samples=num_samples,
            num_classes=len(prob_cols),
            has_valid_probabilities=sums_valid and values_non_negative,
            is_corrupted=False,
            mean_confidence=round(float(np.mean(max_confidences)), 4),
            mean_predictive_entropy=round(float(np.mean(predictive_entropy)), 4),
            has_swag_uncertainty=has_swag,
            mean_swag_uncertainty=round(mean_swag, 4) if mean_swag is not None else None,
        )

        return df, summary

