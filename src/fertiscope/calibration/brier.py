r"""Multiclass Brier score and class-wise calibration loss.

Mathematical formulation:
Multiclass Brier Score (Brier 1950, Murphy 1973):
BS = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2
where y_{ik} \in \{0, 1\} is the one-hot encoded true label.

Brier Skill Score:
BSS = 1 - \frac{BS}{BS_{\text{reference}}}

Methodological governance on reference forecast:
- Evaluation labels must NEVER be used to compute the reference baseline forecast (avoiding circularity).
- An independent prevalence vector (e.g. from training data) or an uninformative uniform baseline (1/K)
  must be explicitly supplied or defaulted.
- If BS_reference is 0 (degenerate target with perfect baseline), BSS is explicitly reported as 0.0.

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from fertiscope.core.exceptions import CalibrationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BrierScoreReport:
    """Multiclass and class-wise Brier score decomposition."""

    num_samples: int
    num_classes: int
    overall_brier_score: float
    reference_brier_score: float
    brier_skill_score: float
    classwise_brier_scores: Dict[int, float]
    reference_type: str
    reference_probs: List[float]

    def to_dict(self) -> Dict:
        return asdict(self)


class MulticlassBrier:
    """Independent implementation of multiclass Brier score and skill metrics."""

    @classmethod
    def compute(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        reference_probs: Optional[np.ndarray] = None,
    ) -> BrierScoreReport:
        """Compute multiclass Brier score, class-wise scores, and Brier skill score.

        Parameters
        ----------
        probs : np.ndarray
            Array of shape (N, K) containing predicted class probabilities.
        labels : np.ndarray
            Array of shape (N,) containing integer ground-truth class labels.
        reference_probs : Optional[np.ndarray], default=None
            Independent reference forecast (e.g., training set marginal frequencies).
            Can have shape (K,) or (N, K).
            If None, defaults to the uninformative uniform prior p_k = 1/K.
            Evaluation set labels are NEVER used to estimate this reference.
        """
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if probs.ndim != 2:
            raise CalibrationError(f"Probability array must be 2D, got shape {probs.shape}")

        if len(labels) != len(probs):
            raise CalibrationError(f"Length mismatch: {len(labels)} labels vs {len(probs)} predictions")

        if np.any(np.isnan(probs)) or np.any(np.isnan(labels)):
            raise CalibrationError("NaNs detected in calibration inputs.")

        num_samples, num_classes = probs.shape
        if num_samples == 0:
            raise CalibrationError("Cannot compute Brier score on empty inputs.")

        if num_classes < 2:
            raise CalibrationError("Multiclass Brier score requires at least 2 classes.")

        # One-hot encode ground truth labels
        y_onehot = np.zeros((num_samples, num_classes), dtype=float)
        for i, lbl in enumerate(labels):
            if lbl < 0 or lbl >= num_classes:
                raise CalibrationError(f"Label {lbl} out of bounds for {num_classes} classes.")
            y_onehot[i, lbl] = 1.0

        # Overall multiclass Brier score: mean squared error summed across classes
        squared_errors = np.sum((probs - y_onehot) ** 2, axis=1)
        overall_bs = float(np.mean(squared_errors))

        # Class-wise Brier scores
        classwise: Dict[int, float] = {}
        for k in range(num_classes):
            cw_err = (probs[:, k] - y_onehot[:, k]) ** 2
            classwise[k] = round(float(np.mean(cw_err)), 4)

        # Reference forecast validation & computation (Independent of evaluation labels)
        if reference_probs is not None:
            ref_arr = np.asarray(reference_probs, dtype=float)
            if ref_arr.ndim == 1:
                if len(ref_arr) != num_classes:
                    raise CalibrationError(
                        f"reference_probs dimension mismatch: expected {num_classes}, got {len(ref_arr)}"
                    )
                if np.any(ref_arr < 0.0) or not np.isclose(np.sum(ref_arr), 1.0, atol=1e-3):
                    raise CalibrationError("reference_probs must be non-negative and sum to 1.0")
                ref_mat = np.tile(ref_arr, (num_samples, 1))
                ref_type = "independent_prevalence"
                persisted_ref = [round(float(x), 4) for x in ref_arr]
            elif ref_arr.ndim == 2:
                if ref_arr.shape != (num_samples, num_classes):
                    raise CalibrationError(
                        f"reference_probs 2D shape mismatch: expected ({num_samples}, {num_classes}), got {ref_arr.shape}"
                    )
                ref_mat = ref_arr
                ref_type = "independent_per_sample"
                persisted_ref = [round(float(x), 4) for x in np.mean(ref_arr, axis=0)]
            else:
                raise CalibrationError("reference_probs must be 1D or 2D array.")
        else:
            # Uninformative uniform baseline: 1/K for each class
            uniform_p = np.full((num_samples, num_classes), 1.0 / num_classes, dtype=float)
            ref_mat = uniform_p
            ref_type = "uniform_uninformative"
            persisted_ref = [round(1.0 / num_classes, 4)] * num_classes

        # Calculate reference forecast Brier score
        ref_squared_errors = np.sum((ref_mat - y_onehot) ** 2, axis=1)
        ref_bs = float(np.mean(ref_squared_errors))

        if ref_bs > 1e-12:
            bss = 1.0 - (overall_bs / ref_bs)
        else:
            bss = 0.0  # Zero/degenerate reference score handled explicitly

        return BrierScoreReport(
            num_samples=num_samples,
            num_classes=num_classes,
            overall_brier_score=round(overall_bs, 4),
            reference_brier_score=round(ref_bs, 4),
            brier_skill_score=round(bss, 4),
            classwise_brier_scores=classwise,
            reference_type=ref_type,
            reference_probs=persisted_ref,
        )
