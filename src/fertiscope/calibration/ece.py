r"""Multiclass Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Class-wise ECE.

Mathematical formulation:
Confidence: \hat{p}_i = \max_k p_{ik}
Predicted class: \hat{y}_i = \arg\max_k p_{ik}
Bins B_m partitioning confidence interval [0, 1].
ECE = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|
MCE = \max_{m=1..M} |\text{acc}(B_m) - \text{conf}(B_m)|

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from fertiscope.core.exceptions import CalibrationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BinStatistics:
    """Statistics for an individual confidence bin."""

    bin_idx: int
    bin_lower: float
    bin_upper: float
    count: int
    accuracy: float
    confidence: float
    calibration_gap: float


@dataclass(frozen=True)
class CalibrationReport:
    """Comprehensive multiclass calibration metrics."""

    num_samples: int
    num_classes: int
    num_bins: int
    binning_strategy: str  # "uniform" or "quantile"
    ece: float
    mce: float
    classwise_ece: Dict[int, float]
    mean_confidence: float
    mean_accuracy: float
    bins: List[BinStatistics]

    def to_dict(self) -> Dict:
        return asdict(self)


class MulticlassCalibration:
    """Independent implementation of multiclass calibration metrics."""

    @staticmethod
    def _validate_inputs(
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate shapes and mathematical probability axioms."""
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if probs.ndim != 2:
            raise CalibrationError(f"Probability array must be 2D (samples, classes), got shape {probs.shape}")

        if len(labels) != len(probs):
            raise CalibrationError(f"Length mismatch: {len(labels)} labels vs {len(probs)} probability vectors")

        if np.any(np.isnan(probs)) or np.any(np.isnan(labels)):
            raise CalibrationError("Probabilities or labels contain NaN values. Corrupted inputs cannot be calibrated.")

        if np.any(probs < -1e-5) or np.any(probs > 1.0 + 1e-5):
            raise CalibrationError("Probabilities must lie strictly within [0, 1].")

        # Normalize across classes to prevent floating-point drift
        row_sums = np.sum(probs, axis=1, keepdims=True)
        if np.any(np.abs(row_sums - 1.0) > 1e-2):
            raise CalibrationError("Probability rows do not sum to 1.0.")

        norm_probs = np.clip(probs / row_sums, 0.0, 1.0)
        return norm_probs, labels

    @classmethod
    def compute_ece(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
        strategy: str = "uniform",
    ) -> CalibrationReport:
        """Compute multiclass ECE, MCE, and confidence bin statistics."""
        probs, labels = cls._validate_inputs(probs, labels)
        num_samples, num_classes = probs.shape

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        accuracies = (preds == labels).astype(float)

        if strategy == "uniform":
            bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        elif strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, num_bins + 1)
            bin_edges = np.percentile(confs, quantiles * 100)
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise CalibrationError(f"Unsupported binning strategy: {strategy}")

        bin_stats: List[BinStatistics] = []
        weighted_gap_sum = 0.0
        max_gap = 0.0

        for i in range(num_bins):
            lower = float(bin_edges[i])
            upper = float(bin_edges[i + 1])

            if i == num_bins - 1:
                in_bin = (confs >= lower) & (confs <= upper)
            else:
                in_bin = (confs >= lower) & (confs < upper)

            count = int(np.sum(in_bin))
            if count > 0:
                bin_acc = float(np.mean(accuracies[in_bin]))
                bin_conf = float(np.mean(confs[in_bin]))
                gap = abs(bin_acc - bin_conf)
                weighted_gap_sum += count * gap
                if gap > max_gap:
                    max_gap = gap
            else:
                bin_acc = 0.0
                bin_conf = 0.0
                gap = 0.0

            bin_stats.append(
                BinStatistics(
                    bin_idx=i,
                    bin_lower=round(lower, 4),
                    bin_upper=round(upper, 4),
                    count=count,
                    accuracy=round(bin_acc, 4),
                    confidence=round(bin_conf, 4),
                    calibration_gap=round(gap, 4),
                )
            )

        ece = weighted_gap_sum / num_samples if num_samples > 0 else 0.0

        # Class-wise ECE (One-vs-Rest)
        classwise: Dict[int, float] = {}
        for k in range(num_classes):
            k_probs = probs[:, k]
            k_labels = (labels == k).astype(float)
            k_bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
            k_gap_sum = 0.0
            for i in range(num_bins):
                low = k_bin_edges[i]
                high = k_bin_edges[i + 1]
                in_k = (k_probs >= low) & (k_probs <= high if i == num_bins - 1 else k_probs < high)
                k_count = np.sum(in_k)
                if k_count > 0:
                    k_acc = np.mean(k_labels[in_k])
                    k_conf = np.mean(k_probs[in_k])
                    k_gap_sum += k_count * abs(k_acc - k_conf)
            classwise[k] = round(float(k_gap_sum / num_samples), 4)

        return CalibrationReport(
            num_samples=num_samples,
            num_classes=num_classes,
            num_bins=num_bins,
            binning_strategy=strategy,
            ece=round(float(ece), 4),
            mce=round(float(max_gap), 4),
            classwise_ece=classwise,
            mean_confidence=round(float(np.mean(confs)), 4),
            mean_accuracy=round(float(np.mean(accuracies)), 4),
            bins=bin_stats,
        )
