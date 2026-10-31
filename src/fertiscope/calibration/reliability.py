"""Reliability diagram generation and post-hoc calibration governance.

Strictly enforces methodological constraints:
- Reliability diagrams illustrate empirical accuracy vs predicted confidence per bin.
- Temperature scaling or post-hoc recalibration MUST NEVER be fitted on the evaluation test set.
  Fitting parameters on test data constitutes data snooping / overfitting.
- An independent calibration set (e.g. holdout validation split) is strictly required.

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import minimize

from fertiscope.calibration.ece import CalibrationReport
from fertiscope.core.exceptions import CalibrationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReliabilityDiagramView:
    """Structured view of a reliability diagram for reporting or visualization."""

    bin_centers: List[float]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    calibration_gaps: List[float]
    ece: float
    mce: float

    def render_ascii(self, width: int = 40) -> str:
        """Render a formatted ASCII text diagram representing calibration alignment."""
        lines = [
            "============================================================",
            f" Reliability Diagram (ECE: {self.ece:.4f}, MCE: {self.mce:.4f})",
            " Bin Range   | Conf   | Acc    | Count | Diagram (A=Acc, C=Conf)",
            "------------------------------------------------------------",
        ]
        for i in range(len(self.bin_centers)):
            c = self.bin_confidences[i]
            a = self.bin_accuracies[i]
            cnt = self.bin_counts[i]
            if cnt == 0:
                lines.append(f" Bin {i:02d}     |   --   |   --   |     0 | [Empty]")
                continue

            # Position of Conf and Acc on scale 0 to width
            pos_c = int(round(c * width))
            pos_a = int(round(a * width))

            bar = [" "] * (width + 1)
            # Perfect calibration marker at diagonal
            diag_pos = int(round(self.bin_centers[i] * width))
            if 0 <= diag_pos <= width:
                bar[diag_pos] = "."

            if 0 <= pos_c <= width:
                bar[pos_c] = "C"
            if 0 <= pos_a <= width:
                if bar[pos_a] == "C":
                    bar[pos_a] = "X"  # Exact alignment
                else:
                    bar[pos_a] = "A"

            bar_str = "".join(bar)
            lines.append(f" Bin {i:02d}     | {c:.4f} | {a:.4f} | {cnt:5d} | |{bar_str}|")

        lines.append("============================================================")
        return "\n".join(lines)


class ReliabilityEngine:
    """Independent engine for reliability analysis and calibration safeguards."""

    @staticmethod
    def from_report(report: CalibrationReport) -> ReliabilityDiagramView:
        """Convert a CalibrationReport into a structured diagram view."""
        centers = []
        accs = []
        confs = []
        counts = []
        gaps = []

        for b in report.bins:
            centers.append(round((b.bin_lower + b.bin_upper) / 2.0, 4))
            accs.append(b.accuracy)
            confs.append(b.confidence)
            counts.append(b.count)
            gaps.append(b.calibration_gap)

        return ReliabilityDiagramView(
            bin_centers=centers,
            bin_accuracies=accs,
            bin_confidences=confs,
            bin_counts=counts,
            calibration_gaps=gaps,
            ece=report.ece,
            mce=report.mce,
        )

    @staticmethod
    def fit_temperature_scaling(
        logits: np.ndarray,
        labels: np.ndarray,
        split_name: str = "validation",
    ) -> float:
        """Fit a single positive temperature scalar T to optimize NLL on an independent split.

        STRICT PROHIBITION:
        Will raise CalibrationError if split_name indicates an evaluation test set.
        """
        split_norm = split_name.strip().lower()
        if "test" in split_norm or "eval" in split_norm:
            raise CalibrationError(
                f"Methodological violation: Temperature scaling cannot be fitted on '{split_name}'. "
                "Fitting calibration parameters on the evaluation test set leads to data leakage and overconfidence. "
                "Provide an independent validation/calibration partition."
            )

        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if len(logits) != len(labels):
            raise CalibrationError("Dimension mismatch between logits and labels.")

        def nll_obj(temp: np.ndarray) -> float:
            t = temp[0]
            if t <= 0.001:
                return 1e9
            scaled_logits = logits / t
            # Stable log-softmax
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            exp_l = np.exp(scaled_logits - max_l)
            sum_exp = np.sum(exp_l, axis=1, keepdims=True)
            log_probs = scaled_logits - max_l - np.log(sum_exp)
            # Pick true class log prob
            nll = -np.mean(log_probs[np.arange(len(labels)), labels])
            return float(nll)

        res = minimize(nll_obj, x0=[1.0], bounds=[(0.01, 20.0)], method="L-BFGS-B")
        optimal_t = float(res.x[0])
        return round(optimal_t, 4)

