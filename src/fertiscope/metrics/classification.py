"""Multiclass classification evaluation metrics.

Provides clean-room implementations and wrappers for:
- Standard accuracy and balanced accuracy.
- Macro, micro, and weighted F1 scores.
- Multiclass confusion matrices with per-class precision and recall.

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fertiscope.core.exceptions import SchemaValidationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationSummary:
    """Summary of multiclass model classification performance."""

    criterion: str
    num_samples: int
    num_classes: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict:
        return asdict(self)


class MulticlassClassificationMetrics:
    """Evaluation metrics for multiclass embryo quality grading."""

    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        criterion: str = "EXP",
        num_classes: Optional[int] = None,
    ) -> ClassificationSummary:
        """Compute standard multiclass performance diagnostics."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        if len(y_true) != len(y_pred):
            raise SchemaValidationError(f"Length mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions")

        if len(y_true) == 0:
            raise SchemaValidationError("Cannot evaluate on empty arrays.")

        labels = sorted(list(set(y_true).union(set(y_pred))))
        if num_classes is not None:
            labels = list(range(num_classes))

        k = len(labels)
        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        return ClassificationSummary(
            criterion=criterion.upper(),
            num_samples=len(y_true),
            num_classes=k,
            accuracy=round(acc, 4),
            balanced_accuracy=round(bacc, 4),
            macro_f1=round(macro_f1, 4),
            micro_f1=round(micro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
            per_class_precision={lbl: round(float(precisions[i]), 4) for i, lbl in enumerate(labels)},
            per_class_recall={lbl: round(float(recalls[i]), 4) for i, lbl in enumerate(labels)},
            per_class_f1={lbl: round(float(f1s[i]), 4) for i, lbl in enumerate(labels)},
            confusion_matrix=cm,
        )

