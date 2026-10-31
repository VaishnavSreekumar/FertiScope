"""Analysis of sister-embryo cluster leakage on model performance and calibration.

Evaluates whether models perform artificially better on test images whose sister
embryos appeared in training/validation cohorts (cycle-overlapping, n=242) compared to
cycle-isolated test cohorts (n=58).

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from fertiscope.core.exceptions import SchemaValidationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SubcohortEvaluationMetrics:
    """Evaluation metrics on a specific subcohort partition."""

    cohort_name: str
    sample_size: int
    num_cycles: int
    accuracy: float
    macro_f1: float
    log_loss_val: float
    mean_confidence: float
    mean_predictive_entropy: float


@dataclass(frozen=True)
class CycleLeakageComparisonReport:
    """Comparative evaluation between leaked and isolated cohorts."""

    model_name: str
    criterion: str
    overlapping_cohort: SubcohortEvaluationMetrics
    isolated_cohort: SubcohortEvaluationMetrics
    accuracy_delta: float  # overlapping - isolated
    log_loss_delta: float  # overlapping - isolated (negative means lower loss on overlapping)
    is_performance_inflated: bool
    methodological_note: str

    def to_dict(self) -> Dict:
        return asdict(self)


class CycleLeakageEvaluator:
    """Compares model evaluation between cycle-isolated and cycle-overlapping test samples."""

    @staticmethod
    def evaluate_cohort(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        cohort_name: str,
        num_cycles: int,
    ) -> SubcohortEvaluationMetrics:
        """Compute standard classification and confidence metrics on a cohort."""
        if len(y_true) == 0:
            return SubcohortEvaluationMetrics(
                cohort_name=cohort_name,
                sample_size=0,
                num_cycles=num_cycles,
                accuracy=0.0,
                macro_f1=0.0,
                log_loss_val=0.0,
                mean_confidence=0.0,
                mean_predictive_entropy=0.0,
            )

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Compute log loss with eps clipping
        eps = 1e-12
        clipped_probs = np.clip(y_prob, eps, 1.0)
        norm_probs = clipped_probs / np.sum(clipped_probs, axis=1, keepdims=True)

        # Unique classes
        n_classes = norm_probs.shape[1]
        try:
            ll = float(log_loss(y_true, norm_probs, labels=list(range(n_classes))))
        except Exception:
            ll = float("nan")

        conf = float(np.mean(np.max(norm_probs, axis=1)))
        # Shannon entropy base 2
        entropy = -np.sum(norm_probs * np.log2(norm_probs), axis=1)
        mean_ent = float(np.mean(entropy))

        return SubcohortEvaluationMetrics(
            cohort_name=cohort_name,
            sample_size=len(y_true),
            num_cycles=num_cycles,
            accuracy=round(acc, 4),
            macro_f1=round(macro_f1, 4),
            log_loss_val=round(ll, 4),
            mean_confidence=round(conf, 4),
            mean_predictive_entropy=round(mean_ent, 4),
        )

    @classmethod
    def compare_cohorts(
        cls,
        test_df: pd.DataFrame,
        model_name: str,
        criterion: str,
        true_col: str,
        pred_col: str,
        prob_cols: List[str],
        cycle_col: str = "cycle_id",
        leaked_cycles: Optional[set] = None,
    ) -> CycleLeakageComparisonReport:
        """Partition test data into leaked vs isolated cohorts and compare performance."""
        if cycle_col not in test_df.columns:
            raise SchemaValidationError(f"Missing '{cycle_col}' in test dataframe.")

        if leaked_cycles is None:
            raise SchemaValidationError("leaked_cycles set must be provided to determine overlap.")

        is_leaked = test_df[cycle_col].isin(leaked_cycles)
        df_leaked = test_df[is_leaked]
        df_isolated = test_df[~is_leaked]

        y_true_leaked = df_leaked[true_col].values.astype(int)
        y_pred_leaked = df_leaked[pred_col].values.astype(int)
        y_prob_leaked = df_leaked[prob_cols].values.astype(float)

        y_true_iso = df_isolated[true_col].values.astype(int)
        y_pred_iso = df_isolated[pred_col].values.astype(int)
        y_prob_iso = df_isolated[prob_cols].values.astype(float)

        metrics_leaked = cls.evaluate_cohort(
            y_true_leaked,
            y_pred_leaked,
            y_prob_leaked,
            cohort_name="overlapping_cycle_cohort",
            num_cycles=df_leaked[cycle_col].nunique(),
        )

        metrics_iso = cls.evaluate_cohort(
            y_true_iso,
            y_pred_iso,
            y_prob_iso,
            cohort_name="isolated_cycle_cohort",
            num_cycles=df_isolated[cycle_col].nunique(),
        )

        acc_delta = metrics_leaked.accuracy - metrics_iso.accuracy
        ll_delta = metrics_leaked.log_loss_val - metrics_iso.log_loss_val

        note = (
            "Cycle identifier reflects an oocyte retrieval cycle, not verified unique patient identity. "
            "Evaluations on overlapping cohorts risk optimistic bias due to shared morphology among sister embryos."
        )

        return CycleLeakageComparisonReport(
            model_name=model_name,
            criterion=criterion.upper(),
            overlapping_cohort=metrics_leaked,
            isolated_cohort=metrics_iso,
            accuracy_delta=round(acc_delta, 4),
            log_loss_delta=round(ll_delta, 4),
            is_performance_inflated=acc_delta > 0,
            methodological_note=note,
        )

