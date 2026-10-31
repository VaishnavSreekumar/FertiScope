r"""Inter-annotator agreement metrics: Cohen's Kappa, Fleiss' Kappa, and pairwise concordance.

Mathematical formulations:
Cohen's Kappa: \kappa = \frac{p_o - p_e}{1 - p_e}
Fleiss' Kappa: generalized multi-rater agreement for fixed number of ratings per subject.
Pairwise concordance: fraction of agreeing pairs across all valid evaluations.

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from fertiscope.core.exceptions import SchemaValidationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgreementReport:
    """Inter-rater reliability summary."""

    metric_name: str
    num_items: int
    num_raters: int
    num_categories: int
    value: float
    interpretation: str  # E.g. "Slight", "Fair", "Moderate", "Substantial", "Almost Perfect"

    def to_dict(self) -> Dict:
        return asdict(self)


class InterAnnotatorAgreement:
    """Independent implementation of multi-rater agreement metrics."""

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Landis and Koch (1977) benchmark ranges."""
        if kappa <= 0.0:
            return "Poor (Chance or less)"
        elif kappa <= 0.20:
            return "Slight agreement"
        elif kappa <= 0.40:
            return "Fair agreement"
        elif kappa <= 0.60:
            return "Moderate agreement"
        elif kappa <= 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    @classmethod
    def cohens_kappa(
        cls,
        rater_a: np.ndarray,
        rater_b: np.ndarray,
        categories: Optional[List[int]] = None,
    ) -> AgreementReport:
        """Calculate Cohen's Kappa between two raters, excluding paired missing values."""
        rater_a = np.asarray(rater_a)
        rater_b = np.asarray(rater_b)

        valid_a = np.array([not is_missing_value(x) for x in rater_a], dtype=bool)
        valid_b = np.array([not is_missing_value(x) for x in rater_b], dtype=bool)
        valid = valid_a & valid_b

        a_valid = rater_a[valid]
        b_valid = rater_b[valid]

        n = len(a_valid)
        if n == 0:
            raise SchemaValidationError("No overlapping non-null ratings between the two raters.")

        if categories is None:
            cats = np.unique(np.concatenate([a_valid, b_valid]))
        else:
            cats = np.array(categories)

        k = len(cats)
        cat_to_idx = {c: i for i, c in enumerate(cats)}

        # Build confusion matrix
        cm = np.zeros((k, k), dtype=float)
        for i in range(n):
            if a_valid[i] in cat_to_idx and b_valid[i] in cat_to_idx:
                cm[cat_to_idx[a_valid[i]], cat_to_idx[b_valid[i]]] += 1.0

        p_o = np.trace(cm) / n
        row_sums = np.sum(cm, axis=1) / n
        col_sums = np.sum(cm, axis=0) / n
        p_e = float(np.sum(row_sums * col_sums))

        if abs(1.0 - p_e) < 1e-12:
            kappa = 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
        else:
            kappa = (p_o - p_e) / (1.0 - p_e)

        return AgreementReport(
            metric_name="Cohens_Kappa",
            num_items=n,
            num_raters=2,
            num_categories=k,
            value=round(float(kappa), 4),
            interpretation=cls.interpret_kappa(kappa),
        )

    @classmethod
    def fleiss_kappa(
        cls,
        rating_matrix: np.ndarray,
    ) -> AgreementReport:
        """Calculate Fleiss' Kappa for an (N x K) matrix of category counts per subject.

        rating_matrix[i, j] is the number of raters who assigned subject i to category j.
        Requires that the total number of raters per subject is constant (m raters).
        """
        mat = np.asarray(rating_matrix, dtype=float)
        if mat.ndim != 2:
            raise SchemaValidationError("rating_matrix must be 2D (subjects x categories).")

        n_subjects, n_categories = mat.shape
        raters_per_subject = np.sum(mat, axis=1)

        # Check for constant number of raters
        m = raters_per_subject[0]
        if not np.allclose(raters_per_subject, m):
            # If slight variation, filter or raise
            raise SchemaValidationError(
                f"Fleiss' Kappa requires a constant number of raters per subject. Found range {raters_per_subject.min()} to {raters_per_subject.max()}."
            )

        if m <= 1:
            raise SchemaValidationError("At least 2 raters per subject are required.")

        # p_j: proportion of all assignments to category j
        p_j = np.sum(mat, axis=0) / (n_subjects * m)
        p_e = float(np.sum(p_j ** 2))

        # P_i: extent of agreement for subject i
        # P_i = (1 / (m * (m - 1))) * (sum_j n_ij^2 - m)
        p_i = (np.sum(mat ** 2, axis=1) - m) / (m * (m - 1))
        p_o = float(np.mean(p_i))

        if abs(1.0 - p_e) < 1e-12:
            kappa = 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
        else:
            kappa = (p_o - p_e) / (1.0 - p_e)

        return AgreementReport(
            metric_name="Fleiss_Kappa",
            num_items=n_subjects,
            num_raters=int(m),
            num_categories=n_categories,
            value=round(float(kappa), 4),
            interpretation=cls.interpret_kappa(kappa),
        )

    @classmethod
    def pairwise_concordance_rate(
        cls,
        ratings_per_subject: List[List[Union[int, str]]],
    ) -> float:
        """Calculate the proportion of agreeing pairs across all possible rater pairs."""
        total_pairs = 0
        agree_pairs = 0

        for r_list in ratings_per_subject:
            clean = [r for r in r_list if not is_missing_value(r)]
            k = len(clean)
            if k < 2:
                continue
            for i in range(k):
                for j in range(i + 1, k):
                    total_pairs += 1
                    if clean[i] == clean[j]:
                        agree_pairs += 1

        if total_pairs == 0:
            return 0.0
        return round(float(agree_pairs / total_pairs), 4)


def is_missing_value(val: object) -> bool:
    """Helper to detect nulls/NaNs across types without array coercion issues."""
    if val is None:
        return True
    if isinstance(val, (int, np.integer)):
        return False
    if isinstance(val, (float, np.floating)):
        return bool(np.isnan(val))
    val_str = str(val).strip().upper()
    return val_str in ("", "NAN", "NONE", "ND", "-1")

