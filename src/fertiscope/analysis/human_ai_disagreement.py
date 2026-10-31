"""Human-AI disagreement and predictive uncertainty correlation analysis.

Analyzes the statistical relationship between human expert disagreement (Shannon voting entropy)
and algorithmic uncertainty (model predictive entropy and SWA-G posterior variance).

Methodological safeguards:
- Human voting disagreement reflects subjective inter-observer variability, NOT ground truth biology.
- Correlation between model entropy and human entropy demonstrates heuristic alignment, NOT cognitive
  understanding or clinical safety.

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from fertiscope.core.exceptions import SchemaValidationError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation metric with confidence intervals and statistical significance."""

    metric: str
    coefficient: float
    p_value: float
    ci_95_lower: float
    ci_95_upper: float
    sample_size: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class CriterionDisagreementReport:
    """Disagreement analysis for a specific Gardner criterion."""

    criterion: str
    num_images: int
    mean_human_entropy: float
    mean_model_entropy: float
    spearman_correlation: CorrelationResult
    pearson_correlation: CorrelationResult
    unanimous_agreement_rate: float
    tied_agreement_rate: float
    high_disagreement_count: int  # Entropy > 1.0 bit

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class ComprehensiveDisagreementReport:
    """Multi-criterion human-AI disagreement and sensitivity analysis."""

    model_name: str
    criteria_reports: Dict[str, CriterionDisagreementReport]
    overall_spearman: CorrelationResult
    overall_pearson: CorrelationResult
    methodological_disclaimer: str

    def to_dict(self) -> Dict:
        return asdict(self)


class HumanAIDisagreementAnalyzer:
    """Independent analyzer relating human inter-rater disagreement to model predictive entropy."""

    @staticmethod
    def compute_fisher_ci(r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute asymptotic confidence intervals strictly for Pearson linear correlation via Fisher z-transform."""
        if n <= 3:
            return -1.0, 1.0

        # Bound r strictly within (-1, 1)
        r_clamped = np.clip(r, -0.99999, 0.99999)
        z = np.arctanh(r_clamped)
        se = 1.0 / np.sqrt(n - 3)
        alpha = 1.0 - confidence
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

        z_low = z - z_crit * se
        z_high = z + z_crit * se

        ci_low = float(np.tanh(z_low))
        ci_high = float(np.tanh(z_high))
        return round(ci_low, 4), round(ci_high, 4)

    @staticmethod
    def compute_spearman_bootstrap_ci(
        x: np.ndarray,
        y: np.ndarray,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        random_seed: int = 42,
    ) -> Tuple[float, float]:
        """Compute empirical percentile bootstrap confidence interval strictly for Spearman rank correlation.

        Methodology:
        - Pairs (x_i, y_i) are resampled with replacement using a deterministic numpy Generator.
        - Fractional ranks resolve ties as implemented in standard Spearman rank statistics.
        - Degenerate resamples (where resampled x or y is constant) are explicitly caught and discarded.
        - Percentile bounds at (alpha/2, 1 - alpha/2) are extracted.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n < 5:
            return -1.0, 1.0

        rng = np.random.default_rng(random_seed)
        boot_rhos: List[float] = []
        indices = np.arange(n)

        for _ in range(n_resamples):
            boot_idx = rng.choice(indices, size=n, replace=True)
            bx, by = x[boot_idx], y[boot_idx]
            if np.all(bx == bx[0]) or np.all(by == by[0]):
                continue
            res = stats.spearmanr(bx, by)
            stat = res.statistic
            if not np.isnan(stat):
                boot_rhos.append(float(stat))

        if len(boot_rhos) < n_resamples // 2:
            logger.warning("Excessive degenerate bootstrap resamples; falling back to wide bounds.")
            return -1.0, 1.0

        alpha = 1.0 - confidence
        low_pct = (alpha / 2.0) * 100.0
        high_pct = (1.0 - alpha / 2.0) * 100.0

        ci_low = float(np.percentile(boot_rhos, low_pct))
        ci_high = float(np.percentile(boot_rhos, high_pct))

        return round(float(np.clip(ci_low, -1.0, 1.0)), 4), round(float(np.clip(ci_high, -1.0, 1.0)), 4)

    @classmethod
    def analyze_criterion(
        cls,
        human_entropies: np.ndarray,
        model_entropies: np.ndarray,
        criterion: str,
        unanimous_flags: Optional[np.ndarray] = None,
        tie_flags: Optional[np.ndarray] = None,
        random_seed: int = 42,
    ) -> CriterionDisagreementReport:
        """Run statistical correlation and distribution checks for a single criterion."""
        h_ent = np.asarray(human_entropies, dtype=float)
        m_ent = np.asarray(model_entropies, dtype=float)

        if len(h_ent) != len(m_ent):
            raise SchemaValidationError(f"Length mismatch: {len(h_ent)} human vs {len(m_ent)} model entropies")

        valid = ~(np.isnan(h_ent) | np.isnan(m_ent))
        h_clean = h_ent[valid]
        m_clean = m_ent[valid]
        n = len(h_clean)

        if n < 5:
            raise SchemaValidationError(f"Insufficient valid data points (n={n}) to compute correlation.")

        # Spearman rank correlation with non-parametric bootstrap CI
        spearman_res = stats.spearmanr(h_clean, m_clean)
        sp_coef = float(spearman_res.statistic)
        sp_p = float(spearman_res.pvalue)
        sp_ci_low, sp_ci_high = cls.compute_spearman_bootstrap_ci(
            h_clean, m_clean, n_resamples=2000, random_seed=random_seed
        )

        sp_report = CorrelationResult(
            metric="Spearman_rho (Bootstrap)",
            coefficient=round(sp_coef, 4),
            p_value=float(f"{sp_p:.4e}"),
            ci_95_lower=sp_ci_low,
            ci_95_upper=sp_ci_high,
            sample_size=n,
        )

        # Pearson correlation with Fisher z CI (valid for linear correlation)
        pearson_res = stats.pearsonr(h_clean, m_clean)
        pe_coef = float(pearson_res.statistic)
        pe_p = float(pearson_res.pvalue)
        pe_ci_low, pe_ci_high = cls.compute_fisher_ci(pe_coef, n)

        pe_report = CorrelationResult(
            metric="Pearson_r (Fisher_z)",
            coefficient=round(pe_coef, 4),
            p_value=float(f"{pe_p:.4e}"),
            ci_95_lower=pe_ci_low,
            ci_95_upper=pe_ci_high,
            sample_size=n,
        )

        unanimous_rate = float(np.mean(unanimous_flags)) if unanimous_flags is not None else 0.0
        tie_rate = float(np.mean(tie_flags)) if tie_flags is not None else 0.0
        high_disagree = int(np.sum(h_clean > 1.0))

        return CriterionDisagreementReport(
            criterion=criterion.upper(),
            num_images=n,
            mean_human_entropy=round(float(np.mean(h_clean)), 4),
            mean_model_entropy=round(float(np.mean(m_clean)), 4),
            spearman_correlation=sp_report,
            pearson_correlation=pe_report,
            unanimous_agreement_rate=round(unanimous_rate, 4),
            tied_agreement_rate=round(tie_rate, 4),
            high_disagreement_count=high_disagree,
        )

    @classmethod
    def evaluate_multi_criterion(
        cls,
        model_name: str,
        df_combined: pd.DataFrame,
    ) -> ComprehensiveDisagreementReport:
        """Run comprehensive analysis across EXP, ICM, and TEQ from combined dataframe."""
        criteria = ["EXP", "ICM", "TEQ"]
        criteria_reports: Dict[str, CriterionDisagreementReport] = {}

        all_human = []
        all_model = []

        for crit in criteria:
            h_col = f"human_entropy_{crit}"
            m_col = f"model_entropy_{crit}"
            u_col = f"is_unanimous_{crit}"
            t_col = f"is_tie_{crit}"

            if h_col in df_combined.columns and m_col in df_combined.columns:
                h_vals = df_combined[h_col].values
                m_vals = df_combined[m_col].values
                u_vals = df_combined[u_col].values if u_col in df_combined.columns else None
                t_vals = df_combined[t_col].values if t_col in df_combined.columns else None

                rep = cls.analyze_criterion(h_vals, m_vals, crit, u_vals, t_vals)
                criteria_reports[crit] = rep

                all_human.extend(h_vals)
                all_model.extend(m_vals)

        # Pooled correlation
        all_h_arr = np.array(all_human, dtype=float)
        all_m_arr = np.array(all_model, dtype=float)
        valid = ~(np.isnan(all_h_arr) | np.isnan(all_m_arr))
        n_pooled = int(np.sum(valid))

        sp_pool = stats.spearmanr(all_h_arr[valid], all_m_arr[valid])
        pe_pool = stats.pearsonr(all_h_arr[valid], all_m_arr[valid])

        sp_ci_l, sp_ci_h = cls.compute_spearman_bootstrap_ci(all_h_arr[valid], all_m_arr[valid], random_seed=42)
        pe_ci_l, pe_ci_h = cls.compute_fisher_ci(float(pe_pool.statistic), n_pooled)

        sp_overall = CorrelationResult(
            metric="Pooled_Spearman_rho (Bootstrap)",
            coefficient=round(float(sp_pool.statistic), 4),
            p_value=float(f"{sp_pool.pvalue:.4e}"),
            ci_95_lower=sp_ci_l,
            ci_95_upper=sp_ci_h,
            sample_size=n_pooled,
        )

        pe_overall = CorrelationResult(
            metric="Pooled_Pearson_r (Fisher_z)",
            coefficient=round(float(pe_pool.statistic), 4),
            p_value=float(f"{pe_pool.pvalue:.4e}"),
            ci_95_lower=pe_ci_l,
            ci_95_upper=pe_ci_h,
            sample_size=n_pooled,
        )

        disclaimer = (
            "Human consensus does not constitute ground-truth biological viability. "
            "Statistical correlation between model predictive entropy and human voting entropy "
            "reflects mutual susceptibility to perceptual ambiguity, not clinical understanding."
        )

        return ComprehensiveDisagreementReport(
            model_name=model_name,
            criteria_reports=criteria_reports,
            overall_spearman=sp_overall,
            overall_pearson=pe_overall,
            methodological_disclaimer=disclaimer,
        )

