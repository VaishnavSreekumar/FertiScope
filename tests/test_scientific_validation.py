"""Independent, hand-calculated mathematical validation tests for FertiScope Phase 2.

Every test case here has an analytically derived expected value computed independently
by hand from first principles, ensuring zero circularity against the library implementation.
"""

import numpy as np
import pytest

from fertiscope.analysis.human_ai_disagreement import HumanAIDisagreementAnalyzer
from fertiscope.calibration.brier import MulticlassBrier
from fertiscope.calibration.ece import MulticlassCalibration
from fertiscope.core.exceptions import CalibrationError, SchemaValidationError
from fertiscope.data.annotations import AnnotationParser
from fertiscope.data.splitting import CycleAwareSplitter
from fertiscope.metrics.agreement import InterAnnotatorAgreement


def test_hand_calculated_multiclass_brier_and_independent_baseline():
    """Verify multiclass Brier score against analytical hand derivation.

    N = 3 samples, K = 3 classes.
    Sample 1: y = 0, p = [0.8, 0.1, 0.1] -> SE = (0.8-1)^2 + 0.1^2 + 0.1^2 = 0.06
    Sample 2: y = 1, p = [0.2, 0.6, 0.2] -> SE = 0.2^2 + (0.6-1)^2 + 0.2^2 = 0.24
    Sample 3: y = 2, p = [0.1, 0.2, 0.7] -> SE = 0.1^2 + 0.2^2 + (0.7-1)^2 = 0.14
    Overall BS = (0.06 + 0.24 + 0.14) / 3 = 0.44 / 3 = 0.146666... -> 0.1467

    Class-wise:
    Class 0: (0.04 + 0.04 + 0.01) / 3 = 0.09 / 3 = 0.0300
    Class 1: (0.01 + 0.16 + 0.04) / 3 = 0.21 / 3 = 0.0700
    Class 2: (0.01 + 0.04 + 0.09) / 3 = 0.14 / 3 = 0.0467

    Independent reference prevalence forecast: p_ref = [0.5, 0.3, 0.2]
    Sample 1: (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
    Sample 2: 0.5^2 + (0.3-1)^2 + 0.2^2 = 0.25 + 0.49 + 0.04 = 0.78
    Sample 3: 0.5^2 + 0.3^2 + (0.2-1)^2 = 0.25 + 0.09 + 0.64 = 0.98
    BS_ref = (0.38 + 0.78 + 0.98) / 3 = 2.14 / 3 = 0.713333... -> 0.7133
    BSS = 1 - (0.44 / 2.14) = 1 - 0.205607 = 0.79439... -> 0.7944
    """
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ])
    labels = np.array([0, 1, 2])
    ref_probs = np.array([0.5, 0.3, 0.2])

    report = MulticlassBrier.compute(probs, labels, reference_probs=ref_probs)

    assert np.isclose(report.overall_brier_score, 0.1467, atol=1e-4)
    assert np.isclose(report.classwise_brier_scores[0], 0.0300, atol=1e-4)
    assert np.isclose(report.classwise_brier_scores[1], 0.0700, atol=1e-4)
    assert np.isclose(report.classwise_brier_scores[2], 0.0467, atol=1e-4)
    assert np.isclose(report.reference_brier_score, 0.7133, atol=1e-4)
    assert np.isclose(report.brier_skill_score, 0.7944, atol=1e-4)
    assert report.reference_type == "independent_prevalence"


def test_hand_calculated_ece_and_mce():
    """Verify multiclass ECE and MCE on hand-derived 4-sample 2-bin case.

    N = 4, K = 2 classes, M = 2 uniform bins [0.0, 0.5), [0.5, 1.0].
    p1 = [0.6, 0.4], y = 0 -> pred = 0 (correct), conf = 0.6
    p2 = [0.9, 0.1], y = 1 -> pred = 0 (wrong), conf = 0.9
    p3 = [0.2, 0.8], y = 1 -> pred = 1 (correct), conf = 0.8
    p4 = [0.1, 0.9], y = 1 -> pred = 1 (correct), conf = 0.9

    All confs in bin [0.5, 1.0]:
    acc = 3 / 4 = 0.75
    conf = (0.6 + 0.9 + 0.8 + 0.9) / 4 = 3.2 / 4 = 0.80
    gap = |0.75 - 0.80| = 0.05
    ECE = (4/4) * 0.05 = 0.0500
    MCE = 0.0500
    """
    probs = np.array([
        [0.6, 0.4],
        [0.9, 0.1],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    labels = np.array([0, 1, 1, 1])

    report = MulticlassCalibration.compute_ece(probs, labels, num_bins=2, strategy="uniform")

    assert np.isclose(report.ece, 0.0500, atol=1e-4)
    assert np.isclose(report.mce, 0.0500, atol=1e-4)
    assert report.bins[0].count == 0
    assert report.bins[1].count == 4
    assert np.isclose(report.bins[1].accuracy, 0.7500, atol=1e-4)
    assert np.isclose(report.bins[1].confidence, 0.8000, atol=1e-4)


def test_hand_calculated_cohens_kappa_chance():
    """Verify Cohen's Kappa against hand-calculated 2x2 contingency table.

    Rater A: [0, 0, 1, 1]
    Rater B: [0, 1, 0, 1]
    cm = [[1, 1], [1, 1]]
    p_o = 2 / 4 = 0.50
    p_e = 0.5*0.5 + 0.5*0.5 = 0.50
    kappa = (0.50 - 0.50) / (1 - 0.50) = 0.0000
    """
    ra = np.array([0, 0, 1, 1])
    rb = np.array([0, 1, 0, 1])

    rep = InterAnnotatorAgreement.cohens_kappa(ra, rb)
    assert np.isclose(rep.value, 0.0000, atol=1e-4)
    assert "chance" in rep.interpretation.lower() or "poor" in rep.interpretation.lower()


def test_hand_calculated_fleiss_kappa():
    """Verify Fleiss' Kappa against hand-calculated 2-subject 3-rater matrix.

    N = 2 subjects, m = 3 raters, K = 2 categories.
    Subject 1: [3, 0]
    Subject 2: [1, 2]
    p0 = 4 / 6 = 2/3, p1 = 2 / 6 = 1/3
    P_e = (2/3)^2 + (1/3)^2 = 5/9 = 0.555556
    P1 = (9 - 3) / 6 = 1.0
    P2 = (1 + 4 - 3) / 6 = 2/6 = 1/3
    P_bar = (1.0 + 1/3) / 2 = 2/3 = 0.666667
    kappa = (2/3 - 5/9) / (1 - 5/9) = (1/9) / (4/9) = 1/4 = 0.2500
    """
    matrix = np.array([
        [3, 0],
        [1, 2],
    ])

    rep = InterAnnotatorAgreement.fleiss_kappa(matrix)
    assert np.isclose(rep.value, 0.2500, atol=1e-4)
    assert rep.num_items == 2
    assert rep.num_raters == 3
    assert rep.num_categories == 2


def test_hand_calculated_shannon_voting_entropy():
    """Verify expert voting Shannon entropy against analytical base-2 formula.

    Votes: 3 for class 'A', 1 for class 'B' (total = 4)
    q(A) = 3/4 = 0.75, q(B) = 1/4 = 0.25
    H = -(0.75 * log2(0.75) + 0.25 * log2(0.25))
      = -(0.75 * -0.415037 + 0.25 * -2.0)
      = 0.311278 + 0.500000 = 0.811278 -> 0.8113 bits
    """
    dist = AnnotationParser.compute_vote_distribution(
        image_name="test_embryo.png",
        votes=["A", "A", "A", "B"],
        criterion="ICM",
    )
    assert np.isclose(dist.entropy, 0.8113, atol=1e-4)
    assert dist.num_votes == 4
    assert dist.consensus_class == "A"
    assert dist.is_unanimous is False


def test_spearman_bootstrap_ci_reproducibility():
    """Verify that bootstrap CI for Spearman correlation is deterministic and handles edge cases."""
    rng = np.random.default_rng(123)
    x = np.arange(20, dtype=float)
    y = x + rng.normal(0, 1.0, 20)

    # Deterministic output under fixed seed
    ci1_low, ci1_high = HumanAIDisagreementAnalyzer.compute_spearman_bootstrap_ci(
        x, y, n_resamples=500, confidence=0.95, random_seed=42
    )
    ci2_low, ci2_high = HumanAIDisagreementAnalyzer.compute_spearman_bootstrap_ci(
        x, y, n_resamples=500, confidence=0.95, random_seed=42
    )
    assert ci1_low == ci2_low
    assert ci1_high == ci2_high
    assert -1.0 <= ci1_low < ci1_high <= 1.0
    assert ci1_low > 0.50  # Strong positive correlation

    # Degenerate input handling
    constant_x = np.ones(10)
    normal_y = np.arange(10, dtype=float)
    ci_deg_low, ci_deg_high = HumanAIDisagreementAnalyzer.compute_spearman_bootstrap_ci(
        constant_x, normal_y, n_resamples=100, random_seed=42
    )
    assert ci_deg_low == -1.0 and ci_deg_high == 1.0


def test_cycle_split_record_conservation_and_diagnostics():
    """Cycle splitter conserves all records, prevents leakage, and reports full diagnostics."""
    import pandas as pd

    # 30 cycles with 1 to 4 embryos each
    records = []
    for c in range(1, 31):
        n_emb = (c % 4) + 1
        for e in range(1, n_emb + 1):
            records.append({
                "image_name": f"{c:04d}_{e:02d}.png",
                "cycle_id": c,
                "embryo_index": e,
                "EXP": (c % 6) + 1,
                "ICM": ["A", "B", "C"][c % 3],
            })
    df_cohort = pd.DataFrame(records)

    train_df, val_df, test_df, diag = CycleAwareSplitter.create_cycle_grouped_split(
        df_cohort,
        group_col="cycle_id",
        test_ratio=0.20,
        val_ratio=0.20,
        random_seed=100,
        return_diagnostics=True,
    )

    # 1. Total records conserved
    assert diag.records_conserved is True
    assert diag.total_images_allocated == len(df_cohort)
    assert diag.total_images_input == len(df_cohort)

    # 2. 0% cycle leakage verified
    assert diag.overlap_report.has_leakage is False
    assert diag.overlap_report.cycle_leakage_rate == 0.0

    # 3. All splits non-empty with reported class counts
    for split_name in ("train", "val", "test"):
        part = diag.partitions[split_name]
        assert part.num_images > 0
        assert part.num_cycles > 0
        assert "EXP" in part.class_counts
        assert "ICM" in part.class_counts

    # 4. Methodological limitation is explicitly stated
    assert "oocyte retrieval cohorts" in diag.methodological_note
    assert "not confirmed unique patient identities" in diag.methodological_note

