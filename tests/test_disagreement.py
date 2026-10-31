"""Unit and integration tests for human-AI disagreement and inter-rater metrics."""

import numpy as np
import pandas as pd
import pytest

from fertiscope.analysis.human_ai_disagreement import HumanAIDisagreementAnalyzer
from fertiscope.metrics.agreement import InterAnnotatorAgreement
from fertiscope.metrics.classification import MulticlassClassificationMetrics


def test_cohens_kappa():
    """Cohen's Kappa reflects complete agreement vs random chance."""
    # Perfect agreement
    r1 = np.array([0, 1, 2, 0, 1])
    r2 = np.array([0, 1, 2, 0, 1])
    rep_perfect = InterAnnotatorAgreement.cohens_kappa(r1, r2)
    assert rep_perfect.value == 1.0
    assert "Almost perfect" in rep_perfect.interpretation

    # Complete disagreement
    r_opp = np.array([1, 2, 0, 1, 2])
    rep_opp = InterAnnotatorAgreement.cohens_kappa(r1, r_opp)
    assert rep_opp.value < 0.0


def test_fleiss_kappa():
    """Fleiss' Kappa computes multi-rater categorical concordance."""
    # 5 subjects evaluated by 4 raters into 3 categories
    # Subject 1: unanimous (4, 0, 0)
    # Subject 2: split (2, 2, 0)
    # Subject 3: unanimous (0, 4, 0)
    # Subject 4: unanimous (0, 0, 4)
    # Subject 5: split (1, 2, 1)
    matrix = np.array([
        [4, 0, 0],
        [2, 2, 0],
        [0, 4, 0],
        [0, 0, 4],
        [1, 2, 1],
    ])

    rep = InterAnnotatorAgreement.fleiss_kappa(matrix)
    assert rep.num_items == 5
    assert rep.num_raters == 4
    assert rep.value > 0.40  # Moderate/substantial


def test_pairwise_concordance():
    """Calculates overall pairwise consensus rate among raters."""
    ratings = [
        ["A", "A", "A"],  # 3 pairs, all agree (3/3)
        ["A", "B", "A"],  # 3 pairs, (A,A) agree, (A,B), (B,A) disagree (1/3)
    ]
    # Total pairs = 6, total agree = 4 -> 4/6 = 0.6667
    concordance = InterAnnotatorAgreement.pairwise_concordance_rate(ratings)
    assert np.isclose(concordance, 0.6667, atol=1e-3)


def test_fisher_ci_calculation():
    """Fisher z-transform returns mathematically valid confidence intervals bounded by [-1, 1]."""
    ci_low, ci_high = HumanAIDisagreementAnalyzer.compute_fisher_ci(r=0.50, n=100)
    assert -1.0 <= ci_low < 0.50 < ci_high <= 1.0
    assert ci_low > 0.25
    assert ci_high < 0.70


def test_disagreement_correlation_analysis():
    """Analyzer computes Spearman and Pearson correlation between entropy vectors."""
    np.random.seed(42)
    # Generate positively correlated entropies
    human_entropy = np.linspace(0.1, 1.5, 30)
    noise = np.random.normal(0, 0.1, 30)
    model_entropy = 0.8 * human_entropy + noise

    report = HumanAIDisagreementAnalyzer.analyze_criterion(
        human_entropies=human_entropy,
        model_entropies=model_entropy,
        criterion="ICM",
    )

    assert report.criterion == "ICM"
    assert report.num_images == 30
    assert report.spearman_correlation.coefficient > 0.80
    assert report.spearman_correlation.p_value < 0.001
    assert report.pearson_correlation.coefficient > 0.80


def test_multiclass_classification_metrics():
    """Metrics calculator produces comprehensive diagnostic summary."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    summary = MulticlassClassificationMetrics.evaluate(y_true, y_pred, criterion="EXP", num_classes=3)
    assert summary.criterion == "EXP"
    assert summary.num_samples == 6
    assert summary.num_classes == 3
    assert 0.0 <= summary.accuracy <= 1.0
    assert 0.0 <= summary.macro_f1 <= 1.0
    assert len(summary.confusion_matrix) == 3

