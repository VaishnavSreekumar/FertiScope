"""Unit and integration tests for calibration, Brier score, and reliability engine."""

import numpy as np
import pytest

from fertiscope.calibration.brier import MulticlassBrier
from fertiscope.calibration.ece import MulticlassCalibration
from fertiscope.calibration.reliability import ReliabilityEngine
from fertiscope.core.exceptions import CalibrationError


def test_multiclass_ece_perfect_calibration():
    """A model predicting exactly calibrated probabilities has ECE near zero."""
    # 100 samples with 2 classes: 50 with [0.8, 0.2] where 40 are class 0 (acc=0.8),
    # 50 with [0.3, 0.7] where 35 are class 1 (acc=0.7).
    probs = np.vstack([
        np.tile([0.8, 0.2], (50, 1)),
        np.tile([0.3, 0.7], (50, 1)),
    ])
    labels = np.array(
        [0] * 40 + [1] * 10 + [1] * 35 + [0] * 15
    )

    report = MulticlassCalibration.compute_ece(probs, labels, num_bins=10)
    assert report.num_samples == 100
    assert report.num_classes == 2
    assert report.ece < 0.05
    assert len(report.classwise_ece) == 2


def test_multiclass_ece_input_validation():
    """Calibration engine strictly rejects NaNs, unnormalized rows, and shape mismatches."""
    # NaN inputs
    with pytest.raises(CalibrationError):
        MulticlassCalibration.compute_ece(np.array([[np.nan, 0.5]]), np.array([0]))

    # Row does not sum to 1.0
    with pytest.raises(CalibrationError):
        MulticlassCalibration.compute_ece(np.array([[0.8, 0.8]]), np.array([0]))

    # Dimension mismatch
    with pytest.raises(CalibrationError):
        MulticlassCalibration.compute_ece(np.array([[0.5, 0.5]]), np.array([0, 1]))


def test_multiclass_brier_score():
    """Brier score correctly computes quadratic penalty and skill score."""
    probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    labels = np.array([0, 1, 2])

    # Perfect predictions have Brier score of 0.0
    report = MulticlassBrier.compute(probs, labels)
    assert report.overall_brier_score == 0.0
    assert report.brier_skill_score == 1.0
    assert report.classwise_brier_scores[0] == 0.0

    # Completely wrong predictions
    wrong_labels = np.array([1, 2, 0])
    wrong_report = MulticlassBrier.compute(probs, wrong_labels)
    # Each sample error: (1-0)^2 + (0-1)^2 = 2.0 -> mean = 2.0
    assert wrong_report.overall_brier_score == 2.0
    assert wrong_report.brier_skill_score < 0.0


def test_reliability_diagram_and_ascii_rendering():
    """Reliability engine generates diagram view and non-empty ASCII representation."""
    probs = np.array([
        [0.9, 0.1],
        [0.85, 0.15],
        [0.4, 0.6],
        [0.2, 0.8],
    ])
    labels = np.array([0, 0, 1, 1])

    cal_report = MulticlassCalibration.compute_ece(probs, labels, num_bins=5)
    view = ReliabilityEngine.from_report(cal_report)

    assert len(view.bin_centers) == 5
    ascii_art = view.render_ascii(width=30)
    assert "Reliability Diagram" in ascii_art
    assert "Bin 00" in ascii_art


def test_temperature_scaling_safeguard():
    """Temperature scaling fits on validation sets but strictly prohibits test sets."""
    # Synthetic validation logits
    np.random.seed(42)
    logits = np.random.randn(50, 3)
    labels = np.random.randint(0, 3, size=50)

    # Valid validation split
    t_opt = ReliabilityEngine.fit_temperature_scaling(logits, labels, split_name="validation")
    assert t_opt > 0.0

    # Test set fitting must raise CalibrationError
    with pytest.raises(CalibrationError) as exc_info:
        ReliabilityEngine.fit_temperature_scaling(logits, labels, split_name="test_rev")

    assert "Methodological violation" in str(exc_info.value)
    assert "test" in str(exc_info.value)

    with pytest.raises(CalibrationError):
        ReliabilityEngine.fit_temperature_scaling(logits, labels, split_name="evaluation_split")

