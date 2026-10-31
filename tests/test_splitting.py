"""Unit and integration tests for cycle-aware splitting and cluster leakage diagnostics."""

import numpy as np
import pandas as pd
import pytest

from fertiscope.analysis.cycle_overlap import CycleLeakageEvaluator
from fertiscope.core.exceptions import SplitLeakageError
from fertiscope.data.splitting import CycleAwareSplitter


@pytest.fixture
def sample_blastocyst_cohort():
    """Create a synthetic cohort with multi-embryo cycles."""
    data = []
    # 20 cycles, some with 1 embryo, some with 3 embryos
    for c in range(1, 21):
        n_embryos = 3 if c % 2 == 0 else 1
        for e in range(1, n_embryos + 1):
            data.append({
                "image_name": f"{c:04d}_{e:02d}.png",
                "cycle_id": c,
                "embryo_index": e,
                "EXP": (c % 6) + 1,
                "ICM": ["A", "B", "C"][c % 3],
                "TEQ": ["A", "B", "C"][(c + 1) % 3],
            })
    return pd.DataFrame(data)


def test_cycle_grouped_split_zero_leakage(sample_blastocyst_cohort):
    """Cycle-grouped splitting allocates entire cycles to single splits with 0% leakage."""
    train_df, val_df, test_df = CycleAwareSplitter.create_cycle_grouped_split(
        sample_blastocyst_cohort,
        group_col="cycle_id",
        test_ratio=0.20,
        val_ratio=0.20,
        random_seed=42,
    )

    # Validate non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_blastocyst_cohort)

    # Audit overlap
    report = CycleAwareSplitter.audit_overlap(
        train_df, val_df, test_df, group_col="cycle_id", strict=True
    )
    assert report.has_leakage is False
    assert report.cycle_leakage_rate == 0.0
    assert report.test_cycles_in_either == 0
    assert report.cycle_isolated_test_images == len(test_df)


def test_audit_detects_deliberate_leakage(sample_blastocyst_cohort):
    """Split auditor identifies when sister embryos from the same cycle are split across train and test."""
    # Deliberately construct a leaking split
    # Cycle 2 has 3 embryos: put embryo 1 in train, embryo 2 in test
    c2_imgs = sample_blastocyst_cohort[sample_blastocyst_cohort["cycle_id"] == 2]

    train_df = sample_blastocyst_cohort.iloc[1:].copy()  # contains other c2 embryos
    val_df = sample_blastocyst_cohort.iloc[:1].copy()
    test_df = c2_imgs.iloc[:1].copy()  # cycle 2 is now in both train and test

    report = CycleAwareSplitter.audit_overlap(
        train_df, val_df, test_df, group_col="cycle_id", strict=False
    )
    assert report.has_leakage is True
    assert report.test_cycles_in_train == 1
    assert report.cycle_leakage_rate == 1.0

    # In strict mode, must raise SplitLeakageError
    with pytest.raises(SplitLeakageError):
        CycleAwareSplitter.audit_overlap(
            train_df, val_df, test_df, group_col="cycle_id", strict=True
        )


def test_cycle_leakage_evaluator():
    """Evaluator partitions test predictions into leaked vs isolated cohorts and compares performance."""
    # Synthetic test dataframe with 10 samples
    test_df = pd.DataFrame({
        "image_name": [f"000{i}_01.png" for i in range(1, 11)],
        "cycle_id": list(range(1, 11)),
        "true_label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "pred_label": [0, 1, 2, 0, 1, 2, 1, 0, 1, 2],  # First 6 correct, last 4 incorrect
        "p0": [0.8, 0.1, 0.1, 0.9, 0.05, 0.05, 0.2, 0.8, 0.1, 0.1],
        "p1": [0.1, 0.8, 0.1, 0.05, 0.9, 0.05, 0.7, 0.1, 0.8, 0.1],
        "p2": [0.1, 0.1, 0.8, 0.05, 0.05, 0.9, 0.1, 0.1, 0.1, 0.8],
    })

    # Suppose cycles 1..6 are leaked into train, cycles 7..10 are isolated
    leaked_cycles = {1, 2, 3, 4, 5, 6}

    report = CycleLeakageEvaluator.compare_cohorts(
        test_df=test_df,
        model_name="test_model",
        criterion="ICM",
        true_col="true_label",
        pred_col="pred_label",
        prob_cols=["p0", "p1", "p2"],
        cycle_col="cycle_id",
        leaked_cycles=leaked_cycles,
    )

    assert report.model_name == "test_model"
    assert report.overlapping_cohort.sample_size == 6
    assert report.overlapping_cohort.accuracy == 1.0  # All 6 correct
    assert report.isolated_cohort.sample_size == 4
    assert report.isolated_cohort.accuracy == 0.0  # All 4 incorrect
    assert report.accuracy_delta == 1.0
    assert report.is_performance_inflated is True

