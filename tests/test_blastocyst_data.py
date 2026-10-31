"""Unit and integration tests for independent blastocyst data loaders, annotations, and predictions."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fertiscope.core.exceptions import (
    AnnotationFormatError,
    CorruptedFileError,
    DatasetNotFoundError,
    SchemaValidationError,
)
from fertiscope.data.annotations import AnnotationParser, ExpertVoteDistribution
from fertiscope.data.blastocyst_dataset import BlastocystDatasetLoader
from fertiscope.data.predictions import ModelPredictionParser


def test_filename_parsing():
    """Loader correctly extracts cycle ID and embryo index from filenames."""
    loader = BlastocystDatasetLoader(data_dir="dummy")

    assert loader.parse_filename("0001_01.png") == (1, 1, "0001")
    assert loader.parse_filename("0042_05.png") == (42, 5, "0042")
    assert loader.parse_filename("198_01.png") == (198, 1, "198")
    assert loader.parse_filename("851_03.png") == (851, 3, "851")

    # Invalid names
    assert loader.parse_filename("invalid_embryo.png") is None
    assert loader.parse_filename("0001_01.txt") is None


def test_build_manifest_and_summary(tmp_path: Path):
    """Loader scans directory, produces reproducible manifest and clustering summary."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create dummy image files
    (img_dir / "0001_01.png").write_text("dummy1")
    (img_dir / "0001_02.png").write_text("dummy2")
    (img_dir / "0002_01.png").write_text("dummy3")
    (img_dir / "198_01.png").write_text("dummy4")

    loader = BlastocystDatasetLoader(data_dir=img_dir)
    manifest = loader.build_manifest(compute_hashes=True)

    assert len(manifest) == 4
    assert list(manifest["cycle_id"]) == [1, 1, 2, 198]
    assert manifest["sha256"].notnull().all()

    summary = loader.summarize(manifest)
    assert summary.total_images == 4
    assert summary.total_unique_cycles == 3
    assert summary.min_embryos_per_cycle == 1
    assert summary.max_embryos_per_cycle == 2
    assert summary.cycles_with_multiple_embryos == 1


def test_annotation_grade_standardization():
    """Annotation parser standardizes grades and maps missing values to None."""
    # Expansion (1-6 -> 0-5)
    assert AnnotationParser.standardize_grade(1, "EXP") == 0
    assert AnnotationParser.standardize_grade("6", "EXP") == 5
    assert AnnotationParser.standardize_grade("ND", "EXP") is None
    assert AnnotationParser.standardize_grade(np.nan, "EXP") is None

    # ICM and TEQ (A, B, C -> 0, 1, 2)
    assert AnnotationParser.standardize_grade("A", "ICM") == 0
    assert AnnotationParser.standardize_grade("B", "TEQ") == 1
    assert AnnotationParser.standardize_grade("c", "ICM") == 2
    assert AnnotationParser.standardize_grade("ND", "TEQ") is None
    assert AnnotationParser.standardize_grade("", "ICM") is None

    # Invalid grades raise AnnotationFormatError
    with pytest.raises(AnnotationFormatError):
        AnnotationParser.standardize_grade("D", "ICM")

    with pytest.raises(AnnotationFormatError):
        AnnotationParser.standardize_grade(7, "EXP")


def test_expert_vote_distribution_and_entropy():
    """Parser computes discrete voting distributions, unanimous status, and Shannon entropy."""
    # Unanimous consensus: 5 votes for 'A'
    dist_unanimous = AnnotationParser.compute_vote_distribution(
        image_name="0001_01.png",
        votes=["A", "A", "A", "A", "A"],
        criterion="ICM",
    )
    assert dist_unanimous.num_votes == 5
    assert dist_unanimous.consensus_class == "A"
    assert dist_unanimous.is_unanimous is True
    assert dist_unanimous.is_tie is False
    assert dist_unanimous.entropy == 0.0

    # Tied consensus: 2 for 'A', 2 for 'B', 1 ND
    dist_tied = AnnotationParser.compute_vote_distribution(
        image_name="0001_02.png",
        votes=["A", "A", "B", "B", "ND"],
        criterion="ICM",
    )
    assert dist_tied.num_votes == 4
    assert dist_tied.is_tie is True
    assert dist_tied.is_unanimous is False
    # P(A) = 0.5, P(B) = 0.5 -> Entropy = -2 * (0.5 * log2(0.5)) = 1.0 bit
    assert np.isclose(dist_tied.entropy, 1.0)


def test_model_prediction_parser(tmp_path: Path):
    """Prediction parser correctly validates probabilities and detects corruption."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    valid_model_dir = models_dir / "valid_model"
    valid_model_dir.mkdir()

    # Create valid exp.csv with 6 classes summing to 1.0
    valid_df = pd.DataFrame({
        "image_name": ["0001_01.png", "0001_02.png"],
        "0": [0.70, 0.10],
        "1": [0.15, 0.60],
        "2": [0.05, 0.10],
        "3": [0.05, 0.10],
        "4": [0.03, 0.05],
        "5": [0.02, 0.05],
        "exp_std": [0.12, 0.18],
    })
    valid_df.to_csv(valid_model_dir / "exp.csv", index=False)

    parser = ModelPredictionParser(models_root=models_dir, strict=True)
    assert "valid_model" in parser.get_available_models()

    df_loaded, summary = parser.load_criterion_probabilities("valid_model", "EXP")
    assert summary.has_valid_probabilities is True
    assert summary.is_corrupted is False
    assert summary.has_swag_uncertainty is True
    assert summary.num_classes == 6
    assert "predictive_entropy" in df_loaded.columns
    assert "confidence" in df_loaded.columns

    # Test corrupted model handling
    corrupted_model_dir = models_dir / "corrupted_model"
    corrupted_model_dir.mkdir()
    corrupt_df = pd.DataFrame({
        "image_name": ["0001_01.png", "0001_02.png"],
        "0": [np.nan, np.nan],
        "1": [np.nan, np.nan],
        "2": [np.nan, np.nan],
    })
    corrupt_df.to_csv(corrupted_model_dir / "teq.csv", index=False)

    # In strict mode, should raise CorruptedFileError
    with pytest.raises(CorruptedFileError):
        parser.load_criterion_probabilities("corrupted_model", "TEQ")

    # In permissive mode, returns diagnostic summary with is_corrupted=True
    permissive_parser = ModelPredictionParser(models_root=models_dir, strict=False)
    _, corrupt_summary = permissive_parser.load_criterion_probabilities("corrupted_model", "TEQ")
    assert corrupt_summary.is_corrupted is True
    assert "NaN" in corrupt_summary.corruption_reason


def test_real_benchmark_integration_if_available():
    """Verify integration against local benchmark files if available."""
    ref_repo = Path(r"C:\Users\vaish\AppData\Local\Temp\ref_blastocyst\repo")
    if not ref_repo.exists():
        pytest.skip("Reference benchmark repository not locally present in temp.")

    # 1. Parse ground truth test set
    test_csv = ref_repo / "annotations" / "test_rev.csv"
    if test_csv.exists():
        df_gt = AnnotationParser.parse_test_ground_truth(test_csv)
        assert len(df_gt) == 300
        assert "EXP_class" in df_gt.columns
        assert "ICM_class" in df_gt.columns
        assert "TEQ_class" in df_gt.columns

    # 2. Check model predictions and corrupted TEQ file
    models_dir = ref_repo / "model_predictions"
    if models_dir.exists():
        parser = ModelPredictionParser(models_root=models_dir, strict=False)
        available = parser.get_available_models()
        assert len(available) == 12

        # Check an intact model: coatnet_nano
        df_exp, sum_exp = parser.load_criterion_probabilities("coatnet_nano", "EXP")
        assert sum_exp.has_valid_probabilities is True
        assert sum_exp.is_corrupted is False
        assert len(df_exp) == 300

        # Check corrupted model: convnextv2_tiny TEQ
        _, corrupt_sum = parser.load_criterion_probabilities("convnextv2_tiny", "TEQ")
        assert corrupt_sum.is_corrupted is True
        assert "NaN" in corrupt_sum.corruption_reason

