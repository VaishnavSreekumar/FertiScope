# Changelog

All notable changes to the FertiScope project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-01

### Added
- **STAGE 2: Clean-Room Independent Implementation** established under strict clean-room governance.
- Zero source code copied or ported from the unlicensed reference GitHub repository; all components developed independently from first principles.
- **Component 1 (Data Infrastructure)**:
  - `fertiscope.data.blastocyst_dataset.BlastocystDatasetLoader`: Typed loader parsing filename patterns (`^(\d+)_(\d+)\.png$`), generating reproducible image manifests, and calculating cohort clustering statistics.
  - `fertiscope.data.annotations.AnnotationParser`: Strict parser for Gardner criteria (EXP 1..6, ICM A..C, TEQ A..C), multi-annotator matrices, discrete voting distributions, unanimous status, and Shannon entropy.
  - `fertiscope.data.predictions.ModelPredictionParser`: Independent parser for 12 deep learning benchmark architectures, validating discrete probability axioms ($\sum p_i = 1.0$), extracting SWA-G posterior standard deviations, and isolating corrupted files.
  - `CorruptedFileError` taxonomy and explicit detection of completely corrupted `convnextv2_tiny/teq.csv` (100% NaNs).
- **Component 2 (Cycle-Aware Evaluation)**:
  - `fertiscope.data.splitting.CycleAwareSplitter`: Group-aware splitting allocating entire retrieval cycles to single partitions (0% cycle leakage across train/val/test).
  - Naive split reproduction replicating the exact 242/300 cycle leakage documented in Kromp et al.
  - Split overlap auditor computing set intersections, sample counts, and leakage proportions with strict zero-leakage enforcement.
  - `fertiscope.analysis.cycle_overlap.CycleLeakageEvaluator`: Quantitative subcohort comparison measuring performance delta between cycle-overlapping ($n=242$) and cycle-isolated ($n=58$) test cohorts.
- **Component 3 (Calibration & Reliability Engine)**:
  - `fertiscope.calibration.ece.MulticlassCalibration`: Multiclass Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Class-wise ECE with uniform and quantile binning.
  - `fertiscope.calibration.brier.MulticlassBrier`: Multiclass Brier score, class-wise quadratic penalties, and Brier Skill Score (BSS) against climatological prior baseline.
  - `fertiscope.calibration.reliability.ReliabilityEngine`: Structured reliability diagram generation and ASCII visualization.
  - Temperature scaling fitting on validation splits with mandatory safeguard raising `CalibrationError` on evaluation test partitions.
- **Component 4 (Human-AI Disagreement Analysis)**:
  - `fertiscope.metrics.agreement.InterAnnotatorAgreement`: Independent implementations of Cohen's Kappa, Fleiss' Kappa, and multi-rater pairwise concordance rates.
  - `fertiscope.metrics.classification.MulticlassClassificationMetrics`: Multiclass accuracy, balanced accuracy, macro/micro F1, and confusion matrix diagnostics.
  - `fertiscope.analysis.human_ai_disagreement.HumanAIDisagreementAnalyzer`: Evaluation of human Shannon voting entropy vs algorithmic predictive entropy and SWA-G uncertainty, with Spearman $\rho$, Pearson $r$, and Fisher $z$-transform 95% confidence intervals.
- Comprehensive test suite extended to 28 passing tests (100% pass rate) across unit, split-integrity, calibration, and benchmark-integration suites.
- Formal documentation in `docs/STAGE_2_IMPLEMENTATION_REPORT.md`.

## [0.1.0] - 2026-11-01

### Added
- **PHASE 1: Research & Data Foundation** initialized.
- Repository structure, clean architecture boundaries, and `.gitignore` protecting patient data privacy.
- Standardized package configuration (`pyproject.toml`) supporting Python 3.11+.
- Structured logging (`fertiscope.core.logging`) and comprehensive domain exception taxonomy (`fertiscope.core.exceptions`).
- Data provenance tracking container (`fertiscope.data.provenance`) with cryptographic SHA-256 computation.
- Automated data auditor (`fertiscope.data.audit.RawDataAuditor`) inspecting raw Excel workbooks, sheet structures, column dimensions, and missingness.
- Biological range validation schemas (`fertiscope.data.schema`) implemented via Pydantic v2.
- Exhaustive test suite (`tests/test_data_foundation.py`) passing with 100% success rate across 5 core verification scenarios.
- Comprehensive Phase 1 Research & Data Foundation audit document (`docs/PHASE_1_DATA_FOUNDATION_AUDIT.md`).
- Primary candidate source investigation documenting Mendeley Data HTTP 451 status and DataCite registration status.

