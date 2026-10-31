# FertiScope — Stage 2: Clean-Room Independent Implementation Report

**Author:** FertiScope Research Engineering  
**Date:** 2026-11-01  
**Project Identity:** Independent Reproductive-Health AI Research Engineering  
**Repository Branch:** `master`  
**License & Attribution:** Clean-room independent software implementation. Dataset attribution preserved under CC BY 4.0 (Florian Kromp et al., 2023, *Scientific Data*, DOI: `10.1038/s41597-023-02182-3`; Figshare DOI: `10.6084/m9.figshare.20123153.v3`).  
**Status:** **STAGE 2 COMPLETE — AWAITING EXPLICIT APPROVAL FOR STAGE 3**

---

## 1. Executive Summary

Stage 2 establishes FertiScope as a fully independent research codebase, implementing four modular components from first principles with zero source code copied, ported, or vendored from the unlicensed reference GitHub repository.

All 28 tests across unit, split-integrity, calibration, and benchmark-integration suites pass with 100% success rate in 4.59s.

### Summary of Delivered Capabilities

| Component | Modules | Key Functions / Classes | Verified Output |
| :--- | :--- | :--- | :--- |
| **1. Data Infrastructure** | `data.blastocyst_dataset`<br>`data.annotations`<br>`data.predictions` | `BlastocystDatasetLoader`<br>`AnnotationParser`<br>`ModelPredictionParser` | Typed loading, SHA-256 provenance, handling of headerless/alternating formats, explicit detection and isolation of corrupted `convnextv2_tiny/teq.csv` (100% NaNs). |
| **2. Cycle-Aware Evaluation** | `data.splitting`<br>`analysis.cycle_overlap` | `CycleAwareSplitter`<br>`CycleLeakageEvaluator` | Cycle extraction (`^(\d+)_(\d+)\.png$`), 0% leakage group-splitting, naive benchmark reproduction (242/300 cycle leakage), leaked vs isolated subcohort performance comparison. |
| **3. Calibration Engine** | `calibration.ece`<br>`calibration.brier`<br>`calibration.reliability` | `MulticlassCalibration`<br>`MulticlassBrier`<br>`ReliabilityEngine` | Multiclass ECE, MCE, class-wise ECE, Brier score, Brier skill score, ASCII reliability diagrams, and strict prohibition of fitting temperature scaling on test sets. |
| **4. Human–AI Disagreement** | `metrics.agreement`<br>`metrics.classification`<br>`analysis.human_ai_disagreement` | `InterAnnotatorAgreement`<br>`MulticlassClassificationMetrics`<br>`HumanAIDisagreementAnalyzer` | Cohen's Kappa, Fleiss' Kappa, pairwise concordance, multiclass confusion metrics, Shannon voting entropy vs predictive entropy, Spearman $\rho$, Pearson $r$, Fisher $z$-transform 95% CIs. |

---

## 2. Clean-Room Governance & Legal Integrity

1. **Zero Source Code Reuse:** The original authors' GitHub repository contains no open-source license, reserving all exclusive rights under default international copyright. FertiScope was written completely from scratch using mathematical definitions, clinical domain standards (Gardner grading), and standard scientific Python libraries (`numpy`, `pandas`, `scipy`, `scikit-learn`, `pydantic`).
2. **Dataset Attribution:** The underlying dataset is distributed under Creative Commons Attribution 4.0 International (CC BY 4.0). FertiScope preserves full attribution to Florian Kromp et al. and Figshare DOI `10.6084/m9.figshare.20123153.v3`.
3. **No Synthetic Fabrication:** No missing annotations or corrupted prediction values are silently fabricated or imputed.

---

## 3. Implemented File Tree

```
c:\Users\vaish\.gemini\antigravity\scratch\fertiscope\
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
├── README.md
├── docs/
│   ├── PHASE_1_DATA_FOUNDATION_AUDIT.md
│   ├── INDEPENDENT_REPOSITORY_AND_RESEARCH_GAP_AUDIT.md
│   ├── RESEARCH_CLAIMS_VERIFICATION.md
│   └── STAGE_2_IMPLEMENTATION_REPORT.md
├── src/fertiscope/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── annotations.py
│   │   ├── audit.py
│   │   ├── blastocyst_dataset.py
│   │   ├── predictions.py
│   │   ├── provenance.py
│   │   ├── schema.py
│   │   └── splitting.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── agreement.py
│   │   └── classification.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── brier.py
│   │   ├── ece.py
│   │   └── reliability.py
│   └── analysis/
│       ├── __init__.py
│       ├── cycle_overlap.py
│       └── human_ai_disagreement.py
└── tests/
    ├── test_data_foundation.py
    ├── test_blastocyst_data.py
    ├── test_splitting.py
    ├── test_calibration.py
    └── test_disagreement.py
```

---

## 4. Mathematical Definitions of Implemented Metrics

### 4.1 Expected Calibration Error (ECE) & Maximum Calibration Error (MCE)

For an evaluation set of $N$ samples with model predicted probability vectors $p_i = (p_{i1}, \dots, p_{iK})$, confidence is defined as $\hat{p}_i = \max_k p_{ik}$ and predicted class as $\hat{y}_i = \arg\max_k p_{ik}$.

Dividing the interval $[0, 1]$ into $M$ bins $B_1, \dots, B_M$:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
$$\text{MCE} = \max_{m=1..M} |\text{acc}(B_m) - \text{conf}(B_m)|$$

**Class-wise ECE (One-vs-Rest):**
For each class $k \in \{1, \dots, K\}$, predictions are treated as binary indicators with predicted probability $p_{ik}$:
$$\text{ECE}_k = \sum_{m=1}^M \frac{|B_{m,k}|}{N} |\text{acc}(B_{m,k}) - \text{conf}(B_{m,k})|$$

### 4.2 Multiclass Brier Score & Brier Skill Score (BSS)

Given one-hot encoded ground truth vectors $y_i = (y_{i1}, \dots, y_{iK})$ where $y_{ik} \in \{0, 1\}$:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$

**Class-wise Brier Score:**
$$\text{BS}_k = \frac{1}{N} \sum_{i=1}^N (p_{ik} - y_{ik})^2$$

**Brier Skill Score:**
Relative to reference baseline $\bar{y}_k = \frac{1}{N} \sum_{i=1}^N y_{ik}$:
$$\text{BS}_{\text{ref}} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (\bar{y}_k - y_{ik})^2$$
$$\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{ref}}}$$
A score of $\text{BSS} > 0$ indicates predictive skill superior to climatological/marginal frequency guessing.

### 4.3 Human Voting Shannon Entropy & Model Predictive Entropy

Given discrete human votes $V(x) = [v_1, \dots, v_R]$ mapped to discrete distribution $q(k) = \frac{1}{R} \sum_{r=1}^R \mathbf{1}(v_r = k)$:
$$H_{\text{human}}(x) = -\sum_{k: q(k) > 0} q(k) \log_2 q(k)$$

Given model softmax output probabilities $p(k)$:
$$H_{\text{model}}(x) = -\sum_{k: p(k) > 0} p(k) \log_2 p(k)$$

### 4.4 Inter-Annotator Agreement (Cohen's & Fleiss' Kappa)

**Cohen's Kappa ($\kappa$):**
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ is observed relative agreement, and $p_e$ is hypothetical chance agreement computed from rater marginals.

**Fleiss' Kappa:**
For $N$ subjects rated by $m$ raters across $K$ categories, where $n_{ij}$ is the count of raters assigning subject $i$ to category $j$:
$$P_i = \frac{1}{m(m-1)} \left( \sum_{j=1}^K n_{ij}^2 - m \right), \quad \bar{P}_o = \frac{1}{N}\sum_{i=1}^N P_i$$
$$p_j = \frac{1}{Nm}\sum_{i=1}^N n_{ij}, \quad \bar{P}_e = \sum_{j=1}^K p_j^2$$
$$\kappa_{\text{Fleiss}} = \frac{\bar{P}_o - \bar{P}_e}{1 - \bar{P}_e}$$

### 4.5 Fisher $z$-Transformation for Correlation 95% Confidence Intervals

For correlation coefficient $r$ across sample size $N$:
$$z = \operatorname{arctanh}(r) = \frac{1}{2} \ln \left(\frac{1+r}{1-r}\right), \quad \text{SE} = \frac{1}{\sqrt{N-3}}$$
$$z_{\text{lower}}, z_{\text{upper}} = z \pm 1.96 \cdot \text{SE}$$
$$\text{CI}_{95\%} = \left[ \tanh(z_{\text{lower}}), \tanh(z_{\text{upper}}) \right]$$

---

## 5. Source File Corruption and Architectural Peculiarities

1. **`convnextv2_tiny/teq.csv` Corruption:**
   - The file contains 100% `NaN` values across all 300 test images.
   - Handled in `ModelPredictionParser` by returning `is_corrupted = True` with reason `"All probability values are NaN"`. In strict mode (`strict=True`), it raises `CorruptedFileError`.
2. **Benchmark Prediction File Structure:**
   - Prediction files (`exp.csv`, `icm.csv`, `teq.csv`) are headerless and have a trailing comma on every row.
   - Columns alternate: `[image_name, prob_0, std_0, prob_1, std_1, ...]`.
   - EXP has 5 classes (10 numeric columns). ICM and TEQ have 4 classes (8 numeric columns, where Class 3 represents the unassessable / not-defined class).
   - `ModelPredictionParser` automatically parses and validates these alternating structures and separates probabilities from SWA-G posterior standard deviations.
3. **Ground Truth `test_rev.csv` Structure:**
   - Also headerless. Contains unassessable / not defined labels (`-1` or `3` for ICM/TEQ; `-1` for EXP).
   - Handled cleanly in `AnnotationParser` by mapping missing/unassessable tokens to `None`.

---

## 6. Demonstration of Split Integrity and Leakage Diagnostics

The cycle-aware splitting and leakage diagnostic suite was verified against synthetic and real benchmark cohorts:

1. **Group-Aware Splitting:**
   - `CycleAwareSplitter.create_cycle_grouped_split()` guarantees 0% cycle leakage across train, val, and test splits.
   - Tested in `test_cycle_grouped_split_zero_leakage`: `report.has_leakage is False`, `report.cycle_leakage_rate == 0.0`.
2. **Detection of Intentional Leakage:**
   - Tested in `test_audit_detects_deliberate_leakage`: Correctly identifies sister embryos split between train and test, raising `SplitLeakageError` under strict enforcement.
3. **Leaked vs Isolated Evaluation:**
   - `CycleLeakageEvaluator.compare_cohorts()` partitions test data into leaked ($n=242$) and isolated ($n=58$) subcohorts, measuring accuracy deltas and log loss shifts.

---

## 7. Methodological Safeguards and Calibration Governance

1. **Test-Set Prohibition:**
   - `ReliabilityEngine.fit_temperature_scaling()` strictly prohibits fitting temperature parameters on evaluation test partitions (`split_name="test"`, `"eval"`, etc.), raising `CalibrationError`.
2. **Distinction of Uncertainty Semantics:**
   - SWA-G posterior standard deviation ($\sigma_{\text{swag}}$) represents model parameter posterior variance under stochastic weight averaging.
   - Softmax predictive entropy ($H(p)$) represents categorical output distribution spread.
   - The two quantities are maintained as separate fields and never conflated.

---

## 8. Complete Test Suite Execution

All 28 tests pass synchronously:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: C:\Users\vaish\.gemini\antigravity\scratch\fertiscope
configfile: pyproject.toml
plugins: anyio-4.11.0, asyncio-1.4.0
collected 28 items

tests/test_blastocyst_data.py::test_filename_parsing PASSED              [  3%]
tests/test_blastocyst_data.py::test_build_manifest_and_summary PASSED    [  7%]
tests/test_blastocyst_data.py::test_annotation_grade_standardization PASSED [ 10%]
tests/test_blastocyst_data.py::test_expert_vote_distribution_and_entropy PASSED [ 14%]
tests/test_blastocyst_data.py::test_model_prediction_parser PASSED       [ 17%]
tests/test_blastocyst_data.py::test_real_benchmark_integration_if_available PASSED [ 21%]
tests/test_calibration.py::test_multiclass_ece_perfect_calibration PASSED [ 25%]
tests/test_calibration.py::test_multiclass_ece_input_validation PASSED   [ 28%]
tests/test_calibration.py::test_multiclass_brier_score PASSED            [ 32%]
tests/test_calibration.py::test_reliability_diagram_and_ascii_rendering PASSED [ 35%]
tests/test_calibration.py::test_temperature_scaling_safeguard PASSED    [ 39%]
tests/test_data_foundation.py::test_provenance_record_immutability_and_export PASSED [ 42%]
tests/test_data_foundation.py::test_sha256_computation PASSED            [ 46%]
tests/test_data_foundation.py::test_raw_auditor_detects_missing_files PASSED [ 50%]
tests/test_data_foundation.py::test_raw_auditor_with_mock_dataset PASSED [ 53%]
tests/test_data_foundation.py::test_schema_validation PASSED             [ 57%]
tests/test_data_foundation.py::test_candidate_hd8yy6gzc5_provenance_record PASSED [ 60%]
tests/test_data_foundation.py::test_candidate_hd8yy6gzc5_auditor PASSED [ 64%]
tests/test_data_foundation.py::test_candidate_hd8yy6gzc5_arithmetic_and_quality_metrics PASSED [ 67%]
tests/test_disagreement.py::test_cohens_kappa PASSED                     [ 71%]
tests/test_disagreement.py::test_fleiss_kappa PASSED                     [ 75%]
tests/test_disagreement.py::test_pairwise_concordance PASSED             [ 78%]
tests/test_disagreement.py::test_fisher_ci_calculation PASSED            [ 82%]
tests/test_disagreement.py::test_disagreement_correlation_analysis PASSED [ 85%]
tests/test_disagreement.py::test_multiclass_classification_metrics PASSED [ 89%]
tests/test_splitting.py::test_cycle_grouped_split_zero_leakage PASSED    [ 92%]
tests/test_splitting.py::test_audit_detects_deliberate_leakage PASSED   [ 96%]
tests/test_splitting.py::test_cycle_leakage_evaluator PASSED             [100%]

============================= 28 passed in 4.59s ==============================
```

---

## 9. Unresolved Scientific Limitations

1. **Cycle Identifier Semantics vs True Patient Identity:**
   - Filename prefix represents an oocyte retrieval cycle / embryo cohort.
   - While this prevents sister-embryo leakage within the same retrieval, approximately 14 repeat cycles come from the same couple. Without an explicit patient master index, true patient-level generalisation cannot be mathematically guaranteed.
2. **Subjectivity of Gardner Consensus:**
   - Gardner grading is inherently subjective; human expert unanimous consensus is achieved on only 29.7% of TEQ evaluations and 39.7% of ICM evaluations.
   - Consensus majority votes represent human inter-rater agreement, not ground-truth biological viability or live-birth potential.
3. **Correlation Does Not Establish Cognitive Comprehension:**
   - The positive correlation between human voting entropy and model predictive entropy ($\rho \approx 0.28$) demonstrates mutual sensitivity to visual ambiguity (e.g. out-of-focus optics, borderline cell boundaries), not that deep neural networks possess clinical reasoning.

---

## 10. Stopping Condition & Sign-off

Stage 2 Clean-Room Independent Implementation is complete.
In adherence to research governance rules:
- No models have been trained.
- No code has been copied from the unlicensed repository.
- No git commits or GitHub pushes have been executed.
- All 28 tests pass.

**FertiScope is stopped here, awaiting user review and explicit approval before advancing to Stage 3.**

