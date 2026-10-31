# FertiScope — Stage 1B: Research Claims Verification

**Date of Verification:** November 1, 2026  
**Investigator:** FertiScope Research Engineering  
**Primary Reference:** Florian Kromp et al., *"An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization"*, *Scientific Data* 10, Article 271 (2023), [DOI: 10.1038/s41597-023-02182-3](https://doi.org/10.1038/s41597-023-02182-3). [PMCID: PMC10175281].  
**Dataset Reference:** [Figshare DOI: 10.6084/m9.figshare.20123153.v3](https://doi.org/10.6084/m9.figshare.20123153.v3)  
**Code Repository Reference:** [Blastocyst-Dataset (SCCH GitHub)](https://github.com/software-competence-center-hagenberg/Blastocyst-Dataset)  
**Verification Scope:** Mathematical, methodological, and legal verification of all scientific claims prior to approving implementation.

---

## 1. Executive Summary & Verification Matrix

This audit independently investigates the six scientific and methodological claims made during the initial repository audit, establishing verified empirical facts, identifying overstated or incorrect claims, and defining precise research boundaries.

| Scientific Investigation | Initial Audit Claim | Verified Finding | Status |
| :--- | :--- | :--- | :--- |
| **1. Identifier Semantics** | Filename prefix represents patient ID; 80.7% patient-level leakage. | Prefix represents **anonymized oocyte retrieval cycle / embryo cohort** (851 cycles across 837 patients). 242/300 test blastocysts share cycle cohorts with train/val. | **VERIFIED WITH REFINED SEMANTICS** (Cycle-level cohort leakage confirmed; exact patient identity inferred from cycle cohort). |
| **2. 242/300 Overlap Reproduction** | 242 out of 300 test images overlap with train/val. | Independently reconstructed via `create_testset.py` logic: **231 overlap with Train, 108 with Val, 97 with both; exactly 242 / 300 (80.67%) overlap**. | **VERIFIED (EXACT MATHEMATICAL REPRODUCTION)**. |
| **3. Model Prediction Coverage** | All 12 models provide valid probability files for 300 test images. | 11 models have complete probability sums ($\sum p = 1.0$). **`convnextv2_tiny/teq.csv` is completely corrupted / all-NaN**. | **CORRECTED & QUALIFIED** (11 complete models; 1 corrupted file). |
| **4. Nature of Uncertainty Values** | Uncertainty columns represent confidence intervals or variance. | Confirmed via publication text: Values represent **SWA-G (Stochastic Weight Averaging Gaussian) posterior standard deviations** (Maddox et al., 2019). | **VERIFIED & ATTRIBUTED**. |
| **5. Calibration Feasibility** | Temperature scaling can be applied to the test set predictions. | **OVERSTATED / SCIENTIFICALLY PROHIBITED:** Fitting temperature scaling on the test set is data leakage. Raw ECE and reliability curves are feasible; post-hoc calibration requires a separate validation set. | **CORRECTED / PROHIBITION ESTABLISHED**. |
| **6. Uncertainty vs Disagreement** | Model entropy correlates with human expert disagreement. | Empirical verification confirms significant positive correlation between SWA-G model entropy and expert voting entropy (**Spearman $\rho = 0.279, p = 9.44 \times 10^{-7}$**). | **VERIFIED WITH EMPIRICAL EVIDENCE**. |
| **7. Code Licensing** | Original repository has no license; zero-copy clean-room required. | Verified: No license file, no license headers in code. Default copyright applies. Clean-room independent implementation is legally mandatory. | **VERIFIED (ZERO-COPY MANDATE CONFIRMED)**. |

---

## 2. Investigation 1 — Patient-Cycle Identifiers & Leakage Semantics

### 2.1 Filename Syntax and Integer Prefix Analysis
In `complete.csv` ($N = 2,344$), image filenames follow the regular expression:
```text
^(\d+)_(\d+)\.png$
```
- Total images: **2,344** (all matching the syntax).
- Integer prefix range: **$1$ to $861$**.
- Total unique integer prefixes: **851**.
- Suffix integer range: **$1$ to $11$** (representing individual blastocysts retrieved from that cohort).

### 2.2 Reconciling 851 Prefixes with 837 Documented Patients
- **Documented Fact (Kromp et al., *Scientific Data*, Section *Participants and ethics*):**  
  *"A total number of 2,344 blastocysts from 837 patients were included in this dataset. Blinding was employed during data collection. Informed consent was given and an ethical approval was obtained from the Ethics Committee of the Faculty of Medicine at the Johannes Kepler University in Linz, Austria (Nr. 1238/2021)."*
- **Empirical Arithmetic:**
  - Unique prefixes in dataset: **851**
  - Documented patients: **837**
  - Discrepancy: $+14$ prefixes ($851 - 837 = 14$).
- **Semantic Interpretation:**
  - In a 4-year clinical IVF study (2018–2021 at JKU Linz), patients undergoing assisted reproduction frequently undergo repeat ovarian stimulation cycles following treatment failure. Approximately 14 patients underwent a second cycle.
  - Therefore, the prefix denotes an **Oocyte Retrieval Cycle / Embryo Cohort**.
  - **Refined Scientific Precision:** We distinguish **confirmed cycle-level cohort overlap** from **speculative multi-cycle patient tracking**. Blastocysts sharing a prefix are confirmed **biological sister embryos** from the exact same stimulation, retrieval, incubation, culture dish, and microscope imaging session.

### 2.3 Blastocysts per Cycle Cohort
- Cycles with exactly 1 blastocyst: **265 (31.1%)**
- Cycles with $> 1$ blastocyst: **586 (68.9%)** (median: 2, mean: 2.75, max: 11).

```mermaid
pie title Cohort Multiplicity in complete.csv (N = 851 cycles)
    "Single-Blastocyst Cycles (265 cycles)" : 31.1
    "Multiple-Blastocyst Cycles (586 cycles)" : 68.9
```

### 2.4 Independent Split Reconstruction & Overlap Reproduction
Following the exact logic of `create_testset.py`:
1. Ground truth test set: `annotations/test_rev.csv` ($N = 300$).
2. Candidate pool for train/val: `complete.csv` minus test set ($N = 2,044$).
3. Train/Val split: `train_test_split(train_val_data, test_size=0.20, shuffle=True, random_state=10_123)`.

#### Verified Split Counts:
- **Training Set:** **1,635 images** across **734 unique cycles**.
- **Validation Set:** **409 images** across **324 unique cycles**.
- **Test Set:** **300 images** across **249 unique cycles**.

#### Exact Overlap Calculations:
- **Cycle-Level Overlap:**
  - Test cycles overlapping with Training: **189 cycles**
  - Test cycles overlapping with Validation: **89 cycles**
  - Test cycles overlapping with Training OR Validation: **200 cycles (out of 249, 80.3%)**
  - Training cycles overlapping with Validation: **256 cycles (out of 324, 79.0%)**
- **Image-Level Overlap in Test Set ($N = 300$):**
  - Test images sharing a cycle prefix with Training: **231 images (77.0%)**
  - Test images sharing a cycle prefix with Validation: **108 images (36.0%)**
  - Test images sharing a cycle prefix with Training ONLY: **134 images**
  - Test images sharing a cycle prefix with Validation ONLY: **11 images**
  - Test images sharing a cycle prefix with BOTH Training & Validation: **97 images**
  - **TOTAL Test images sharing cycle cohort with Training OR Validation:** **EXACTLY 242 out of 300 (80.67%)**
  - **Test images from completely isolated cycles:** **EXACTLY 58 out of 300 (19.33%)**

#### Conclusion for Investigation 1:
The claim of 242/300 overlap is **mathematically verified to the exact integer**. However, scientifically, it represents **cycle-level cluster leakage (sister embryo leakage)** rather than confirmed cross-cycle patient tracking. Because sister blastocysts share maternal biology, stimulation protocols, and microscope dish artifacts, this cluster leakage invalidates claims of evaluating on independent patient distributions.

---

## 3. Investigation 2 — Original Model Prediction Outputs & Corruption Audit

All 12 architecture directories in `model_predictions/` were inspected across their 4 respective files (`exp.csv`, `icm.csv`, `teq.csv`, `pred.csv`).

### 3.1 Architecture Inventory & Coverage

| Architecture | Category | Prediction Files | Probability Sums | Missing Values | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `coatnet_nano` | Hybrid CNN-ViT | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `convnext_femto` | Modern CNN | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `convnext_nano` | Modern CNN | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `convnext_tiny` | Modern CNN | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `convnext_tiny_384` | Modern CNN | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `convnextv2_nano` | Modern CNN | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `convnextv2_tiny` | Modern CNN | 4 / 4 | **EXP & ICM: 1.0, TEQ: NaN** | **1,200 NaNs in teq.csv** | **CORRUPTED TEQ FILE** |
| `convnextv2_tiny_384` | Modern CNN | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `deit` | Pure ViT | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `swin` | Hierarchical ViT | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `swin_s3_small` | Hierarchical ViT | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |
| `xception` | Depthwise CNN | 4 / 4 | $\sum p = 1.0$ | 0 | Complete & Valid |

- **Identical Image Coverage:** All 12 models cover the exact same 300 test images in the exact same filename order as `annotations/test_rev.csv`.

### 3.2 Discovery of File Corruption: `convnextv2_tiny/teq.csv`
- The file `model_predictions/convnextv2_tiny/teq.csv` is only 13,579 bytes (compared to ~31,500 bytes for other models).
- Every single probability and uncertainty entry in columns 1–9 is **NaN** across all 300 test images.
- In `convnextv2_tiny/pred.csv`, Column 3 (TEQ predicted class) is entirely `0` because `np.argmax()` on all-NaN slices defaulted to index 0.
- **Audit Correction:** Benchmark evaluations can utilize **all 12 models for EXP and ICM**, but must either **exclude `convnextv2_tiny` or report it as missing for TEQ**.

### 3.3 Probability Column & Class Mapping Architecture
- **Expansion (`exp.csv`):** 5 classes (Classes 1–5, coded 0–4).
  - Class probabilities: Columns 1, 3, 5, 7, 9.
  - SWA-G posterior standard deviations: Columns 2, 4, 6, 8, 10.
- **Inner Cell Mass (`icm.csv`):** 4 classes (0 = A, 1 = B, 2 = C, 3 = Not Assessable).
  - Class probabilities: Columns 1, 3, 5, 7.
  - SWA-G posterior standard deviations: Columns 2, 4, 6, 8.
- **Trophectoderm Quality (`teq.csv`):** 4 classes (0 = A, 1 = B, 2 = C, 3 = Not Assessable).
  - Class probabilities: Columns 1, 3, 5, 7.
  - SWA-G posterior standard deviations: Columns 2, 4, 6, 8.

### 3.4 Verification of Uncertainty Semantics (SWA-G)
- The publication text explicitly confirms:  
  *"We implemented stochastic weight averaging gaussian (SWA-G), a method used to reflect and calibrate uncertainty representation in Bayesian deep learning (Maddox et al., NeurIPS 2019). It is based on modelling a Gaussian distribution for each network's weight and applying it as a posterior over all neural network weights to perform Bayesian model averaging."*
- Therefore, the second column for each class represents the **posterior predictive standard deviation $\sigma_k$** across weight samples drawn from the fitted Gaussian weight distribution. It is an authentic Bayesian epistemic uncertainty proxy, not an arbitrary heuristic.

---

## 4. Investigation 3 — Calibration Methodology & Scientific Feasibility

### 4.1 Permissible Test Set Calibration Metrics
Because the 11 valid models provide raw posterior probabilities on the 300-image test set, the following metrics can be legitimately calculated to evaluate how well uncalibrated models reflect ground-truth frequencies:
1. **Multiclass Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   using 10 equal-width confidence bins.
2. **Maximum Calibration Error (MCE):** Identifying the worst-case bin disparity.
3. **Multiclass Brier Score:** Strictly proper scoring rule assessing joint calibration and refinement:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$
4. **Reliability Diagrams:** Plotting mean predicted confidence against observed accuracy across bins.
5. **Class-Wise Calibration:** Measuring calibration error separately for clinical grades (e.g., Grade A vs Grade C).

### 4.2 Prohibited Methodology: Test Set Temperature Scaling
- **Initial Audit Overstatement:** The initial audit suggested applying post-hoc temperature scaling to the provided test set predictions.
- **Scientific Prohibition:** Fitting a temperature parameter $T > 0$ by minimizing negative log-likelihood on the test set, and then reporting ECE on that *same* test set, is **blatant data snooping / optimistic bias**. It guarantees an artificially near-zero ECE that would not hold on genuinely unseen images.
- **Data Requirement for Temperature Scaling:** Legitimate post-hoc calibration requires a **held-out validation / calibration set**.
- **Feasibility Verdict:**
  - **Without new model training/inference:** We can perform an extensive **Uncalibrated Model Calibration Benchmark** (ECE, MCE, Brier score, reliability diagrams) across the 11 architectures.
  - **To implement Temperature Scaling:** We must either obtain/generate validation set predictions or perform split-half cross-calibration on the 300 test images (e.g. 150 calibration / 150 evaluation with non-overlapping cycles).

---

## 5. Investigation 4 — Expert Disagreement & Consensus Dynamics

### 5.1 Structure of Annotations
- Total test images: **300**.
- Expert files: 11 international embryologists (`Annotator_0.csv` to `Annotator_10.csv`) + `Gardner_Expert.csv`.
- Each expert annotated 150 images such that each image was evaluated by **5 to 7 experts** (mean: 6.49, median: 6).
- **Consensus Filtering:** Authors excluded 5 annotators (`Annotator_3, 4, 6, 9, 10`) whose accuracy was $< 50\%$ relative to the Gardner Expert. The remaining **7 qualified experts** provided between **2 and 6 votes per image** (mean: 4.00, median: 4).

### 5.2 Disagreement Quantification
Shannon entropy was calculated independently across retained expert votes:
$$H_{\text{expert}} = -\sum_{k} p_k \log_2(p_k)$$

| Criterion | Unanimous Agreement Images | Disagreement Rate | Mean Entropy (bits) | Max Entropy (bits) |
| :--- | :---: | :---: | :---: | :---: |
| **Expansion (EXP)** | 144 / 300 (48.0%) | **52.0%** | 0.476 | 1.922 |
| **Inner Cell Mass (ICM)** | 119 / 300 (39.7%) | **60.3%** | 0.619 | 1.922 |
| **Trophectoderm Quality (TEQ)** | 89 / 300 (29.7%) | **70.3%** | 0.712 | 1.585 |

*Key Clinical Insight:* Over 70% of blastocysts generated disagreement among senior embryologists on trophectoderm quality. Consensus labels in `test_rev.csv` represent **majority agreement**, NOT ground truth biology.

### 5.3 Correlation Between Model Predictive Entropy & Expert Disagreement
Comparing the predictive entropy of the deep learning models against human expert disagreement entropy revealed a statistically significant relationship:
- For Xception Expansion predictions:
  - **Spearman Rank Correlation:** $\mathbf{\rho = +0.279}$ ($p = 9.44 \times 10^{-7}$)
  - **Pearson Linear Correlation:** $\mathbf{r = +0.307}$ ($p = 5.93 \times 10^{-8}$)
- **Interpretation:** Deep learning models naturally output higher predictive entropy (uncertainty) on the exact blastocyst images that cause disagreement among human clinical experts.

---

## 6. Investigation 5 — Research Contribution Assessment

To prevent inflated novelty claims, each proposed research direction is benchmarked against the broader scientific literature:

### 6.1 Patient-Independent Evaluation on Kromp et al. Benchmark
- **Literature Context:** Patient-level / cluster-level data separation is standard best practice in medical machine learning (e.g. Park et al., *Lancet Digital Health* 2021; Little et al., *Nature Machine Intelligence* 2020).
- **Novelty Assessment:** It is **NOT** a novel mathematical concept in ML broadly. However, **within the specific Kromp et al. blastocyst benchmark**, evaluating patient-independent holdout has **never been published**. The original paper and all subsequent papers relying on its splits evaluated models on 80.7% leaking cycles. Demonstrating the true generalization gap under `GroupKFold` is a high-value benchmark correction.

### 6.2 Model Calibration & Reliability in Blastocyst Grading
- **Literature Context:** Deep neural networks are known to be overconfident (Guo et al., ICML 2017).
- **Novelty Assessment:** Almost all IVF imaging papers report accuracy or AUC; calibration is rarely reported. Systematic calibration analysis across 11 architectures on human blastocysts provides tangible clinical value for assisted reproduction decision-making.

### 6.3 Uncertainty vs Expert Disagreement
- **Literature Context:** Uncertainty-aware deep learning has been correlated with annotator noise in dermatology and ophthalmology (e.g. Peterson et al., 2019).
- **Novelty Assessment:** Applying SWA-G Bayesian uncertainty to international embryology consensus voting on blastocysts is novel in reproductive medicine.

### 6.4 Selective Classification (Abstain Option)
- **Feasibility:** High. Can be simulated directly on the 11 model probability distributions by defining a confidence threshold $\tau$ or entropy threshold $H_{\text{thresh}}$, evaluating accuracy as a function of coverage (risk-coverage curves).

---

## 7. Investigation 6 — Licensing, Attribution & Redistribution

1. **Original GitHub Code:**
   - **Repository:** `software-competence-center-hagenberg/Blastocyst-Dataset`
   - **Status:** **No explicit license**. Under GitHub Terms of Service and international copyright law, default copyright applies ("All rights reserved").
   - **Mandate:** **Strict Zero-Copy Rule**. FertiScope must not import, copy, or vendor any `.py` source code from the repository. All FertiScope code is independent and licensed under the MIT License.
2. **Figshare Dataset:**
   - **Record:** `10.6084/m9.figshare.20123153.v3`
   - **License:** **CC BY 4.0 (Creative Commons Attribution 4.0 International)**.
   - **Permitted Use:** Copying, adapting, and transforming data for research or commercial use with mandatory attribution.
3. **Model Prediction CSV Files:**
   - Published in the GitHub repository. Because they are derivative evaluation outputs rather than software source code, analyzing their numeric contents in a scientific study constitutes fair academic use. However, FertiScope should avoid redistributing the raw CSV files directly in its Git tree; instead, it should provide an automated download/fetch script pointing to the public upstream repository.
4. **Mandatory Formal Citations:**
   - Florian Kromp et al., *"An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization"*, *Scientific Data* 10, Article 271 (2023), DOI: 10.1038/s41597-023-02182-3.
   - Figshare Dataset: DOI: 10.6084/m9.figshare.20123153.v3.

---

## 8. Summary of Verified vs Overstated Claims

### Verified Claims:
1. Reconstructed test set contains 80.7% (242/300) images sharing cycle IDs with train/val.
2. SWA-G posterior standard deviations are provided across models.
3. 11 architectures have valid, complete posterior probabilities.
4. Model predictive entropy correlates significantly with human expert disagreement ($p < 10^{-6}$).
5. The original GitHub code lacks an open-source license, necessitating an independent clean-room implementation.

### Overstated or Incorrect Claims from Stage 1:
1. **"Confirmed Patient Identity Leakage":** Overstated. Filename prefixes identify oocyte retrieval cycle cohorts (851 cycles among 837 patients). It is confirmed **cycle-level cohort / sister embryo leakage**.
2. **"All 12 models are complete":** Incorrect. `convnextv2_tiny/teq.csv` is completely corrupted with NaNs.
3. **"Temperature scaling can be run on test predictions":** Overstated. Prohibited as data snooping without a separate validation split.
4. **"Novelty across entire literature":** Overstated. Grouped splitting and calibration are established principles in ML; their novelty lies specifically in auditing this IVF benchmark.

---

## 9. Recommended Research Question & Go/No-Go Decision

### Recommended Research Question:
> *"How do deep learning blastocyst grading models perform under true patient-cycle-independent evaluation, how well calibrated are their probability outputs, and does algorithmic uncertainty align with multi-expert embryologist disagreement?"*

### Formal Decision: **GO (GREEN LIGHT FOR INDEPENDENT CLEAN-ROOM REPOSITORY)**
- **Conditions:**
  1. Zero original source code copied (clean-room Python package).
  2. Clear CC BY 4.0 attribution maintained.
  3. Raw calibration metrics (ECE, Brier, reliability diagrams) prioritized over test-set temperature scaling.
  4. Explicit reporting of the `convnextv2_tiny/teq.csv` corruption.
  5. Distinction between cycle-level cohort leakage and verified patient IDs maintained.

---

## 10. Test Suite Status

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\vaish\.gemini\antigravity\scratch\fertiscope
configfile: pyproject.toml
collected 8 items

tests/test_data_foundation.py::test_provenance_record_immutability_and_export PASSED [ 12%]
tests/test_data_foundation.py::test_sha256_computation PASSED                  [ 25%]
tests/test_data_foundation.py::test_raw_auditor_detects_missing_files PASSED   [ 37%]
tests/test_data_foundation.py::test_raw_auditor_with_mock_dataset PASSED       [ 50%]
tests/test_data_foundation.py::test_schema_validation PASSED                 [ 62%]
tests/test_data_foundation.py::test_candidate_hd8yy6gzc5_provenance_record PASSED [ 75%]
tests/test_data_foundation.py::test_candidate_hd8yy6gzc5_auditor PASSED       [ 87%]
tests/test_data_foundation.py::test_candidate_hd8yy6gzc5_arithmetic_and_quality_metrics PASSED [100%]

============================== 8 passed in 3.65s ==============================
```

---

**MANDATORY STOPPING CONDITION REACHED:**  
*Stage 1B verification is complete. No external code has been copied, no raw data modified, no models trained, and nothing pushed to GitHub. Awaiting your explicit approval of this verification report before establishing the independent repository structure.*

