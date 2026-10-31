# FertiScope

> **A Reproducible Research Platform for Clinical and Machine Learning Investigation of Reproductive-Health and IVF Outcomes**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture: Clean Room](https://img.shields.io/badge/architecture-clean--room-emerald.svg)](#technical-architecture)
[![Test Suite: 35 Passing](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)](#running-tests)

---

## 1. Project Motivation & Research Context

Assisted Reproductive Technologies (ART), specifically In Vitro Fertilization and Embryo Transfer (IVF-ET), involve intricate interactions between maternal physiology, paternal gamete quality, embryological developmental kinetics, and clinical protocol selections. 

Despite substantial advances in reproductive endocrinology, cycle outcomes exhibit notable variance, and clinician decision support remains constrained by:
1. **Opaque and Non-reproducible Pipelines:** Many published machine learning models in reproductive medicine lack reproducible preprocessing, fail to guard against cluster/cohort leakage, or omit cryptographic checksums of raw datasets.
2. **Cluster & Cycle Leakage:** Sister embryos from the same retrieval cycle are frequently partitioned across training and testing sets, resulting in optimistic generalization bias from shared morphology.
3. **Black-box Predictions Without Clinical Interpretability:** Complex models often report discrimination metrics (AUROC, accuracy) without assessing calibration (ECE, Brier score), expert disagreement dynamics, or uncertainty reliability.

**FertiScope** is an independent, open-source scientific and engineering research initiative designed to build a transparent, modular, and fully reproducible research platform. It evaluates how statistical data engineering, cycle-aware validation methodology, and calibration metrics enhance the rigorous analysis of reproductive health datasets.

---

## 2. Core Research Questions

FertiScope investigates the following foundational questions:
1. **Cluster Leakage Dynamics:** How does sister-embryo cycle overlap (where sister blastocysts appear across train and test sets) affect deep learning generalization performance compared to cycle-isolated evaluation cohorts?
2. **Calibration vs. Discrimination:** In clinical decision support, calibrated probabilities are critical for informed patient consent. How well calibrated are vision architectures across Gardner criteria (EXP, ICM, TEQ) using multiclass ECE and Brier scores?
3. **Human–AI Uncertainty Alignment:** Does algorithmic predictive entropy correlate monotonically with human expert voting disagreement (Shannon entropy across international embryologists)?
4. **Clean-Room Engineering:** Can clinical reproductive health benchmarks be implemented with complete legal, mathematical, and architectural independence?

---

## 3. Dataset Provenance & Governance

FertiScope enforces rigorous data provenance. All raw patient records are isolated, cryptographically audited, and excluded from version control to prevent privacy compromises.

### Benchmarked Datasets
1. **Multi-Annotator Blastocyst Grading Benchmark:**
   - **Source:** Software Competence Center Hagenberg & Kepler University Hospital Linz
   - **Scope:** 2,344 microscopic blastocyst images, Gardner criteria (Expansion, ICM, TEQ), 11 international expert embryologists, and probability outputs across 12 deep learning architectures.
   - **Permanent Identifier:** Figshare DOI `10.6084/m9.figshare.20123153.v3` / Scientific Data DOI `10.1038/s41597-023-02182-3`
   - **Licensing:** CC BY 4.0
   - **Implementation Governance:** Clean-room independent implementation. Zero source code reused from unlicensed repositories.

2. **Multimodal IVF Clinical Tabular Cohort (Audit Candidate):**
   - **Scope:** 5,000 IVF cycles with clinical and embryological variables.
   - **Permanent Identifier:** Mendeley Data DOI `10.17632/hd8yy6gzc5.1` (CC BY 4.0)

> [!IMPORTANT]
> **Data Privacy & Compliance:** Raw patient data files are never committed to version control. The `.gitignore` configuration explicitly excludes all raw, interim, and processed patient records (`.xlsx`, `.csv`, `.parquet`, `.zip`). Checksums and schema definitions are tracked independently.

---

## 4. Technical Architecture

FertiScope is organized as a decoupled, multi-layer Python package:

```text
fertiscope/
├── docs/                     # Scientific audits, claims verifications, and validation reports
├── src/
│   └── fertiscope/
│       ├── core/             # Structured logging, domain exception taxonomy
│       ├── data/             # Ingestion, provenance, Pydantic schemas, cycle-aware splitting
│       ├── metrics/          # Cohen's & Fleiss' Kappa, concordance, classification metrics
│       ├── calibration/      # Multiclass ECE, MCE, Brier score, reliability diagrams
│       └── analysis/         # Human-AI disagreement, sister-embryo cycle leakage
└── tests/                    # 35 unit, integration, and mathematical validation tests
```

### Key Engineering Decisions:
- **Clean-Room Design:** Completely independent implementation from mathematical first principles.
- **Cycle-Aware Splitting:** Whole oocyte retrieval cycles are allocated as units, guaranteeing 0% cycle leakage across train, val, and test.
- **Evaluation Independence:** Brier Skill Score baselines are decoupled from evaluation set labels. Post-hoc temperature scaling on test sets is strictly prohibited by code safeguards.
- **Pydantic Validation:** Ingested clinical features and manifests are verified against biological bounds and strict data integrity schemas.

---

## 5. Development Phases

The project follows a disciplined five-phase lifecycle:

| Phase | Focus | Status |
| :--- | :--- | :--- |
| **Phase 1** | **Research & Data Foundation** (Provenance, schema, integrity audit) | **Complete** |
| **Phase 2** | **Clinical Data Analytics** (Cycle-aware splitting, calibration, human-AI disagreement) | **Complete & Validated** |
| **Phase 3** | **Predictive Machine Learning** (Baselines, penalized models, tree ensembles) | Awaiting approval |
| **Phase 4** | **Research Validation & Explainability** (Subgroup audits, feature attributions) | Awaiting approval |
| **Phase 5** | **Research Platform & Publication** (Interactive dashboard, artifact reports) | Awaiting approval |

---

## 6. Installation & Reproducibility

### 6.1 Prerequisites
- Python >= 3.11
- Windows, Linux, or macOS

### 6.2 Setup
```bash
# Clone the repository
git clone https://github.com/VaishnavSreekumar/FertiScope.git
cd FertiScope

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install core dependencies in editable mode
pip install -e .
```

### 6.3 Running Tests
```bash
pytest -v
```
All 35 tests pass deterministically in ~6 seconds.

---

## 7. Known Limitations & Scientific Caution

1. **Cycle Identifier Semantics:** Filename prefixes denote clinical oocyte retrieval cycles, not confirmed unique patient master records. While sister embryos within a retrieval are isolated, repeat retrieval cycles from the same patient (~14 cycles) cannot be separated without external clinical linkage.
2. **Subjectivity of Expert Consensus:** Human consensus is not biological ground truth. TEQ shows unanimous agreement on only 29.7% of images; consensus represents inter-rater agreement on visual appearance, not embryo viability or live-birth potential.
3. **Observational Nature:** All predictive associations identified reflect observational correlations and **must not be interpreted as causal relationships or clinical advice**.
4. **No Clinical Advice:** FertiScope is a technical research tool for methodological exploration and is **not certified as a medical device or clinical diagnostic tool**.

---

## 8. License & Attribution

- **Software Code:** MIT License — see the [LICENSE](LICENSE) file.
- **Dataset Attribution:** Florian Kromp et al., *Scientific Data* (2023), DOI: `10.1038/s41597-023-02182-3` under CC BY 4.0.
