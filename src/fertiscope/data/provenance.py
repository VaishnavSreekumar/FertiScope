"""Dataset provenance and integrity management for FertiScope."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetProvenanceRecord:
    """Immutable metadata record tracking dataset provenance and integrity."""

    dataset_id: str
    doi: str
    source_url: str
    version: str
    license: str
    reported_cohort_size: int
    reported_feature_count: int
    collection_period: str
    source_institution: str
    download_date: Optional[str]
    verification_status: str  # e.g., "UNVERIFIED_SOURCE_BLOCKED", "VERIFIED", "PENDING_LOCAL_FILE"
    notes: str
    expected_files: List[str]
    file_checksums_sha256: Dict[str, str]

    def to_dict(self) -> Dict:
        """Convert record to dictionary."""
        return asdict(self)

    def save_json(self, destination: Path) -> None:
        """Serialize provenance record to JSON."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 cryptographic hash of a local file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Target file does not exist: {file_path}")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


PRIMARY_CANDIDATE_PROVENANCE = DatasetProvenanceRecord(
    dataset_id="mendeley_x5s3ky9cpv_1",
    doi="10.17632/x5s3ky9cpv.1",
    source_url="https://data.mendeley.com/datasets/x5s3ky9cpv/1",
    version="1.0",
    license="CC BY 4.0 (reported)",
    reported_cohort_size=2542,
    reported_feature_count=36,
    collection_period="January 2016 – August 2020",
    source_institution="Reproductive Medical Center, Gansu Provincial Maternity and Child-care Hospital",
    download_date=None,
    verification_status="UNVERIFIED_SOURCE_BLOCKED_HTTP_451",
    notes=(
        "Primary candidate dataset was queried on 2026-11-01. Mendeley Data API returned HTTP 451 "
        "(Unavailable For Legal Reasons) with blocked_reason: 'As per author's request'. "
        "The public web portal and REST endpoints suppress file downloads. Independent SHA-256 verification "
        "and data dictionary validation require local provision of 'Embryo dataset_2542.xlsx' and 'Data set splitting.zip'."
    ),
    expected_files=["Embryo dataset_2542.xlsx", "Data set splitting.zip"],
    file_checksums_sha256={},
)


CANDIDATE_HD8YY6GZC5_PROVENANCE = DatasetProvenanceRecord(
    dataset_id="mendeley_hd8yy6gzc5_1",
    doi="10.17632/hd8yy6gzc5.1",
    source_url="https://data.mendeley.com/datasets/hd8yy6gzc5/1",
    version="1.0",
    license="CC BY 4.0",
    reported_cohort_size=5000,
    reported_feature_count=20,
    collection_period="January 2019 – June 2025 (derived from Egg retrieval date span)",
    source_institution="Erbil Polytechnic University, Erbil, Kurdistan Region, Iraq",
    download_date="2026-11-01",
    verification_status="ACQUIRED_AND_AUDITED_CANDIDATE",
    notes=(
        "Candidate dataset downloaded and verified locally. Contains 5,000 cycles with 20 claimed input variables "
        "and 1 target variable. Independent SHA-256 integrity verified against original ZIP. Audit identified critical "
        "target leakage in Column 20 (β-HCG), non-standardized sperm parameter scales, embedded Excel formula rows, "
        "and lack of unique patient identifiers."
    ),
    expected_files=["Dataset Clean.xlsx", "Normalized CSV.csv"],
    file_checksums_sha256={
        "Comprehensive Dataset Profile of In Vitro Fertiliz.zip": "358cfe945948fdde8714042933f8b69ecb157185661104298c8819892e40f3ed",
        "Dataset Clean.xlsx": "8914bf7a331c01bc1c66eeb81bf405b3476da3d80674536385a166f5d52dbf32",
        "Normalized CSV.csv": "fdda6c56c8ac3f66e8981685fa8732b9aaa57818f39e53325463a5e6c0acabc0",
    },
)


FIGSHARE_BLASTOCYST_PROVENANCE = DatasetProvenanceRecord(
    dataset_id="figshare_20123153_v3",
    doi="10.6084/m9.figshare.20123153.v3",
    source_url="https://doi.org/10.6084/m9.figshare.20123153.v3",
    version="3.0",
    license="CC BY 4.0",
    reported_cohort_size=2344,
    reported_feature_count=3,
    collection_period="Retrospective IVF clinical cohort (published 2023)",
    source_institution="Software Competence Center Hagenberg & Kepler University Hospital Linz",
    download_date="2026-11-01",
    verification_status="INDEPENDENT_CLEAN_ROOM_VERIFIED",
    notes=(
        "Standard multi-annotator blastocyst grading benchmark under CC BY 4.0. Contains 2,344 microscopic images "
        "graded for Expansion (EXP 1-6), Inner Cell Mass (ICM A-C), and Trophectoderm Quality (TEQ A-C). "
        "Test cohort comprises 300 images evaluated by up to 11 international embryologists and 12 deep learning models. "
        "Filename prefix corresponds to clinical cycle ID (oocyte retrieval cycle), showing 242/300 cycle leakage in original split."
    ),
    expected_files=[
        "complete.csv",
        "annotations/test_rev.csv",
        "annotations/Gardner_Expert.csv",
    ],
    file_checksums_sha256={},
)



