"""Independent, clean-room dataset loader and metadata manifest for blastocyst image data.

Conforms to CC BY 4.0 data attribution (Florian Kromp et al., 2023).
No original code is reused or copied.
"""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field

from fertiscope.core.exceptions import (
    DatasetNotFoundError,
    IntegrityVerificationError,
    SchemaValidationError,
)
from fertiscope.core.logging import get_logger
from fertiscope.data.provenance import compute_sha256

logger = get_logger(__name__)

FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


class BlastocystMetadata(BaseModel):
    """Pydantic validated metadata record for an individual blastocyst observation."""

    image_name: str
    cycle_id: int = Field(ge=1, description="Extracted clinical retrieval cycle ID")
    embryo_index: int = Field(ge=1, description="Embryo index within the cycle cohort")
    cycle_str: str = Field(description="Original cycle string representation")
    file_path: Optional[str] = None
    sha256: Optional[str] = None


@dataclass(frozen=True)
class DatasetSummary:
    """Statistical summary of the blastocyst image cohort."""

    total_images: int
    total_unique_cycles: int
    min_embryos_per_cycle: int
    max_embryos_per_cycle: int
    mean_embryos_per_cycle: float
    cycles_with_multiple_embryos: int


class BlastocystDatasetLoader:
    """Clean-room loader for the Kromp et al. blastocyst image benchmark."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    def parse_filename(self, filename: str) -> Optional[Tuple[int, int, str]]:
        """Extract (cycle_id, embryo_index, cycle_str) from a standardized filename.

        Handles both zero-padded (e.g. 0001_01.png) and non-padded (e.g. 198_01.png) names.
        """
        match = FILENAME_PATTERN.match(filename)
        if not match:
            return None
        cycle_str = match.group(1)
        cycle_id = int(cycle_str)
        embryo_idx = int(match.group(2))
        return cycle_id, embryo_idx, cycle_str

    def build_manifest(self, compute_hashes: bool = False) -> pd.DataFrame:
        """Scan directory and generate a reproducible image and cycle manifest."""
        if not self.data_dir.exists():
            raise DatasetNotFoundError(f"Data directory does not exist: {self.data_dir}")

        image_files = sorted(
            [p for p in self.data_dir.iterdir() if p.is_file() and FILENAME_PATTERN.match(p.name)]
        )

        if not image_files:
            # Check for images subdirectory
            images_subdir = self.data_dir / "images"
            if images_subdir.exists():
                image_files = sorted(
                    [p for p in images_subdir.iterdir() if p.is_file() and FILENAME_PATTERN.match(p.name)]
                )

        if not image_files:
            raise DatasetNotFoundError(
                f"No blastocyst image files matching pattern found in: {self.data_dir}"
            )

        records: List[Dict] = []
        for file_path in image_files:
            parsed = self.parse_filename(file_path.name)
            if not parsed:
                continue
            cycle_id, embryo_idx, cycle_str = parsed
            sha = compute_sha256(file_path) if compute_hashes else None
            meta = BlastocystMetadata(
                image_name=file_path.name,
                cycle_id=cycle_id,
                embryo_index=embryo_idx,
                cycle_str=cycle_str,
                file_path=str(file_path.resolve()),
                sha256=sha,
            )
            records.append(meta.model_dump())

        df = pd.DataFrame(records)
        df.sort_values(by=["cycle_id", "embryo_index"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def summarize(self, manifest_df: pd.DataFrame) -> DatasetSummary:
        """Compute cohort-level clustering diagnostics."""
        if manifest_df.empty:
            raise SchemaValidationError("Cannot summarize empty manifest dataframe.")

        cycle_counts = manifest_df["cycle_id"].value_counts()
        total_images = len(manifest_df)
        total_unique_cycles = len(cycle_counts)
        min_emb = int(cycle_counts.min())
        max_emb = int(cycle_counts.max())
        mean_emb = float(cycle_counts.mean())
        multi_cycles = int((cycle_counts > 1).sum())

        return DatasetSummary(
            total_images=total_images,
            total_unique_cycles=total_unique_cycles,
            min_embryos_per_cycle=min_emb,
            max_embryos_per_cycle=max_emb,
            mean_embryos_per_cycle=round(mean_emb, 3),
            cycles_with_multiple_embryos=multi_cycles,
        )

    def load_complete_table(self, csv_path: Path | str) -> pd.DataFrame:
        """Load and validate the complete annotations table (e.g. complete.csv)."""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise DatasetNotFoundError(f"Annotation file not found: {csv_file}")

        df = pd.read_csv(csv_file)
        # Expected column names in benchmark: Image / image_name, EXP, ICM, TEQ
        col_map = {}
        for c in df.columns:
            if c.lower() in ("image", "image_name", "img", "filename"):
                col_map[c] = "image_name"
            elif c.upper() == "EXP":
                col_map[c] = "EXP"
            elif c.upper() == "ICM":
                col_map[c] = "ICM"
            elif c.upper() in ("TEQ", "TE"):
                col_map[c] = "TEQ"
        df.rename(columns=col_map, inplace=True)

        required_cols = {"image_name", "EXP", "ICM", "TEQ"}
        missing = required_cols - set(df.columns)
        if missing:
            raise SchemaValidationError(f"Missing required columns in complete table: {missing}")

        # Extract cycle_id for each row
        cycle_info = df["image_name"].apply(self.parse_filename)
        if cycle_info.isnull().any():
            invalid_names = df[cycle_info.isnull()]["image_name"].tolist()
            raise SchemaValidationError(f"Invalid image filenames in table: {invalid_names[:5]}")

        df["cycle_id"] = [info[0] for info in cycle_info]
        df["embryo_index"] = [info[1] for info in cycle_info]
        df["cycle_str"] = [info[2] for info in cycle_info]

        # Validate domains
        # EXP should be integers 1 to 6
        valid_exp = df["EXP"].dropna().isin([1, 2, 3, 4, 5, 6, "1", "2", "3", "4", "5", "6"])
        if not valid_exp.all():
            invalid = df[~valid_exp]["EXP"].unique()
            raise SchemaValidationError(f"EXP column contains values outside [1..6]: {invalid}")

        # ICM should be A, B, C or 0, 1, 2
        valid_icm = df["ICM"].dropna().astype(str).str.upper().isin(["A", "B", "C", "0", "1", "2"])
        if not valid_icm.all():
            invalid = df[~valid_icm]["ICM"].unique()
            raise SchemaValidationError(f"ICM column contains values outside [A, B, C]: {invalid}")

        # TEQ should be A, B, C or 0, 1, 2
        valid_teq = df["TEQ"].dropna().astype(str).str.upper().isin(["A", "B", "C", "0", "1", "2"])
        if not valid_teq.all():
            invalid = df[~valid_teq]["TEQ"].unique()
            raise SchemaValidationError(f"TEQ column contains values outside [A, B, C]: {invalid}")

        return df

