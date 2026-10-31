"""Independent, clean-room annotation parser for Gardner blastocyst grading benchmark.

Conforms to Gardner & Schoolcraft criteria:
- Expansion (EXP): grades 1, 2, 3, 4, 5, 6
- Inner Cell Mass (ICM): grades A, B, C (mapped to 0, 1, 2)
- Trophectoderm Quality (TEQ): grades A, B, C (mapped to 0, 1, 2)

No original code is copied or reused.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fertiscope.core.exceptions import AnnotationFormatError, DatasetNotFoundError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)

# Standard mappings
EXP_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
EXP_INV_MAP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
GRADE_MAP = {"A": 0, "B": 1, "C": 2, 0: 0, 1: 1, 2: 2}
GRADE_INV_MAP = {0: "A", 1: "B", 2: "C"}


class ExpertVoteDistribution(BaseModel):
    """Distribution of human expert votes for a single image and criterion."""

    image_name: str
    criterion: str  # "EXP", "ICM", or "TEQ"
    num_votes: int = Field(ge=0)
    vote_counts: Dict[str, int]
    vote_proportions: Dict[str, float]
    consensus_class: Optional[str] = None
    is_unanimous: bool = False
    is_tie: bool = False
    entropy: float = Field(ge=0.0)


class AnnotationParser:
    """Clean-room parser for expert annotations and ground truth evaluation sets."""

    @staticmethod
    def standardize_grade(val: Union[str, int, float, None], criterion: str) -> Optional[int]:
        """Convert raw annotation string/number to standardized 0-indexed integer class.

        Returns None for missing / not defined ('ND', NaN, '', etc.).
        """
        if pd.isna(val) or val is None:
            return None
        val_str = str(val).strip().upper()
        if val_str in ("ND", "NA", "NAN", "", "NONE", "-1"):
            return None

        crit = criterion.upper()
        if crit == "EXP":
            try:
                exp_int = int(float(val_str))
                if exp_int == -1:
                    return None
                if 1 <= exp_int <= 6:
                    return EXP_MAP[exp_int]
                elif 0 <= exp_int <= 5:
                    return exp_int
            except ValueError:
                pass
            raise AnnotationFormatError(f"Invalid EXP grade: {val}")

        elif crit in ("ICM", "TEQ", "TE"):
            if val_str in GRADE_MAP:
                return GRADE_MAP[val_str]
            try:
                idx = int(float(val_str))
                if idx in (-1, 3):
                    # 3 or -1 represents unassessable / not defined in Gardner benchmark
                    return None
                if 0 <= idx <= 2:
                    return idx
            except ValueError:
                pass
            raise AnnotationFormatError(f"Invalid {crit} grade: {val}")

        raise AnnotationFormatError(f"Unknown grading criterion: {criterion}")

    @staticmethod
    def parse_test_ground_truth(csv_path: Path | str) -> pd.DataFrame:
        """Parse ground truth evaluation set (e.g. test_rev.csv). Handles headered and headerless CSVs."""
        path = Path(csv_path)
        if not path.exists():
            raise DatasetNotFoundError(f"Test annotations file not found: {path}")

        # Check first line to detect if header is present
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        if ".png" in first_line.lower() or ".jpg" in first_line.lower():
            # Headerless file with image filename in first column
            df = pd.read_csv(path, header=None, names=["image_name", "EXP", "ICM", "TEQ"])
        else:
            df = pd.read_csv(path)
            col_map = {}
            for c in df.columns:
                cl = c.strip().lower()
                if cl in ("image", "image_name", "img"):
                    col_map[c] = "image_name"
                elif cl == "exp":
                    col_map[c] = "EXP"
                elif cl == "icm":
                    col_map[c] = "ICM"
                elif cl in ("teq", "te"):
                    col_map[c] = "TEQ"
            df.rename(columns=col_map, inplace=True)

        for crit in ("EXP", "ICM", "TEQ"):
            if crit not in df.columns:
                raise AnnotationFormatError(f"Missing ground truth column {crit} in {path}")
            df[f"{crit}_raw"] = df[crit]
            df[f"{crit}_class"] = df[crit].apply(
                lambda v, c=crit: AnnotationParser.standardize_grade(v, c)
            )

        return df

    @staticmethod
    def parse_expert_matrix(
        csv_path: Path | str,
        valid_annotators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Parse multi-annotator voting matrix (e.g. Gardner_Expert.csv)."""
        path = Path(csv_path)
        if not path.exists():
            raise DatasetNotFoundError(f"Expert matrix file not found: {path}")

        df = pd.read_csv(path)
        # Standardize Image column
        if "Image" in df.columns:
            df.rename(columns={"Image": "image_name"}, inplace=True)
        elif "image_name" not in df.columns:
            raise AnnotationFormatError(f"No image column found in {path}")

        if valid_annotators is not None:
            # Filter columns to only image_name and valid annotator columns
            keep_cols = ["image_name"]
            for col in df.columns:
                if col == "image_name":
                    continue
                # E.g. col is 'EXP_Annotator_0' or 'Annotator_0_EXP'
                if any(annot in col for annot in valid_annotators):
                    keep_cols.append(col)
            df = df[keep_cols]

        return df

    @classmethod
    def compute_vote_distribution(
        cls,
        image_name: str,
        votes: List[Union[str, int, float, None]],
        criterion: str,
    ) -> ExpertVoteDistribution:
        """Compute the discrete voting distribution, consensus, and entropy for an image."""
        standardized_votes = [
            cls.standardize_grade(v, criterion) for v in votes if cls.standardize_grade(v, criterion) is not None
        ]
        num_votes = len(standardized_votes)

        if criterion.upper() == "EXP":
            all_classes = [0, 1, 2, 3, 4, 5]
            str_labels = ["1", "2", "3", "4", "5", "6"]
            inv_map = EXP_INV_MAP
        else:
            all_classes = [0, 1, 2]
            str_labels = ["A", "B", "C"]
            inv_map = GRADE_INV_MAP

        counts = {str(inv_map[c]): 0 for c in all_classes}
        for v in standardized_votes:
            counts[str(inv_map[v])] += 1

        if num_votes > 0:
            props = {k: v / num_votes for k, v in counts.items()}
            # Shannon entropy in bits (base 2)
            prob_vec = np.array([p for p in props.values() if p > 0.0])
            entropy = -float(np.sum(prob_vec * np.log2(prob_vec)))
        else:
            props = {k: 0.0 for k in counts}
            entropy = 0.0

        # Majority consensus and tie detection
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        is_tie = False
        consensus_class = None
        is_unanimous = False

        if num_votes > 0:
            top_label, top_count = sorted_counts[0]
            if len(sorted_counts) > 1 and sorted_counts[1][1] == top_count:
                is_tie = True
                consensus_class = top_label  # Tie flag set
            else:
                consensus_class = top_label
            if top_count == num_votes:
                is_unanimous = True

        return ExpertVoteDistribution(
            image_name=image_name,
            criterion=criterion.upper(),
            num_votes=num_votes,
            vote_counts=counts,
            vote_proportions=props,
            consensus_class=consensus_class,
            is_unanimous=is_unanimous,
            is_tie=is_tie,
            entropy=round(entropy, 4),
        )
