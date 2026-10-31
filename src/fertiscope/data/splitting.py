"""Cycle-aware dataset splitting, cluster-leakage diagnostics, and split audit.

Provides clean-room implementations for:
1. Group-aware splitting on clinical cycle identifiers (preventing sister-embryo leakage).
2. Diagnostic reporting of class distributions, cycle counts, and record conservation.
3. Reproduction of naive/original splits demonstrating sister-embryo cluster leakage.
4. Rigorous split overlap auditing and manifest generation.

Methodological note:
Filename prefixes denote clinical oocyte retrieval cycles, not verified unique patient identities.
Repeat cycles from the same patient may exist.

No original code is copied or reused.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fertiscope.core.exceptions import SchemaValidationError, SplitLeakageError
from fertiscope.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitOverlapReport:
    """Quantitative summary of group/cycle overlap between dataset partitions."""

    total_images_train: int
    total_images_val: int
    total_images_test: int
    unique_cycles_train: int
    unique_cycles_val: int
    unique_cycles_test: int
    test_cycles_in_train: int
    test_cycles_in_val: int
    test_cycles_in_either: int
    test_images_in_train_cycles: int
    test_images_in_val_cycles: int
    test_images_in_either_cycles: int
    cycle_isolated_test_images: int
    cycle_leakage_rate: float
    has_leakage: bool

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitPartitionDiagnostics:
    """Diagnostics for an individual split partition."""

    split_name: str
    num_images: int
    num_cycles: int
    image_proportion: float
    cycle_proportion: float
    class_counts: Dict[str, Dict[str, int]]
    missing_classes: Dict[str, List[str]]


@dataclass(frozen=True)
class CycleGroupSplitDiagnostics:
    """Comprehensive diagnostics for cluster-independent splitting."""

    total_images_input: int
    total_cycles_input: int
    total_images_allocated: int
    records_conserved: bool
    partitions: Dict[str, SplitPartitionDiagnostics]
    overlap_report: SplitOverlapReport
    warnings: List[str]
    methodological_note: str

    def to_dict(self) -> Dict:
        return asdict(self)


class CycleAwareSplitter:
    """Group-aware dataset splitter ensuring zero cycle overlap across partitions."""

    @staticmethod
    def audit_overlap(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_col: str = "cycle_id",
        strict: bool = False,
    ) -> SplitOverlapReport:
        """Audit whether any cycle IDs leak between train, val, and test splits."""
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if group_col not in df.columns:
                raise SchemaValidationError(f"Missing group column '{group_col}' in {name} dataframe.")

        train_cycles: Set = set(train_df[group_col].unique())
        val_cycles: Set = set(val_df[group_col].unique())
        test_cycles: Set = set(test_df[group_col].unique())

        in_train = test_cycles.intersection(train_cycles)
        in_val = test_cycles.intersection(val_cycles)
        in_either = in_train.union(in_val)

        test_in_train_imgs = int(test_df[test_df[group_col].isin(in_train)].shape[0])
        test_in_val_imgs = int(test_df[test_df[group_col].isin(in_val)].shape[0])
        test_in_either_imgs = int(test_df[test_df[group_col].isin(in_either)].shape[0])
        isolated_imgs = len(test_df) - test_in_either_imgs
        leakage_rate = test_in_either_imgs / len(test_df) if len(test_df) > 0 else 0.0

        has_leakage = len(in_either) > 0

        report = SplitOverlapReport(
            total_images_train=len(train_df),
            total_images_val=len(val_df),
            total_images_test=len(test_df),
            unique_cycles_train=len(train_cycles),
            unique_cycles_val=len(val_cycles),
            unique_cycles_test=len(test_cycles),
            test_cycles_in_train=len(in_train),
            test_cycles_in_val=len(in_val),
            test_cycles_in_either=len(in_either),
            test_images_in_train_cycles=test_in_train_imgs,
            test_images_in_val_cycles=test_in_val_imgs,
            test_images_in_either_cycles=test_in_either_imgs,
            cycle_isolated_test_images=isolated_imgs,
            cycle_leakage_rate=round(leakage_rate, 4),
            has_leakage=has_leakage,
        )

        if has_leakage and strict:
            raise SplitLeakageError(
                f"Split leakage detected: {test_in_either_imgs}/{len(test_df)} test images ({leakage_rate:.2%}) "
                f"share cycles with train/val."
            )

        return report

    @classmethod
    def generate_split_diagnostics(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        total_input_records: int,
        group_col: str = "cycle_id",
        target_cols: Optional[List[str]] = None,
    ) -> CycleGroupSplitDiagnostics:
        """Generate structured diagnostics inspecting class representation and record conservation."""
        overlap = cls.audit_overlap(train_df, val_df, test_df, group_col=group_col, strict=False)
        total_allocated = len(train_df) + len(val_df) + len(test_df)
        conserved = total_allocated == total_input_records

        total_input_cycles = (
            set(train_df[group_col]) | set(val_df[group_col]) | set(test_df[group_col])
        )
        total_c_count = len(total_input_cycles)

        if target_cols is None:
            target_cols = [c for c in ["EXP", "ICM", "TEQ"] if c in train_df.columns]

        partitions: Dict[str, SplitPartitionDiagnostics] = {}
        warnings: List[str] = []

        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            n_imgs = len(split_df)
            n_cycles = split_df[group_col].nunique()
            img_prop = n_imgs / total_allocated if total_allocated > 0 else 0.0
            cyc_prop = n_cycles / total_c_count if total_c_count > 0 else 0.0

            cls_counts: Dict[str, Dict[str, int]] = {}
            missing_cls: Dict[str, List[str]] = {}

            for col in target_cols:
                if col in split_df.columns:
                    counts = split_df[col].dropna().astype(str).value_counts().to_dict()
                    cls_counts[col] = {k: int(v) for k, v in counts.items()}

                    # Check if any class present in train is missing in val or test
                    all_classes = train_df[col].dropna().astype(str).unique()
                    miss = [c for c in all_classes if c not in counts]
                    if miss:
                        missing_cls[col] = miss
                        warnings.append(
                            f"Partition '{name}' is missing classes {miss} for column '{col}'."
                        )

            partitions[name] = SplitPartitionDiagnostics(
                split_name=name,
                num_images=n_imgs,
                num_cycles=n_cycles,
                image_proportion=round(img_prop, 4),
                cycle_proportion=round(cyc_prop, 4),
                class_counts=cls_counts,
                missing_classes=missing_cls,
            )

        if not conserved:
            warnings.append(
                f"Record conservation violated: {total_input_records} input vs {total_allocated} allocated."
            )

        note = (
            "Cycle identifiers denote oocyte retrieval cohorts, not confirmed unique patient identities. "
            "Sister embryos within a single retrieval are isolated, but repeat retrieval cycles from the "
            "same patient may exist."
        )

        return CycleGroupSplitDiagnostics(
            total_images_input=total_input_records,
            total_cycles_input=total_c_count,
            total_images_allocated=total_allocated,
            records_conserved=conserved,
            partitions=partitions,
            overlap_report=overlap,
            warnings=warnings,
            methodological_note=note,
        )

    @classmethod
    def create_cycle_grouped_split(
        cls,
        df: pd.DataFrame,
        group_col: str = "cycle_id",
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        return_diagnostics: bool = False,
        stratify_cols: Optional[List[str]] = None,
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CycleGroupSplitDiagnostics],
    ]:
        """Create a cluster-independent split where entire cycles are allocated as units.

        Ensures 0% cycle leakage across train, val, and test.
        """
        if group_col not in df.columns:
            raise SchemaValidationError(f"Group column '{group_col}' missing from dataframe.")

        if test_ratio <= 0.0 or val_ratio <= 0.0 or (test_ratio + val_ratio >= 1.0):
            raise SchemaValidationError(
                f"Invalid split ratios: test_ratio={test_ratio}, val_ratio={val_ratio}. Sum must be < 1.0."
            )

        unique_cycles = np.array(sorted(df[group_col].unique()))
        rng = np.random.default_rng(random_seed)
        shuffled_cycles = rng.permutation(unique_cycles)

        total_cycles = len(shuffled_cycles)
        n_test = int(np.round(total_cycles * test_ratio))
        n_val = int(np.round(total_cycles * val_ratio))

        test_cycle_ids = set(shuffled_cycles[:n_test])
        val_cycle_ids = set(shuffled_cycles[n_test : n_test + n_val])
        train_cycle_ids = set(shuffled_cycles[n_test + n_val :])

        train_df = df[df[group_col].isin(train_cycle_ids)].copy().reset_index(drop=True)
        val_df = df[df[group_col].isin(val_cycle_ids)].copy().reset_index(drop=True)
        test_df = df[df[group_col].isin(test_cycle_ids)].copy().reset_index(drop=True)

        # Audit split to mathematically confirm zero leakage
        cls.audit_overlap(train_df, val_df, test_df, group_col=group_col, strict=True)

        if return_diagnostics:
            diag = cls.generate_split_diagnostics(
                train_df,
                val_df,
                test_df,
                total_input_records=len(df),
                group_col=group_col,
                target_cols=stratify_cols,
            )
            return train_df, val_df, test_df, diag

        return train_df, val_df, test_df

    @classmethod
    def reconstruct_naive_benchmark_split(
        cls,
        df: pd.DataFrame,
        random_seed: int = 10_123,
        stratify_col: str = "EXP",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Reconstruct the published image-level naive split with known sister-embryo leakage.

        Reproduces the exact 242/300 cycle leakage documented in Kromp et al.
        """
        if "test_rev" in df.columns or "is_test" in df.columns:
            flag_col = "test_rev" if "test_rev" in df.columns else "is_test"
            test_df = df[df[flag_col] == 1].copy()
            pool_df = df[df[flag_col] == 0].copy()
        else:
            strat = df[stratify_col] if stratify_col in df.columns else None
            pool_df, test_df = train_test_split(
                df, test_size=300, random_state=random_seed, stratify=strat
            )

        strat_pool = pool_df[stratify_col] if stratify_col in pool_df.columns else None
        train_df, val_df = train_test_split(
            pool_df, test_size=0.20, random_state=random_seed, stratify=strat_pool
        )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
