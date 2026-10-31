"""Custom exception taxonomy for FertiScope research workflows."""


class FertiScopeError(Exception):
    """Base exception for all FertiScope errors."""


class DatasetNotFoundError(FertiScopeError):
    """Raised when expected raw or processed dataset files are missing."""


class IntegrityVerificationError(FertiScopeError):
    """Raised when SHA-256 checksums or raw file integrity checks fail."""


class SchemaValidationError(FertiScopeError):
    """Raised when tabular data fails pydantic or schema integrity constraints."""


class DataProvenanceError(FertiScopeError):
    """Raised when dataset licensing, source, or split provenance is unverifiable."""


class CorruptedFileError(FertiScopeError):
    """Raised when data or prediction files are corrupted, unreadable, or filled with invalid NaNs."""


class SplitLeakageError(FertiScopeError):
    """Raised when train/val/test splits violate cluster grouping or leak identifiers."""


class CalibrationError(FertiScopeError):
    """Raised when calibration calculation inputs are invalid or test-set fitting is attempted."""


class AnnotationFormatError(FertiScopeError):
    """Raised when clinical annotations violate expected grading standards or encodings."""


