# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""pdftopdfa - Convert PDF files to PDF/A format."""

from importlib.metadata import PackageNotFoundError, version

from .converter import (
    ConversionResult,
    PDFUAReviewFinding,
    PDFUAStatus,
    ProfileValidationResult,
    PublicationPolicy,
    convert_directory,
    convert_files,
    convert_to_pdfa,
)
from .exceptions import (
    ConversionError,
    FontEmbeddingError,
    OCRError,
    PDFToPDFAError,
    UnsupportedPDFError,
    ValidationError,
    VeraPDFError,
)
from .ocr import OCRSession, recognize_image
from .table import (
    TableBoundingBox,
    TableCell,
    TableRecognitionResult,
    TableType,
    recognize_table,
)

try:
    __version__ = version("pdftopdfa")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "convert_to_pdfa",
    "convert_files",
    "convert_directory",
    "OCRSession",
    "recognize_image",
    "recognize_table",
    "ConversionResult",
    "PDFUAReviewFinding",
    "PDFUAStatus",
    "ProfileValidationResult",
    "PublicationPolicy",
    "TableType",
    "TableBoundingBox",
    "TableCell",
    "TableRecognitionResult",
    "PDFToPDFAError",
    "ConversionError",
    "ValidationError",
    "FontEmbeddingError",
    "UnsupportedPDFError",
    "OCRError",
    "VeraPDFError",
]
