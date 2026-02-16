# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Custom exceptions for pdftopdfa."""


class PDFToPDFAError(Exception):
    """Base exception for all pdftopdfa errors."""


class ConversionError(PDFToPDFAError):
    """Error during conversion."""


class ValidationError(PDFToPDFAError):
    """PDF/A validation failed."""


class FontEmbeddingError(PDFToPDFAError):
    """Font could not be embedded."""


class UnsupportedPDFError(PDFToPDFAError):
    """PDF format is not supported."""


class OCRError(PDFToPDFAError):
    """Error during OCR processing."""


class VeraPDFError(PDFToPDFAError):
    """Error during veraPDF validation."""
