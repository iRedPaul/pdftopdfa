# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for PDF/A conversion."""

import logging
import sys
from collections.abc import Generator
from typing import Any

from pikepdf import Dictionary, Pdf

from .exceptions import ConversionError

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Supported target levels for PDF/A conversion (no "a" levels — those require
# Tagged PDF structure which this tool does not produce)
SUPPORTED_LEVELS = frozenset({"2b", "2u", "3b", "3u"})

# Required PDF versions for PDF/A levels
REQUIRED_PDF_VERSIONS = {
    "2b": "1.7",
    "2u": "1.7",
    "3b": "1.7",
    "3u": "1.7",
}


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configures logging for pdftopdfa.

    Args:
        verbose: If True, DEBUG level is used.
        quiet: If True, only ERROR and higher are output.
            Takes precedence over verbose.

    Returns:
        Configured logger for pdftopdfa.
    """
    # Determine log level (quiet takes precedence)
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger for pdftopdfa
    pdftopdfa_logger = logging.getLogger("pdftopdfa")
    pdftopdfa_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    pdftopdfa_logger.handlers.clear()

    # Create and configure handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    pdftopdfa_logger.addHandler(handler)

    logger.debug("Logging configured with level: %s", logging.getLevelName(level))
    return pdftopdfa_logger


def is_pdf_encrypted(pdf: Pdf) -> bool:
    """Checks if a PDF is encrypted.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        True if the PDF is encrypted.
    """
    return pdf.is_encrypted


def get_pdf_version(pdf: Pdf) -> str:
    """Gets the PDF version.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        PDF version as string (e.g., "1.7").
    """
    return pdf.pdf_version


def validate_pdfa_level(level: str) -> str:
    """Validates and normalizes a PDF/A target level.

    Args:
        level: PDF/A conformance level string (e.g., '3b', '2U').

    Returns:
        Lowercased level string.

    Raises:
        ConversionError: If the level is not a supported target level.
    """
    level_lower = level.lower()
    if level_lower not in SUPPORTED_LEVELS:
        raise ConversionError(
            f"Invalid PDF/A level: {level}. "
            f"Allowed: {', '.join(sorted(SUPPORTED_LEVELS))}"
        )
    return level_lower


def get_required_pdf_version(level: str) -> str:
    """Returns the required PDF version for the given level.

    Args:
        level: PDF/A conformance level.

    Returns:
        Required PDF version as string (e.g., "1.7").
    """
    return REQUIRED_PDF_VERSIONS.get(level, "1.7")


def resolve_indirect(obj: Any) -> Any:
    """Resolve indirect object reference if needed.

    pikepdf objects may be indirect references that need to be resolved.
    This safely handles the resolution without using hasattr which can
    throw exceptions on certain pikepdf object types.

    Args:
        obj: A pikepdf object that may be an indirect reference.

    Returns:
        The resolved object.
    """
    try:
        return obj.get_object()
    except Exception:
        return obj


def iter_type3_fonts(
    resources, visited: set[tuple[int, int]]
) -> Generator[tuple[str, Dictionary], None, None]:
    """Yield Type3 font objects from a Resources dictionary.

    Extracts fonts with ``/Subtype /Type3`` from ``/Resources/Font``,
    using cycle detection via ``objgen`` to avoid infinite loops.

    Args:
        resources: A resolved Resources dictionary.
        visited: Set of ``(obj_num, gen)`` tuples for cycle detection.

    Yields:
        ``(font_name, font_dict)`` tuples for each Type3 font found.
    """
    resources = resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return

    fonts = resources.get("/Font")
    if not fonts:
        return

    fonts = resolve_indirect(fonts)
    if not isinstance(fonts, Dictionary):
        return

    for font_name in list(fonts.keys()):
        try:
            font = resolve_indirect(fonts[font_name])
        except (AttributeError, TypeError, ValueError):
            continue

        if not isinstance(font, Dictionary):
            continue

        subtype = font.get("/Subtype")
        if subtype is None or str(subtype) != "/Type3":
            continue

        # Cycle detection
        objgen = font.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        yield font_name, font
