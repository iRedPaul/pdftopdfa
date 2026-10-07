# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for PDF/A conversion."""

import io
import logging
import sys
from collections.abc import Generator
from typing import IO, Any

from pikepdf import Array, Dictionary, Object, Pdf

from .exceptions import ConversionError

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Supported target levels for PDF/A conversion
SUPPORTED_LEVELS = frozenset({"2a", "2b", "2u", "3a", "3b", "3u"})

# Required PDF versions for PDF/A levels
REQUIRED_PDF_VERSIONS = {
    "2a": "1.7",
    "2b": "1.7",
    "2u": "1.7",
    "3a": "1.7",
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


def log_suppressed_error(
    module_logger: logging.Logger,
    exc: BaseException,
    msg: str,
    *args: object,
    level: int = logging.DEBUG,
) -> None:
    """Logs an exception swallowed by a robustness handler.

    Broken PDFs routinely raise data errors that are safe to skip, so
    they are logged at ``level`` (DEBUG by default). AttributeError and
    NameError, however, almost always indicate a programming error rather
    than a data error; they are logged at WARNING so they stay visible
    instead of being silently mistaken for malformed input.

    Args:
        module_logger: Logger of the calling module.
        exc: The suppressed exception.
        msg: %-style log message (same as for logging calls).
        *args: Arguments for the log message.
        level: Log level for ordinary data errors.
    """
    if isinstance(exc, (AttributeError, NameError)):
        module_logger.warning(
            msg + " (%s may indicate a programming error)",
            *args,
            type(exc).__name__,
        )
    else:
        module_logger.log(level, msg, *args)


class _SpoolWriter(io.RawIOBase):
    """Writable view of a spool that exposes no file descriptor."""

    def __init__(self, spool: IO[bytes]) -> None:
        super().__init__()
        self._spool = spool

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        return self._spool.write(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._spool.seek(offset, whence)

    def tell(self) -> int:
        return self._spool.tell()

    def flush(self) -> None:
        if not self.closed:
            self._spool.flush()


def save_pdf_to_spool(pdf: Pdf, spool: IO[bytes], **save_options: Any) -> None:
    """Saves a PDF into a SpooledTemporaryFile without forcing rollover.

    pikepdf 10.14+ calls ``fileno()`` on save streams to detect overwriting the
    input, and ``SpooledTemporaryFile.fileno()`` always rolls over to disk.
    A fresh spool can never be the input, so the descriptor is hidden.

    Args:
        pdf: Opened pikepdf PDF object.
        spool: Empty writable spool.
        **save_options: Keyword arguments for ``pikepdf.Pdf.save``.
    """
    writer = _SpoolWriter(spool)
    try:
        pdf.save(writer, **save_options)
    finally:
        writer.close()


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
    # pikepdf resolves indirect references transparently and has no
    # ``get_object`` method: on a pikepdf object the call below only fails
    # (as a missing dictionary key), which costs ~6 us per call. Return
    # pikepdf objects and None directly; keep the fallback for other types.
    if obj is None or isinstance(obj, Object):
        return obj
    try:
        return obj.get_object()
    except Exception:
        return obj


def iter_type3_fonts(
    resources, visited: set[tuple[int, int]]
) -> Generator[tuple[str, Dictionary], None, None]:
    """Yield Type3 font objects from a Resources dictionary.

    Extracts fonts with ``/Subtype /Type3`` from ``/Resources/Font`` and
    ``/Resources/ExtGState/*/Font``,
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

    fonts = resolve_indirect(resources.get("/Font"))
    candidates = list(fonts.items()) if isinstance(fonts, Dictionary) else []
    states = resolve_indirect(resources.get("/ExtGState"))
    if isinstance(states, Dictionary):
        for name, state in states.items():
            if not isinstance(state, Dictionary):
                continue
            font = resolve_indirect(state.get("/Font"))
            if isinstance(font, Array) and len(font) == 2:
                candidates.append((name, font[0]))

    for font_name, font in candidates:
        try:
            font = resolve_indirect(font)
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
