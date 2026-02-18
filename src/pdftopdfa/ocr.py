# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""OCR functionality for pdftopdfa.

This module provides functions for optical character recognition (OCR)
in image-based PDFs (scanned documents).
"""

# Standard Library
import contextlib
import enum
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING

# Optional import of ocrmypdf
try:
    import ocrmypdf
    from ocrmypdf.exceptions import (
        EncryptedPdfError,
        MissingDependencyError,
        PriorOcrFoundError,
    )

    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    ocrmypdf = None  # type: ignore[assignment]

    class EncryptedPdfError(Exception):  # type: ignore[no-redef]
        pass

    class MissingDependencyError(Exception):  # type: ignore[no-redef]
        pass

    class PriorOcrFoundError(Exception):  # type: ignore[no-redef]
        pass


# Optional import of OpenCV (used for OCR image preprocessing)
try:
    import cv2  # noqa: F401

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

if TYPE_CHECKING:
    import pikepdf

# Local
from .exceptions import OCRError

logger = logging.getLogger(__name__)

_path_lock = threading.Lock()


@contextlib.contextmanager
def _temporary_tesseract_path():
    """Temporarily add TESSERACT_PATH parent to PATH (thread-safe).

    ocrmypdf does not support a custom env parameter for subprocess calls,
    so we must modify os.environ temporarily. A lock serializes access to
    prevent concurrent PATH mutations from different threads.
    """
    tesseract_path = os.environ.get("TESSERACT_PATH")
    if not tesseract_path:
        yield
        return
    p = Path(tesseract_path)
    tesseract_dir = str(p) if p.is_dir() else str(p.parent)
    with _path_lock:
        saved = os.environ.get("PATH", "")
        os.environ["PATH"] = tesseract_dir + os.pathsep + saved
        try:
            yield
        finally:
            os.environ["PATH"] = saved


class OcrQuality(enum.Enum):
    """OCR quality presets controlling the speed/quality trade-off.

    Attributes:
        FAST: Minimal processing, fastest. Does not alter the document visually.
        DEFAULT: Best quality without visual changes to the document.
        BEST: Best quality, may alter the document visually (deskew, rotate, etc.).
    """

    FAST = "fast"
    DEFAULT = "default"
    BEST = "best"


OCR_SETTINGS: dict[OcrQuality, dict] = {
    OcrQuality.FAST: {
        "skip_text": True,
        "deskew": False,
        "rotate_pages": False,
        "optimize": 0,
        "tesseract_timeout": 120,
        "progress_bar": False,
    },
    OcrQuality.DEFAULT: {
        "skip_text": True,
        "deskew": False,
        "rotate_pages": False,
        "oversample": 300,
        "optimize": 0,
        "tesseract_timeout": 120,
        "progress_bar": False,
    },
    OcrQuality.BEST: {
        "skip_text": True,
        "deskew": True,
        "rotate_pages": True,
        "oversample": 300,
        "optimize": 0,
        "tesseract_timeout": 120,
        "progress_bar": False,
    },
}

# Quality levels that benefit from OpenCV preprocessing
_PREPROCESS_QUALITIES = frozenset({OcrQuality.DEFAULT, OcrQuality.BEST})


def is_ocr_available() -> bool:
    """Checks if OCR functionality is available.

    Returns:
        True if ocrmypdf is installed, False otherwise.
    """
    return HAS_OCR


def needs_ocr(pdf: "pikepdf.Pdf", *, threshold: float = 0.5) -> bool:
    """Analyzes whether a PDF needs OCR.

    Checks each page for the presence of images without recognizable text.
    A page is considered to need OCR if it contains images but has no
    text operators (Tj/TJ) in the content stream.

    Args:
        pdf: The pikepdf.Pdf object to analyze.
        threshold: Proportion of pages that must need OCR (0.0-1.0).
            Default: 0.5 (50% of pages).

    Returns:
        True if at least `threshold` of the pages need OCR.
    """
    if len(pdf.pages) == 0:
        return False

    pages_needing_ocr = 0

    for page in pdf.pages:
        has_images = _page_has_images(page)
        has_text = _page_has_text(page)

        if has_images and not has_text:
            pages_needing_ocr += 1

    ratio = pages_needing_ocr / len(pdf.pages)
    logger.debug(
        "OCR analysis: %d/%d pages need OCR (%.1f%%, threshold: %.1f%%)",
        pages_needing_ocr,
        len(pdf.pages),
        ratio * 100,
        threshold * 100,
    )

    return ratio >= threshold


def _page_has_images(page: "pikepdf.Page") -> bool:
    """Checks if a page contains images.

    Args:
        page: The pikepdf.Page to check.

    Returns:
        True if the page contains at least one image.
    """
    try:
        resources = page.get("/Resources")
        if resources is None:
            return False

        xobjects = resources.get("/XObject")
        if xobjects is None:
            return False

        for name in xobjects.keys():
            try:
                xobj = xobjects[name].get_object()
            except (AttributeError, TypeError, ValueError):
                xobj = xobjects[name]
            try:
                subtype = xobj.get("/Subtype")
                if subtype is not None and str(subtype) == "/Image":
                    return True
            except Exception:
                continue
    except Exception as e:
        logger.debug("Error during image analysis: %s", e)

    return False


def _page_has_text(page: "pikepdf.Page") -> bool:
    """Checks if a page contains text operators.

    Uses pikepdf.parse_content_stream for reliable operator detection
    instead of raw byte matching (which can false-positive on binary data).
    Also checks Form XObjects referenced from the page, since text is
    commonly rendered inside Form XObjects (e.g. overlaid text, headers/footers,
    or existing OCR layers).

    Args:
        page: The pikepdf.Page to check.

    Returns:
        True if the page contains text operators.
    """
    import pikepdf

    text_operators = frozenset(["Tj", "TJ", "'", '"'])

    try:
        for _operands, operator in pikepdf.parse_content_stream(page):
            if str(operator) in text_operators:
                return True
    except Exception as e:
        logger.debug("Error during text analysis: %s", e)

    # Check Form XObjects for text operators
    try:
        resources = page.get("/Resources")
        if resources is None:
            return False
        xobjects = resources.get("/XObject")
        if xobjects is None:
            return False

        visited: set[tuple[int, int]] = set()
        for name in xobjects.keys():
            try:
                xobj = xobjects[name].get_object()
            except (AttributeError, TypeError, ValueError):
                xobj = xobjects[name]
            if _form_xobject_has_text(xobj, text_operators, visited):
                return True
    except Exception as e:
        logger.debug("Error checking XObjects for text: %s", e)

    return False


def _form_xobject_has_text(
    xobj: "pikepdf.Object",
    text_operators: frozenset[str],
    visited: set[tuple[int, int]],
) -> bool:
    """Recursively checks a Form XObject for text operators.

    Args:
        xobj: The XObject to check.
        text_operators: Set of PDF text operator names.
        visited: Set of already-visited object IDs to prevent cycles.

    Returns:
        True if the Form XObject (or nested Form XObjects) contains text.
    """
    import pikepdf

    try:
        subtype = xobj.get("/Subtype")
        if subtype is None or str(subtype) != "/Form":
            return False
    except Exception:
        return False

    try:
        objgen = xobj.objgen
    except Exception:
        objgen = (0, 0)
    if objgen != (0, 0):
        if objgen in visited:
            return False
        visited.add(objgen)

    try:
        for _operands, operator in pikepdf.parse_content_stream(xobj):
            if str(operator) in text_operators:
                return True
    except Exception as e:
        logger.debug("Error parsing Form XObject content stream: %s", e)
        return False

    # Check nested Form XObjects
    try:
        resources = xobj.get("/Resources")
        if resources is None:
            return False
        nested_xobjects = resources.get("/XObject")
        if nested_xobjects is None:
            return False

        for name in nested_xobjects.keys():
            try:
                nested = nested_xobjects[name].get_object()
            except (AttributeError, TypeError, ValueError):
                nested = nested_xobjects[name]
            if _form_xobject_has_text(nested, text_operators, visited):
                return True
    except Exception as e:
        logger.debug("Error checking nested XObjects: %s", e)

    return False


def apply_ocr(
    input_path: Path,
    output_path: Path,
    languages: list[str] | None = None,
    *,
    quality: OcrQuality = OcrQuality.DEFAULT,
    force: bool = False,
) -> Path:
    """Performs OCR on a PDF.

    Uses ocrmypdf for text recognition. Pages that already contain text
    are skipped unless ``force=True``.

    Args:
        input_path: Path to the input PDF.
        output_path: Path for the OCR-processed PDF.
        languages: List of Tesseract language codes (default: ``["eng"]``).
            Example: ``["deu", "eng"]`` for German + English.
        quality: OCR quality preset (default: OcrQuality.DEFAULT).
        force: If True, use ocrmypdf's ``redo_ocr`` mode to remove the
            existing OCR layer and re-apply OCR (default: False).

    Returns:
        Path to the OCR-processed PDF.

    Raises:
        OCRError: If OCR is not available or fails.
    """
    if languages is None:
        languages = ["eng"]
    if not HAS_OCR:
        raise OCRError(
            "OCR not available. Install the OCR dependency: pip install pdftopdfa[ocr]"
        )

    logger.info(
        "Starting OCR for %s (languages: %s, quality: %s, force: %s)",
        input_path,
        "+".join(languages),
        quality.value,
        force,
    )

    try:
        ocr_kwargs = dict(OCR_SETTINGS[quality])

        if force:
            ocr_kwargs.pop("skip_text", None)
            ocr_kwargs["redo_ocr"] = True

        if quality in _PREPROCESS_QUALITIES:
            if HAS_OPENCV:
                ocr_kwargs["plugins"] = ["pdftopdfa.ocr_preprocess"]
                logger.debug("OpenCV preprocessing plugin enabled")
            else:
                logger.warning(
                    "OpenCV not available; skipping image preprocessing. "
                    "Install opencv-python-headless for better OCR quality."
                )

        with _temporary_tesseract_path():
            ocrmypdf.ocr(
                input_path,
                output_path,
                language=languages,
                output_type="pdf",
                rasterizer="pypdfium",
                **ocr_kwargs,
            )
        logger.info("OCR completed successfully: %s", output_path)
        return output_path

    except EncryptedPdfError as e:
        raise OCRError(f"OCR failed: PDF is encrypted ({input_path})") from e

    except PriorOcrFoundError:
        # PDF already has OCR text, just copy it
        logger.info("PDF already contains OCR text, skipping OCR")
        shutil.copy2(input_path, output_path)
        return output_path

    except MissingDependencyError as e:
        raise OCRError(f"OCR failed: {e}") from e

    except Exception as e:
        raise OCRError(f"OCR failed: {e}") from e
