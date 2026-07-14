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
import math
import os
import shutil
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

# Optional import of ocrmypdf
try:
    import ocrmypdf
    from ocrmypdf._exec.tesseract import ThresholdingMethod
    from ocrmypdf.exceptions import (
        EncryptedPdfError,
        MissingDependencyError,
        PriorOcrFoundError,
    )

    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    ocrmypdf = None  # type: ignore[assignment]

    class ThresholdingMethod(enum.IntEnum):  # type: ignore[no-redef]
        AUTO = 0
        OTSU = 0
        ADAPTIVE_OTSU = 1
        SAUVOLA = 2

    class EncryptedPdfError(Exception):  # type: ignore[no-redef]
        pass

    class MissingDependencyError(Exception):  # type: ignore[no-redef]
        pass

    class PriorOcrFoundError(Exception):  # type: ignore[no-redef]
        pass


if TYPE_CHECKING:
    import pikepdf

# Local
from .exceptions import OCRError
from .orientation import normalize_pdf_orientation
from .utils import log_suppressed_error

logger = logging.getLogger(__name__)

_path_lock = threading.Lock()
_ROTATION_FIX_PLUGIN = "pdftopdfa.ocr_rotation_fix"


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
    """

    FAST = "fast"
    DEFAULT = "default"


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
        # Higher oversampling helps small text blocks on large pages.
        "oversample": 600,
        # Sparse-text mode is more robust when text covers only part of a page.
        "tesseract_pagesegmode": 11,
        # Let modern Tesseract handle local thresholding directly.
        "tesseract_thresholding": int(ThresholdingMethod.ADAPTIVE_OTSU),
        "optimize": 0,
        "tesseract_timeout": 300,
        "progress_bar": False,
    },
}

DEFAULT_OCR_FALLBACK_QUALITY = OcrQuality.FAST
DEFAULT_OCR_FALLBACK_AFTER_SECONDS = 60.0
_OCR_QUALITY_RANK = {
    OcrQuality.FAST: 0,
    OcrQuality.DEFAULT: 1,
}

# ocrmypdf rejects these options when redo_ocr is enabled.
_REDO_OCR_INCOMPATIBLE_OPTIONS = frozenset(
    {"deskew", "clean_final", "remove_background"}
)

_ROTATION_FIX_QUALITIES = frozenset({OcrQuality.DEFAULT})


def _get_ocr_plugins(
    quality: OcrQuality,
    *,
    rotate_pages: bool = False,
) -> list[str]:
    """Build the plugin list for the current OCR run."""
    plugins: list[str] = []
    if rotate_pages or quality in _ROTATION_FIX_QUALITIES:
        plugins.append(_ROTATION_FIX_PLUGIN)
    return plugins


def _is_faster_ocr_quality(candidate: OcrQuality, quality: OcrQuality) -> bool:
    """Return True if candidate is a faster OCR quality than quality."""
    return _OCR_QUALITY_RANK[candidate] < _OCR_QUALITY_RANK[quality]


def _effective_fallback_quality(
    quality: OcrQuality,
    fallback_quality: OcrQuality | None,
) -> OcrQuality | None:
    """Return the fallback quality to use for quality, if any."""
    if fallback_quality is None:
        return None
    if not _is_faster_ocr_quality(fallback_quality, quality):
        return None
    return fallback_quality


def _remove_redo_ocr_incompatible_options(ocr_kwargs: dict[str, object]) -> list[str]:
    """Remove options that ocrmypdf rejects together with redo_ocr."""
    removed_options: list[str] = []

    for option in sorted(_REDO_OCR_INCOMPATIBLE_OPTIONS):
        if option not in ocr_kwargs:
            continue

        value = ocr_kwargs.pop(option)
        if value:
            removed_options.append(option)

    return removed_options


def _format_ocr_exception(exc: BaseException) -> str:
    """Return a stable, non-empty error description for OCR exceptions."""
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _cap_tesseract_timeout(
    ocr_kwargs: dict[str, object],
    fallback_after_seconds: float | None,
) -> None:
    """Cap Tesseract OCR timeout so fallback can happen promptly."""
    if fallback_after_seconds is None:
        return
    timeout = max(1, math.ceil(fallback_after_seconds))
    existing_timeout = ocr_kwargs.get("tesseract_timeout")
    if existing_timeout is None or float(existing_timeout) > timeout:
        ocr_kwargs["tesseract_timeout"] = timeout


def _count_pdf_pages(pdf_path: Path) -> int:
    """Return the page count of a PDF, or 1 if it cannot be determined."""
    import pikepdf

    try:
        with pikepdf.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as exc:
        log_suppressed_error(
            logger, exc, "Could not count pages of %s: %s", pdf_path, exc
        )
        return 1


def _extract_text_matrix_angles(page) -> list[float]:
    """Collect text-matrix rotation angles from a page's content stream."""
    angles: list[float] = []

    try:
        import pikepdf

        for operands, operator in pikepdf.parse_content_stream(page):
            if str(operator) != "Tm" or len(operands) != 6:
                continue
            a, b, c, d, _e, _f = [float(value) for value in operands]
            if abs(a) < 1e-6 and abs(b) < 1e-6:
                continue
            angle = math.degrees(math.atan2(b, a))
            if abs(d) > 1e-6 or abs(c) > 1e-6:
                shear = math.degrees(math.atan2(c, d))
            else:
                shear = 0.0
            if abs(angle - (-shear)) > 0.25:
                continue
            angles.append(angle)
    except Exception as exc:
        log_suppressed_error(logger, exc, "Failed to inspect text matrix skew: %s", exc)

    return angles


def _detect_consistent_text_skew(page) -> float | None:
    """Detect a dominant small skew angle on a text-only page."""
    angles = _extract_text_matrix_angles(page)
    if len(angles) < 2:
        return None

    median_angle = sorted(angles)[len(angles) // 2]
    abs_median = abs(median_angle)
    if abs_median < 0.5 or abs_median > 10.0:
        return None

    consistent = [angle for angle in angles if abs(angle - median_angle) <= 0.5]
    if len(consistent) / len(angles) < 0.8:
        return None

    return sum(consistent) / len(consistent)


def _apply_page_content_transform(pdf, page, *, angle_degrees: float) -> None:
    """Apply a global counter-rotation while preserving the page format."""
    import pikepdf

    media_box = [float(value) for value in page.mediabox]
    page_width = media_box[2] - media_box[0]
    page_height = media_box[3] - media_box[1]
    radians = math.radians(angle_degrees)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    rotated_width = abs(page_width * cos_theta) + abs(page_height * sin_theta)
    rotated_height = abs(page_width * sin_theta) + abs(page_height * cos_theta)
    scale = min(page_width / rotated_width, page_height / rotated_height, 1.0)

    a = scale * cos_theta
    b = scale * sin_theta
    c = -scale * sin_theta
    d = scale * cos_theta
    center_x = media_box[0] + page_width / 2.0
    center_y = media_box[1] + page_height / 2.0
    translate_x = center_x - (a * center_x + c * center_y)
    translate_y = center_y - (b * center_x + d * center_y)

    prefix = pdf.make_stream(
        (
            "q\n"
            f"{a:.12f} {b:.12f} "
            f"{c:.12f} {d:.12f} "
            f"{translate_x:.12f} {translate_y:.12f} cm\n"
        ).encode("ascii")
    )
    suffix = pdf.make_stream(b"\nQ\n")

    contents = page.obj.get("/Contents")
    if isinstance(contents, pikepdf.Array):
        wrapped_contents = pikepdf.Array([prefix, *contents, suffix])
    else:
        wrapped_contents = pikepdf.Array([prefix, contents, suffix])
    page.obj[pikepdf.Name.Contents] = wrapped_contents


def _normalize_text_page_skew(pdf_path: Path) -> list[tuple[int, float]]:
    """Deskew text-only pages whose text matrices are consistently slanted."""
    try:
        import pikepdf
    except ImportError:
        return []

    normalized: list[tuple[int, float]] = []
    output_tmp = pdf_path.with_name(f"{pdf_path.stem}_deskew.pdf")

    with pikepdf.open(pdf_path) as pdf:
        changed = False
        for page_index, page in enumerate(pdf.pages):
            rotate = int(page.obj.get("/Rotate", 0) or 0) % 360
            if rotate != 0:
                continue
            if not _page_has_text(page) or _page_has_images(page):
                continue

            skew_angle = _detect_consistent_text_skew(page)
            if skew_angle is None:
                continue

            _apply_page_content_transform(
                pdf,
                page,
                angle_degrees=-skew_angle,
            )
            normalized.append((page_index + 1, skew_angle))
            changed = True

        if not changed:
            return []

        pdf.save(output_tmp)

    output_tmp.replace(pdf_path)

    logger.info(
        "Deskewed OCR-skipped text page(s): %s",
        ", ".join(f"{page_no} ({angle:.2f}deg)" for page_no, angle in normalized),
    )
    return normalized


def is_ocr_available() -> bool:
    """Checks if OCR functionality is available.

    Returns:
        True if ocrmypdf is installed, False otherwise.
    """
    return HAS_OCR


def needs_ocr(pdf: "pikepdf.Pdf", *, threshold: float = 0.5) -> bool:
    """Analyzes whether a PDF needs OCR.

    Public library helper; not called by the conversion pipeline itself
    (OCR is opt-in via ``ocr_languages``/``--ocr``). Use it to decide
    programmatically whether to enable OCR for a document.

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
        log_suppressed_error(logger, e, "Error during image analysis: %s", e)

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
        log_suppressed_error(logger, e, "Error during text analysis: %s", e)

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
        log_suppressed_error(logger, e, "Error checking XObjects for text: %s", e)

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
        log_suppressed_error(
            logger, e, "Error parsing Form XObject content stream: %s", e
        )
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
        log_suppressed_error(logger, e, "Error checking nested XObjects: %s", e)

    return False


def apply_ocr(
    input_path: Path,
    output_path: Path,
    languages: list[str] | None = None,
    *,
    quality: OcrQuality = OcrQuality.DEFAULT,
    fallback_quality: OcrQuality | None = DEFAULT_OCR_FALLBACK_QUALITY,
    fallback_after_seconds: float | None = DEFAULT_OCR_FALLBACK_AFTER_SECONDS,
    force: bool = False,
    deskew: bool = False,
    rotate_pages: bool = False,
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
        fallback_quality: Faster OCR quality to retry with if the initial OCR
            run exceeds the fallback threshold. Use ``None`` to disable.
        fallback_after_seconds: Per-page runtime budget for OCR fallback. The
            retry triggers when the whole run takes longer than this value
            multiplied by the page count, so large documents are not penalized
            for their size. The per-page Tesseract timeout of the initial run
            is capped to this value so fallback can happen promptly. Use
            ``None`` to disable time-based retry.
        force: If True, use ocrmypdf's ``redo_ocr`` mode to remove the
            existing OCR layer and re-apply OCR. This cannot be combined
            with ``deskew`` (default: False).
        deskew: If True, straighten skewed pages independently of the quality
            preset (default: False).
        rotate_pages: If True, normalize page orientation with the bundled
            Paddle model independently of the quality preset (default: False).

    Returns:
        Path to the OCR-processed PDF.

    Raises:
        OCRError: If OCR is not available or fails.
    """
    if languages is None:
        languages = ["eng"]
    if force and deskew:
        raise OCRError("Deskew cannot be combined with forced OCR")
    if not HAS_OCR:
        raise OCRError(
            "OCR not available. Install the OCR dependency: pip install pdftopdfa[ocr]"
        )

    logger.info(
        "Starting OCR for %s (languages: %s, quality: %s, force: %s, "
        "deskew: %s, rotate pages: %s)",
        input_path,
        "+".join(languages),
        quality.value,
        force,
        deskew,
        rotate_pages,
    )

    fallback = _effective_fallback_quality(quality, fallback_quality)
    per_page_limit = fallback_after_seconds if fallback is not None else None
    # The threshold is a per-page budget: scale it by the page count so a
    # large document that OCRs each page quickly does not trigger fallback.
    total_limit: float | None = None
    if per_page_limit is not None:
        total_limit = per_page_limit * max(1, _count_pdf_pages(input_path))

    ocr_input_path = input_path
    orientation_elapsed = 0.0
    orientation_temp: TemporaryDirectory[str] | None = None

    def run_ocr_with_quality(
        run_quality: OcrQuality,
        *,
        timeout_limit: float | None = None,
    ) -> float:
        ocr_kwargs = dict(OCR_SETTINGS[run_quality])
        ocr_kwargs["deskew"] = deskew
        _cap_tesseract_timeout(ocr_kwargs, timeout_limit)
        plugins = _get_ocr_plugins(run_quality, rotate_pages=rotate_pages)

        if force:
            ocr_kwargs.pop("skip_text", None)
            removed_options = _remove_redo_ocr_incompatible_options(ocr_kwargs)
            if removed_options:
                logger.info(
                    "force=True disables redo_ocr-incompatible OCR options: %s",
                    ", ".join(removed_options),
                )
            ocr_kwargs["redo_ocr"] = True

        if plugins:
            ocr_kwargs["plugins"] = plugins

        started = time.perf_counter()
        with _temporary_tesseract_path():
            ocrmypdf.ocr(
                ocr_input_path,
                output_path,
                language=languages,
                output_type="pdf",
                rasterizer="pypdfium",
                **ocr_kwargs,
            )
        return time.perf_counter() - started

    try:
        if rotate_pages:
            orientation_temp = TemporaryDirectory(
                prefix="pdftopdfa_paddle_orientation_"
            )
            ocr_input_path = (
                Path(orientation_temp.name) / f"{input_path.stem}_oriented.pdf"
            )
            orientation_started = time.perf_counter()
            normalize_pdf_orientation(input_path, ocr_input_path)
            orientation_elapsed = time.perf_counter() - orientation_started
            logger.info(
                "Paddle orientation preflight completed in %.2fs",
                orientation_elapsed,
            )

        elapsed = orientation_elapsed + run_ocr_with_quality(
            quality,
            timeout_limit=per_page_limit,
        )

        if fallback is not None and total_limit is not None and elapsed > total_limit:
            logger.warning(
                "Retrying OCR with fallback quality '%s' after '%s' took %.2fs "
                "(limit: %.2fs)",
                fallback.value,
                quality.value,
                elapsed,
                total_limit,
            )
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
            run_ocr_with_quality(fallback)

        if deskew and output_path.exists():
            _normalize_text_page_skew(output_path)
        logger.info("OCR completed successfully: %s", output_path)
        return output_path

    except EncryptedPdfError as e:
        raise OCRError(f"OCR failed: PDF is encrypted ({input_path})") from e

    except PriorOcrFoundError:
        # PDF already has OCR text, just copy it
        logger.info("PDF already contains OCR text, skipping OCR")
        shutil.copy2(ocr_input_path, output_path)
        return output_path

    except MissingDependencyError as e:
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e

    except OCRError:
        raise

    except Exception as e:
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e

    finally:
        if orientation_temp is not None:
            orientation_temp.cleanup()
