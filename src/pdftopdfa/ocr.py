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
import subprocess
import threading
from dataclasses import dataclass
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
        # Higher oversampling helps small text blocks on large pages.
        "oversample": 600,
        # Sparse-text mode is more robust when text covers only part of a page.
        "tesseract_pagesegmode": 11,
        # Let modern Tesseract handle local thresholding directly.
        "tesseract_thresholding": int(ThresholdingMethod.ADAPTIVE_OTSU),
        "optimize": 0,
        "tesseract_timeout": 120,
        "progress_bar": False,
    },
    OcrQuality.BEST: {
        "skip_text": True,
        "deskew": True,
        "rotate_pages": True,
        # Lower threshold so sideways scanned pages are rotated more reliably.
        "rotate_pages_threshold": 5.0,
        # Match DEFAULT's OCR-friendly sampling so BEST truly builds on it.
        "oversample": 600,
        "tesseract_pagesegmode": 11,
        "tesseract_thresholding": int(ThresholdingMethod.ADAPTIVE_OTSU),
        "optimize": 0,
        "tesseract_timeout": 120,
        "progress_bar": False,
    },
}

# ocrmypdf rejects these options when redo_ocr is enabled.
_REDO_OCR_INCOMPATIBLE_OPTIONS = frozenset(
    {"deskew", "clean_final", "remove_background"}
)

_ROTATION_FIX_QUALITIES = frozenset({OcrQuality.DEFAULT, OcrQuality.BEST})


@dataclass(frozen=True)
class _OrientationResult:
    """Orientation analysis result returned by Tesseract OSD."""

    rotate: int
    confidence: float


def _get_ocr_plugins(quality: OcrQuality) -> list[str]:
    """Build the plugin list for the current OCR run."""
    plugins: list[str] = []
    if quality in _ROTATION_FIX_QUALITIES:
        plugins.append(_ROTATION_FIX_PLUGIN)
    return plugins


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


def _parse_tesseract_osd(output: str) -> _OrientationResult | None:
    """Parse Tesseract OSD output into a structured result."""
    rotate = None
    confidence = None

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Rotate:"):
            try:
                rotate = int(stripped.split(":", 1)[1].strip()) % 360
            except ValueError:
                return None
        elif stripped.startswith("Orientation confidence:"):
            try:
                confidence = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                return None

    if rotate is None or confidence is None:
        return None

    return _OrientationResult(rotate=rotate, confidence=confidence)


def _run_tesseract_orientation(image_path: Path) -> _OrientationResult | None:
    """Run Tesseract OSD on an image and return the orientation result."""
    command = ["tesseract", os.fspath(image_path), "stdout", "--psm", "0"]

    try:
        with _temporary_tesseract_path():
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
    except FileNotFoundError:
        logger.debug("Tesseract not available for OCR rotation normalization")
        return None
    except Exception as exc:
        logger.debug("Tesseract OSD failed for %s: %s", image_path, exc)
        return None

    if completed.returncode != 0:
        logger.debug(
            "Tesseract OSD returned %s for %s: %s",
            completed.returncode,
            image_path,
            completed.stderr.strip(),
        )

    return _parse_tesseract_osd(completed.stdout)


def _render_pdf_page_preview(
    pdf_path: Path, *, page_index: int, output_path: Path
) -> None:
    """Render a PDF page to an image using pypdfium2."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(os.fspath(pdf_path))
    try:
        page = pdf[page_index]
        image = page.render(scale=1.5).to_pil()
        image.save(output_path)
    finally:
        close = getattr(pdf, "close", None)
        if callable(close):
            close()


def _write_single_page_with_rotate(
    input_path: Path,
    *,
    page_index: int,
    rotate: int,
    output_path: Path,
) -> Path:
    """Write a one-page PDF copy with an overridden /Rotate value."""
    import pikepdf

    with pikepdf.open(input_path) as source_pdf:
        with pikepdf.Pdf.new() as temp_pdf:
            copied_page = temp_pdf.copy_foreign(source_pdf.pages[page_index].obj)
            temp_pdf.pages.append(pikepdf.Page(copied_page))
            temp_pdf.pages[0].Rotate = rotate
            temp_pdf.save(output_path)
    return output_path


def _should_clear_page_rotate(
    existing_rotate: int,
    current_orientation: _OrientationResult | None,
    cleared_orientation: _OrientationResult | None,
) -> bool:
    """Decide whether clearing /Rotate improves the page orientation."""
    if existing_rotate % 360 == 0:
        return False
    if current_orientation is None or cleared_orientation is None:
        return False
    if current_orientation.rotate == 0:
        return False
    if cleared_orientation.rotate != 0:
        return False
    return cleared_orientation.confidence >= current_orientation.confidence


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
        logger.debug("Failed to inspect text matrix skew: %s", exc)

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


def _normalize_best_quality_text_page_skew(pdf_path: Path) -> list[tuple[int, float]]:
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


def _normalize_best_quality_text_page_rotations(pdf_path: Path) -> list[int]:
    """Clear suspicious /Rotate flags on text-only pages skipped by OCR."""
    try:
        import pikepdf
    except ImportError:
        return []

    changed_pages: list[int] = []

    with pikepdf.open(pdf_path) as pdf:
        candidates: list[tuple[int, int]] = []
        for page_index, page in enumerate(pdf.pages):
            rotate = int(page.obj.get("/Rotate", 0) or 0) % 360
            if rotate == 0:
                continue
            if not _page_has_text(page) or _page_has_images(page):
                continue
            candidates.append((page_index, rotate))

    if not candidates:
        return changed_pages

    try:
        with TemporaryDirectory(prefix="pdftopdfa_rotfix_") as tmpdir:
            temp_dir = Path(tmpdir)
            for page_index, rotate in candidates:
                current_preview = temp_dir / f"page_{page_index + 1}_current.png"
                cleared_pdf = temp_dir / f"page_{page_index + 1}_cleared.pdf"
                cleared_preview = temp_dir / f"page_{page_index + 1}_cleared.png"

                _render_pdf_page_preview(
                    pdf_path,
                    page_index=page_index,
                    output_path=current_preview,
                )
                _write_single_page_with_rotate(
                    pdf_path,
                    page_index=page_index,
                    rotate=0,
                    output_path=cleared_pdf,
                )
                _render_pdf_page_preview(
                    cleared_pdf,
                    page_index=0,
                    output_path=cleared_preview,
                )

                current_orientation = _run_tesseract_orientation(current_preview)
                cleared_orientation = _run_tesseract_orientation(cleared_preview)

                if not _should_clear_page_rotate(
                    rotate,
                    current_orientation,
                    cleared_orientation,
                ):
                    continue

                changed_pages.append(page_index)
    except ImportError:
        logger.debug("pypdfium2 not available for OCR rotation normalization")
        return []
    except Exception as exc:
        logger.debug("OCR rotation normalization skipped for %s: %s", pdf_path, exc)
        return []

    if not changed_pages:
        return []

    output_tmp = pdf_path.with_name(f"{pdf_path.stem}_rotfix.pdf")
    with pikepdf.open(pdf_path) as pdf:
        for page_index in changed_pages:
            pdf.pages[page_index].Rotate = 0
        pdf.save(output_tmp)

    output_tmp.replace(pdf_path)

    normalized_pages = [page_index + 1 for page_index in changed_pages]
    logger.info(
        "Cleared suspicious /Rotate on OCR-skipped text page(s): %s",
        ", ".join(str(page_no) for page_no in normalized_pages),
    )
    return normalized_pages


def _normalize_best_quality_skipped_text_pages(pdf_path: Path) -> None:
    """Normalize skipped text pages for best-quality OCR output."""
    _normalize_best_quality_text_page_rotations(pdf_path)
    _normalize_best_quality_text_page_skew(pdf_path)


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
            existing OCR layer and re-apply OCR. Options incompatible with
            ``redo_ocr`` are disabled automatically (default: False).

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
        plugins = _get_ocr_plugins(quality)

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

        with _temporary_tesseract_path():
            ocrmypdf.ocr(
                input_path,
                output_path,
                language=languages,
                output_type="pdf",
                rasterizer="pypdfium",
                **ocr_kwargs,
            )
        if quality == OcrQuality.BEST and output_path.exists():
            _normalize_best_quality_skipped_text_pages(output_path)
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
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e

    except Exception as e:
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e
