# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""OCR functionality for pdftopdfa.

This module provides functions for optical character recognition (OCR)
in image-based PDFs (scanned documents).
"""

import logging
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory
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


if TYPE_CHECKING:
    import pikepdf

# Local
from .exceptions import OCRError
from .orientation import _effective_page_rotate, normalize_pdf_orientation
from .utils import log_suppressed_error

logger = logging.getLogger(__name__)

_PADDLE_OCR_PLUGIN = "pdftopdfa.ocr_paddle"
_ROTATION_FIX_PLUGIN = "pdftopdfa.ocr_rotation_fix"
_PADDLE_LANGUAGE_TAGS = {
    "ch": "zh-Hans",
    "chinese_cht": "zh-Hant",
    "french": "fr",
    "german": "de",
    "japan": "ja",
    "rs_latin": "sr-Latn",
}
PADDLE_OCR_LANGUAGES = frozenset(
    {
        "af",
        "az",
        "bs",
        "ca",
        "ch",
        "chinese_cht",
        "cs",
        "cy",
        "da",
        "de",
        "en",
        "es",
        "et",
        "eu",
        "fi",
        "fr",
        "french",
        "ga",
        "german",
        "gl",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "japan",
        "ku",
        "la",
        "lb",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pl",
        "pt",
        "qu",
        "rm",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
    }
)


def validate_ocr_languages(languages: list[str]) -> list[str]:
    """Validate and return PaddleOCR 3.7 PP-OCRv6 language codes."""
    if not languages or any(not language for language in languages):
        raise ValueError("At least one OCR language must be specified")

    unsupported = sorted(set(languages) - PADDLE_OCR_LANGUAGES)
    if unsupported:
        supported = ", ".join(sorted(PADDLE_OCR_LANGUAGES))
        raise ValueError(
            f"Unsupported PaddleOCR language code(s): {', '.join(unsupported)}. "
            f"Supported codes: {supported}"
        )
    return languages


def _format_ocr_exception(exc: BaseException) -> str:
    """Return a stable, non-empty error description for OCR exceptions."""
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _pdf_has_annotations(pdf_path: Path) -> bool:
    """Return whether a PDF has page annotations.

    Inspection failures conservatively disable deskewing so annotation geometry
    cannot be corrupted by a page transformation.
    """
    import pikepdf

    try:
        with pikepdf.open(pdf_path) as pdf:
            return any(
                (annots := page.obj.get("/Annots")) is not None and len(annots) > 0
                for page in pdf.pages
            )
    except Exception as exc:
        log_suppressed_error(
            logger,
            exc,
            "Could not inspect annotations before deskewing %s: %s",
            pdf_path,
            exc,
        )
        return True


def _ocr_form_names(pdf_path: Path) -> list[frozenset[str]]:
    """Return resource names that must not be treated as newly grafted OCR."""
    import pikepdf

    with pikepdf.open(pdf_path) as pdf:
        names = []
        for page in pdf.pages:
            xobjects = page.resources.get("/XObject")
            names.append(
                frozenset(
                    str(name)
                    for name in xobjects.keys()
                    if str(name).startswith("/OCR-")
                )
                if xobjects is not None
                else frozenset()
            )
        return names


def _strip_invisible_text_from_form(form: "pikepdf.Stream") -> bool:
    """Remove invisible text objects from a Form XObject."""
    import pikepdf

    try:
        instructions = list(pikepdf.parse_content_stream(form))
    except (pikepdf.PdfError, TypeError, ValueError) as exc:
        raise OCRError("Could not remove the existing OCR text layer") from exc

    output = []
    text_object = []
    text_show_modes: list[tuple[int, int]] = []
    in_text_object = False
    render_mode = 0
    render_mode_stack: list[int] = []
    changed = False
    text_show_operators = frozenset({"Tj", "TJ", "'", '"'})

    for instruction in instructions:
        operands, operator = instruction
        operator_name = str(operator)

        if operator_name == "Tr":
            try:
                render_mode = int(operands[0])
            except (IndexError, TypeError, ValueError) as exc:
                raise OCRError("Could not remove the existing OCR text layer") from exc
        elif operator_name == "q":
            render_mode_stack.append(render_mode)
        elif operator_name == "Q" and render_mode_stack:
            render_mode = render_mode_stack.pop()

        if not in_text_object:
            if operator_name == "BT":
                in_text_object = True
                text_object.append(instruction)
                text_show_modes.clear()
            else:
                output.append(instruction)
            continue

        text_object.append(instruction)
        if operator_name in text_show_operators:
            text_show_modes.append((len(text_object) - 1, render_mode))
        if operator_name != "ET":
            continue

        in_text_object = False
        if text_show_modes and all(mode == 3 for _, mode in text_show_modes):
            for index, _mode in text_show_modes:
                show_operands, show_operator = text_object[index]
                show_operands = list(show_operands)
                show_name = str(show_operator)
                try:
                    if show_name == "TJ":
                        show_operands[0] = pikepdf.Array(
                            pikepdf.String("")
                            if isinstance(value, pikepdf.String)
                            else value
                            for value in show_operands[0]
                        )
                    else:
                        show_operands[-1] = pikepdf.String("")
                except (IndexError, TypeError, ValueError) as exc:
                    raise OCRError(
                        "Could not remove the existing OCR text layer"
                    ) from exc
                text_object[index] = (show_operands, show_operator)
            changed = True
        output.extend(text_object)
        text_object.clear()
        text_show_modes.clear()

    if text_object:
        output.extend(text_object)

    if changed:
        form.write(pikepdf.unparse_content_stream(output))
    return changed


def _finalize_ocr_output(
    pdf_path: Path,
    languages: list[str],
    existing_ocr_form_names: list[frozenset[str]],
    *,
    strip_existing_ocr_text: bool = False,
) -> None:
    """Set OCR metadata and finalize OCR Form XObjects."""
    import pikepdf

    changed = False
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        if "/Lang" not in pdf.Root:
            language = _PADDLE_LANGUAGE_TAGS.get(languages[0], languages[0])
            pdf.Root[pikepdf.Name.Lang] = pikepdf.String(language)
            changed = True

        for page_index, page in enumerate(pdf.pages):
            if page_index >= len(existing_ocr_form_names):
                continue

            xobjects = page.resources.get("/XObject")
            if xobjects is None:
                continue

            for name, xobject in xobjects.items():
                if not str(name).startswith("/OCR-"):
                    continue
                is_existing = str(name) in existing_ocr_form_names[page_index]
                if (
                    strip_existing_ocr_text
                    and is_existing
                    and xobject.get("/Subtype") == pikepdf.Name.Form
                ):
                    changed = _strip_invisible_text_from_form(xobject) or changed

                if is_existing:
                    continue
                if xobject.get("/Subtype") != pikepdf.Name.Form:
                    continue
                rotation = _effective_page_rotate(page)
                if rotation not in {90, 270}:
                    continue

                box = xobject.get("/BBox")
                if box is None or len(box) != 4:
                    continue
                media_box = [float(value) for value in page.MediaBox]
                width = media_box[2] - media_box[0]
                height = media_box[3] - media_box[1]
                values = [float(value) for value in box]
                box_width = values[2] - values[0]
                box_height = values[3] - values[1]
                if abs(box_width - width) > 0.01 or abs(box_height - height) > 0.01:
                    continue

                xobject[pikepdf.Name.BBox] = pikepdf.Array([0, 0, height, width])
                changed = True

        if changed:
            pdf.save(pdf_path)


def is_ocr_available() -> bool:
    """Checks if OCR functionality is available.

    Returns:
        True if ocrmypdf is installed, False otherwise.
    """
    return HAS_OCR


def needs_ocr(pdf: "pikepdf.Pdf", *, threshold: float = 0.5) -> bool:
    """Analyzes whether a PDF needs OCR.

    Public library helper; not called by the conversion pipeline itself
    (OCR is opt-in via an explicit model-directory pair and
    ``ocr_languages``/``--ocr``). Use it to decide programmatically whether
    to enable OCR for a document.

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
    detection_model_dir: Path,
    recognition_model_dir: Path,
    force: bool = False,
    deskew: bool = False,
    rotate_pages: bool = False,
) -> Path:
    """Performs OCR on a PDF.

    Uses PaddleOCR for recognition and OCRmyPDF for rasterization, text-layer
    rendering, and PDF merging. Pages that already contain text are skipped
    unless ``force=True``.

    Args:
        input_path: Path to the input PDF.
        output_path: Path for the OCR-processed PDF.
        languages: PaddleOCR 3.7 PP-OCRv6 language codes (default: ``["en"]``).
            Example: ``["de", "en"]`` for German + English metadata.
        detection_model_dir: Verified PP-OCRv6 Medium detection model directory.
        recognition_model_dir: Verified PP-OCRv6 Medium recognition model directory.
        force: If True, use ocrmypdf's ``redo_ocr`` mode to remove the
            existing OCR layer and re-apply OCR. This cannot be combined
            with ``deskew`` (default: False).
        deskew: If True, straighten skewed pages with PaddleOCR (default: False).
        rotate_pages: If True, normalize page orientation with the bundled
            Paddle model (default: False).

    Returns:
        Path to the OCR-processed PDF.

    Raises:
        OCRError: If OCR is not available or fails.
    """
    if languages is None:
        languages = ["en"]
    if force and deskew:
        raise OCRError("Deskew cannot be combined with forced OCR")
    if not HAS_OCR:
        raise OCRError(
            "OCR not available. Install the OCR dependency: pip install pdftopdfa[ocr]"
        )

    from .ocr_paddle import validate_model_directories

    try:
        validate_ocr_languages(languages)
    except ValueError as exc:
        raise OCRError(str(exc)) from exc

    detection_model_dir, recognition_model_dir = validate_model_directories(
        detection_model_dir,
        recognition_model_dir,
    )

    if deskew and _pdf_has_annotations(input_path):
        logger.warning(
            "Deskew disabled because the PDF contains annotations whose geometry "
            "cannot be transformed safely"
        )
        deskew = False

    logger.info(
        "Starting PaddleOCR for %s (languages: %s, force: %s, deskew: %s, "
        "rotate pages: %s)",
        input_path,
        "+".join(languages),
        force,
        deskew,
        rotate_pages,
    )

    ocr_input_path = input_path
    orientation_temp: TemporaryDirectory[str] | None = None
    existing_ocr_form_names: list[frozenset[str]] = []

    def run_ocr() -> None:
        ocr_kwargs: dict[str, object] = {
            "ocr_engine": "paddle",
            "pdf_renderer": "fpdf2",
            "rasterizer": "pypdfium",
            "output_type": "pdf",
            "oversample": 600,
            "optimize": 0,
            "jobs": 1,
            "skip_text": True,
            "deskew": deskew,
            "rotate_pages": False,
            "progress_bar": False,
            "plugins": [_PADDLE_OCR_PLUGIN, _ROTATION_FIX_PLUGIN],
            "paddle_detection_model_dir": detection_model_dir,
            "paddle_recognition_model_dir": recognition_model_dir,
        }
        if force:
            ocr_kwargs.pop("skip_text", None)
            ocr_kwargs["redo_ocr"] = True

        ocrmypdf.ocr(
            ocr_input_path,
            output_path,
            language=languages,
            **ocr_kwargs,
        )

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
            logger.info(
                "Paddle orientation preflight completed in %.2fs",
                time.perf_counter() - orientation_started,
            )

        existing_ocr_form_names = _ocr_form_names(ocr_input_path)
        run_ocr()

        _finalize_ocr_output(
            output_path,
            languages,
            existing_ocr_form_names,
            strip_existing_ocr_text=force,
        )
        logger.info("OCR completed successfully: %s", output_path)
        return output_path

    except EncryptedPdfError as e:
        raise OCRError(f"OCR failed: PDF is encrypted ({input_path})") from e

    except PriorOcrFoundError:
        # PDF already has OCR text, just copy it
        logger.info("PDF already contains OCR text, skipping OCR")
        shutil.copy2(ocr_input_path, output_path)
        _finalize_ocr_output(
            output_path,
            languages,
            existing_ocr_form_names,
            strip_existing_ocr_text=force,
        )
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
