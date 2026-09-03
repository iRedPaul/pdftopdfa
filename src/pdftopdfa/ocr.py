# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""OCR functionality for pdftopdfa.

This module provides functions for optical character recognition (OCR)
in image-based PDFs (scanned documents).
"""

import copy
import gc
import json
import logging
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory, mkstemp
from typing import TYPE_CHECKING, Any

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
from ._ocr_runtime import execution_provider_base, onnxruntime_engine_config
from .exceptions import ConversionError, OCRError
from .orientation import (
    ORIENTATION_RENDER_SCALE,
    _effective_page_rotate,
    normalize_pdf_orientation,
)
from .staging import (
    StagedFileSnapshot,
    private_staging_directory,
    publish_staged_file,
    rollback_staged_publication,
    staged_file_snapshot,
)
from .utils import log_suppressed_error

logger = logging.getLogger(__name__)

_PADDLE_OCR_PLUGIN = "pdftopdfa.ocr_paddle"
_ROTATION_FIX_PLUGIN = "pdftopdfa.ocr_rotation_fix"
_SCAN_IMAGE_AREA_RATIO = 0.8
_FULL_PAGE_IMAGE_AREA_RATIO = 1.0
_OCR_MANIFEST_SCHEMA_VERSION = 1
_OCR_PAGE_MANIFEST_TYPE = "pdftopdfa-ocr-page"
_OCR_DOCUMENT_MANIFEST_TYPE = "pdftopdfa-ocr-document"
_OCR_RASTER_DPI = 600
_OCR_FALLBACK_RASTER_DPI = 300
# Keeps one grayscale raster near 250 MB while admitting A0 at fallback DPI.
_OCR_MAX_PAGE_RASTER_PIXELS = 250_000_000
# Bound non-raster page inspection and planning work as well.
_OCR_MAX_DOCUMENT_PAGES = 100_000
_OCR_MANIFEST_GEOMETRY_TOLERANCE = 1e-6
_MAX_PDF_USER_UNIT = 75_000.0
_ObjectKey = tuple[int, int] | tuple[str, bytes]
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
_NON_LATIN_OCR_LANGUAGES = frozenset({"ch", "chinese_cht", "japan"})


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


def _latin_only_ocr_languages(languages: list[str]) -> bool:
    """Return whether OCR decoding should reject non-Latin letters."""
    return not _NON_LATIN_OCR_LANGUAGES.intersection(languages)


def _require_ocr_runtime(ocr_execution_provider: str) -> None:
    onnxruntime_engine_config(ocr_execution_provider)
    if HAS_OCR:
        return
    extra = (
        "directml"
        if execution_provider_base(ocr_execution_provider) == "directml"
        else "ocr"
    )
    raise OCRError(
        f"OCR not available. Install the OCR dependency: pip install pdftopdfa[{extra}]"
    )


class OCRSession:
    """Reuse one PP-OCRv6 model session across images from one document."""

    def __init__(
        self,
        *,
        detection_model_dir: Path,
        recognition_model_dir: Path,
        ocr_execution_provider: str = "cpu",
    ) -> None:
        """Configure a document-bound OCR session."""
        _require_ocr_runtime(ocr_execution_provider)

        self._detection_model_dir = detection_model_dir
        self._recognition_model_dir = recognition_model_dir
        self._ocr_execution_provider = ocr_execution_provider
        self._backend: Any | None = None
        self._closed = False
        self._lock = threading.RLock()

    def __enter__(self) -> "OCRSession":
        with self._lock:
            if self._closed:
                raise RuntimeError("OCRSession is closed")
            return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except Exception as close_error:
            if exc is None:
                raise
            exc.add_note(f"OCR model cleanup also failed: {close_error}")

    def close(self) -> None:
        """Release the session's OCR models exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            backend = self._backend
            self._backend = None
            if backend is None:
                return
            try:
                backend.close()
            finally:
                gc.collect()

    def recognize_image(
        self,
        input_path: str | Path,
        *,
        layout: str = "auto",
        allowed_characters: str | None = None,
        languages: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Recognize one image while retaining the session's loaded models."""
        with self._lock:
            if self._closed:
                raise RuntimeError("OCRSession is closed")
            latin_only = False
            if languages is not None:
                latin_only = _latin_only_ocr_languages(
                    validate_ocr_languages(languages)
                )
            if self._backend is None:
                from .ocr_paddle import _ImageOCRSession

                self._backend = _ImageOCRSession(
                    detection_model_dir=self._detection_model_dir,
                    recognition_model_dir=self._recognition_model_dir,
                    ocr_execution_provider=self._ocr_execution_provider,
                )
            return self._backend.recognize_image(
                input_path,
                layout=layout,
                allowed_characters=allowed_characters,
                latin_only=latin_only,
            )


def recognize_image(
    input_path: str | Path,
    *,
    detection_model_dir: Path,
    recognition_model_dir: Path,
    ocr_execution_provider: str = "cpu",
    layout: str = "auto",
    allowed_characters: str | None = None,
) -> list[tuple[str, float]]:
    """Recognize text and confidence values in an image with PP-OCRv6.

    ``layout="auto"`` uses the normal text detection and recognition pipeline.
    ``layout="single_line"`` bypasses text detection and sends the complete
    image directly to the recognition model. When ``allowed_characters`` is
    provided, disallowed CTC classes are masked before decoding.
    """
    with OCRSession(
        detection_model_dir=detection_model_dir,
        recognition_model_dir=recognition_model_dir,
        ocr_execution_provider=ocr_execution_provider,
    ) as session:
        return session.recognize_image(
            input_path,
            layout=layout,
            allowed_characters=allowed_characters,
        )


def _release_ocr_models() -> None:
    """Release PaddleOCR models and their native ONNX Runtime sessions."""
    from .ocr_paddle import _release_model_cache
    from .orientation import _release_model_cache as release_orientation_model_cache

    _release_model_cache()
    release_orientation_model_cache()
    gc.collect()


def _format_ocr_exception(exc: BaseException) -> str:
    """Return a stable, non-empty error description for OCR exceptions."""
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


@dataclass
class _PagePaintAnalysis:
    """Conservative paint information used to identify scan-like pages."""

    event: int = 0
    visible_text: list[int] = field(default_factory=list)
    image_candidates: list[tuple[float, int]] = field(default_factory=list)
    has_vector: bool = False
    unsafe: bool = False


@dataclass
class _ContentAnalysisFrame:
    """Mutable state for one content stream in the iterative paint walk."""

    instructions: list[object]
    resources: "pikepdf.Object | None"
    ctm: "pikepdf.Matrix"
    render_mode: int
    font_is_type3: bool | None
    fill_alpha: float
    stroke_alpha: float
    blend_mode: object
    soft_mask: object
    overprint: bool
    clip_unknown: bool
    text_clip: bool
    optional_content_depth: int
    transparency_group: bool
    active_form_key: _ObjectKey | None = None
    index: int = 0
    state_stack: list[tuple[object, ...]] = field(default_factory=list)
    marked_content_stack: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class _DeskewPlan:
    deskew_pages: tuple[int, ...]
    regular_ocr_pages: tuple[int, ...]
    redo_ocr_pages: tuple[int, ...]
    strip_text_pages: tuple[int, ...]
    ambiguous_scan_pages: tuple[int, ...]


def _object_key(value: "pikepdf.Object") -> _ObjectKey:
    """Return a recursion key for an indirect or direct PDF object."""
    try:
        objgen = value.objgen
    except Exception:
        objgen = (0, 0)
    if objgen != (0, 0):
        return objgen
    try:
        serialized = value.unparse()
    except Exception:
        serialized = repr(value).encode("utf-8", errors="backslashreplace")
    return ("direct", serialized)


def _matrix_from_operands(operands: object) -> "pikepdf.Matrix":
    """Return a PDF matrix, raising for malformed operands."""
    import pikepdf

    values = [float(value) for value in operands]  # type: ignore[union-attr]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("Invalid PDF transformation matrix")
    return pikepdf.Matrix(*values)


def _rect_coverage(
    matrix: "pikepdf.Matrix",
    visible_box: tuple[float, float, float, float],
) -> float:
    """Return the visible-page fraction covered by an axis-aligned rectangle."""
    values = (matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f)
    if not all(math.isfinite(value) for value in values):
        return 0.0
    tolerance = 1e-7
    if not (
        (abs(matrix.b) <= tolerance and abs(matrix.c) <= tolerance)
        or (abs(matrix.a) <= tolerance and abs(matrix.d) <= tolerance)
    ):
        return 0.0

    corners = [
        matrix.transform(point)
        for point in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    ]
    image_left = min(point[0] for point in corners)
    image_bottom = min(point[1] for point in corners)
    image_right = max(point[0] for point in corners)
    image_top = max(point[1] for point in corners)
    page_left, page_bottom, page_right, page_top = visible_box
    page_area = (page_right - page_left) * (page_top - page_bottom)
    if page_area <= 0:
        return 0.0

    intersection_width = max(
        0.0,
        min(image_right, page_right) - max(image_left, page_left),
    )
    intersection_height = max(
        0.0,
        min(image_top, page_top) - max(image_bottom, page_bottom),
    )
    return intersection_width * intersection_height / page_area


def _normal_blend_mode(value: object) -> bool:
    """Return whether a PDF blend mode is opaque-compatible."""
    if value is None:
        return True
    if str(value) in {"/Normal", "/Compatible"}:
        return True
    try:
        values = list(value)  # type: ignore[call-overload]
    except TypeError:
        return False
    return bool(values) and all(
        str(item) in {"/Normal", "/Compatible"} for item in values
    )


def _image_color_space_marks(
    value: object,
    resources: "pikepdf.Object | None",
    aliases: set[str] | None = None,
) -> bool:
    """Return whether an image color space is known to paint page content."""
    aliases = set() if aliases is None else aliases
    seen_complex: set[_ObjectKey] = set()
    while value is not None:
        name = str(value)
        device_default = {
            "/DeviceGray": "/DefaultGray",
            "/DeviceRGB": "/DefaultRGB",
            "/DeviceCMYK": "/DefaultCMYK",
        }.get(name)
        if device_default is not None:
            try:
                color_spaces = resources.get("/ColorSpace") if resources else None
                resolved = (
                    color_spaces.get(device_default)
                    if color_spaces is not None
                    else None
                )
            except (AttributeError, TypeError, ValueError):
                return False
            if resolved is None:
                return True
            if device_default in aliases:
                return False
            aliases.add(device_default)
            value = resolved
            continue
        if name.startswith("/"):
            if name in aliases:
                return False
            try:
                color_spaces = resources.get("/ColorSpace") if resources else None
                resolved = color_spaces.get(value) if color_spaces is not None else None
            except (AttributeError, TypeError, ValueError):
                return False
            if resolved is None:
                return False
            aliases.add(name)
            value = resolved
            continue

        key = _object_key(value)  # type: ignore[arg-type]
        if key in seen_complex:
            return False
        seen_complex.add(key)
        try:
            values = list(value)  # type: ignore[call-overload]
        except TypeError:
            return False
        if not values:
            return False

        family = str(values[0])
        if family == "/Separation":
            return len(values) >= 4 and str(values[1]) != "/None"
        if family == "/DeviceN":
            if len(values) < 4:
                return False
            try:
                colorants = list(values[1])
            except TypeError:
                return False
            return bool(colorants) and any(
                str(colorant) != "/None" for colorant in colorants
            )
        if family == "/Indexed":
            if len(values) < 4:
                return False
            value = values[1]
            continue
        return family in {"/CalGray", "/CalRGB", "/Lab", "/ICCBased"}
    return False


def _uses_jpx_decode(value: object) -> bool:
    """Return whether an image filter chain includes JPXDecode."""
    if str(value) == "/JPXDecode":
        return True
    try:
        return any(str(item) == "/JPXDecode" for item in value)  # type: ignore[union-attr]
    except TypeError:
        return False


def _font_is_type3(value: object) -> bool:
    """Return whether a resolved font dictionary is a Type 3 font."""
    try:
        try:
            value = value.get_object()  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError):
            pass
        subtype = value.get("/Subtype")  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid font") from exc
    if subtype is None:
        raise ValueError("Missing font subtype")
    return str(subtype) == "/Type3"


def _soft_mask_hides_all(value: object) -> bool:
    """Return whether an alpha soft mask is provably fully transparent."""
    if value is None or str(value) == "/None":
        return False
    try:
        group = value.get("/G")  # type: ignore[attr-defined]
        group_attributes = group.get("/Group")
        box = [float(item) for item in group.get("/BBox")]
        return (
            str(value.get("/S")) == "/Alpha"  # type: ignore[attr-defined]
            and value.get("/TR") is None  # type: ignore[attr-defined]
            and str(group.get("/Type")) == "/XObject"
            and str(group.get("/Subtype")) == "/Form"
            and str(group_attributes.get("/S")) == "/Transparency"
            and len(box) == 4
            and all(math.isfinite(item) for item in box)
            and box[2] >= box[0]
            and box[3] >= box[1]
            and not group.read_bytes().strip()
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _text_is_visible(
    render_mode: int,
    fill_alpha: float,
    stroke_alpha: float,
    soft_mask: object,
) -> bool:
    """Return whether a text-show operation paints glyphs."""
    if _soft_mask_hides_all(soft_mask):
        return False
    if render_mode in {0, 4}:
        return fill_alpha > 0
    if render_mode in {1, 5}:
        return stroke_alpha > 0
    if render_mode in {2, 6}:
        return fill_alpha > 0 or stroke_alpha > 0
    return False


def _analyze_content(
    container: "pikepdf.Object",
    resources: "pikepdf.Object | None",
    visible_box: tuple[float, float, float, float],
    analysis: _PagePaintAnalysis,
    *,
    ctm: "pikepdf.Matrix",
    render_mode: int = 0,
    font_is_type3: bool | None = None,
    fill_alpha: float = 1.0,
    stroke_alpha: float = 1.0,
    blend_mode: object = None,
    soft_mask: object = None,
    overprint: bool = False,
    clip_unknown: bool = False,
    text_clip: bool = False,
    optional_content: bool = False,
    transparency_group: bool = False,
    active_forms: set[_ObjectKey],
) -> None:
    """Walk a content stream in paint order and collect conservative page facts."""
    import pikepdf

    text_show_operators = frozenset({"Tj", "TJ", "'", '"'})
    path_paint_operators = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})
    frames = [
        _ContentAnalysisFrame(
            list(pikepdf.parse_content_stream(container)),
            resources,
            ctm,
            render_mode,
            font_is_type3,
            fill_alpha,
            stroke_alpha,
            blend_mode,
            soft_mask,
            overprint,
            clip_unknown,
            text_clip,
            int(optional_content),
            transparency_group,
        )
    ]

    def finish_frame() -> None:
        frame = frames.pop()
        if frame.active_form_key is not None:
            active_forms.remove(frame.active_form_key)

    try:
        while frames:
            frame = frames[-1]
            if frame.index >= len(frame.instructions):
                if frame.state_stack or frame.marked_content_stack:
                    analysis.unsafe = True
                finish_frame()
                continue

            operands, operator = frame.instructions[frame.index]  # type: ignore[misc]
            frame.index += 1
            operator_name = str(operator)
            if operator_name == "q":
                frame.state_stack.append(
                    (
                        frame.ctm,
                        frame.render_mode,
                        frame.font_is_type3,
                        frame.fill_alpha,
                        frame.stroke_alpha,
                        frame.blend_mode,
                        frame.soft_mask,
                        frame.overprint,
                        frame.clip_unknown,
                        frame.text_clip,
                    )
                )
            elif operator_name == "Q":
                if not frame.state_stack:
                    analysis.unsafe = True
                    finish_frame()
                    continue
                (
                    frame.ctm,
                    frame.render_mode,
                    frame.font_is_type3,
                    frame.fill_alpha,
                    frame.stroke_alpha,
                    frame.blend_mode,
                    frame.soft_mask,
                    frame.overprint,
                    frame.clip_unknown,
                    frame.text_clip,
                ) = frame.state_stack.pop()
            elif operator_name == "cm":
                try:
                    frame.ctm = _matrix_from_operands(operands) @ frame.ctm
                except (TypeError, ValueError):
                    analysis.unsafe = True
                    finish_frame()
            elif operator_name == "Tr":
                try:
                    frame.render_mode = int(operands[0])
                except (IndexError, TypeError, ValueError):
                    analysis.unsafe = True
                    finish_frame()
                    continue
                if frame.render_mode not in range(8):
                    analysis.unsafe = True
                    finish_frame()
            elif operator_name == "Tf":
                try:
                    if len(operands) != 2 or not math.isfinite(float(operands[1])):
                        raise ValueError("Invalid Tf operands")
                    fonts = frame.resources.get("/Font") if frame.resources else None
                    font = fonts.get(operands[0]) if fonts is not None else None
                    if font is None:
                        raise ValueError("Missing font")
                    frame.font_is_type3 = _font_is_type3(font)
                except (IndexError, TypeError, ValueError):
                    analysis.unsafe = True
                    finish_frame()
            elif operator_name == "gs":
                try:
                    extgstates = (
                        frame.resources.get("/ExtGState") if frame.resources else None
                    )
                    graphics_state = (
                        extgstates.get(operands[0]) if extgstates is not None else None
                    )
                    if graphics_state is None:
                        raise ValueError("Missing ExtGState")
                    frame.fill_alpha = float(
                        graphics_state.get("/ca", frame.fill_alpha)
                    )
                    frame.stroke_alpha = float(
                        graphics_state.get("/CA", frame.stroke_alpha)
                    )
                    frame.blend_mode = graphics_state.get("/BM", frame.blend_mode)
                    frame.soft_mask = graphics_state.get("/SMask", frame.soft_mask)
                    font_setting = graphics_state.get("/Font")
                    if font_setting is not None:
                        font_values = list(font_setting)
                        if len(font_values) != 2 or not math.isfinite(
                            float(font_values[1])
                        ):
                            raise ValueError("Invalid ExtGState font")
                        frame.font_is_type3 = _font_is_type3(font_values[0])
                    frame.overprint = bool(
                        graphics_state.get("/OP", frame.overprint)
                        or graphics_state.get("/op", frame.overprint)
                    )
                    if not (
                        math.isfinite(frame.fill_alpha)
                        and math.isfinite(frame.stroke_alpha)
                        and 0 <= frame.fill_alpha <= 1
                        and 0 <= frame.stroke_alpha <= 1
                    ):
                        raise ValueError("Invalid alpha")
                except (IndexError, TypeError, ValueError):
                    analysis.unsafe = True
                    finish_frame()
            elif operator_name in text_show_operators:
                analysis.event += 1
                if frame.font_is_type3 is not False:
                    analysis.unsafe = True
                    finish_frame()
                    continue
                if _text_is_visible(
                    frame.render_mode,
                    frame.fill_alpha,
                    frame.stroke_alpha,
                    frame.soft_mask,
                ):
                    if (
                        frame.optional_content_depth
                        or frame.clip_unknown
                        or not _normal_blend_mode(frame.blend_mode)
                        or frame.overprint
                        or frame.transparency_group
                        or (
                            frame.soft_mask is not None
                            and str(frame.soft_mask) != "/None"
                        )
                    ):
                        analysis.unsafe = True
                        finish_frame()
                        continue
                    analysis.visible_text.append(analysis.event)
                if frame.render_mode in {4, 5, 6, 7}:
                    if frame.optional_content_depth:
                        analysis.unsafe = True
                        finish_frame()
                        continue
                    frame.clip_unknown = True
                    frame.text_clip = True
            elif operator_name in {"BMC", "BDC"}:
                if not operands:
                    analysis.unsafe = True
                    finish_frame()
                    continue
                enters_optional_content = str(operands[0]) == "/OC"
                frame.marked_content_stack.append(enters_optional_content)
                frame.optional_content_depth += int(enters_optional_content)
            elif operator_name == "EMC":
                if not frame.marked_content_stack:
                    analysis.unsafe = True
                    finish_frame()
                    continue
                frame.optional_content_depth -= int(frame.marked_content_stack.pop())
            elif operator_name in {"W", "W*"}:
                frame.clip_unknown = True
            elif operator_name in path_paint_operators or operator_name == "sh":
                if frame.optional_content_depth:
                    analysis.unsafe = True
                    finish_frame()
                    continue
                analysis.has_vector = True
            elif operator_name == "INLINE IMAGE":
                if frame.optional_content_depth or frame.text_clip:
                    analysis.unsafe = True
                    finish_frame()
                    continue
                if (
                    frame.soft_mask is not None
                    and str(frame.soft_mask) != "/None"
                    and not _soft_mask_hides_all(frame.soft_mask)
                    and frame.fill_alpha > 0
                ):
                    analysis.unsafe = True
                    finish_frame()
                    continue
                try:
                    inline_image = operands[0]
                    if bool(inline_image.image_mask):
                        analysis.unsafe = True
                        finish_frame()
                        continue
                    analysis.event += 1
                    image_is_opaque = (
                        int(inline_image.width) > 0
                        and int(inline_image.height) > 0
                        and _image_color_space_marks(
                            inline_image.obj.get("/ColorSpace"),
                            frame.resources,
                        )
                        and frame.fill_alpha == 1.0
                        and _normal_blend_mode(frame.blend_mode)
                        and (frame.soft_mask is None or str(frame.soft_mask) == "/None")
                        and not frame.overprint
                        and not frame.transparency_group
                        and not frame.clip_unknown
                    )
                except (AttributeError, IndexError, TypeError, ValueError):
                    analysis.unsafe = True
                    finish_frame()
                    continue
                if image_is_opaque:
                    coverage = _rect_coverage(frame.ctm, visible_box)
                    if coverage >= _SCAN_IMAGE_AREA_RATIO:
                        analysis.image_candidates.append((coverage, analysis.event))
            elif operator_name == "Do":
                try:
                    xobjects = (
                        frame.resources.get("/XObject") if frame.resources else None
                    )
                    xobject = (
                        xobjects.get(operands[0]) if xobjects is not None else None
                    )
                    if xobject is None:
                        raise ValueError("Missing XObject")
                    try:
                        xobject = xobject.get_object()
                    except (AttributeError, TypeError, ValueError):
                        pass
                    subtype = str(xobject.get("/Subtype"))
                    xobject_has_optional_content = xobject.get("/OC") is not None
                except (IndexError, TypeError, ValueError):
                    analysis.unsafe = True
                    finish_frame()
                    continue

                if subtype == "/Image":
                    if (
                        frame.optional_content_depth
                        or xobject_has_optional_content
                        or frame.text_clip
                    ):
                        analysis.unsafe = True
                        finish_frame()
                        continue
                    if (
                        frame.soft_mask is not None
                        and str(frame.soft_mask) != "/None"
                        and not _soft_mask_hides_all(frame.soft_mask)
                        and frame.fill_alpha > 0
                    ):
                        analysis.unsafe = True
                        finish_frame()
                        continue
                    try:
                        if bool(xobject.get("/ImageMask", False)):
                            analysis.unsafe = True
                            finish_frame()
                            continue
                        analysis.event += 1
                        width = int(xobject.get("/Width", 0))
                        height = int(xobject.get("/Height", 0))
                        image_color_space = xobject.get("/ColorSpace")
                        color_space_marks = (
                            _image_color_space_marks(image_color_space, frame.resources)
                            if image_color_space is not None
                            else _uses_jpx_decode(xobject.get("/Filter"))
                        )
                    except (TypeError, ValueError):
                        width = height = 0
                        color_space_marks = False
                    image_is_opaque = (
                        width > 0
                        and height > 0
                        and xobject.get("/Mask") is None
                        and xobject.get("/SMask") is None
                        and not bool(xobject.get("/SMaskInData", False))
                        and xobject.get("/OC") is None
                        and color_space_marks
                        and frame.fill_alpha == 1.0
                        and _normal_blend_mode(frame.blend_mode)
                        and (frame.soft_mask is None or str(frame.soft_mask) == "/None")
                        and not frame.overprint
                        and not frame.transparency_group
                        and not frame.clip_unknown
                    )
                    if image_is_opaque:
                        coverage = _rect_coverage(frame.ctm, visible_box)
                        if coverage >= _SCAN_IMAGE_AREA_RATIO:
                            analysis.image_candidates.append((coverage, analysis.event))
                    continue

                if subtype != "/Form":
                    analysis.unsafe = True
                    finish_frame()
                    continue
                key = _object_key(xobject)
                if key in active_forms:
                    analysis.unsafe = True
                    finish_frame()
                    continue
                active_forms.add(key)
                try:
                    form_matrix = xobject.get("/Matrix")
                    form_ctm = (
                        _matrix_from_operands(form_matrix) @ frame.ctm
                        if form_matrix is not None
                        else frame.ctm
                    )
                    form_clip_unknown = frame.clip_unknown
                    form_box = xobject.get("/BBox")
                    if form_box is None:
                        raise ValueError("Missing Form BBox")
                    box_values = [float(value) for value in form_box]
                    if len(box_values) != 4:
                        raise ValueError("Invalid Form BBox")
                    box_matrix = pikepdf.Matrix(
                        box_values[2] - box_values[0],
                        0,
                        0,
                        box_values[3] - box_values[1],
                        box_values[0],
                        box_values[1],
                    )
                    if (
                        _rect_coverage(box_matrix @ form_ctm, visible_box)
                        < _FULL_PAGE_IMAGE_AREA_RATIO
                    ):
                        form_clip_unknown = True
                    form_resources = xobject.get("/Resources", frame.resources)
                    group = xobject.get("/Group")
                    if group is not None and str(group.get("/S")) != "/Transparency":
                        raise ValueError("Invalid Form Group")
                    instructions = list(pikepdf.parse_content_stream(xobject))
                except (TypeError, ValueError):
                    active_forms.remove(key)
                    analysis.unsafe = True
                    finish_frame()
                    continue
                except BaseException:
                    active_forms.remove(key)
                    raise

                frames.append(
                    _ContentAnalysisFrame(
                        instructions,
                        form_resources,
                        form_ctm,
                        frame.render_mode,
                        frame.font_is_type3,
                        frame.fill_alpha,
                        frame.stroke_alpha,
                        frame.blend_mode,
                        frame.soft_mask,
                        frame.overprint,
                        form_clip_unknown,
                        frame.text_clip,
                        int(
                            bool(
                                frame.optional_content_depth
                                or xobject_has_optional_content
                            )
                        ),
                        frame.transparency_group
                        or (
                            group is not None
                            and str(group.get("/S")) == "/Transparency"
                        ),
                        key,
                    )
                )
    finally:
        for frame in frames:
            if frame.active_form_key is not None:
                active_forms.discard(frame.active_form_key)


def _page_paint_analysis(page: "pikepdf.Page") -> _PagePaintAnalysis:
    """Return conservative paint facts, failing closed for malformed content."""
    import pikepdf

    analysis = _PagePaintAnalysis()
    try:
        visible_box_values = tuple(float(value) for value in page.cropbox)
        if (
            len(visible_box_values) != 4
            or not all(math.isfinite(value) for value in visible_box_values)
            or visible_box_values[2] <= visible_box_values[0]
            or visible_box_values[3] <= visible_box_values[1]
        ):
            raise ValueError("Invalid CropBox")
        group = page.obj.get("/Group")
        if group is not None and str(group.get("/S")) != "/Transparency":
            raise ValueError("Invalid Page Group")
        _analyze_content(
            page,
            page.resources,
            visible_box_values,
            analysis,
            ctm=pikepdf.Matrix(),
            transparency_group=group is not None,
            active_forms=set(),
        )
    except Exception as exc:
        log_suppressed_error(
            logger,
            exc,
            "Could not inspect page content before deskewing: %s",
            exc,
        )
        analysis.unsafe = True
    return analysis


def _plan_deskew_ocr(
    pdf_path: Path,
    *,
    annotated_pages: frozenset[int] | None = None,
) -> _DeskewPlan | None:
    """Return disjoint page sets for regular OCR and scan-only deskew OCR.

    ``None`` indicates that inspection failed and deskewing should be disabled.
    """
    import pikepdf
    from ocrmypdf.pdfinfo import PdfInfo

    try:
        pdfinfo = PdfInfo(pdf_path, max_workers=1)
        with pikepdf.open(pdf_path) as pdf:
            if len(pdfinfo) != len(pdf.pages):
                raise OCRError("Page count changed during deskew analysis")

            deskew_pages = []
            regular_ocr_pages = []
            redo_ocr_pages = []
            strip_text_pages = []
            ambiguous_scan_pages = []
            annotated_scan_pages = 0
            for page_number, (page_info, page) in enumerate(
                zip(pdfinfo.pages, pdf.pages, strict=True),
                start=1,
            ):
                if page_info is None:
                    continue

                analysis = _page_paint_analysis(page)
                safe_raster_content = not (
                    page_info.has_vector or analysis.has_vector or analysis.unsafe
                )
                has_renderable_image = any(
                    image.renderable for image in page_info.images
                )
                full_page_image_event = max(
                    (
                        event
                        for coverage, event in analysis.image_candidates
                        if coverage >= _FULL_PAGE_IMAGE_AREA_RATIO
                    ),
                    default=-1,
                )
                if not safe_raster_content or not has_renderable_image:
                    if page_info.has_text and full_page_image_event >= 0:
                        ambiguous_scan_pages.append(page_number)
                    elif not page_info.has_text:
                        regular_ocr_pages.append(page_number)
                    continue

                has_visible_text = any(
                    event > full_page_image_event for event in analysis.visible_text
                )
                if full_page_image_event >= 0 and has_visible_text:
                    redo_ocr_pages.append(page_number)
                elif analysis.image_candidates and not has_visible_text:
                    page_has_annotations = (
                        page_number in annotated_pages
                        if annotated_pages is not None
                        else (
                            (annots := page.obj.get("/Annots")) is not None
                            and len(annots) > 0
                        )
                    )
                    if page_has_annotations:
                        annotated_scan_pages += 1
                        regular_ocr_pages.append(page_number)
                        if page_info.has_text:
                            strip_text_pages.append(page_number)
                        continue
                    deskew_pages.append(page_number)
                    if page_info.has_text:
                        strip_text_pages.append(page_number)
                elif not page_info.has_text and not has_visible_text:
                    regular_ocr_pages.append(page_number)

        logger.info(
            "OCR planning selected %d deskew and %d mixed-content redo page(s) "
            "from %d page(s)",
            len(deskew_pages),
            len(redo_ocr_pages),
            len(pdfinfo),
        )
        if annotated_scan_pages:
            logger.warning(
                "Deskew skipped for %d annotated scan-like page(s)",
                annotated_scan_pages,
            )
        return _DeskewPlan(
            tuple(deskew_pages),
            tuple(regular_ocr_pages),
            tuple(redo_ocr_pages),
            tuple(strip_text_pages),
            tuple(ambiguous_scan_pages),
        )
    except Exception as exc:
        log_suppressed_error(
            logger,
            exc,
            "Could not inspect pages before deskewing %s: %s",
            pdf_path,
            exc,
        )
        return None


def _find_deskew_pages(pdf_path: Path) -> list[int] | None:
    """Return one-based scan-like page numbers safe to deskew."""
    plan = _plan_deskew_ocr(pdf_path)
    return None if plan is None else list(plan.deskew_pages)


def _private_resources(
    resources: "pikepdf.Object | None",
) -> "pikepdf.Object":
    """Copy a resource dictionary and its XObject mapping for page-local edits."""
    import pikepdf

    private = copy.copy(resources) if resources is not None else pikepdf.Dictionary()
    xobjects = resources.get("/XObject") if resources is not None else None
    if xobjects is not None:
        private[pikepdf.Name.XObject] = copy.copy(xobjects)
    return private


def _strip_text_show_operators(
    pdf: "pikepdf.Pdf",
    container: "pikepdf.Object",
    resources: "pikepdf.Object",
    active_forms: set[_ObjectKey],
    *,
    is_page: bool = False,
) -> None:
    """Remove text-show operators, cloning referenced Forms before editing."""
    import pikepdf

    operations: list[tuple[object, ...]] = [("process", container, resources, is_page)]
    added_forms: list[_ObjectKey] = []
    try:
        while operations:
            operation = operations.pop()
            action = operation[0]
            if action == "leave":
                _, key, xobjects, name, form = operation
                xobjects[name] = form  # type: ignore[index]
                active_forms.remove(key)  # type: ignore[arg-type]
                added_forms.pop()
                continue
            if action == "rewrite":
                _, current, instructions, current_is_page = operation
                rewritten = pikepdf.unparse_content_stream(
                    [
                        instruction
                        for instruction in instructions  # type: ignore[union-attr]
                        if str(instruction[1]) not in {"Tj", "TJ", "'", '"'}
                    ]
                )
                if current_is_page:
                    current.obj[pikepdf.Name.Contents] = pdf.make_stream(  # type: ignore[union-attr]
                        rewritten
                    )
                else:
                    current.write(rewritten)  # type: ignore[union-attr]
                continue
            if action == "enter":
                _, inherited_resources, xobjects, name = operation
                xobject = xobjects.get(name)  # type: ignore[union-attr]
                if xobject is None:
                    continue
                try:
                    xobject = xobject.get_object()
                except (AttributeError, TypeError, ValueError):
                    pass
                if str(xobject.get("/Subtype")) != "/Form":
                    continue

                key = _object_key(xobject)
                if key in active_forms:
                    raise OCRError("Could not remove the existing OCR text layer")
                active_forms.add(key)
                added_forms.append(key)
                form = copy.copy(xobject)
                form_resources = _private_resources(
                    xobject.get("/Resources", inherited_resources)
                )
                form[pikepdf.Name.Resources] = form_resources
                operations.append(("leave", key, xobjects, name, form))
                operations.append(("process", form, form_resources, False))
                continue

            _, current, current_resources, current_is_page = operation
            try:
                instructions = list(pikepdf.parse_content_stream(current))
            except (pikepdf.PdfError, TypeError, ValueError) as exc:
                raise OCRError("Could not remove the existing OCR text layer") from exc

            operations.append(("rewrite", current, instructions, current_is_page))
            xobjects = current_resources.get("/XObject")
            if xobjects is None:
                continue
            form_calls = [
                operands[0]
                for operands, operator in instructions
                if str(operator) == "Do" and operands
            ]
            operations.extend(
                ("enter", current_resources, xobjects, name)
                for name in reversed(form_calls)
            )
    finally:
        for key in reversed(added_forms):
            active_forms.discard(key)


def _prepare_deskew_input(
    input_path: Path,
    output_path: Path,
    pages: tuple[int, ...],
) -> None:
    """Create a copy without text-showing operators on the selected image pages."""
    import pikepdf

    try:
        with pikepdf.open(input_path) as pdf:
            for page_number in pages:
                page = pdf.pages[page_number - 1]
                resources = _private_resources(page.resources)
                page.obj[pikepdf.Name.Resources] = resources
                _strip_text_show_operators(
                    pdf,
                    page,
                    resources,
                    set(),
                    is_page=True,
                )
            pdf.save(output_path)
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError("Could not remove the existing OCR text layer") from exc


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


def _write_json_atomic(output_path: Path, value: dict[str, Any]) -> None:
    """Write UTF-8 JSON without exposing a partially written manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            file_descriptor = -1
            json.dump(
                value,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)


def _manifest_mapping(
    value: Any,
    expected_keys: frozenset[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OCRError(f"Invalid OCR manifest {context}: expected an object")
    keys = frozenset(value)
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        unexpected = sorted(keys - expected_keys)
        raise OCRError(
            f"Invalid OCR manifest {context}: missing keys {missing}, "
            f"unexpected keys {unexpected}"
        )
    return value


def _manifest_number(
    value: Any,
    context: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OCRError(f"Invalid OCR manifest {context}: expected a number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise OCRError(f"Invalid OCR manifest {context}: invalid number") from exc
    if not math.isfinite(number) or positive and number <= 0:
        raise OCRError(f"Invalid OCR manifest {context}: invalid number")
    return number


def _validate_manifest_bbox(
    value: Any,
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
    parent: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    bbox = _manifest_mapping(
        value,
        frozenset({"left", "top", "right", "bottom"}),
        context,
    )
    left = _manifest_number(bbox["left"], f"{context}.left")
    top = _manifest_number(bbox["top"], f"{context}.top")
    right = _manifest_number(bbox["right"], f"{context}.right")
    bottom = _manifest_number(bbox["bottom"], f"{context}.bottom")
    if right <= left or bottom <= top:
        raise OCRError(f"Invalid OCR manifest {context}: empty bounding box")
    coordinates = (left, top, right, bottom)
    tolerance = _OCR_MANIFEST_GEOMETRY_TOLERANCE
    if bounds is not None:
        width, height = bounds
        if (
            left < -tolerance
            or top < -tolerance
            or right > width + tolerance
            or bottom > height + tolerance
        ):
            raise OCRError(f"Invalid OCR manifest {context}: outside page coordinates")
    if parent is not None and (
        left < parent[0] - tolerance
        or top < parent[1] - tolerance
        or right > parent[2] + tolerance
        or bottom > parent[3] + tolerance
    ):
        raise OCRError(f"Invalid OCR manifest {context}: outside parent line")
    return coordinates


def _validate_manifest_polygon(
    value: Any,
    context: str,
    *,
    bbox: tuple[float, float, float, float],
    bounds: tuple[float, float],
) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise OCRError(f"Invalid OCR manifest {context}: expected four points")
    points = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise OCRError(f"Invalid OCR manifest {context}[{index}]: expected a point")
        points.append(
            (
                _manifest_number(point[0], f"{context}[{index}][0]"),
                _manifest_number(point[1], f"{context}[{index}][1]"),
            )
        )
    area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                points,
                [*points[1:], points[0]],
                strict=True,
            )
        )
    )
    if area <= 1e-6:
        raise OCRError(f"Invalid OCR manifest {context}: empty polygon")
    tolerance = _OCR_MANIFEST_GEOMETRY_TOLERANCE
    width, height = bounds
    if any(
        x < -tolerance
        or y < -tolerance
        or x > width + tolerance
        or y > height + tolerance
        for x, y in points
    ):
        raise OCRError(f"Invalid OCR manifest {context}: outside page coordinates")
    polygon_bbox = (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )
    if any(
        not math.isclose(
            actual,
            expected,
            rel_tol=1e-9,
            abs_tol=tolerance,
        )
        for actual, expected in zip(polygon_bbox, bbox, strict=True)
    ):
        raise OCRError(
            f"Invalid OCR manifest {context}: polygon does not match bounding box"
        )


def _validate_manifest_element(
    value: Any,
    context: str,
    *,
    bounds: tuple[float, float],
    parent_bbox: tuple[float, float, float, float] | None = None,
    line_index: int | None = None,
    word_index: int | None = None,
    expected_mcid: int | None = None,
) -> None:
    common_keys = {
        "ocr_class",
        "text",
        "confidence",
        "bbox",
        "polygon",
        "language",
        "direction",
        "baseline",
        "text_angle",
    }
    expected_keys = set(common_keys)
    if line_index is not None:
        expected_keys.update({"mcid", "words"})
    else:
        expected_keys.add("index")
    element = _manifest_mapping(value, frozenset(expected_keys), context)

    expected_class = "ocr_line" if line_index is not None else "ocrx_word"
    if element["ocr_class"] != expected_class:
        raise OCRError(
            f"Invalid OCR manifest {context}.ocr_class: expected {expected_class}"
        )
    if not isinstance(element["text"], str) or not element["text"]:
        raise OCRError(f"Invalid OCR manifest {context}.text: expected text")

    confidence = element["confidence"]
    if confidence is not None:
        confidence_value = _manifest_number(confidence, f"{context}.confidence")
        if not 0.0 <= confidence_value <= 1.0:
            raise OCRError(f"Invalid OCR manifest {context}.confidence: outside 0..1")

    bbox = _validate_manifest_bbox(
        element["bbox"],
        f"{context}.bbox",
        bounds=bounds,
        parent=parent_bbox,
    )
    _validate_manifest_polygon(
        element["polygon"],
        f"{context}.polygon",
        bbox=bbox,
        bounds=bounds,
    )
    language = element["language"]
    if language is not None and (not isinstance(language, str) or not language):
        raise OCRError(f"Invalid OCR manifest {context}.language")
    if element["direction"] not in {None, "ltr", "rtl"}:
        raise OCRError(f"Invalid OCR manifest {context}.direction")

    baseline = element["baseline"]
    if baseline is not None:
        baseline = _manifest_mapping(
            baseline,
            frozenset({"slope", "intercept"}),
            f"{context}.baseline",
        )
        _manifest_number(baseline["slope"], f"{context}.baseline.slope")
        _manifest_number(baseline["intercept"], f"{context}.baseline.intercept")
    text_angle = element["text_angle"]
    if text_angle is not None:
        _manifest_number(text_angle, f"{context}.text_angle")

    if line_index is not None:
        mcid = element["mcid"]
        if isinstance(mcid, bool) or not isinstance(mcid, int) or mcid < 0:
            raise OCRError(f"Invalid OCR manifest {context}.mcid")
        if expected_mcid is not None and mcid != expected_mcid:
            raise OCRError(
                f"Invalid OCR manifest {context}.mcid: expected {expected_mcid}"
            )
        words = element["words"]
        if not isinstance(words, list) or not words:
            raise OCRError(f"Invalid OCR manifest {context}.words")
        for index, word in enumerate(words):
            _validate_manifest_element(
                word,
                f"{context}.words[{index}]",
                bounds=bounds,
                parent_bbox=bbox,
                word_index=index,
            )
    elif (
        isinstance(element["index"], bool)
        or not isinstance(element["index"], int)
        or element["index"] != word_index
    ):
        raise OCRError(f"Invalid OCR manifest {context}.index: expected {word_index}")


def _validate_ocr_page_manifest(
    value: Any,
    context: str,
    *,
    document_page: bool = False,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "type",
        "page_index",
        "raster",
        "coordinates",
        "languages",
        "layout",
        "lines",
    }
    if document_page:
        expected_keys.add("form_name")
    page = _manifest_mapping(value, frozenset(expected_keys), context)
    if (
        isinstance(page["schema_version"], bool)
        or page["schema_version"] != _OCR_MANIFEST_SCHEMA_VERSION
    ):
        raise OCRError(f"Invalid OCR manifest {context}.schema_version")
    if page["type"] != _OCR_PAGE_MANIFEST_TYPE:
        raise OCRError(f"Invalid OCR manifest {context}.type")
    page_index = page["page_index"]
    if (
        isinstance(page_index, bool)
        or not isinstance(page_index, int)
        or page_index < 0
    ):
        raise OCRError(f"Invalid OCR manifest {context}.page_index")

    raster = _manifest_mapping(
        page["raster"],
        frozenset({"width", "height", "dpi"}),
        f"{context}.raster",
    )
    for dimension in ("width", "height"):
        if (
            isinstance(raster[dimension], bool)
            or not isinstance(raster[dimension], int)
            or raster[dimension] <= 0
        ):
            raise OCRError(f"Invalid OCR manifest {context}.raster.{dimension}")
    raster_dpi = _manifest_number(
        raster["dpi"],
        f"{context}.raster.dpi",
        positive=True,
    )

    coordinates = _manifest_mapping(
        page["coordinates"],
        frozenset({"width", "height", "dpi", "scale_from_raster"}),
        f"{context}.coordinates",
    )
    coordinate_values = {}
    for name in ("width", "height", "dpi", "scale_from_raster"):
        coordinate_values[name] = _manifest_number(
            coordinates[name],
            f"{context}.coordinates.{name}",
            positive=True,
        )
    scale = coordinate_values["scale_from_raster"]
    expected_coordinates = {
        "width": raster["width"] * scale,
        "height": raster["height"] * scale,
        "dpi": raster_dpi * scale,
    }
    if any(
        not math.isclose(
            coordinate_values[name],
            expected,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for name, expected in expected_coordinates.items()
    ):
        raise OCRError(f"Invalid OCR manifest {context}.coordinates: inconsistent")
    coordinate_bounds = (
        coordinate_values["width"],
        coordinate_values["height"],
    )

    languages = page["languages"]
    if (
        not isinstance(languages, list)
        or not languages
        or any(not isinstance(language, str) or not language for language in languages)
    ):
        raise OCRError(f"Invalid OCR manifest {context}.languages")

    layout = _manifest_mapping(
        page["layout"],
        frozenset({"reading_order_applied", "sections", "selected_columns"}),
        f"{context}.layout",
    )
    if not isinstance(layout["reading_order_applied"], bool):
        raise OCRError(f"Invalid OCR manifest {context}.layout.reading_order_applied")
    sections = layout["sections"]
    if not isinstance(sections, list):
        raise OCRError(f"Invalid OCR manifest {context}.layout.sections")
    for section_index, section_value in enumerate(sections):
        section_context = f"{context}.layout.sections[{section_index}]"
        section = _manifest_mapping(
            section_value,
            frozenset({"regions", "columns"}),
            section_context,
        )
        for collection_name in ("regions", "columns"):
            collection = section[collection_name]
            if not isinstance(collection, list):
                raise OCRError(
                    f"Invalid OCR manifest {section_context}.{collection_name}"
                )
            for region_index, region in enumerate(collection):
                _validate_manifest_bbox(
                    region,
                    f"{section_context}.{collection_name}[{region_index}]",
                    bounds=coordinate_bounds,
                )
    selected_columns = layout["selected_columns"]
    if not isinstance(selected_columns, list) or not selected_columns:
        raise OCRError(f"Invalid OCR manifest {context}.layout.selected_columns")
    for column_index, column in enumerate(selected_columns):
        _validate_manifest_bbox(
            column,
            f"{context}.layout.selected_columns[{column_index}]",
            bounds=coordinate_bounds,
        )

    lines = page["lines"]
    if not isinstance(lines, list):
        raise OCRError(f"Invalid OCR manifest {context}.lines")
    # Page sidecars are written before rendering, so their MCIDs are still the
    # line indexes. Document manifests are reconciled against the rendered OCR
    # Forms, which drops suppressed lines and leaves gaps in the MCID sequence.
    previous_mcid = -1
    for line_index, line in enumerate(lines):
        _validate_manifest_element(
            line,
            f"{context}.lines[{line_index}]",
            bounds=coordinate_bounds,
            line_index=line_index,
            expected_mcid=None if document_page else line_index,
        )
        mcid = line["mcid"]
        if mcid <= previous_mcid:
            raise OCRError(
                f"Invalid OCR manifest {context}.lines[{line_index}].mcid: "
                "expected a strictly increasing MCID"
            )
        previous_mcid = mcid

    if document_page:
        form_name = page["form_name"]
        if not isinstance(form_name, str) or not form_name.startswith("/OCR-"):
            raise OCRError(f"Invalid OCR manifest {context}.form_name")
    return page


def _validate_ocr_document_manifest(
    value: Any,
    context: str,
    *,
    expected_page_count: int | None = None,
) -> dict[str, Any]:
    document = _manifest_mapping(
        value,
        frozenset({"schema_version", "type", "page_count", "languages", "pages"}),
        context,
    )
    if (
        isinstance(document["schema_version"], bool)
        or document["schema_version"] != _OCR_MANIFEST_SCHEMA_VERSION
    ):
        raise OCRError(f"Invalid OCR manifest {context}.schema_version")
    if document["type"] != _OCR_DOCUMENT_MANIFEST_TYPE:
        raise OCRError(f"Invalid OCR manifest {context}.type")

    page_count = document["page_count"]
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 0
        or expected_page_count is not None
        and page_count != expected_page_count
    ):
        raise OCRError(f"Invalid OCR manifest {context}.page_count")

    languages = document["languages"]
    if (
        not isinstance(languages, list)
        or not languages
        or any(not isinstance(language, str) or not language for language in languages)
        or len(set(languages)) != len(languages)
    ):
        raise OCRError(f"Invalid OCR manifest {context}.languages")

    pages = document["pages"]
    if not isinstance(pages, list):
        raise OCRError(f"Invalid OCR manifest {context}.pages")
    page_indexes = []
    for index, page_value in enumerate(pages):
        page = _validate_ocr_page_manifest(
            page_value,
            f"{context}.pages[{index}]",
            document_page=True,
        )
        page_index = page["page_index"]
        if page_index >= page_count or page["languages"] != languages:
            raise OCRError(f"Invalid OCR manifest {context}.pages[{index}]")
        page_indexes.append(page_index)
    if page_indexes != sorted(set(page_indexes)):
        raise OCRError(f"Invalid OCR manifest {context}.pages: invalid page order")
    return document


def _read_ocr_run_sidecars(directory: Path) -> dict[int, dict[str, Any]]:
    """Read and validate one isolated OCRmyPDF run's page sidecars."""
    pages = {}
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        raise OCRError(f"Could not read OCR manifest sidecars: {exc}") from exc
    for entry in entries:
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
            raise OCRError(f"Invalid OCR manifest sidecar entry: {entry.name}")
        try:
            value = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OCRError(
                f"Could not read OCR manifest sidecar {entry.name}: {exc}"
            ) from exc
        page = _validate_ocr_page_manifest(value, entry.name)
        page_index = page["page_index"]
        if entry.name != f"page-{page_index:06d}.json":
            raise OCRError(
                f"Invalid OCR manifest sidecar filename for page {page_index}"
            )
        if page_index in pages:
            raise OCRError(f"Duplicate OCR manifest page {page_index}")
        pages[page_index] = page
    return pages


def _emitted_form_mcids(form, context: str) -> set[int]:
    """Return the MCIDs the renderer actually emitted into an OCR Form."""
    import pikepdf

    try:
        instructions = pikepdf.parse_content_stream(form)
    except Exception as exc:
        raise OCRError(f"Could not parse {context}: {exc}") from exc

    mcids: set[int] = set()
    for instruction in instructions:
        if not isinstance(instruction, pikepdf.ContentStreamInstruction):
            continue
        operands = instruction.operands
        if str(instruction.operator) != "BDC" or len(operands) < 2:
            continue
        properties = operands[1]
        if not isinstance(properties, pikepdf.Dictionary):
            raise OCRError(f"Unexpected marked content properties in {context}")
        if "/MCID" not in properties:
            continue
        mcid = properties["/MCID"]
        try:
            mcid = int(mcid)
        except (TypeError, ValueError) as exc:
            raise OCRError(f"Invalid marked content MCID in {context}") from exc
        if mcid < 0 or mcid in mcids:
            raise OCRError(f"Invalid marked content MCID in {context}")
        mcids.add(mcid)
    return mcids


def _reconcile_manifest_lines(page, manifest_page: dict[str, Any]) -> None:
    """Drop manifest lines the renderer suppressed instead of emitting them.

    The fpdf2 renderer discards implausible OCR lines (for example when the
    aspect-ratio check fails), so the page sidecar can declare more lines than
    the rendered OCR Form contains. The manifest must describe the PDF as it
    was written, not as it was planned.
    """
    page_index = manifest_page["page_index"]
    form_name = manifest_page["form_name"]
    context = f"OCR Form {form_name} on page {page_index}"
    try:
        xobjects = page.resources.get("/XObject")
        form = xobjects[form_name] if xobjects is not None else None
    except (AttributeError, KeyError):
        form = None
    if form is None:
        raise OCRError(f"Could not read {context}")

    emitted = _emitted_form_mcids(form, context)
    declared = {line["mcid"] for line in manifest_page["lines"]}
    unexpected = sorted(emitted - declared)
    if unexpected:
        raise OCRError(f"{context} contains undeclared MCIDs: {unexpected}")

    suppressed = sorted(declared - emitted)
    if not suppressed:
        return
    manifest_page["lines"] = [
        line for line in manifest_page["lines"] if line["mcid"] in emitted
    ]
    logger.warning(
        "OCR page %d: %d line(s) were suppressed by the renderer and removed "
        "from the OCR manifest (MCIDs %s)",
        page_index,
        len(suppressed),
        suppressed,
    )


def _reconciled_document_pages(
    pdf,
    languages: list[str],
    pages: dict[int, dict[str, Any]],
    form_names: dict[int, tuple[str, ...]],
    page_count: int,
) -> list[dict[str, Any]]:
    """Return document manifest pages reconciled against the rendered PDF."""
    document_pages = []
    for page_index in sorted(pages):
        if page_index >= page_count:
            raise OCRError(f"OCR manifest page {page_index} is outside the PDF")
        page_form_names = form_names[page_index]
        if len(page_form_names) != 1:
            raise OCRError(
                f"OCR manifest page {page_index} has {len(page_form_names)} "
                "new OCR Forms"
            )
        page = copy.deepcopy(pages[page_index])
        if page["languages"] != languages:
            raise OCRError(
                f"OCR manifest page {page_index} languages do not match the run"
            )
        page["form_name"] = page_form_names[0]
        _reconcile_manifest_lines(pdf.pages[page_index], page)
        _validate_ocr_page_manifest(
            page,
            f"pages[{page_index}]",
            document_page=True,
        )
        document_pages.append(page)
    return document_pages


def _write_ocr_document_manifest(
    output_path: Path,
    pdf_path: Path,
    languages: list[str],
    pages: dict[int, dict[str, Any]],
    form_names: dict[int, tuple[str, ...]],
) -> None:
    """Merge validated page sidecars with their final OCR Form names."""
    import pikepdf

    manifest_page_indexes = frozenset(pages)
    form_page_indexes = frozenset(form_names)
    if manifest_page_indexes != form_page_indexes:
        raise OCRError(
            "OCR manifest pages do not match final OCR Forms: "
            f"pages={sorted(manifest_page_indexes)}, "
            f"forms={sorted(form_page_indexes)}"
        )

    with pikepdf.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        document_pages = _reconciled_document_pages(
            pdf,
            languages,
            pages,
            form_names,
            page_count,
        )

    document = {
        "schema_version": _OCR_MANIFEST_SCHEMA_VERSION,
        "type": _OCR_DOCUMENT_MANIFEST_TYPE,
        "page_count": page_count,
        "languages": list(languages),
        "pages": document_pages,
    }
    _validate_ocr_document_manifest(
        document,
        "document",
        expected_page_count=page_count,
    )
    _write_json_atomic(output_path, document)
    try:
        serialized = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_ocr_document_manifest(
            serialized,
            output_path.name,
            expected_page_count=page_count,
        )
    except OCRError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OCRError(f"Could not revalidate OCR document manifest: {exc}") from exc


def _finalize_ocr_output(
    pdf_path: Path,
    languages: list[str],
    existing_ocr_form_names: list[frozenset[str]],
    *,
    strip_existing_ocr_text: bool = False,
) -> dict[int, tuple[str, ...]]:
    """Set OCR metadata and finalize OCR Form XObjects."""
    import pikepdf

    changed = False
    new_ocr_form_names: dict[int, list[str]] = {}
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        if "/Lang" not in pdf.Root:
            language = _PADDLE_LANGUAGE_TAGS.get(languages[0], languages[0])
            pdf.Root[pikepdf.Name.Lang] = pikepdf.String(language)
            changed = True

        for page_index, page in enumerate(pdf.pages):
            existing_names = (
                existing_ocr_form_names[page_index]
                if page_index < len(existing_ocr_form_names)
                else frozenset()
            )
            xobjects = page.resources.get("/XObject")
            if xobjects is None:
                continue

            for name, xobject in xobjects.items():
                if not str(name).startswith("/OCR-"):
                    continue
                is_existing = str(name) in existing_names
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
                new_ocr_form_names.setdefault(page_index, []).append(str(name))
                rotation = _validated_ocr_page_rotation(page, page_index + 1)
                try:
                    media_box = [float(value) for value in page.mediabox]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise OCRError(
                        f"OCR page {page_index + 1} has an invalid MediaBox"
                    ) from exc
                if len(media_box) != 4 or not all(
                    math.isfinite(value) for value in media_box
                ):
                    raise OCRError(f"OCR page {page_index + 1} has an invalid MediaBox")
                width = media_box[2] - media_box[0]
                height = media_box[3] - media_box[1]
                if width <= 0 or height <= 0:
                    raise OCRError(f"OCR page {page_index + 1} has an invalid MediaBox")
                form_width, form_height = (
                    (height, width) if rotation in {90, 270} else (width, height)
                )
                expected_box = (0.0, 0.0, form_width, form_height)
                box = xobject.get("/BBox")
                try:
                    current_box = (
                        tuple(float(value) for value in box)
                        if box is not None and len(box) == 4
                        else ()
                    )
                except (TypeError, ValueError, OverflowError):
                    current_box = ()
                if current_box != expected_box:
                    xobject[pikepdf.Name.BBox] = pikepdf.Array(expected_box)
                    changed = True

        if changed:
            pdf.save(pdf_path)
    return {
        page_index: tuple(sorted(names))
        for page_index, names in new_ocr_form_names.items()
    }


def is_ocr_available() -> bool:
    """Checks if OCR functionality is available.

    Returns:
        True if ocrmypdf is installed, False otherwise.
    """
    return HAS_OCR


def _validate_ocr_content_work_budget(
    pdf: "pikepdf.Pdf",
    *,
    force: bool = False,
) -> None:
    """Validate canonical PDF content before any OCR-specific content walk."""
    import pikepdf

    from .digital_layout import (
        _SERIALIZED_PDF_MEMORY_LIMIT,
        _DecodedContentBudget,
        _validate_content_work_budget,
    )

    if len(pdf.pages) > _OCR_MAX_DOCUMENT_PAGES:
        raise OCRError(
            f"OCR input contains more than {_OCR_MAX_DOCUMENT_PAGES:,} pages"
        )
    try:
        with SpooledTemporaryFile(
            max_size=_SERIALIZED_PDF_MEMORY_LIMIT,
            mode="w+b",
        ) as serialized:
            pdf.save(serialized, compress_streams=False)
            serialized.seek(0)
            with pikepdf.open(serialized) as canonical_pdf:
                if force:
                    _include_forced_ocr_forms_in_content_preflight(canonical_pdf)
                _validate_content_work_budget(
                    canonical_pdf,
                    frozenset(range(len(canonical_pdf.pages))),
                    _DecodedContentBudget(),
                )
    except OCRError:
        raise
    except ConversionError as exc:
        raise OCRError(f"OCR content-stream preflight failed: {exc}") from exc
    except Exception as exc:
        raise OCRError(f"Could not preflight OCR content: {exc}") from exc


def needs_ocr(pdf: "pikepdf.Pdf", *, threshold: float = 0.5) -> bool:
    """Analyzes whether a PDF needs OCR.

    Public library helper; not called by the conversion pipeline itself
    (OCR is opt-in via an explicit model-directory pair and
    ``ocr_languages``/``--ocr``). Use it to decide programmatically whether
    to enable OCR for a document.

    Checks each page for the presence of images without recognizable text.
    A page is considered to need OCR if it contains images but has no
    non-whitespace text operands in the content stream.

    Args:
        pdf: The pikepdf.Pdf object to analyze.
        threshold: Proportion of pages that must need OCR (0.0-1.0).
            Default: 0.5 (50% of pages).

    Returns:
        True if at least `threshold` of the pages need OCR.
    """
    if len(pdf.pages) == 0:
        return False

    _validate_ocr_content_work_budget(pdf)

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
    """Checks if a page contains non-whitespace text operators.

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

    form_calls = []
    try:
        for operands, operator in pikepdf.parse_content_stream(page):
            operator_name = str(operator)
            if operator_name in text_operators and _text_show_has_content(
                operands, operator_name
            ):
                return True
            if operator_name == "Do" and len(operands) == 1:
                form_calls.append(operands[0])
    except Exception as e:
        log_suppressed_error(logger, e, "Error during text analysis: %s", e)

    # Check only invoked Form XObjects; unused resources do not paint page text.
    try:
        resources = page.resources
        if not isinstance(resources, pikepdf.Dictionary):
            return False
        xobjects = resources.get("/XObject")
        if not isinstance(xobjects, pikepdf.Dictionary):
            return False

        visited: set[tuple[_ObjectKey, _ObjectKey]] = set()
        for name in form_calls:
            try:
                xobj = xobjects[name].get_object()
            except (AttributeError, TypeError, ValueError):
                xobj = xobjects.get(name)
            if _form_xobject_has_text(
                xobj,
                resources,
                text_operators,
                visited,
            ):
                return True
    except Exception as e:
        log_suppressed_error(logger, e, "Error checking XObjects for text: %s", e)

    return False


def _form_xobject_has_text(
    xobj: "pikepdf.Object",
    inherited_resources: "pikepdf.Object",
    text_operators: frozenset[str],
    visited: set[tuple[_ObjectKey, _ObjectKey]],
) -> bool:
    """Check a Form XObject and its descendants for text operators.

    Args:
        xobj: The XObject to check.
        inherited_resources: Resources inherited by a Form without its own.
        text_operators: Set of PDF text operator names.
        visited: Set of already-visited Form/resource contexts to prevent cycles.

    Returns:
        True if the Form XObject (or nested Form XObjects) contains text.
    """
    import pikepdf

    pending = [(xobj, inherited_resources)]
    while pending:
        current, inherited = pending.pop()
        try:
            subtype = current.get("/Subtype")
            if subtype is None or str(subtype) != "/Form":
                continue
        except Exception:
            continue

        resources = current.get("/Resources")
        if not isinstance(resources, pikepdf.Dictionary) or not resources:
            resources = inherited
        context = (_object_key(current), _object_key(resources))
        if context in visited:
            continue
        visited.add(context)

        try:
            form_calls = []
            for operands, operator in pikepdf.parse_content_stream(current):
                operator_name = str(operator)
                if operator_name in text_operators and _text_show_has_content(
                    operands, operator_name
                ):
                    return True
                if operator_name == "Do" and len(operands) == 1:
                    form_calls.append(operands[0])
        except Exception as e:
            log_suppressed_error(
                logger, e, "Error parsing Form XObject content stream: %s", e
            )
            continue

        try:
            if not isinstance(resources, pikepdf.Dictionary):
                continue
            nested_xobjects = resources.get("/XObject")
            if not isinstance(nested_xobjects, pikepdf.Dictionary):
                continue

            nested_forms = []
            for name in form_calls:
                try:
                    nested = nested_xobjects[name].get_object()
                except (AttributeError, TypeError, ValueError):
                    nested = nested_xobjects.get(name)
                nested_forms.append((nested, resources))
            pending.extend(reversed(nested_forms))
        except Exception as e:
            log_suppressed_error(logger, e, "Error checking nested XObjects: %s", e)

    return False


def _text_show_has_content(operands: object, operator: str) -> bool:
    """Return whether a text-showing operation contains non-whitespace bytes."""
    import pikepdf

    try:
        values = operands[0] if operator == "TJ" else (operands[-1],)  # type: ignore[index]
        for value in values:
            if getattr(value, "_type_code", None) != pikepdf.ObjectType.string:
                if operator == "TJ" and isinstance(value, Number):
                    continue
                return True
            if bytes(value).strip(b" \t\n\f\r"):
                return True
        return False
    except (IndexError, TypeError, ValueError):
        return True


def _find_whitespace_only_text_pages(pdf_path: Path) -> tuple[int, ...]:
    """Return image pages OCRmyPDF misclassifies because text is only whitespace."""
    import pikepdf
    from ocrmypdf.pdfinfo import PdfInfo

    try:
        pdfinfo = PdfInfo(pdf_path, max_workers=1)
        with pikepdf.open(pdf_path) as pdf:
            if len(pdfinfo) != len(pdf.pages):
                raise OCRError("Page count changed during OCR text analysis")
            pages = tuple(
                page_number
                for page_number, (page_info, page) in enumerate(
                    zip(pdfinfo.pages, pdf.pages, strict=True),
                    start=1,
                )
                if page_info is not None
                and page_info.has_text
                and any(image.renderable for image in page_info.images)
                and not _page_has_text(page)
            )
        if pages:
            logger.info(
                "Ignoring whitespace-only text on %d image page(s)",
                len(pages),
            )
        return pages
    except Exception as exc:
        log_suppressed_error(
            logger,
            exc,
            "Could not inspect whitespace-only page text in %s: %s",
            pdf_path,
            exc,
        )
        return ()


def _ocr_page_box_coordinates(
    value: object,
    page_number: int,
    name: str,
) -> tuple[float, float, float, float]:
    import pikepdf

    if not isinstance(value, pikepdf.Array) or len(value) != 4:
        raise OCRError(f"OCR page {page_number} has a malformed {name}")
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OCRError(f"OCR page {page_number} has a malformed {name}") from exc
    if not all(math.isfinite(item) for item in coordinates):
        raise OCRError(f"OCR page {page_number} has a malformed {name}")
    left, bottom, right, top = coordinates
    if right <= left or top <= bottom:
        raise OCRError(f"OCR page {page_number} has a non-normalized {name}")

    from .sanitizers.page_boxes import (
        _MAX_PAGE_BOUNDARY_SIZE,
        _MIN_PAGE_BOUNDARY_SIZE,
    )

    width = right - left
    height = top - bottom
    if not (
        _MIN_PAGE_BOUNDARY_SIZE <= width <= _MAX_PAGE_BOUNDARY_SIZE
        and _MIN_PAGE_BOUNDARY_SIZE <= height <= _MAX_PAGE_BOUNDARY_SIZE
    ):
        raise OCRError(
            f"OCR page {page_number} {name} is outside PDF page-boundary limits"
        )
    return coordinates


def _validated_ocr_page_rotation(page: Any, page_number: int) -> int:
    """Return an inherited page rotation only when it is an exact 90-degree step."""
    current = page.obj
    visited: set[tuple[int, int]] = set()
    while current is not None:
        objgen = getattr(current, "objgen", None)
        if isinstance(objgen, tuple):
            if objgen in visited:
                break
            visited.add(objgen)

        if "/Rotate" in current:
            raw_rotation = current.get("/Rotate")
            try:
                numeric_rotation = float(raw_rotation)
            except (TypeError, ValueError, OverflowError) as exc:
                raise OCRError(
                    f"OCR page {page_number} has an invalid rotation"
                ) from exc
            if (
                not math.isfinite(numeric_rotation)
                or not numeric_rotation.is_integer()
                or int(numeric_rotation) % 90
            ):
                raise OCRError(
                    f"OCR page {page_number} rotation is not a multiple of 90"
                )
            return _effective_page_rotate(page)
        current = current.get("/Parent")
    return 0


def _include_forced_ocr_forms_in_content_preflight(pdf: "pikepdf.Pdf") -> None:
    """Make every existing OCR Form visible to the read-only work validator."""
    import pikepdf

    from .digital_layout import _inherited_page_resources

    for page_number, page in enumerate(pdf.pages, start=1):
        resources = _inherited_page_resources(page)
        xobjects = resources.get("/XObject")
        if xobjects is None:
            continue
        if not isinstance(xobjects, pikepdf.Dictionary):
            raise ConversionError(
                f"OCR page {page_number} XObject resources are malformed"
            )
        calls = []
        for name, value in xobjects.items():
            if not str(name).startswith("/OCR-"):
                continue
            try:
                xobject = value.get_object()
            except (AttributeError, TypeError, ValueError):
                xobject = value
            if not isinstance(xobject, (pikepdf.Dictionary, pikepdf.Stream)):
                raise ConversionError(
                    f"OCR page {page_number} OCR Form resource is malformed"
                )
            if (
                isinstance(xobject, pikepdf.Stream)
                and str(xobject.get("/Subtype")) == "/Form"
            ):
                calls.append(pikepdf.Name(str(name)).unparse() + b" Do")
        if not calls:
            continue
        validation_stream = pdf.make_stream(b"\n".join(calls))
        contents = page.obj.get("/Contents")
        if isinstance(contents, pikepdf.Array):
            page.obj[pikepdf.Name.Contents] = pikepdf.Array(
                [*contents, validation_stream]
            )
        elif contents is None:
            page.obj[pikepdf.Name.Contents] = validation_stream
        else:
            page.obj[pikepdf.Name.Contents] = pikepdf.Array(
                [contents, validation_stream]
            )


def _ocr_raster_pixel_count(
    page_dimensions: tuple[float, float],
    dpi: float,
) -> int:
    """Return a conservative raster pixel count for physical page dimensions."""
    pixel_width = math.ceil(page_dimensions[0] * dpi / 72.0)
    pixel_height = math.ceil(page_dimensions[1] * dpi / 72.0)
    return pixel_width * pixel_height


def _preflight_ocr_input(
    pdf_path: Path,
    *,
    force: bool = False,
    deskew: bool = False,
    rotate_pages: bool = False,
) -> int:
    """Reject unsafe raster work and return the planned OCR oversample DPI."""
    import pikepdf
    from ocrmypdf.pdfinfo import PdfInfo

    try:
        page_dimensions = []
        non_whitespace_text = []
        scan_like_pages = []
        with pikepdf.open(pdf_path) as pdf:
            if not pdf.pages:
                raise OCRError("OCR input contains no pages")
            _validate_ocr_content_work_budget(pdf, force=force)
            for page_number, page in enumerate(pdf.pages, start=1):
                media_box = _ocr_page_box_coordinates(
                    page.mediabox,
                    page_number,
                    "MediaBox",
                )
                crop_box = _ocr_page_box_coordinates(
                    page.cropbox,
                    page_number,
                    "CropBox",
                )
                tolerance = _OCR_MANIFEST_GEOMETRY_TOLERANCE
                if (
                    crop_box[0] < media_box[0] - tolerance
                    or crop_box[1] < media_box[1] - tolerance
                    or crop_box[2] > media_box[2] + tolerance
                    or crop_box[3] > media_box[3] + tolerance
                ):
                    raise OCRError(
                        f"OCR page {page_number} CropBox is outside its MediaBox"
                    )
                for name in ("/BleedBox", "/TrimBox", "/ArtBox"):
                    value = page.obj.get(name)
                    if value is None:
                        continue
                    box = _ocr_page_box_coordinates(
                        value,
                        page_number,
                        name[1:],
                    )
                    if (
                        box[0] < media_box[0] - tolerance
                        or box[1] < media_box[1] - tolerance
                        or box[2] > media_box[2] + tolerance
                        or box[3] > media_box[3] + tolerance
                    ):
                        raise OCRError(
                            f"OCR page {page_number} {name[1:]} is outside its MediaBox"
                        )

                _validated_ocr_page_rotation(page, page_number)
                user_unit_value = page.obj.get("/UserUnit", 1.0)
                try:
                    user_unit = float(user_unit_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise OCRError(
                        f"OCR page {page_number} has an invalid UserUnit"
                    ) from exc
                if (
                    not math.isfinite(user_unit)
                    or user_unit <= 0
                    or user_unit > _MAX_PDF_USER_UNIT
                ):
                    raise OCRError(f"OCR page {page_number} has an invalid UserUnit")

                page_dimensions.append(
                    (
                        (
                            (media_box[2] - media_box[0]) * user_unit,
                            (media_box[3] - media_box[1]) * user_unit,
                        ),
                        (
                            (crop_box[2] - crop_box[0]) * user_unit,
                            (crop_box[3] - crop_box[1]) * user_unit,
                        ),
                    )
                )
                non_whitespace_text.append(_page_has_text(page))
                scan_like_pages.append(
                    any(
                        coverage >= _FULL_PAGE_IMAGE_AREA_RATIO
                        for coverage, _event in _page_paint_analysis(
                            page
                        ).image_candidates
                    )
                )

        pdfinfo = PdfInfo(pdf_path, max_workers=1)
        if len(pdfinfo.pages) != len(page_dimensions):
            raise OCRError("Page count changed during OCR resource preflight")
        raster_pages = []
        for page_number, (page_info, dimensions, has_text, scan_like) in enumerate(
            zip(
                pdfinfo.pages,
                page_dimensions,
                non_whitespace_text,
                scan_like_pages,
                strict=True,
            ),
            start=1,
        ):
            if page_info is None:
                raise OCRError(
                    f"Could not inspect OCR resources for page {page_number}"
                )
            media_dimensions, crop_dimensions = dimensions
            if rotate_pages:
                orientation_dpi = 72.0 * ORIENTATION_RENDER_SCALE
                orientation_width = math.ceil(
                    crop_dimensions[0] * ORIENTATION_RENDER_SCALE
                )
                orientation_height = math.ceil(
                    crop_dimensions[1] * ORIENTATION_RENDER_SCALE
                )
                orientation_pixels = orientation_width * orientation_height
                if orientation_pixels > _OCR_MAX_PAGE_RASTER_PIXELS:
                    raise OCRError(
                        f"Paddle orientation page {page_number} would require "
                        f"{orientation_pixels:,} pixels at {orientation_dpi:g} dpi; "
                        f"the safety limit is {_OCR_MAX_PAGE_RASTER_PIXELS:,} pixels"
                    )
            renderable_image = any(image.renderable for image in page_info.images)
            will_rasterize = (
                force
                or not page_info.has_text
                or not has_text
                or scan_like
                or deskew
                and renderable_image
            )
            if not will_rasterize:
                continue
            native_dpi = max(
                float(page_info.dpi.x),
                float(page_info.dpi.y),
            )
            raster_pages.append((page_number, media_dimensions, native_dpi))

        oversized_pages = {}
        for page_number, media_dimensions, native_dpi in raster_pages:
            raster_dpi = max(float(_OCR_RASTER_DPI), native_dpi)
            pixel_count = _ocr_raster_pixel_count(
                media_dimensions,
                raster_dpi,
            )
            if pixel_count > _OCR_MAX_PAGE_RASTER_PIXELS:
                oversized_pages[page_number] = (pixel_count, raster_dpi)

        planned_oversample = (
            _OCR_FALLBACK_RASTER_DPI if oversized_pages else _OCR_RASTER_DPI
        )
        planned_pages = []
        for page_number, media_dimensions, native_dpi in raster_pages:
            raster_dpi = max(float(planned_oversample), native_dpi)
            pixel_count = _ocr_raster_pixel_count(media_dimensions, raster_dpi)
            if pixel_count > _OCR_MAX_PAGE_RASTER_PIXELS:
                capped_pixel_count = _ocr_raster_pixel_count(
                    media_dimensions,
                    planned_oversample,
                )
                if capped_pixel_count > _OCR_MAX_PAGE_RASTER_PIXELS:
                    raise OCRError(
                        f"OCR page {page_number} would require "
                        f"{capped_pixel_count:,} pixels at {planned_oversample} dpi; "
                        "the safety limit is "
                        f"{_OCR_MAX_PAGE_RASTER_PIXELS:,} pixels"
                    )
                raster_dpi = float(planned_oversample)
                pixel_count = capped_pixel_count
            planned_pages.append((page_number, raster_dpi, pixel_count))
        for page_number, raster_dpi, pixel_count in planned_pages:
            if page_number not in oversized_pages:
                continue
            preferred_pixels, preferred_raster_dpi = oversized_pages[page_number]
            logger.warning(
                "OCR page %d would require %s pixels with the preferred %d dpi "
                "oversampling (effective %.3g dpi); using %d dpi for this OCR "
                "run (%s pixels expected at an effective %.3g dpi)",
                page_number,
                f"{preferred_pixels:,}",
                _OCR_RASTER_DPI,
                preferred_raster_dpi,
                _OCR_FALLBACK_RASTER_DPI,
                f"{pixel_count:,}",
                raster_dpi,
            )
        return planned_oversample
    except OCRError:
        raise
    except ConversionError as exc:
        raise OCRError(f"OCR content-stream preflight failed: {exc}") from exc
    except Exception as exc:
        raise OCRError(f"Could not preflight OCR input: {exc}") from exc


def _cleanup_ocr_resources(
    temporary_directories: tuple[
        tuple[str, TemporaryDirectory[str] | None],
        ...,
    ],
    *,
    preserve_primary_error: bool = False,
) -> None:
    """Release every resource, preserving an already propagating OCR error."""
    failures: list[tuple[str, Exception]] = []
    try:
        _release_ocr_models()
    except Exception as exc:
        failures.append(("release OCR models", exc))

    for label, directory in temporary_directories:
        if directory is None:
            continue
        try:
            directory.cleanup()
        except Exception as exc:
            failures.append((f"clean up {label}", exc))

    if not failures:
        return
    if preserve_primary_error:
        for action, exc in failures:
            logger.warning("Could not %s: %s", action, exc)
        return
    details = "; ".join(f"could not {action}: {exc}" for action, exc in failures)
    raise OCRError(f"OCR cleanup failed: {details}") from failures[0][1]


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
    ocr_execution_provider: str = "cpu",
    layout: bool = False,
    _annotated_pages: frozenset[int] | None = None,
    _manifest_output_path: Path | None = None,
) -> Path:
    """Performs OCR on a PDF.

    Uses PaddleOCR for recognition and OCRmyPDF for rasterization, text-layer
    rendering, and PDF merging. Digital text pages are skipped unless
    ``force=True``. Raster-dominant full-page scans with a visible native text
    overlay are re-OCRed while retaining that native text.

    Args:
        input_path: Path to the input PDF.
        output_path: Path for the OCR-processed PDF.
        languages: PaddleOCR 3.7 PP-OCRv6 language codes (default: ``["en"]``).
            Latin-script languages restrict recognition to Latin letters.
            Example: ``["de", "en"]`` for German + English recognition.
        detection_model_dir: Compatible PP-OCRv6 Medium detection model directory.
        recognition_model_dir: Compatible PP-OCRv6 Medium recognition model directory.
        force: If True, use ocrmypdf's ``redo_ocr`` mode to remove the
            existing OCR layer and re-apply OCR. This cannot be combined
            with ``deskew`` (default: False).
        deskew: If True, straighten scan-like, raster-dominant pages with
            PaddleOCR (default: False).
        rotate_pages: If True, normalize page orientation with the bundled
            Paddle model (default: False).
        ocr_execution_provider: ONNX Runtime provider to use for all Paddle
            models: ``"cpu"`` (default), ``"directml"`` or
            ``"directml:<index>"`` to select a specific adapter.
        layout: If True, order OCR lines by detected page columns.
        _manifest_output_path: Private Level-A integration path for an atomic,
            versioned OCR document manifest. Supplying it also enables MCID
            markers and layout-derived reading order in the OCR text Forms.

    Returns:
        Path to the OCR-processed PDF.

    Raises:
        OCRError: If OCR is not available or fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    manifest_output_path = (
        Path(_manifest_output_path) if _manifest_output_path is not None else None
    )
    if manifest_output_path is not None:
        try:
            manifest_target = manifest_output_path.resolve()
            protected_targets = {input_path.resolve(), output_path.resolve()}
        except OSError as exc:
            raise OCRError(f"Could not resolve OCR output paths: {exc}") from exc
        if manifest_target in protected_targets:
            raise OCRError(
                "OCR manifest path must differ from the input and PDF output paths"
            )
    if languages is None:
        languages = ["en"]
    if force and deskew:
        raise OCRError("Deskew cannot be combined with forced OCR")
    _require_ocr_runtime(ocr_execution_provider)

    from .ocr_paddle import validate_model_directories

    try:
        validate_ocr_languages(languages)
    except ValueError as exc:
        raise OCRError(str(exc)) from exc

    planned_oversample = _preflight_ocr_input(
        input_path,
        force=force,
        deskew=deskew,
        rotate_pages=rotate_pages,
    )

    detection_model_dir, recognition_model_dir = validate_model_directories(
        detection_model_dir,
        recognition_model_dir,
    )

    logger.info(
        "Starting PaddleOCR for %s (languages: %s, force: %s, deskew: %s, "
        "rotate pages: %s, execution provider: %s, layout: %s)",
        input_path,
        "+".join(languages),
        force,
        deskew,
        rotate_pages,
        ocr_execution_provider,
        layout,
    )

    ocr_input_path = input_path
    orientation_temp: TemporaryDirectory[str] | None = None
    pipeline_temp: TemporaryDirectory[str] | None = None
    manifest_temp: TemporaryDirectory[str] | None = None
    output_staging: TemporaryDirectory[str] | None = None
    manifest_staging: TemporaryDirectory[str] | None = None
    staged_manifest_path: Path | None = None
    try:
        output_staging = private_staging_directory(
            output_path.parent,
            prefix=f".{output_path.stem}_ocr_",
            delete=False,
        )
        staged_output_path = Path(output_staging.name) / "output.pdf"
        if manifest_output_path is not None:
            manifest_staging = private_staging_directory(
                manifest_output_path.parent,
                prefix=f".{manifest_output_path.stem}_ocr_",
                delete=False,
            )
            staged_manifest_path = Path(manifest_staging.name) / "manifest.json"
    except OSError as exc:
        for directory in (manifest_staging, output_staging):
            if directory is None:
                continue
            try:
                directory.cleanup()
            except Exception as cleanup_exc:
                logger.warning(
                    "Could not remove private OCR staging directory %s: %s",
                    directory.name,
                    cleanup_exc,
                )
        raise OCRError(f"Could not stage OCR output: {exc}") from exc
    manifest_pages: dict[int, dict[str, Any]] = {}
    manifest_run_number = 0
    existing_ocr_form_names: list[frozenset[str]] = []
    completed_successfully = False
    staged_output_snapshot: StagedFileSnapshot | None = None
    staged_manifest_snapshot: StagedFileSnapshot | None = None

    def run_ocr(
        source: Path,
        destination: Path,
        *,
        pages: tuple[int, ...] | None = None,
        deskew_run: bool = False,
        redo: bool = False,
    ) -> None:
        nonlocal manifest_run_number

        ocr_kwargs: dict[str, object] = {
            "ocr_engine": "paddle",
            "pdf_renderer": "fpdf2",
            "rasterizer": "pypdfium",
            "output_type": "pdf",
            "oversample": planned_oversample,
            "max_image_mpixels": _OCR_MAX_PAGE_RASTER_PIXELS / 1_000_000,
            "optimize": 0,
            "jobs": 1,
            "skip_text": True,
            "deskew": deskew_run,
            "rotate_pages": False,
            "progress_bar": False,
            "plugins": [_PADDLE_OCR_PLUGIN, _ROTATION_FIX_PLUGIN],
            "paddle_detection_model_dir": detection_model_dir,
            "paddle_recognition_model_dir": recognition_model_dir,
            "paddle_execution_provider": ocr_execution_provider,
            "paddle_layout": layout,
        }
        if pages is not None:
            ocr_kwargs["pages"] = ",".join(str(page) for page in pages)
        if redo:
            ocr_kwargs.pop("skip_text", None)
            ocr_kwargs["redo_ocr"] = True

        run_manifest_directory: Path | None = None
        if manifest_temp is not None:
            run_manifest_directory = (
                Path(manifest_temp.name) / f"run-{manifest_run_number:04d}"
            )
            manifest_run_number += 1
            run_manifest_directory.mkdir()
            ocr_kwargs["paddle_manifest_dir"] = run_manifest_directory

        from .ocr_fpdf import install_fpdf_renderer

        with install_fpdf_renderer():
            ocrmypdf.ocr(
                source,
                destination,
                language=languages,
                **ocr_kwargs,
            )
        if run_manifest_directory is not None:
            run_pages = _read_ocr_run_sidecars(run_manifest_directory)
            duplicate_pages = frozenset(manifest_pages).intersection(run_pages)
            if duplicate_pages:
                raise OCRError(
                    "OCR manifest pages were emitted by multiple runs: "
                    f"{sorted(duplicate_pages)}"
                )
            manifest_pages.update(run_pages)

    try:
        if manifest_output_path is not None:
            manifest_temp = TemporaryDirectory(prefix="pdftopdfa_ocr_manifest_")

        if rotate_pages:
            orientation_temp = TemporaryDirectory(
                prefix="pdftopdfa_paddle_orientation_"
            )
            ocr_input_path = (
                Path(orientation_temp.name) / f"{input_path.stem}_oriented.pdf"
            )
            orientation_started = time.perf_counter()
            normalize_pdf_orientation(
                input_path,
                ocr_input_path,
                execution_provider=ocr_execution_provider,
            )
            logger.info(
                "Paddle orientation preflight completed in %.2fs",
                time.perf_counter() - orientation_started,
            )

        existing_ocr_form_names = _ocr_form_names(ocr_input_path)

        if not force:
            whitespace_text_pages = _find_whitespace_only_text_pages(ocr_input_path)
            if whitespace_text_pages:
                pipeline_temp = TemporaryDirectory(prefix="pdftopdfa_paddle_ocr_")
                prepared_input = Path(pipeline_temp.name) / "ocr_input.pdf"
                _prepare_deskew_input(
                    ocr_input_path,
                    prepared_input,
                    whitespace_text_pages,
                )
                ocr_input_path = prepared_input

        if force:
            run_ocr(ocr_input_path, staged_output_path, redo=True)
        elif not deskew:
            plan = _plan_deskew_ocr(
                ocr_input_path,
                annotated_pages=_annotated_pages,
            )
            if plan is None:
                run_ocr(ocr_input_path, staged_output_path)
            else:
                if plan.ambiguous_scan_pages:
                    raise OCRError(
                        "OCR cannot safely replace an existing text layer on "
                        "ambiguous scan-like page(s): "
                        f"{list(plan.ambiguous_scan_pages)}"
                    )
                regular_pages = tuple(
                    sorted((*plan.regular_ocr_pages, *plan.deskew_pages))
                )
                if (regular_pages or plan.redo_ocr_pages) and pipeline_temp is None:
                    pipeline_temp = TemporaryDirectory(prefix="pdftopdfa_paddle_ocr_")
                current_input = ocr_input_path
                if plan.strip_text_pages:
                    prepared_input = Path(pipeline_temp.name) / "ocr_input.pdf"
                    _prepare_deskew_input(
                        current_input,
                        prepared_input,
                        plan.strip_text_pages,
                    )
                    current_input = prepared_input
                if regular_pages:
                    regular_output = (
                        staged_output_path
                        if not plan.redo_ocr_pages
                        else Path(pipeline_temp.name) / "regular_ocr.pdf"
                    )
                    run_ocr(
                        current_input,
                        regular_output,
                        pages=regular_pages,
                    )
                    current_input = regular_output
                if plan.redo_ocr_pages:
                    run_ocr(
                        current_input,
                        staged_output_path,
                        pages=plan.redo_ocr_pages,
                        redo=True,
                    )
                elif not regular_pages:
                    shutil.copy2(current_input, staged_output_path)
        else:
            plan = _plan_deskew_ocr(
                ocr_input_path,
                annotated_pages=_annotated_pages,
            )
            if plan is None:
                logger.warning(
                    "Deskew disabled because scan-like pages could not be "
                    "identified safely"
                )
                run_ocr(ocr_input_path, staged_output_path)
            elif plan.ambiguous_scan_pages:
                raise OCRError(
                    "OCR cannot safely replace an existing text layer on "
                    "ambiguous scan-like page(s): "
                    f"{list(plan.ambiguous_scan_pages)}"
                )
            elif (
                not plan.deskew_pages
                and not plan.regular_ocr_pages
                and not plan.redo_ocr_pages
            ):
                shutil.copy2(ocr_input_path, staged_output_path)
            else:
                if pipeline_temp is None:
                    pipeline_temp = TemporaryDirectory(
                        prefix="pdftopdfa_paddle_deskew_"
                    )
                current_input = ocr_input_path

                if plan.strip_text_pages:
                    prepared_input = Path(pipeline_temp.name) / "ocr_input.pdf"
                    _prepare_deskew_input(
                        current_input,
                        prepared_input,
                        plan.strip_text_pages,
                    )
                    current_input = prepared_input

                if plan.regular_ocr_pages:
                    regular_output = (
                        staged_output_path
                        if not plan.deskew_pages and not plan.redo_ocr_pages
                        else Path(pipeline_temp.name) / "regular_ocr.pdf"
                    )
                    run_ocr(
                        current_input,
                        regular_output,
                        pages=plan.regular_ocr_pages,
                    )
                    current_input = regular_output

                if plan.deskew_pages:
                    deskew_output = (
                        staged_output_path
                        if not plan.redo_ocr_pages
                        else Path(pipeline_temp.name) / "deskew_ocr.pdf"
                    )
                    run_ocr(
                        current_input,
                        deskew_output,
                        pages=plan.deskew_pages,
                        deskew_run=True,
                    )
                    current_input = deskew_output

                if plan.redo_ocr_pages:
                    run_ocr(
                        current_input,
                        staged_output_path,
                        pages=plan.redo_ocr_pages,
                        redo=True,
                    )

        new_ocr_form_names = _finalize_ocr_output(
            staged_output_path,
            languages,
            existing_ocr_form_names,
            strip_existing_ocr_text=force,
        )
        if staged_manifest_path is not None:
            _write_ocr_document_manifest(
                staged_manifest_path,
                staged_output_path,
                languages,
                manifest_pages,
                new_ocr_form_names,
            )
        staged_output_snapshot = staged_file_snapshot(staged_output_path)
        if staged_manifest_path is not None:
            staged_manifest_snapshot = staged_file_snapshot(staged_manifest_path)
        completed_successfully = True

    except EncryptedPdfError as e:
        raise OCRError(f"OCR failed: PDF is encrypted ({input_path})") from e

    except PriorOcrFoundError as e:
        raise OCRError(
            "OCR failed: the selected page already contains an OCR text layer; "
            "refusing to publish it as PaddleOCR output"
        ) from e

    except MissingDependencyError as e:
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e

    except OCRError:
        raise

    except Exception as e:
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e

    finally:
        cleanup_completed = False
        try:
            _cleanup_ocr_resources(
                (
                    ("OCR manifest temporary directory", manifest_temp),
                    ("OCR pipeline temporary directory", pipeline_temp),
                    ("OCR orientation temporary directory", orientation_temp),
                ),
                preserve_primary_error=not completed_successfully,
            )
            cleanup_completed = True
        finally:
            if not completed_successfully or not cleanup_completed:
                for directory in (manifest_staging, output_staging):
                    if directory is None:
                        continue
                    try:
                        directory.cleanup()
                    except Exception as exc:
                        logger.warning(
                            "Could not remove private OCR staging directory %s: %s",
                            directory.name,
                            exc,
                        )

    output_backup = Path(output_staging.name) / f"backup{output_path.suffix}"
    manifest_backup = (
        Path(manifest_staging.name) / f"backup{manifest_output_path.suffix}"
        if manifest_staging is not None and manifest_output_path is not None
        else None
    )
    output_publication_attempted = False
    manifest_publication_attempted = False
    keep_output_backup = False
    keep_manifest_backup = False

    def rollback_publication(
        destination: Path,
        candidate: StagedFileSnapshot,
        backup: Path,
    ) -> None:
        if backup.exists():
            original = staged_file_snapshot(backup)
        else:
            try:
                current = staged_file_snapshot(destination)
            except ConversionError:
                return
            if (current.device, current.inode) != (candidate.device, candidate.inode):
                return
            original = None
        rollback_staged_publication(
            destination,
            candidate,
            original=original,
            backup=backup if original is not None else None,
        )

    try:
        if staged_output_snapshot is None:
            raise OCRError("OCR output was not finalized for publication")
        if staged_manifest_path is not None:
            if staged_manifest_snapshot is None:
                raise OCRError("OCR manifest was not finalized for publication")

        output_publication_attempted = True
        publish_staged_file(
            staged_output_path,
            output_path,
            staged_output_snapshot,
            backup=output_backup,
        )
        if staged_manifest_path is not None and manifest_output_path is not None:
            manifest_publication_attempted = True
            publish_staged_file(
                staged_manifest_path,
                manifest_output_path,
                staged_manifest_snapshot,
                backup=manifest_backup,
            )
    except BaseException as exc:
        recovery_failures: list[str] = []
        if (
            manifest_publication_attempted
            and manifest_output_path is not None
            and manifest_backup is not None
        ):
            try:
                rollback_publication(
                    manifest_output_path,
                    staged_manifest_snapshot,
                    manifest_backup,
                )
            except BaseException as recovery_exc:
                keep_manifest_backup = bool(
                    manifest_backup is not None and manifest_backup.exists()
                )
                recovery_failures.append(
                    "manifest recovery failed: "
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
        if output_publication_attempted:
            try:
                rollback_publication(
                    output_path,
                    staged_output_snapshot,
                    output_backup,
                )
            except BaseException as recovery_exc:
                keep_output_backup = output_backup.exists()
                recovery_failures.append(
                    "output recovery failed: "
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
        recovery_message = ""
        if recovery_failures:
            recovery_message = f"; {'; '.join(recovery_failures)}"
            if keep_output_backup:
                recovery_message += f"; recovery copy retained at {output_backup}"
            if keep_manifest_backup and manifest_backup is not None:
                recovery_message += (
                    f"; manifest recovery copy retained at {manifest_backup}"
                )
        if not isinstance(exc, Exception):
            if recovery_message:
                exc.add_note(f"OCR publication recovery{recovery_message}")
            raise
        raise OCRError(
            f"Could not publish OCR output atomically: {exc}{recovery_message}"
        ) from exc
    finally:
        directories = []
        if not keep_manifest_backup:
            directories.append(manifest_staging)
        if not keep_output_backup:
            directories.append(output_staging)
        for directory in directories:
            if directory is None:
                continue
            try:
                directory.cleanup()
            except Exception as exc:
                logger.warning(
                    "Could not remove private OCR staging directory %s: %s",
                    directory.name,
                    exc,
                )

    logger.info("OCR completed successfully: %s", output_path)
    return output_path
