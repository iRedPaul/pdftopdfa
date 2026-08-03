# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""OCR functionality for pdftopdfa.

This module provides functions for optical character recognition (OCR)
in image-based PDFs (scanned documents).
"""

import copy
import gc
import logging
import math
import shutil
import time
from dataclasses import dataclass, field
from numbers import Number
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
from ._ocr_runtime import execution_provider_base, onnxruntime_engine_config
from .exceptions import OCRError
from .orientation import _effective_page_rotate, normalize_pdf_orientation
from .utils import log_suppressed_error

logger = logging.getLogger(__name__)

_PADDLE_OCR_PLUGIN = "pdftopdfa.ocr_paddle"
_ROTATION_FIX_PLUGIN = "pdftopdfa.ocr_rotation_fix"
_SCAN_IMAGE_AREA_RATIO = 0.8
_FULL_PAGE_IMAGE_AREA_RATIO = 1.0
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
    onnxruntime_engine_config(ocr_execution_provider)
    if not HAS_OCR:
        extra = (
            "directml"
            if execution_provider_base(ocr_execution_provider) == "directml"
            else "ocr"
        )
        raise OCRError(
            "OCR not available. Install the OCR dependency: "
            f"pip install pdftopdfa[{extra}]"
        )

    from .ocr_paddle import recognize_image as recognize_with_paddle

    try:
        return recognize_with_paddle(
            input_path,
            detection_model_dir=detection_model_dir,
            recognition_model_dir=recognition_model_dir,
            ocr_execution_provider=ocr_execution_provider,
            layout=layout,
            allowed_characters=allowed_characters,
        )
    finally:
        _release_ocr_models()


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
    strip_text_pages: tuple[int, ...]


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
    media_box: tuple[float, float, float, float],
) -> float:
    """Return the page-area fraction covered by an axis-aligned image rectangle."""
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
    page_left, page_bottom, page_right, page_top = media_box
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
    media_box: tuple[float, float, float, float],
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
                    if frame.optional_content_depth:
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
                    coverage = _rect_coverage(frame.ctm, media_box)
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
                        coverage = _rect_coverage(frame.ctm, media_box)
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
                        _rect_coverage(box_matrix @ form_ctm, media_box)
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
        media_box_values = tuple(float(value) for value in page.MediaBox)
        if len(media_box_values) != 4 or not all(
            math.isfinite(value) for value in media_box_values
        ):
            raise ValueError("Invalid MediaBox")
        group = page.obj.get("/Group")
        if group is not None and str(group.get("/S")) != "/Transparency":
            raise ValueError("Invalid Page Group")
        _analyze_content(
            page,
            page.resources,
            media_box_values,
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
            strip_text_pages = []
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
                if not safe_raster_content or not has_renderable_image:
                    if not page_info.has_text:
                        regular_ocr_pages.append(page_number)
                    continue

                full_page_image_event = max(
                    (
                        event
                        for coverage, event in analysis.image_candidates
                        if coverage >= _FULL_PAGE_IMAGE_AREA_RATIO
                    ),
                    default=-1,
                )
                has_visible_text = any(
                    event > full_page_image_event for event in analysis.visible_text
                )
                if analysis.image_candidates and not has_visible_text:
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
                        if not page_info.has_text:
                            regular_ocr_pages.append(page_number)
                        continue
                    deskew_pages.append(page_number)
                    if page_info.has_text:
                        strip_text_pages.append(page_number)
                elif not page_info.has_text and not has_visible_text:
                    regular_ocr_pages.append(page_number)

        logger.info(
            "Deskew selected %d/%d scan-like page(s)",
            len(deskew_pages),
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
            tuple(strip_text_pages),
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

    try:
        for operands, operator in pikepdf.parse_content_stream(page):
            operator_name = str(operator)
            if operator_name in text_operators and _text_show_has_content(
                operands, operator_name
            ):
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

        visited: set[_ObjectKey] = set()
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
    visited: set[_ObjectKey],
) -> bool:
    """Check a Form XObject and its descendants for text operators.

    Args:
        xobj: The XObject to check.
        text_operators: Set of PDF text operator names.
        visited: Set of already-visited object IDs to prevent cycles.

    Returns:
        True if the Form XObject (or nested Form XObjects) contains text.
    """
    import pikepdf

    pending = [xobj]
    while pending:
        current = pending.pop()
        try:
            subtype = current.get("/Subtype")
            if subtype is None or str(subtype) != "/Form":
                continue
        except Exception:
            continue

        key = _object_key(current)
        if key in visited:
            continue
        visited.add(key)

        try:
            for operands, operator in pikepdf.parse_content_stream(current):
                operator_name = str(operator)
                if operator_name in text_operators and _text_show_has_content(
                    operands, operator_name
                ):
                    return True
        except Exception as e:
            log_suppressed_error(
                logger, e, "Error parsing Form XObject content stream: %s", e
            )
            continue

        try:
            resources = current.get("/Resources")
            if resources is None:
                continue
            nested_xobjects = resources.get("/XObject")
            if nested_xobjects is None:
                continue

            nested_forms = []
            for name in nested_xobjects.keys():
                try:
                    nested = nested_xobjects[name].get_object()
                except (AttributeError, TypeError, ValueError):
                    nested = nested_xobjects[name]
                nested_forms.append(nested)
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
) -> Path:
    """Performs OCR on a PDF.

    Uses PaddleOCR for recognition and OCRmyPDF for rasterization, text-layer
    rendering, and PDF merging. Pages that already contain text are skipped
    unless ``force=True`` or a scan-like page is selected for deskewing.

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

    Returns:
        Path to the OCR-processed PDF.

    Raises:
        OCRError: If OCR is not available or fails.
    """
    if languages is None:
        languages = ["en"]
    if force and deskew:
        raise OCRError("Deskew cannot be combined with forced OCR")
    onnxruntime_engine_config(ocr_execution_provider)
    if not HAS_OCR:
        extra = (
            "directml"
            if execution_provider_base(ocr_execution_provider) == "directml"
            else "ocr"
        )
        raise OCRError(
            "OCR not available. Install the OCR dependency: "
            f"pip install pdftopdfa[{extra}]"
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
    existing_ocr_form_names: list[frozenset[str]] = []
    completed_input_path = input_path

    def run_ocr(
        source: Path,
        destination: Path,
        *,
        pages: tuple[int, ...] | None = None,
        deskew_run: bool = False,
        redo: bool = False,
    ) -> None:
        ocr_kwargs: dict[str, object] = {
            "ocr_engine": "paddle",
            "pdf_renderer": "fpdf2",
            "rasterizer": "pypdfium",
            "output_type": "pdf",
            "oversample": 600,
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

        from .ocr_fpdf import install_fpdf_renderer

        with install_fpdf_renderer():
            ocrmypdf.ocr(
                source,
                destination,
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
        completed_input_path = ocr_input_path

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
                completed_input_path = prepared_input

        if force:
            run_ocr(ocr_input_path, output_path, redo=True)
        elif not deskew:
            run_ocr(ocr_input_path, output_path)
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
                run_ocr(ocr_input_path, output_path)
            elif not plan.deskew_pages and not plan.regular_ocr_pages:
                shutil.copy2(ocr_input_path, output_path)
            else:
                if pipeline_temp is None:
                    pipeline_temp = TemporaryDirectory(
                        prefix="pdftopdfa_paddle_deskew_"
                    )
                current_input = ocr_input_path

                if plan.regular_ocr_pages:
                    regular_output = (
                        output_path
                        if not plan.deskew_pages
                        else Path(pipeline_temp.name) / "regular_ocr.pdf"
                    )
                    run_ocr(
                        current_input,
                        regular_output,
                        pages=plan.regular_ocr_pages,
                    )
                    current_input = regular_output
                    completed_input_path = current_input

                if plan.deskew_pages:
                    if plan.strip_text_pages:
                        prepared_input = Path(pipeline_temp.name) / "deskew_input.pdf"
                        _prepare_deskew_input(
                            current_input,
                            prepared_input,
                            plan.strip_text_pages,
                        )
                        current_input = prepared_input
                    run_ocr(
                        current_input,
                        output_path,
                        pages=plan.deskew_pages,
                        deskew_run=True,
                    )

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
        shutil.copy2(completed_input_path, output_path)
        _finalize_ocr_output(
            output_path,
            languages,
            existing_ocr_form_names,
            strip_existing_ocr_text=False,
        )
        return output_path

    except MissingDependencyError as e:
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e

    except OCRError:
        raise

    except Exception as e:
        raise OCRError(f"OCR failed: {_format_ocr_exception(e)}") from e

    finally:
        _release_ocr_models()
        if pipeline_temp is not None:
            pipeline_temp.cleanup()
        if orientation_temp is not None:
            orientation_temp.cleanup()
