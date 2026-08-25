# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Read-only painting provenance for digitally generated PDF pages."""

from __future__ import annotations

import re
import string
import zlib
from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal
from io import BytesIO
from math import hypot, isfinite, ulp
from tempfile import SpooledTemporaryFile
from typing import Literal, cast

import pikepdf
from pdfminer.casting import safe_float, safe_int
from pdfminer.converter import PDFPageAggregator
from pdfminer.fontmetrics import FONT_METRICS
from pdfminer.layout import LTChar, LTLayoutContainer, LTPage
from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdfdevice import PDFTextSeq
from pdfminer.pdfinterp import (
    LITERAL_FORM,
    LITERAL_IMAGE,
    PDFGraphicState,
    PDFInterpreterError,
    PDFPageInterpreter,
    PDFResourceManager,
    PDFStackT,
    PDFTextState,
)
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import PDFStream, dict_value, resolve1
from pdfminer.psparser import literal_name
from pdfminer.utils import (
    Matrix,
    Rect,
    apply_matrix_pt,
    apply_matrix_rect,
    mult_matrix,
    translate_matrix,
)

from .exceptions import ConversionError
from .utils import resolve_indirect

__all__ = [
    "BBox",
    "ClipPolygon",
    "DigitalPageLayout",
    "DirectTextSpan",
    "DirectXObjectSpan",
    "InvocationPaintState",
    "PaintingSpan",
    "TextRun",
    "XObjectKind",
    "extract_digital_layout",
]

BBox = tuple[float, float, float, float]
Point = tuple[float, float]
ClipPolygon = tuple[Point, ...]
XObjectKind = Literal["form", "image", "inline_image"]
_ImageVisibility = Literal["visible", "invisible", "uncertain"]
_SERIALIZED_PDF_MEMORY_LIMIT = 16 * 1024 * 1024
_MAX_DIGITAL_OPERATORS_PER_PAGE = 100_000
_MAX_DIGITAL_OPERATORS_PER_DOCUMENT = 1_000_000
_MAX_FORM_INVOCATIONS_PER_RESOURCE_PER_PAGE = 1_024
_MAX_FORM_NESTING_DEPTH = 64
_MAX_CONTENT_STREAMS_PER_PAGE = 8_192
_MAX_ENCODED_CONTENT_BYTES_PER_CONTAINER = 32 * 1024 * 1024
_MAX_DECODED_CONTENT_BYTES_PER_CONTAINER = 32 * 1024 * 1024
_MAX_DECODED_CONTENT_BYTES_PER_PAGE = 128 * 1024 * 1024
_MAX_DECODED_CONTENT_BYTES_PER_DOCUMENT = 512 * 1024 * 1024
_MAX_ENCODED_CONTENT_BYTES_PER_PAGE = _MAX_DECODED_CONTENT_BYTES_PER_PAGE
_MAX_ENCODED_CONTENT_BYTES_PER_DOCUMENT = _MAX_DECODED_CONTENT_BYTES_PER_DOCUMENT
_MAX_ENCODED_IMAGE_VISIBILITY_BYTES = 32 * 1024 * 1024
_MAX_ENCODED_IMAGE_VISIBILITY_BYTES_PER_DOCUMENT = 512 * 1024 * 1024
_MAX_DECODED_IMAGE_VISIBILITY_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_VISIBILITY_PIXELS = 16_777_216
_MAX_IMAGE_VISIBILITY_BYTES_PER_DOCUMENT = 512 * 1024 * 1024
_MAX_IMAGE_VISIBILITY_SAMPLES_PER_DOCUMENT = 134_217_728
_MAX_IMAGE_VISIBILITY_DECODES_PER_DOCUMENT = 4_096
_DECODE_CHUNK_SIZE = 1024 * 1024
_DIRECT_TEXT_OPERAND_COUNTS = {"Tj": 1, "TJ": 1, "'": 1, '"': 3}
_PATH_START_OPERATORS = frozenset({"m", "re"})
_PATH_CONTINUATION_OPERATORS = frozenset({"l", "c", "v", "y", "h"})
_PATH_PAINTING_OPERATORS = frozenset(
    {"n", "S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}
)
_TEXT_BASE14_FONTS = frozenset(
    {
        "Courier",
        "Courier-Bold",
        "Courier-BoldOblique",
        "Courier-Oblique",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-BoldOblique",
        "Helvetica-Oblique",
        "Times-Bold",
        "Times-BoldItalic",
        "Times-Italic",
        "Times-Roman",
    }
)
_CID_FALLBACK = re.compile(r"(?:\(cid:[0-9]+\))+")


class _NonRectangularClipError(PDFInterpreterError):
    pass


@dataclass(frozen=True, slots=True)
class InvocationPaintState:
    """Graphics parameters inherited by one Form XObject invocation."""

    line_width: float = 1.0
    line_cap: int = 0
    line_join: int = 0
    miter_limit: float = 10.0
    dash_array: tuple[float, ...] = ()
    dash_phase: float = 0.0
    stroke_adjust: bool = False
    stroke_alpha: float = 1.0
    fill_alpha: float = 1.0
    soft_mask_active: bool = False
    stroke_overprint: bool = False
    fill_overprint: bool = False
    overprint_mode: int = 0
    stroke_color_complex: bool = False
    fill_color_complex: bool = False
    blend_mode_complex: bool = False
    text_render_mode: int = 0
    clip_visibility_uncertain: bool = False


def _polygon_area(polygon: ClipPolygon) -> float:
    return 0.5 * sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(polygon, (*polygon[1:], polygon[0]))
    )


def _polygon_tolerance(points: tuple[Point, ...]) -> float:
    if not points:
        return 1e-7
    values = tuple(value for point in points for value in point)
    extent = max(
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
        1.0,
    )
    return max(
        1e-7,
        extent * 1e-9,
        max(ulp(abs(value)) for value in values) * 4,
    )


def _normalize_polygon(points: tuple[Point, ...]) -> ClipPolygon:
    if not points or any(not isfinite(value) for point in points for value in point):
        return ()
    tolerance = _polygon_tolerance(points)
    normalized: list[Point] = []
    for point in points:
        if not normalized or any(
            abs(first - second) > tolerance
            for first, second in zip(normalized[-1], point)
        ):
            normalized.append(point)
    if len(normalized) > 1 and all(
        abs(first - second) <= tolerance
        for first, second in zip(normalized[0], normalized[-1])
    ):
        normalized.pop()
    if len(normalized) < 3:
        return ()
    polygon = tuple(normalized)
    area = _polygon_area(polygon)
    if abs(area) <= tolerance * tolerance:
        return ()
    return polygon if area > 0 else tuple(reversed(polygon))


def _parallelogram_polygon(points: tuple[Point, ...]) -> ClipPolygon | None:
    if len(points) != 4:
        return None
    tolerance = _polygon_tolerance(points)
    if any(
        all(abs(first - second) <= tolerance for first, second in zip(point, other))
        for index, point in enumerate(points)
        for other in points[index + 1 :]
    ):
        return None
    if any(
        abs(points[0][axis] + points[2][axis] - points[1][axis] - points[3][axis])
        > tolerance * 4
        for axis in (0, 1)
    ):
        return None
    polygon = _normalize_polygon(points)
    return polygon if len(polygon) == 4 else None


def _rect_polygon(matrix: Matrix, bbox: Rect | BBox) -> ClipPolygon:
    left, bottom, right, top = (float(value) for value in bbox)
    left, right = sorted((left, right))
    bottom, top = sorted((bottom, top))
    return _normalize_polygon(
        tuple(
            apply_matrix_pt(matrix, point)
            for point in (
                (left, bottom),
                (right, bottom),
                (right, top),
                (left, top),
            )
        )
    )


def _polygon_intersection(
    current: ClipPolygon | None,
    update: ClipPolygon,
) -> ClipPolygon:
    if not update:
        return ()
    if current is None:
        return update
    if not current:
        return ()
    output = list(current)
    tolerance = _polygon_tolerance((*current, *update))
    for edge_start, edge_end in zip(update, (*update[1:], update[0])):
        input_points = output
        output = []
        if not input_points:
            break

        def signed_distance(point: Point) -> float:
            return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
                edge_end[1] - edge_start[1]
            ) * (point[0] - edge_start[0])

        previous = input_points[-1]
        previous_distance = signed_distance(previous)
        for point in input_points:
            distance = signed_distance(point)
            point_inside = distance >= -tolerance
            previous_inside = previous_distance >= -tolerance
            if point_inside != previous_inside:
                denominator = previous_distance - distance
                if abs(denominator) > tolerance:
                    ratio = previous_distance / denominator
                    output.append(
                        (
                            previous[0] + ratio * (point[0] - previous[0]),
                            previous[1] + ratio * (point[1] - previous[1]),
                        )
                    )
            if point_inside:
                output.append(point)
            previous = point
            previous_distance = distance
    return _normalize_polygon(tuple(output))


def _polygon_bbox(polygon: ClipPolygon | None) -> BBox | None:
    if polygon is None:
        return None
    if not polygon:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


def _clip_bbox_to_polygon(bbox: BBox, polygon: ClipPolygon | None) -> BBox | None:
    if polygon is None:
        return bbox
    intersection = _polygon_intersection(
        _rect_polygon((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), bbox),
        polygon,
    )
    result = _polygon_bbox(intersection)
    return None if result == (0.0, 0.0, 0.0, 0.0) else result


def _transform_polygon(polygon: ClipPolygon, matrix: Matrix) -> ClipPolygon:
    return _normalize_polygon(
        tuple(apply_matrix_pt(matrix, point) for point in polygon)
    )


def _inverse_matrix(matrix: Matrix) -> Matrix:
    if not all(isfinite(value) for value in matrix):
        raise ValueError("Non-finite transformation matrix")
    a, b, c, d, e, f = matrix
    determinant = a * d - b * c
    product_scale = max(abs(a * d), abs(b * c))
    tolerance = max(
        product_scale * 1e-15,
        ulp(abs(a * d)) * 4,
        ulp(abs(b * c)) * 4,
    )
    if not isfinite(determinant) or abs(determinant) <= tolerance:
        raise ValueError("Singular transformation matrix")
    inverse = (
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
        (c * f - d * e) / determinant,
        (b * e - a * f) / determinant,
    )
    if not all(isfinite(value) for value in inverse):
        raise ValueError("Non-finite inverse transformation matrix")
    return inverse


def _is_finite_pdf_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _has_valid_image_dimensions(width: object, height: object) -> bool:
    return (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
    )


def _image_filter_names(image: pikepdf.Stream) -> tuple[str, ...] | None:
    filters = resolve_indirect(image.get("/Filter"))
    if filters is None:
        return ()
    if isinstance(filters, pikepdf.Name):
        return (str(filters),)
    if not isinstance(filters, pikepdf.Array):
        return None
    resolved = tuple(resolve_indirect(item) for item in filters)
    if not all(isinstance(item, pikepdf.Name) for item in resolved):
        return None
    return tuple(str(item) for item in resolved)


def _image_decode_parameters(
    image: pikepdf.Stream,
    filter_count: int,
) -> tuple[pikepdf.Dictionary | None, ...] | None:
    parameters = resolve_indirect(image.get("/DecodeParms"))
    if parameters is None:
        return (None,) * filter_count
    if isinstance(parameters, pikepdf.Dictionary) and filter_count == 1:
        return (parameters,)
    if not isinstance(parameters, pikepdf.Array) or len(parameters) != filter_count:
        return None
    resolved = tuple(resolve_indirect(item) for item in parameters)
    if not all(
        item is None or isinstance(item, pikepdf.Dictionary) for item in resolved
    ):
        return None
    return cast(tuple[pikepdf.Dictionary | None, ...], resolved)


def _image_component_count(image: pikepdf.Stream) -> int | None:
    color_space = resolve_indirect(image.get("/ColorSpace"))
    if isinstance(color_space, pikepdf.Name):
        return {
            "/DeviceGray": 1,
            "/G": 1,
            "/DeviceRGB": 3,
            "/RGB": 3,
            "/DeviceCMYK": 4,
            "/CMYK": 4,
        }.get(str(color_space))
    if not isinstance(color_space, pikepdf.Array) or not color_space:
        return None
    family = resolve_indirect(color_space[0])
    if not isinstance(family, pikepdf.Name):
        return None
    family_name = str(family)
    if family_name in {"/Indexed", "/I", "/Separation", "/CalGray"}:
        return 1
    if family_name in {"/CalRGB", "/Lab"}:
        return 3
    if family_name == "/ICCBased" and len(color_space) >= 2:
        profile = resolve_indirect(color_space[1])
        components = (
            resolve_indirect(profile.get("/N"))
            if isinstance(profile, pikepdf.Stream)
            else None
        )
        if (
            isinstance(components, int)
            and not isinstance(components, bool)
            and 1 <= components <= 32
        ):
            return components
    if family_name == "/DeviceN" and len(color_space) >= 2:
        names = resolve_indirect(color_space[1])
        if isinstance(names, pikepdf.Array) and 1 <= len(names) <= 32:
            return len(names)
    return None


def _image_sample_geometry(
    image: pikepdf.Stream,
    components: int,
    *,
    stencil: bool = False,
) -> tuple[int, int, int, int] | None:
    width = resolve_indirect(image.get("/Width"))
    height = resolve_indirect(image.get("/Height"))
    if not _has_valid_image_dimensions(width, height):
        return None
    assert isinstance(width, int) and isinstance(height, int)
    if width * height > _MAX_IMAGE_VISIBILITY_PIXELS:
        return None
    raw_bpc = resolve_indirect(image.get("/BitsPerComponent"))
    if stencil and raw_bpc is None:
        bits_per_component = 1
    elif (
        isinstance(raw_bpc, int)
        and not isinstance(raw_bpc, bool)
        and raw_bpc in {1, 2, 4, 8, 16}
    ):
        bits_per_component = raw_bpc
    else:
        return None
    if stencil and bits_per_component != 1:
        return None
    row_bytes = (width * components * bits_per_component + 7) // 8
    decoded_bytes = row_bytes * height
    if decoded_bytes > _MAX_DECODED_IMAGE_VISIBILITY_BYTES:
        return None
    return width, height, bits_per_component, decoded_bytes


def _raw_image_stream_bytes(
    image: pikepdf.Stream,
    budget: _ImageVisibilityBudget,
    source_key: tuple[object, ...],
) -> bytes | None:
    declared_length = resolve_indirect(image.get("/Length"))
    if (
        not isinstance(declared_length, int)
        or isinstance(declared_length, bool)
        or declared_length < 0
        or declared_length > _MAX_ENCODED_IMAGE_VISIBILITY_BYTES
        or not budget._begin_encoded_stream(source_key, declared_length)
    ):
        return None
    try:
        raw_buffer = image.get_raw_stream_buffer()
    except Exception:
        return None
    if not budget._finish_encoded_stream(declared_length, len(raw_buffer)):
        return None
    return bytes(raw_buffer)


def _decoded_image_samples(
    image: pikepdf.Stream,
    decoded_limit: int,
    budget: _ImageVisibilityBudget,
    source_key: tuple[object, ...],
) -> bytes | None:
    raw = _raw_image_stream_bytes(image, budget, source_key)
    filters = _image_filter_names(image)
    if raw is None or filters is None:
        return None
    parameters = _image_decode_parameters(image, len(filters))
    if parameters is None:
        return None
    decoded: bytes | memoryview = raw
    decoders = {
        "/FlateDecode": _bounded_flate_decode,
        "/Fl": _bounded_flate_decode,
        "/ASCIIHexDecode": _bounded_ascii_hex_decode,
        "/AHx": _bounded_ascii_hex_decode,
        "/ASCII85Decode": _bounded_ascii85_decode,
        "/A85": _bounded_ascii85_decode,
        "/RunLengthDecode": _bounded_run_length_decode,
        "/RL": _bounded_run_length_decode,
    }
    try:
        for filter_name, decode_parameters in zip(filters, parameters):
            if decode_parameters is not None:
                predictor = resolve_indirect(decode_parameters.get("/Predictor", 1))
                if (
                    not isinstance(predictor, int)
                    or isinstance(predictor, bool)
                    or predictor != 1
                ):
                    return None
            decoder = decoders.get(filter_name)
            if decoder is None:
                return None
            decoded = decoder(decoded, decoded_limit)
    except ConversionError:
        return None
    if len(decoded) != decoded_limit:
        return None
    return bytes(decoded)


def _iter_image_samples(
    data: bytes,
    width: int,
    height: int,
    components: int,
    bits_per_component: int,
):
    samples_per_row = width * components
    row_bytes = (samples_per_row * bits_per_component + 7) // 8
    mask = (1 << bits_per_component) - 1
    for row_index in range(height):
        row = data[row_index * row_bytes : (row_index + 1) * row_bytes]
        if bits_per_component == 8:
            yield from row[:samples_per_row]
            continue
        if bits_per_component == 16:
            for offset in range(0, samples_per_row * 2, 2):
                yield int.from_bytes(row[offset : offset + 2], "big")
            continue
        for sample_index in range(samples_per_row):
            bit_offset = sample_index * bits_per_component
            byte_index = bit_offset // 8
            shift = 8 - bits_per_component - bit_offset % 8
            yield (row[byte_index] >> shift) & mask


def _decode_pair(image: pikepdf.Stream) -> tuple[float, float] | None:
    raw_decode = resolve_indirect(image.get("/Decode"))
    if raw_decode is None:
        return 0.0, 1.0
    if not isinstance(raw_decode, pikepdf.Array) or len(raw_decode) != 2:
        return None
    try:
        pair = tuple(float(resolve_indirect(value)) for value in raw_decode)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(value) for value in pair):
        return None
    return cast(tuple[float, float], pair)


@dataclass(slots=True)
class _ImageVisibilityBudget:
    cache: dict[tuple[object, ...], _ImageVisibility] = field(default_factory=dict)
    active: set[tuple[object, ...]] = field(default_factory=set)
    encoded_streams: set[tuple[object, ...]] = field(default_factory=set)
    encoded_bytes: int = 0
    decoded_bytes: int = 0
    decoded_samples: int = 0
    decodes: int = 0

    def _begin_encoded_stream(
        self,
        source_key: tuple[object, ...],
        declared_length: int,
    ) -> bool:
        if (
            source_key in self.encoded_streams
            or self.encoded_bytes + declared_length
            > _MAX_ENCODED_IMAGE_VISIBILITY_BYTES_PER_DOCUMENT
        ):
            return False
        self.encoded_streams.add(source_key)
        self.encoded_bytes += declared_length
        return True

    def _finish_encoded_stream(
        self,
        declared_length: int,
        actual_length: int,
    ) -> bool:
        self.encoded_bytes += actual_length - declared_length
        return (
            actual_length <= _MAX_ENCODED_IMAGE_VISIBILITY_BYTES
            and self.encoded_bytes <= _MAX_ENCODED_IMAGE_VISIBILITY_BYTES_PER_DOCUMENT
        )

    def _charge(self, decoded_bytes: int, decoded_samples: int) -> bool:
        if (
            self.decodes >= _MAX_IMAGE_VISIBILITY_DECODES_PER_DOCUMENT
            or decoded_bytes < 0
            or decoded_samples < 0
            or self.decoded_bytes + decoded_bytes
            > _MAX_IMAGE_VISIBILITY_BYTES_PER_DOCUMENT
            or self.decoded_samples + decoded_samples
            > _MAX_IMAGE_VISIBILITY_SAMPLES_PER_DOCUMENT
        ):
            return False
        self.decodes += 1
        self.decoded_bytes += decoded_bytes
        self.decoded_samples += decoded_samples
        return True

    def classify_stream(
        self,
        image: pikepdf.Stream,
        *,
        cache_key: tuple[object, ...] | None = None,
    ) -> _ImageVisibility:
        key = cache_key or cast(tuple[object, ...], _content_object_key(image))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if key in self.active:
            return "uncertain"
        self.active.add(key)
        try:
            visibility = _intrinsic_image_visibility(image, self, key)
        except Exception:
            visibility = "uncertain"
        finally:
            self.active.remove(key)
        self.cache[key] = visibility
        return visibility

    def classify_mask_stream(
        self,
        image: pikepdf.Stream,
        *,
        mode: Literal["soft_mask", "stencil_mask"],
    ) -> _ImageVisibility:
        key = (mode, *_content_object_key(image))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if key in self.active:
            return "uncertain"
        self.active.add(key)
        try:
            source_key = cast(tuple[object, ...], _content_object_key(image))
            visibility = (
                _soft_mask_visibility(image, self, source_key)
                if mode == "soft_mask"
                else _stencil_visibility(image, self, source_key)
            )
        except Exception:
            visibility = "uncertain"
        finally:
            self.active.remove(key)
        self.cache[key] = visibility
        return visibility

    def classify_inline(
        self,
        image: pikepdf.PdfInlineImage,
        cache_key: tuple[object, ...],
    ) -> _ImageVisibility:
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            raw = bytes(image._data._inline_image_raw_bytes())
        except Exception:
            self.cache[cache_key] = "uncertain"
            return "uncertain"
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        elif raw[-1:] in {b"\x00", b"\t", b"\n", b"\x0c", b"\r", b" "}:
            raw = raw[:-1]
        else:
            self.cache[cache_key] = "uncertain"
            return "uncertain"
        if len(raw) > _MAX_ENCODED_IMAGE_VISIBILITY_BYTES:
            self.cache[cache_key] = "uncertain"
            return "uncertain"
        try:
            with pikepdf.Pdf.new() as temporary_pdf:
                stream = temporary_pdf.make_stream(raw)
                stream["/Type"] = pikepdf.Name.XObject
                stream["/Subtype"] = pikepdf.Name.Image
                for name in image.obj.keys():
                    stream[name] = image.obj[name]
                visibility = self.classify_stream(stream, cache_key=cache_key)
        except Exception:
            visibility = "uncertain"
            self.cache[cache_key] = visibility
        return visibility


def _stencil_visibility(
    image: pikepdf.Stream,
    budget: _ImageVisibilityBudget,
    source_key: tuple[object, ...],
) -> _ImageVisibility:
    if resolve_indirect(image.get("/ImageMask", False)) is not True:
        return "uncertain"
    if (
        resolve_indirect(image.get("/Subtype")) != pikepdf.Name.Image
        or image.get("/ColorSpace") is not None
        or image.get("/SMask") is not None
        or image.get("/Mask") is not None
        or image.get("/SMaskInData") is not None
    ):
        return "uncertain"
    geometry = _image_sample_geometry(image, 1, stencil=True)
    decode = _decode_pair(image)
    if geometry is None or decode not in {(0.0, 1.0), (1.0, 0.0)}:
        return "uncertain"
    width, height, bits_per_component, decoded_bytes = geometry
    if not budget._charge(decoded_bytes, width * height):
        return "uncertain"
    data = _decoded_image_samples(image, decoded_bytes, budget, source_key)
    if data is None:
        return "uncertain"
    transparent_sample = 1 if decode == (0.0, 1.0) else 0
    return (
        "invisible"
        if all(
            sample == transparent_sample
            for sample in _iter_image_samples(
                data,
                width,
                height,
                1,
                bits_per_component,
            )
        )
        else "visible"
    )


def _soft_mask_visibility(
    image: pikepdf.Stream,
    budget: _ImageVisibilityBudget,
    source_key: tuple[object, ...],
) -> _ImageVisibility:
    if (
        resolve_indirect(image.get("/Subtype")) != pikepdf.Name.Image
        or resolve_indirect(image.get("/ImageMask", False)) is not False
        or resolve_indirect(image.get("/ColorSpace"))
        not in {pikepdf.Name.DeviceGray, pikepdf.Name("/G")}
        or image.get("/SMask") is not None
        or image.get("/Mask") is not None
        or image.get("/SMaskInData") is not None
    ):
        return "uncertain"
    geometry = _image_sample_geometry(image, 1)
    decode = _decode_pair(image)
    if geometry is None or decode is None:
        return "uncertain"
    width, height, bits_per_component, decoded_bytes = geometry
    if not budget._charge(decoded_bytes, width * height):
        return "uncertain"
    data = _decoded_image_samples(image, decoded_bytes, budget, source_key)
    if data is None:
        return "uncertain"
    maximum_sample = (1 << bits_per_component) - 1
    d_min, d_max = decode
    for sample in _iter_image_samples(
        data,
        width,
        height,
        1,
        bits_per_component,
    ):
        alpha = d_min + sample / maximum_sample * (d_max - d_min)
        if min(1.0, max(0.0, alpha)) > 0:
            return "visible"
    return "invisible"


def _color_key_visibility(
    image: pikepdf.Stream,
    mask: pikepdf.Array,
    budget: _ImageVisibilityBudget,
    source_key: tuple[object, ...],
) -> _ImageVisibility:
    components = _image_component_count(image)
    if components is None or len(mask) != components * 2:
        return "uncertain"
    geometry = _image_sample_geometry(image, components)
    if geometry is None:
        return "uncertain"
    width, height, bits_per_component, decoded_bytes = geometry
    maximum_sample = (1 << bits_per_component) - 1
    ranges: list[tuple[int, int]] = []
    for index in range(0, len(mask), 2):
        lower = resolve_indirect(mask[index])
        upper = resolve_indirect(mask[index + 1])
        if (
            not isinstance(lower, int)
            or isinstance(lower, bool)
            or not isinstance(upper, int)
            or isinstance(upper, bool)
            or not 0 <= lower <= upper <= maximum_sample
        ):
            return "uncertain"
        ranges.append((lower, upper))
    if not budget._charge(decoded_bytes, width * height * components):
        return "uncertain"
    data = _decoded_image_samples(image, decoded_bytes, budget, source_key)
    if data is None:
        return "uncertain"
    samples = iter(
        _iter_image_samples(
            data,
            width,
            height,
            components,
            bits_per_component,
        )
    )
    for _pixel in range(width * height):
        pixel = tuple(next(samples) for _range in ranges)
        if any(
            not lower <= sample <= upper
            for sample, (lower, upper) in zip(pixel, ranges)
        ):
            return "visible"
    return "invisible"


def _jpx_alpha_visibility(
    image: pikepdf.Stream,
    budget: _ImageVisibilityBudget,
    source_key: tuple[object, ...],
) -> _ImageVisibility:
    filters = _image_filter_names(image)
    width = resolve_indirect(image.get("/Width"))
    height = resolve_indirect(image.get("/Height"))
    raw = _raw_image_stream_bytes(image, budget, source_key)
    if (
        filters != ("/JPXDecode",)
        or not _has_valid_image_dimensions(width, height)
        or raw is None
    ):
        return "uncertain"
    assert isinstance(width, int) and isinstance(height, int)
    if width * height > _MAX_IMAGE_VISIBILITY_PIXELS:
        return "uncertain"
    decoded_bytes = width * height * 4
    if decoded_bytes > _MAX_DECODED_IMAGE_VISIBILITY_BYTES:
        return "uncertain"
    if not budget._charge(decoded_bytes, width * height * 4):
        return "uncertain"
    try:
        from PIL import Image

        with Image.open(BytesIO(raw)) as decoded:
            if (
                decoded.size != (width, height)
                or decoded.width * decoded.height > _MAX_IMAGE_VISIBILITY_PIXELS
                or "A" not in decoded.getbands()
            ):
                return "uncertain"
            decoded.load()
            extrema = decoded.getchannel("A").getextrema()
    except Exception:
        return "uncertain"
    maximum = extrema[1] if isinstance(extrema, tuple) else extrema
    return "invisible" if maximum == 0 else "visible"


def _intrinsic_image_visibility(
    image: pikepdf.Stream,
    budget: _ImageVisibilityBudget,
    source_key: tuple[object, ...],
) -> _ImageVisibility:
    if resolve_indirect(image.get("/Subtype")) != pikepdf.Name.Image:
        return "uncertain"
    width = resolve_indirect(image.get("/Width"))
    height = resolve_indirect(image.get("/Height"))
    if not _has_valid_image_dimensions(width, height):
        return "uncertain"

    raw_image_mask = resolve_indirect(image.get("/ImageMask", False))
    if not isinstance(raw_image_mask, bool):
        return "uncertain"
    soft_mask = resolve_indirect(image.get("/SMask"))
    explicit_mask = resolve_indirect(image.get("/Mask"))
    raw_smask_in_data = resolve_indirect(image.get("/SMaskInData", 0))
    if (
        not isinstance(raw_smask_in_data, int)
        or isinstance(raw_smask_in_data, bool)
        or raw_smask_in_data not in {0, 1, 2}
    ):
        return "uncertain"
    intrinsic_sources = sum(
        (
            soft_mask is not None,
            explicit_mask is not None,
            raw_smask_in_data != 0,
        )
    )
    if raw_image_mask:
        if intrinsic_sources:
            return "uncertain"
        return _stencil_visibility(image, budget, source_key)
    if intrinsic_sources > 1:
        return "uncertain"
    if soft_mask is not None:
        return (
            budget.classify_mask_stream(soft_mask, mode="soft_mask")
            if isinstance(soft_mask, pikepdf.Stream)
            and resolve_indirect(soft_mask.get("/Width")) == width
            and resolve_indirect(soft_mask.get("/Height")) == height
            else "uncertain"
        )
    if isinstance(explicit_mask, pikepdf.Stream):
        return (
            budget.classify_mask_stream(explicit_mask, mode="stencil_mask")
            if resolve_indirect(explicit_mask.get("/Width")) == width
            and resolve_indirect(explicit_mask.get("/Height")) == height
            else "uncertain"
        )
    if isinstance(explicit_mask, pikepdf.Array):
        return _color_key_visibility(image, explicit_mask, budget, source_key)
    if explicit_mask is not None:
        return "uncertain"
    if raw_smask_in_data:
        return _jpx_alpha_visibility(image, budget, source_key)
    return "visible"


def _valid_direct_text_operands(operator_name: str, operands: pikepdf.Array) -> bool:
    if operator_name in {"Tj", "'"}:
        return len(operands) == 1 and isinstance(operands[0], pikepdf.String)
    if operator_name == "TJ":
        if len(operands) != 1 or not isinstance(operands[0], pikepdf.Array):
            return False
        return all(
            isinstance(item, pikepdf.String) or _is_finite_pdf_number(item)
            for item in operands[0]
        )
    if operator_name == '"':
        return (
            len(operands) == 3
            and _is_finite_pdf_number(operands[0])
            and _is_finite_pdf_number(operands[1])
            and isinstance(operands[2], pikepdf.String)
        )
    return True


@dataclass(frozen=True, slots=True)
class TextRun:
    """Text and effective style produced by one text-show operation."""

    text: str
    bbox: BBox
    font_name: str | None
    font_size: float
    render_mode: int
    invisible: bool
    final_paint_uncertain: bool


@dataclass(frozen=True, slots=True)
class DirectTextSpan(TextRun):
    """A text-show operation executed directly by one content container."""

    direct_text_index: int
    clip_bbox: BBox | None = None
    clip_polygon: ClipPolygon | None = None


@dataclass(frozen=True, slots=True)
class DirectXObjectSpan:
    """One direct Form, Image, or inline-image invocation."""

    kind: XObjectKind
    bbox: BBox
    text_runs: tuple[TextRun, ...]
    direct_xobject_index: int
    resource_name: str | None
    matrix: Matrix
    children: tuple[DirectTextSpan | DirectXObjectSpan, ...] = ()
    clip_bbox: BBox | None = None
    clip_polygon: ClipPolygon | None = None
    entry_state: InvocationPaintState = InvocationPaintState()
    invisible: bool = False
    final_paint_uncertain: bool = False
    intrinsic_visibility_uncertain: bool = False
    non_intrinsic_visibility_uncertain: bool = False

    @property
    def text(self) -> str:
        """Return text painted inside the invocation in execution order."""

        return "".join(run.text for run in self.text_runs)

    def local_to_page_bbox(self, bbox: BBox) -> BBox:
        """Transform a local Form box to pdfminer page coordinates."""

        return _bbox(apply_matrix_rect(self.matrix, bbox))


PaintingSpan = DirectTextSpan | DirectXObjectSpan


@dataclass(frozen=True, slots=True)
class DigitalPageLayout:
    """Neutral direct-painting information for one zero-based PDF page."""

    page_index: int
    width: float
    height: float
    spans: tuple[PaintingSpan, ...]


@dataclass(slots=True)
class _XObjectContext:
    kind: XObjectKind
    bbox: BBox
    resource_name: str | None
    matrix: Matrix
    clip_bbox: BBox | None
    clip_polygon: ClipPolygon | None
    entry_state: InvocationPaintState
    invisible: bool = False
    final_paint_uncertain: bool = False
    non_intrinsic_visibility_uncertain: bool = False
    text_runs: list[TextRun] = field(default_factory=list)
    children: list[PaintingSpan] = field(default_factory=list)
    direct_text_index: int = 0
    direct_xobject_index: int = 0


def _bbox(rect: Rect) -> BBox:
    x0, y0, x1, y1 = (float(value) for value in rect)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _bbox_union(current: BBox | None, update: BBox) -> BBox:
    if current is None:
        return update
    return (
        min(current[0], update[0]),
        min(current[1], update[1]),
        max(current[2], update[2]),
        max(current[3], update[3]),
    )


def _font_name(textstate: PDFTextState, characters: tuple[LTChar, ...]) -> str | None:
    value = (
        characters[0].fontname
        if characters
        else getattr(textstate.font, "fontname", None)
    )
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace")
    return str(value)


def _font_size(
    textstate: PDFTextState,
    characters: tuple[LTChar, ...],
    text_matrix: Matrix,
) -> float:
    if characters:
        return abs(float(characters[0].size))
    font = textstate.font
    if font is not None and font.is_vertical():
        scale = hypot(text_matrix[0], text_matrix[1])
    else:
        scale = hypot(text_matrix[2], text_matrix[3])
    return abs(float(textstate.fontsize)) * scale


def _make_text_run(
    textstate: PDFTextState,
    graphicstate: PDFGraphicState,
    characters: tuple[LTChar, ...],
    empty_bbox: BBox,
    text_matrix: Matrix,
    paint_state: InvocationPaintState,
) -> TextRun:
    render_mode = int(textstate.render)
    uses_fill = render_mode in {0, 2, 4, 6}
    uses_stroke = render_mode in {1, 2, 5, 6}
    transparent = (not uses_fill or paint_state.fill_alpha <= 0) and (
        not uses_stroke or paint_state.stroke_alpha <= 0
    )

    def device_cmyk_has_ink(*, stroke: bool) -> bool:
        color_space = graphicstate.scs if stroke else graphicstate.ncs
        color = graphicstate.scolor if stroke else graphicstate.ncolor
        if getattr(color_space, "name", None) != "DeviceCMYK" or not isinstance(
            color, tuple
        ):
            return False
        values = tuple(safe_float(value) for value in color)
        return (
            len(values) == 4
            and all(value is not None and isfinite(value) for value in values)
            and any(value is not None and value > 0 for value in values)
        )

    fill_uncertain = (
        uses_fill
        and paint_state.fill_alpha > 0
        and (
            paint_state.soft_mask_active
            or (paint_state.fill_overprint and not device_cmyk_has_ink(stroke=False))
            or paint_state.fill_color_complex
            or paint_state.blend_mode_complex
        )
    )
    stroke_uncertain = (
        uses_stroke
        and paint_state.stroke_alpha > 0
        and (
            paint_state.soft_mask_active
            or paint_state.stroke_adjust
            or paint_state.line_width == 0
            or (paint_state.stroke_overprint and not device_cmyk_has_ink(stroke=True))
            or paint_state.stroke_color_complex
            or paint_state.blend_mode_complex
            or bool(paint_state.dash_array)
        )
    )
    proven_visible_channel = (
        uses_fill and paint_state.fill_alpha > 0 and not fill_uncertain
    ) or (uses_stroke and paint_state.stroke_alpha > 0 and not stroke_uncertain)
    has_visible_channel = (uses_fill and paint_state.fill_alpha > 0) or (
        uses_stroke and paint_state.stroke_alpha > 0
    )
    uncertain = (
        (fill_uncertain or stroke_uncertain) and not proven_visible_channel
    ) or (paint_state.clip_visibility_uncertain and has_visible_channel)
    return TextRun(
        text="".join(character.get_text() for character in characters),
        bbox=(
            _bbox(
                (
                    min(character.x0 for character in characters),
                    min(character.y0 for character in characters),
                    max(character.x1 for character in characters),
                    max(character.y1 for character in characters),
                )
            )
            if characters
            else empty_bbox
        ),
        font_name=_font_name(textstate, characters),
        font_size=_font_size(textstate, characters, text_matrix),
        render_mode=render_mode,
        invisible=render_mode in {3, 7} or transparent,
        final_paint_uncertain=uncertain,
    )


def _base14_cid_fallback(
    textstate: PDFTextState,
    seq: PDFTextSeq,
    run: TextRun,
    origin_matrix: Matrix,
) -> tuple[TextRun, float] | None:
    font = textstate.font
    font_name = run.font_name
    if (
        font is None
        or font_name not in _TEXT_BASE14_FONTS
        or font.is_multibyte()
        or font.is_vertical()
        or getattr(font, "cid2unicode", None) != {}
        or not _CID_FALLBACK.fullmatch(run.text)
        or (run.bbox[0] != run.bbox[2] and run.bbox[1] != run.bbox[3])
    ):
        return None
    raw = b"".join(item for item in seq if isinstance(item, bytes))
    if not raw or any(value < 32 or value > 126 for value in raw):
        return None
    cids = tuple(int(value) for value in re.findall(r"\(cid:([0-9]+)\)", run.text))
    if cids != tuple(raw):
        return None
    text = raw.decode("ascii")
    _properties, widths = FONT_METRICS[font_name]
    if any(character not in widths for character in text):
        return None

    font_size = float(textstate.fontsize)
    scaling = float(textstate.scaling) * 0.01
    charspace = float(textstate.charspace) * scaling
    wordspace = float(textstate.wordspace) * scaling
    glyph_advance = 0.0
    total_advance = 0.0
    needs_charspace = False
    for item in seq:
        if isinstance(item, (int, float)):
            total_advance -= float(item) * 0.001 * font_size * scaling
            needs_charspace = True
            continue
        if not isinstance(item, bytes):
            return None
        for value in item:
            if needs_charspace:
                total_advance += charspace
            advance = float(widths[chr(value)]) * 0.001 * font_size * scaling
            glyph_advance += advance
            total_advance += advance
            if value == 32:
                total_advance += wordspace
            needs_charspace = True
    if glyph_advance <= 0 or total_advance <= 0:
        return None
    delta_x = origin_matrix[0] * total_advance
    delta_y = origin_matrix[1] * total_advance
    translated_bbox = _bbox(
        (
            run.bbox[0] + delta_x,
            run.bbox[1] + delta_y,
            run.bbox[2] + delta_x,
            run.bbox[3] + delta_y,
        )
    )
    return (
        TextRun(
            text=text,
            bbox=_bbox_union(run.bbox, translated_bbox),
            font_name=run.font_name,
            font_size=(
                run.font_size
                or abs(font_size) * hypot(origin_matrix[2], origin_matrix[3])
            ),
            render_mode=run.render_mode,
            invisible=run.invisible,
            final_paint_uncertain=run.final_paint_uncertain,
        ),
        glyph_advance,
    )


class _ProvenanceDevice(PDFPageAggregator):
    def __init__(self, rsrcmgr: PDFResourceManager) -> None:
        super().__init__(rsrcmgr, laparams=None)
        self.pages: list[DigitalPageLayout] = []
        self._page_spans: list[PaintingSpan] = []
        self._direct_text_index = 0
        self._direct_xobject_index = 0
        self._ctm_stack: list[Matrix] = []
        self._clip_bbox: BBox | None = None
        self._clip_polygon: ClipPolygon | None = None
        self._clip_stack: list[tuple[BBox | None, ClipPolygon | None]] = []
        self._text_clip_stack: list[BBox | None] = []
        self._xobject_stack: list[_XObjectContext] = []
        self._next_entry_state: InvocationPaintState | None = None
        self._paint_state = InvocationPaintState()
        self.page_index: int | None = None

    def _current_ctm(self) -> Matrix:
        if self.ctm is None:
            raise PDFInterpreterError("Missing current transformation matrix")
        return self.ctm

    def begin_page(self, page: PDFPage, ctm: Matrix) -> None:
        if self._xobject_stack:
            raise PDFInterpreterError("Unclosed figure before page start")
        self._page_spans = []
        self._direct_text_index = 0
        self._direct_xobject_index = 0
        self._clip_bbox = None
        self._clip_polygon = _rect_polygon(ctm, page.cropbox)
        self._clip_stack = []
        self._text_clip_stack = []
        self._next_entry_state = None
        self._paint_state = InvocationPaintState()
        self.set_ctm(ctm)
        super().begin_page(page, ctm)

    def set_clip(
        self,
        bbox: BBox | None,
        polygon: ClipPolygon | None,
    ) -> None:
        self._clip_bbox = bbox
        self._clip_polygon = polygon

    def set_next_entry_state(self, state: InvocationPaintState) -> None:
        if self._next_entry_state is not None:
            raise PDFInterpreterError("Unconsumed XObject graphics state")
        self._next_entry_state = state

    def set_text_paint_state(self, state: InvocationPaintState) -> None:
        self._paint_state = state

    def begin_text_object(self) -> None:
        self._text_clip_stack.append(None)

    def end_text_object(self) -> BBox | None:
        if not self._text_clip_stack:
            raise PDFInterpreterError("Text-object stack underflow")
        return self._text_clip_stack.pop()

    def end_page(self, page: PDFPage) -> None:
        if self._xobject_stack:
            raise PDFInterpreterError("Unclosed figure at page end")
        if self._text_clip_stack:
            raise PDFInterpreterError("Unclosed text object at page end")
        if self.page_index is None:
            raise PDFInterpreterError("Missing physical page index")
        layout = cast(LTPage, self.cur_item)
        result = DigitalPageLayout(
            page_index=self.page_index,
            width=float(layout.width),
            height=float(layout.height),
            spans=tuple(self._page_spans),
        )
        super().end_page(page)
        self.pages.append(result)

    def begin_figure(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        current_ctm = self._current_ctm()
        self._ctm_stack.append(current_ctm)
        effective_matrix = mult_matrix(matrix, current_ctm)
        painted_polygon = _rect_polygon(effective_matrix, bbox)
        painted_bbox = _polygon_bbox(painted_polygon) or (0.0, 0.0, 0.0, 0.0)
        self._clip_stack.append((self._clip_bbox, self._clip_polygon))
        figure_polygon = _polygon_intersection(self._clip_polygon, painted_polygon)
        figure_clip = _polygon_bbox(figure_polygon)
        entry_state = self._next_entry_state or InvocationPaintState()
        self._next_entry_state = None
        self._clip_bbox = figure_clip
        self._clip_polygon = figure_polygon
        self._xobject_stack.append(
            _XObjectContext(
                kind="form",
                bbox=painted_bbox,
                resource_name=str(name).removeprefix("/"),
                matrix=effective_matrix,
                clip_bbox=figure_clip,
                clip_polygon=figure_polygon,
                entry_state=entry_state,
            )
        )
        super().begin_figure(name, bbox, matrix)

    def end_figure(self, name: str) -> None:
        if not self._xobject_stack or not self._ctm_stack or not self._clip_stack:
            raise PDFInterpreterError("Figure stack underflow")
        super().end_figure(name)
        self.set_ctm(self._ctm_stack.pop())
        context = self._xobject_stack.pop()
        parent = self._xobject_stack[-1] if self._xobject_stack else None
        direct_xobject_index = (
            parent.direct_xobject_index
            if parent is not None
            else self._direct_xobject_index
        )
        invocation = DirectXObjectSpan(
            kind=context.kind,
            bbox=context.bbox,
            text_runs=tuple(context.text_runs),
            direct_xobject_index=direct_xobject_index,
            resource_name=context.resource_name,
            matrix=context.matrix,
            children=tuple(context.children) if context.kind == "form" else (),
            clip_bbox=context.clip_bbox,
            clip_polygon=context.clip_polygon,
            entry_state=context.entry_state,
            invisible=context.invisible,
            final_paint_uncertain=context.final_paint_uncertain,
            non_intrinsic_visibility_uncertain=(
                context.non_intrinsic_visibility_uncertain
            ),
        )
        if parent is not None:
            parent.children.append(invocation)
            parent.text_runs.extend(invocation.text_runs)
            parent.direct_xobject_index += 1
        else:
            self._page_spans.append(invocation)
            self._direct_xobject_index += 1
        self._clip_bbox, self._clip_polygon = self._clip_stack.pop()

    def render_image(self, name: str, stream: PDFStream) -> None:
        if not self._xobject_stack:
            raise PDFInterpreterError("Missing direct image context")
        context = self._xobject_stack[-1]
        if stream.objid is None:
            context.kind = "inline_image"
            context.resource_name = None
        else:
            context.kind = "image"
        context.invisible = context.entry_state.fill_alpha <= 0
        image_mask = resolve1(stream.get("ImageMask", stream.get("IM", False))) is True
        context.non_intrinsic_visibility_uncertain = (
            context.entry_state.fill_alpha > 0
            and (
                context.entry_state.soft_mask_active
                or context.entry_state.clip_visibility_uncertain
                or context.entry_state.fill_overprint
                or (image_mask and context.entry_state.fill_color_complex)
                or context.entry_state.blend_mode_complex
            )
        )
        context.final_paint_uncertain = context.non_intrinsic_visibility_uncertain
        super().render_image(name, stream)

    def render_string(
        self,
        textstate: PDFTextState,
        seq: PDFTextSeq,
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState,
    ) -> None:
        container = cast(LTLayoutContainer, self.cur_item)
        first_new_child = len(container)
        text_matrix = mult_matrix(textstate.matrix, self._current_ctm())
        origin_matrix = translate_matrix(text_matrix, textstate.linematrix)
        empty_bbox = _bbox(
            (
                origin_matrix[4],
                origin_matrix[5],
                origin_matrix[4],
                origin_matrix[5],
            )
        )
        super().render_string(textstate, seq, ncs, graphicstate)
        characters = tuple(
            item
            for item in container._objs[first_new_child:]
            if isinstance(item, LTChar)
        )
        run = _make_text_run(
            textstate,
            graphicstate,
            characters,
            empty_bbox,
            text_matrix,
            self._paint_state,
        )
        fallback = _base14_cid_fallback(textstate, seq, run, origin_matrix)
        if fallback is not None:
            run, glyph_advance = fallback
            line_x, line_y = textstate.linematrix
            textstate.linematrix = (line_x + glyph_advance, line_y)
        has_text_glyphs = any(isinstance(item, bytes) and item for item in seq)
        if run.render_mode in {4, 5, 6, 7} and has_text_glyphs:
            if not self._text_clip_stack:
                raise PDFInterpreterError("Text clipping outside a text object")
            self._text_clip_stack[-1] = _bbox_union(
                self._text_clip_stack[-1],
                run.bbox,
            )
        if self._xobject_stack:
            context = self._xobject_stack[-1]
            context.text_runs.append(run)
            context.children.append(
                DirectTextSpan(
                    text=run.text,
                    bbox=run.bbox,
                    font_name=run.font_name,
                    font_size=run.font_size,
                    render_mode=run.render_mode,
                    invisible=run.invisible,
                    final_paint_uncertain=run.final_paint_uncertain,
                    direct_text_index=context.direct_text_index,
                    clip_bbox=self._clip_bbox,
                    clip_polygon=self._clip_polygon,
                )
            )
            context.direct_text_index += 1
            return
        self._page_spans.append(
            DirectTextSpan(
                text=run.text,
                bbox=run.bbox,
                font_name=run.font_name,
                font_size=run.font_size,
                render_mode=run.render_mode,
                invisible=run.invisible,
                final_paint_uncertain=run.final_paint_uncertain,
                direct_text_index=self._direct_text_index,
                clip_bbox=self._clip_bbox,
                clip_polygon=self._clip_polygon,
            )
        )
        self._direct_text_index += 1


class _ProvenanceInterpreter(PDFPageInterpreter):
    def init_state(self, ctm: Matrix) -> None:
        super().init_state(ctm)
        inherited_textstate = getattr(self, "_inherited_textstate", None)
        inherited_graphicstate = getattr(self, "_inherited_graphicstate", None)
        inherited_paint_state = getattr(self, "_inherited_paint_state", None)
        if isinstance(inherited_textstate, PDFTextState):
            self.textstate = inherited_textstate.copy()
            self.textstate.reset()
        if isinstance(inherited_graphicstate, PDFGraphicState):
            self.graphicstate = inherited_graphicstate.copy()
        self._paint_state = (
            inherited_paint_state
            if isinstance(inherited_paint_state, InvocationPaintState)
            else InvocationPaintState()
        )
        device = cast(_ProvenanceDevice, self.device)
        self._clip_bbox = device._clip_bbox
        self._clip_polygon = device._clip_polygon
        self._clip_is_exact = not self._paint_state.clip_visibility_uncertain
        self._clip_stack: list[
            tuple[
                BBox | None,
                ClipPolygon | None,
                bool,
                InvocationPaintState,
            ]
        ] = []
        self._clip_pending = False
        self._pending_clip_bbox: BBox = (0.0, 0.0, 0.0, 0.0)
        self._pending_clip_polygon: ClipPolygon = ()
        self._pending_clip_is_exact = True

    def subinterp(self) -> PDFPageInterpreter:
        interpreter = cast(_ProvenanceInterpreter, super().subinterp())
        interpreter._inherited_textstate = self.textstate.copy()
        interpreter._inherited_graphicstate = self.graphicstate.copy()
        interpreter._inherited_paint_state = self._paint_state
        return interpreter

    def _set_clip(
        self,
        bbox: BBox | None,
        polygon: ClipPolygon | None,
    ) -> None:
        self._clip_bbox = bbox
        self._clip_polygon = polygon
        cast(_ProvenanceDevice, self.device).set_clip(bbox, polygon)

    def _rectangular_clip_polygon(self) -> ClipPolygon:
        if not self.curpath:
            raise PDFInterpreterError("Clipping operator has no current path")
        segments = list(self.curpath)
        while segments and segments[-1][0] == "h":
            segments.pop()
        if (
            len(segments) not in {4, 5}
            or segments[0][0] != "m"
            or any(segment[0] != "l" for segment in segments[1:])
        ):
            raise _NonRectangularClipError(
                "Non-rectangular clipping path cannot be represented safely"
            )
        points: list[Point] = []
        for segment in segments:
            x = safe_float(segment[1])
            y = safe_float(segment[2])
            if x is None or y is None:
                raise PDFInterpreterError("Malformed clipping path")
            points.append((x, y))
        coordinates = tuple(value for point in points for value in point)
        extent = max(
            max(point[0] for point in points) - min(point[0] for point in points),
            max(point[1] for point in points) - min(point[1] for point in points),
            1.0,
        )
        tolerance = max(
            1e-7,
            extent * 1e-9,
            max(ulp(abs(value)) for value in coordinates) * 4,
        )

        def close(first: float, second: float) -> bool:
            return abs(first - second) <= tolerance

        if len(points) == 5:
            if not all(close(a, b) for a, b in zip(points[0], points[-1])):
                raise _NonRectangularClipError(
                    "Non-rectangular clipping path cannot be represented safely"
                )
            points.pop()

        if close(
            min(point[0] for point in points),
            max(point[0] for point in points),
        ) or close(
            min(point[1] for point in points),
            max(point[1] for point in points),
        ):
            return ()

        polygon = _parallelogram_polygon(
            tuple(apply_matrix_pt(self.ctm, point) for point in points)
        )
        if polygon is None:
            raise _NonRectangularClipError(
                "Non-rectangular clipping path cannot be represented safely"
            )
        return polygon

    def _commit_clip(self) -> None:
        if not self._clip_pending:
            return
        current_is_empty = self._clip_polygon == ()
        pending_is_empty = (
            self._pending_clip_is_exact and self._pending_clip_polygon == ()
        )
        if current_is_empty or pending_is_empty:
            self._clip_is_exact = True
            self._set_clip((0.0, 0.0, 0.0, 0.0), ())
        elif self._clip_is_exact and self._pending_clip_is_exact:
            polygon = _polygon_intersection(
                self._clip_polygon,
                self._pending_clip_polygon,
            )
            self._set_clip(_polygon_bbox(polygon), polygon)
        else:
            self._clip_is_exact = False
        self._clip_pending = False

    def _operation_paint_state(self) -> InvocationPaintState:
        return replace(
            self._paint_state,
            clip_visibility_uncertain=not self._clip_is_exact,
        )

    def do_q(self) -> None:
        self._clip_stack.append(
            (
                self._clip_bbox,
                self._clip_polygon,
                self._clip_is_exact,
                self._paint_state,
            )
        )
        super().do_q()

    def do_Q(self) -> None:  # noqa: N802
        super().do_Q()
        if self._clip_stack:
            (
                clip_bbox,
                clip_polygon,
                self._clip_is_exact,
                self._paint_state,
            ) = self._clip_stack.pop()
            self._set_clip(clip_bbox, clip_polygon)

    def do_BT(self) -> None:  # noqa: N802
        cast(_ProvenanceDevice, self.device).begin_text_object()
        super().do_BT()

    def do_ET(self) -> None:  # noqa: N802
        super().do_ET()
        text_clip_bbox = cast(_ProvenanceDevice, self.device).end_text_object()
        if text_clip_bbox is not None and self._clip_bbox != (0.0, 0.0, 0.0, 0.0):
            self._clip_is_exact = False

    def do_W(self) -> None:  # noqa: N802
        try:
            self._pending_clip_polygon = self._rectangular_clip_polygon()
            self._pending_clip_bbox = _polygon_bbox(self._pending_clip_polygon) or (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            self._pending_clip_is_exact = True
        except _NonRectangularClipError:
            self._pending_clip_is_exact = False
        self._clip_pending = True

    def do_W_a(self) -> None:  # noqa: N802
        self.do_W()

    def do_n(self) -> None:
        self._commit_clip()
        super().do_n()

    def do_S(self) -> None:  # noqa: N802
        self._commit_clip()
        super().do_S()

    def do_s(self) -> None:
        self._commit_clip()
        super().do_s()

    def do_f(self) -> None:
        self._commit_clip()
        super().do_f()

    def do_F(self) -> None:  # noqa: N802
        self._commit_clip()
        super().do_f()

    def do_f_a(self) -> None:
        self._commit_clip()
        super().do_f_a()

    def do_B(self) -> None:  # noqa: N802
        self._commit_clip()
        super().do_B()

    def do_B_a(self) -> None:  # noqa: N802
        self._commit_clip()
        super().do_B_a()

    def do_b(self) -> None:
        self._commit_clip()
        super().do_b()

    def do_b_a(self) -> None:
        self._commit_clip()
        super().do_b_a()

    def do_w(self, linewidth: PDFStackT) -> None:
        value = safe_float(linewidth)
        if value is None or not isfinite(value) or value < 0:
            raise PDFInterpreterError("Invalid line width")
        self._paint_state = replace(self._paint_state, line_width=value)
        super().do_w(linewidth)

    def do_J(self, linecap: PDFStackT) -> None:  # noqa: N802
        value = safe_int(linecap)
        if value not in {0, 1, 2}:
            raise PDFInterpreterError("Invalid line cap")
        self._paint_state = replace(self._paint_state, line_cap=value)
        super().do_J(linecap)

    def do_j(self, linejoin: PDFStackT) -> None:
        value = safe_int(linejoin)
        if value not in {0, 1, 2}:
            raise PDFInterpreterError("Invalid line join")
        self._paint_state = replace(self._paint_state, line_join=value)
        super().do_j(linejoin)

    def do_M(self, miterlimit: PDFStackT) -> None:  # noqa: N802
        value = safe_float(miterlimit)
        if value is None or not isfinite(value) or value < 1:
            raise PDFInterpreterError("Invalid miter limit")
        self._paint_state = replace(self._paint_state, miter_limit=value)
        super().do_M(miterlimit)

    def do_d(self, dash: PDFStackT, phase: PDFStackT) -> None:
        if not isinstance(dash, list):
            raise PDFInterpreterError("Invalid line dash pattern")
        dash_values = tuple(safe_float(value) for value in dash)
        phase_value = safe_float(phase)
        if (
            phase_value is None
            or not isfinite(phase_value)
            or any(
                value is None or not isfinite(value) or value < 0
                for value in dash_values
            )
            or (dash_values and not any(value for value in dash_values))
        ):
            raise PDFInterpreterError("Invalid line dash pattern")
        self._paint_state = replace(
            self._paint_state,
            dash_array=cast(tuple[float, ...], dash_values),
            dash_phase=phase_value,
        )
        super().do_d(dash, phase)

    def _color_space_is_complex(self, name: PDFStackT) -> bool:
        try:
            resource_name = literal_name(name)
        except Exception as exc:
            raise PDFInterpreterError("Invalid color space name") from exc
        device_spaces = {
            "DeviceGray",
            "DeviceRGB",
            "DeviceCMYK",
            "G",
            "RGB",
            "CMYK",
        }
        if resource_name in device_spaces:
            return False
        try:
            color_space = resolve1(
                dict_value(self.resources.get("ColorSpace"))[resource_name]
            )
        except Exception:
            return True
        try:
            if isinstance(color_space, (list, tuple)) and color_space:
                family = literal_name(resolve1(color_space[0]))
                return family not in {
                    "CalGray",
                    "CalRGB",
                    "Lab",
                    "ICCBased",
                    "Indexed",
                    "I",
                }
            return literal_name(color_space) not in device_spaces
        except Exception:
            return True

    def do_CS(self, name: PDFStackT) -> None:  # noqa: N802
        super().do_CS(name)
        self._paint_state = replace(
            self._paint_state,
            stroke_color_complex=self._color_space_is_complex(name),
        )

    def do_cs(self, name: PDFStackT) -> None:
        super().do_cs(name)
        self._paint_state = replace(
            self._paint_state,
            fill_color_complex=self._color_space_is_complex(name),
        )

    def do_G(self, gray: PDFStackT) -> None:  # noqa: N802
        super().do_G(gray)
        self._paint_state = replace(self._paint_state, stroke_color_complex=False)

    def do_g(self, gray: PDFStackT) -> None:
        super().do_g(gray)
        self._paint_state = replace(self._paint_state, fill_color_complex=False)

    def do_RG(self, r: PDFStackT, g: PDFStackT, b: PDFStackT) -> None:  # noqa: N802
        super().do_RG(r, g, b)
        self._paint_state = replace(self._paint_state, stroke_color_complex=False)

    def do_rg(self, r: PDFStackT, g: PDFStackT, b: PDFStackT) -> None:
        super().do_rg(r, g, b)
        self._paint_state = replace(self._paint_state, fill_color_complex=False)

    def do_K(  # noqa: N802
        self,
        c: PDFStackT,
        m: PDFStackT,
        y: PDFStackT,
        k: PDFStackT,
    ) -> None:
        super().do_K(c, m, y, k)
        self._paint_state = replace(self._paint_state, stroke_color_complex=False)

    def do_k(
        self,
        c: PDFStackT,
        m: PDFStackT,
        y: PDFStackT,
        k: PDFStackT,
    ) -> None:
        super().do_k(c, m, y, k)
        self._paint_state = replace(self._paint_state, fill_color_complex=False)

    def do_gs(self, name: PDFStackT) -> None:
        try:
            resource_name = literal_name(name)
            parameters = resolve1(
                dict_value(self.resources.get("ExtGState"))[resource_name]
            )
        except Exception as exc:
            raise PDFInterpreterError("Invalid ExtGState resource reference") from exc
        if not isinstance(parameters, dict):
            raise PDFInterpreterError("ExtGState resource is not a dictionary")

        state = self._paint_state

        def number(key: str, *, minimum: float, maximum: float | None = None):
            if key not in parameters:
                return None
            value = safe_float(resolve1(parameters[key]))
            if (
                value is None
                or not isfinite(value)
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                raise PDFInterpreterError(f"Invalid ExtGState /{key} value")
            return value

        line_width = number("LW", minimum=0)
        miter_limit = number("ML", minimum=1)
        stroke_alpha = number("CA", minimum=0, maximum=1)
        fill_alpha = number("ca", minimum=0, maximum=1)
        if line_width is not None:
            state = replace(state, line_width=line_width)
        if miter_limit is not None:
            state = replace(state, miter_limit=miter_limit)
        if stroke_alpha is not None:
            state = replace(state, stroke_alpha=stroke_alpha)
        if fill_alpha is not None:
            state = replace(state, fill_alpha=fill_alpha)
        for key, field_name in (("LC", "line_cap"), ("LJ", "line_join")):
            if key not in parameters:
                continue
            value = safe_int(resolve1(parameters[key]))
            if value not in {0, 1, 2}:
                raise PDFInterpreterError(f"Invalid ExtGState /{key} value")
            state = replace(state, **{field_name: value})
        if "D" in parameters:
            dash = resolve1(parameters["D"])
            if not isinstance(dash, list) or len(dash) != 2:
                raise PDFInterpreterError("Invalid ExtGState /D value")
            dash_array = resolve1(dash[0])
            phase = safe_float(resolve1(dash[1]))
            if not isinstance(dash_array, list) or phase is None:
                raise PDFInterpreterError("Invalid ExtGState /D value")
            values = tuple(safe_float(resolve1(value)) for value in dash_array)
            if (
                not isfinite(phase)
                or any(
                    value is None or not isfinite(value) or value < 0
                    for value in values
                )
                or (values and not any(value for value in values))
            ):
                raise PDFInterpreterError("Invalid ExtGState /D value")
            state = replace(
                state,
                dash_array=cast(tuple[float, ...], values),
                dash_phase=phase,
            )
        if "SMask" in parameters:
            soft_mask = resolve1(parameters["SMask"])
            try:
                is_none = literal_name(soft_mask) == "None"
            except Exception:
                is_none = False
            if not is_none and not isinstance(soft_mask, (dict, PDFStream)):
                raise PDFInterpreterError("Invalid ExtGState /SMask value")
            state = replace(state, soft_mask_active=not is_none)
        if "SA" in parameters:
            stroke_adjust = resolve1(parameters["SA"])
            if not isinstance(stroke_adjust, bool):
                raise PDFInterpreterError("Invalid ExtGState /SA value")
            state = replace(state, stroke_adjust=stroke_adjust)
        stroke_overprint = None
        fill_overprint = None
        for key in ("OP", "op"):
            if key not in parameters:
                continue
            value = resolve1(parameters[key])
            if not isinstance(value, bool):
                raise PDFInterpreterError(f"Invalid ExtGState /{key} value")
            if key == "OP":
                stroke_overprint = value
            else:
                fill_overprint = value
        if stroke_overprint is not None:
            state = replace(
                state,
                stroke_overprint=stroke_overprint,
                fill_overprint=(
                    stroke_overprint if fill_overprint is None else fill_overprint
                ),
            )
        elif fill_overprint is not None:
            state = replace(state, fill_overprint=fill_overprint)
        if "OPM" in parameters:
            raw_mode = resolve1(parameters["OPM"])
            mode = safe_int(raw_mode)
            if isinstance(raw_mode, bool) or mode not in {0, 1}:
                raise PDFInterpreterError("Invalid ExtGState /OPM value")
            state = replace(state, overprint_mode=mode)
        if "BM" in parameters:
            raw_blend_mode = resolve1(parameters["BM"])
            values = (
                raw_blend_mode if isinstance(raw_blend_mode, list) else [raw_blend_mode]
            )
            if not values:
                raise PDFInterpreterError("Invalid ExtGState /BM value")
            try:
                blend_modes = tuple(literal_name(resolve1(value)) for value in values)
            except Exception as exc:
                raise PDFInterpreterError("Invalid ExtGState /BM value") from exc
            state = replace(
                state,
                blend_mode_complex=any(
                    value not in {"Normal", "Compatible"} for value in blend_modes
                ),
            )
        self._paint_state = state

    def do_Tf(self, fontid: PDFStackT, fontsize: PDFStackT) -> None:  # noqa: N802
        try:
            resource_name = literal_name(fontid)
        except Exception as exc:
            raise PDFInterpreterError("Invalid font resource reference") from exc
        if resource_name not in self.fontmap or safe_float(fontsize) is None:
            raise PDFInterpreterError("Invalid font selection")
        super().do_Tf(fontid, fontsize)

    def do_TJ(self, seq: PDFStackT) -> None:  # noqa: N802
        if self.textstate.font is None:
            raise PDFInterpreterError("Text-show operation has no selected font")
        cast(_ProvenanceDevice, self.device).set_text_paint_state(
            self._operation_paint_state()
        )
        super().do_TJ(seq)

    def do_Tr(self, render: PDFStackT) -> None:  # noqa: N802
        render_mode = safe_int(render)
        if render_mode is None or not 0 <= render_mode <= 7:
            raise PDFInterpreterError("Invalid text rendering mode")
        super().do_Tr(render)
        self._paint_state = replace(self._paint_state, text_render_mode=render_mode)

    def do_Do(self, xobjid_arg: PDFStackT) -> None:  # noqa: N802
        try:
            resource_name = literal_name(xobjid_arg)
            xobject = resolve1(self.xobjmap[resource_name])
        except Exception as exc:
            raise PDFInterpreterError("Invalid XObject resource reference") from exc
        if not isinstance(xobject, PDFStream):
            raise PDFInterpreterError("XObject resource is not a stream")

        subtype = xobject.get("Subtype")
        if subtype is LITERAL_FORM:
            if "BBox" not in xobject:
                raise PDFInterpreterError("Form XObject has no BBox")
            if xobject.objid is not None and xobject.objid in (
                self.parent_stream_ids | self.stream_ids
            ):
                raise PDFInterpreterError("Circular Form XObject invocation")
        elif subtype is LITERAL_IMAGE:
            width = resolve1(xobject.get("Width"))
            height = resolve1(xobject.get("Height"))
            if not _has_valid_image_dimensions(width, height):
                raise PDFInterpreterError("Image XObject has invalid dimensions")
        elif subtype not in {LITERAL_FORM, LITERAL_IMAGE}:
            raise PDFInterpreterError("Unsupported XObject subtype")
        operation_state = self._operation_paint_state()
        cast(_ProvenanceDevice, self.device).set_next_entry_state(operation_state)
        paint_state = self._paint_state
        self._paint_state = operation_state
        try:
            super().do_Do(xobjid_arg)
        finally:
            self._paint_state = paint_state

    def do_EI(self, obj: PDFStackT) -> None:  # noqa: N802
        if not isinstance(obj, PDFStream):
            raise PDFInterpreterError("Malformed inline image")
        width = obj.get("W", obj.get("Width"))
        height = obj.get("H", obj.get("Height"))
        if not _has_valid_image_dimensions(width, height):
            raise PDFInterpreterError("Inline image has invalid dimensions")
        obj.attrs["W"] = width
        obj.attrs["H"] = height
        cast(_ProvenanceDevice, self.device).set_next_entry_state(
            self._operation_paint_state()
        )
        super().do_EI(obj)

    def do_sh(self, name: object) -> None:
        try:
            resource_name = literal_name(name)
            shading = resolve1(dict_value(self.resources.get("Shading"))[resource_name])
        except Exception as exc:
            raise PDFInterpreterError("Invalid Shading resource reference") from exc
        if not isinstance(shading, (dict, PDFStream)):
            raise PDFInterpreterError("Shading resource is not a dictionary or stream")
        super().do_sh(name)


def _validate_input_pages(
    pdf: pikepdf.Pdf,
    page_indices: frozenset[int],
) -> None:
    for page_index, page in enumerate(pdf.pages):
        if page_index not in page_indices:
            continue
        contents = page.obj.get("/Contents")
        if isinstance(contents, pikepdf.Array):
            valid_contents = all(isinstance(item, pikepdf.Stream) for item in contents)
        else:
            valid_contents = contents is None or isinstance(contents, pikepdf.Stream)
        resources = page.obj.get("/Resources")
        if not valid_contents or (
            resources is not None and not isinstance(resources, pikepdf.Dictionary)
        ):
            raise ConversionError(
                f"Digital layout extraction found a malformed page {page_index + 1}"
            )


def _content_object_key(
    value: pikepdf.Object,
) -> tuple[str, int, int] | tuple[str, int]:
    objgen = getattr(value, "objgen", (0, 0))
    if objgen != (0, 0):
        return "indirect", int(objgen[0]), int(objgen[1])
    return "direct", id(value)


def _content_streams(
    owner: pikepdf.Page | pikepdf.Stream,
) -> tuple[pikepdf.Stream, ...]:
    if isinstance(owner, pikepdf.Stream):
        return (owner,)
    contents = resolve_indirect(owner.obj.get("/Contents"))
    if contents is None:
        return ()
    if isinstance(contents, pikepdf.Stream):
        return (contents,)
    if isinstance(contents, pikepdf.Array):
        if len(contents) > _MAX_CONTENT_STREAMS_PER_PAGE:
            raise ConversionError("Digital layout page content-stream budget exceeded")
        streams = tuple(resolve_indirect(item) for item in contents)
        if all(isinstance(item, pikepdf.Stream) for item in streams):
            return cast(tuple[pikepdf.Stream, ...], streams)
    raise ConversionError("Digital layout page contents are malformed")


def _extend_bounded(output: bytearray, data: bytes, limit: int) -> None:
    output.extend(data)
    if len(output) > limit:
        raise ConversionError("Digital layout content container byte budget exceeded")


def _bounded_flate_decode(raw: bytes | memoryview, limit: int) -> bytes:
    if not raw:
        return b""
    decoder = zlib.decompressobj()
    output = bytearray()
    try:
        for offset in range(0, len(raw), _DECODE_CHUNK_SIZE):
            pending = raw[offset : offset + _DECODE_CHUNK_SIZE]
            while pending:
                maximum = min(
                    _DECODE_CHUNK_SIZE,
                    limit - len(output) + 1,
                )
                decoded = decoder.decompress(pending, maximum)
                _extend_bounded(output, decoded, limit)
                pending = decoder.unconsumed_tail
        maximum = limit - len(output) + 1
        _extend_bounded(output, decoder.flush(maximum), limit)
    except zlib.error as exc:
        raise ConversionError(
            "Digital layout Flate content stream is malformed"
        ) from exc
    if not decoder.eof or decoder.unused_data.strip(b"\x00\t\n\f\r "):
        raise ConversionError("Digital layout Flate content stream is malformed")
    return bytes(output)


def _bounded_ascii_hex_decode(raw: bytes | memoryview, limit: int) -> bytes:
    digits = bytearray()
    ended = False
    for value in raw:
        if value in b"\x00\t\n\f\r ":
            continue
        if value == ord(">"):
            ended = True
            continue
        if ended or chr(value) not in string.hexdigits:
            raise ConversionError("Digital layout ASCIIHex content stream is malformed")
        digits.append(value)
        if len(digits) > limit * 2:
            raise ConversionError(
                "Digital layout content container byte budget exceeded"
            )
    if len(digits) % 2:
        digits.append(ord("0"))
    try:
        return bytes.fromhex(digits.decode("ascii"))
    except ValueError as exc:
        raise ConversionError(
            "Digital layout ASCIIHex content stream is malformed"
        ) from exc


def _bounded_ascii85_decode(raw: bytes | memoryview, limit: int) -> bytes:
    values = bytearray()
    output = bytearray()
    index = 0
    if len(raw) >= 2 and bytes(raw[:2]) == b"<~":
        index = 2
    ended = False
    while index < len(raw):
        value = raw[index]
        index += 1
        if value in b"\x00\t\n\f\r ":
            continue
        if value == ord("~"):
            if index >= len(raw) or raw[index] != ord(">"):
                raise ConversionError(
                    "Digital layout ASCII85 content stream is malformed"
                )
            index += 1
            ended = True
            break
        if value == ord("z"):
            if values:
                raise ConversionError(
                    "Digital layout ASCII85 content stream is malformed"
                )
            _extend_bounded(output, b"\x00\x00\x00\x00", limit)
            continue
        if not ord("!") <= value <= ord("u"):
            raise ConversionError("Digital layout ASCII85 content stream is malformed")
        values.append(value - ord("!"))
        if len(values) == 5:
            number = 0
            for item in values:
                number = number * 85 + item
            if number > 0xFFFFFFFF:
                raise ConversionError(
                    "Digital layout ASCII85 content stream is malformed"
                )
            _extend_bounded(output, number.to_bytes(4, "big"), limit)
            values.clear()
    if not ended or any(value not in b"\x00\t\n\f\r " for value in raw[index:]):
        raise ConversionError("Digital layout ASCII85 content stream is malformed")
    if len(values) == 1:
        raise ConversionError("Digital layout ASCII85 content stream is malformed")
    if values:
        original_length = len(values)
        values.extend([84] * (5 - original_length))
        number = 0
        for item in values:
            number = number * 85 + item
        if number > 0xFFFFFFFF:
            raise ConversionError("Digital layout ASCII85 content stream is malformed")
        _extend_bounded(
            output,
            number.to_bytes(4, "big")[: original_length - 1],
            limit,
        )
    return bytes(output)


def _bounded_run_length_decode(raw: bytes | memoryview, limit: int) -> bytes:
    output = bytearray()
    index = 0
    ended = False
    while index < len(raw):
        length = raw[index]
        index += 1
        if length == 128:
            ended = True
            break
        if length <= 127:
            count = length + 1
            if index + count > len(raw):
                raise ConversionError(
                    "Digital layout RunLength content stream is malformed"
                )
            _extend_bounded(output, bytes(raw[index : index + count]), limit)
            index += count
            continue
        if index >= len(raw):
            raise ConversionError(
                "Digital layout RunLength content stream is malformed"
            )
        _extend_bounded(output, bytes([raw[index]]) * (257 - length), limit)
        index += 1
    if not ended or any(value not in b"\x00\t\n\f\r " for value in raw[index:]):
        raise ConversionError("Digital layout RunLength content stream is malformed")
    return bytes(output)


def _content_stream_sizes(
    stream: pikepdf.Stream,
    decoded_limit: int,
) -> tuple[int, int]:
    filters = resolve_indirect(stream.get("/Filter"))
    if filters is None:
        filter_names: tuple[str, ...] = ()
    elif isinstance(filters, pikepdf.Name):
        filter_names = (str(filters),)
    elif isinstance(filters, pikepdf.Array):
        resolved = tuple(resolve_indirect(item) for item in filters)
        if not all(isinstance(item, pikepdf.Name) for item in resolved):
            raise ConversionError("Digital layout content filters are malformed")
        filter_names = tuple(str(item) for item in resolved)
    else:
        raise ConversionError("Digital layout content filters are malformed")
    declared_length = resolve_indirect(stream.get("/Length"))
    missing_empty_length = declared_length is None
    if not missing_empty_length and (
        not isinstance(declared_length, int)
        or isinstance(declared_length, bool)
        or declared_length < 0
    ):
        raise ConversionError("Digital layout content stream length is malformed")
    if (
        isinstance(declared_length, int)
        and declared_length > _MAX_ENCODED_CONTENT_BYTES_PER_CONTAINER
    ):
        raise ConversionError(
            "Digital layout encoded content container byte budget exceeded"
        )
    try:
        raw = memoryview(stream.get_raw_stream_buffer())
    except Exception as exc:
        raise ConversionError("Could not read raw digital layout content") from exc
    if missing_empty_length and raw:
        raise ConversionError("Digital layout content stream length is malformed")
    if len(raw) > _MAX_ENCODED_CONTENT_BYTES_PER_CONTAINER:
        raise ConversionError(
            "Digital layout encoded content container byte budget exceeded"
        )
    if missing_empty_length:
        return 0, 0
    decoded: bytes | memoryview = raw
    decoders = {
        "/FlateDecode": _bounded_flate_decode,
        "/Fl": _bounded_flate_decode,
        "/ASCIIHexDecode": _bounded_ascii_hex_decode,
        "/AHx": _bounded_ascii_hex_decode,
        "/ASCII85Decode": _bounded_ascii85_decode,
        "/A85": _bounded_ascii85_decode,
        "/RunLengthDecode": _bounded_run_length_decode,
        "/RL": _bounded_run_length_decode,
    }
    for filter_name in filter_names:
        decoder = decoders.get(filter_name)
        if decoder is None:
            raise ConversionError(
                "Digital layout content filter cannot be decoded with a strict bound"
            )
        decoded = decoder(decoded, decoded_limit)
    size = len(decoded)
    if size > decoded_limit:
        raise ConversionError("Digital layout content container byte budget exceeded")
    return len(raw), size


def _decoded_content_stream_size(stream: pikepdf.Stream, limit: int) -> int:
    return _content_stream_sizes(stream, limit)[1]


class _DecodedContentBudget:
    def __init__(
        self,
        sizes: dict[
            tuple[str, int, int] | tuple[str, int],
            tuple[int, int],
        ]
        | None = None,
    ) -> None:
        self._sizes = {} if sizes is None else sizes
        self._page_encoded_bytes: Counter[int] = Counter()
        self._document_encoded_bytes = 0
        self._page_bytes: Counter[int] = Counter()
        self._document_bytes = 0
        self._unique_page_streams: set[
            tuple[int, tuple[str, int, int] | tuple[str, int]]
        ] = set()
        self._unique_document_streams: set[tuple[str, int, int] | tuple[str, int]] = (
            set()
        )

    def new_counter(self) -> _DecodedContentBudget:
        return _DecodedContentBudget(self._sizes)

    def _stream_sizes(
        self,
        stream: pikepdf.Stream,
    ) -> tuple[tuple[str, int, int] | tuple[str, int], tuple[int, int]]:
        key = _content_object_key(stream)
        sizes = self._sizes.get(key)
        if sizes is None:
            sizes = _content_stream_sizes(
                stream,
                _MAX_DECODED_CONTENT_BYTES_PER_CONTAINER,
            )
            self._sizes[key] = sizes
        return key, sizes

    def charge(
        self,
        owner: pikepdf.Page | pikepdf.Stream,
        page_index: int,
    ) -> None:
        streams = _content_streams(owner)
        for stream in streams:
            _key, (encoded_size, decoded_size) = self._stream_sizes(stream)
            self._charge(encoded_size, decoded_size, page_index)
        self._charge(0, max(0, len(streams) - 1), page_index)

    def charge_once(
        self,
        owner: pikepdf.Page | pikepdf.Stream,
        page_index: int | None,
    ) -> None:
        streams = _content_streams(owner)
        for stream in streams:
            key = _content_object_key(stream)
            new_page_stream = (
                page_index is not None
                and (page_index, key) not in self._unique_page_streams
            )
            new_document_stream = key not in self._unique_document_streams
            if not new_page_stream and not new_document_stream:
                continue
            _key, (encoded_size, decoded_size) = self._stream_sizes(stream)
            if new_page_stream:
                self._unique_page_streams.add((page_index, key))
                self._charge_page(encoded_size, decoded_size, page_index)
            if new_document_stream:
                self._unique_document_streams.add(key)
                self._charge_document(encoded_size, decoded_size)
        separator_bytes = max(0, len(streams) - 1)
        if page_index is not None:
            self._charge_page(0, separator_bytes, page_index)
        self._charge_document(0, separator_bytes)

    def _charge(
        self,
        encoded_size: int,
        decoded_size: int,
        page_index: int,
    ) -> None:
        self._charge_page(encoded_size, decoded_size, page_index)
        self._charge_document(encoded_size, decoded_size)

    def _charge_page(
        self,
        encoded_size: int,
        decoded_size: int,
        page_index: int,
    ) -> None:
        self._page_encoded_bytes[page_index] += encoded_size
        if self._page_encoded_bytes[page_index] > _MAX_ENCODED_CONTENT_BYTES_PER_PAGE:
            raise ConversionError(
                "Digital layout page encoded-content byte budget exceeded on page "
                f"{page_index + 1}"
            )
        self._page_bytes[page_index] += decoded_size
        if self._page_bytes[page_index] > _MAX_DECODED_CONTENT_BYTES_PER_PAGE:
            raise ConversionError(
                "Digital layout page decoded-content byte budget exceeded on page "
                f"{page_index + 1}"
            )

    def _charge_document(self, encoded_size: int, decoded_size: int) -> None:
        self._document_encoded_bytes += encoded_size
        if self._document_encoded_bytes > _MAX_ENCODED_CONTENT_BYTES_PER_DOCUMENT:
            raise ConversionError(
                "Digital layout document encoded-content byte budget exceeded"
            )
        self._document_bytes += decoded_size
        if self._document_bytes > _MAX_DECODED_CONTENT_BYTES_PER_DOCUMENT:
            raise ConversionError(
                "Digital layout document decoded-content byte budget exceeded"
            )


def _inherited_page_resources(page: pikepdf.Page) -> pikepdf.Dictionary:
    current = page.obj
    visited: set[tuple[str, int, int] | tuple[str, int]] = set()
    while isinstance(current, pikepdf.Dictionary):
        key = _content_object_key(current)
        if key in visited:
            raise ConversionError("Digital layout page tree is recursive")
        visited.add(key)
        if "/Resources" in current:
            resources = resolve_indirect(current.get("/Resources"))
            if not isinstance(resources, pikepdf.Dictionary):
                raise ConversionError("Digital layout page resources are malformed")
            return resources
        current = resolve_indirect(current.get("/Parent"))
    return pikepdf.Dictionary()


def _validate_content_work_budget(
    pdf: pikepdf.Pdf,
    page_indices: frozenset[int],
    decoded_content_budget: _DecodedContentBudget | None = None,
    *,
    strict_provenance: bool = True,
    max_form_nesting_depth: int | None = _MAX_FORM_NESTING_DEPTH,
) -> None:
    document_operator_count = 0
    decoded_content_budget = decoded_content_budget or _DecodedContentBudget()
    for page_index, page in enumerate(pdf.pages):
        if page_index not in page_indices:
            continue
        page_operator_count = 0
        form_invocations: Counter[tuple[str, int, int] | tuple[str, int]] = Counter()
        stack: list[
            tuple[
                pikepdf.Page | pikepdf.Object,
                pikepdf.Dictionary,
                frozenset[tuple[str, int, int] | tuple[str, int]],
            ]
        ] = [(page, _inherited_page_resources(page), frozenset())]
        while stack:
            owner, resources, active_forms = stack.pop()
            decoded_content_budget.charge(owner, page_index)
            pending_clip = False
            has_current_path = False
            for instruction in pikepdf.parse_content_stream(owner):
                if strict_provenance:
                    page_operator_count += 1
                    document_operator_count += 1
                    if page_operator_count > _MAX_DIGITAL_OPERATORS_PER_PAGE:
                        raise ConversionError(
                            "Digital layout page operator budget exceeded on page "
                            f"{page_index + 1}"
                        )
                    if document_operator_count > _MAX_DIGITAL_OPERATORS_PER_DOCUMENT:
                        raise ConversionError(
                            "Digital layout document operator budget exceeded"
                        )
                if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                    if strict_provenance:
                        if pending_clip:
                            raise ConversionError(
                                "Digital layout clipping path has no immediate "
                                "path-painting terminator"
                            )
                        if has_current_path:
                            raise ConversionError(
                                "Digital layout path object is interrupted before "
                                "painting"
                            )
                        inline_image = instruction.iimage.obj
                        width = resolve_indirect(inline_image.get("/Width"))
                        height = resolve_indirect(inline_image.get("/Height"))
                        if not _has_valid_image_dimensions(width, height):
                            raise ConversionError(
                                "Digital layout inline image has invalid dimensions"
                            )
                    continue
                operator_name = str(instruction.operator)
                if strict_provenance:
                    if pending_clip and operator_name not in _PATH_PAINTING_OPERATORS:
                        raise ConversionError(
                            "Digital layout clipping path has no immediate "
                            "path-painting terminator"
                        )
                    if pending_clip:
                        pending_clip = False
                    if operator_name in _PATH_START_OPERATORS:
                        has_current_path = True
                    elif operator_name in _PATH_CONTINUATION_OPERATORS:
                        if not has_current_path:
                            raise ConversionError(
                                "Digital layout path construction operator has no "
                                "current subpath"
                            )
                    elif operator_name in {"W", "W*"}:
                        if not has_current_path:
                            raise ConversionError(
                                "Digital layout clipping operator has no current path"
                            )
                        pending_clip = True
                    elif operator_name in _PATH_PAINTING_OPERATORS:
                        has_current_path = False
                    elif has_current_path:
                        raise ConversionError(
                            "Digital layout path object is interrupted before painting"
                        )
                expected_operands = _DIRECT_TEXT_OPERAND_COUNTS.get(operator_name)
                if (
                    strict_provenance
                    and expected_operands is not None
                    and (
                        len(instruction.operands) != expected_operands
                        or not _valid_direct_text_operands(
                            operator_name,
                            instruction.operands,
                        )
                    )
                ):
                    raise ConversionError(
                        "Digital layout direct text provenance is malformed"
                    )
                if operator_name != "Do":
                    continue
                if len(instruction.operands) != 1:
                    if not strict_provenance:
                        continue
                    raise ConversionError(
                        "Digital layout direct XObject provenance is malformed"
                    )
                xobjects = resolve_indirect(resources.get("/XObject"))
                xobject = (
                    resolve_indirect(xobjects.get(instruction.operands[0]))
                    if isinstance(xobjects, pikepdf.Dictionary)
                    else None
                )
                if not isinstance(xobject, pikepdf.Stream):
                    if not strict_provenance:
                        continue
                    raise ConversionError(
                        "Digital layout XObject resource cannot be resolved"
                    )
                subtype = str(resolve_indirect(xobject.get("/Subtype")))
                if subtype == "/Image":
                    continue
                if subtype != "/Form":
                    if not strict_provenance:
                        continue
                    raise ConversionError(
                        "Digital layout XObject subtype is unsupported"
                    )
                form = xobject
                form_key = _content_object_key(form)
                if form_key in active_forms:
                    raise ConversionError("Digital layout Form XObject is recursive")
                if (
                    max_form_nesting_depth is not None
                    and len(active_forms) >= max_form_nesting_depth
                ):
                    raise ConversionError(
                        "Digital layout Form XObject nesting depth budget exceeded "
                        f"on page {page_index + 1}"
                    )
                form_invocations[form_key] += 1
                if (
                    form_invocations[form_key]
                    > _MAX_FORM_INVOCATIONS_PER_RESOURCE_PER_PAGE
                ):
                    raise ConversionError(
                        "Digital layout Form XObject invocation budget exceeded "
                        f"on page {page_index + 1}"
                    )
                form_resources = resolve_indirect(form.get("/Resources"))
                if form_resources is None or (
                    isinstance(form_resources, pikepdf.Dictionary)
                    and not form_resources
                ):
                    form_resources = resources
                elif not isinstance(form_resources, pikepdf.Dictionary):
                    raise ConversionError(
                        "Digital layout Form XObject resources are malformed"
                    )
                stack.append(
                    (
                        form,
                        form_resources,
                        active_forms | frozenset({form_key}),
                    )
                )
            if strict_provenance and pending_clip:
                raise ConversionError(
                    "Digital layout clipping path has no path-painting terminator"
                )
            if strict_provenance and has_current_path:
                raise ConversionError(
                    "Digital layout path object has no path-painting terminator"
                )


def _apply_intrinsic_image_visibility(
    pdf: pikepdf.Pdf,
    layouts: tuple[DigitalPageLayout, ...],
) -> tuple[DigitalPageLayout, ...]:
    budget = _ImageVisibilityBudget()

    def effective_form_resources(
        form: pikepdf.Stream,
        inherited: pikepdf.Dictionary,
    ) -> pikepdf.Dictionary:
        raw_resources = form.get("/Resources")
        resources = resolve_indirect(raw_resources)
        if raw_resources is None or (
            isinstance(resources, pikepdf.Dictionary) and not resources
        ):
            return inherited
        if not isinstance(resources, pikepdf.Dictionary):
            raise ConversionError("Digital layout Form resources are malformed")
        return resources

    def apply_visibility(
        span: DirectXObjectSpan,
        visibility: _ImageVisibility,
    ) -> DirectXObjectSpan:
        if span.invisible:
            return span
        if visibility == "invisible":
            return replace(
                span,
                invisible=True,
                final_paint_uncertain=False,
                intrinsic_visibility_uncertain=False,
                non_intrinsic_visibility_uncertain=False,
            )
        if visibility == "uncertain":
            return replace(
                span,
                final_paint_uncertain=True,
                intrinsic_visibility_uncertain=True,
            )
        return span

    def annotate_container(
        owner: pikepdf.Page | pikepdf.Stream,
        spans: tuple[PaintingSpan, ...],
        resources: pikepdf.Dictionary,
        active_forms: frozenset[tuple[str, int, int] | tuple[str, int]],
    ) -> tuple[PaintingSpan, ...]:
        direct_xobjects = {
            span.direct_xobject_index: span
            for span in spans
            if isinstance(span, DirectXObjectSpan)
        }
        replacements: dict[int, DirectXObjectSpan] = {}
        xobject_index = 0
        try:
            instructions = pikepdf.parse_content_stream(owner)
        except Exception as exc:
            raise ConversionError(
                "Digital layout image provenance cannot be parsed"
            ) from exc
        for instruction in instructions:
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                span = direct_xobjects.get(xobject_index)
                if span is None or span.kind != "inline_image":
                    raise ConversionError(
                        "Digital layout inline-image provenance no longer matches"
                    )
                container = owner.obj if isinstance(owner, pikepdf.Page) else owner
                cache_key = (
                    "inline",
                    *_content_object_key(container),
                    xobject_index,
                )
                replacements[xobject_index] = apply_visibility(
                    span,
                    budget.classify_inline(instruction.iimage, cache_key),
                )
                xobject_index += 1
                continue
            if str(instruction.operator) != "Do":
                continue
            if len(instruction.operands) != 1:
                raise ConversionError(
                    "Digital layout direct XObject provenance no longer matches"
                )
            span = direct_xobjects.get(xobject_index)
            operand = resolve_indirect(instruction.operands[0])
            xobjects = resolve_indirect(resources.get("/XObject"))
            xobject = (
                resolve_indirect(xobjects.get(operand))
                if isinstance(operand, pikepdf.Name)
                and isinstance(xobjects, pikepdf.Dictionary)
                else None
            )
            if span is None or not isinstance(xobject, pikepdf.Stream):
                raise ConversionError(
                    "Digital layout direct XObject provenance no longer matches"
                )
            expected_name = str(operand).removeprefix("/")
            if span.resource_name != expected_name:
                raise ConversionError(
                    "Digital layout XObject resource provenance no longer matches"
                )
            subtype = resolve_indirect(xobject.get("/Subtype"))
            if subtype == pikepdf.Name.Image and span.kind == "image":
                replacement = apply_visibility(
                    span,
                    budget.classify_stream(xobject),
                )
            elif subtype == pikepdf.Name.Form and span.kind == "form":
                form_key = _content_object_key(xobject)
                if form_key in active_forms:
                    raise ConversionError("Digital layout Form XObject is recursive")
                replacement = replace(
                    span,
                    children=annotate_container(
                        xobject,
                        span.children,
                        effective_form_resources(xobject, resources),
                        active_forms | frozenset({form_key}),
                    ),
                )
            else:
                raise ConversionError(
                    "Digital layout XObject subtype provenance no longer matches"
                )
            replacements[xobject_index] = replacement
            xobject_index += 1
        if set(direct_xobjects) != set(range(xobject_index)):
            raise ConversionError(
                "Digital layout direct XObject provenance no longer matches"
            )
        return tuple(
            replacements[span.direct_xobject_index]
            if isinstance(span, DirectXObjectSpan)
            else span
            for span in spans
        )

    annotated = []
    for layout in layouts:
        page = pdf.pages[layout.page_index]
        annotated.append(
            replace(
                layout,
                spans=annotate_container(
                    page,
                    layout.spans,
                    _inherited_page_resources(page),
                    frozenset(),
                ),
            )
        )
    return tuple(annotated)


def extract_digital_layout(
    pdf: pikepdf.Pdf,
    *,
    page_indices: frozenset[int] | None = None,
) -> tuple[DigitalPageLayout, ...]:
    """Extract direct painting provenance without changing ``pdf``.

    The document is serialized to a bounded-memory, automatically deleted
    temporary stream for pdfminer. Any serialization, permission, parsing, or
    extraction failure is reported as a :class:`ConversionError` instead of
    returning partial layout information.
    """

    try:
        if not pdf.allow.extract:
            raise ConversionError(
                "Digital layout extraction is not permitted by the PDF"
            )
        page_count = len(pdf.pages)
        selected_pages = (
            frozenset(range(page_count))
            if page_indices is None
            else frozenset(page_indices)
        )
        if any(
            isinstance(page_index, bool)
            or not isinstance(page_index, int)
            or page_index < 0
            or page_index >= page_count
            for page_index in selected_pages
        ):
            raise ConversionError("Digital layout extraction page selection is invalid")
        _validate_input_pages(pdf, selected_pages)
        _validate_content_work_budget(pdf, selected_pages)
        with SpooledTemporaryFile(
            max_size=_SERIALIZED_PDF_MEMORY_LIMIT,
            mode="w+b",
        ) as serialized:
            pdf.save(serialized)
            serialized.seek(0)
            resource_manager = PDFResourceManager(caching=False)
            device = _ProvenanceDevice(resource_manager)
            interpreter = _ProvenanceInterpreter(resource_manager, device)
            for page_index, page in enumerate(
                PDFPage.get_pages(
                    serialized,
                    caching=False,
                    check_extractable=True,
                )
            ):
                if page_index not in selected_pages:
                    continue
                device.page_index = page_index
                interpreter.process_page(page)
            if {page.page_index for page in device.pages} != selected_pages:
                raise ConversionError(
                    "Digital layout extraction returned an incomplete page set"
                )
            serialized.seek(0)
            with pikepdf.Pdf.open(serialized) as canonical_pdf:
                return _apply_intrinsic_image_visibility(
                    canonical_pdf,
                    tuple(device.pages),
                )
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError("Could not extract digital PDF layout") from exc
