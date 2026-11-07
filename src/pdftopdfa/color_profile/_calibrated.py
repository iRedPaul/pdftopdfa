# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Compatibility entry point for the former calibrated-space conversion."""

import logging
import math
from decimal import Decimal

from pikepdf import (
    Array,
    ContentStreamInlineImage,
    Dictionary,
    Name,
    Page,
    Pdf,
    Stream,
    parse_content_stream,
)

from ..exceptions import ConversionError
from ..utils import iter_type3_fonts, log_suppressed_error
from ._types import ColorSpaceType

logger = logging.getLogger(__name__)


def _calibration_numbers(value, count: int) -> bool:
    """Check a calibration entry's dimensions and finite PDF-number values."""
    return (
        isinstance(value, Array)
        and len(value) == count
        and all(
            not isinstance(number, bool)
            and isinstance(number, (int, float, Decimal))
            and math.isfinite(number)
            for number in value
        )
    )


def _validate_calibrated_space(cs) -> None:
    """Reject malformed calibration rather than substituting unrelated colorimetry."""
    if len(cs) != 2 or not isinstance(cs[1], Dictionary):
        raise ConversionError(
            f"Malformed {cs[0]} color space: expected calibration dictionary"
        )
    params = cs[1]
    white = params.get("/WhitePoint")
    if not _calibration_numbers(white, 3) or not (
        white[0] > 0 and white[1] == 1 and white[2] > 0
    ):
        raise ConversionError(f"Malformed {cs[0]} color space: invalid /WhitePoint")
    black = params.get("/BlackPoint")
    if "/BlackPoint" in params and (
        not _calibration_numbers(black, 3) or any(number < 0 for number in black)
    ):
        raise ConversionError(f"Malformed {cs[0]} color space: invalid /BlackPoint")
    if "/Gamma" in params:
        gamma = params.Gamma
        if cs[0] == Name.CalGray:
            valid = (
                not isinstance(gamma, bool)
                and isinstance(gamma, (int, float, Decimal))
                and math.isfinite(gamma)
                and gamma > 0
            )
        else:
            valid = _calibration_numbers(gamma, 3) and all(
                number > 0 for number in gamma
            )
        if not valid:
            raise ConversionError(f"Malformed {cs[0]} color space: invalid /Gamma")
    if (
        cs[0] == Name.CalRGB
        and "/Matrix" in params
        and not _calibration_numbers(params.Matrix, 9)
    ):
        raise ConversionError("Malformed /CalRGB color space: invalid /Matrix")


def _acroform_resources(pdf: Pdf) -> list:
    """Return the default resources of an interactive form and of its fields."""
    acroform = pdf.Root.get("/AcroForm")
    if not isinstance(acroform, Dictionary):
        return []
    found = [acroform.get("/DR")]
    fields = acroform.get("/Fields")
    pending = list(fields) if isinstance(fields, Array) else []
    seen = set()
    while pending:
        field = pending.pop()
        if not isinstance(field, Dictionary):
            continue
        if field.objgen != (0, 0):
            if field.objgen in seen:
                continue
            seen.add(field.objgen)
        found.append(field.get("/DR"))
        kids = field.get("/Kids")
        if isinstance(kids, Array):
            pending.extend(kids)
    return found


def _inline_image_spaces(owner) -> list:
    """Return color spaces that inline images in a content stream declare directly.

    Named spaces are found through resources. Only a stream that can contain both
    an inline image and a calibrated space, possibly in a hex-escaped name, is parsed.
    A page's /Contents array counts as one stream for this purpose.
    """
    is_page = owner.get("/Type") == Name.Page
    contents = owner.get("/Contents") if is_page else owner
    streams = list(contents) if isinstance(contents, Array) else [contents]
    try:
        # A page may divide an inline image between the streams of its /Contents.
        data = [stream.read_bytes() for stream in streams if isinstance(stream, Stream)]
        if not any(b"BI" in part for part in data) or not any(
            b"Cal" in part or b"#" in part for part in data
        ):
            return []
        return [
            space
            for instruction in parse_content_stream(Page(owner) if is_page else owner)
            if isinstance(instruction, ContentStreamInlineImage)
            and isinstance(
                space := instruction.iimage.obj.get("/ColorSpace"),
                Array,
            )
        ]
    except Exception as exc:
        # Unreadable content is reported by the stages that have to interpret it.
        log_suppressed_error(logger, exc, "Error reading inline images: %s", exc)
        return []


def validate_calibrated_colorspaces(pdf: Pdf) -> None:
    """Validate declared color spaces without interpreting unrelated private data.

    Leave the PDF unchanged.

    Raises:
        ConversionError: If a CalGray or CalRGB array encountered in a declared
            color context has malformed calibration parameters.
    """
    from ..fonts.traversal import get_page_resources

    pending = []
    for page in pdf.pages:
        pending.append(("owner", page.obj))
        pending.append(("resources", get_page_resources(page)))
        pending.extend(
            ("appearances", annotation.get("/AP"))
            for annotation in (page.get("/Annots") or [])
            if isinstance(annotation, Dictionary)
        )
    pending.extend(("resources", found) for found in _acroform_resources(pdf))
    visited = set()
    fonts_seen = set()
    while pending:
        kind, value = pending.pop()
        if not isinstance(value, (Array, Dictionary, Stream)):
            continue
        if value.objgen != (0, 0):
            key = (kind, value.objgen)
            if key in visited:
                continue
            visited.add(key)
        if kind == "space" and isinstance(value, Array):
            if (
                value
                and isinstance(value[0], Name)
                and value[0] in (Name.CalGray, Name.CalRGB)
            ):
                _validate_calibrated_space(value)
            elif value:
                family = value[0]
                # Inline images may abbreviate Indexed as /I.
                if family in (Name.Indexed, Name.I, Name.Pattern) and len(value) > 1:
                    pending.append(("space", value[1]))
                elif family in (Name.Separation, Name.DeviceN) and len(value) > 2:
                    pending.append(("space", value[2]))
                    attrs = value[4] if len(value) > 4 else None
                    if family == Name.DeviceN and isinstance(attrs, Dictionary):
                        process = attrs.get("/Process")
                        if isinstance(process, Dictionary):
                            pending.append(("space", process.get("/ColorSpace")))
                        colorants = attrs.get("/Colorants")
                        if isinstance(colorants, Dictionary):
                            pending.extend(("space", cs) for cs in colorants.values())
                elif (
                    family == Name.ICCBased
                    and len(value) > 1
                    and isinstance(value[1], Stream)
                ):
                    pending.append(("space", value[1].get("/Alternate")))
        elif kind == "resources" and isinstance(value, Dictionary):
            spaces = value.get("/ColorSpace")
            if isinstance(spaces, Dictionary):
                pending.extend(("space", cs) for cs in spaces.values())
            for category, target in (
                ("/XObject", "owner"),
                ("/Pattern", "pattern"),
                ("/Shading", "shading"),
            ):
                table = value.get(category)
                if isinstance(table, Dictionary):
                    pending.extend((target, obj) for obj in table.values())
            states = value.get("/ExtGState")
            if isinstance(states, Dictionary):
                for state in states.values():
                    mask = (
                        state.get("/SMask") if isinstance(state, Dictionary) else None
                    )
                    if isinstance(mask, Dictionary):
                        pending.append(("owner", mask.get("/G")))
            for _, font in iter_type3_fonts(value, fonts_seen):
                pending.append(("resources", font.get("/Resources")))
                glyphs = font.get("/CharProcs")
                if isinstance(glyphs, Dictionary):
                    for glyph in glyphs.values():
                        pending.extend(
                            ("space", space) for space in _inline_image_spaces(glyph)
                        )
        elif kind == "appearances" and isinstance(value, Dictionary):
            pending.extend(("appearance", value.get(key)) for key in ("/N", "/R", "/D"))
        elif kind == "appearance":
            if isinstance(value, Stream):
                pending.append(("owner", value))
            elif isinstance(value, Dictionary):
                pending.extend(
                    ("owner", entry)
                    for entry in value.values()
                    if isinstance(entry, Stream)
                )
        elif kind == "pattern" and isinstance(value, (Dictionary, Stream)):
            if value.get("/PatternType") == 1:
                pending.append(("tiling", value))
            elif value.get("/PatternType") == 2:
                pending.append(("shading", value.get("/Shading")))
                state = value.get("/ExtGState")
                mask = state.get("/SMask") if isinstance(state, Dictionary) else None
                if isinstance(mask, Dictionary):
                    pending.append(("owner", mask.get("/G")))
        elif kind == "shading" and isinstance(value, (Dictionary, Stream)):
            pending.append(("space", value.get("/ColorSpace")))
        elif kind in ("owner", "tiling") and isinstance(value, (Dictionary, Stream)):
            form_or_page = (
                value.get("/Subtype") == Name.Form or value.get("/Type") == Name.Page
            )
            if form_or_page or kind == "tiling":
                pending.append(("resources", value.get("/Resources")))
                pending.extend(
                    ("space", space) for space in _inline_image_spaces(value)
                )
            group = value.get("/Group") if form_or_page else None
            if isinstance(group, Dictionary) and group.get("/S") == Name.Transparency:
                pending.append(("space", group.get("/CS")))
            if value.get("/Subtype") == Name.Image:
                pending.append(("space", value.get("/ColorSpace")))
                pending.append(("owner", value.get("/SMask")))
                pending.append(("owner", value.get("/Mask")))
                alternates = value.get("/Alternates")
                if isinstance(alternates, Array):
                    pending.extend(
                        ("owner", item.get("/Image"))
                        for item in alternates
                        if isinstance(item, Dictionary)
                    )


def _convert_calibrated_colorspaces(
    pdf: Pdf,
    icc_stream_cache: dict[ColorSpaceType, Stream],
) -> int:
    """Preserve calibrated spaces requested through the deprecated option.

    CalGray and CalRGB are already permitted by PDF/A-2 and PDF/A-3. Replacing
    them with a generic grayscale or sRGB ICC profile discards their gamma,
    white point, black point, and matrix without converting the color values.
    Until equivalent ICC profiles can be generated from those parameters,
    retain the original color spaces and report that no conversion occurred.

    The signature and zero replacement count are retained for compatibility.
    Neither the PDF nor the supplied ICC stream cache is modified.

    Raises:
        ConversionError: If declared CalGray/CalRGB calibration is malformed.
    """
    logger.warning(
        "convert_calibrated=True is deprecated: CalGray/CalRGB color spaces "
        "are preserved because generic ICC replacement changes their calibration"
    )
    validate_calibrated_colorspaces(pdf)
    return 0
