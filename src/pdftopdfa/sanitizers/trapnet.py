# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Structural requirements for permitted ISO 32000-1 trap network annotations."""

import math
import re
from decimal import Decimal

from pikepdf import Array, Dictionary, Name, Page, Pdf, Stream, String

from ..exceptions import ConversionError


def _numbers(value, count: int) -> bool:
    """Check finite PDF numbers in an appearance rectangle or transformation."""
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


def _date(value):
    """Require complete PDF date syntax as well as valid calendar components."""
    from ..metadata import _parse_pdf_date

    if not isinstance(value, String) or not re.fullmatch(
        r"D:\d{4}(?:\d{2}){0,5}(?:Z|[+-]\d{2}(?:'\d{2}'?)?)?", str(value)
    ):
        return None
    try:
        return _parse_pdf_date(str(value))
    except (ValueError, OverflowError):
        return None


def _key(value):
    """Compare Version references by indirect identity or direct PDF value."""
    if value is None:
        return None
    if not isinstance(value, (Dictionary, Stream, Array, Name, String)):
        return (type(value).__name__, value)
    return value.objgen if value.objgen != (0, 0) else value.unparse()


def _version_objects(owner):
    """Collect page content and resource references required by a Version snapshot."""
    from ..fonts.traversal import get_page_resources

    required = set()
    content = owner.get("/Contents")
    if content is not None:
        required.update(
            _key(item)
            for item in (content if isinstance(content, Array) else [content])
        )
    pending = [get_page_resources(Page(owner))]
    visited = set()
    while pending:
        resources = pending.pop()
        if not isinstance(resources, Dictionary):
            continue
        if resources.objgen != (0, 0):
            if resources.objgen in visited:
                continue
            visited.add(resources.objgen)
        for category, table in resources.items():
            if category == "/ProcSet" or not isinstance(table, Dictionary):
                continue
            for resource in table.values():
                if resource is None:
                    continue
                required.add(_key(resource))
                if category == "/XObject" and isinstance(resource, Stream):
                    if resource.get("/Subtype") == Name.Form:
                        if resource.objgen in visited:
                            continue
                        visited.add(resource.objgen)
                        pending.append(resource.get("/Resources"))
                    opi = resource.get("/OPI")
                    if isinstance(opi, Dictionary):
                        required.add(_key(opi))
    return required


def trapnet_count(annotations) -> int:
    """Count trap networks in an annotation array without interpreting other entries."""
    if not isinstance(annotations, (Array, list, tuple)):
        return 0
    return sum(
        isinstance(item, Dictionary) and item.get("/Subtype") == Name.TrapNet
        for item in annotations
    )


def usable_form_appearance(stream) -> bool:
    """Require a well-formed Form before preserving or flattening an appearance."""
    if (
        not isinstance(stream, Stream)
        or stream.get("/Subtype") != Name.Form
        or stream.get("/Type") not in (None, Name.XObject)
        or not _numbers(stream.get("/BBox"), 4)
    ):
        return False
    form_type = stream.get("/FormType")
    if form_type is not None and (
        isinstance(form_type, bool) or not isinstance(form_type, int) or form_type != 1
    ):
        return False
    return (stream.get("/Matrix") is None or _numbers(stream.Matrix, 6)) and (
        stream.get("/Resources") is None or isinstance(stream.Resources, Dictionary)
    )


def valid_trapnet(annotation, owner, annotations, index: int, count: int) -> bool:
    """Check trap cardinality, state, flags, appearances and tracking fields."""
    if owner.get("/Type") != Name.Page or count != 1 or index != len(annotations) - 1:
        return False
    # ISO 32000-1, 14.11.6.2: only Print (4) and ReadOnly (64) are set.
    flags = annotation.get("/F")
    if isinstance(flags, bool) or not isinstance(flags, int) or flags != 68:
        return False
    state = annotation.get("/AS")
    appearance = annotation.get("/AP")
    if not isinstance(state, Name) or not isinstance(appearance, Dictionary):
        return False
    normal = appearance.get("/N")
    if isinstance(normal, Stream):
        streams = [normal]
    elif isinstance(normal, Dictionary) and normal and str(state) in normal:
        streams = list(normal.values())
    else:
        return False
    for stream in streams:
        if not usable_form_appearance(stream) or stream.get("/PCM") not in (
            Name.DeviceGray,
            Name.DeviceRGB,
            Name.DeviceCMYK,
            Name.DeviceCMY,
            Name.DeviceRGBK,
            Name.DeviceN,
        ):
            return False
        colors = stream.get("/SeparationColorNames")
        if colors is not None and (
            not isinstance(colors, Array)
            or not all(isinstance(color, Name) for color in colors)
        ):
            return False
        regions = stream.get("/TrapRegions")
        if regions is not None and (
            not isinstance(regions, Array)
            or not all(
                isinstance(region, Dictionary) and region.objgen != (0, 0)
                for region in regions
            )
        ):
            return False
        if stream.get("/TrapStyles") is not None and not isinstance(
            stream.TrapStyles, String
        ):
            return False
    fonts = annotation.get("/FontFauxing")
    if fonts is not None and (
        not isinstance(fonts, Array)
        or not all(
            isinstance(font, Dictionary) and font.get("/Type") == Name.Font
            for font in fonts
        )
    ):
        return False
    modified = annotation.get("/LastModified")
    version, states = annotation.get("/Version"), annotation.get("/AnnotStates")
    if modified is not None:
        if version is not None or states is not None:
            return False
        trap_date = _date(modified)
        if trap_date is None:
            return False
        page_date = owner.get("/LastModified")
        return (
            page_date is None
            or _date(page_date) is not None
            and _date(page_date) <= trap_date
        )
    if not isinstance(version, Array) or not isinstance(states, Array):
        return False
    if len(states) != len(annotations) - 1:
        return False
    for other, saved in zip(annotations[:-1], states, strict=True):
        if not isinstance(other, Dictionary) or (
            saved is not None and not isinstance(saved, Name)
        ):
            return False
        if other.get("/AS") != saved:
            return False
    return _version_objects(owner) <= {_key(item) for item in version}


def validate_preserved_trapnets(pdf: Pdf) -> None:
    """Reject retained trap networks invalidated by later conversion edits."""
    for page in pdf.pages:
        annotations = page.get("/Annots", [])
        if not isinstance(annotations, Array):
            continue
        count = trapnet_count(annotations)
        for index, annotation in enumerate(annotations):
            if (
                isinstance(annotation, Dictionary)
                and annotation.get("/Subtype") == Name.TrapNet
            ):
                if not valid_trapnet(annotation, page.obj, annotations, index, count):
                    raise ConversionError(
                        "Conversion invalidated a preserved TrapNet annotation"
                    )
