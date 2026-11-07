# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Structural requirements for permitted ISO 32000-1 trap network annotations."""

import hashlib
import math
from decimal import Decimal

from pikepdf import Array, Dictionary, Name, Page, Pdf, PdfError, Stream, String

from ..exceptions import ConversionError
from .navigation import _pdf_date as _date


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


def has_valid_rect(annotation) -> bool:
    """Check the required base annotation rectangle that positions an appearance."""
    return _numbers(annotation.get("/Rect"), 4)


def valid_trapnet(annotation, owner, annotations, index: int, count: int) -> bool:
    """Check trap cardinality, geometry, state, flags, appearances and tracking."""
    if owner.get("/Type") != Name.Page or count != 1 or index != len(annotations) - 1:
        return False
    if annotation.get("/Type") not in (None, Name.Annot):
        return False
    if not has_valid_rect(annotation):
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
    # An extra entry refers to an object that has since left the page description.
    return _version_objects(owner) == {_key(item) for item in version}


# Besides its appearance and state, these entries decide how an annotation prints.
# PDF/A sanitization may change them, for example by setting the required Print flag.
_ANNOTATION_RENDERING_KEYS = ("/F", "/Rect", "/CA", "/ca", "/BM", "/OC")


def _printed_appearance(annotation: Dictionary):
    """Reduce /AP to the normal appearance that printing and trapping depend on.

    Rollover and down appearances only apply to interactive use, and PDF/A
    sanitization removes them. Their removal cannot make a trap network stale.
    """
    appearance = annotation.get("/AP")
    return appearance.get("/N") if isinstance(appearance, Dictionary) else appearance


def _page_snapshot(page: Page) -> bytes:
    """Hash page content, resource graphs, geometry, and other annotation states."""
    from ..fonts.traversal import get_page_resources

    inherited = Dictionary()
    owner = page.obj
    parents = set()
    while isinstance(owner, Dictionary):
        if owner.objgen != (0, 0):
            if owner.objgen in parents:
                break
            parents.add(owner.objgen)
        for key in ("/MediaBox", "/CropBox", "/Rotate"):
            if key not in inherited and owner.get(key) is not None:
                inherited[key] = owner[key]
        owner = owner.get("/Parent")
    cropbox = inherited.get("/CropBox", inherited.get("/MediaBox"))
    if cropbox is not None:
        inherited["/CropBox"] = cropbox
    if "/Rotate" not in inherited:
        inherited.Rotate = 0
    states = Array(
        Array([item.get("/AS"), _printed_appearance(item)])
        for item in _other_annotations(page)
    )
    description = Array(
        [
            page.get("/Contents"),
            get_page_resources(page),
            inherited,
            states,
            page.get("/Group"),
            page.get("/UserUnit", 1),
            *[page.get(key, cropbox) for key in ("/BleedBox", "/TrimBox", "/ArtBox")],
            page.get("/SeparationInfo"),
        ]
    )
    return _graph_digest(description)


def _annotation_rendering_snapshot(page: Page) -> bytes:
    """Hash the entries that decide how the other annotations of a page print."""
    return _graph_digest(
        Array(
            Array([item.get(key) for key in _ANNOTATION_RENDERING_KEYS])
            for item in _other_annotations(page)
        )
    )


def _other_annotations(page: Page) -> list:
    """Return the annotations of a page other than its trap network."""
    return [
        item
        for item in page.get("/Annots", [])
        if isinstance(item, Dictionary) and item.get("/Subtype") != Name.TrapNet
    ]


def _graph_digest(description: Array) -> bytes:
    """Hash a PDF object graph independently of object numbers and stream encoding."""
    digest = hashlib.sha256()

    def record(data: bytes) -> None:
        """Frame each token so distinct PDF graphs cannot concatenate identically."""
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    pending = [description]
    visited = {}
    while pending:
        value = pending.pop()
        if isinstance(value, (Dictionary, Stream, Array)):
            if value.objgen != (0, 0):
                if value.objgen in visited:
                    record(f"reference:{visited[value.objgen]}".encode())
                    continue
                # Local traversal IDs survive object renumbering during OCR saves.
                visited[value.objgen] = len(visited)
                record(f"object:{visited[value.objgen]}".encode())
            if isinstance(value, Array):
                record(f"array:{len(value)}".encode())
                pending.extend(reversed(value))
            else:
                # Table 366 excludes obsolete procedure-set lists from tracking.
                items = sorted(
                    (key, item) for key, item in value.items() if key != "/ProcSet"
                )
                if isinstance(value, Stream):
                    ignored = {"/Length"}
                    try:
                        data = value.read_bytes()
                    except PdfError:
                        data = value.read_raw_bytes()
                    else:
                        ignored.update(("/Filter", "/DecodeParms"))
                    items = [(key, item) for key, item in items if key not in ignored]
                    record(b"stream")
                    record(data)
                record(f"dictionary:{len(items)}".encode())
                for key, item in reversed(items):
                    pending.extend((item, key.encode()))
        elif isinstance(value, bytes):
            record(b"key:" + value)
        elif isinstance(value, String):
            record(b"string:" + bytes(value))
        elif isinstance(value, Name):
            record(b"name:" + value.unparse())
        elif isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
            number = format(Decimal(str(value)), "f")
            if "." in number:
                number = number.rstrip("0").rstrip(".")
            record(b"number:" + ("0" if Decimal(number) == 0 else number).encode())
        else:
            record(f"{type(value).__name__}:{value!r}".encode())
    return digest.digest()


def capture_trapnet_snapshots(pdf: Pdf) -> dict[int, tuple[bytes, bytes]]:
    """Capture immutable page descriptions before any converter or OCR edits.

    Returns:
        Hashes of the page description and of the printing properties of its other
        annotations, keyed by zero-based page index, including only pages that
        currently contain TrapNet annotations. Pass these snapshots to
        validate_preserved_trapnets after conversion.
    """
    return {
        index: (_page_snapshot(page), _annotation_rendering_snapshot(page))
        for index, page in enumerate(pdf.pages)
        if trapnet_count(page.get("/Annots", []))
    }


def validate_preserved_trapnets(
    pdf: Pdf, snapshots: dict[int, tuple[bytes, bytes]] | None = None
) -> int:
    """Reject or remove retained trap networks invalidated by conversion edits.

    A required change to another annotation, such as setting its Print flag, only
    makes the trapping of the page stale. Removing the trap network therefore keeps
    the conversion, while changed page content still rejects it.

    Args:
        pdf: Converted PDF to check, whose stale trap networks are removed.
        snapshots: Page hashes from capture_trapnet_snapshots before conversion,
            using the same page indices. None is accepted only when no TrapNet
            annotations remain.

    Returns:
        Number of trap networks removed because another annotation changed.

    Raises:
        ConversionError: If a retained TrapNet is structurally invalid, its page
            snapshot is missing, or its tracked page description has changed.
    """
    removed = 0
    for page_index, page in enumerate(pdf.pages):
        annotations = page.get("/Annots", [])
        if not isinstance(annotations, Array):
            continue
        count = trapnet_count(annotations)
        for index, annotation in enumerate(annotations):
            if (
                not isinstance(annotation, Dictionary)
                or annotation.get("/Subtype") != Name.TrapNet
            ):
                continue
            snapshot = snapshots.get(page_index) if snapshots is not None else None
            if (
                not valid_trapnet(annotation, page.obj, annotations, index, count)
                or snapshot is None
                or snapshot[0] != _page_snapshot(page)
            ):
                raise ConversionError(
                    "Conversion invalidated a preserved TrapNet annotation"
                )
            if snapshot[1] != _annotation_rendering_snapshot(page):
                # A valid trap network is the only one on its page, and the last entry.
                del annotations[index]
                if len(annotations) == 0:
                    del page.obj["/Annots"]
                removed += 1
                break
    return removed
