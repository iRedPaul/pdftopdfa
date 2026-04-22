# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Annotation handling for PDF/A compliance."""

import logging
import threading
from collections.abc import Iterator

from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..utils import resolve_indirect as _resolve_indirect
from .base import (
    ANNOT_FLAG_HIDDEN,
    ANNOT_FLAG_INVISIBLE,
    ANNOT_FLAG_NOROTATE,
    ANNOT_FLAG_NOVIEW,
    ANNOT_FLAG_NOZOOM,
    ANNOT_FLAG_PRINT,
    ANNOT_FLAG_TOGGLENOVIEW,
    FORBIDDEN_ANNOTATION_SUBTYPES,
)
from .widget_appearance import create_widget_appearance

logger = logging.getLogger(__name__)

# Annotation subtypes defined in ISO 32000-1 / ISO 32000-2.
# PDF/A rule 6.3.1 forbids annotation types not defined there.
DEFINED_ANNOTATION_SUBTYPES = frozenset(
    {
        "/Text",
        "/Link",
        "/FreeText",
        "/Line",
        "/Square",
        "/Circle",
        "/Polygon",
        "/PolyLine",
        "/Highlight",
        "/Underline",
        "/Squiggly",
        "/StrikeOut",
        "/Stamp",
        "/Caret",
        "/Ink",
        "/Popup",
        "/FileAttachment",
        "/Sound",
        "/Movie",
        "/Widget",
        "/Screen",
        "/PrinterMark",
        "/TrapNet",
        "/Watermark",
        "/3D",
        "/Redact",
        "/RichMedia",
    }
)

PROPRIETARY_STAMP_SUBTYPES = frozenset(
    {
        "/GdPicture-AnnotationTypeEmbeddedImage",
        "/GdPicture-AnnotationTypeFreeHandHighlighter",
        "/GdPicture-AnnotationTypeLine",
        "/GdPicture-AnnotationTypeStickyNote",
        "/GdPicture-AnnotationTypeText",
    }
)


def _iter_annotation_arrays(
    pdf: Pdf,
) -> Iterator[tuple[Dictionary | Stream, Array]]:
    """Yields all /Annots arrays found in document dictionaries."""
    seen_owners: set[tuple[int, int]] = set()

    for obj in pdf.objects:
        try:
            owner = _resolve_indirect(obj)
            if not isinstance(owner, (Dictionary, Stream)):
                continue
        except Exception:
            continue

        try:
            owner_objgen = owner.objgen
        except Exception:
            owner_objgen = (0, 0)
        if owner_objgen != (0, 0):
            if owner_objgen in seen_owners:
                continue
            seen_owners.add(owner_objgen)

        try:
            annots = owner.get("/Annots")
            if annots is None:
                continue
            annots = _resolve_indirect(annots)
            if isinstance(annots, Array):
                yield owner, annots
        except Exception:
            continue


_annotation_arrays_cache: tuple[Pdf, list[tuple[Dictionary | Stream, Array]]] | None = (
    None
)
_annotation_arrays_lock = threading.Lock()


def _get_annotation_arrays(
    pdf: Pdf,
) -> list[tuple[Dictionary | Stream, Array]]:
    """Return cached annotation arrays for *pdf*, scanning only once."""
    global _annotation_arrays_cache  # noqa: PLW0603
    with _annotation_arrays_lock:
        if _annotation_arrays_cache is not None and _annotation_arrays_cache[0] is pdf:
            return _annotation_arrays_cache[1]
    result = list(_iter_annotation_arrays(pdf))
    with _annotation_arrays_lock:
        _annotation_arrays_cache = (pdf, result)
    return result


def _invalidate_annotation_arrays_cache(pdf: Pdf) -> None:
    """Invalidate the cache after structural annotation changes."""
    global _annotation_arrays_cache  # noqa: PLW0603
    with _annotation_arrays_lock:
        if _annotation_arrays_cache is not None and _annotation_arrays_cache[0] is pdf:
            _annotation_arrays_cache = None


def _is_non_compliant_annotation_subtype(subtype: str) -> bool:
    """Checks whether an annotation subtype violates PDF/A 6.3.1."""
    if subtype in FORBIDDEN_ANNOTATION_SUBTYPES:
        return True
    return subtype not in DEFINED_ANNOTATION_SUBTYPES


def _is_proprietary_stamp_subtype(subtype: str) -> bool:
    """Return ``True`` for known proprietary stamp-like annotations."""
    return subtype in PROPRIETARY_STAMP_SUBTYPES


def _annotation_is_hidden(resolved: Dictionary) -> bool:
    """Return ``True`` when an annotation is explicitly non-visible."""
    raw_flags = resolved.get("/F")
    flags = int(raw_flags) if raw_flags is not None else 0
    return bool(flags & (ANNOT_FLAG_INVISIBLE | ANNOT_FLAG_HIDDEN | ANNOT_FLAG_NOVIEW))


def _extract_normal_appearance_stream(annot: Dictionary) -> Stream | None:
    """Return a usable normal appearance stream for an annotation."""
    ap = annot.get("/AP")
    if ap is None:
        return None

    ap = _resolve_indirect(ap)
    if not isinstance(ap, Dictionary) or isinstance(ap, Stream):
        return None

    normal = ap.get("/N")
    if normal is None:
        return None

    normal = _resolve_indirect(normal)
    if isinstance(normal, Stream):
        return normal

    if isinstance(normal, Dictionary):
        return _extract_stream_from_state_dict(normal, annot)

    return None


def _ensure_page_xobject_resources(page) -> Dictionary:
    """Return the page ``/Resources/XObject`` dictionary, creating it if needed."""
    resources = page.obj.get("/Resources")
    resources = _resolve_indirect(resources) if resources is not None else None
    if not isinstance(resources, Dictionary):
        resources = Dictionary()
        page.obj[Name.Resources] = resources

    xobjects = resources.get("/XObject")
    xobjects = _resolve_indirect(xobjects) if xobjects is not None else None
    if not isinstance(xobjects, Dictionary):
        xobjects = Dictionary()
        resources[Name.XObject] = xobjects

    return xobjects


def _append_page_content_stream(page, stream: Stream) -> None:
    """Append a content stream so flattened appearances render on top."""
    contents = page.obj.get("/Contents")
    if contents is None:
        page.obj[Name.Contents] = stream
        return

    resolved = _resolve_indirect(contents)
    if isinstance(resolved, Array):
        resolved.append(stream)
        return

    page.obj[Name.Contents] = Array([contents, stream])


def _format_pdf_number(value: float) -> str:
    """Format a float as a compact PDF numeric literal."""
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text or "0"


def _appearance_invocation_stream(
    pdf: Pdf, resource_name: Name, annot, ap_stream
) -> Stream:
    """Build a page content stream that paints an annotation appearance."""
    rect = annot.get("/Rect")
    bbox = ap_stream.get("/BBox")
    if rect is None or bbox is None or len(rect) != 4 or len(bbox) != 4:
        command = f"q {str(resource_name)} Do Q\n".encode("ascii")
        return pdf.make_stream(command)

    try:
        rect_coords = [float(value) for value in rect]
        bbox_coords = [float(value) for value in bbox]
    except Exception:
        command = f"q {str(resource_name)} Do Q\n".encode("ascii")
        return pdf.make_stream(command)

    bbox_width = bbox_coords[2] - bbox_coords[0]
    bbox_height = bbox_coords[3] - bbox_coords[1]

    if bbox_width == 0 or bbox_height == 0:
        command = f"q {str(resource_name)} Do Q\n".encode("ascii")
        return pdf.make_stream(command)

    scale_x = (rect_coords[2] - rect_coords[0]) / bbox_width
    scale_y = (rect_coords[3] - rect_coords[1]) / bbox_height
    translate_x = rect_coords[0] - (bbox_coords[0] * scale_x)
    translate_y = rect_coords[1] - (bbox_coords[1] * scale_y)

    command = (
        "q "
        f"{_format_pdf_number(scale_x)} 0 0 {_format_pdf_number(scale_y)} "
        f"{_format_pdf_number(translate_x)} {_format_pdf_number(translate_y)} cm "
        f"{str(resource_name)} Do Q\n"
    ).encode("ascii")
    return pdf.make_stream(command)


def flatten_non_compliant_annotations(pdf: Pdf, level: str = "3b") -> int:
    """Flatten visible non-compliant page annotations into page content.

    Proprietary or otherwise non-standard annotations cannot remain in a
    PDF/A file.  When such an annotation has a usable normal appearance
    stream, this function paints that appearance into the owning page's
    content stream and removes the annotation entry afterwards so the
    visual result is preserved.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level (reserved for future use).

    Returns:
        Number of annotations flattened into page content.
    """
    del level  # reserved for future use

    flattened_count = 0

    for page in pdf.pages:
        annots = page.obj.get("/Annots")
        if annots is None:
            continue

        annots = _resolve_indirect(annots)
        if not isinstance(annots, Array):
            continue

        indices_to_remove: list[int] = []
        page_xobjects: Dictionary | None = None

        for index, annot in enumerate(annots):
            try:
                resolved = _resolve_indirect(annot)
                if not isinstance(resolved, Dictionary):
                    continue

                subtype = resolved.get("/Subtype")
                if subtype is None:
                    continue

                subtype_str = str(subtype)
                if not _is_non_compliant_annotation_subtype(subtype_str):
                    continue

                if _annotation_is_hidden(resolved):
                    continue

                appearance = _extract_normal_appearance_stream(resolved)
                if appearance is None:
                    continue

                page_xobjects = (
                    page_xobjects
                    if page_xobjects is not None
                    else _ensure_page_xobject_resources(page)
                )
                resource_index = len(page_xobjects)
                resource_name = Name(f"/FmFlattenAnnot{resource_index}")
                while page_xobjects.get(resource_name) is not None:
                    resource_index += 1
                    resource_name = Name(f"/FmFlattenAnnot{resource_index}")

                page_xobjects[resource_name] = appearance
                content_stream = _appearance_invocation_stream(
                    pdf, resource_name, resolved, appearance
                )
                _append_page_content_stream(page, content_stream)

                indices_to_remove.append(index)
                flattened_count += 1
                logger.debug(
                    "Flattened non-compliant annotation subtype %s into page content",
                    subtype_str,
                )
            except Exception as e:
                logger.debug("Error flattening non-compliant annotation: %s", e)

        for index in reversed(indices_to_remove):
            del annots[index]

        if len(annots) == 0 and indices_to_remove:
            del page.obj["/Annots"]

    if flattened_count > 0:
        _invalidate_annotation_arrays_cache(pdf)
        logger.info(
            "%d non-compliant annotation(s) flattened into page content",
            flattened_count,
        )

    return flattened_count


def normalize_proprietary_stamp_annotations(pdf: Pdf, level: str = "3b") -> int:
    """Convert known proprietary stamp annotations to standard ``/Stamp``.

    This is intentionally narrower than flattening: only known stamp-like
    vendor annotations with a usable normal appearance stream are normalized.
    Other non-standard annotations still fall through to the robust flattening
    path or, if they cannot be flattened, removal.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level (reserved for future use).

    Returns:
        Number of proprietary stamp annotations normalized.
    """
    del level  # reserved for future use

    normalized_count = 0

    for _, annots in _get_annotation_arrays(pdf):
        for annot in annots:
            try:
                resolved = _resolve_indirect(annot)
                if not isinstance(resolved, Dictionary):
                    continue

                subtype = resolved.get("/Subtype")
                if subtype is None:
                    continue

                subtype_str = str(subtype)
                if not _is_proprietary_stamp_subtype(subtype_str):
                    continue

                if _annotation_is_hidden(resolved):
                    continue

                if _extract_normal_appearance_stream(resolved) is None:
                    continue

                resolved[Name.Subtype] = Name.Stamp
                normalized_count += 1
                logger.debug(
                    "Normalized proprietary annotation subtype %s to /Stamp",
                    subtype_str,
                )
            except Exception as e:
                logger.debug("Error normalizing proprietary stamp annotation: %s", e)

    if normalized_count > 0:
        logger.info(
            "%d proprietary stamp annotation(s) normalized to /Stamp",
            normalized_count,
        )

    return normalized_count


def remove_forbidden_annotations(pdf: Pdf, level: str = "3b") -> int:
    """Removes forbidden and undefined annotation subtypes from the PDF.

    PDF/A rule 6.3.1 forbids certain multimedia annotation subtypes
    and also any annotation subtype not defined in ISO 32000.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level (for future compatibility).

    Returns:
        Number of non-compliant annotations removed.
    """
    removed_count = 0

    for owner, annots in _get_annotation_arrays(pdf):
        try:
            indices_to_remove = []
            for i, annot in enumerate(annots):
                try:
                    resolved = _resolve_indirect(annot)
                    if not isinstance(resolved, Dictionary):
                        continue
                    subtype = resolved.get("/Subtype")
                    if subtype is None:
                        continue

                    subtype_str = str(subtype)
                    if _is_non_compliant_annotation_subtype(subtype_str):
                        indices_to_remove.append(i)
                        logger.debug(
                            "Found non-compliant annotation subtype %s",
                            subtype_str,
                        )
                except Exception:
                    continue

            for i in reversed(indices_to_remove):
                del annots[i]
                removed_count += 1

            if len(annots) == 0:
                del owner["/Annots"]

        except Exception as e:
            logger.debug("Error processing annotation array: %s", e)

    if removed_count > 0:
        _invalidate_annotation_arrays_cache(pdf)
        logger.info("%d non-compliant annotation(s) removed", removed_count)
    return removed_count


def fix_annotation_flags(pdf: Pdf, level: str = "3b") -> int:
    """Sets the Print flag on all annotations for PDF/A compliance.

    PDF/A requires each annotation to have an /F entry with the Print
    flag set, while Hidden, Invisible and NoView must not be set.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level ('2b' or '3b').

    Returns:
        Number of annotations fixed.
    """
    fixed_count = 0

    for _, annots in _get_annotation_arrays(pdf):
        try:
            for annot in annots:
                try:
                    resolved = _resolve_indirect(annot)
                    if not isinstance(resolved, Dictionary):
                        continue

                    raw_flags = resolved.get("/F")
                    current_flags = int(raw_flags) if raw_flags is not None else 0
                    new_flags = current_flags
                    was_fixed = raw_flags is None

                    # Remove Invisible flag (PDF/A forbids it for undefined
                    # annotation types; clearing unconditionally is safe
                    # since it has no effect on standard types)
                    if current_flags & ANNOT_FLAG_INVISIBLE:
                        new_flags = new_flags & ~ANNOT_FLAG_INVISIBLE
                        was_fixed = True

                    # Remove Hidden flag (PDF/A forbids hidden annotations)
                    if current_flags & ANNOT_FLAG_HIDDEN:
                        new_flags = new_flags & ~ANNOT_FLAG_HIDDEN
                        was_fixed = True

                    # Remove NoView flag (PDF/A forbids NoView annotations)
                    if current_flags & ANNOT_FLAG_NOVIEW:
                        new_flags = new_flags & ~ANNOT_FLAG_NOVIEW
                        was_fixed = True

                    # Remove ToggleNoView flag (PDF/A forbids it)
                    if current_flags & ANNOT_FLAG_TOGGLENOVIEW:
                        new_flags = new_flags & ~ANNOT_FLAG_TOGGLENOVIEW
                        was_fixed = True

                    # Ensure Print flag is set for every annotation.
                    subtype = resolved.get("/Subtype")
                    if not (new_flags & ANNOT_FLAG_PRINT):
                        new_flags = new_flags | ANNOT_FLAG_PRINT
                        was_fixed = True

                    # PDF/A-2/3: Text annotations require NoZoom and
                    # NoRotate flags (ISO 19005-2, 6.5.2)
                    is_text = subtype is not None and str(subtype) == "/Text"
                    if is_text:
                        if not (new_flags & ANNOT_FLAG_NOZOOM):
                            new_flags = new_flags | ANNOT_FLAG_NOZOOM
                            was_fixed = True
                        if not (new_flags & ANNOT_FLAG_NOROTATE):
                            new_flags = new_flags | ANNOT_FLAG_NOROTATE
                            was_fixed = True

                    # Update flags if changed
                    if was_fixed:
                        resolved[Name("/F")] = new_flags
                        fixed_count += 1
                        logger.debug(
                            "Fixed annotation flags: %d -> %d",
                            current_flags,
                            new_flags,
                        )

                except Exception as e:
                    logger.debug("Error processing annotation flags: %s", e)

        except Exception as e:
            logger.debug("Error processing annotation array: %s", e)

    if fixed_count > 0:
        logger.info("%d annotation flag(s) fixed", fixed_count)
    return fixed_count


def _create_minimal_appearance_stream(pdf: Pdf, annot) -> object:
    """Creates a minimal empty Form XObject appearance stream.

    Args:
        pdf: Opened pikepdf PDF object.
        annot: The annotation dictionary to derive BBox from.

    Returns:
        A pikepdf Stream object suitable for use as /AP /N.
    """
    bbox = Array([0, 0, 0, 0])
    try:
        rect = annot.get("/Rect")
        if rect is not None and len(rect) == 4:
            x1 = float(rect[0])
            y1 = float(rect[1])
            x2 = float(rect[2])
            y2 = float(rect[3])
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            bbox = Array([0, 0, width, height])
    except Exception:
        pass

    stream = pdf.make_stream(b"")
    stream[Name.Type] = Name.XObject
    stream[Name.Subtype] = Name.Form
    stream[Name.BBox] = bbox
    return stream


def remove_needs_appearances(pdf: Pdf) -> bool:
    """Remove /NeedAppearances from /AcroForm.

    PDF/A requires that /NeedAppearances is either absent or false.
    When set to true, it tells a viewer to regenerate appearance streams
    on the fly, which is forbidden in PDF/A.

    This should be called after ensure_appearance_streams() so that all
    widgets already have valid /AP entries.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        True if the flag was removed, False otherwise.
    """
    try:
        root = pdf.Root
        if root is None:
            return False
        acroform = root.get("/AcroForm")
        if acroform is None:
            return False
        acroform = _resolve_indirect(acroform)
        removed = False
        # Correct key per ISO 32000.
        if acroform.get("/NeedAppearances") is not None:
            del acroform["/NeedAppearances"]
            removed = True
        # Backward-compatible cleanup for legacy typo in earlier sanitizer
        # versions and malformed input files.
        if acroform.get("/NeedsAppearances") is not None:
            del acroform["/NeedsAppearances"]
            removed = True
        if not removed:
            return False
        logger.debug("Removed /NeedAppearances from /AcroForm")
        return True
    except Exception as e:
        logger.debug("Error removing /NeedAppearances: %s", e)
        return False


def _fix_appearance_state_dict(
    pdf: Pdf,
    annot,
    ap,
) -> int:
    """Replace /AP/N state dictionary with a single Stream when required.

    PDF/A rule 6.3.3 requires that for non-Widget annotations (and Widget
    annotations without /FT=Btn), /AP/N must be a single appearance Stream,
    not a state sub-dictionary.  If /AP/N is a Dictionary, this function
    extracts a usable Stream from it (matching /AS or the first available)
    or creates a minimal Form XObject.

    Returns:
        1 if the appearance was fixed, 0 otherwise.
    """
    n = ap.get("/N")
    if n is None:
        return 0
    n = _resolve_indirect(n)

    # Already a Stream — nothing to fix
    if isinstance(n, Stream):
        return 0

    # Not a Dictionary either — replace with minimal stream
    if not isinstance(n, Dictionary):
        ap[Name.N] = _create_minimal_appearance_stream(pdf, annot)
        logger.debug("Replaced invalid /AP/N with minimal stream")
        return 1

    # /AP/N is a Dictionary (state dict) — check if this is allowed
    subtype = annot.get("/Subtype")
    is_widget = subtype is not None and str(subtype) == "/Widget"

    if is_widget:
        ft = _get_field_type(annot)
        if ft == "/Btn":
            # Btn widgets are allowed to have state dicts
            return 0

    # Non-Widget or Widget-non-Btn: collapse state dict to single Stream
    stream = _extract_stream_from_state_dict(n, annot)
    if stream is not None:
        ap[Name.N] = stream
    else:
        ap[Name.N] = _create_minimal_appearance_stream(pdf, annot)

    # Remove /AS since the appearance is no longer a state dict
    if annot.get("/AS") is not None:
        del annot["/AS"]

    logger.debug("Collapsed /AP/N state dict to single stream")
    return 1


def _extract_stream_from_state_dict(state_dict, annot):
    """Try to extract a usable Stream from an appearance state dictionary.

    Prefers the entry matching /AS, then falls back to the first Stream.

    Returns:
        A Stream object, or None if no usable stream is found.
    """
    # Try the entry matching /AS first
    as_val = annot.get("/AS")
    if as_val is not None:
        as_str = str(as_val)
        entry = state_dict.get(as_str)
        if entry is not None:
            entry = _resolve_indirect(entry)
            if isinstance(entry, Stream):
                return entry

    # Fall back to first Stream in the dictionary
    for key in state_dict.keys():
        try:
            entry = _resolve_indirect(state_dict.get(key))
            if isinstance(entry, Stream):
                return entry
        except Exception:
            continue
    return None


def ensure_appearance_streams(pdf: Pdf, level: str = "3b") -> int:
    """Ensures all annotations have appearance streams (/AP /N).

    PDF/A-2/3 (ISO 19005-2/3, clause 6.5.3) requires all annotations
    (except Popup) to have an /AP dictionary with at least an /N
    (Normal appearance) entry. For Widget annotations (form fields),
    visible appearance streams are generated; for other annotation types,
    minimal empty Form XObjects are used.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level (for future compatibility).

    Returns:
        Number of appearance streams added.
    """
    added_count = 0

    # Resolve AcroForm once for the entire PDF
    acroform = None
    try:
        root = pdf.Root
        if root is not None:
            af = root.get("/AcroForm")
            if af is not None:
                acroform = _resolve_indirect(af)
    except Exception:
        pass

    for _, annots in _get_annotation_arrays(pdf):
        for annot in annots:
            try:
                resolved = _resolve_indirect(annot)

                # Skip Popup annotations (exempt per spec)
                subtype = resolved.get("/Subtype")
                if subtype is not None and str(subtype) == "/Popup":
                    continue

                # Skip Link annotations (exempt per rule 6.3.3-1)
                if subtype is not None and str(subtype) == "/Link":
                    continue

                # Skip zero-size annotations (invisible, no /AP needed).
                # Per ISO 19005-2 rule 6.3.3, both pairs must be equal:
                # "value 1 is equal to value 3 AND value 2 is equal to value 4".
                rect = resolved.get("/Rect")
                if rect is not None:
                    try:
                        coords = [float(rect[i]) for i in range(4)]
                        if coords[0] == coords[2] and coords[1] == coords[3]:
                            continue
                    except Exception:
                        pass

                ap = resolved.get("/AP")
                if ap is None:
                    # No /AP at all — create /AP dict with /N
                    is_widget = subtype is not None and str(subtype) == "/Widget"
                    if is_widget:
                        appearance = create_widget_appearance(pdf, resolved, acroform)
                    else:
                        appearance = _create_minimal_appearance_stream(pdf, resolved)

                    resolved[Name.AP] = Dictionary(N=appearance)
                    added_count += 1
                    logger.debug("Added /AP /N to annotation")
                else:
                    # /AP exists — check for /N
                    ap = _resolve_indirect(ap)

                    if ap.get("/N") is None:
                        is_widget = subtype is not None and str(subtype) == "/Widget"
                        if is_widget:
                            appearance = create_widget_appearance(
                                pdf, resolved, acroform
                            )
                        else:
                            appearance = _create_minimal_appearance_stream(
                                pdf, resolved
                            )
                        ap[Name.N] = appearance
                        added_count += 1
                        logger.debug("Added /N to existing /AP")
                    else:
                        # /AP/N exists — check it's the right type
                        added_count += _fix_appearance_state_dict(
                            pdf,
                            resolved,
                            ap,
                        )

            except Exception as e:
                logger.debug(
                    "Error processing annotation: %s",
                    e,
                )

    if added_count > 0:
        logger.info("%d appearance stream(s) added", added_count)
    return added_count


def remove_non_normal_appearance_keys(pdf: Pdf) -> int:
    """Removes /R and /D keys from annotation /AP dictionaries.

    PDF/A rule 6.3.3 allows only /N (normal appearance) in annotation
    appearance dictionaries. Rollover (/R) and down (/D) appearances
    are removed when present.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of removed /AP keys.
    """
    removed_count = 0

    for _, annots in _get_annotation_arrays(pdf):
        try:
            for annot in annots:
                try:
                    resolved = _resolve_indirect(annot)
                    if not isinstance(resolved, Dictionary):
                        continue

                    ap = resolved.get("/AP")
                    if ap is None:
                        continue
                    ap = _resolve_indirect(ap)
                    if not isinstance(ap, Dictionary) or isinstance(ap, Stream):
                        continue

                    for key in ("/R", "/D"):
                        if ap.get(key) is not None:
                            del ap[key]
                            removed_count += 1
                except Exception as e:
                    logger.debug("Error processing annotation /AP keys: %s", e)
        except Exception as e:
            logger.debug("Error processing annotation array for /AP keys: %s", e)

    if removed_count > 0:
        logger.info("%d annotation /AP key(s) removed (/R,/D)", removed_count)
    return removed_count


def remove_annotation_colors(pdf: Pdf, level: str = "3b") -> int:
    """Removes /C and /IC color arrays from annotations.

    /C (border color) and /IC (interior color) are Device color values
    (DeviceRGB or DeviceGray arrays) which violate PDF/A's requirement
    for device-independent color (ISO 19005-2).  Since /AP appearance
    streams determine the actual visual rendering, these keys can be
    safely removed.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level (for future compatibility).

    Returns:
        Number of color arrays removed.
    """
    removed_count = 0

    for _, annots in _get_annotation_arrays(pdf):
        for annot in annots:
            try:
                resolved = _resolve_indirect(annot)

                for key in ("/C", "/IC"):
                    if resolved.get(key) is not None:
                        del resolved[key]
                        removed_count += 1
                        logger.debug("Removed %s from annotation", key)

            except Exception as e:
                logger.debug(
                    "Error processing annotation colors: %s",
                    e,
                )

    if removed_count > 0:
        logger.info("%d annotation color array(s) removed", removed_count)
    return removed_count


def _get_field_type(annot) -> str | None:
    """Get the field type of a widget annotation, walking /Parent chain.

    Args:
        annot: Resolved annotation dictionary.

    Returns:
        The /FT value as a string (e.g. "/Btn"), or None if not found.
    """
    current = annot
    seen = set()
    while current is not None:
        objgen = current.objgen
        if objgen != (0, 0) and objgen in seen:
            break
        if objgen != (0, 0):
            seen.add(objgen)

        ft = current.get("/FT")
        if ft is not None:
            return str(ft)

        parent = current.get("/Parent")
        if parent is None:
            break
        try:
            current = _resolve_indirect(parent)
        except Exception:
            break
    return None


def fix_button_appearance_subdicts(pdf: Pdf) -> int:
    """Ensure Btn widget /AP/N is a state subdictionary, not a bare Stream.

    PDF/A rule 6.3.3 requires that if an annotation's /Subtype is /Widget
    and /FT is /Btn, the /AP/N value must be a subdictionary mapping state
    names to appearance streams (not a single Stream).

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of annotations fixed.
    """
    fixed_count = 0

    for _, annots in _get_annotation_arrays(pdf):
        for annot in annots:
            try:
                resolved = _resolve_indirect(annot)

                subtype = resolved.get("/Subtype")
                if subtype is None or str(subtype) != "/Widget":
                    continue

                ft = _get_field_type(resolved)
                if ft != "/Btn":
                    continue

                ap = resolved.get("/AP")
                if ap is None:
                    continue
                ap = _resolve_indirect(ap)

                n = ap.get("/N")
                if n is None:
                    continue
                n = _resolve_indirect(n)

                # Already a proper subdictionary (not a Stream)
                if isinstance(n, Dictionary) and not isinstance(n, Stream):
                    continue

                if not isinstance(n, Stream):
                    continue

                # Determine state key: /AS -> /V -> fallback "Yes"
                state_key = None
                as_val = resolved.get("/AS")
                if as_val is not None:
                    state_key = str(as_val).lstrip("/")
                if not state_key:
                    v_val = resolved.get("/V")
                    if v_val is not None:
                        state_key = str(v_val).lstrip("/")
                if not state_key:
                    state_key = "Yes"

                # Wrap the Stream in a state Dictionary
                state_dict = Dictionary()
                state_dict[Name("/" + state_key)] = n
                ap[Name("/N")] = state_dict

                # Ensure /AS matches so the viewer picks the right state
                resolved[Name("/AS")] = Name("/" + state_key)

                fixed_count += 1
                logger.debug(
                    "Wrapped Btn /AP/N Stream in state dict (key=%s)",
                    state_key,
                )
            except Exception as e:
                logger.debug("Error fixing Btn AP subdict: %s", e)

    if fixed_count > 0:
        logger.info(
            "%d Btn widget /AP/N stream(s) wrapped in state dict",
            fixed_count,
        )
    return fixed_count


def fix_annotation_opacity(pdf: Pdf, level: str = "3b") -> int:
    """Fixes annotation-level /CA (opacity) for PDF/A compliance.

    ISO 19005-2, clause 6.5.3 requires that the /CA key on annotation
    dictionaries (not to be confused with ExtGState /CA) must be 1.0
    if present.  Non-conforming values are set to 1.0; missing /CA is
    acceptable and left untouched.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        level: PDF/A conformance level (for future compatibility).

    Returns:
        Number of annotations fixed.
    """
    fixed_count = 0

    for _, annots in _get_annotation_arrays(pdf):
        for annot in annots:
            try:
                resolved = _resolve_indirect(annot)

                ca = resolved.get("/CA")
                if ca is None:
                    continue

                try:
                    ca_val = float(ca)
                except (TypeError, ValueError):
                    # Non-numeric /CA — force to 1.0
                    resolved[Name("/CA")] = 1.0
                    fixed_count += 1
                    logger.debug("Fixed non-numeric annotation /CA")
                    continue

                if ca_val != 1.0:
                    resolved[Name("/CA")] = 1.0
                    fixed_count += 1
                    logger.debug(
                        "Fixed annotation /CA %.4f -> 1.0",
                        ca_val,
                    )

            except Exception as e:
                logger.debug(
                    "Error processing annotation /CA: %s",
                    e,
                )

    if fixed_count > 0:
        logger.info("%d annotation /CA opacity value(s) fixed", fixed_count)
    return fixed_count
