# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""XObject handling for PDF/A compliance."""

import logging

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream
from pikepdf import parse_content_stream as _parse_content_stream
from pikepdf import unparse_content_stream as _unparse_content_stream

from ..utils import resolve_indirect as _resolve_indirect
from .base import FORBIDDEN_XOBJECT_SUBTYPES

logger = logging.getLogger(__name__)

# Allowed BitsPerComponent values per ISO 19005-2, Clause 6.2.8
VALID_BITS_PER_COMPONENT = frozenset({1, 2, 4, 8, 16})


def _remove_forbidden_form_keys(xobj: Stream, key_name: str) -> int:
    """Remove Form-XObject keys forbidden by PDF/A 6.2.9."""
    removed = 0

    if "/Ref" in xobj:
        del xobj["/Ref"]
        removed += 1
        logger.debug("Removed /Ref from XObject: %s", key_name)

    subtype2 = xobj.get("/Subtype2")
    if subtype2 is not None:
        subtype2 = _resolve_indirect(subtype2)
        if isinstance(subtype2, Name) and str(subtype2) == "/PS":
            del xobj["/Subtype2"]
            removed += 1
            logger.debug("Removed /Subtype2 /PS from Form XObject: %s", key_name)

    if "/PS" in xobj:
        del xobj["/PS"]
        removed += 1
        logger.debug("Removed /PS from Form XObject: %s", key_name)

    return removed


def _process_xobjects_for_removal(
    xobjects: pikepdf.Dictionary,
    pdf: Pdf,
    visited: set[tuple[int, int]],
) -> tuple[list[str], int]:
    """Processes XObjects for forbidden subtypes, alternates and OPI.

    Recursively processes XObjects, collecting forbidden subtypes for removal
    and removing /Alternates arrays. Nested forbidden XObjects are removed
    immediately during recursion.

    Args:
        xobjects: XObject dictionary from page or Form XObject resources.
        pdf: Parent PDF (for object ID tracking).
        visited: Set of already-visited objgen tuples for cycle detection.

    Returns:
        Tuple of (list of keys to remove at this level, total count of
        elements removed including nested XObjects and Alternates).
    """
    keys_to_remove: list[str] = []
    removed_count = 0

    for key in xobjects.keys():
        try:
            xobj = xobjects.get(key)
            if xobj is None:
                continue

            xobj = _resolve_indirect(xobj)

            # Cycle detection using objgen
            obj_key = xobj.objgen
            if obj_key != (0, 0):
                if obj_key in visited:
                    continue
                visited.add(obj_key)

            # Check subtype
            subtype = xobj.get("/Subtype")
            if subtype is not None:
                subtype_str = str(subtype)

                # Mark forbidden subtypes for removal
                if subtype_str in FORBIDDEN_XOBJECT_SUBTYPES:
                    keys_to_remove.append(str(key))
                    logger.debug("Found forbidden XObject %s: %s", subtype_str, key)
                    continue

                # Remove /Alternates from any XObject
                if "/Alternates" in xobj:
                    del xobj["/Alternates"]
                    removed_count += 1
                    logger.debug("Removed /Alternates from XObject: %s", key)

                # Remove /OPI from any XObject (ISO 19005-2, 6.2.4)
                if "/OPI" in xobj:
                    del xobj["/OPI"]
                    removed_count += 1
                    logger.debug("Removed /OPI from XObject: %s", key)

                # Recurse into Form XObjects for nested XObjects
                if subtype_str == "/Form":
                    removed_count += _remove_forbidden_form_keys(xobj, str(key))
                    nested_resources = xobj.get("/Resources")
                    if nested_resources is not None:
                        nested_resources = _resolve_indirect(nested_resources)
                        nested_xobjects = nested_resources.get("/XObject")
                        if nested_xobjects is not None:
                            nested_xobjects = _resolve_indirect(nested_xobjects)
                            nested_keys, nested_removed = _process_xobjects_for_removal(
                                nested_xobjects, pdf, visited
                            )
                            # Remove forbidden XObjects from nested resources
                            for nested_key in nested_keys:
                                del nested_xobjects[nested_key]
                                removed_count += 1
                                logger.debug(
                                    "Removed nested forbidden XObject: %s", nested_key
                                )
                            removed_count += nested_removed

        except Exception as e:
            logger.debug("Error processing XObject %s: %s", key, e)

    return keys_to_remove, removed_count


def _remove_forbidden_in_ap_stream(
    ap_entry, pdf: Pdf, visited: set[tuple[int, int]]
) -> int:
    """Remove forbidden XObjects from an annotation appearance stream entry.

    An AP entry value can be a Form XObject (stream) directly, or a
    dictionary of sub-state Form XObjects (e.g. /Yes, /Off).

    Args:
        ap_entry: An appearance entry (N, R, or D value).
        pdf: Parent PDF (for object ID tracking).
        visited: Set of (objnum, gen) tuples for cycle detection.

    Returns:
        Number of forbidden elements removed.
    """
    removed = 0
    ap_entry = _resolve_indirect(ap_entry)

    if isinstance(ap_entry, Stream):
        resources = ap_entry.get("/Resources")
        if resources:
            resources = _resolve_indirect(resources)
            xobjects = resources.get("/XObject")
            if xobjects:
                xobjects = _resolve_indirect(xobjects)
                keys_to_remove, nested_removed = _process_xobjects_for_removal(
                    xobjects, pdf, visited
                )
                for key in keys_to_remove:
                    del xobjects[key]
                    removed += 1
                removed += nested_removed
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                resources = state_stream.get("/Resources")
                if resources:
                    resources = _resolve_indirect(resources)
                    xobjects = resources.get("/XObject")
                    if xobjects:
                        xobjects = _resolve_indirect(xobjects)
                        keys_to_remove, nested_removed = _process_xobjects_for_removal(
                            xobjects, pdf, visited
                        )
                        for key in keys_to_remove:
                            del xobjects[key]
                            removed += 1
                        removed += nested_removed

    return removed


def remove_forbidden_xobjects(pdf: Pdf) -> int:
    """Removes forbidden XObjects from the PDF.

    PDF/A forbids:
    - PostScript XObjects (/Subtype /PS)
    - Reference XObjects (/Subtype /Ref)
    - /Alternates arrays in XObjects
    - /OPI dictionaries in XObjects
    - In Form XObjects: /Ref, /Subtype2 /PS, and /PS keys

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of forbidden elements removed.
    """
    removed_count = 0
    visited: set[tuple[int, int]] = set()

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve_indirect(page.obj)

            # 1. Page → Resources → XObject
            resources = page_dict.get("/Resources")
            if resources is not None:
                resources = _resolve_indirect(resources)

                xobjects = resources.get("/XObject")
                if xobjects is not None:
                    xobjects = _resolve_indirect(xobjects)

                    keys_to_remove, nested_removed = _process_xobjects_for_removal(
                        xobjects, pdf, visited
                    )

                    # Remove forbidden XObjects (two-phase deletion)
                    for key in keys_to_remove:
                        del xobjects[key]
                        removed_count += 1
                        logger.debug(
                            "Removed forbidden XObject on page %d: %s",
                            page_num,
                            key,
                        )

                    removed_count += nested_removed

            # 2. Page → Annots → AP streams
            annots = page_dict.get("/Annots")
            if annots:
                annots = _resolve_indirect(annots)
                for annot in annots:
                    annot = _resolve_indirect(annot)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = annot.get("/AP")
                    if not ap:
                        continue
                    ap = _resolve_indirect(ap)
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry is not None:
                            removed_count += _remove_forbidden_in_ap_stream(
                                ap_entry, pdf, visited
                            )

        except Exception as e:
            logger.debug("Error processing XObjects on page %d: %s", page_num, e)

    if removed_count > 0:
        logger.info("%d forbidden XObject element(s) removed", removed_count)
    return removed_count


def _fix_interpolate_in_xobjects(
    xobjects: pikepdf.Dictionary,
    visited: set[tuple[int, int]],
    visited_inline_streams: set[tuple[int, int]],
) -> int:
    """Fixes /Interpolate on Image XObjects within an XObject dictionary.

    Recursively processes XObjects: for each Image with Interpolate=true,
    sets it to false. Recurses into Form XObjects for nested images.

    Args:
        xobjects: XObject dictionary from page or Form XObject resources.
        visited: Set of already-visited objgen tuples for cycle detection.

    Returns:
        Number of images fixed.
    """
    fixed_count = 0

    for key in xobjects.keys():
        try:
            xobj = xobjects.get(key)
            if xobj is None:
                continue

            xobj = _resolve_indirect(xobj)

            # Cycle detection using objgen
            obj_key = xobj.objgen
            if obj_key != (0, 0):
                if obj_key in visited:
                    continue
                visited.add(obj_key)

            subtype = xobj.get("/Subtype")
            if subtype is None:
                continue
            subtype_str = str(subtype)

            if subtype_str == "/Image":
                interp = xobj.get("/Interpolate")
                if interp is not None and bool(interp):
                    xobj["/Interpolate"] = False
                    fixed_count += 1
                    logger.debug("Set /Interpolate to false on Image XObject: %s", key)

            elif subtype_str == "/Form":
                fixed_count += _fix_inline_interpolate_in_stream_once(
                    xobj, visited_inline_streams
                )
                nested_resources = xobj.get("/Resources")
                if nested_resources is not None:
                    nested_resources = _resolve_indirect(nested_resources)
                    nested_xobjects = nested_resources.get("/XObject")
                    if nested_xobjects is not None:
                        nested_xobjects = _resolve_indirect(nested_xobjects)
                        fixed_count += _fix_interpolate_in_xobjects(
                            nested_xobjects, visited, visited_inline_streams
                        )

        except Exception as e:
            logger.debug("Error checking /Interpolate on XObject %s: %s", key, e)

    return fixed_count


def _fix_interpolate_in_ap_stream(
    ap_entry,
    visited: set[tuple[int, int]],
    visited_inline_streams: set[tuple[int, int]],
) -> int:
    """Fix /Interpolate on images in an annotation appearance stream entry.

    An AP entry value can be a Form XObject (stream) directly, or a
    dictionary of sub-state Form XObjects (e.g. /Yes, /Off).

    Args:
        ap_entry: An appearance entry (N, R, or D value).
        visited: Set of (objnum, gen) tuples for cycle detection.

    Returns:
        Number of images fixed.
    """
    fixed = 0
    ap_entry = _resolve_indirect(ap_entry)

    if isinstance(ap_entry, Stream):
        fixed += _fix_inline_interpolate_in_stream_once(
            ap_entry, visited_inline_streams
        )
        resources = ap_entry.get("/Resources")
        if resources:
            resources = _resolve_indirect(resources)
            xobjects = resources.get("/XObject")
            if xobjects:
                xobjects = _resolve_indirect(xobjects)
                fixed += _fix_interpolate_in_xobjects(
                    xobjects, visited, visited_inline_streams
                )
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                fixed += _fix_inline_interpolate_in_stream_once(
                    state_stream, visited_inline_streams
                )
                resources = state_stream.get("/Resources")
                if resources:
                    resources = _resolve_indirect(resources)
                    xobjects = resources.get("/XObject")
                    if xobjects:
                        xobjects = _resolve_indirect(xobjects)
                        fixed += _fix_interpolate_in_xobjects(
                            xobjects, visited, visited_inline_streams
                        )

    return fixed


def _serialize_inline_token(token) -> bytes:
    """Serialize one inline-image metadata token."""
    if isinstance(token, bool):
        return b"true" if token else b"false"
    return str(token).encode("ascii")


def _serialize_inline_tokens(tokens: list) -> bytes:
    """Serialize inline-image metadata tokens to bytes."""
    return b" ".join(_serialize_inline_token(token) for token in tokens)


def _create_inline_image_instruction(
    image_tokens: list, image_data: bytes
) -> pikepdf.ContentStreamInlineImage | None:
    """Create a parsed inline-image instruction from token and payload bytes."""
    metadata = _serialize_inline_tokens(image_tokens)
    inline_bytes = b"BI\n" + metadata + b"\nID\n" + image_data + b"EI\n"

    with Pdf.new() as temp_pdf:
        temp_stream = Stream(temp_pdf, inline_bytes)
        for item in _parse_content_stream(temp_stream):
            if isinstance(item, pikepdf.ContentStreamInlineImage):
                return item
    return None


def _extract_inline_image_payload(inline_image) -> bytes | None:
    """Extract raw payload bytes from ``PdfInlineImage.unparse()``."""
    inline_bytes = inline_image.unparse()
    marker = b"\nID\n"
    marker_idx = inline_bytes.find(marker)
    if marker_idx < 0:
        return None

    payload_and_end = inline_bytes[marker_idx + len(marker) :]
    if payload_and_end.endswith(b" EI"):
        return payload_and_end[:-3]
    if payload_and_end.endswith(b"\nEI"):
        return payload_and_end[:-3]
    if payload_and_end.endswith(b"EI"):
        return payload_and_end[:-2].rstrip(b" \r\n")
    return None


def _fix_inline_image_interpolate_in_stream(stream: Stream) -> int:
    """Set /I or /Interpolate to false in inline images of one content stream."""
    try:
        instructions = list(_parse_content_stream(stream))
    except Exception:
        return 0

    fixed_count = 0
    changed = False

    for index, item in enumerate(instructions):
        if not isinstance(item, pikepdf.ContentStreamInlineImage):
            continue
        if not item.operands:
            continue

        inline_image = item.operands[0]
        # Private pikepdf API (tested with pikepdf 8.x–9.x): _image_object
        # holds the inline image's metadata key/value token tuple.
        image_tokens = list(inline_image._image_object)

        local_changed = False
        for token_idx in range(0, len(image_tokens) - 1, 2):
            key = image_tokens[token_idx]
            if not isinstance(key, Name):
                continue
            key_str = str(key)
            if key_str not in ("/I", "/Interpolate"):
                continue

            value = image_tokens[token_idx + 1]
            if bool(value):
                image_tokens[token_idx + 1] = False
                fixed_count += 1
                local_changed = True

        if not local_changed:
            continue

        payload = _extract_inline_image_payload(inline_image)
        if payload is None:
            continue

        replacement = _create_inline_image_instruction(image_tokens, payload)
        if replacement is None:
            continue

        instructions[index] = replacement
        changed = True

    if changed:
        stream.write(_unparse_content_stream(instructions))

    return fixed_count


def _fix_inline_interpolate_in_stream_once(
    stream: Stream, visited_inline_streams: set[tuple[int, int]]
) -> int:
    """Apply inline-image interpolation fix once per indirect stream."""
    objgen = stream.objgen
    if objgen != (0, 0):
        if objgen in visited_inline_streams:
            return 0
        visited_inline_streams.add(objgen)
    return _fix_inline_image_interpolate_in_stream(stream)


def _fix_inline_interpolate_in_contents(
    contents, visited_inline_streams: set[tuple[int, int]]
) -> int:
    """Apply inline-image interpolation fix to a page's /Contents."""
    fixed = 0
    contents = _resolve_indirect(contents)

    if isinstance(contents, Stream):
        return _fix_inline_interpolate_in_stream_once(contents, visited_inline_streams)

    if isinstance(contents, Array):
        for stream_ref in contents:
            stream_obj = _resolve_indirect(stream_ref)
            if isinstance(stream_obj, Stream):
                fixed += _fix_inline_interpolate_in_stream_once(
                    stream_obj, visited_inline_streams
                )

    return fixed


def fix_image_interpolate(pdf: Pdf) -> int:
    """Sets interpolation flags to false on all image dictionaries.

    ISO 19005-2, Clause 6.2.8 requires:
    - image dictionaries: /Interpolate must be false
    - inline image dictionaries: /I must be false

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of images where /Interpolate was fixed.
    """
    fixed_count = 0
    visited: set[tuple[int, int]] = set()
    visited_inline_streams: set[tuple[int, int]] = set()

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve_indirect(page.obj)

            # 0. Page /Contents inline images
            contents = page_dict.get("/Contents")
            if contents is not None:
                fixed_count += _fix_inline_interpolate_in_contents(
                    contents, visited_inline_streams
                )

            # 1. Page → Resources → XObject
            resources = page_dict.get("/Resources")
            if resources is not None:
                resources = _resolve_indirect(resources)

                xobjects = resources.get("/XObject")
                if xobjects is not None:
                    xobjects = _resolve_indirect(xobjects)
                    fixed_count += _fix_interpolate_in_xobjects(
                        xobjects, visited, visited_inline_streams
                    )

            # 2. Page → Annots → AP streams
            annots = page_dict.get("/Annots")
            if annots:
                annots = _resolve_indirect(annots)
                for annot in annots:
                    annot = _resolve_indirect(annot)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = annot.get("/AP")
                    if not ap:
                        continue
                    ap = _resolve_indirect(ap)
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry is not None:
                            fixed_count += _fix_interpolate_in_ap_stream(
                                ap_entry, visited, visited_inline_streams
                            )

        except Exception as e:
            logger.debug("Error checking /Interpolate on page %d: %s", page_num, e)

    if fixed_count > 0:
        logger.info("%d image(s) had /Interpolate set to false", fixed_count)
    return fixed_count


def _validate_bpc_in_xobjects(
    xobjects: pikepdf.Dictionary,
    visited: set[tuple[int, int]],
) -> dict[str, int]:
    """Validates BitsPerComponent on Image XObjects within an XObject dictionary.

    Recursively processes XObjects: for each Image, checks that BPC is one of
    the allowed values (1, 2, 4, 8, 16) and that image masks have BPC == 1.
    Recurses into Form XObjects for nested images.

    Args:
        xobjects: XObject dictionary from page or Form XObject resources.
        visited: Set of already-visited objgen tuples for cycle detection.

    Returns:
        Dictionary with invalid_bpc and mask_bpc_invalid counts.
    """
    result = {"invalid_bpc": 0, "mask_bpc_invalid": 0}

    for key in xobjects.keys():
        try:
            xobj = xobjects.get(key)
            if xobj is None:
                continue

            xobj = _resolve_indirect(xobj)

            # Cycle detection using objgen
            obj_key = xobj.objgen
            if obj_key != (0, 0):
                if obj_key in visited:
                    continue
                visited.add(obj_key)

            subtype = xobj.get("/Subtype")
            if subtype is None:
                continue
            subtype_str = str(subtype)

            if subtype_str == "/Image":
                bpc = xobj.get("/BitsPerComponent")
                if bpc is not None:
                    try:
                        bpc_val = int(bpc)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Image XObject %s has non-integer BitsPerComponent: %s",
                            key,
                            bpc,
                        )
                        result["invalid_bpc"] += 1
                        continue

                    if bpc_val not in VALID_BITS_PER_COMPONENT:
                        logger.warning(
                            "Image XObject %s has invalid "
                            "BitsPerComponent %d "
                            "(allowed: 1, 2, 4, 8, 16)",
                            key,
                            bpc_val,
                        )
                        result["invalid_bpc"] += 1

                    # Image masks must have BPC == 1
                    is_mask = xobj.get("/ImageMask")
                    if is_mask is not None and bool(is_mask):
                        if bpc_val != 1:
                            logger.warning(
                                "Image mask %s has BitsPerComponent %d (must be 1)",
                                key,
                                bpc_val,
                            )
                            result["mask_bpc_invalid"] += 1

            elif subtype_str == "/Form":
                nested_resources = xobj.get("/Resources")
                if nested_resources is not None:
                    nested_resources = _resolve_indirect(nested_resources)
                    nested_xobjects = nested_resources.get("/XObject")
                    if nested_xobjects is not None:
                        nested_xobjects = _resolve_indirect(nested_xobjects)
                        nested = _validate_bpc_in_xobjects(nested_xobjects, visited)
                        result["invalid_bpc"] += nested["invalid_bpc"]
                        result["mask_bpc_invalid"] += nested["mask_bpc_invalid"]

        except Exception as e:
            logger.debug("Error validating BPC on XObject %s: %s", key, e)

    return result


def _validate_bpc_in_ap_stream(
    ap_entry,
    visited: set[tuple[int, int]],
) -> dict[str, int]:
    """Validate BitsPerComponent on images in an annotation AP stream entry.

    Args:
        ap_entry: An appearance entry (N, R, or D value).
        visited: Set of (objnum, gen) tuples for cycle detection.

    Returns:
        Dictionary with invalid_bpc and mask_bpc_invalid counts.
    """
    result = {"invalid_bpc": 0, "mask_bpc_invalid": 0}
    ap_entry = _resolve_indirect(ap_entry)

    if isinstance(ap_entry, Stream):
        resources = ap_entry.get("/Resources")
        if resources:
            resources = _resolve_indirect(resources)
            xobjects = resources.get("/XObject")
            if xobjects:
                xobjects = _resolve_indirect(xobjects)
                r = _validate_bpc_in_xobjects(xobjects, visited)
                result["invalid_bpc"] += r["invalid_bpc"]
                result["mask_bpc_invalid"] += r["mask_bpc_invalid"]
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                resources = state_stream.get("/Resources")
                if resources:
                    resources = _resolve_indirect(resources)
                    xobjects = resources.get("/XObject")
                    if xobjects:
                        xobjects = _resolve_indirect(xobjects)
                        r = _validate_bpc_in_xobjects(xobjects, visited)
                        result["invalid_bpc"] += r["invalid_bpc"]
                        result["mask_bpc_invalid"] += r["mask_bpc_invalid"]

    return result


def validate_bits_per_component(pdf: Pdf) -> dict[str, int]:
    """Validates BitsPerComponent on all Image XObjects.

    ISO 19005-2, Clause 6.2.8 requires:
    - BitsPerComponent must be 1, 2, 4, 8, or 16
    - Image masks (/ImageMask true) must have BitsPerComponent == 1

    Logs a warning for each invalid value found.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        Dictionary with keys:
        - invalid_bpc: Number of images with invalid BitsPerComponent
        - mask_bpc_invalid: Number of image masks with BPC != 1
    """
    total: dict[str, int] = {"invalid_bpc": 0, "mask_bpc_invalid": 0}
    visited: set[tuple[int, int]] = set()

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve_indirect(page.obj)

            # Page → Resources → XObject
            resources = page_dict.get("/Resources")
            if resources is not None:
                resources = _resolve_indirect(resources)
                xobjects = resources.get("/XObject")
                if xobjects is not None:
                    xobjects = _resolve_indirect(xobjects)
                    r = _validate_bpc_in_xobjects(xobjects, visited)
                    total["invalid_bpc"] += r["invalid_bpc"]
                    total["mask_bpc_invalid"] += r["mask_bpc_invalid"]

            # Page → Annots → AP streams
            annots = page_dict.get("/Annots")
            if annots:
                annots = _resolve_indirect(annots)
                for annot in annots:
                    annot = _resolve_indirect(annot)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = annot.get("/AP")
                    if not ap:
                        continue
                    ap = _resolve_indirect(ap)
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry is not None:
                            r = _validate_bpc_in_ap_stream(ap_entry, visited)
                            total["invalid_bpc"] += r["invalid_bpc"]
                            total["mask_bpc_invalid"] += r["mask_bpc_invalid"]

        except Exception as e:
            logger.debug("Error validating BPC on page %d: %s", page_num, e)

    warnings = total["invalid_bpc"] + total["mask_bpc_invalid"]
    if warnings > 0:
        logger.info(
            "BitsPerComponent validation: %d invalid BPC, %d invalid mask BPC",
            total["invalid_bpc"],
            total["mask_bpc_invalid"],
        )
    return total
