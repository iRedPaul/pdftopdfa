# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Glyph usage collection from PDF content streams.

Parses all content streams (pages, Form XObjects, Tiling Patterns,
Annotation Appearance Streams) and collects character codes used
with each font. This is needed for font subsetting — only glyphs
that are actually used need to be kept in the font program.
"""

import logging
from collections.abc import Iterator

import pikepdf

from ..utils import resolve_indirect as _resolve_indirect
from .utils import check_visited as _check_visited

logger = logging.getLogger(__name__)

# Text-showing operators that contain character strings
_TEXT_OPERATORS = frozenset(
    {
        pikepdf.Operator("Tj"),
        pikepdf.Operator("TJ"),
        pikepdf.Operator("'"),
        pikepdf.Operator('"'),
    }
)

_TF_OPERATOR = pikepdf.Operator("Tf")


def _is_cidfont(font_obj: pikepdf.Object) -> bool:
    """Checks if a font is a CIDFont (Type0).

    Args:
        font_obj: pikepdf font object.

    Returns:
        True if the font is Type0 (CIDFont).
    """
    try:
        subtype = font_obj.get("/Subtype")
        if subtype is not None and str(subtype) == "/Type0":
            return True
    except Exception:
        pass
    return False


def _extract_char_codes(
    string_operand: pikepdf.Object,
    is_cid: bool,
) -> set[int]:
    """Extracts character codes from a text string operand.

    For simple fonts, each byte is one character code (0-255).
    For CIDFonts with Identity-H/V encoding, each pair of bytes
    (big-endian) forms one character code (0-65535).

    Args:
        string_operand: pikepdf String object from a text operator.
        is_cid: True if the current font is a CIDFont.

    Returns:
        Set of character codes found in the string.
    """
    codes: set[int] = set()
    try:
        raw = bytes(string_operand)
    except Exception:
        return codes

    if is_cid:
        # 2-byte big-endian encoding
        if len(raw) % 2 != 0:
            logger.warning(
                "Odd-length CID string (%d bytes); trailing byte dropped",
                len(raw),
            )
        for i in range(0, len(raw) - 1, 2):
            code = (raw[i] << 8) | raw[i + 1]
            codes.add(code)
    else:
        # 1-byte encoding
        for b in raw:
            codes.add(b)

    return codes


def _iter_content_streams_with_resources(
    page: pikepdf.Page,
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object]]:
    """Yields (content_stream_owner, resources) for all nested structures on a page.

    Traverses page-level content, Form XObjects, Tiling Patterns,
    and Annotation Appearance Streams recursively.

    Args:
        page: A pikepdf Page object.

    Yields:
        Tuples of (stream_owner, resources_dict).
    """
    visited: set[tuple[int, int]] = set()

    # Page-level
    resources = page.get("/Resources")
    if resources is not None:
        try:
            resources = _resolve_indirect(resources)
        except Exception:
            resources = None

    if resources is not None:
        yield (page.obj, resources)
        yield from _iter_nested_streams(resources, visited)

    # Annotation Appearance Streams
    annots = page.get("/Annots")
    if annots is None:
        return

    try:
        annots = _resolve_indirect(annots)
    except Exception:
        return

    for annot_ref in annots:
        try:
            annot = _resolve_indirect(annot_ref)
            ap = annot.get("/AP")
            if ap is None:
                continue
            ap = _resolve_indirect(ap)

            for ap_key in ("/N", "/R", "/D"):
                ap_entry = ap.get(ap_key)
                if ap_entry is None:
                    continue

                try:
                    ap_entry = _resolve_indirect(ap_entry)
                except Exception:
                    continue

                if isinstance(ap_entry, pikepdf.Stream):
                    _yield_from_form_xobject(ap_entry, visited)
                    res = ap_entry.get("/Resources")
                    if res is not None:
                        res = _resolve_indirect(res)
                        yield (ap_entry, res)
                        yield from _iter_nested_streams(res, visited)
                elif isinstance(ap_entry, pikepdf.Dictionary):
                    for sub_key in list(ap_entry.keys()):
                        try:
                            sub = _resolve_indirect(ap_entry[sub_key])
                            if isinstance(sub, pikepdf.Stream):
                                if not _check_visited(sub, visited):
                                    res = sub.get("/Resources")
                                    if res is not None:
                                        res = _resolve_indirect(res)
                                        yield (sub, res)
                                        yield from _iter_nested_streams(res, visited)
                        except Exception:
                            continue
        except Exception:
            continue


def _yield_from_form_xobject(
    xobj: pikepdf.Object,
    visited: set[tuple[int, int]],
) -> None:
    """Mark a form XObject as visited (helper for AP streams)."""
    _check_visited(xobj, visited)


def _iter_nested_streams(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object]]:
    """Yields (stream_owner, resources) from nested XObjects and Patterns.

    Args:
        resources: A PDF Resources dictionary.
        visited: Set for cycle detection.

    Yields:
        Tuples of (stream_owner, resources_dict).
    """
    # Form XObjects
    xobject_dict = resources.get("/XObject")
    if xobject_dict is not None:
        try:
            xobject_dict = _resolve_indirect(xobject_dict)
        except Exception:
            xobject_dict = None

    if xobject_dict is not None:
        for xobj_key in list(xobject_dict.keys()):
            try:
                xobj = _resolve_indirect(xobject_dict[xobj_key])
                if _check_visited(xobj, visited):
                    continue
                subtype = xobj.get("/Subtype")
                if subtype is not None and str(subtype) == "/Form":
                    nested_res = xobj.get("/Resources")
                    if nested_res is not None:
                        nested_res = _resolve_indirect(nested_res)
                        yield (xobj, nested_res)
                        yield from _iter_nested_streams(nested_res, visited)
            except Exception:
                continue

    # Tiling Patterns
    pattern_dict = resources.get("/Pattern")
    if pattern_dict is not None:
        try:
            pattern_dict = _resolve_indirect(pattern_dict)
        except Exception:
            pattern_dict = None

    if pattern_dict is not None:
        for pat_key in list(pattern_dict.keys()):
            try:
                pattern = _resolve_indirect(pattern_dict[pat_key])
                if _check_visited(pattern, visited):
                    continue
                pattern_type = pattern.get("/PatternType")
                if pattern_type is not None and int(pattern_type) == 1:
                    nested_res = pattern.get("/Resources")
                    if nested_res is not None:
                        nested_res = _resolve_indirect(nested_res)
                        yield (pattern, nested_res)
                        yield from _iter_nested_streams(nested_res, visited)
            except Exception:
                continue


def _resolve_font_object(
    font_name_in_stream: str,
    resources: pikepdf.Object,
) -> pikepdf.Object | None:
    """Resolves a font name from a content stream to its font object.

    Args:
        font_name_in_stream: Font name as used in Tf operator (e.g. "/F1").
        resources: Resources dictionary containing the Font sub-dictionary.

    Returns:
        The resolved font object, or None if not found.
    """
    font_dict = resources.get("/Font")
    if font_dict is None:
        return None

    try:
        font_dict = _resolve_indirect(font_dict)
    except Exception:
        return None

    # The font name in the stream includes the leading "/" — use it as a key
    font_ref = font_dict.get(font_name_in_stream)
    if font_ref is None:
        return None

    try:
        return _resolve_indirect(font_ref)
    except Exception:
        return None


def collect_font_usage(
    pdf: pikepdf.Pdf,
) -> dict[tuple[int, int], set[int]]:
    """Collects character codes used with each font across the entire PDF.

    Iterates all pages and their nested structures (Form XObjects,
    Tiling Patterns, Annotation APs), parses content streams, and
    records which character codes are used with each font.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        Dictionary mapping font objgen (object_number, generation)
        to the set of character codes used with that font.
        Only fonts with objgen != (0,0) are included.
    """
    usage: dict[tuple[int, int], set[int]] = {}

    for page in pdf.pages:
        for stream_owner, resources in _iter_content_streams_with_resources(page):
            _process_content_stream(stream_owner, resources, usage)

    return usage


def _process_content_stream(
    stream_owner: pikepdf.Object,
    resources: pikepdf.Object,
    usage: dict[tuple[int, int], set[int]],
) -> None:
    """Parses a content stream and records character code usage.

    Args:
        stream_owner: Object that owns the content stream (page or XObject).
        resources: Resources dictionary for font resolution.
        usage: Accumulator mapping font objgen -> used character codes.
    """
    try:
        instructions = pikepdf.parse_content_stream(stream_owner)
    except Exception:
        return

    current_font: pikepdf.Object | None = None
    current_font_is_cid = False

    for operands, operator in instructions:
        if operator == _TF_OPERATOR:
            # Tf: set current font
            if operands:
                font_name = str(operands[0])
                font_obj = _resolve_font_object(font_name, resources)
                if font_obj is not None:
                    current_font = font_obj
                    current_font_is_cid = _is_cidfont(font_obj)
                else:
                    current_font = None
                    current_font_is_cid = False

        elif operator in _TEXT_OPERATORS and current_font is not None:
            # Get objgen for the current font
            try:
                objgen = current_font.objgen
            except Exception:
                continue

            if objgen == (0, 0):
                continue

            if operator == pikepdf.Operator("TJ"):
                # TJ takes an array of strings and numbers
                if operands and isinstance(operands[0], pikepdf.Array):
                    for item in operands[0]:
                        if isinstance(item, pikepdf.String):
                            codes = _extract_char_codes(item, current_font_is_cid)
                            if codes:
                                if objgen not in usage:
                                    usage[objgen] = set()
                                usage[objgen].update(codes)
            elif operator == pikepdf.Operator('"'):
                # " takes: aw ac string
                if len(operands) >= 3:
                    codes = _extract_char_codes(operands[2], current_font_is_cid)
                    if codes:
                        if objgen not in usage:
                            usage[objgen] = set()
                        usage[objgen].update(codes)
            else:
                # Tj and ' take a single string
                if operands:
                    codes = _extract_char_codes(operands[0], current_font_is_cid)
                    if codes:
                        if objgen not in usage:
                            usage[objgen] = set()
                        usage[objgen].update(codes)
