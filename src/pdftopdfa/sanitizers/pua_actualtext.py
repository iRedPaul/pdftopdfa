# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Wrap PUA-mapped characters in /ActualText marked-content sequences.

ISO 19005 Rule 6.2.11.7.3-1 requires that any character mapped to a
Unicode Private Use Area (PUA) code point be wrapped in a marked-content
sequence with /ActualText.  This module inserts the required BDC/EMC
wrappers around text-showing operators whose character codes resolve
to PUA via the font's ToUnicode CMap.
"""

import logging

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream, String

from ..fonts.subsetter import _resolve_simple_font_encoding
from ..fonts.tounicode import parse_tounicode_cmap, resolve_glyph_to_unicode
from ..utils import iter_type3_fonts as _iter_type3_fonts
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Text-showing operators whose string operands may reference PUA
_TEXT_OPERATORS = frozenset({"Tj", "'", '"'})


# ---------------------------------------------------------------------------
# PUA detection
# ---------------------------------------------------------------------------


def _is_pua(code_point: int) -> bool:
    """Check if a Unicode code point falls in any Private Use Area range.

    PUA ranges: U+E000..U+F8FF (BMP), U+F0000..U+FFFFD (Supplementary A),
    U+100000..U+10FFFD (Supplementary B).
    """
    return (
        0xE000 <= code_point <= 0xF8FF
        or 0xF0000 <= code_point <= 0xFFFFD
        or 0x100000 <= code_point <= 0x10FFFD
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sanitize_pua_actualtext(pdf: Pdf) -> dict[str, int]:
    """Wraps PUA-mapped characters in /ActualText marked-content sequences.

    Scans all content streams for text-showing operators whose character
    codes resolve to PUA Unicode values via the font's ToUnicode CMap,
    and wraps them in ``/Span <</ActualText ...>> BDC ... EMC``.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"pua_actualtext_added": N,
        "pua_actualtext_warnings": N}``.
    """
    total_added = 0
    total_warnings = 0
    visited: set[tuple[int, int]] = set()
    tounicode_cache: dict[tuple[int, int], dict[int, int]] = {}
    encoding_cache: dict[tuple[int, int], dict[int, str] | None] = {}

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve(page.obj)

            resources = page_dict.get("/Resources")
            if resources is not None:
                resources = _resolve(resources)
            font_map = _build_font_map(resources) if resources else {}

            # 1. Page Contents
            a, w = _fix_pua_in_page_contents(
                page_dict, font_map, tounicode_cache, encoding_cache
            )
            total_added += a
            total_warnings += w

            # 2. Form XObjects (recursive)
            if resources is not None:
                a, w = _fix_pua_in_form_xobjects(
                    resources, visited, tounicode_cache, encoding_cache
                )
                total_added += a
                total_warnings += w

                # 3. Tiling Patterns (recursive)
                a, w = _fix_pua_in_patterns(
                    resources, visited, tounicode_cache, encoding_cache
                )
                total_added += a
                total_warnings += w

                # 4. Type3 CharProcs
                a, w = _fix_pua_in_type3_charprocs(
                    resources, visited, tounicode_cache, encoding_cache
                )
                total_added += a
                total_warnings += w

            # 5. Annotation AP streams
            annots = page_dict.get("/Annots")
            if annots:
                annots = _resolve(annots)
                for annot_ref in annots:
                    annot = _resolve(annot_ref)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = annot.get("/AP")
                    if not ap:
                        continue
                    ap = _resolve(ap)
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry:
                            a, w = _fix_pua_in_ap_stream(
                                ap_entry,
                                visited,
                                tounicode_cache,
                                encoding_cache,
                            )
                            total_added += a
                            total_warnings += w

        except Exception as e:
            logger.debug(
                "Error fixing PUA ActualText on page %d: %s",
                page_num,
                e,
            )

    if total_added > 0:
        logger.info("PUA ActualText: %d text operators wrapped", total_added)
    if total_warnings > 0:
        logger.warning(
            "PUA ActualText: %d unresolvable PUA characters",
            total_warnings,
        )

    return {
        "pua_actualtext_added": total_added,
        "pua_actualtext_warnings": total_warnings,
    }


# ---------------------------------------------------------------------------
# Font map building
# ---------------------------------------------------------------------------


def _build_font_map(
    resources: pikepdf.Object,
) -> dict[str, pikepdf.Object]:
    """Builds a mapping from font resource name to font dictionary."""
    font_map: dict[str, pikepdf.Object] = {}
    resources = _resolve(resources)
    if not isinstance(resources, Dictionary):
        return font_map
    font_dict = resources.get("/Font")
    if font_dict is None:
        return font_map
    font_dict = _resolve(font_dict)
    if not isinstance(font_dict, Dictionary):
        return font_map
    for key in list(font_dict.keys()):
        try:
            font_obj = _resolve(font_dict[key])
            font_map[str(key)] = font_obj
        except Exception:
            continue
    return font_map


def _is_cidfont(font_obj: pikepdf.Object) -> bool:
    """Checks if a font is a CIDFont (Type0)."""
    try:
        subtype = font_obj.get("/Subtype")
        if subtype is not None and str(subtype) == "/Type0":
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# ToUnicode and encoding helpers
# ---------------------------------------------------------------------------


def _get_tounicode_map(
    font_obj: pikepdf.Object,
    cache: dict[tuple[int, int], dict[int, int]],
) -> dict[int, int]:
    """Returns the ToUnicode mapping for a font, cached by objgen."""
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        return {}

    objgen = font_obj.objgen
    if objgen != (0, 0) and objgen in cache:
        return cache[objgen]

    tounicode = font_obj.get("/ToUnicode")
    if tounicode is None:
        result: dict[int, int] = {}
    else:
        tounicode = _resolve(tounicode)
        try:
            data = bytes(tounicode.read_bytes())
            result = parse_tounicode_cmap(data)
        except Exception:
            result = {}

    if objgen != (0, 0):
        cache[objgen] = result
    return result


def _resolve_pua_to_actual_unicode(
    code: int,
    font_obj: pikepdf.Object,
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> int | None:
    """Try to find real Unicode for a PUA character code via encoding.

    For simple fonts, resolves code -> glyph name -> Unicode via AGL.
    For CIDFonts, returns None (no encoding-based resolution possible).
    """
    if _is_cidfont(font_obj):
        return None

    font_obj = _resolve(font_obj)
    objgen = font_obj.objgen

    if objgen != (0, 0) and objgen in encoding_cache:
        encoding = encoding_cache[objgen]
    else:
        encoding = _resolve_simple_font_encoding(font_obj)
        if objgen != (0, 0):
            encoding_cache[objgen] = encoding

    if encoding is None:
        return None

    glyph_name = encoding.get(code)
    if glyph_name is None:
        return None

    unicode_val = resolve_glyph_to_unicode(glyph_name)
    if unicode_val is not None and not _is_pua(unicode_val):
        return unicode_val
    return None


# ---------------------------------------------------------------------------
# ActualText construction
# ---------------------------------------------------------------------------


def _encode_actualtext(text: str) -> bytes:
    """Encode a string as UTF-16BE with BOM for PDF ActualText."""
    return b"\xfe\xff" + text.encode("utf-16-be")


def _has_pua_codes(raw: bytes, tounicode: dict[int, int], is_cid: bool) -> bool:
    """Check if any character codes in raw bytes map to PUA."""
    if is_cid:
        for i in range(0, len(raw) - 1, 2):
            code = (raw[i] << 8) | raw[i + 1]
            unicode_val = tounicode.get(code)
            if unicode_val is not None and _is_pua(unicode_val):
                return True
    else:
        for byte_val in raw:
            unicode_val = tounicode.get(byte_val)
            if unicode_val is not None and _is_pua(unicode_val):
                return True
    return False


def _extract_text_bytes(op_str: str, operands: list) -> bytes | None:
    """Extract raw text bytes from a text operator's operands."""
    if op_str == "TJ":
        if not operands or not isinstance(operands[0], Array):
            return None
        parts = bytearray()
        for elem in operands[0]:
            if isinstance(elem, String):
                parts.extend(bytes(elem))
        return bytes(parts) if parts else None
    elif op_str == '"':
        if len(operands) >= 3 and isinstance(operands[2], String):
            return bytes(operands[2])
        return None
    else:
        # Tj or '
        if operands and isinstance(operands[0], String):
            return bytes(operands[0])
        return None


def _build_actualtext_value(
    raw: bytes,
    tounicode: dict[int, int],
    is_cid: bool,
    font_obj: pikepdf.Object,
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> tuple[str, int]:
    """Build the ActualText string for a text operand.

    For each character code: if non-PUA Unicode, use it; if PUA, try
    to resolve via font encoding and AGL; if unresolvable, omit.

    Returns:
        Tuple of (actualtext_string, num_unresolvable_pua).
    """
    chars: list[str] = []
    warnings = 0

    if is_cid:
        for i in range(0, len(raw) - 1, 2):
            code = (raw[i] << 8) | raw[i + 1]
            unicode_val = tounicode.get(code)
            if unicode_val is None:
                continue
            if _is_pua(unicode_val):
                resolved = _resolve_pua_to_actual_unicode(
                    code, font_obj, encoding_cache
                )
                if resolved is not None:
                    chars.append(chr(resolved))
                else:
                    warnings += 1
            else:
                chars.append(chr(unicode_val))
    else:
        for byte_val in raw:
            unicode_val = tounicode.get(byte_val)
            if unicode_val is None:
                continue
            if _is_pua(unicode_val):
                resolved = _resolve_pua_to_actual_unicode(
                    byte_val, font_obj, encoding_cache
                )
                if resolved is not None:
                    chars.append(chr(resolved))
                else:
                    warnings += 1
            else:
                chars.append(chr(unicode_val))

    return "".join(chars), warnings


# ---------------------------------------------------------------------------
# Content stream fixing
# ---------------------------------------------------------------------------


def _fix_pua_in_stream(
    stream_obj: Stream,
    font_map: dict[str, pikepdf.Object],
    tounicode_cache: dict[tuple[int, int], dict[int, int]],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> tuple[int, int]:
    """Core stream processor.

    Parses a content stream, identifies text operators with PUA codes,
    and wraps them in BDC /Span <</ActualText ...>> ... EMC.

    Returns:
        Tuple of (wrapped_count, warning_count).
    """
    try:
        instructions = list(pikepdf.parse_content_stream(stream_obj))
    except Exception:
        return 0, 0

    # First pass: find text operator indices inside /ActualText BDC
    covered: set[int] = set()
    stack: list[bool] = []
    actualtext_depth = 0

    for idx, item in enumerate(instructions):
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            continue

        op_str = str(item.operator)

        if op_str == "BDC":
            has_actualtext = False
            if len(item.operands) >= 2:
                props = item.operands[1]
                if isinstance(props, Dictionary):
                    try:
                        if props.get("/ActualText") is not None:
                            has_actualtext = True
                    except Exception:
                        pass
            if has_actualtext:
                actualtext_depth += 1
            stack.append(has_actualtext)
        elif op_str == "BMC":
            stack.append(False)
        elif op_str == "EMC":
            if stack:
                if stack.pop():
                    actualtext_depth -= 1
        elif actualtext_depth > 0:
            if op_str in _TEXT_OPERATORS or op_str == "TJ":
                covered.add(idx)

    # Second pass: wrap PUA text operators
    new_instructions: list = []
    current_font_name: str | None = None
    wrapped_count = 0
    warning_count = 0

    for idx, item in enumerate(instructions):
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            new_instructions.append(item)
            continue

        operands, operator = item.operands, item.operator
        op_str = str(operator)

        # Track font changes
        if op_str == "Tf" and len(operands) >= 1:
            try:
                current_font_name = str(operands[0])
            except Exception:
                current_font_name = None
            new_instructions.append(item)
            continue

        # Skip non-text operators
        if op_str not in _TEXT_OPERATORS and op_str != "TJ":
            new_instructions.append(item)
            continue

        # Skip if already covered by existing /ActualText
        if idx in covered:
            new_instructions.append(item)
            continue

        # Skip if no font context
        if current_font_name is None:
            new_instructions.append(item)
            continue

        font_obj = font_map.get(current_font_name)
        if font_obj is None:
            new_instructions.append(item)
            continue

        tounicode = _get_tounicode_map(font_obj, tounicode_cache)
        if not tounicode:
            new_instructions.append(item)
            continue

        is_cid = _is_cidfont(font_obj)

        # Collect raw bytes from the text operand
        raw = _extract_text_bytes(op_str, operands)
        if raw is None:
            new_instructions.append(item)
            continue

        # Check if any codes map to PUA
        if not _has_pua_codes(raw, tounicode, is_cid):
            new_instructions.append(item)
            continue

        # Build ActualText
        text, warnings = _build_actualtext_value(
            raw, tounicode, is_cid, font_obj, encoding_cache
        )
        warning_count += warnings

        # Create BDC/EMC wrapper
        actualtext_bytes = _encode_actualtext(text)
        props = Dictionary()
        props[Name("/ActualText")] = String(actualtext_bytes)
        bdc = pikepdf.ContentStreamInstruction(
            [Name("/Span"), props], pikepdf.Operator("BDC")
        )
        emc = pikepdf.ContentStreamInstruction([], pikepdf.Operator("EMC"))

        new_instructions.append(bdc)
        new_instructions.append(item)
        new_instructions.append(emc)
        wrapped_count += 1

    if wrapped_count > 0:
        stream_obj.write(pikepdf.unparse_content_stream(new_instructions))

    return wrapped_count, warning_count


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------


def _fix_pua_in_page_contents(
    page_dict: Dictionary,
    font_map: dict[str, pikepdf.Object],
    tounicode_cache: dict[tuple[int, int], dict[int, int]],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> tuple[int, int]:
    """Fixes PUA references in page Contents."""
    contents = page_dict.get("/Contents")
    if contents is None:
        return 0, 0

    contents = _resolve(contents)
    total_added = 0
    total_warnings = 0

    if isinstance(contents, Stream):
        a, w = _fix_pua_in_stream(contents, font_map, tounicode_cache, encoding_cache)
        total_added += a
        total_warnings += w
    elif isinstance(contents, Array):
        for item in contents:
            item = _resolve(item)
            if isinstance(item, Stream):
                a, w = _fix_pua_in_stream(
                    item, font_map, tounicode_cache, encoding_cache
                )
                total_added += a
                total_warnings += w

    return total_added, total_warnings


def _fix_pua_in_form_xobjects(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
    tounicode_cache: dict[tuple[int, int], dict[int, int]],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> tuple[int, int]:
    """Recurses into Form XObjects to fix PUA references."""
    total_added = 0
    total_warnings = 0
    resources = _resolve(resources)
    if not isinstance(resources, Dictionary):
        return 0, 0

    xobjects = resources.get("/XObject")
    if not xobjects:
        return 0, 0
    xobjects = _resolve(xobjects)
    if not isinstance(xobjects, Dictionary):
        return 0, 0

    for xobj_name in list(xobjects.keys()):
        xobj = _resolve(xobjects[xobj_name])
        if not isinstance(xobj, Stream):
            continue

        subtype = xobj.get("/Subtype")
        if subtype is None or str(subtype) != "/Form":
            continue

        objgen = xobj.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        # Build font map from Form XObject's own resources
        form_resources = xobj.get("/Resources")
        if form_resources:
            form_resources = _resolve(form_resources)
            form_font_map = _build_font_map(form_resources)
        else:
            form_font_map = {}

        a, w = _fix_pua_in_stream(xobj, form_font_map, tounicode_cache, encoding_cache)
        total_added += a
        total_warnings += w

        # Recurse into nested Form XObjects and Patterns
        if form_resources:
            a, w = _fix_pua_in_form_xobjects(
                form_resources, visited, tounicode_cache, encoding_cache
            )
            total_added += a
            total_warnings += w
            a, w = _fix_pua_in_patterns(
                form_resources, visited, tounicode_cache, encoding_cache
            )
            total_added += a
            total_warnings += w

    return total_added, total_warnings


def _fix_pua_in_patterns(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
    tounicode_cache: dict[tuple[int, int], dict[int, int]],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> tuple[int, int]:
    """Recurses into Tiling Patterns to fix PUA references."""
    total_added = 0
    total_warnings = 0
    resources = _resolve(resources)
    if not isinstance(resources, Dictionary):
        return 0, 0

    patterns = resources.get("/Pattern")
    if not patterns:
        return 0, 0
    patterns = _resolve(patterns)
    if not isinstance(patterns, Dictionary):
        return 0, 0

    for pat_name in list(patterns.keys()):
        try:
            pattern = _resolve(patterns[pat_name])
            if not isinstance(pattern, Stream):
                continue

            # Only process Tiling Patterns (PatternType 1)
            pattern_type = pattern.get("/PatternType")
            if pattern_type is None or int(pattern_type) != 1:
                continue

            objgen = pattern.objgen
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            # Build font map from pattern's own resources
            pat_resources = pattern.get("/Resources")
            if pat_resources:
                pat_resources = _resolve(pat_resources)
                pat_font_map = _build_font_map(pat_resources)
            else:
                pat_font_map = {}

            a, w = _fix_pua_in_stream(
                pattern,
                pat_font_map,
                tounicode_cache,
                encoding_cache,
            )
            total_added += a
            total_warnings += w

            # Recurse into nested Form XObjects and Patterns
            if pat_resources:
                a, w = _fix_pua_in_form_xobjects(
                    pat_resources,
                    visited,
                    tounicode_cache,
                    encoding_cache,
                )
                total_added += a
                total_warnings += w
                a, w = _fix_pua_in_patterns(
                    pat_resources,
                    visited,
                    tounicode_cache,
                    encoding_cache,
                )
                total_added += a
                total_warnings += w
        except Exception:
            continue

    return total_added, total_warnings


def _fix_pua_in_ap_stream(
    ap_entry: pikepdf.Object,
    visited: set[tuple[int, int]],
    tounicode_cache: dict[tuple[int, int], dict[int, int]],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> tuple[int, int]:
    """Fixes PUA references in an annotation appearance stream entry."""
    total_added = 0
    total_warnings = 0
    ap_entry = _resolve(ap_entry)

    if isinstance(ap_entry, Stream):
        ap_resources = ap_entry.get("/Resources")
        ap_font_map = _build_font_map(ap_resources) if ap_resources else {}
        a, w = _fix_pua_in_stream(
            ap_entry, ap_font_map, tounicode_cache, encoding_cache
        )
        total_added += a
        total_warnings += w
        if ap_resources:
            ap_resources = _resolve(ap_resources)
            a, w = _fix_pua_in_form_xobjects(
                ap_resources, visited, tounicode_cache, encoding_cache
            )
            total_added += a
            total_warnings += w
            a, w = _fix_pua_in_patterns(
                ap_resources, visited, tounicode_cache, encoding_cache
            )
            total_added += a
            total_warnings += w
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                st_resources = state_stream.get("/Resources")
                st_font_map = _build_font_map(st_resources) if st_resources else {}
                a, w = _fix_pua_in_stream(
                    state_stream,
                    st_font_map,
                    tounicode_cache,
                    encoding_cache,
                )
                total_added += a
                total_warnings += w
                if st_resources:
                    st_resources = _resolve(st_resources)
                    a, w = _fix_pua_in_form_xobjects(
                        st_resources,
                        visited,
                        tounicode_cache,
                        encoding_cache,
                    )
                    total_added += a
                    total_warnings += w
                    a, w = _fix_pua_in_patterns(
                        st_resources,
                        visited,
                        tounicode_cache,
                        encoding_cache,
                    )
                    total_added += a
                    total_warnings += w

    return total_added, total_warnings


def _fix_pua_in_type3_charprocs(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
    tounicode_cache: dict[tuple[int, int], dict[int, int]],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
) -> tuple[int, int]:
    """Fixes PUA references in Type3 font CharProcs."""
    total_added = 0
    total_warnings = 0

    for _font_name, font in _iter_type3_fonts(resources, visited):
        charprocs = font.get("/CharProcs")
        if charprocs is None:
            continue
        charprocs = _resolve(charprocs)
        if not isinstance(charprocs, Dictionary):
            continue

        font_resources = font.get("/Resources")
        cp_font_map = _build_font_map(font_resources) if font_resources else {}

        for cp_name in list(charprocs.keys()):
            cp_stream = _resolve(charprocs[cp_name])
            if isinstance(cp_stream, Stream):
                a, w = _fix_pua_in_stream(
                    cp_stream,
                    cp_font_map,
                    tounicode_cache,
                    encoding_cache,
                )
                total_added += a
                total_warnings += w

    return total_added, total_warnings
