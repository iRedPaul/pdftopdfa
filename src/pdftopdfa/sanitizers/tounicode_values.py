# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Sanitize invalid Unicode values in existing ToUnicode CMaps.

PDF/A (veraPDF rule 6.2.11.7.2) forbids U+0000, U+FEFF, U+FFFE,
and Unicode surrogate code points (U+D800–U+DFFF) in ToUnicode CMap
mappings. This sanitizer detects and replaces these values in
pre-existing ToUnicode streams with Private Use Area codepoints.

For PDF/A-2u and PDF/A-3u (Unicode levels), every glyph used in content
streams must be mappable to Unicode via the ToUnicode CMap. The
``fill_tounicode_gaps`` function ensures complete coverage by adding
PUA codepoints for any character codes that appear in content streams
but are missing from the font's ToUnicode CMap.
"""

import logging

import pikepdf
from pikepdf import Pdf, Stream

from ..fonts.analysis import get_font_type
from ..fonts.tounicode import (
    _is_invalid_unicode,
    filter_invalid_unicode_values,
    generate_cidfont_tounicode_cmap,
    generate_tounicode_cmap_data,
    parse_tounicode_cmap,
)
from ..fonts.traversal import iter_all_page_fonts
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Text-showing operators (PDF Reference, Table 5.6)
_TEXT_OPERATORS = frozenset({"Tj", "'", '"'})
_TJ_OPERATOR = "TJ"


def sanitize_tounicode_values(pdf: Pdf) -> dict[str, int]:
    """Replaces forbidden Unicode values in existing ToUnicode CMaps.

    Iterates all fonts and checks their ToUnicode streams for U+0000,
    U+FEFF, or U+FFFE. When found, regenerates the CMap with PUA
    replacement codepoints.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"tounicode_values_fixed": N}``.
    """
    total_fixed = 0
    processed: set[tuple[int, int]] = set()

    for page in pdf.pages:
        for _font_key, font_obj in iter_all_page_fonts(page):
            try:
                obj_key = font_obj.objgen
                if obj_key != (0, 0):
                    if obj_key in processed:
                        continue
                    processed.add(obj_key)

                tounicode = font_obj.get("/ToUnicode")
                if tounicode is None:
                    continue

                tounicode = _resolve(tounicode)
                if not isinstance(tounicode, Stream):
                    continue

                # Parse existing CMap
                try:
                    cmap_data = bytes(tounicode.read_bytes())
                except Exception:
                    continue

                code_to_unicode = parse_tounicode_cmap(cmap_data)
                if not code_to_unicode:
                    continue

                # Check for invalid values (including surrogates)
                has_invalid = any(
                    _is_invalid_unicode(v) for v in code_to_unicode.values()
                )
                if not has_invalid:
                    continue

                # Replace invalid values
                fixed = filter_invalid_unicode_values(code_to_unicode)

                # Regenerate CMap based on font type
                font_type = get_font_type(font_obj)
                if font_type == "CIDFont":
                    new_cmap = generate_cidfont_tounicode_cmap(fixed)
                else:
                    new_cmap = generate_tounicode_cmap_data(fixed)

                # Replace the ToUnicode stream
                new_stream = Stream(pdf, new_cmap)
                font_obj[pikepdf.Name.ToUnicode] = pdf.make_indirect(new_stream)

                total_fixed += 1
                logger.debug(
                    "Fixed invalid Unicode values in ToUnicode for font %s",
                    _font_key,
                )

            except Exception as e:
                logger.debug(
                    "Error fixing ToUnicode values for font %s: %s",
                    _font_key,
                    e,
                )
                continue

    if total_fixed > 0:
        logger.info("ToUnicode values: %d font(s) fixed", total_fixed)

    return {"tounicode_values_fixed": total_fixed}


def fill_tounicode_gaps(pdf: Pdf) -> dict[str, int]:
    """Fills gaps in ToUnicode CMaps for character codes used in content streams.

    For PDF/A-2u and PDF/A-3u compliance (veraPDF rule 6.2.11.7.2), every
    glyph used in a content stream must be mappable to Unicode. This function
    parses all content streams to discover which character codes are actually
    used per font, then checks the existing ToUnicode CMap for gaps. Any
    unmapped codes are assigned Private Use Area (PUA) codepoints.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"tounicode_gaps_filled": N}``.
    """
    total_filled = 0

    # Phase 1: Collect used character codes per font (by objgen)
    # Indirect fonts use objgen as key; direct fonts (rare) use
    # a stable key derived from BaseFont + FirstChar + LastChar so
    # the same direct font on different pages is consolidated.
    font_used_codes: dict[tuple[int, int], set[int]] = {}
    font_objs: dict[tuple[int, int], pikepdf.Object] = {}
    direct_font_keys: dict[str, tuple[int, int]] = {}
    _next_direct_id = -1

    for page in pdf.pages:
        page_font_map = _build_page_font_map(page)
        if not page_font_map:
            continue

        used = _extract_used_codes_from_page(page, page_font_map)

        for font_key, codes in used.items():
            font_obj = page_font_map[font_key]
            obj_key = font_obj.objgen
            if obj_key == (0, 0):
                # Direct object — derive a stable key from font properties
                bf = font_obj.get("/BaseFont")
                fc = font_obj.get("/FirstChar")
                lc = font_obj.get("/LastChar")
                stable_key = f"{bf}:{fc}:{lc}"
                if stable_key in direct_font_keys:
                    obj_key = direct_font_keys[stable_key]
                else:
                    obj_key = (_next_direct_id, 0)
                    _next_direct_id -= 1
                    direct_font_keys[stable_key] = obj_key
            if obj_key not in font_used_codes:
                font_used_codes[obj_key] = set()
                font_objs[obj_key] = font_obj
            font_used_codes[obj_key].update(codes)

    # Phase 2: For each font with used codes, check ToUnicode gaps
    for obj_key, used_codes in font_used_codes.items():
        font_obj = font_objs[obj_key]

        try:
            tounicode = font_obj.get("/ToUnicode")
            if tounicode is None:
                continue

            tounicode = _resolve(tounicode)
            if not isinstance(tounicode, Stream):
                continue

            cmap_data = bytes(tounicode.read_bytes())
            code_to_unicode = parse_tounicode_cmap(cmap_data)

            # Find codes used in content but missing from ToUnicode
            missing_codes = used_codes - set(code_to_unicode.keys())
            if not missing_codes:
                continue

            # Assign PUA codepoints to missing codes
            existing_pua = {
                v for v in code_to_unicode.values() if 0xE000 <= v <= 0xF8FF
            }
            next_pua = 0xE000
            for code in sorted(missing_codes):
                while next_pua in existing_pua and next_pua <= 0xF8FF:
                    next_pua += 1
                if next_pua <= 0xF8FF:
                    code_to_unicode[code] = next_pua
                    existing_pua.add(next_pua)
                    next_pua += 1
                else:
                    # BMP PUA exhausted, skip remaining
                    break

            # Regenerate CMap based on font type
            font_type = get_font_type(font_obj)
            if font_type == "CIDFont":
                new_cmap = generate_cidfont_tounicode_cmap(code_to_unicode)
            else:
                new_cmap = generate_tounicode_cmap_data(code_to_unicode)

            new_stream = Stream(pdf, new_cmap)
            font_obj[pikepdf.Name.ToUnicode] = pdf.make_indirect(new_stream)

            total_filled += 1
            base_font = font_obj.get("/BaseFont")
            font_label = str(base_font) if base_font is not None else str(obj_key)
            logger.warning(
                "PDF/A 'u' level: font %s has %d character codes mapped to "
                "PUA codepoints (U+E000-U+F8FF) — text extraction will not "
                "produce meaningful Unicode for these characters",
                font_label,
                len(missing_codes),
            )

        except Exception as e:
            logger.debug(
                "Error filling ToUnicode gaps for font objgen %s: %s",
                obj_key,
                e,
            )
            continue

    if total_filled > 0:
        logger.info(
            "ToUnicode gaps: %d font(s) patched with PUA mappings",
            total_filled,
        )

    return {"tounicode_gaps_filled": total_filled}


def _build_page_font_map(
    page: pikepdf.Object,
) -> dict[str, pikepdf.Object]:
    """Builds a mapping from font resource keys to font objects for a page.

    Args:
        page: A pikepdf Page object.

    Returns:
        Dictionary mapping font keys (e.g. "/F0") to resolved font objects.
    """
    result: dict[str, pikepdf.Object] = {}

    for font_key, font_obj in iter_all_page_fonts(page):
        result[font_key] = font_obj

    return result


def _extract_used_codes_from_page(
    page: pikepdf.Object,
    font_map: dict[str, pikepdf.Object],
) -> dict[str, set[int]]:
    """Extracts used character codes from a page's content stream.

    Parses the content stream to track the current font (via Tf) and
    extract character codes from text-showing operators (Tj, TJ, ', ").

    Args:
        page: A pikepdf Page object.
        font_map: Mapping from font keys to font objects.

    Returns:
        Dictionary mapping font keys to sets of used character codes.
    """
    used: dict[str, set[int]] = {}

    try:
        instructions = pikepdf.parse_content_stream(page)
    except Exception:
        return used

    current_font_key: str | None = None

    for item in instructions:
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            continue

        operands, operator = item.operands, item.operator
        op_str = str(operator)

        # Track font changes via Tf operator
        if op_str == "Tf" and len(operands) >= 1:
            try:
                current_font_key = str(operands[0])
            except Exception:
                current_font_key = None
            continue

        if current_font_key is None or current_font_key not in font_map:
            continue

        font_obj = font_map[current_font_key]
        is_cidfont = get_font_type(font_obj) == "CIDFont"

        # Single-string text operators: Tj, ', "
        if op_str in _TEXT_OPERATORS:
            for operand in operands:
                if isinstance(operand, pikepdf.String):
                    codes = _extract_codes_from_string(bytes(operand), is_cidfont)
                    if codes:
                        if current_font_key not in used:
                            used[current_font_key] = set()
                        used[current_font_key].update(codes)

        # TJ operator: array of strings and adjustments
        elif op_str == _TJ_OPERATOR:
            for operand in operands:
                if isinstance(operand, pikepdf.Array):
                    for item_val in operand:
                        if isinstance(item_val, pikepdf.String):
                            codes = _extract_codes_from_string(
                                bytes(item_val), is_cidfont
                            )
                            if codes:
                                if current_font_key not in used:
                                    used[current_font_key] = set()
                                used[current_font_key].update(codes)

    return used


def _extract_codes_from_string(raw: bytes, is_cidfont: bool) -> list[int]:
    """Extracts character codes from a raw PDF string.

    For CIDFonts (Type0), codes are 2-byte big-endian values.
    For simple fonts (Type1, TrueType, Type3), codes are 1-byte values.

    Args:
        raw: Raw bytes from a pikepdf.String.
        is_cidfont: True if the font is a CIDFont (2-byte codes).

    Returns:
        List of character codes.
    """
    codes: list[int] = []
    if is_cidfont:
        for i in range(0, len(raw) - 1, 2):
            code = (raw[i] << 8) | raw[i + 1]
            codes.append(code)
    else:
        for b in raw:
            codes.append(b)
    return codes
