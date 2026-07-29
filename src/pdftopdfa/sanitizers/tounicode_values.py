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
import re

import pikepdf
from pikepdf import Pdf, Stream

from ..fonts.analysis import get_font_type
from ..fonts.tounicode import (
    _is_invalid_unicode,
    generate_cidfont_tounicode_cmap,
    generate_tounicode_cmap_data,
    parse_tounicode_cmap,
)
from ..fonts.traversal import iter_all_page_fonts
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Text-showing operators (PDF Reference, Table 5.6)
_TEXT_OPERATORS = frozenset({"Tj", "'", '"'})
_TJ_OPERATOR = "TJ"
_BFCHAR_BLOCK_PATTERN = re.compile(r"(beginbfchar\b)(.*?)(\bendbfchar)", re.DOTALL)
_BFCHAR_ENTRY_PATTERN = re.compile(r"(<[0-9A-Fa-f]+>\s*)<([0-9A-Fa-f]+)>")
_BFRANGE_BLOCK_PATTERN = re.compile(r"(beginbfrange\b)(.*?)(\bendbfrange)", re.DOTALL)
_BFRANGE_ENTRY_PATTERN = re.compile(
    r"(<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*)"
    r"(?:<([0-9A-Fa-f]+)>|\[([^\]]*)\])"
)
_HEX_TOKEN_PATTERN = re.compile(r"<([0-9A-Fa-f]+)>")


def _sanitize_tounicode_cmap(cmap_data: bytes) -> bytes:
    """Replace invalid destinations without rebuilding the CMap."""
    try:
        text = cmap_data.decode("ascii")
    except UnicodeDecodeError:
        return cmap_data

    parsed = parse_tounicode_cmap(cmap_data)
    used_pua = {value for value in parsed.values() if 0xE000 <= value <= 0xF8FF}
    next_pua = 0xE000

    def sanitize_destination(hex_value: str) -> str:
        nonlocal next_pua

        if len(hex_value) % 4 != 0:
            return hex_value

        units = [
            int(hex_value[index : index + 4], 16)
            for index in range(0, len(hex_value), 4)
        ]
        changed = False
        index = 0

        while index < len(units):
            value = units[index]
            if (
                0xD800 <= value <= 0xDBFF
                and index + 1 < len(units)
                and 0xDC00 <= units[index + 1] <= 0xDFFF
            ):
                index += 2
                continue

            if _is_invalid_unicode(value):
                while next_pua in used_pua and next_pua <= 0xF8FF:
                    next_pua += 1
                if next_pua <= 0xF8FF:
                    units[index] = next_pua
                    used_pua.add(next_pua)
                    next_pua += 1
                    changed = True

            index += 1

        if not changed:
            return hex_value
        return "".join(f"{value:04X}" for value in units)

    def sanitize_bfchar_block(match: re.Match[str]) -> str:
        def sanitize_entry(entry: re.Match[str]) -> str:
            destination = sanitize_destination(entry.group(2))
            return f"{entry.group(1)}<{destination}>"

        body = _BFCHAR_ENTRY_PATTERN.sub(sanitize_entry, match.group(2))
        return f"{match.group(1)}{body}{match.group(3)}"

    def sanitize_bfrange_block(match: re.Match[str]) -> str:
        def sanitize_entry(entry: re.Match[str]) -> str:
            destination = entry.group(4)
            if destination is not None:
                start = int(entry.group(2), 16)
                end = int(entry.group(3), 16)
                if end < start:
                    return entry.group(0)

                destination_start = int(destination, 16)
                destinations = [
                    f"{destination_start + offset:0{len(destination)}X}"
                    for offset in range(end - start + 1)
                ]
                sanitized = [sanitize_destination(value) for value in destinations]
                if sanitized == destinations:
                    return entry.group(0)
                values = " ".join(f"<{value}>" for value in sanitized)
                return f"{entry.group(1)}[{values}]"

            array_body = _HEX_TOKEN_PATTERN.sub(
                lambda token: f"<{sanitize_destination(token.group(1))}>",
                entry.group(5),
            )
            return f"{entry.group(1)}[{array_body}]"

        body = _BFRANGE_ENTRY_PATTERN.sub(sanitize_entry, match.group(2))
        return f"{match.group(1)}{body}{match.group(3)}"

    text = _BFCHAR_BLOCK_PATTERN.sub(sanitize_bfchar_block, text)
    text = _BFRANGE_BLOCK_PATTERN.sub(sanitize_bfrange_block, text)
    return text.encode("ascii")


def sanitize_tounicode_values(pdf: Pdf) -> dict[str, int]:
    """Replaces forbidden Unicode values in existing ToUnicode CMaps.

    Iterates all fonts and checks their ToUnicode streams for forbidden
    values. Affected destinations are replaced with PUA codepoints while
    valid mappings remain unchanged.

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

                new_cmap = _sanitize_tounicode_cmap(cmap_data)
                if new_cmap == cmap_data:
                    continue

                # Replace the ToUnicode stream
                new_stream = Stream(pdf, new_cmap)
                font_obj[pikepdf.Name.ToUnicode] = pdf.make_indirect(new_stream)

                total_fixed += 1
                logger.debug(
                    "Fixed invalid Unicode values in ToUnicode for font %s",
                    _font_key,
                )

            except Exception as e:
                log_suppressed_error(
                    logger,
                    e,
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
            log_suppressed_error(
                logger,
                e,
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
