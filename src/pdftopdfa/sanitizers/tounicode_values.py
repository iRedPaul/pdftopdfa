# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Sanitize invalid Unicode values in existing ToUnicode CMaps.

PDF/A (veraPDF rule 6.2.11.7.2) forbids U+0000, U+FEFF, U+FFFE,
and Unicode surrogate code points (U+D800–U+DFFF) in ToUnicode CMap
mappings. This sanitizer detects and replaces these values in
pre-existing ToUnicode streams with Private Use Area codepoints.

For PDF/A-2u, PDF/A-3u, PDF/A-2a, and PDF/A-3a, every glyph used in
content streams must be mappable to Unicode via the ToUnicode CMap.
The ``fill_tounicode_gaps`` function ensures complete coverage by adding
semantic mappings for known UTF-16/UCS-2 CMaps and PUA codepoints for
otherwise unresolvable character codes.
"""

import logging
import re
from collections.abc import Iterator

import pikepdf
from pikepdf import Pdf, Stream

from ..fonts.constants import UTF16_ENCODING_NAMES
from ..fonts.glyph_usage import (
    _iter_content_streams_with_resources,
    _resolve_font_object,
)
from ..fonts.tounicode import (
    _is_invalid_unicode,
    get_font_code_space_ranges,
    parse_tounicode_cmap_sequences,
    split_cmap_codes,
)
from ..fonts.utils import get_encoding_name
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

    parsed = parse_tounicode_cmap_sequences(cmap_data)
    used_pua = {
        value
        for values in parsed.values()
        for value in values
        if 0xE000 <= value <= 0xF8FF
    }
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

    for _font_key, font_obj in _iter_fonts_from_effective_resources(pdf):
        try:
            objgen = font_obj.objgen
            if objgen != (0, 0):
                if objgen in processed:
                    continue
                processed.add(objgen)

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


def _iter_fonts_from_effective_resources(
    pdf: Pdf,
) -> Iterator[tuple[str, pikepdf.Object]]:
    """Yield fonts from every resource context reached by content traversal."""
    for page in pdf.pages:
        for _owner, resources in _iter_content_streams_with_resources(page):
            resources = _resolve(resources)
            if not isinstance(resources, pikepdf.Dictionary):
                continue
            font_dict = _resolve(resources.get("/Font"))
            if not isinstance(font_dict, pikepdf.Dictionary):
                continue
            for font_key in list(font_dict.keys()):
                try:
                    font = _resolve(font_dict[font_key])
                except Exception:
                    continue
                if isinstance(font, pikepdf.Dictionary):
                    yield str(font_key), font


def fill_tounicode_gaps(pdf: Pdf) -> dict[str, int]:
    """Fills gaps in ToUnicode CMaps for character codes used in content streams.

    For PDF/A Unicode and level A compliance (veraPDF rule 6.2.11.7.2),
    every glyph used in a content stream must be mappable to Unicode. This
    function parses all content streams to discover which character codes are
    actually used per font, then checks the existing ToUnicode CMap for gaps.
    Codes from known UTF-16/UCS-2 CMaps retain their inherent Unicode
    semantics; otherwise unmapped codes receive Private Use Area (PUA)
    codepoints.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"tounicode_gaps_filled": N}``.
    """
    total_filled = 0
    font_used_codes, font_objs = _collect_used_codes(pdf)

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
            code_to_unicode = parse_tounicode_cmap_sequences(cmap_data)

            # Find codes used in content but missing from ToUnicode
            missing_codes = used_codes - set(code_to_unicode.keys())
            if not missing_codes:
                continue

            # Preserve the inherent Unicode semantics of UTF-16/UCS-2 CMaps.
            # Only codes whose meaning cannot be derived receive PUA values.
            existing_pua = {
                value
                for values in code_to_unicode.values()
                for value in values
                if 0xE000 <= value <= 0xF8FF
            }
            next_pua = 0xE000
            additions: dict[bytes, int] = {}
            for code in sorted(missing_codes):
                unicode_value = _unicode_cmap_identity(font_obj, code)
                if unicode_value is not None:
                    additions[code] = unicode_value
                    if 0xE000 <= unicode_value <= 0xF8FF:
                        existing_pua.add(unicode_value)
                    continue
                while next_pua in existing_pua and next_pua <= 0xF8FF:
                    next_pua += 1
                if next_pua <= 0xF8FF:
                    additions[code] = next_pua
                    existing_pua.add(next_pua)
                    next_pua += 1
                else:
                    # BMP PUA exhausted, skip remaining
                    break

            if not additions:
                continue
            new_cmap = _append_tounicode_mappings(cmap_data, additions)
            if new_cmap is None:
                continue

            new_stream = Stream(pdf, new_cmap)
            font_obj[pikepdf.Name.ToUnicode] = pdf.make_indirect(new_stream)

            total_filled += 1
            base_font = font_obj.get("/BaseFont")
            font_label = str(base_font) if base_font is not None else str(obj_key)
            pua_additions = sum(
                0xE000 <= value <= 0xF8FF for value in additions.values()
            )
            if pua_additions:
                logger.warning(
                    "PDF/A Unicode/level A: font %s has %d character codes mapped to "
                    "PUA codepoints (U+E000-U+F8FF) — text extraction will not "
                    "produce meaningful Unicode for these characters",
                    font_label,
                    pua_additions,
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
            "ToUnicode gaps: %d font(s) patched with ToUnicode mappings",
            total_filled,
        )

    return {"tounicode_gaps_filled": total_filled}


def _unicode_cmap_identity(font_obj: pikepdf.Object, code: bytes) -> int | None:
    """Decode a character code whose predefined CMap defines Unicode semantics."""
    encoding = _resolve(font_obj.get("/Encoding"))
    if encoding is None or get_encoding_name(encoding) not in UTF16_ENCODING_NAMES:
        return None
    try:
        text = code.decode("utf-16-be")
    except UnicodeDecodeError:
        return None
    if len(text) != 1:
        return None
    value = ord(text)
    return None if _is_invalid_unicode(value) else value


def _append_tounicode_mappings(
    cmap_data: bytes,
    additions: dict[bytes, int],
) -> bytes | None:
    """Insert new bfchar entries while preserving every existing mapping."""
    if not additions:
        return cmap_data
    matches = list(re.finditer(rb"\bendcmap\b", cmap_data))
    if not matches:
        return None

    blocks: list[bytes] = []
    ordered = sorted(additions.items())
    for offset in range(0, len(ordered), 100):
        chunk = ordered[offset : offset + 100]
        lines = [f"{len(chunk)} beginbfchar".encode("ascii")]
        lines.extend(
            (
                f"<{code.hex().upper()}> "
                f"<{chr(value).encode('utf-16-be').hex().upper()}>"
            ).encode("ascii")
            for code, value in chunk
        )
        lines.append(b"endbfchar")
        blocks.append(b"\n".join(lines))
    insertion = b"\n" + b"\n".join(blocks) + b"\n"
    position = matches[-1].start()
    return cmap_data[:position] + insertion + cmap_data[position:]


def _collect_used_codes(
    pdf: Pdf,
) -> tuple[
    dict[tuple[int, int], set[bytes]],
    dict[tuple[int, int], pikepdf.Object],
]:
    """Collect width-preserving character codes from every rendered text stream."""
    usage: dict[tuple[int, int], set[bytes]] = {}
    fonts: dict[tuple[int, int], pikepdf.Object] = {}
    ranges_by_font: dict[tuple[int, int], tuple[tuple[bytes, bytes], ...]] = {}
    next_direct_id = -1

    def font_key(font: pikepdf.Object) -> tuple[int, int]:
        nonlocal next_direct_id
        key = font.objgen
        if key != (0, 0):
            return key
        key = (next_direct_id, 0)
        next_direct_id -= 1
        return key

    def code_ranges(
        font: pikepdf.Object,
        key: tuple[int, int],
    ) -> tuple[tuple[bytes, bytes], ...]:
        cached = ranges_by_font.get(key)
        if cached is not None:
            return cached
        tounicode_data = None
        tounicode = _resolve(font.get("/ToUnicode"))
        if isinstance(tounicode, Stream):
            try:
                tounicode_data = bytes(tounicode.read_bytes())
            except Exception:
                pass
        ranges = get_font_code_space_ranges(font, tounicode_data)
        ranges_by_font[key] = ranges
        return ranges

    for page in pdf.pages:
        for owner, resources in _iter_content_streams_with_resources(page):
            try:
                instructions = pikepdf.parse_content_stream(owner)
            except Exception:
                continue

            current: (
                tuple[
                    pikepdf.Object,
                    tuple[int, int],
                    tuple[tuple[bytes, bytes], ...],
                ]
                | None
            ) = None
            stack: list[
                tuple[
                    pikepdf.Object,
                    tuple[int, int],
                    tuple[tuple[bytes, bytes], ...],
                ]
                | None
            ] = []
            for item in instructions:
                if isinstance(item, pikepdf.ContentStreamInlineImage):
                    continue
                operands, operator = item.operands, str(item.operator)
                if operator == "q":
                    stack.append(current)
                    continue
                if operator == "Q":
                    current = stack.pop() if stack else None
                    continue
                if operator == "Tf":
                    current = None
                    if operands:
                        font = _resolve_font_object(str(operands[0]), resources)
                        if font is not None:
                            key = font_key(font)
                            fonts.setdefault(key, font)
                            current = (font, key, code_ranges(font, key))
                    continue
                if current is None:
                    continue

                strings: list[pikepdf.String] = []
                if operator == _TJ_OPERATOR:
                    if operands and isinstance(operands[0], pikepdf.Array):
                        strings.extend(
                            value
                            for value in operands[0]
                            if isinstance(value, pikepdf.String)
                        )
                elif operator == '"':
                    if len(operands) >= 3 and isinstance(operands[2], pikepdf.String):
                        strings.append(operands[2])
                elif operator in {"Tj", "'"}:
                    if operands and isinstance(operands[0], pikepdf.String):
                        strings.append(operands[0])

                _font, key, ranges = current
                for string in strings:
                    codes = split_cmap_codes(bytes(string), ranges)
                    if codes:
                        usage.setdefault(key, set()).update(codes)

    return usage, fonts
