# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ToUnicode CMap generation for PDF/A compliance."""

import logging
import re

import pikepdf
from fontTools.agl import AGL2UV

from ..utils import resolve_indirect as _resolve_indirect
from .encodings import STANDARD_ENCODING, SYMBOL_ENCODING, ZAPFDINGBATS_ENCODING
from .glyph_mapping import SYMBOL_GLYPH_TO_UNICODE, ZAPFDINGBATS_GLYPH_TO_UNICODE
from .utils import safe_str as _safe_str

logger = logging.getLogger(__name__)

# Unicode values forbidden in PDF/A ToUnicode CMaps (veraPDF rule 6.2.11.7.2)
INVALID_UNICODE_VALUES = frozenset({0x0000, 0xFEFF, 0xFFFE})

# Unicode surrogate code points (U+D800–U+DFFF) — not valid Unicode scalar
# values. ISO 19005-2 §6.2.11.7.2 forbids these in ToUnicode mappings.
_SURROGATE_RANGE = range(0xD800, 0xE000)


def _is_invalid_unicode(val: int) -> bool:
    """Return True if a Unicode value is forbidden in PDF/A ToUnicode CMaps."""
    return val in INVALID_UNICODE_VALUES or val in _SURROGATE_RANGE


def filter_invalid_unicode_values(
    code_to_unicode: dict[int, int],
) -> dict[int, int]:
    """Replaces forbidden Unicode values with Private Use Area codepoints.

    PDF/A (veraPDF rule 6.2.11.7.2) forbids U+0000, U+FEFF, and U+FFFE
    in ToUnicode mappings. This replaces them with PUA codepoints (U+E000+)
    while avoiding collisions with existing PUA values.

    Args:
        code_to_unicode: Mapping from character codes to Unicode codepoints.

    Returns:
        New mapping with invalid values replaced by PUA codepoints.
    """
    if not any(_is_invalid_unicode(v) for v in code_to_unicode.values()):
        return code_to_unicode

    # Collect existing PUA values to avoid collisions
    existing_pua = {v for v in code_to_unicode.values() if 0xE000 <= v <= 0xF8FF}
    next_pua = 0xE000
    result = {}

    for code, unicode_val in code_to_unicode.items():
        if _is_invalid_unicode(unicode_val):
            # Find next available PUA codepoint
            while next_pua in existing_pua and next_pua <= 0xF8FF:
                next_pua += 1
            if next_pua <= 0xF8FF:
                result[code] = next_pua
                existing_pua.add(next_pua)
                next_pua += 1
            # else: PUA exhausted, skip this entry
        else:
            result[code] = unicode_val

    return result


def fill_tounicode_gaps_with_pua(
    code_to_unicode: dict[int, int],
    first_char: int = 0,
    last_char: int = 255,
) -> dict[int, int]:
    """Fills gaps in a code-to-Unicode mapping with PUA codepoints.

    For every character code in [first_char, last_char] that has no mapping,
    assigns a Private Use Area codepoint (U+E000-U+F8FF). This ensures
    complete ToUnicode coverage for symbolic fonts where the encoding is
    unknown but PDF/A requires every code to be mapped.

    Existing mappings are preserved. PUA assignments avoid collisions with
    any PUA values already present in the mapping.

    Args:
        code_to_unicode: Existing code-to-Unicode mapping (preserved as-is).
        first_char: First character code to cover (inclusive).
        last_char: Last character code to cover (inclusive).

    Returns:
        New mapping with gaps filled by PUA codepoints.
    """
    # Find codes that need a mapping
    missing_codes = [
        c for c in range(first_char, last_char + 1) if c not in code_to_unicode
    ]
    if not missing_codes:
        return code_to_unicode

    # Collect existing PUA values to avoid collisions
    existing_pua = {v for v in code_to_unicode.values() if 0xE000 <= v <= 0xF8FF}
    next_pua = 0xE000
    result = dict(code_to_unicode)

    pua_count = 0
    for code in missing_codes:
        while next_pua in existing_pua and next_pua <= 0xF8FF:
            next_pua += 1
        if next_pua <= 0xF8FF:
            result[code] = next_pua
            existing_pua.add(next_pua)
            next_pua += 1
            pua_count += 1
        # else: PUA exhausted, skip

    if pua_count > 0:
        logger.warning(
            "%d character codes mapped to PUA codepoints (U+E000-U+F8FF) "
            "— text extraction will not produce meaningful Unicode "
            "for these characters",
            pua_count,
        )

    return result


def generate_to_unicode_for_simple_font(font_name: str) -> bytes:
    """Generates ToUnicode CMap for Simple Fonts (Standard-14 replacements).

    This enables text extraction and copy/paste for PDF/A-2b compliance.
    Simple fonts use 8-bit encoding (codes 0-255) unlike CIDFonts which use
    16-bit encoding.

    Args:
        font_name: Name of the Standard-14 font being replaced.

    Returns:
        CMap data in PostScript format as bytes.
    """
    # Build code -> Unicode mapping based on font type
    code_to_unicode: dict[int, int] = {}

    if font_name == "Symbol":
        # Symbol font: use SYMBOL_ENCODING + glyph-to-unicode mappings
        for code, glyph_name in SYMBOL_ENCODING.items():
            unicode_val = resolve_symbol_glyph_to_unicode(glyph_name)
            if unicode_val is not None:
                code_to_unicode[code] = unicode_val
    elif font_name == "ZapfDingbats":
        # ZapfDingbats: use ZAPFDINGBATS_ENCODING + glyph-to-unicode mappings
        for code, glyph_name in ZAPFDINGBATS_ENCODING.items():
            unicode_val = ZAPFDINGBATS_GLYPH_TO_UNICODE.get(glyph_name)
            if unicode_val is not None:
                code_to_unicode[code] = unicode_val
    else:
        # Standard fonts (Helvetica, Times, Courier): WinAnsiEncoding (CP1252)
        code_to_unicode = generate_tounicode_for_winansi()

    # Generate Adobe CMap format (8-bit codespacerange)
    return generate_tounicode_cmap_data(code_to_unicode)


def resolve_symbol_glyph_to_unicode(glyph_name: str) -> int | None:
    """Resolves a Symbol font glyph name to its Unicode codepoint.

    Checks SYMBOL_GLYPH_TO_UNICODE first (for special/variant glyphs),
    then falls back to the standard Adobe Glyph List (AGL2UV).

    Args:
        glyph_name: Adobe glyph name from SYMBOL_ENCODING.

    Returns:
        Unicode codepoint, or None if the glyph has no Unicode equivalent.
    """
    # Check custom Symbol mapping first (for exceptions and variants)
    if glyph_name in SYMBOL_GLYPH_TO_UNICODE:
        return SYMBOL_GLYPH_TO_UNICODE[glyph_name]

    # Fall back to standard Adobe Glyph List
    if glyph_name in AGL2UV:
        return AGL2UV[glyph_name]

    return None


def generate_tounicode_for_winansi() -> dict[int, int]:
    """Generates code-to-Unicode mapping for WinAnsiEncoding (CP1252).

    Returns:
        Dictionary mapping character codes to Unicode codepoints.
    """
    code_to_unicode: dict[int, int] = {}
    for code in range(256):
        try:
            char = bytes([code]).decode("cp1252")
            code_to_unicode[code] = ord(char)
        except UnicodeDecodeError:
            pass
    return code_to_unicode


def generate_tounicode_for_macroman() -> dict[int, int]:
    """Generates code-to-Unicode mapping for MacRomanEncoding.

    Returns:
        Dictionary mapping character codes to Unicode codepoints.
    """
    code_to_unicode: dict[int, int] = {}
    for code in range(256):
        try:
            char = bytes([code]).decode("mac_roman")
            code_to_unicode[code] = ord(char)
        except UnicodeDecodeError:
            pass
    return code_to_unicode


def generate_tounicode_for_standard_encoding() -> dict[int, int]:
    """Generates code-to-Unicode mapping for StandardEncoding.

    Uses STANDARD_ENCODING glyph names resolved via Adobe Glyph List (AGL).

    Returns:
        Dictionary mapping character codes to Unicode codepoints.
    """
    code_to_unicode: dict[int, int] = {}
    for code, glyph_name in STANDARD_ENCODING.items():
        if glyph_name in AGL2UV:
            code_to_unicode[code] = AGL2UV[glyph_name]
    return code_to_unicode


def generate_tounicode_from_encoding_dict(
    encoding: pikepdf.Object,
) -> dict[int, int]:
    """Generates code-to-Unicode mapping from an Encoding dictionary.

    Handles BaseEncoding and Differences array.

    Args:
        encoding: Encoding dictionary object.

    Returns:
        Dictionary mapping character codes to Unicode codepoints.
    """
    # Dereference if needed
    encoding = _resolve_indirect(encoding)

    # Start with base encoding
    base_encoding = encoding.get("/BaseEncoding")
    if base_encoding is not None:
        base_name = _safe_str(base_encoding)
        if base_name == "/WinAnsiEncoding":
            code_to_unicode = generate_tounicode_for_winansi()
        elif base_name == "/MacRomanEncoding":
            code_to_unicode = generate_tounicode_for_macroman()
        elif base_name == "/StandardEncoding":
            code_to_unicode = generate_tounicode_for_standard_encoding()
        else:
            code_to_unicode = generate_tounicode_for_winansi()
    else:
        # Default to StandardEncoding (per PDF spec, the implicit base
        # encoding for non-symbolic Type1 fonts is StandardEncoding)
        code_to_unicode = generate_tounicode_for_standard_encoding()

    # Apply Differences array
    differences = encoding.get("/Differences")
    if differences is not None:
        code_to_unicode = apply_differences_to_mapping(code_to_unicode, differences)

    return code_to_unicode


def apply_differences_to_mapping(
    base_mapping: dict[int, int],
    differences: pikepdf.Array,
) -> dict[int, int]:
    """Applies a Differences array to a code-to-Unicode mapping.

    Uses Adobe Glyph List (AGL) to resolve glyph names to Unicode.

    Args:
        base_mapping: Starting code-to-Unicode mapping.
        differences: PDF Differences array.

    Returns:
        Updated mapping with differences applied.
    """
    result = base_mapping.copy()
    current_code = 0

    for item in differences:
        try:
            current_code = int(item)
            continue
        except (TypeError, ValueError):
            pass
        if isinstance(item, pikepdf.Name):
            glyph_name = _safe_str(item)[1:]  # Remove leading "/"
            unicode_val = resolve_glyph_to_unicode(glyph_name)
            if unicode_val is not None:
                result[current_code] = unicode_val
            current_code += 1

    return result


def resolve_glyph_to_unicode(glyph_name: str) -> int | None:
    """Resolves a glyph name to its Unicode codepoint.

    Uses Adobe Glyph List (AGL) for standard glyph names.

    Args:
        glyph_name: Adobe glyph name.

    Returns:
        Unicode codepoint, or None if not found.
    """
    # Check Adobe Glyph List
    if glyph_name in AGL2UV:
        return AGL2UV[glyph_name]

    # Handle uniXXXX format (e.g., uni0041 = 'A')
    if glyph_name.startswith("uni") and len(glyph_name) == 7:
        try:
            val = int(glyph_name[3:], 16)
            if _is_invalid_unicode(val):
                return None
            return val
        except ValueError:
            pass

    # Handle uXXXX or uXXXXX format
    if glyph_name.startswith("u") and len(glyph_name) in (5, 6):
        try:
            val = int(glyph_name[1:], 16)
            if _is_invalid_unicode(val):
                return None
            return val
        except ValueError:
            pass

    return None


def generate_tounicode_for_type3_font(
    font_obj: pikepdf.Object,
) -> dict[int, int]:
    """Generates code-to-Unicode mapping for a Type3 font.

    Type3 fonts often use custom glyph names that are not resolvable via AGL.
    For PDF/A-2/3 compliance (all levels, rule 6.2.11.7.2), every character
    code must map to Unicode. Unresolvable glyph names are mapped to the
    Unicode Private Use Area (U+E000-U+F8FF) to ensure complete coverage.

    Args:
        font_obj: pikepdf Type3 font object.

    Returns:
        Dictionary mapping character codes to Unicode codepoints.
    """
    encoding = font_obj.get("/Encoding")
    first_char = 0
    last_char = 255

    try:
        fc = font_obj.get("/FirstChar")
        if fc is not None:
            first_char = int(fc)
    except (TypeError, ValueError):
        pass

    try:
        lc = font_obj.get("/LastChar")
        if lc is not None:
            last_char = int(lc)
    except (TypeError, ValueError):
        pass

    # Build code → glyph name mapping from encoding
    code_to_glyph: dict[int, str] = {}

    try:
        encoding.get
        has_get = True
    except Exception:
        has_get = False

    if isinstance(encoding, pikepdf.Dictionary) or (
        encoding is not None and not isinstance(encoding, pikepdf.Name) and has_get
    ):
        encoding = _resolve_indirect(encoding)
        differences = encoding.get("/Differences")
        if differences is not None:
            current_code = 0
            for item in differences:
                try:
                    current_code = int(item)
                    continue
                except (TypeError, ValueError):
                    pass
                if isinstance(item, pikepdf.Name):
                    glyph_name = _safe_str(item)[1:]  # Remove "/"
                    code_to_glyph[current_code] = glyph_name
                    current_code += 1
    elif isinstance(encoding, pikepdf.Name):
        enc_name = _safe_str(encoding)
        if enc_name == "/WinAnsiEncoding":
            return generate_tounicode_for_winansi()
        elif enc_name == "/MacRomanEncoding":
            return generate_tounicode_for_macroman()
        elif enc_name == "/StandardEncoding":
            return generate_tounicode_for_standard_encoding()

    # Resolve glyph names to Unicode, with PUA fallback
    code_to_unicode: dict[int, int] = {}
    pua_glyphs: list[str] = []
    next_pua = 0xE000

    for code in range(first_char, last_char + 1):
        glyph_name = code_to_glyph.get(code)
        if glyph_name is None or glyph_name == ".notdef":
            continue

        unicode_val = resolve_glyph_to_unicode(glyph_name)
        if unicode_val is not None:
            code_to_unicode[code] = unicode_val
        else:
            # Map to Private Use Area for PDF/A-2/3 compliance
            if next_pua <= 0xF8FF:
                code_to_unicode[code] = next_pua
                next_pua += 1
                if next_pua > 0xF8FF:
                    next_pua = 0xF0000  # Supplementary PUA-A
                pua_glyphs.append(glyph_name)
            elif next_pua <= 0xFFFFD:
                code_to_unicode[code] = next_pua
                next_pua += 1
                pua_glyphs.append(glyph_name)

    if pua_glyphs:
        logger.warning(
            "Type3 font: %d glyph(s) mapped to PUA codepoints "
            "(U+E000-U+F8FF) — unresolvable names: %s",
            len(pua_glyphs),
            ", ".join(pua_glyphs[:10]) + (" ..." if len(pua_glyphs) > 10 else ""),
        )

    return code_to_unicode


def generate_tounicode_cmap_data(
    code_to_unicode: dict[int, int],
) -> bytes:
    """Generates ToUnicode CMap data for simple fonts (8-bit encoding).

    Args:
        code_to_unicode: Mapping from character codes to Unicode.

    Returns:
        CMap data as bytes.
    """
    code_to_unicode = filter_invalid_unicode_values(code_to_unicode)
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo <<",
        "  /Registry (Adobe)",
        "  /Ordering (UCS)",
        "  /Supplement 0",
        ">> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<00> <FF>",
        "endcodespacerange",
    ]

    # Group entries into chunks (max 100 per block)
    sorted_codes = sorted(code_to_unicode.keys())
    chunk_size = 100

    for i in range(0, len(sorted_codes), chunk_size):
        chunk = sorted_codes[i : i + chunk_size]
        lines.append(f"{len(chunk)} beginbfchar")
        for code in chunk:
            unicode_val = code_to_unicode[code]
            if unicode_val <= 0xFFFF:
                lines.append(f"<{code:02X}> <{unicode_val:04X}>")
            else:
                # Surrogate pair for Unicode > 0xFFFF
                high = 0xD800 + ((unicode_val - 0x10000) >> 10)
                low = 0xDC00 + ((unicode_val - 0x10000) & 0x3FF)
                lines.append(f"<{code:02X}> <{high:04X}{low:04X}>")
        lines.append("endbfchar")

    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )

    result = "\n".join(lines).encode("ascii")
    validate_tounicode_cmap(result)
    return result


def generate_cidfont_tounicode_cmap(
    code_to_unicode: dict[int, int],
) -> bytes:
    """Generates ToUnicode CMap data for CIDFonts (16-bit encoding).

    Args:
        code_to_unicode: Mapping from character codes (CID/GID) to Unicode.

    Returns:
        CMap data as bytes.
    """
    code_to_unicode = filter_invalid_unicode_values(code_to_unicode)
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo <<",
        "  /Registry (Adobe)",
        "  /Ordering (UCS)",
        "  /Supplement 0",
        ">> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]

    # Group entries into chunks (max 100 per block)
    sorted_codes = sorted(code_to_unicode.keys())
    chunk_size = 100

    for i in range(0, len(sorted_codes), chunk_size):
        chunk = sorted_codes[i : i + chunk_size]
        lines.append(f"{len(chunk)} beginbfchar")
        for code in chunk:
            unicode_val = code_to_unicode[code]
            if unicode_val <= 0xFFFF:
                lines.append(f"<{code:04X}> <{unicode_val:04X}>")
            else:
                # Surrogate pair for Unicode > 0xFFFF
                high = 0xD800 + ((unicode_val - 0x10000) >> 10)
                low = 0xDC00 + ((unicode_val - 0x10000) & 0x3FF)
                lines.append(f"<{code:04X}> <{high:04X}{low:04X}>")
        lines.append("endbfchar")

    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )

    result = "\n".join(lines).encode("ascii")
    validate_tounicode_cmap(result)
    return result


def validate_tounicode_cmap(data: bytes) -> None:
    """Validates the structural syntax of a generated ToUnicode CMap.

    Checks for required PostScript elements, balanced begin/end blocks,
    correct bfchar entry counts, and valid hex values.

    Args:
        data: CMap data as bytes.

    Raises:
        ValueError: If the CMap syntax is invalid.
    """
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as e:
        raise ValueError(f"CMap contains non-ASCII bytes: {e}") from e

    # Required structural elements
    required = [
        "/CIDInit /ProcSet findresource begin",
        "begincmap",
        "endcmap",
        "/CIDSystemInfo",
        "/Registry (Adobe)",
        "/Ordering (UCS)",
        "begincodespacerange",
        "endcodespacerange",
        "CMapName currentdict /CMap defineresource pop",
    ]
    for element in required:
        if element not in text:
            raise ValueError(f"Missing required CMap element: {element}")

    # Validate codespacerange
    codespace_match = re.search(
        r"(\d+)\s+begincodespacerange\s*(.*?)\s*endcodespacerange",
        text,
        re.DOTALL,
    )
    if codespace_match is None:
        raise ValueError("Invalid codespacerange block")
    declared_count = int(codespace_match.group(1))
    range_entries = re.findall(
        r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
        codespace_match.group(2),
    )
    if len(range_entries) != declared_count:
        raise ValueError(
            f"codespacerange declares {declared_count} entries "
            f"but contains {len(range_entries)}"
        )

    # Validate bfchar blocks
    bfchar_blocks = re.finditer(
        r"(\d+)\s+beginbfchar\s*(.*?)\s*endbfchar", text, re.DOTALL
    )
    hex_entry = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")

    for block in bfchar_blocks:
        declared = int(block.group(1))
        if declared > 100:
            raise ValueError(f"bfchar block declares {declared} entries (max 100)")
        entries = hex_entry.findall(block.group(2))
        if len(entries) != declared:
            raise ValueError(
                f"bfchar block declares {declared} entries but contains {len(entries)}"
            )

    # Check balanced begin/end for cmap
    if text.count("begincmap") != text.count("endcmap"):
        raise ValueError("Unbalanced begincmap/endcmap")


def parse_cidtogidmap_stream(stream_data: bytes) -> dict[int, int]:
    """Parses a CIDToGIDMap stream into a CID-to-GID mapping.

    The stream contains 2-byte big-endian GID values, indexed by CID.
    CID 0 maps to the first 2 bytes, CID 1 to the next 2 bytes, etc.

    Args:
        stream_data: Raw bytes of the CIDToGIDMap stream.

    Returns:
        Dictionary mapping CID to GID, excluding GID=0 (.notdef).
    """
    if len(stream_data) % 2 != 0:
        logger.warning(
            "CIDToGIDMap stream has odd length %d; possibly truncated",
            len(stream_data),
        )
    cid_to_gid: dict[int, int] = {}
    num_entries = len(stream_data) // 2
    for cid in range(num_entries):
        gid = (stream_data[cid * 2] << 8) | stream_data[cid * 2 + 1]
        if gid != 0:
            cid_to_gid[cid] = gid
    return cid_to_gid


def build_identity_unicode_mapping(cmap: dict[int, str]) -> dict[int, int]:
    """Builds an identity Unicode mapping for UTF-16 encoded CIDFonts.

    For UTF-16/UCS-2 encodings, character codes are already Unicode values,
    so the ToUnicode map is simply each code mapping to itself.

    Args:
        cmap: The font's cmap table (unicode_val -> glyph_name).

    Returns:
        Dictionary mapping Unicode values to themselves.
    """
    return {unicode_val: unicode_val for unicode_val in cmap}


def _decode_unicode_hex(hex_str: str) -> int:
    """Decodes a hex string from a CMap entry to a Unicode codepoint.

    Handles both BMP values (4 hex digits) and surrogate pairs (8 hex digits).

    Args:
        hex_str: Hex string like "0041" or "D800DC00".

    Returns:
        Unicode codepoint as integer.
    """
    if len(hex_str) == 8:
        # Surrogate pair
        high = int(hex_str[:4], 16)
        low = int(hex_str[4:], 16)
        if 0xD800 <= high <= 0xDBFF and 0xDC00 <= low <= 0xDFFF:
            return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
    return int(hex_str, 16)


def parse_tounicode_cmap(data: bytes) -> dict[int, int]:
    """Parses a ToUnicode CMap stream into a code-to-Unicode mapping.

    Extracts entries from beginbfchar/endbfchar and beginbfrange/endbfrange
    blocks.

    Args:
        data: Raw CMap stream bytes.

    Returns:
        Dictionary mapping character codes to Unicode codepoints.
    """
    code_to_unicode: dict[int, int] = {}

    try:
        text = data.decode("ascii", errors="replace")
    except Exception:
        return code_to_unicode

    # Parse bfchar blocks: <src_code> <unicode_value>
    bfchar_pattern = re.compile(r"beginbfchar\s*(.*?)\s*endbfchar", re.DOTALL)
    entry_pattern = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")

    for block_match in bfchar_pattern.finditer(text):
        block = block_match.group(1)
        for entry_match in entry_pattern.finditer(block):
            src_hex = entry_match.group(1)
            dst_hex = entry_match.group(2)
            try:
                code = int(src_hex, 16)
                unicode_val = _decode_unicode_hex(dst_hex)
                code_to_unicode[code] = unicode_val
            except ValueError:
                continue

    # Parse bfrange blocks: <start> <end> <unicode_start> or
    # <start> <end> [<u1> <u2> ... <un>]
    bfrange_pattern = re.compile(r"beginbfrange\s*(.*?)\s*endbfrange", re.DOTALL)
    range_inc_pattern = re.compile(
        r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>"
    )
    range_array_pattern = re.compile(
        r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[([^\]]*)\]"
    )
    array_element_pattern = re.compile(r"<([0-9A-Fa-f]+)>")

    for block_match in bfrange_pattern.finditer(text):
        block = block_match.group(1)
        # Incrementing destination form
        for entry_match in range_inc_pattern.finditer(block):
            start_hex = entry_match.group(1)
            end_hex = entry_match.group(2)
            dst_hex = entry_match.group(3)
            try:
                start_code = int(start_hex, 16)
                end_code = int(end_hex, 16)
                unicode_start = _decode_unicode_hex(dst_hex)
                for offset in range(end_code - start_code + 1):
                    code_to_unicode[start_code + offset] = unicode_start + offset
            except ValueError:
                continue
        # Array destination form
        for entry_match in range_array_pattern.finditer(block):
            start_hex = entry_match.group(1)
            end_hex = entry_match.group(2)
            array_body = entry_match.group(3)
            try:
                start_code = int(start_hex, 16)
                end_code = int(end_hex, 16)
                elements = array_element_pattern.findall(array_body)
                for offset, elem_hex in enumerate(
                    elements[: end_code - start_code + 1]
                ):
                    unicode_val = _decode_unicode_hex(elem_hex)
                    code_to_unicode[start_code + offset] = unicode_val
            except ValueError:
                continue

    return code_to_unicode
