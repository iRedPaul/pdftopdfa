# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ToUnicode CMap generation for PDF/A compliance."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import pikepdf
from fontTools import agl
from pdfminer.cmapdb import CMapDB

from ..utils import resolve_indirect as _resolve_indirect
from .constants import UTF16_ENCODING_NAMES
from .encodings import STANDARD_ENCODING, SYMBOL_ENCODING, ZAPFDINGBATS_ENCODING
from .glyph_mapping import SYMBOL_GLYPH_TO_UNICODE, ZAPFDINGBATS_GLYPH_TO_UNICODE
from .utils import safe_str as _safe_str

logger = logging.getLogger(__name__)

UnicodeValue = int | tuple[int, ...]

# Unicode values forbidden in PDF/A ToUnicode CMaps (veraPDF rule 6.2.11.7.2)
INVALID_UNICODE_VALUES = frozenset({0x0000, 0xFEFF, 0xFFFE})

# Unicode surrogate code points (U+D800–U+DFFF) — not valid Unicode scalar
# values. ISO 19005-2 §6.2.11.7.2 forbids these in ToUnicode mappings.
_SURROGATE_RANGE = range(0xD800, 0xE000)

# Adobe CFF specification, Appendix B: ExpertEncoding code-to-SID table.
# fmt: off
_CFF_EXPERT_ENCODING_SIDS = (
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 229, 230, 0, 231, 232, 233, 234,
    235, 236, 237, 238, 13, 14, 15, 99,
    239, 240, 241, 242, 243, 244, 245, 246,
    247, 248, 27, 28, 249, 250, 251, 252,
    0, 253, 254, 255, 256, 257, 0, 0,
    0, 258, 0, 0, 259, 260, 261, 262,
    0, 0, 263, 264, 265, 0, 266, 109,
    110, 267, 268, 269, 0, 270, 271, 272,
    273, 274, 275, 276, 277, 278, 279, 280,
    281, 282, 283, 284, 285, 286, 287, 288,
    289, 290, 291, 292, 293, 294, 295, 296,
    297, 298, 299, 300, 301, 302, 303, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 304, 305, 306, 0, 0, 307, 308,
    309, 310, 311, 0, 312, 0, 0, 313,
    0, 0, 314, 315, 0, 0, 316, 317,
    318, 0, 0, 0, 158, 155, 163, 319,
    320, 321, 322, 323, 324, 325, 0, 0,
    326, 150, 164, 169, 327, 328, 329, 330,
    331, 332, 333, 334, 335, 336, 337, 338,
    339, 340, 341, 342, 343, 344, 345, 346,
    347, 348, 349, 350, 351, 352, 353, 354,
    355, 356, 357, 358, 359, 360, 361, 362,
    363, 364, 365, 366, 367, 368, 369, 370,
    371, 372, 373, 374, 375, 376, 377, 378,
)
# fmt: on


def _is_invalid_unicode(val: int) -> bool:
    """Return True if a Unicode value is forbidden in PDF/A ToUnicode CMaps."""
    return val in INVALID_UNICODE_VALUES or val in _SURROGATE_RANGE


def _unicode_scalars(value: UnicodeValue) -> tuple[int, ...]:
    """Return a mapping destination as a tuple of Unicode scalars."""
    return (value,) if isinstance(value, int) else value


def filter_invalid_unicode_values(
    code_to_unicode: dict[int, UnicodeValue],
) -> dict[int, UnicodeValue]:
    """Replaces forbidden Unicode values with Private Use Area codepoints.

    PDF/A (veraPDF rule 6.2.11.7.2) forbids U+0000, U+FEFF, and U+FFFE
    in ToUnicode mappings. This replaces them with PUA codepoints (U+E000+)
    while avoiding collisions with existing PUA values.

    Args:
        code_to_unicode: Mapping from character codes to Unicode codepoints.

    Returns:
        New mapping with invalid values replaced by PUA codepoints.
    """
    if not any(
        _is_invalid_unicode(value)
        for unicode_value in code_to_unicode.values()
        for value in _unicode_scalars(unicode_value)
    ):
        return code_to_unicode

    # Collect existing PUA values to avoid collisions
    existing_pua = {
        value
        for unicode_value in code_to_unicode.values()
        for value in _unicode_scalars(unicode_value)
        if 0xE000 <= value <= 0xF8FF
    }
    next_pua = 0xE000
    result = {}

    for code, unicode_val in code_to_unicode.items():
        values = list(_unicode_scalars(unicode_val))
        complete = True
        for index, value in enumerate(values):
            if not _is_invalid_unicode(value):
                continue
            # Find next available PUA codepoint
            while next_pua in existing_pua and next_pua <= 0xF8FF:
                next_pua += 1
            if next_pua > 0xF8FF:
                complete = False
                break
            values[index] = next_pua
            existing_pua.add(next_pua)
            next_pua += 1
        if complete:
            result[code] = values[0] if isinstance(unicode_val, int) else tuple(values)

    return result


def fill_tounicode_gaps_with_pua(
    code_to_unicode: dict[int, UnicodeValue],
    first_char: int = 0,
    last_char: int = 255,
) -> dict[int, UnicodeValue]:
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
    existing_pua = {
        value
        for unicode_value in code_to_unicode.values()
        for value in _unicode_scalars(unicode_value)
        if 0xE000 <= value <= 0xF8FF
    }
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
    if glyph_name in agl.AGL2UV:
        return agl.AGL2UV[glyph_name]

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
        if glyph_name in agl.AGL2UV:
            code_to_unicode[code] = agl.AGL2UV[glyph_name]
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


def resolve_glyph_to_unicode_sequence(glyph_name: str) -> tuple[int, ...] | None:
    """Resolve an Adobe glyph name to its complete Unicode sequence."""
    try:
        text = agl.toUnicode(glyph_name)
    except (TypeError, ValueError):
        return None
    values = tuple(ord(char) for char in text)
    if not values or any(_is_invalid_unicode(value) for value in values):
        return None
    return values


def resolve_glyph_to_unicode(glyph_name: str) -> int | None:
    """Resolve a glyph name when it represents exactly one Unicode scalar."""
    values = resolve_glyph_to_unicode_sequence(glyph_name)
    return values[0] if values is not None and len(values) == 1 else None


def parse_type1_font_program(font_data: bytes):
    """Parse an embedded Type 1 PFA/PFB program with fontTools.

    Some PDF producers omit the clear-text Type 1 trailer because the PDF
    stream's ``Length1`` and ``Length2`` values already delimit the program.
    fontTools still requires the trailer and an ASCII clear-text header.

    Args:
        font_data: Raw PFA or PFB font bytes.

    Returns:
        A parsed ``T1Font``, or ``None`` when the program cannot be parsed.
    """
    import os
    import tempfile

    from fontTools.t1Lib import T1Font

    prepared_data = font_data
    if font_data.startswith(b"%!"):
        header, marker, encrypted = font_data.partition(b"eexec")
        if marker:
            header = bytes(value if value < 128 else ord("?") for value in header)
            prepared_data = header + marker + encrypted
            if b"cleartomark" not in encrypted:
                prepared_data += b"\n" + (b"0" * 512) + b"\ncleartomark\n"

    suffix = ".pfa" if font_data[:2] == b"%!" else ".pfb"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, prepared_data)
    finally:
        os.close(tmp_fd)

    try:
        font = T1Font(tmp_path)
        font.parse()
        return font
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _glyph_name_unicode_value(glyph_name: str) -> UnicodeValue | None:
    """Resolve one glyph name, preserving multi-scalar Unicode sequences."""
    values = resolve_glyph_to_unicode_sequence(glyph_name)
    if values is None:
        return None
    return values[0] if len(values) == 1 else values


def generate_tounicode_from_type1_program(
    font_data: bytes,
) -> dict[int, UnicodeValue]:
    """Build a Unicode mapping from an embedded Type 1 program's Encoding."""
    font = parse_type1_font_program(font_data)
    if font is None:
        return {}

    encoding = font.font.get("Encoding")
    if not isinstance(encoding, list):
        return {}

    result: dict[int, UnicodeValue] = {}
    for code, glyph_name in enumerate(encoding):
        glyph_name = str(glyph_name)
        if glyph_name == ".notdef":
            continue
        unicode_val = _glyph_name_unicode_value(glyph_name)
        if unicode_val is not None:
            result[code] = unicode_val
    return result


def generate_tounicode_from_cff_program(
    font_data: bytes,
    font_name: str | None = None,
) -> dict[int, UnicodeValue]:
    """Build a Unicode mapping from a Type1C program's effective Encoding."""
    from io import BytesIO

    from fontTools.cffLib import CFFFontSet, cffStandardStrings

    try:
        font_set = CFFFontSet()
        font_set.decompile(BytesIO(font_data), None)
        if not font_set.fontNames:
            return {}
        selected_name = font_set.fontNames[0]
        if font_name is not None:
            requested_name = font_name.removeprefix("/")
            if requested_name in font_set.fontNames:
                selected_name = requested_name
            elif len(font_set.fontNames) > 1:
                return {}
        top_dict = font_set[selected_name]
        encoding = top_dict.Encoding
    except Exception:
        return {}

    if isinstance(encoding, list):
        code_to_glyph = enumerate(encoding)
    elif encoding == "StandardEncoding":
        code_to_glyph = STANDARD_ENCODING.items()
    elif encoding == "ExpertEncoding":
        code_to_glyph = (
            (code, cffStandardStrings[sid])
            for code, sid in enumerate(_CFF_EXPERT_ENCODING_SIDS)
            if sid
        )
    else:
        return {}

    charset = {str(glyph_name) for glyph_name in top_dict.charset}
    result: dict[int, UnicodeValue] = {}
    for code, glyph_name in code_to_glyph:
        glyph_name = str(glyph_name)
        if glyph_name == ".notdef" or glyph_name not in charset:
            continue
        unicode_val = _glyph_name_unicode_value(glyph_name)
        if unicode_val is not None:
            result[code] = unicode_val
    return result


def generate_tounicode_for_type3_font(
    font_obj: pikepdf.Object,
) -> dict[int, UnicodeValue]:
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

    # Simple fonts use an 8-bit codespace (<00> <FF>); codes outside that
    # range would produce invalid multi-byte entries in the ToUnicode CMap.
    first_char = max(first_char, 0)
    last_char = min(last_char, 255)

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
    code_to_unicode: dict[int, UnicodeValue] = {}
    pua_glyphs: list[str] = []
    next_pua = 0xE000

    for code in range(first_char, last_char + 1):
        glyph_name = code_to_glyph.get(code)
        if glyph_name is None or glyph_name == ".notdef":
            continue

        unicode_val = _glyph_name_unicode_value(glyph_name)
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


def _unicode_value_to_utf16_hex(unicode_value: UnicodeValue) -> str:
    """Encode one Unicode scalar or sequence as a ToUnicode destination."""
    text = "".join(chr(value) for value in _unicode_scalars(unicode_value))
    return text.encode("utf-16-be").hex().upper()


def _format_tounicode_cmap(
    code_to_unicode: dict[int, UnicodeValue],
    *,
    source_hex_width: int,
) -> bytes:
    """Format a code-to-Unicode mapping as an Adobe ToUnicode CMap."""
    max_source = "F" * source_hex_width
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
        f"<{'0' * source_hex_width}> <{max_source}>",
        "endcodespacerange",
    ]

    sorted_codes = sorted(code_to_unicode)
    for i in range(0, len(sorted_codes), 100):
        chunk = sorted_codes[i : i + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for code in chunk:
            destination = _unicode_value_to_utf16_hex(code_to_unicode[code])
            lines.append(f"<{code:0{source_hex_width}X}> <{destination}>")
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


def generate_tounicode_cmap_data(
    code_to_unicode: dict[int, UnicodeValue],
) -> bytes:
    """Generates ToUnicode CMap data for simple fonts (8-bit encoding).

    Args:
        code_to_unicode: Mapping from character codes to Unicode.

    Returns:
        CMap data as bytes.
    """
    code_to_unicode = filter_invalid_unicode_values(code_to_unicode)
    # The declared codespace is <00> <FF>; codes outside the 8-bit range
    # cannot be represented and are dropped.
    code_to_unicode = {c: u for c, u in code_to_unicode.items() if 0 <= c <= 255}
    return _format_tounicode_cmap(code_to_unicode, source_hex_width=2)


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
    return _format_tounicode_cmap(code_to_unicode, source_hex_width=4)


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
    # Both destination forms in one pattern so that each entry is consumed
    # atomically: hex tokens inside an array body must never be matched as
    # a separate incrementing entry.
    range_entry_pattern = re.compile(
        r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(?:<([0-9A-Fa-f]+)>|\[([^\]]*)\])"
    )
    array_element_pattern = re.compile(r"<([0-9A-Fa-f]+)>")

    for block_match in bfrange_pattern.finditer(text):
        block = block_match.group(1)
        for entry_match in range_entry_pattern.finditer(block):
            start_hex = entry_match.group(1)
            end_hex = entry_match.group(2)
            dst_hex = entry_match.group(3)
            try:
                start_code = int(start_hex, 16)
                end_code = int(end_hex, 16)
                if dst_hex is not None:
                    # Incrementing destination form
                    unicode_start = _decode_unicode_hex(dst_hex)
                    for offset in range(end_code - start_code + 1):
                        code_to_unicode[start_code + offset] = unicode_start + offset
                else:
                    # Array destination form
                    elements = array_element_pattern.findall(entry_match.group(4))
                    for offset, elem_hex in enumerate(
                        elements[: end_code - start_code + 1]
                    ):
                        unicode_val = _decode_unicode_hex(elem_hex)
                        code_to_unicode[start_code + offset] = unicode_val
            except ValueError:
                continue

    return code_to_unicode


def parse_cmap_codespace_ranges(
    data: bytes,
) -> tuple[tuple[bytes, bytes], ...]:
    """Return the byte ranges declared by CMap codespace blocks."""
    try:
        text = data.decode("ascii", errors="replace")
    except Exception:
        return ()

    ranges: list[tuple[bytes, bytes]] = []
    block_pattern = re.compile(
        r"begincodespacerange\s*(.*?)\s*endcodespacerange",
        re.DOTALL,
    )
    entry_pattern = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
    for block_match in block_pattern.finditer(text):
        for entry_match in entry_pattern.finditer(block_match.group(1)):
            try:
                lower = bytes.fromhex(entry_match.group(1))
                upper = bytes.fromhex(entry_match.group(2))
            except ValueError:
                continue
            if 1 <= len(lower) <= 4 and len(lower) == len(upper) and lower <= upper:
                ranges.append((lower, upper))
    return tuple(ranges)


@dataclass(frozen=True)
class CIDEncodingMap:
    """Effective character-code to CID mappings for a Type 0 CMap."""

    ranges: tuple[tuple[bytes, bytes, int], ...]
    code_space_ranges: tuple[tuple[bytes, bytes], ...]
    unmapped_to_zero: bool
    notdef_ranges: tuple[tuple[bytes, bytes, int], ...] = ()
    decoder: Callable[[bytes], int | None] | None = None
    base: "CIDEncodingMap | None" = None

    @staticmethod
    def _map_ranges(
        ranges: tuple[tuple[bytes, bytes, int], ...],
        raw: bytes | None,
        value: int,
    ) -> int | None:
        for start, end, cid_start in reversed(ranges):
            if raw is not None:
                if len(raw) != len(start) or not start <= raw <= end:
                    continue
                return cid_start + (int.from_bytes(raw, "big") - int.from_bytes(start))
            start_value = int.from_bytes(start, "big")
            end_value = int.from_bytes(end, "big")
            if start_value <= value <= end_value:
                return cid_start + (value - start_value)
        return None

    def _map_regular(self, raw: bytes | None, value: int) -> int | None:
        current: CIDEncodingMap | None = self
        seen: set[int] = set()
        while current is not None:
            identity = id(current)
            if identity in seen:
                return None
            seen.add(identity)

            if current.decoder is not None:
                candidates: list[bytes] = []
                if raw is not None:
                    candidates.append(raw)
                else:
                    for lower, upper in current.code_space_ranges:
                        width = len(lower)
                        try:
                            candidate = value.to_bytes(width, "big")
                        except OverflowError:
                            continue
                        if lower <= candidate <= upper and candidate not in candidates:
                            candidates.append(candidate)
                for candidate in candidates:
                    mapped = current.decoder(candidate)
                    if mapped is not None:
                        return mapped
            mapped = self._map_ranges(current.ranges, raw, value)
            if mapped is not None:
                return mapped
            current = current.base
        return None

    def _map_notdef(self, raw: bytes | None, value: int) -> int | None:
        current: CIDEncodingMap | None = self
        seen: set[int] = set()
        while current is not None:
            identity = id(current)
            if identity in seen:
                return None
            seen.add(identity)
            mapped = self._map_ranges(current.notdef_ranges, raw, value)
            if mapped is not None:
                return mapped
            current = current.base
        return None

    def map_code(self, code: int | bytes) -> int:
        """Map one encoded character code to its descendant-font CID."""
        raw = code if isinstance(code, bytes) else None
        value = int.from_bytes(code, "big") if isinstance(code, bytes) else code
        mapped = self._map_regular(raw, value)
        if mapped is None:
            mapped = self._map_notdef(raw, value)
        if mapped is not None:
            return mapped
        return 0 if self.unmapped_to_zero else value

    def has_mapping(self, code: int | bytes) -> bool:
        """Return whether the CMap explicitly maps an encoded character code."""
        raw = code if isinstance(code, bytes) else None
        value = int.from_bytes(code, "big") if isinstance(code, bytes) else code
        return (
            self._map_regular(raw, value) is not None
            or self._map_notdef(raw, value) is not None
        )


_CID_BLOCK_RE = re.compile(
    rb"begin(?P<kind>cidchar|cidrange|notdefchar|notdefrange)\b"
    rb"(?P<body>.*?)end(?P=kind)\b",
    re.DOTALL,
)
_CIDCHAR_RE = re.compile(
    rb"<(?P<src>[0-9A-Fa-f]+)>\s+"
    rb"(?:<(?P<cid_hex>[0-9A-Fa-f]+)>|(?P<cid_int>-?\d+))"
)
_CIDRANGE_RE = re.compile(
    rb"<(?P<src>[0-9A-Fa-f]+)>\s+<(?P<end>[0-9A-Fa-f]+)>\s+"
    rb"(?:<(?P<cid_hex>[0-9A-Fa-f]+)>|(?P<cid_int>-?\d+))"
)
_CODE_SPACE_BLOCK_RE = re.compile(
    rb"(?:\d+\s+)?begincodespacerange\b.*?endcodespacerange\b",
    re.DOTALL,
)
_CID_MAPPING_BLOCK_RE = re.compile(
    rb"(?:\d+\s+)?begin(?:cidchar|cidrange|notdefchar|notdefrange)\b"
    rb".*?end(?:cidchar|cidrange|notdefchar|notdefrange)\b",
    re.DOTALL,
)


def parse_cid_encoding_cmap(data: bytes) -> CIDEncodingMap:
    """Parse CID mappings and code-space ranges from an embedded CMap."""
    uncommented = re.sub(rb"%[^\r\n]*", b"", data)
    ranges: list[tuple[bytes, bytes, int]] = []
    notdef_ranges: list[tuple[bytes, bytes, int]] = []
    for block in _CID_BLOCK_RE.finditer(uncommented):
        kind = block.group("kind")
        entry_pattern = _CIDCHAR_RE if kind.endswith(b"char") else _CIDRANGE_RE
        for entry in entry_pattern.finditer(block.group("body")):
            try:
                start = bytes.fromhex(entry.group("src").decode("ascii"))
                end_group = entry.groupdict().get("end")
                end = (
                    bytes.fromhex(end_group.decode("ascii"))
                    if end_group is not None
                    else start
                )
                cid_hex = entry.group("cid_hex")
                cid_start = (
                    int(cid_hex, 16)
                    if cid_hex is not None
                    else int(entry.group("cid_int"))
                )
            except (TypeError, ValueError):
                continue
            if len(start) == len(end) and start <= end:
                target = notdef_ranges if kind.startswith(b"notdef") else ranges
                target.append((start, end, cid_start))

    return CIDEncodingMap(
        ranges=tuple(ranges),
        code_space_ranges=parse_cmap_codespace_ranges(uncommented),
        unmapped_to_zero=not bool(re.search(rb"\busecmap\b", uncommented)),
        notdef_ranges=tuple(notdef_ranges),
    )


def _flatten_cid_ranges(
    mapping: CIDEncodingMap,
) -> (
    tuple[
        list[tuple[bytes, bytes, int]],
        list[tuple[bytes, bytes, int]],
    ]
    | None
):
    """Return inherited mappings in base-to-child definition order."""
    chain: list[CIDEncodingMap] = []
    seen: set[int] = set()
    current: CIDEncodingMap | None = mapping
    while current is not None:
        identity = id(current)
        if identity in seen or current.decoder is not None:
            return None
        seen.add(identity)
        chain.append(current)
        current = current.base

    regular: list[tuple[bytes, bytes, int]] = []
    notdef: list[tuple[bytes, bytes, int]] = []
    for layer in reversed(chain):
        regular.extend(layer.ranges)
        notdef.extend(layer.notdef_ranges)
    return regular, notdef


def flatten_cid_encoding_cmap(
    data: bytes,
    mapping: CIDEncodingMap,
) -> bytes | None:
    """Materialize an embedded CMap's effective inherited mappings."""
    flattened = _flatten_cid_ranges(mapping)
    if flattened is None:
        return None
    regular, notdef = flattened
    chunks: list[bytes] = []

    code_spaces = tuple(dict.fromkeys(mapping.code_space_ranges))
    for offset in range(0, len(code_spaces), 100):
        block = code_spaces[offset : offset + 100]
        lines = [f"{len(block)} begincodespacerange".encode("ascii")]
        lines.extend(
            b"<"
            + lower.hex().upper().encode("ascii")
            + b"> <"
            + upper.hex().upper().encode("ascii")
            + b">"
            for lower, upper in block
        )
        lines.append(b"endcodespacerange")
        chunks.append(b"\n".join(lines))

    def add_blocks(
        entries: list[tuple[bytes, bytes, int]],
        char_kind: bytes,
        range_kind: bytes,
    ) -> None:
        chars = [entry for entry in entries if entry[0] == entry[1]]
        ranges = [entry for entry in entries if entry[0] != entry[1]]
        for block_entries, kind in ((chars, char_kind), (ranges, range_kind)):
            for offset in range(0, len(block_entries), 100):
                block = block_entries[offset : offset + 100]
                lines = [str(len(block)).encode("ascii") + b" begin" + kind]
                for start, end, cid_start in block:
                    source = b"<" + start.hex().upper().encode("ascii") + b">"
                    if kind.endswith(b"range"):
                        source += b" <" + end.hex().upper().encode("ascii") + b">"
                    lines.append(source + b" " + str(cid_start).encode("ascii"))
                lines.append(b"end" + kind)
                chunks.append(b"\n".join(lines))

    add_blocks(regular, b"cidchar", b"cidrange")
    add_blocks(notdef, b"notdefchar", b"notdefrange")

    rewritten = _CODE_SPACE_BLOCK_RE.sub(b"", data)
    rewritten = _CID_MAPPING_BLOCK_RE.sub(b"", rewritten)
    rewritten = re.sub(
        rb"(?m)^[ \t]*/?[^\s/]+\s+usecmap[ \t]*$",
        b"",
        rewritten,
    )
    marker = re.search(rb"\bendcmap\b", rewritten)
    if marker is None:
        return None
    materialized = b"\n".join(chunks)
    return (
        rewritten[: marker.start()] + materialized + b"\n" + rewritten[marker.start() :]
    )


_UCS2_CODE_SPACES = ((b"\x00\x00", b"\xff\xff"),)
_UTF16_CODE_SPACES = (
    (b"\x00\x00", b"\xd7\xff"),
    (b"\xd8\x00\xdc\x00", b"\xdb\xff\xdf\xff"),
    (b"\xe0\x00", b"\xff\xff"),
)


def _pdfminer_cmap_decoder(name: str) -> Callable[[bytes], int | None] | None:
    """Load an authoritative predefined Adobe character-code CMap."""
    try:
        cmap = CMapDB.get_cmap(name)
    except Exception:
        return None

    def decode(raw: bytes) -> int | None:
        try:
            cids = list(cmap.decode(raw))
        except Exception:
            return None
        return cids[0] if len(cids) == 1 else None

    return decode


def _named_cid_encoding_map(name: str) -> CIDEncodingMap | None:
    normalized = name.lstrip("/")
    if normalized in {"Identity-H", "Identity-V"}:
        return CIDEncodingMap(
            ranges=(),
            code_space_ranges=((b"\x00\x00", b"\xff\xff"),),
            unmapped_to_zero=False,
        )

    if normalized not in UTF16_ENCODING_NAMES:
        return None
    decoder = _pdfminer_cmap_decoder(normalized)
    if decoder is None:
        return None
    code_spaces = _UTF16_CODE_SPACES if "-UTF16-" in normalized else _UCS2_CODE_SPACES
    return CIDEncodingMap(
        ranges=(),
        code_space_ranges=code_spaces,
        unmapped_to_zero=True,
        decoder=decoder,
    )


def _get_cid_encoding_map(
    encoding: pikepdf.Object,
) -> CIDEncodingMap | None:
    """Resolve an embedded CMap's /UseCMap chain without Python recursion."""
    layers: list[CIDEncodingMap] = []
    seen: set[tuple[int, int]] = set()
    current = _resolve_indirect(encoding)
    terminal: CIDEncodingMap | None = None
    has_terminal = False

    while True:
        if isinstance(current, pikepdf.Name):
            terminal = _named_cid_encoding_map(str(current))
            has_terminal = True
            break
        if not isinstance(current, pikepdf.Stream):
            return None

        objgen = current.objgen
        if objgen != (0, 0):
            if objgen in seen:
                return None
            seen.add(objgen)

        try:
            data = bytes(current.read_bytes())
        except Exception:
            return None
        layers.append(parse_cid_encoding_cmap(data))

        usecmap = current.get("/UseCMap")
        if usecmap is not None:
            current = _resolve_indirect(usecmap)
            continue

        match = re.search(rb"/([^\s/]+)\s+usecmap\b", data)
        if match is not None:
            terminal = _named_cid_encoding_map(match.group(1).decode("latin-1"))
            has_terminal = True
        break

    if has_terminal and terminal is None:
        return None

    effective = terminal
    for mapping in reversed(layers):
        if effective is None:
            effective = mapping
            continue
        code_spaces = tuple(
            dict.fromkeys(effective.code_space_ranges + mapping.code_space_ranges)
        )
        effective = CIDEncodingMap(
            ranges=mapping.ranges,
            code_space_ranges=code_spaces,
            unmapped_to_zero=effective.unmapped_to_zero,
            notdef_ranges=mapping.notdef_ranges,
            decoder=mapping.decoder,
            base=effective,
        )
    return effective


def get_type0_cid_encoding_map(
    font_obj: pikepdf.Object,
) -> CIDEncodingMap | None:
    """Return the effective Type 0 character-code to CID mapping."""
    font_obj = _resolve_indirect(font_obj)
    if not isinstance(font_obj, pikepdf.Dictionary):
        return None
    encoding = font_obj.get("/Encoding")
    if encoding is None:
        return None
    return _get_cid_encoding_map(encoding)


def map_type0_character_codes_to_cids(
    font_obj: pikepdf.Object,
    character_codes: set[int | bytes],
) -> set[int] | None:
    """Translate content-stream character codes through a Type 0 CMap."""
    mapping = get_type0_cid_encoding_map(font_obj)
    if mapping is None:
        return None
    return {mapping.map_code(code) for code in character_codes}


def split_cmap_codes(
    raw: bytes,
    ranges: tuple[tuple[bytes, bytes], ...],
) -> list[bytes]:
    """Split encoded text bytes according to a CMap codespace."""
    lengths = sorted({len(lower) for lower, _upper in ranges})
    codes: list[bytes] = []
    offset = 0
    while offset < len(raw):
        matched = None
        for length in lengths:
            candidate = raw[offset : offset + length]
            if len(candidate) != length:
                continue
            if any(
                len(lower) == length and lower <= candidate <= upper
                for lower, upper in ranges
            ):
                matched = candidate
                break
        if matched is None:
            if lengths and len(raw) - offset < min(lengths):
                break
            matched = raw[offset : offset + 1]
        codes.append(matched)
        offset += len(matched)
    return codes


def get_font_code_space_ranges(
    font_obj: pikepdf.Object,
    tounicode_data: bytes | None = None,
) -> tuple[tuple[bytes, bytes], ...]:
    """Return the effective character-code ranges for a PDF font."""
    font_obj = _resolve_indirect(font_obj)
    if not isinstance(font_obj, pikepdf.Dictionary):
        return ()
    if str(font_obj.get("/Subtype")) != "/Type0":
        return ((b"\x00", b"\xff"),)

    effective_mapping = get_type0_cid_encoding_map(font_obj)
    if effective_mapping is not None and effective_mapping.code_space_ranges:
        return effective_mapping.code_space_ranges

    encoding = _resolve_indirect(font_obj.get("/Encoding"))
    if isinstance(encoding, pikepdf.Stream):
        try:
            ranges = parse_cmap_codespace_ranges(bytes(encoding.read_bytes()))
        except Exception:
            ranges = ()
        if ranges:
            return ranges
    elif isinstance(encoding, pikepdf.Name) and str(encoding) in {
        "/Identity-H",
        "/Identity-V",
    }:
        return ((b"\x00\x00", b"\xff\xff"),)

    if tounicode_data is None:
        tounicode = _resolve_indirect(font_obj.get("/ToUnicode"))
        if isinstance(tounicode, pikepdf.Stream):
            try:
                tounicode_data = bytes(tounicode.read_bytes())
            except Exception:
                pass

    if tounicode_data is not None:
        ranges = parse_cmap_codespace_ranges(tounicode_data)
        if ranges:
            return ranges
    return ((b"\x00\x00", b"\xff\xff"),)


def parse_tounicode_cmap_sequences(
    data: bytes,
) -> dict[bytes, tuple[int, ...]]:
    """Parse a ToUnicode CMap without losing source widths or Unicode sequences."""
    try:
        text = data.decode("ascii", errors="replace")
    except Exception:
        return {}

    def decode_destination(hex_value: str) -> tuple[int, ...] | None:
        try:
            raw = bytes.fromhex(hex_value)
            if not raw or len(raw) % 2:
                return None
            return tuple(ord(char) for char in raw.decode("utf-16-be"))
        except (UnicodeDecodeError, ValueError):
            return None

    mapping: dict[bytes, tuple[int, ...]] = {}
    bfchar_pattern = re.compile(r"beginbfchar\s*(.*?)\s*endbfchar", re.DOTALL)
    entry_pattern = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
    for block_match in bfchar_pattern.finditer(text):
        for entry_match in entry_pattern.finditer(block_match.group(1)):
            try:
                source = bytes.fromhex(entry_match.group(1))
            except ValueError:
                continue
            destination = decode_destination(entry_match.group(2))
            if source and destination is not None:
                mapping[source] = destination

    bfrange_pattern = re.compile(r"beginbfrange\s*(.*?)\s*endbfrange", re.DOTALL)
    range_entry_pattern = re.compile(
        r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*"
        r"(?:<([0-9A-Fa-f]+)>|\[([^\]]*)\])"
    )
    array_element_pattern = re.compile(r"<([0-9A-Fa-f]+)>")
    for block_match in bfrange_pattern.finditer(text):
        for entry_match in range_entry_pattern.finditer(block_match.group(1)):
            start_hex, end_hex = entry_match.group(1), entry_match.group(2)
            if len(start_hex) != len(end_hex) or len(start_hex) % 2:
                continue
            try:
                start = int(start_hex, 16)
                end = int(end_hex, 16)
            except ValueError:
                continue
            if end < start:
                continue
            source_width = len(start_hex) // 2
            destinations: list[str]
            destination_start = entry_match.group(3)
            if destination_start is not None:
                width = len(destination_start)
                try:
                    value = int(destination_start, 16)
                except ValueError:
                    continue
                destinations = [
                    f"{value + offset:0{width}X}" for offset in range(end - start + 1)
                ]
            else:
                destinations = array_element_pattern.findall(entry_match.group(4))

            for offset, destination_hex in enumerate(destinations[: end - start + 1]):
                destination = decode_destination(destination_hex)
                if destination is None:
                    continue
                source = (start + offset).to_bytes(source_width, "big")
                mapping[source] = destination

    return mapping
