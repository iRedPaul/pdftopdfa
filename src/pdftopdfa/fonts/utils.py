# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for font handling."""

import logging

import pikepdf

logger = logging.getLogger(__name__)

SYMBOL_CMAP_OFFSETS = (0x0000, 0xF000, 0xF100, 0xF200)


def safe_str(obj: pikepdf.Object, fallback: str = "Unknown") -> str:
    """Converts a pikepdf object to string, handling non-UTF-8 bytes.

    Args:
        obj: pikepdf object to convert.
        fallback: Value to return if conversion fails entirely.

    Returns:
        String representation of the object.
    """
    try:
        return str(obj)
    except (UnicodeDecodeError, UnicodeEncodeError):
        try:
            return bytes(obj).decode("latin-1")
        except Exception:
            return fallback


# fsType bit masks (OpenType OS/2 table)
FSTYPE_RESTRICTED_LICENSE = 0x0002
FSTYPE_PREVIEW_AND_PRINT = 0x0004
FSTYPE_EDITABLE = 0x0008
FSTYPE_NO_SUBSETTING = 0x0100
FSTYPE_BITMAP_ONLY = 0x0200


def get_fstype(font_data: bytes) -> int | None:
    """Extracts the fsType embedding permission field from font data.

    Reads the OS/2 table from TrueType/OpenType font data and returns
    the fsType value that defines embedding restrictions.

    Args:
        font_data: Raw TrueType or OpenType font bytes.

    Returns:
        The fsType value as an integer, or None if the OS/2 table
        is not present or cannot be read.
    """
    try:
        from io import BytesIO

        from fontTools.ttLib import TTFont

        tt_font = TTFont(BytesIO(font_data))
        try:
            os2_table = tt_font.get("OS/2")
            if os2_table is None:
                return None
            return os2_table.fsType
        finally:
            tt_font.close()
    except Exception:
        return None


def get_any_cmap(tt_font) -> dict[int, str]:
    """Returns the most useful character map available from a font.

    Tries ``getBestCmap()`` first; for fonts where that yields nothing
    (symbol fonts, Mac-only fonts) falls back to Windows Symbol (3,0),
    then Mac Roman (1,0), then any non-empty subtable.

    Args:
        tt_font: fonttools TTFont object.

    Returns:
        Mapping of codepoint to glyph name; empty dict if none available.
    """
    try:
        cmap = tt_font.getBestCmap()
    except KeyError:
        cmap = None
    if cmap:
        return cmap
    if "cmap" not in tt_font:
        return {}
    cmap_table = tt_font["cmap"]
    for subtable in cmap_table.tables:
        mapping = getattr(subtable, "cmap", None)
        if subtable.platformID == 3 and subtable.platEncID == 0 and mapping:
            return mapping
    for subtable in cmap_table.tables:
        mapping = getattr(subtable, "cmap", None)
        if subtable.platformID == 1 and subtable.platEncID == 0 and mapping:
            return mapping
    for subtable in cmap_table.tables:
        mapping = getattr(subtable, "cmap", None)
        if mapping:
            return mapping
    return {}


def symbol_cmap_code_to_byte(code: int) -> int | None:
    """Returns the PDF byte represented by a Microsoft Symbol cmap key."""
    if 0 <= code <= 0xFF:
        return code
    for offset in SYMBOL_CMAP_OFFSETS[1:]:
        if offset <= code <= offset + 0xFF:
            return code - offset
    return None


def resolve_symbol_cmap_glyph(cmap: dict[int, str], code: int) -> str | None:
    """Resolves a PDF byte through the four valid Microsoft Symbol ranges."""
    if not 0 <= code <= 0xFF:
        return None
    parsed = parse_symbol_cmap(cmap)
    return parsed[1].get(code) if parsed is not None else None


def parse_symbol_cmap(cmap: dict[int, str]) -> tuple[int, dict[int, str]] | None:
    """Returns the single unambiguous byte range used by a Symbol cmap."""
    if any(symbol_cmap_code_to_byte(code) is None for code in cmap):
        return None
    ranges: list[tuple[int, dict[int, str]]] = []
    for offset in SYMBOL_CMAP_OFFSETS:
        mapping = {
            code - offset: glyph_name
            for code, glyph_name in cmap.items()
            if offset <= code <= offset + 0xFF
        }
        if mapping:
            ranges.append((offset, mapping))
    if not ranges:
        return None

    if len(ranges) != 1:
        return None
    return ranges[0]


def symbol_cmap_to_byte_mapping(cmap: dict[int, str]) -> dict[int, str]:
    """Converts an unambiguous Microsoft Symbol cmap to PDF byte mappings."""
    parsed = parse_symbol_cmap(cmap)
    return parsed[1] if parsed is not None else {}


def get_truetype_byte_encoding(
    tt_font,
) -> tuple[tuple[int, int], int, dict[int, str]] | None:
    """Returns the byte cmap used by a TrueType font without PDF Encoding.

    Microsoft Symbol (3,0) is preferred over Macintosh Roman (1,0), matching
    the TrueType glyph-selection algorithm in ISO 32000. The result contains
    the platform/encoding IDs, the Microsoft Symbol range offset (zero for a
    Macintosh cmap), and the normalized byte-to-glyph mapping.
    """
    cmap_table = tt_font.get("cmap")
    if cmap_table is None:
        return None

    symbol_encodings: list[tuple[int, dict[int, str]]] = []
    for subtable in cmap_table.tables:
        mapping = getattr(subtable, "cmap", None)
        if subtable.platformID == 3 and subtable.platEncID == 0 and mapping:
            parsed = parse_symbol_cmap(mapping)
            if parsed is None:
                return None
            symbol_encodings.append(parsed)
    if symbol_encodings:
        offset, byte_mapping = symbol_encodings[0]
        if any(candidate != symbol_encodings[0] for candidate in symbol_encodings[1:]):
            return None
        return (3, 0), offset, byte_mapping

    mac_encodings: list[dict[int, str]] = []
    for subtable in cmap_table.tables:
        mapping = getattr(subtable, "cmap", None)
        if subtable.platformID == 1 and subtable.platEncID == 0 and mapping:
            if any(not 0 <= code <= 0xFF for code in mapping):
                return None
            byte_mapping = {
                code: glyph_name
                for code, glyph_name in mapping.items()
                if 0 <= code <= 0xFF
            }
            if byte_mapping:
                mac_encodings.append(byte_mapping)
    if mac_encodings:
        byte_mapping = mac_encodings[0]
        if any(candidate != byte_mapping for candidate in mac_encodings[1:]):
            return None
        return (1, 0), 0, byte_mapping

    return None


def get_encoding_name(encoding: pikepdf.Object) -> str:
    """Extracts the encoding name from a Name or CMap Stream.

    For Name objects (e.g. /Identity-H), returns the name without
    the leading slash. For CMap streams, reads /CMapName from the
    stream dictionary.

    Args:
        encoding: pikepdf Name or Stream object.

    Returns:
        The encoding name string, or empty string if not extractable.
    """
    if isinstance(encoding, pikepdf.Name):
        return safe_str(encoding).lstrip("/")
    # CMap stream: extract /CMapName from the stream dictionary
    try:
        cmap_name = encoding.get("/CMapName")
        if cmap_name is not None:
            return safe_str(cmap_name).lstrip("/")
    except Exception:
        pass
    return ""


def check_fstype_restrictions(
    fstype: int,
) -> tuple[bool, bool, list[str]]:
    """Checks fsType for embedding and subsetting restrictions.

    Args:
        fstype: The fsType value from the OS/2 table.

    Returns:
        Tuple of (embedding_allowed, subsetting_allowed, warnings).
        embedding_allowed is False for Restricted License (0x0002) and
        Bitmap-only embedding (0x0200).
        subsetting_allowed is False if the No Subsetting bit (0x0100) is set.
        warnings contains human-readable descriptions of restrictions found.
    """
    warnings: list[str] = []
    embedding_allowed = True
    subsetting_allowed = True

    if fstype & FSTYPE_RESTRICTED_LICENSE:
        embedding_allowed = False
        warnings.append("Restricted License embedding (fsType bit 1)")

    if fstype & FSTYPE_PREVIEW_AND_PRINT:
        warnings.append("Preview & Print embedding only (fsType bit 2)")

    if fstype & FSTYPE_EDITABLE:
        warnings.append("Editable embedding allowed (fsType bit 3)")

    if fstype & FSTYPE_NO_SUBSETTING:
        subsetting_allowed = False
        warnings.append("No subsetting allowed (fsType bit 8)")

    if fstype & FSTYPE_BITMAP_ONLY:
        embedding_allowed = False
        warnings.append("Bitmap embedding only (fsType bit 9)")

    return embedding_allowed, subsetting_allowed, warnings


def is_permitted_fstype_notice(message: str) -> bool:
    """Return whether an fsType message is informational only.

    ``Preview & Print`` and ``Editable`` permit outline embedding, so they
    should be logged for traceability without surfacing as conversion warnings.
    """
    return "Preview & Print" in message or "Editable" in message
