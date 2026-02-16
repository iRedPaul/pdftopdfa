# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for font handling."""

import logging

import pikepdf

logger = logging.getLogger(__name__)


def obj_key(obj: pikepdf.Object) -> tuple[int, int] | None:
    """Returns a stable identity key for cycle detection.

    Uses pikepdf's objgen (object number, generation) which is stable
    across repeated accesses, unlike Python id() which can be reused
    for transient wrapper objects.

    Args:
        obj: A pikepdf object.

    Returns:
        The (obj_num, gen) tuple for indirect objects, or None for
        direct objects (which cannot form reference cycles).
    """
    try:
        og = obj.objgen
        if og != (0, 0):
            return og
    except Exception:
        pass
    return None


def check_visited(obj: pikepdf.Object, visited: set[tuple[int, int]]) -> bool:
    """Checks if an object has been visited and marks it if not.

    Args:
        obj: A pikepdf object to check.
        visited: Set of objgen tuples already visited.

    Returns:
        True if the object was already visited (should be skipped),
        False if it's new (and has now been added to visited).
    """
    key = obj_key(obj)
    if key is None:
        # Direct object — cannot form cycles, always process
        return False
    if key in visited:
        return True
    visited.add(key)
    return False


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
        embedding_allowed is False only for Restricted License (0x0002).
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

    if fstype & FSTYPE_NO_SUBSETTING:
        subsetting_allowed = False
        warnings.append("No subsetting allowed (fsType bit 8)")

    if fstype & FSTYPE_BITMAP_ONLY:
        warnings.append("Bitmap embedding only (fsType bit 9)")

    return embedding_allowed, subsetting_allowed, warnings
