# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Validate and fix .notdef glyph presence in embedded fonts.

ISO 19005-2, Clause 6.3.3 requires every embedded font to contain a
.notdef glyph.  The subsetter already preserves .notdef for newly
embedded fonts, but fonts that were already embedded in the original
PDF are never validated.  This module adds that check and, when the
glyph is missing, inserts a minimal empty .notdef.
"""

import io
import logging
import struct
from collections.abc import Iterator

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf

from ..fonts.traversal import iter_all_page_fonts
from ..fonts.utils import safe_str as _safe_str
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)


def sanitize_font_notdef(pdf: Pdf) -> dict[str, int]:
    """Ensures every embedded font has a .notdef glyph.

    Iterates all embedded fonts, checks the glyph order for .notdef,
    and adds a minimal empty .notdef glyph when it is missing.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"notdef_fixed": N}``.
    """
    result: dict[str, int] = {"notdef_fixed": 0}

    for font_name, font_obj, font_file_key in _iter_all_embedded_fonts(pdf):
        try:
            if _fix_notdef(pdf, font_obj, font_file_key, font_name):
                result["notdef_fixed"] += 1
        except Exception as e:
            logger.debug(
                "Skipping .notdef validation for font %s: %s",
                font_name,
                e,
            )
            continue

    if result["notdef_fixed"] > 0:
        logger.info(
            "Font .notdef sanitization: %d fonts fixed",
            result["notdef_fixed"],
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_all_embedded_fonts(
    pdf: Pdf,
) -> Iterator[tuple[str, pikepdf.Object, str]]:
    """Yields ``(font_name, font_descriptor_owner, font_file_key)`` for each
    embedded font.

    *font_descriptor_owner* is the font dict (simple) or descendant CIDFont
    dict that owns the ``/FontDescriptor``.

    Skips Type3 fonts and fonts without embedded data.
    """
    seen_objgens: set[tuple[int, int]] = set()

    for page in pdf.pages:
        for _font_key, font_obj in iter_all_page_fonts(pikepdf.Page(page)):
            font = _resolve(font_obj)
            if not isinstance(font, Dictionary):
                continue

            objgen = font.objgen
            if objgen != (0, 0):
                if objgen in seen_objgens:
                    continue
                seen_objgens.add(objgen)

            subtype = font.get("/Subtype")
            if subtype is None:
                continue
            subtype_str = _safe_str(subtype)

            if subtype_str == "/Type3":
                continue

            if subtype_str == "/Type0":
                descendants = font.get("/DescendantFonts")
                if descendants is None:
                    continue
                descendants = _resolve(descendants)
                if not isinstance(descendants, Array) or len(descendants) == 0:
                    continue
                desc_font = _resolve(descendants[0])
                if not isinstance(desc_font, Dictionary):
                    continue
                font_file_key = _get_font_file_key(desc_font)
                if font_file_key is None:
                    continue
                font_name = _get_font_name(font)
                yield font_name, desc_font, font_file_key
            elif subtype_str in ("/Type1", "/TrueType", "/MMType1"):
                font_file_key = _get_font_file_key(font)
                if font_file_key is None:
                    continue
                font_name = _get_font_name(font)
                yield font_name, font, font_file_key


def _get_font_name(font: pikepdf.Object) -> str:
    base_font = font.get("/BaseFont")
    if base_font is not None:
        return _safe_str(base_font)[1:]
    return "Unknown"


def _get_font_file_key(font: pikepdf.Object) -> str | None:
    """Returns the FontDescriptor key holding embedded data, or ``None``."""
    fd = font.get("/FontDescriptor")
    if fd is None:
        return None
    fd = _resolve(fd)
    for key in ("/FontFile2", "/FontFile3", "/FontFile"):
        if fd.get(key) is not None:
            return key
    return None


def _fix_notdef(
    pdf: Pdf,
    font: pikepdf.Object,
    font_file_key: str,
    font_name: str,
) -> bool:
    """Checks a single font for .notdef and adds it if missing.

    Args:
        pdf: Opened pikepdf PDF object.
        font: Font dictionary (simple font or CIDFont descendant) that
            owns the ``/FontDescriptor``.
        font_file_key: One of ``/FontFile``, ``/FontFile2``, ``/FontFile3``.
        font_name: Human-readable font name for logging.

    Returns:
        ``True`` if the font was modified.
    """
    from fontTools.ttLib import TTFont

    fd = _resolve(font.get("/FontDescriptor"))
    stream = _resolve(fd.get(font_file_key))
    font_data = bytes(stream.read_bytes())

    try:
        tt_font = TTFont(io.BytesIO(font_data))
    except Exception:
        logger.debug("Font %s: cannot parse embedded font program", font_name)
        return False

    try:
        glyph_order = tt_font.getGlyphOrder()
        if ".notdef" in glyph_order:
            return False

        # .notdef is missing — add a minimal empty one
        logger.info("Font %s: adding missing .notdef glyph", font_name)
        _add_notdef_glyph(tt_font)

        # Serialize and write back
        buf = io.BytesIO()
        tt_font.save(buf)
        buf.seek(0)
        new_data = buf.read()

        new_stream = pdf.make_stream(new_data)
        # Preserve the original stream's metadata keys
        if font_file_key == "/FontFile2":
            new_stream[Name.Length1] = len(new_data)
        elif font_file_key == "/FontFile":
            new_stream[Name.Length1] = len(new_data)
        elif font_file_key == "/FontFile3":
            original_subtype = stream.get("/Subtype")
            if original_subtype is not None:
                new_stream[Name("/Subtype")] = original_subtype
        fd[Name(font_file_key)] = pdf.make_indirect(new_stream)

        # Inserting .notdef at GID 0 shifts every existing GID by +1.
        # Update CIDToGIDMap (both explicit streams and /Identity) to match.
        _update_cidtogidmap(pdf, font, len(glyph_order))

        return True
    finally:
        tt_font.close()


def _add_notdef_glyph(tt_font) -> None:
    """Inserts an empty .notdef glyph at position 0 in *tt_font*."""
    # Force decompilation of lazy-loaded tables *before* changing the
    # glyph order, otherwise fontTools may lose the name-to-glyph mapping.
    if "glyf" in tt_font:
        _ = list(tt_font["glyf"].keys())
    if "hmtx" in tt_font:
        _ = list(tt_font["hmtx"].metrics.keys())
    if "CFF " in tt_font:
        _ = tt_font["CFF "]

    glyph_order = tt_font.getGlyphOrder()

    # Insert .notdef at the front
    glyph_order.insert(0, ".notdef")
    tt_font.setGlyphOrder(glyph_order)

    # Add an empty glyph outline if glyf table exists (TrueType)
    if "glyf" in tt_font:
        from fontTools.ttLib.tables._g_l_y_f import Glyph

        tt_font["glyf"][".notdef"] = Glyph()

    # If CFF table exists (Type1/CFF)
    if "CFF " in tt_font:
        cff = tt_font["CFF "]
        top_dict = cff.cff.topDictIndex[0]
        char_strings = top_dict.CharStrings
        from fontTools.misc.psCharStrings import T2CharString

        cs = T2CharString()
        cs.program = [0, "hmoveto", "endchar"]
        char_strings[".notdef"] = cs

    # Add horizontal metrics (width 0, lsb 0)
    if "hmtx" in tt_font:
        tt_font["hmtx"][".notdef"] = (0, 0)

    # Update maxp numGlyphs
    if "maxp" in tt_font:
        tt_font["maxp"].numGlyphs = len(glyph_order)


def _update_cidtogidmap(
    pdf: Pdf, font: pikepdf.Object, original_glyph_count: int
) -> None:
    """Updates CIDToGIDMap after .notdef insertion at GID 0.

    When .notdef is inserted at GID 0, all existing GIDs shift by +1.
    Both explicit CIDToGIDMap streams and /Identity mappings must be
    updated to reflect this shift.
    """
    cidtogidmap = font.get("/CIDToGIDMap")
    if cidtogidmap is None:
        return
    resolved = _resolve(cidtogidmap)
    # /Identity means CID == GID — must replace with an explicit stream
    # where CID i → GID i+1 to account for the .notdef shift.
    if isinstance(resolved, Name):
        n = original_glyph_count
        shifted = [min(i + 1, 0xFFFF) for i in range(n)]
        new_data = struct.pack(f">{n}H", *shifted)
        new_stream = pdf.make_stream(new_data)
        font[Name("/CIDToGIDMap")] = pdf.make_indirect(new_stream)
        return
    data = bytes(resolved.read_bytes())
    if len(data) < 2 or len(data) % 2 != 0:
        return
    n = len(data) // 2
    gids = struct.unpack(f">{n}H", data)
    shifted = []
    for g in gids:
        if g + 1 > 0xFFFF:
            logger.warning(
                "CIDToGIDMap GID overflow: GID %d would exceed 0xFFFF "
                "after .notdef shift, clamping to 0xFFFF",
                g,
            )
            shifted.append(0xFFFF)
        else:
            shifted.append(g + 1)
    new_gids = struct.pack(f">{n}H", *shifted)
    new_stream = pdf.make_stream(new_gids)
    font[Name("/CIDToGIDMap")] = pdf.make_indirect(new_stream)
