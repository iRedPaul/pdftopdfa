# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Repair truncated metric tables in embedded font programs.

Some producers embed TrueType programs whose ``hmtx`` table is shorter than
``hhea.numberOfHMetrics`` and ``maxp.numGlyphs`` require, typically because
the trailing left side bearings were dropped.  veraPDF still reads the
advance widths from such a table, but fontTools refuses to decompile it, so
the .notdef, glyph coverage and width sanitizers silently skip the font and
mismatches between the font dictionary and the font program survive
(ISO 19005-2, rule 6.2.11.5).

When every advance width is present, padding the missing side bearings
restores a well-formed table without changing any glyph advance.
"""

import io
import logging
import struct
from collections.abc import Iterator

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf

from ..fonts.traversal import iter_acroform_dr_fonts, iter_all_page_fonts
from ..fonts.utils import safe_str as _safe_str
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Font file streams that can hold an sfnt program with an hmtx table
_SFNT_FONT_FILE_KEYS = ("/FontFile2", "/FontFile3")


def sanitize_font_programs(pdf: Pdf) -> dict[str, int]:
    """Repairs truncated ``hmtx`` tables in all embedded sfnt font programs.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"font_hmtx_repaired": N}``.
    """
    result: dict[str, int] = {"font_hmtx_repaired": 0}

    for font_name, stream, font_file_key in _iter_embedded_sfnt_streams(pdf):
        try:
            new_data = _pad_truncated_hmtx(bytes(stream.read_bytes()))
            if new_data is None:
                continue
            # Rewrite in place so every descriptor sharing the program sees it
            stream.write(new_data)
            if font_file_key == "/FontFile2":
                stream[Name.Length1] = len(new_data)
        except Exception as e:
            log_suppressed_error(
                logger,
                e,
                "Skipping font program repair for font %s: %s",
                font_name,
                e,
            )
            continue
        logger.info("Font %s: padded truncated hmtx table", font_name)
        result["font_hmtx_repaired"] += 1

    return result


def _iter_embedded_sfnt_streams(
    pdf: Pdf,
) -> Iterator[tuple[str, pikepdf.Stream, str]]:
    """Yields ``(font_name, font_file_stream, font_file_key)`` once per stream."""
    seen_objgens: set[tuple[int, int]] = set()

    for font_obj in _iter_font_objects(pdf):
        try:
            font = _resolve(font_obj)
            if not isinstance(font, Dictionary):
                continue
            owner = font
            if _safe_str(font.get("/Subtype", Name.Null)) == "/Type0":
                descendants = _resolve(font.get("/DescendantFonts"))
                if not isinstance(descendants, Array) or len(descendants) == 0:
                    continue
                owner = _resolve(descendants[0])
                if not isinstance(owner, Dictionary):
                    continue
            fd = _resolve(owner.get("/FontDescriptor"))
            if not isinstance(fd, Dictionary):
                continue
            for key in _SFNT_FONT_FILE_KEYS:
                stream = _resolve(fd.get(key))
                if not isinstance(stream, pikepdf.Stream):
                    continue
                objgen = stream.objgen
                if objgen != (0, 0):
                    if objgen in seen_objgens:
                        continue
                    seen_objgens.add(objgen)
                yield _get_font_name(font), stream, key
        except Exception as e:
            log_suppressed_error(logger, e, "Skipping broken font object: %s", e)
            continue


def _iter_font_objects(pdf: Pdf) -> Iterator[pikepdf.Object]:
    """Yields page, nested-resource and AcroForm /DR font objects."""
    for page in pdf.pages:
        for _font_key, font_obj in iter_all_page_fonts(pikepdf.Page(page)):
            yield font_obj
    for _font_key, font_obj in iter_acroform_dr_fonts(pdf):
        yield font_obj


def _get_font_name(font: pikepdf.Object) -> str:
    base_font = font.get("/BaseFont")
    if base_font is not None:
        return _safe_str(base_font)[1:]
    return "Unknown"


def _pad_truncated_hmtx(font_data: bytes) -> bytes | None:
    """Returns *font_data* with the missing trailing hmtx side bearings added.

    Each missing left side bearing is set to the glyph's ``xMin``, which is
    what TrueType rasterizers expect.  All other tables are copied verbatim.

    Returns:
        The repaired program, or ``None`` when the data is not a parsable
        sfnt, its ``hmtx`` table is complete, or advance widths themselves are
        missing (padding would have to invent them).
    """
    from fontTools.ttLib import TTFont
    from fontTools.ttLib.sfnt import SFNTWriter

    try:
        tt_font = TTFont(io.BytesIO(font_data), lazy=True)
    except Exception:
        return None

    try:
        reader = tt_font.reader
        if not all(tag in reader for tag in ("hmtx", "hhea", "maxp")):
            return None

        hmtx_data = reader["hmtx"]
        num_glyphs = tt_font["maxp"].numGlyphs
        num_metrics = min(int(tt_font["hhea"].numberOfHMetrics), num_glyphs)
        long_metrics_size = 4 * num_metrics
        if len(hmtx_data) >= long_metrics_size + 2 * (num_glyphs - num_metrics):
            return None
        if num_metrics == 0 or len(hmtx_data) < long_metrics_size:
            return None

        present = (len(hmtx_data) - long_metrics_size) // 2
        try:
            glyf = tt_font["glyf"] if "glyf" in tt_font else None
        except Exception:
            glyf = None
        glyph_order = tt_font.getGlyphOrder()
        padding = [
            struct.pack(">h", _left_side_bearing(glyf, glyph_order, gid))
            for gid in range(num_metrics + present, num_glyphs)
        ]
        hmtx_data = hmtx_data[: long_metrics_size + 2 * present] + b"".join(padding)

        # Keep the original table order; SFNTWriter recomputes the checksums.
        tags = sorted(reader.keys(), key=lambda tag: reader.tables[tag].offset)
        buf = io.BytesIO()
        writer = SFNTWriter(buf, len(tags), reader.sfntVersion)
        for tag in tags:
            writer[tag] = hmtx_data if tag == "hmtx" else reader[tag]
        writer.close()
        return buf.getvalue()
    finally:
        tt_font.close()


def _left_side_bearing(glyf, glyph_order: list[str], gid: int) -> int:
    """Returns the glyph's ``xMin``, or 0 for empty or non-glyf glyphs."""
    if glyf is None or gid >= len(glyph_order):
        return 0
    try:
        x_min = int(getattr(glyf[glyph_order[gid]], "xMin", 0))
    except Exception:
        return 0
    return max(-32768, min(32767, x_min))
