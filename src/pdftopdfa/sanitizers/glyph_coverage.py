# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Ensure all referenced glyphs exist in embedded fonts.

ISO 19005-2, Clause 6.2.11.4.1 requires that embedded fonts define all
glyphs referenced for rendering within the conforming file.  When a
content stream references a GID that does not exist in the font program
(e.g. because the original font was incomplete or the subsetter skipped
out-of-range GIDs), veraPDF reports rule 6.2.11.4.1 failure.

This module adds minimal empty glyph outlines for any referenced but
missing glyphs, ensuring full glyph coverage.
"""

import io
import logging
import struct

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf

from ..fonts.glyph_usage import collect_font_usage
from ..fonts.tounicode import parse_cidtogidmap_stream
from ..fonts.traversal import iter_all_page_fonts
from ..fonts.utils import safe_str as _safe_str
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)


def sanitize_glyph_coverage(pdf: Pdf) -> dict[str, int]:
    """Adds empty glyph outlines for referenced but missing glyphs.

    Iterates all embedded fonts and checks whether every glyph
    referenced in content streams actually exists in the font program.
    Missing glyphs are filled with minimal empty outlines.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"glyphs_added": N}``.
    """
    result: dict[str, int] = {"glyphs_added": 0}

    # Collect character codes used with each font across the PDF
    usage = collect_font_usage(pdf)
    if not usage:
        return result

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

            # Skip fonts with no recorded usage
            if objgen not in usage:
                continue
            used_codes = usage[objgen]
            if not used_codes:
                continue

            subtype = font.get("/Subtype")
            if subtype is None:
                continue
            subtype_str = _safe_str(subtype)

            if subtype_str == "/Type3":
                continue

            try:
                if subtype_str == "/Type0":
                    added = _process_type0_font(pdf, font, used_codes)
                elif subtype_str in ("/Type1", "/TrueType", "/MMType1"):
                    added = _process_simple_font(pdf, font, used_codes)
                else:
                    continue
                result["glyphs_added"] += added
            except Exception as e:
                font_name = _get_font_name(font)
                logger.debug(
                    "Skipping glyph coverage for font %s: %s",
                    font_name,
                    e,
                )

    if result["glyphs_added"] > 0:
        logger.info(
            "Glyph coverage: %d missing glyphs added",
            result["glyphs_added"],
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _process_type0_font(
    pdf: Pdf, type0_font: pikepdf.Object, used_codes: set[int]
) -> int:
    """Processes a Type0 (CID) font for glyph coverage.

    Maps used character codes (CIDs) to GIDs via the CIDToGIDMap,
    then checks whether those GIDs exist in the font program.

    Returns:
        Number of missing referenced glyphs that were added.
    """
    descendants = type0_font.get("/DescendantFonts")
    if descendants is None:
        return 0
    descendants = _resolve(descendants)
    if not isinstance(descendants, Array) or len(descendants) == 0:
        return 0
    desc_font = _resolve(descendants[0])
    if not isinstance(desc_font, Dictionary):
        return 0

    font_file_key = _get_font_file_key(desc_font)
    if font_file_key is None:
        return 0

    font_name = _get_font_name(type0_font)

    # CIDFontType0 (CFF CID-keyed): CIDs map via CFF charset, not GIDs
    cidfont_subtype = desc_font.get("/Subtype")
    if cidfont_subtype is not None and _safe_str(cidfont_subtype) == "/CIDFontType0":
        return _fix_missing_cids_in_cff(
            pdf, desc_font, font_file_key, font_name, used_codes
        )

    # Determine CID-to-GID mapping
    cid_to_gid_obj = desc_font.get("/CIDToGIDMap")
    cid_to_gid: dict[int, int] | None = None

    if cid_to_gid_obj is not None:
        cid_to_gid_obj = _resolve(cid_to_gid_obj)
        if isinstance(cid_to_gid_obj, Name):
            if _safe_str(cid_to_gid_obj) == "/Identity":
                cid_to_gid = None  # Identity: CID == GID
            else:
                return 0
        elif isinstance(cid_to_gid_obj, pikepdf.Stream):
            stream_data = bytes(cid_to_gid_obj.read_bytes())
            cid_to_gid = parse_cidtogidmap_stream(stream_data)
        else:
            return 0
    # If CIDToGIDMap is absent, treat as Identity (cid_to_gid stays None)

    # Map used character codes (CIDs) to GIDs
    referenced_gids: set[int] = set()
    for code in used_codes:
        if cid_to_gid is not None:
            gid = cid_to_gid.get(code)
            if gid is not None:
                referenced_gids.add(gid)
        else:
            # Identity mapping: CID == GID
            referenced_gids.add(code)

    if not referenced_gids:
        return 0

    # Parse /W and /DW to build target widths for new glyphs
    target_widths = _parse_cidfont_target_widths(desc_font)

    return _fix_missing_gids(
        pdf,
        desc_font,
        font_file_key,
        font_name,
        referenced_gids,
        target_widths=target_widths,
    )


def _process_simple_font(pdf: Pdf, font: pikepdf.Object, used_codes: set[int]) -> int:
    """Processes a simple font for glyph coverage.

    Maps used character codes through the font encoding to glyph names,
    then checks whether those glyph names exist in the font program.

    Returns:
        Number of missing referenced glyphs that were added.
    """
    font_file_key = _get_font_file_key(font)
    if font_file_key is None:
        return 0

    font_name = _get_font_name(font)

    # Resolve encoding to get code -> glyph_name mapping
    encoding = _resolve_encoding(font)
    if not encoding:
        return 0

    fd = _resolve(font.get("/FontDescriptor"))
    stream = _resolve(fd.get(font_file_key))
    font_data = bytes(stream.read_bytes())

    from fontTools.ttLib import TTFont

    is_bare_cff = False
    try:
        tt_font = TTFont(io.BytesIO(font_data))
    except Exception:
        # Try wrapping bare CFF (Type1C) in OTF container
        if font_file_key == "/FontFile3":
            try:
                otf_data = _wrap_cff_in_otf(font_data)
                tt_font = TTFont(io.BytesIO(otf_data))
                is_bare_cff = True
            except Exception:
                logger.debug("Font %s: cannot parse font program", font_name)
                return 0
        elif font_file_key == "/FontFile":
            # Type1 PFA/PFB — handle separately
            return _process_type1_simple_font(
                pdf,
                font,
                fd,
                font_name,
                font_data,
                encoding,
                used_codes,
            )
        else:
            logger.debug("Font %s: cannot parse font program", font_name)
            return 0

    try:
        glyph_order = tt_font.getGlyphOrder()
        glyph_set = set(glyph_order)

        missing_names: list[str] = []
        missing_widths: dict[str, int] = {}
        # Get PDF Widths array for width consistency
        first_char = int(font.get("/FirstChar", 0))
        widths_arr = font.get("/Widths")
        if widths_arr is not None:
            widths_arr = _resolve(widths_arr)

        for code in sorted(used_codes):
            name = encoding.get(code)
            if name and name not in glyph_set:
                missing_names.append(name)
                if widths_arr is not None:
                    idx = code - first_char
                    if 0 <= idx < len(widths_arr):
                        missing_widths[name] = int(widths_arr[idx])

        if not missing_names:
            return 0

        logger.info(
            "Font %s: adding %d missing glyphs by name",
            font_name,
            len(missing_names),
        )

        _add_empty_glyphs_by_name(tt_font, missing_names, missing_widths)
        if is_bare_cff:
            _write_back_cff(pdf, fd, tt_font)
        else:
            _write_back_font(pdf, fd, font_file_key, tt_font)

        # Remove stale CharSet — it no longer lists all glyphs (6.2.11.4.2)
        if "/CharSet" in fd:
            del fd[Name("/CharSet")]

        return len(missing_names)
    except Exception as e:
        logger.debug("Font %s: error fixing glyph coverage: %s", font_name, e)
        return 0
    finally:
        tt_font.close()


def _resolve_encoding(font: pikepdf.Object) -> dict[int, str] | None:
    """Resolves a simple font's encoding to a code-to-glyph-name mapping."""
    from ..fonts.subsetter import _resolve_simple_font_encoding

    try:
        return _resolve_simple_font_encoding(font)
    except Exception:
        return None


def _parse_cidfont_target_widths(
    desc_font: pikepdf.Object,
) -> dict[int, int]:
    """Parses /W and /DW from a CIDFont to build a GID-to-PDF-width mapping.

    For Identity CIDToGIDMap (or absent), CID == GID, so the mapping
    is directly usable.  For explicit CIDToGIDMap, callers should
    already have resolved CID→GID; here we treat CIDs as GIDs for
    the width lookup (works for Identity which is the common case).

    Returns:
        Dictionary mapping GID (=CID) to declared PDF width (1000-unit space).
    """
    dw = int(desc_font.get("/DW", 1000))
    widths: dict[int, int] = {}

    w_arr = desc_font.get("/W")
    if w_arr is not None:
        w_arr = _resolve(w_arr)
        i = 0
        while i < len(w_arr):
            start = int(w_arr[i])
            i += 1
            if i >= len(w_arr):
                break
            next_val = _resolve(w_arr[i])
            if isinstance(next_val, Array):
                for j, w in enumerate(next_val):
                    widths[start + j] = int(w)
                i += 1
            else:
                end = int(next_val)
                i += 1
                if i >= len(w_arr):
                    break
                width = int(w_arr[i])
                for c in range(start, end + 1):
                    widths[c] = width
                i += 1

    # Store the default width for GIDs not explicitly listed
    widths.setdefault(-1, dw)  # sentinel key for DW
    return widths


def _fix_missing_gids(
    pdf: Pdf,
    desc_font: pikepdf.Object,
    font_file_key: str,
    font_name: str,
    referenced_gids: set[int],
    *,
    target_widths: dict[int, int] | None = None,
) -> int:
    """Fixes missing GIDs in a CIDFont's embedded font program.

    Identifies GIDs that are referenced in content streams but do not
    exist in the font's glyph order, and adds empty glyph outlines
    for them (plus any intermediate positions to keep the order
    contiguous).

    Returns:
        Number of missing referenced GIDs that were fixed.
    """
    from fontTools.ttLib import TTFont

    fd = _resolve(desc_font.get("/FontDescriptor"))
    stream = _resolve(fd.get(font_file_key))
    font_data = bytes(stream.read_bytes())

    is_bare_cff = False
    try:
        tt_font = TTFont(io.BytesIO(font_data))
    except Exception:
        # Try wrapping bare CFF (CIDFontType0C / Type1C) in OTF
        if font_file_key == "/FontFile3":
            try:
                otf_data = _wrap_cff_in_otf(font_data)
                tt_font = TTFont(io.BytesIO(otf_data))
                is_bare_cff = True
            except Exception:
                logger.debug("Font %s: cannot parse font program", font_name)
                return 0
        else:
            logger.debug("Font %s: cannot parse font program", font_name)
            return 0

    try:
        glyph_order = tt_font.getGlyphOrder()
        num_glyphs = len(glyph_order)

        # Find GIDs that are referenced but beyond the glyph order
        missing_gids = sorted(gid for gid in referenced_gids if gid >= num_glyphs)

        if not missing_gids:
            return 0

        logger.info(
            "Font %s: adding %d missing glyphs "
            "(max referenced GID: %d, font has %d glyphs)",
            font_name,
            len(missing_gids),
            max(referenced_gids),
            num_glyphs,
        )

        _add_empty_glyphs_by_gid(
            tt_font,
            missing_gids,
            num_glyphs,
            target_widths=target_widths,
        )
        if is_bare_cff:
            _write_back_cff(pdf, fd, tt_font)
        else:
            _write_back_font(pdf, fd, font_file_key, tt_font)

        return len(missing_gids)
    except Exception as e:
        logger.debug("Font %s: error fixing glyph coverage: %s", font_name, e)
        return 0
    finally:
        tt_font.close()


def _fix_missing_cids_in_cff(
    pdf: Pdf,
    desc_font: pikepdf.Object,
    font_file_key: str,
    font_name: str,
    used_cids: set[int],
) -> int:
    """Fixes missing CIDs in a CID-keyed CFF font program.

    For CIDFontType0 fonts, CIDs map to glyphs via the CFF charset
    (e.g. ``cid00041``), not by GID position. This checks which CIDs
    are present in the charset and adds empty charstrings for any
    missing ones.

    Returns:
        Number of missing CID glyphs added.
    """
    from fontTools.misc.psCharStrings import T2CharString
    from fontTools.ttLib import TTFont

    fd = _resolve(desc_font.get("/FontDescriptor"))
    stream = _resolve(fd.get(font_file_key))
    font_data = bytes(stream.read_bytes())

    is_bare_cff = False
    try:
        tt_font = TTFont(io.BytesIO(font_data))
    except Exception:
        if font_file_key == "/FontFile3":
            try:
                otf_data = _wrap_cff_in_otf(font_data)
                tt_font = TTFont(io.BytesIO(otf_data))
                is_bare_cff = True
            except Exception:
                return 0
        else:
            return 0

    try:
        if "CFF " not in tt_font:
            return 0

        cff = tt_font["CFF "]
        top_dict = cff.cff.topDictIndex[0]
        char_strings = top_dict.CharStrings

        # Build set of CIDs present in the CFF charset
        existing_cids: set[int] = set()
        for name in char_strings.keys():
            if name == ".notdef":
                existing_cids.add(0)
            elif name.startswith("cid"):
                try:
                    existing_cids.add(int(name[3:]))
                except ValueError:
                    pass

        missing_cids = sorted(cid for cid in used_cids if cid not in existing_cids)
        if not missing_cids:
            return 0

        logger.info(
            "Font %s: adding %d missing CID glyphs in CFF",
            font_name,
            len(missing_cids),
        )

        # Determine Private dict for new charstrings
        if hasattr(top_dict, "FDArray") and top_dict.FDArray:
            private = top_dict.FDArray[0].Private
        else:
            private = top_dict.Private

        # Get DW for width consistency
        dw = int(desc_font.get("/DW", 1000))
        # Parse W array for specific CID widths
        cid_widths: dict[int, int] = {}
        w_arr = desc_font.get("/W")
        if w_arr is not None:
            w_arr = _resolve(w_arr)
            i = 0
            arr_len = len(w_arr)
            while i < arr_len:
                if i + 1 >= arr_len:
                    break
                start = int(w_arr[i])
                next_val = _resolve(w_arr[i + 1])
                if isinstance(next_val, Array):
                    for j, w in enumerate(next_val):
                        cid_widths[start + j] = int(w)
                    i += 2
                else:
                    if i + 2 >= arr_len:
                        break
                    end = int(next_val)
                    width = int(w_arr[i + 2])
                    for c in range(start, end + 1):
                        cid_widths[c] = width
                    i += 3

        nominal_w = getattr(private, "nominalWidthX", 0)
        default_w = getattr(private, "defaultWidthX", 0)

        glyph_order = tt_font.getGlyphOrder()

        for cid in missing_cids:
            cid_name = f"cid{cid:05d}"
            glyph_order.append(cid_name)
            target_w = cid_widths.get(cid, dw)
            cs = T2CharString()
            if target_w != default_w:
                cs.program = [target_w - nominal_w, 0, "hmoveto", "endchar"]
            else:
                cs.program = [0, "hmoveto", "endchar"]
            cs.private = private
            new_idx = len(char_strings.charStringsIndex)
            char_strings.charStrings[cid_name] = new_idx
            char_strings.charStringsIndex.append(cs)

        tt_font.setGlyphOrder(glyph_order)
        if "maxp" in tt_font:
            tt_font["maxp"].numGlyphs = len(glyph_order)

        if is_bare_cff:
            _write_back_cff(pdf, fd, tt_font)
        else:
            _write_back_font(pdf, fd, font_file_key, tt_font)

        return len(missing_cids)
    except Exception as e:
        logger.debug("Font %s: error fixing CID glyph coverage: %s", font_name, e)
        return 0
    finally:
        tt_font.close()


def _add_empty_glyphs_by_gid(
    tt_font,
    missing_gids: list[int],
    current_count: int,
    *,
    target_widths: dict[int, int] | None = None,
) -> None:
    """Adds empty glyphs at specified GID positions.

    Extends the glyph order up to the maximum referenced GID and
    fills all new positions with empty glyph outlines.  Intermediate
    positions (not in *missing_gids*) also receive empty glyphs to
    keep the glyph order contiguous.

    Args:
        tt_font: fontTools TTFont object.
        missing_gids: Sorted list of GIDs to add.
        current_count: Current number of glyphs in the font.
        target_widths: Optional mapping of GID to declared PDF width
            (1000-unit space).  Key -1 holds the /DW default width.
    """
    if not missing_gids:
        return

    max_gid = max(missing_gids)

    # Force decompilation of lazy-loaded tables before modifying
    if "glyf" in tt_font:
        _ = list(tt_font["glyf"].keys())
    if "hmtx" in tt_font:
        _ = list(tt_font["hmtx"].metrics.keys())
    if "CFF " in tt_font:
        _ = tt_font["CFF "]

    glyph_order = tt_font.getGlyphOrder()

    # Determine if this is a CID-keyed CFF font
    is_cid_cff = False
    if "CFF " in tt_font:
        cff = tt_font["CFF "]
        top_dict = cff.cff.topDictIndex[0]
        is_cid_cff = hasattr(top_dict, "FDArray") and top_dict.FDArray is not None

    # Extend glyph order to accommodate max_gid
    for gid in range(current_count, max_gid + 1):
        if is_cid_cff:
            glyph_name = f"cid{gid:05d}"
        else:
            glyph_name = f"glyph{gid:05d}"
        glyph_order.append(glyph_name)
    tt_font.setGlyphOrder(glyph_order)

    # Add empty glyph outlines for all new positions
    if "glyf" in tt_font:
        from fontTools.ttLib.tables._g_l_y_f import Glyph

        for gid in range(current_count, max_gid + 1):
            tt_font["glyf"][glyph_order[gid]] = Glyph()

    # Compute font-unit scale factor for target widths (PDF 1000-unit → font units)
    units_per_em = tt_font["head"].unitsPerEm if "head" in tt_font else 1000
    dw_pdf = 0  # default PDF width for GIDs not in target_widths
    if target_widths is not None:
        dw_pdf = target_widths.get(-1, 1000)

    def _get_font_units_width(gid: int) -> int:
        """Returns the advance width in font units for a given GID."""
        if target_widths is None:
            return 0
        pdf_w = target_widths.get(gid, dw_pdf)
        return round(pdf_w * units_per_em / 1000)

    if "CFF " in tt_font:
        from fontTools.misc.psCharStrings import T2CharString

        char_strings = top_dict.CharStrings
        if is_cid_cff:
            # CID-keyed CFF: use FDArray[0]'s Private for charstrings
            private = top_dict.FDArray[0].Private
        else:
            private = top_dict.Private
        nominal_w = getattr(private, "nominalWidthX", 0)
        default_w = getattr(private, "defaultWidthX", 0)
        for gid in range(current_count, max_gid + 1):
            cs = T2CharString()
            fw = _get_font_units_width(gid)
            if fw != default_w:
                cs.program = [fw - nominal_w, 0, "hmoveto", "endchar"]
            else:
                cs.program = [0, "hmoveto", "endchar"]
            cs.private = private
            new_idx = len(char_strings.charStringsIndex)
            char_strings.charStrings[glyph_order[gid]] = new_idx
            char_strings.charStringsIndex.append(cs)

    # Add horizontal metrics with target widths (in font units)
    if "hmtx" in tt_font:
        for gid in range(current_count, max_gid + 1):
            fw = _get_font_units_width(gid)
            tt_font["hmtx"][glyph_order[gid]] = (fw, 0)

    # Update glyph count
    if "maxp" in tt_font:
        tt_font["maxp"].numGlyphs = len(glyph_order)


def _add_empty_glyphs_by_name(
    tt_font,
    missing_names: list[str],
    target_widths: dict[str, int] | None = None,
) -> None:
    """Adds empty glyphs with the specified names to the font program.

    Args:
        tt_font: fontTools TTFont object.
        missing_names: Glyph names to add.
        target_widths: Optional mapping of glyph name to target width
            (from the PDF Widths array) to ensure width consistency.
    """
    if not missing_names:
        return

    if target_widths is None:
        target_widths = {}

    # Force decompilation of lazy-loaded tables before modifying
    if "glyf" in tt_font:
        _ = list(tt_font["glyf"].keys())
    if "hmtx" in tt_font:
        _ = list(tt_font["hmtx"].metrics.keys())
    if "CFF " in tt_font:
        _ = tt_font["CFF "]

    glyph_order = tt_font.getGlyphOrder()

    for name in missing_names:
        glyph_order.append(name)
    tt_font.setGlyphOrder(glyph_order)

    if "glyf" in tt_font:
        from fontTools.ttLib.tables._g_l_y_f import Glyph

        for name in missing_names:
            tt_font["glyf"][name] = Glyph()

    # Scale factor: PDF widths are in 1000-unit space, font tables use font units
    units_per_em = tt_font["head"].unitsPerEm if "head" in tt_font else 1000

    def _to_font_units(pdf_width: int) -> int:
        return round(pdf_width * units_per_em / 1000)

    if "CFF " in tt_font:
        from fontTools.misc.psCharStrings import T2CharString

        cff = tt_font["CFF "]
        top_dict = cff.cff.topDictIndex[0]
        char_strings = top_dict.CharStrings
        private = top_dict.Private
        nominal_w = getattr(private, "nominalWidthX", 0)
        default_w = getattr(private, "defaultWidthX", 0)
        for name in missing_names:
            cs = T2CharString()
            w_pdf = target_widths.get(name)
            if w_pdf is not None:
                w_font = _to_font_units(w_pdf)
                if w_font != default_w:
                    # Encode explicit width delta from nominalWidthX
                    cs.program = [w_font - nominal_w, 0, "hmoveto", "endchar"]
                else:
                    cs.program = [0, "hmoveto", "endchar"]
            else:
                cs.program = [0, "hmoveto", "endchar"]
            cs.private = private
            new_idx = len(char_strings.charStringsIndex)
            char_strings.charStrings[name] = new_idx
            char_strings.charStringsIndex.append(cs)

    if "hmtx" in tt_font:
        for name in missing_names:
            w = _to_font_units(target_widths.get(name, 0))
            tt_font["hmtx"][name] = (w, 0)

    if "maxp" in tt_font:
        tt_font["maxp"].numGlyphs = len(glyph_order)


def _wrap_cff_in_otf(cff_data: bytes) -> bytes:
    """Wraps standalone CFF data in a minimal OTF container."""
    tag = b"CFF "
    offset = 12 + 16  # sfnt header (12) + one table record (16)
    length = len(cff_data)
    pad_len = (4 - length % 4) % 4
    padded = cff_data + b"\x00" * pad_len
    checksum = 0
    for i in range(0, len(padded), 4):
        checksum = (checksum + struct.unpack(">I", padded[i : i + 4])[0]) & 0xFFFFFFFF
    header = struct.pack(">4sHHHH", b"OTTO", 1, 16, 0, 16)
    table_record = struct.pack(">4sIII", tag, checksum, offset, length)
    return header + table_record + cff_data


def _write_back_cff(pdf: Pdf, fd: pikepdf.Object, tt_font) -> None:
    """Extracts CFF table data and writes it back as a FontFile3 stream."""
    # Preserve original subtype (Type1C or CIDFontType0C)
    existing = fd.get("/FontFile3")
    subtype = "/Type1C"
    if existing is not None:
        existing = _resolve(existing)
        st = existing.get("/Subtype")
        if st is not None:
            subtype = _safe_str(st)
    new_cff_data = tt_font.getTableData("CFF ")
    new_stream = pdf.make_stream(new_cff_data)
    new_stream[Name.Subtype] = Name(subtype)
    fd[Name("/FontFile3")] = pdf.make_indirect(new_stream)


def _write_back_font(pdf: Pdf, fd: pikepdf.Object, font_file_key: str, tt_font) -> None:
    """Serializes the modified font and writes it back to the PDF."""
    buf = io.BytesIO()
    tt_font.save(buf)
    buf.seek(0)
    new_data = buf.read()

    # Preserve /Subtype from existing FontFile3 stream (e.g. /OpenType)
    subtype = None
    if font_file_key == "/FontFile3":
        existing = fd.get(font_file_key)
        if existing is not None:
            existing = _resolve(existing)
            st = existing.get("/Subtype")
            if st is not None:
                subtype = _safe_str(st)

    new_stream = pdf.make_stream(new_data)
    if font_file_key in ("/FontFile2", "/FontFile"):
        new_stream[Name.Length1] = len(new_data)
    if subtype is not None:
        new_stream[Name.Subtype] = Name(subtype)
    fd[Name(font_file_key)] = pdf.make_indirect(new_stream)


# ---------------------------------------------------------------------------
# Type1 PFA/PFB glyph coverage
# ---------------------------------------------------------------------------


def _process_type1_simple_font(
    pdf: Pdf,
    font: pikepdf.Object,
    fd: pikepdf.Object,
    font_name: str,
    font_data: bytes,
    encoding: dict[int, str],
    used_codes: set[int],
) -> int:
    """Adds missing glyphs to a Type1 (PFA/PFB) font program.

    Parses the Type1 font to determine which glyphs exist, identifies
    missing ones based on the encoding, and adds minimal empty
    charstrings for them by modifying the eexec-encrypted section.

    Returns:
        Number of missing glyphs added.
    """
    existing_glyphs = _get_type1_glyph_names(font_data)
    if existing_glyphs is None:
        logger.debug("Font %s: cannot parse Type1 glyph names", font_name)
        return 0

    # Determine missing glyphs and their target widths
    first_char = int(font.get("/FirstChar", 0))
    widths_arr = font.get("/Widths")
    if widths_arr is not None:
        widths_arr = _resolve(widths_arr)

    missing_names: list[str] = []
    missing_widths: dict[str, int] = {}
    for code in sorted(used_codes):
        name = encoding.get(code)
        if name and name not in existing_glyphs:
            missing_names.append(name)
            if widths_arr is not None:
                idx = code - first_char
                if 0 <= idx < len(widths_arr):
                    missing_widths[name] = int(widths_arr[idx])

    if not missing_names:
        return 0

    # Get Length1/Length2 from the font file stream for proper splitting
    ff_stream = _resolve(fd.get("/FontFile"))
    length1 = int(ff_stream.get("/Length1", 0))
    length2 = int(ff_stream.get("/Length2", 0))

    new_data = _add_glyphs_to_type1(
        font_data,
        missing_names,
        missing_widths,
        length1=length1,
        length2=length2,
    )
    if new_data is None:
        logger.debug("Font %s: failed to add glyphs to Type1 data", font_name)
        return 0

    logger.info(
        "Font %s: added %d missing glyphs to Type1 font",
        font_name,
        len(missing_names),
    )

    # Write modified font back with correct Length1/Length2/Length3
    length3 = int(ff_stream.get("/Length3", 0))
    _write_back_type1(
        pdf,
        fd,
        font_data,
        new_data,
        length1,
        length2,
        length3,
    )

    # Remove stale CharSet
    if "/CharSet" in fd:
        del fd[Name("/CharSet")]

    return len(missing_names)


def _get_type1_glyph_names(font_data: bytes) -> set[str] | None:
    """Extracts glyph names from a Type1 PFA/PFB font program.

    Uses fontTools T1Font to parse the font and extract glyph names
    from the CharStrings dictionary.

    Returns:
        Set of glyph names, or None if parsing fails.
    """
    import os
    import tempfile

    suffix = ".pfa" if font_data[:2] == b"%!" else ".pfb"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, font_data)
    finally:
        os.close(tmp_fd)
    try:
        from fontTools.t1Lib import T1Font

        t1 = T1Font(tmp_path)
        return set(t1.getGlyphSet().keys())
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _add_glyphs_to_type1(
    font_data: bytes,
    missing_names: list[str],
    target_widths: dict[str, int],
    *,
    length1: int = 0,
    length2: int = 0,
) -> bytes | None:
    """Adds empty glyph charstrings to a Type1 font program.

    Decrypts the eexec section, creates minimal charstring entries
    for the missing glyphs (with correct widths), inserts them into
    the CharStrings dictionary, and re-encrypts.

    Args:
        font_data: Raw Type1 font data (PFA format with binary eexec).
        missing_names: Glyph names to add.
        target_widths: Mapping of glyph name to target width
            (in 1000-unit space).
        length1: Cleartext portion length (from /Length1 stream param).
        length2: Encrypted portion length (from /Length2 stream param).

    Returns:
        Modified font data, or None on error.
    """
    import re

    from fontTools.misc.eexec import decrypt as eexec_decrypt
    from fontTools.misc.eexec import encrypt as eexec_encrypt

    # Split using Length1/Length2 if available, else find eexec marker
    if length1 > 0 and length2 > 0:
        cleartext = font_data[:length1]
        encrypted_data = font_data[length1 : length1 + length2]
        trailing = font_data[length1 + length2 :]
    else:
        text = font_data.decode("latin-1")
        eexec_marker = "currentfile eexec"
        eexec_pos = text.find(eexec_marker)
        if eexec_pos < 0:
            return None
        header_end = text.index("\n", eexec_pos) + 1
        cleartext = font_data[:header_end]
        # Without Length2, try to find the boundary
        font_data[header_end:]
        # Use fontTools to find encrypted chunks
        from fontTools.t1Lib import findEncryptedChunks

        try:
            chunks = findEncryptedChunks(font_data)
            for is_encrypted, chunk in chunks:
                if is_encrypted:
                    encrypted_data = chunk
                    break
            else:
                return None
            trailing_start = font_data.index(encrypted_data) + len(encrypted_data)
            trailing = font_data[trailing_start:]
        except Exception:
            return None

    # Decrypt eexec (first 4 bytes are random seed)
    decrypted_full, _ = eexec_decrypt(encrypted_data, 55665)
    eexec_seed = decrypted_full[:4]
    decrypted = decrypted_full[4:]

    # Find lenIV (default 4)
    len_iv = 4
    leniv_match = re.search(rb"/lenIV\s+(\d+)", decrypted)
    if leniv_match:
        len_iv = int(leniv_match.group(1))

    # Find CharStrings section
    cs_match = re.search(rb"/CharStrings\s+(\d+)\s+dict\s+", decrypted)
    if cs_match is None:
        return None

    old_count = int(cs_match.group(1))
    new_count = old_count + len(missing_names)

    # Detect RD/ND command names (some fonts use -| and |-)
    rd_cmd = b"RD"
    nd_cmd = b"ND"
    if re.search(rb"\s-\|\s", decrypted) and not re.search(rb"\sRD\s", decrypted):
        rd_cmd = b"-|"
        nd_cmd = b"|-"

    # Find insertion point: the first 'end' after CharStrings entries
    # We need to scan past all charstring entries (which contain binary
    # data after RD commands) to find the closing 'end'
    insert_pos = _find_charstrings_end(decrypted, cs_match.end(), rd_cmd, nd_cmd)
    if insert_pos is None:
        return None

    # Build new charstring entries
    new_entries = b""
    for name in missing_names:
        width = target_widths.get(name, 0)
        cs_data = _create_type1_charstring(width, len_iv)
        entry = (
            b"/"
            + name.encode("latin-1")
            + b" "
            + str(len(cs_data)).encode()
            + b" "
            + rd_cmd
            + b" "
            + cs_data
            + b" "
            + nd_cmd
            + b"\n"
        )
        new_entries += entry

    # Update CharStrings count
    new_decrypted = (
        decrypted[: cs_match.start(1)]
        + str(new_count).encode()
        + decrypted[cs_match.end(1) : insert_pos]
        + new_entries
        + decrypted[insert_pos:]
    )

    # Re-encrypt with eexec (prepend same seed bytes)
    new_plaintext = eexec_seed + new_decrypted
    new_encrypted, _ = eexec_encrypt(new_plaintext, 55665)

    # Reassemble font data
    return cleartext + new_encrypted + trailing


def _find_charstrings_end(
    data: bytes,
    start: int,
    rd_cmd: bytes,
    nd_cmd: bytes,
) -> int | None:
    """Finds the 'end' keyword that closes the CharStrings dict.

    Scans through charstring entries (skipping binary data after RD
    commands) to find the closing 'end' keyword.

    Args:
        data: Decrypted eexec data.
        start: Byte offset after the CharStrings dict header.
        rd_cmd: The RD command name (b'RD' or b'-|').
        nd_cmd: The ND command name (b'ND' or b'|-').

    Returns:
        Byte offset of the 'end' keyword, or None if not found.
    """
    import re

    pos = start
    # Pattern to match charstring entry: /name N RD
    entry_pattern = re.compile(rb"/[\w.]+\s+(\d+)\s+" + re.escape(rd_cmd) + rb"\s")

    while pos < len(data):
        # Try to match a charstring entry
        m = entry_pattern.search(data, pos)

        # Check if 'end' comes before the next entry
        end_match = re.search(rb"\bend\b", data[pos:])
        if end_match is not None:
            end_pos = pos + end_match.start()
            if m is None or end_pos < m.start():
                return end_pos

        if m is None:
            break

        # Skip past the binary charstring data
        cs_length = int(m.group(1))
        pos = m.end() + cs_length

        # Skip past ND
        nd_pos = data.find(nd_cmd, pos)
        if nd_pos < 0:
            break
        pos = nd_pos + len(nd_cmd)

    return None


def _create_type1_charstring(width: int, len_iv: int) -> bytes:
    """Creates a minimal Type1 charstring for an empty glyph.

    Generates the bytecode for ``width 0 hsbw endchar``, prepends
    lenIV padding bytes, and encrypts with charstring encryption
    (R=4330).

    Args:
        width: Advance width in Type1 units (typically 1000 upm).
        len_iv: Number of initial random bytes (from font's /lenIV).

    Returns:
        Encrypted charstring bytes.
    """
    from fontTools.misc.eexec import encrypt as cs_encrypt

    # Encode the charstring program: width 0 hsbw endchar
    program = _encode_t1_number(width) + _encode_t1_number(0)
    program += bytes([13, 14])  # hsbw=13, endchar=14

    # Prepend lenIV padding bytes (zeros work fine)
    plaintext = bytes(len_iv) + program

    # Encrypt with charstring key
    ciphertext, _ = cs_encrypt(plaintext, 4330)
    return ciphertext


def _encode_t1_number(n: int) -> bytes:
    """Encodes an integer in Type1 charstring number format."""
    if -107 <= n <= 107:
        return bytes([n + 139])
    elif 108 <= n <= 1131:
        v = n - 108
        return bytes([v // 256 + 247, v % 256])
    elif -1131 <= n <= -108:
        v = -(n + 108)
        return bytes([v // 256 + 251, v % 256])
    else:
        return bytes([255]) + n.to_bytes(4, "big", signed=True)


def _write_back_type1(
    pdf: Pdf,
    fd: pikepdf.Object,
    old_data: bytes,
    new_data: bytes,
    length1: int,
    old_length2: int,
    length3: int,
) -> None:
    """Writes modified Type1 font data back to the PDF.

    Uses the original Length1 and Length3 (unchanged), and computes
    the new Length2 from the data size difference.
    """
    new_length2 = len(new_data) - length1 - length3

    new_stream = pdf.make_stream(new_data)
    new_stream[Name.Length1] = length1
    new_stream[Name.Length2] = new_length2
    new_stream[Name.Length3] = length3
    fd[Name("/FontFile")] = pdf.make_indirect(new_stream)
