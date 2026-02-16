# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font width validation and correction for PDF/A compliance.

ISO 19005-2, Clause 6.3.7 requires that glyph widths declared in the PDF
font dictionary are consistent with the widths in the embedded font program.
This module validates and corrects widths for already-embedded fonts.
"""

import io
import logging
from collections.abc import Iterator

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf

from ..fonts.metrics import FontMetricsExtractor
from ..fonts.tounicode import (
    generate_tounicode_for_macroman,
    generate_tounicode_for_standard_encoding,
    generate_tounicode_for_winansi,
    generate_tounicode_from_encoding_dict,
    parse_cidtogidmap_stream,
)
from ..fonts.traversal import iter_all_page_fonts
from ..fonts.utils import safe_str as _safe_str
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Tolerance for width comparison — veraPDF uses ±1, so we must match
_WIDTH_TOLERANCE = 1

# Valid range for unitsPerEm (OpenType spec)
_MIN_UNITS_PER_EM = 16
_MAX_UNITS_PER_EM = 16384

_metrics = FontMetricsExtractor()


def _validate_font_program(tt_font, font_name: str = "") -> bool:
    """Validates basic structural integrity of the embedded font program.

    Checks that essential tables exist and contain plausible data.
    A font that passes this check may still be subtly corrupted, but
    obviously broken fonts are rejected.

    Args:
        tt_font: fontTools TTFont object.
        font_name: Font name for logging.

    Returns:
        True if the font program appears structurally valid.
    """
    # Check essential tables
    for table in ("head", "hhea", "hmtx", "maxp"):
        if table not in tt_font:
            logger.debug(
                "Font %s: font program missing required '%s' table",
                font_name,
                table,
            )
            return False

    # Check unitsPerEm is in valid range
    units_per_em = tt_font["head"].unitsPerEm
    if not (_MIN_UNITS_PER_EM <= units_per_em <= _MAX_UNITS_PER_EM):
        logger.debug(
            "Font %s: font program has invalid unitsPerEm: %d",
            font_name,
            units_per_em,
        )
        return False

    # Check hmtx is non-empty and consistent with maxp
    tt_font["maxp"].numGlyphs
    num_hmtx = len(tt_font["hmtx"].metrics)
    if num_hmtx == 0:
        logger.debug("Font %s: font program has empty hmtx table", font_name)
        return False

    # Check for negative advance widths
    for _name, (advance, _lsb) in tt_font["hmtx"].metrics.items():
        if advance < 0:
            logger.debug(
                "Font %s: font program has negative advance width",
                font_name,
            )
            return False

    # If all glyphs have zero width, the font program is likely damaged
    all_zero = all(
        advance == 0 for _name, (advance, _lsb) in tt_font["hmtx"].metrics.items()
    )
    if all_zero and num_hmtx > 1:
        logger.debug("Font %s: font program has all zero-width glyphs", font_name)
        return False

    return True


def sanitize_font_widths(pdf: Pdf) -> dict[str, int]:
    """Validates and corrects font widths for all embedded fonts.

    Iterates all fonts in the PDF, compares declared widths against the
    embedded font program, and corrects mismatches in-place.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with counts of fixes applied.
    """
    result: dict[str, int] = {
        "simple_font_widths_fixed": 0,
        "cidfont_widths_fixed": 0,
        "type3_font_widths_fixed": 0,
    }

    for font_name, font_obj, font_type in _iter_all_embedded_fonts(pdf):
        try:
            if font_type in ("Type1", "TrueType", "MMType1"):
                if _fix_simple_font_widths(font_obj, font_name):
                    result["simple_font_widths_fixed"] += 1
            elif font_type == "CIDFont":
                if _fix_cidfont_widths(font_obj, font_name):
                    result["cidfont_widths_fixed"] += 1
            elif font_type == "Type3":
                if _fix_type3_font_widths(font_obj, font_name):
                    result["type3_font_widths_fixed"] += 1
        except Exception as e:
            logger.debug(
                "Skipping width validation for font %s: %s",
                font_name,
                e,
            )
            continue

    total = (
        result["simple_font_widths_fixed"]
        + result["cidfont_widths_fixed"]
        + result["type3_font_widths_fixed"]
    )
    if total > 0:
        logger.info(
            "Font width sanitization: %d simple, %d CIDFont, %d Type3 fixed",
            result["simple_font_widths_fixed"],
            result["cidfont_widths_fixed"],
            result["type3_font_widths_fixed"],
        )

    return result


def _iter_all_embedded_fonts(
    pdf: Pdf,
) -> Iterator[tuple[str, pikepdf.Object, str]]:
    """Yields (font_name, font_obj, font_type) for all embedded fonts.

    Includes Type3 fonts (which have CharProcs instead of font programs)
    and skips fonts without embedded data.

    Args:
        pdf: Opened pikepdf PDF object.

    Yields:
        Tuples of (font_name, resolved_font_dict, font_type_str).
    """
    seen_objgens: set[tuple[int, int]] = set()

    for page in pdf.pages:
        for _font_key, font_obj in iter_all_page_fonts(pikepdf.Page(page)):
            font = _resolve(font_obj)
            if not isinstance(font, Dictionary):
                continue

            # Deduplicate by objgen
            objgen = font.objgen
            if objgen != (0, 0):
                if objgen in seen_objgens:
                    continue
                seen_objgens.add(objgen)

            subtype = font.get("/Subtype")
            if subtype is None:
                continue
            subtype_str = _safe_str(subtype)

            # Type3: procedurally defined via CharProcs, no font program
            if subtype_str == "/Type3":
                char_procs = font.get("/CharProcs")
                if char_procs is not None:
                    font_name = _get_font_name(font)
                    yield font_name, font, "Type3"
                continue

            if subtype_str == "/Type0":
                # CIDFont: check descendant
                descendants = font.get("/DescendantFonts")
                if descendants is None:
                    continue
                descendants = _resolve(descendants)
                if not isinstance(descendants, Array) or len(descendants) == 0:
                    continue
                desc_font = _resolve(descendants[0])
                if not isinstance(desc_font, Dictionary):
                    continue
                if not _has_embedded_data(desc_font):
                    continue
                font_name = _get_font_name(font)
                yield font_name, font, "CIDFont"
            elif subtype_str in ("/Type1", "/TrueType", "/MMType1"):
                if not _has_embedded_data(font):
                    continue
                font_name = _get_font_name(font)
                type_map = {
                    "/Type1": "Type1",
                    "/TrueType": "TrueType",
                    "/MMType1": "MMType1",
                }
                yield font_name, font, type_map[subtype_str]


def _get_font_name(font: pikepdf.Object) -> str:
    """Extracts font name from a font dictionary."""
    base_font = font.get("/BaseFont")
    if base_font is not None:
        return _safe_str(base_font)[1:]  # Remove leading "/"
    return "Unknown"


def _has_embedded_data(font: pikepdf.Object) -> bool:
    """Checks if a font has embedded data via its FontDescriptor."""
    fd = font.get("/FontDescriptor")
    if fd is None:
        return False
    fd = _resolve(fd)
    for key in ("/FontFile", "/FontFile2", "/FontFile3"):
        if fd.get(key) is not None:
            return True
    return False


def _parse_type1_font(font_data: bytes):
    """Parses Type1 PFA/PFB font data into a TTFont-like object.

    Uses fontTools.t1Lib to read the font and creates a minimal TTFont
    with CFF table for width extraction compatibility.

    Args:
        font_data: Raw PFA or PFB font data.

    Returns:
        fontTools TTFont object with CFF table, or None if parsing fails.
    """
    import os
    import tempfile

    from fontTools.ttLib import TTFont

    # T1Font requires a file path, not a BytesIO
    suffix = ".pfa" if font_data[:2] == b"%!" else ".pfb"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, font_data)
    finally:
        os.close(tmp_fd)
    try:
        from fontTools.t1Lib import T1Font

        t1 = T1Font(tmp_path)
        glyphset = t1.getGlyphSet()

        # Extract widths from Type1 charstrings (hsbw/sbw operators)
        glyph_widths: dict[str, int] = {}
        for gname, cs in glyphset.items():
            cs.decompile()
            prog = cs.program
            for i, token in enumerate(prog):
                if token == "hsbw" and i >= 2:
                    glyph_widths[gname] = int(prog[i - 1])
                    break
                elif token == "sbw" and i >= 4:
                    glyph_widths[gname] = int(prog[i - 2])
                    break

        if not glyph_widths:
            return None

        # Create a minimal TTFont with synthetic hmtx/head tables
        # so the standard width comparison path works
        tt = TTFont()

        # Synthetic head table
        from fontTools.ttLib.tables._h_e_a_d import table__h_e_a_d

        head = table__h_e_a_d()
        head.unitsPerEm = 1000  # Type1 fonts use 1000 upm
        head.xMin = head.yMin = head.xMax = head.yMax = 0
        head.magicNumber = 0x5F0F3CF5
        head.flags = 0
        head.macStyle = 0
        head.indexToLocFormat = 0
        head.glyphDataFormat = 0
        head.created = head.modified = 0
        head.lowestRecPPEM = 8
        head.fontDirectionHint = 2
        head.fontRevision = 1.0
        tt["head"] = head

        # Synthetic hhea table
        from fontTools.ttLib.tables._h_h_e_a import table__h_h_e_a

        hhea = table__h_h_e_a()
        hhea.ascent = 800
        hhea.descent = -200
        hhea.lineGap = 0
        hhea.advanceWidthMax = max(glyph_widths.values()) if glyph_widths else 0
        hhea.numberOfHMetrics = len(glyph_widths)
        tt["hhea"] = hhea

        # Synthetic maxp table
        from fontTools.ttLib.tables._m_a_x_p import table__m_a_x_p

        maxp = table__m_a_x_p()
        maxp.version = 0x00005000
        maxp.numGlyphs = len(glyph_widths)
        tt["maxp"] = maxp

        # hmtx table with extracted widths
        from fontTools.ttLib.tables._h_m_t_x import table__h_m_t_x

        hmtx = table__h_m_t_x()
        hmtx.metrics = {gname: (w, 0) for gname, w in glyph_widths.items()}
        tt["hmtx"] = hmtx

        # Set glyph order
        glyph_order = list(glyph_widths.keys())
        if ".notdef" in glyph_order:
            glyph_order.remove(".notdef")
            glyph_order.insert(0, ".notdef")
        tt.setGlyphOrder(glyph_order)

        # Build cmap from glyph names via AGL (Adobe Glyph List)
        from fontTools.agl import AGL2UV
        from fontTools.ttLib.tables._c_m_a_p import table__c_m_a_p

        cmap_dict: dict[int, str] = {}
        for gname in glyph_widths:
            if gname == ".notdef":
                continue
            uv = AGL2UV.get(gname)
            if uv is not None:
                cmap_dict[uv] = gname

        if cmap_dict:
            cmap_table = table__c_m_a_p()
            cmap_table.tableVersion = 0
            from fontTools.ttLib.tables._c_m_a_p import cmap_format_4

            subtable = cmap_format_4(4)
            subtable.platEncID = 3
            subtable.platformID = 3
            subtable.format = 4
            subtable.reserved = 0
            subtable.length = 0
            subtable.language = 0
            subtable.cmap = cmap_dict
            cmap_table.tables = [subtable]
            tt["cmap"] = cmap_table

        return tt
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _extract_font_program(font: pikepdf.Object):
    """Extracts and parses the embedded font program.

    Args:
        font: Font dictionary (simple font or CIDFont descendant).

    Returns:
        fontTools TTFont object, or None if extraction fails.
    """
    from fontTools.ttLib import TTFont

    fd = _resolve(font.get("/FontDescriptor"))
    font_data = None
    font_file_key = None
    for key in ("/FontFile2", "/FontFile3", "/FontFile"):
        stream = fd.get(key)
        if stream is not None:
            stream = _resolve(stream)
            font_data = bytes(stream.read_bytes())
            font_file_key = key
            break

    if font_data is None:
        return None

    try:
        return TTFont(io.BytesIO(font_data))
    except Exception:
        # Try wrapping bare CFF (Type1C / CIDFontType0C) in OTF container
        if font_file_key == "/FontFile3":
            try:
                from .glyph_coverage import _wrap_cff_in_otf

                otf_data = _wrap_cff_in_otf(font_data)
                return TTFont(io.BytesIO(otf_data))
            except Exception:
                return None
        # Try parsing Type1 PFB/PFA via fontTools.t1Lib
        if font_file_key == "/FontFile":
            return _parse_type1_font(font_data)
        return None


def _get_missing_width(font: pikepdf.Object, tt_font) -> int | None:
    """Gets the fallback width for glyphs not found in the font program.

    veraPDF checks rule 6.3.6 by comparing the /Widths entry against the
    width from the embedded font program.  For glyphs not present in the
    font's cmap, the font program returns the .notdef glyph width.
    Therefore the fallback must use the .notdef width from the font
    program (not /MissingWidth from the FontDescriptor) to match what
    veraPDF considers the "widthFromFontProgram".

    Lookup order:
      1. .notdef width from the font program (scaled to 1000 units)
      2. /MissingWidth from FontDescriptor (only if .notdef unavailable)

    Returns:
        Fallback width in PDF units, or None if unavailable.
    """
    # Priority 1: .notdef width from font program (matches veraPDF)
    try:
        if "hmtx" in tt_font and "head" in tt_font:
            hmtx = tt_font["hmtx"]
            head = tt_font["head"]
            notdef = hmtx.metrics.get(".notdef")
            if notdef is not None:
                return round(notdef[0] * 1000.0 / head.unitsPerEm)
    except Exception:
        pass

    # Priority 2: MissingWidth from FontDescriptor
    try:
        fd = font.get("/FontDescriptor")
        if fd is not None:
            fd = _resolve(fd)
            mw = fd.get("/MissingWidth")
            if mw is not None:
                return int(mw)
    except Exception:
        pass

    return None


def _get_encoding_mapping(font: pikepdf.Object) -> dict[int, int] | None:
    """Builds a code-to-Unicode mapping from the font's encoding.

    Args:
        font: Simple font dictionary.

    Returns:
        Dictionary mapping char codes to Unicode codepoints, or None.
    """
    encoding = font.get("/Encoding")
    if encoding is None:
        # Default to WinAnsiEncoding for TrueType, StandardEncoding for Type1
        subtype = font.get("/Subtype")
        if subtype is not None and _safe_str(subtype) == "/TrueType":
            return generate_tounicode_for_winansi()
        return generate_tounicode_for_standard_encoding()

    encoding = _resolve(encoding)

    if isinstance(encoding, Name):
        enc_name = _safe_str(encoding)
        if enc_name == "/WinAnsiEncoding":
            return generate_tounicode_for_winansi()
        elif enc_name == "/MacRomanEncoding":
            return generate_tounicode_for_macroman()
        elif enc_name == "/StandardEncoding":
            return generate_tounicode_for_standard_encoding()
        else:
            return generate_tounicode_for_winansi()

    if isinstance(encoding, Dictionary):
        return generate_tounicode_from_encoding_dict(encoding)

    return None


def _compute_widths_by_name(font: pikepdf.Object, tt_font) -> dict[int, int]:
    """Computes glyph widths by looking up glyphs by name in the font.

    veraPDF resolves glyphs for simple fonts by mapping character codes
    through the encoding to Adobe glyph names, then checking if those
    names exist in the font program (via the post table or glyph order).
    This function replicates that lookup for glyphs not reachable via
    the cmap table (e.g. glyphs added by glyph_coverage sanitizer).

    Args:
        font: Simple font dictionary.
        tt_font: fontTools TTFont object.

    Returns:
        Dictionary mapping char_code to width (scaled to 1000 units)
        for glyphs found by name but not through the cmap.
    """
    from ..fonts.subsetter import _resolve_simple_font_encoding

    try:
        encoding = _resolve_simple_font_encoding(font)
    except Exception:
        return {}
    if not encoding:
        return {}

    if "hmtx" not in tt_font or "head" not in tt_font:
        return {}

    hmtx = tt_font["hmtx"]
    units_per_em = tt_font["head"].unitsPerEm
    scale = 1000.0 / units_per_em

    result: dict[int, int] = {}
    for code, glyph_name in encoding.items():
        if glyph_name in hmtx.metrics:
            result[code] = round(hmtx.metrics[glyph_name][0] * scale)

    return result


def _extract_cff_glyph_widths(tt_font) -> dict[str, int]:
    """Extracts glyph widths from CFF charstrings.

    For bare CFF fonts that lack hmtx/head tables, widths are encoded
    directly in the CFF charstring programs.  The width encoding in
    Type 2 charstrings depends on the first operator:

    - Operators that take an even number of args (hstem, vstem, etc.):
      an odd total means the first arg is the width.
    - hmoveto/vmoveto: 2 args means first is width (normally 1).
    - rmoveto: 3 args means first is width (normally 2).
    - endchar: 1+ args means first is width (normally 0 or 4 for seac).

    Handles CFF FontMatrix scaling (for fonts with non-standard upm)
    and FDSelect-based per-glyph Private dict lookup for CID-keyed fonts.

    Args:
        tt_font: fontTools TTFont with a CFF table.

    Returns:
        Dictionary mapping glyph name to width (scaled to 1000 units).
    """
    if "CFF " not in tt_font:
        return {}

    cff = tt_font["CFF "]
    top_dict = cff.cff.topDictIndex[0]
    char_strings = top_dict.CharStrings

    # Determine FontMatrix scale factor.
    # Standard CFF FontMatrix is [0.001, 0, 0, 0.001, 0, 0] → scale = 1.0
    # Non-standard (e.g. upm=2048) might be [0.000488, ...] → scale = 0.488
    font_matrix = getattr(top_dict, "FontMatrix", None)
    if font_matrix is not None:
        fm_scale = font_matrix[0] * 1000.0
    else:
        fm_scale = 1.0

    # Build per-glyph Private dict lookup for FDArray/FDSelect fonts
    has_fd_array = hasattr(top_dict, "FDArray") and top_dict.FDArray
    fd_select = None
    if has_fd_array:
        fd_select = getattr(top_dict, "FDSelect", None)

    def _get_private(gid: int):
        """Returns (nominalWidthX, defaultWidthX) for a given GID."""
        if has_fd_array and fd_select is not None:
            fd_idx = fd_select[gid]
            private = top_dict.FDArray[fd_idx].Private
        elif has_fd_array:
            private = top_dict.FDArray[0].Private
        else:
            private = top_dict.Private
        return (
            getattr(private, "nominalWidthX", 0),
            getattr(private, "defaultWidthX", 0),
        )

    # Operators that consume an even number of stack arguments
    # (pairs): if stack has an odd count, first is width
    pair_ops = frozenset(
        [
            "hstem",
            "vstem",
            "hstemhm",
            "vstemhm",
            "hintmask",
            "cntrmask",
        ]
    )

    glyph_order = char_strings.keys()
    # Build glyph name → GID mapping for FDSelect lookup
    glyph_name_to_gid: dict[str, int] = {}
    for gid, gname in enumerate(tt_font.getGlyphOrder() if has_fd_array else []):
        glyph_name_to_gid[gname] = gid

    result: dict[str, int] = {}
    for gname in glyph_order:
        cs = char_strings[gname]
        cs.decompile()
        prog = cs.program

        gid = glyph_name_to_gid.get(gname, 0)
        nominal_w, default_w = _get_private(gid)

        # Collect numbers before first operator
        stack: list = []
        first_op = None
        for token in prog:
            if isinstance(token, (int, float)):
                stack.append(token)
            else:
                first_op = token
                break

        has_width = False
        if first_op in pair_ops:
            # Even-arg operators: odd stack means width present
            has_width = len(stack) % 2 == 1
        elif first_op == "hmoveto" or first_op == "vmoveto":
            has_width = len(stack) == 2
        elif first_op == "rmoveto":
            has_width = len(stack) == 3
        elif first_op == "endchar":
            has_width = len(stack) in (1, 5)

        if has_width and stack:
            raw_width = stack[0] + nominal_w
        else:
            raw_width = default_w

        # Apply FontMatrix scaling
        result[gname] = round(raw_width * fm_scale)

    return result


def _fix_type3_font_widths(font: pikepdf.Object, font_name: str) -> bool:
    """Validates and fixes widths for a Type3 font.

    Type3 fonts define glyphs via CharProcs (content streams).
    The glyph width is specified by d0 or d1 operators in each
    CharProc's content stream.

    Args:
        font: Type3 font dictionary.
        font_name: Font name for logging.

    Returns:
        True if widths were corrected.
    """
    first_char_obj = font.get("/FirstChar")
    last_char_obj = font.get("/LastChar")
    widths_obj = font.get("/Widths")
    char_procs = font.get("/CharProcs")

    if (
        first_char_obj is None
        or last_char_obj is None
        or widths_obj is None
        or char_procs is None
    ):
        return False

    first_char = int(first_char_obj)
    last_char = int(last_char_obj)
    widths_obj = _resolve(widths_obj)
    declared_widths = [int(w) for w in widths_obj]

    if len(declared_widths) != (last_char - first_char + 1):
        return False

    char_procs = _resolve(char_procs)

    # Build code→glyphname mapping from /Encoding
    code_to_glyph: dict[int, str] = {}
    encoding = font.get("/Encoding")
    if encoding is not None:
        encoding = _resolve(encoding)
        if isinstance(encoding, Dictionary):
            diffs = encoding.get("/Differences")
            if diffs is not None:
                diffs = _resolve(diffs)
                current_code = 0
                for item in diffs:
                    try:
                        current_code = int(item)
                        continue
                    except (TypeError, ValueError):
                        pass
                    if isinstance(item, Name):
                        glyph_name = _safe_str(item)[1:]
                        code_to_glyph[current_code] = glyph_name
                        current_code += 1

    # Determine FontMatrix scale for width comparison.
    # Type3 FontMatrix is typically [0.001, 0, 0, 0.001, 0, 0]
    # meaning glyph space is 1/1000 of text space. The /Widths
    # array is in text space (like other font types), but d0/d1
    # widths are in glyph space and must be scaled by FontMatrix[0]*1000.
    font_matrix = font.get("/FontMatrix")
    if font_matrix is not None:
        fm = _resolve(font_matrix)
        fm_scale = float(fm[0]) * 1000.0
    else:
        fm_scale = 1.0

    # Extract widths from CharProc streams (d0/d1 operators)
    glyph_widths: dict[str, int] = {}
    for glyph_name_key in dict(char_procs):
        glyph_name = _safe_str(glyph_name_key)[1:]
        stream = _resolve(char_procs[glyph_name_key])
        try:
            ops = pikepdf.parse_content_stream(stream)
            for operands, op in ops:
                op_str = str(op)
                if op_str in ("d0", "d1"):
                    # d0: wx wy d0 — width is first operand
                    # d1: wx wy llx lly urx ury d1 — width is first operand
                    raw_width = float(operands[0])
                    glyph_widths[glyph_name] = round(raw_width * fm_scale)
                    break
        except Exception:
            continue

    if not glyph_widths:
        return False

    # Compare declared vs CharProc widths
    mismatches = 0
    corrected_widths = list(declared_widths)
    for i, code in enumerate(range(first_char, last_char + 1)):
        glyph_name = code_to_glyph.get(code)
        if glyph_name and glyph_name in glyph_widths:
            actual = glyph_widths[glyph_name]
            if abs(declared_widths[i] - actual) > _WIDTH_TOLERANCE:
                corrected_widths[i] = actual
                mismatches += 1

    if mismatches == 0:
        return False

    font[Name.Widths] = Array(corrected_widths)
    logger.info(
        "Fixed %d width mismatches in Type3 font %s",
        mismatches,
        font_name,
    )
    return True


def _fix_simple_font_widths(font: pikepdf.Object, font_name: str) -> bool:
    """Validates and fixes widths for a simple font (Type1/TrueType/MMType1).

    Args:
        font: Font dictionary.
        font_name: Font name for logging.

    Returns:
        True if widths were corrected.
    """
    # Get declared widths
    first_char_obj = font.get("/FirstChar")
    last_char_obj = font.get("/LastChar")
    widths_obj = font.get("/Widths")

    if first_char_obj is None or last_char_obj is None or widths_obj is None:
        return False

    first_char = int(first_char_obj)
    last_char = int(last_char_obj)
    widths_obj = _resolve(widths_obj)
    declared_widths = [int(w) for w in widths_obj]

    if len(declared_widths) != (last_char - first_char + 1):
        return False

    # Extract and parse font program
    tt_font = _extract_font_program(font)
    if tt_font is None:
        return False

    try:
        # For CFF fonts without hmtx, use CFF-specific width extraction
        is_cff_only = "CFF " in tt_font and "hmtx" not in tt_font
        if is_cff_only:
            return _fix_simple_font_widths_cff(
                font,
                font_name,
                tt_font,
                first_char,
                last_char,
                declared_widths,
            )

        if not _validate_font_program(tt_font, font_name):
            return False

        # Build encoding mapping
        code_to_unicode = _get_encoding_mapping(font)
        if code_to_unicode is None:
            return False

        # Compute expected widths
        expected = _metrics.compute_widths_for_encoding(tt_font, code_to_unicode)

        # Build a name-based width lookup for glyphs not reachable via
        # the cmap (e.g. glyphs added by glyph_coverage sanitizer).
        # veraPDF resolves glyphs by name through the post table / AGL
        # when the cmap lookup fails, so we must do the same.
        name_widths = _compute_widths_by_name(font, tt_font)

        # Determine fallback width for codes not found in font at all.
        # veraPDF uses the .notdef glyph width from the font program
        # as widthFromFontProgram for completely missing glyphs.
        fallback_width = _get_missing_width(font, tt_font)

        # Compare declared vs expected
        mismatches = 0
        comparable = 0
        corrected_widths = list(declared_widths)
        for i, code in enumerate(range(first_char, last_char + 1)):
            if code in expected:
                comparable += 1
                declared = declared_widths[i]
                actual = expected[code]
                if abs(declared - actual) > _WIDTH_TOLERANCE:
                    corrected_widths[i] = actual
                    mismatches += 1
            elif code in name_widths:
                # Glyph found by name (e.g. added by glyph_coverage)
                # but not reachable through the cmap.
                comparable += 1
                declared = declared_widths[i]
                actual = name_widths[code]
                if abs(declared - actual) > _WIDTH_TOLERANCE:
                    corrected_widths[i] = actual
                    mismatches += 1
            elif fallback_width is not None and code in code_to_unicode:
                # Code has an encoding entry but no glyph was found in
                # the font program — use .notdef width as the expected
                # width (matches veraPDF behaviour).
                declared = declared_widths[i]
                if abs(declared - fallback_width) > _WIDTH_TOLERANCE:
                    corrected_widths[i] = fallback_width
                    mismatches += 1

        if mismatches == 0:
            return False

        if comparable > 0 and mismatches > comparable * 0.5:
            logger.info(
                "Font %s: %d/%d widths mismatch (%.0f%%) — correcting",
                font_name,
                mismatches,
                comparable,
                100 * mismatches / comparable,
            )

        # Replace the /Widths array
        font[Name.Widths] = Array(corrected_widths)
        logger.info(
            "Fixed %d width mismatches in simple font %s",
            mismatches,
            font_name,
        )
        return True
    except Exception as e:
        logger.debug("Error validating widths for %s: %s", font_name, e)
        return False
    finally:
        tt_font.close()


def _fix_simple_font_widths_cff(
    font: pikepdf.Object,
    font_name: str,
    tt_font,
    first_char: int,
    last_char: int,
    declared_widths: list[int],
) -> bool:
    """Fixes widths for a simple font with bare CFF font program.

    CFF fonts loaded via a minimal OTF wrapper lack hmtx/head tables,
    so widths must be extracted directly from the CFF charstrings.
    """
    cff_widths = _extract_cff_glyph_widths(tt_font)
    if not cff_widths:
        return False

    # Resolve encoding to glyph names
    from ..fonts.subsetter import _resolve_simple_font_encoding

    encoding = _resolve_simple_font_encoding(font)
    if not encoding:
        return False

    mismatches = 0
    comparable = 0
    corrected_widths = list(declared_widths)
    for i, code in enumerate(range(first_char, last_char + 1)):
        glyph_name = encoding.get(code)
        if glyph_name and glyph_name in cff_widths:
            comparable += 1
            actual = cff_widths[glyph_name]
            declared = declared_widths[i]
            if abs(declared - actual) > _WIDTH_TOLERANCE:
                corrected_widths[i] = actual
                mismatches += 1

    if mismatches == 0:
        return False

    font[Name.Widths] = Array(corrected_widths)
    logger.info(
        "Fixed %d width mismatches in CFF simple font %s",
        mismatches,
        font_name,
    )
    return True


def _parse_w_array(w_array: pikepdf.Object) -> dict[int, int]:
    """Parses a CIDFont /W array into a CID-to-width mapping.

    The /W array uses two formats:
    - [cid [w1 w2 ...]] — individual widths for consecutive CIDs
    - [cid_first cid_last width] — same width for a range of CIDs

    Args:
        w_array: pikepdf Array representing the /W entry.

    Returns:
        Dictionary mapping CID to declared width.
    """
    w_array = _resolve(w_array)
    result: dict[int, int] = {}
    i = 0
    items = list(w_array)

    while i < len(items):
        start_cid = int(items[i])
        i += 1
        if i >= len(items):
            break

        next_item = _resolve(items[i])
        if isinstance(next_item, Array):
            # Format 1: start_cid [w1, w2, ...]
            for j, w in enumerate(next_item):
                result[start_cid + j] = int(w)
            i += 1
        else:
            # Format 2: cid_first cid_last width
            end_cid = int(next_item)
            i += 1
            if i >= len(items):
                break
            width = int(items[i])
            i += 1
            for cid in range(start_cid, end_cid + 1):
                result[cid] = width

    return result


def _fix_cidfont_widths(font: pikepdf.Object, font_name: str) -> bool:
    """Validates and fixes widths for a CIDFont (Type0).

    Args:
        font: Type0 font dictionary.
        font_name: Font name for logging.

    Returns:
        True if widths were corrected.
    """
    descendants = _resolve(font.get("/DescendantFonts"))
    desc_font = _resolve(descendants[0])

    # Get /W array
    w_array = desc_font.get("/W")

    if w_array is None:
        return False

    declared_widths = _parse_w_array(w_array)
    if not declared_widths:
        return False

    # Extract and parse font program from the descendant font
    tt_font = _extract_font_program(desc_font)
    if tt_font is None:
        return False

    try:
        # For CFF CID-keyed fonts without hmtx, use CFF-specific path
        is_cff_only = "CFF " in tt_font and "hmtx" not in tt_font
        if is_cff_only:
            return _fix_cidfont_widths_cff(
                desc_font,
                font_name,
                tt_font,
                declared_widths,
            )

        if not _validate_font_program(tt_font, font_name):
            return False

        # Determine CID→GID mapping
        cid_to_gid_obj = desc_font.get("/CIDToGIDMap")
        cid_to_gid: dict[int, int] | None = None

        if cid_to_gid_obj is not None:
            cid_to_gid_obj = _resolve(cid_to_gid_obj)
            if isinstance(cid_to_gid_obj, Name):
                if _safe_str(cid_to_gid_obj) == "/Identity":
                    cid_to_gid = None  # Identity: CID == GID
                else:
                    return False
            elif isinstance(cid_to_gid_obj, pikepdf.Stream):
                stream_data = bytes(cid_to_gid_obj.read_bytes())
                cid_to_gid = parse_cidtogidmap_stream(stream_data)
            else:
                return False

        # Build set of GIDs we need widths for
        gids_needed: set[int] = set()
        cid_to_gid_map: dict[int, int] = {}
        for cid in declared_widths:
            if cid_to_gid is not None:
                gid = cid_to_gid.get(cid)
                if gid is None:
                    continue
            else:
                gid = cid  # Identity mapping
            gids_needed.add(gid)
            cid_to_gid_map[cid] = gid

        # Compute expected widths for those GIDs
        expected_gid_widths = _metrics.compute_widths_for_gids(tt_font, gids_needed)

        # Compare
        mismatches = 0
        comparable = 0
        for cid, declared in declared_widths.items():
            gid = cid_to_gid_map.get(cid)
            if gid is None:
                continue
            expected = expected_gid_widths.get(gid)
            if expected is None:
                continue
            comparable += 1
            if abs(declared - expected) > _WIDTH_TOLERANCE:
                mismatches += 1

        if mismatches == 0:
            return False

        if comparable > 0 and mismatches > comparable * 0.5:
            logger.info(
                "CIDFont %s: %d/%d widths mismatch (%.0f%%) — correcting",
                font_name,
                mismatches,
                comparable,
                100 * mismatches / comparable,
            )

        # Rebuild /W array from the font program
        new_w_array = _metrics.build_cidfont_w_array(tt_font)
        desc_font[Name.W] = Array(_convert_w_array_to_pikepdf(new_w_array))

        # Set /DW to .notdef (GID 0) width from the font program
        glyph_order = tt_font.getGlyphOrder()
        if glyph_order:
            notdef_name = glyph_order[0]
            hmtx = tt_font["hmtx"]
            if notdef_name in hmtx.metrics:
                units_per_em = tt_font["head"].unitsPerEm
                scale = 1000.0 / units_per_em
                dw = round(hmtx.metrics[notdef_name][0] * scale)
                desc_font[Name.DW] = dw

        logger.info(
            "Fixed %d width mismatches in CIDFont %s",
            mismatches,
            font_name,
        )
        return True
    except Exception as e:
        logger.debug("Error validating CIDFont widths for %s: %s", font_name, e)
        return False
    finally:
        tt_font.close()


def _fix_cidfont_widths_cff(
    desc_font: pikepdf.Object,
    font_name: str,
    tt_font,
    declared_widths: dict[int, int],
) -> bool:
    """Fixes widths for a CIDFontType0 with bare CFF font program.

    CFF CID-keyed fonts loaded via a minimal OTF wrapper lack hmtx/head,
    so widths must be extracted directly from CFF charstrings.
    """
    cff_widths = _extract_cff_glyph_widths(tt_font)
    if not cff_widths:
        return False

    # Build CID→charstring-name mapping
    # For CID-keyed CFF: glyph names are "cid00041" etc.
    cid_to_width: dict[int, int] = {}
    for gname, width in cff_widths.items():
        if gname == ".notdef":
            cid_to_width[0] = width
        elif gname.startswith("cid"):
            try:
                cid_to_width[int(gname[3:])] = width
            except ValueError:
                pass

    # Compare declared vs CFF
    mismatches = 0
    comparable = 0
    for cid, declared in declared_widths.items():
        actual = cid_to_width.get(cid)
        if actual is None:
            continue
        comparable += 1
        if abs(declared - actual) > _WIDTH_TOLERANCE:
            mismatches += 1

    if mismatches == 0:
        return False

    # Rebuild /W array from CFF widths
    # Build sorted list of (cid, width) for all known CIDs
    all_widths: list[tuple[int, int]] = sorted(cid_to_width.items())
    w_array: list = []
    i = 0
    while i < len(all_widths):
        start_cid, start_w = all_widths[i]
        # Collect consecutive CIDs
        seq = [start_w]
        j = i + 1
        while j < len(all_widths) and all_widths[j][0] == start_cid + (j - i):
            seq.append(all_widths[j][1])
            j += 1
        w_array.append(start_cid)
        w_array.append(seq)
        i = j

    desc_font[Name.W] = Array(_convert_w_array_to_pikepdf(w_array))

    # Set /DW to .notdef width
    notdef_w = cid_to_width.get(0)
    if notdef_w is not None:
        desc_font[Name.DW] = notdef_w

    logger.info(
        "Fixed %d width mismatches in CFF CIDFont %s",
        mismatches,
        font_name,
    )
    return True


def _convert_w_array_to_pikepdf(w_array: list) -> list:
    """Converts a Python W array (from FontMetricsExtractor) to pikepdf objects.

    The W array from build_cidfont_w_array contains integers and lists.
    pikepdf Array needs all items to be pikepdf-compatible.

    Args:
        w_array: Python list with integers and sub-lists.

    Returns:
        List suitable for pikepdf.Array constructor.
    """
    result = []
    for item in w_array:
        if isinstance(item, list):
            result.append(Array(item))
        else:
            result.append(item)
    return result
