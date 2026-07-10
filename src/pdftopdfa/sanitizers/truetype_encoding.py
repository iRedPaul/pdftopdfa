# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""TrueType font encoding sanitizer for PDF/A compliance.

Covers ISO 19005-2 rules 6.2.11.6-1 through 6.2.11.6-4:
- 6.2.11.6-1: Non-symbolic TrueType must have a (3,1) or non-(3,0) cmap
- 6.2.11.6-2: Non-symbolic TrueType /Encoding must name WinAnsiEncoding or
              MacRomanEncoding (and Differences must use AGL names only);
              fonts with valid Differences also get a (3,1) cmap ensured
- 6.2.11.6-3: Symbolic TrueType must NOT have /Encoding; Symbolic flag must be set
- 6.2.11.6-4: Symbolic TrueType with multiple cmaps must have a (3,0) subtable

Additionally, embedded non-symbolic Type1/MMType1 fonts without an
/Encoding entry get /WinAnsiEncoding added.

This sanitizer must run after font subsetting: cmap subtables added here
would otherwise be pruned by the subsetter.
"""

import logging
from io import BytesIO

import pikepdf
from fontTools.agl import AGL2UV
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import cmap_format_4
from pikepdf import Dictionary, Name, Pdf, Stream

from ..fonts.analysis import is_symbolic_font
from ..fonts.traversal import iter_acroform_dr_fonts, iter_all_page_fonts
from ..fonts.utils import safe_str as _safe_str
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Encoding names considered compliant for non-symbolic TrueType fonts (6.2.11.6-2)
_VALID_ENCODINGS = frozenset({"/WinAnsiEncoding", "/MacRomanEncoding"})


def _is_agl_glyph_name(glyph_name: str) -> bool:
    """Return True when a glyph name is explicitly listed in Adobe Glyph List."""
    return glyph_name == ".notdef" or glyph_name in AGL2UV


def sanitize_truetype_encoding(pdf: Pdf) -> dict[str, int]:
    """Fixes TrueType font encoding issues for PDF/A compliance.

    Iterates all embedded TrueType fonts and applies fixes per
    ISO 19005-2, clause 6.2.11.6.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with counts of fixes applied:
        - tt_nonsymbolic_cmap_added: (3,1) cmap added to non-symbolic font
        - tt_nonsymbolic_encoding_fixed: /Encoding fixed for non-symbolic font
        - tt_symbolic_encoding_removed: /Encoding removed from symbolic font
        - tt_symbolic_flag_set: Symbolic bit set in /Flags
        - tt_symbolic_cmap_added: (3,0) cmap added to symbolic font
        - type1_encoding_added: /WinAnsiEncoding added to Type1/MMType1 font
    """
    result: dict[str, int] = {
        "tt_nonsymbolic_cmap_added": 0,
        "tt_nonsymbolic_encoding_fixed": 0,
        "tt_symbolic_encoding_removed": 0,
        "tt_symbolic_flag_set": 0,
        "tt_symbolic_cmap_added": 0,
        "type1_encoding_added": 0,
    }

    for font, fd, subtype in _iter_embedded_simple_fonts(pdf):
        try:
            if subtype in ("/Type1", "/MMType1"):
                _fix_type1_encoding(font, result)
            elif is_symbolic_font(font):
                _fix_symbolic_truetype(pdf, font, fd, result)
            else:
                _fix_nonsymbolic_truetype(pdf, font, fd, result)
        except Exception as e:
            name = _safe_str(font.get("/BaseFont") or b"")
            log_suppressed_error(logger, e, "Error processing font %s: %s", name, e)
            continue

    total = sum(result.values())
    if total > 0:
        logger.info(
            "TrueType encoding sanitization: %d (3,1) cmaps added, "
            "%d encodings fixed, %d /Encoding entries removed, "
            "%d Symbolic flags set, %d (3,0) cmaps added, "
            "%d Type1 encodings added",
            result["tt_nonsymbolic_cmap_added"],
            result["tt_nonsymbolic_encoding_fixed"],
            result["tt_symbolic_encoding_removed"],
            result["tt_symbolic_flag_set"],
            result["tt_symbolic_cmap_added"],
            result["type1_encoding_added"],
        )

    return result


def _get_truetype_font_stream(
    fd: pikepdf.Object,
) -> tuple[str, pikepdf.Object] | None:
    """Returns the embedded TrueType/OpenType stream and descriptor key.

    Simple TrueType font dictionaries may embed an OpenType program through
    ``/FontFile3 /OpenType`` as well as a TrueType program through
    ``/FontFile2``.  Other FontFile3 subtypes contain different program
    formats and are not handled by ``TTFont`` here.

    Args:
        fd: Resolved FontDescriptor Dictionary.

    Returns:
        Tuple of descriptor key and resolved font stream, or None when no
        supported embedded program is present.
    """
    font_file2 = fd.get("/FontFile2")
    if font_file2 is not None:
        try:
            return "/FontFile2", _resolve(font_file2)
        except Exception:
            return None

    font_file3 = fd.get("/FontFile3")
    if font_file3 is None:
        return None

    try:
        stream = _resolve(font_file3)
        if not isinstance(stream, Stream):
            return None
        if _safe_str(stream.get("/Subtype") or b"") != "/OpenType":
            return None
        return "/FontFile3", stream
    except Exception:
        return None


def _iter_embedded_simple_fonts(pdf: Pdf):
    """Yields (font, fd, subtype) tuples for embedded simple fonts.

    Covers TrueType fonts with /FontFile2 or /FontFile3 /OpenType and
    Type1/MMType1 fonts with any embedded font program.  Scans page
    resources (recursively) and AcroForm /DR Default Resources.

    Args:
        pdf: Opened pikepdf PDF object.

    Yields:
        Tuples of (resolved font Dictionary, resolved FontDescriptor
        Dictionary, subtype string).
    """
    seen_objgens: set[tuple[int, int]] = set()

    def _all_fonts():
        for page in pdf.pages:
            yield from iter_all_page_fonts(pikepdf.Page(page))
        # AcroForm /DR fonts render widget field appearances and must
        # satisfy the same encoding rules as page fonts.
        yield from iter_acroform_dr_fonts(pdf)

    for _font_key, font_obj in _all_fonts():
        try:
            entry = _as_embedded_simple_font(font_obj, seen_objgens)
            if entry is not None:
                yield entry
        except Exception as e:
            log_suppressed_error(logger, e, "Error iterating font: %s", e)
            continue


def _as_embedded_simple_font(
    font_obj: pikepdf.Object,
    seen_objgens: set[tuple[int, int]],
) -> tuple[pikepdf.Object, pikepdf.Object, str] | None:
    """Returns (font, fd, subtype) for an embedded simple font, or None.

    Deduplicates indirect fonts via ``seen_objgens`` and filters out
    unsupported subtypes and non-embedded fonts.
    """
    font = _resolve(font_obj)
    if not isinstance(font, Dictionary):
        return None

    # Deduplicate by objgen
    objgen = font.objgen
    if objgen != (0, 0):
        if objgen in seen_objgens:
            return None
        seen_objgens.add(objgen)

    subtype = font.get("/Subtype")
    if subtype is None:
        return None
    subtype_str = _safe_str(subtype)
    if subtype_str not in ("/TrueType", "/Type1", "/MMType1"):
        return None

    # Must have a FontDescriptor
    fd_obj = font.get("/FontDescriptor")
    if fd_obj is None:
        return None
    fd = _resolve(fd_obj)
    if not isinstance(fd, Dictionary):
        return None

    # Must be embedded: TrueType handling supports /FontFile2 and
    # /FontFile3 /OpenType; Type1/MMType1 accepts any embedded
    # font program.
    if subtype_str == "/TrueType":
        if _get_truetype_font_stream(fd) is None:
            return None
    elif all(fd.get(key) is None for key in ("/FontFile", "/FontFile2", "/FontFile3")):
        return None

    return font, fd, subtype_str


def _load_tt_font(fd: pikepdf.Object) -> TTFont | None:
    """Loads a TTFont from a supported stream in a FontDescriptor.

    Args:
        fd: Resolved FontDescriptor Dictionary.

    Returns:
        TTFont object, or None on failure.
    """
    try:
        font_file_info = _get_truetype_font_stream(fd)
        if font_file_info is None:
            return None
        _font_file_key, font_file = font_file_info
        data = bytes(font_file.read_bytes())
        return TTFont(BytesIO(data))
    except Exception as e:
        log_suppressed_error(
            logger, e, "Could not load embedded TrueType/OpenType font: %s", e
        )
        return None


def _save_tt_font(pdf: Pdf, fd: pikepdf.Object, tt_font: TTFont) -> None:
    """Saves a modified TTFont back to its original descriptor entry.

    Args:
        pdf: The PDF to make indirect objects in.
        fd: Resolved FontDescriptor Dictionary.
        tt_font: The modified TTFont to save.
    """
    font_file_info = _get_truetype_font_stream(fd)
    if font_file_info is None:
        raise ValueError("FontDescriptor has no supported TrueType/OpenType stream")
    font_file_key, _old_stream = font_file_info

    out = BytesIO()
    tt_font.save(out)
    new_font_data = out.getvalue()
    new_stream = Stream(pdf, new_font_data)
    if font_file_key == "/FontFile2":
        new_stream[Name.Length1] = len(new_font_data)
    else:
        new_stream[Name.Subtype] = Name.OpenType
    fd[Name(font_file_key)] = pdf.make_indirect(new_stream)


def _fix_type1_encoding(
    font: pikepdf.Object,
    result: dict[str, int],
) -> None:
    """Adds /WinAnsiEncoding to embedded non-symbolic Type1/MMType1 fonts.

    Args:
        font: Resolved font Dictionary.
        result: Result counters dict (modified in place).
    """
    if font.get("/Encoding") is not None:
        return
    if is_symbolic_font(font):
        return
    font[Name.Encoding] = Name.WinAnsiEncoding
    result["type1_encoding_added"] += 1
    logger.info(
        "Added /WinAnsiEncoding to Type1 font: %s (ISO 19005-2, 6.2.11.6)",
        _safe_str(font.get("/BaseFont") or b""),
    )


def _fix_nonsymbolic_truetype(
    pdf: Pdf,
    font: pikepdf.Object,
    fd: pikepdf.Object,
    result: dict[str, int],
) -> None:
    """Applies rules 6.2.11.6-1 and 6.2.11.6-2 for non-symbolic TrueType.

    Args:
        pdf: The PDF object.
        font: Resolved font Dictionary.
        fd: Resolved FontDescriptor Dictionary.
        result: Result counters dict (modified in place).
    """
    # Rule 6.2.11.6-2 first (encoding dict), then rule 6.2.11.6-1 (cmap)
    _apply_rule_6_2_11_6_2(pdf, font, fd, result)
    _apply_rule_6_2_11_6_1(pdf, fd, result)


def _fix_symbolic_truetype(
    pdf: Pdf,
    font: pikepdf.Object,
    fd: pikepdf.Object,
    result: dict[str, int],
) -> None:
    """Applies rules 6.2.11.6-3 and 6.2.11.6-4 for symbolic TrueType.

    Args:
        pdf: The PDF object.
        font: Resolved font Dictionary.
        fd: Resolved FontDescriptor Dictionary.
        result: Result counters dict (modified in place).
    """
    _apply_rule_6_2_11_6_3(font, fd, result)
    _apply_rule_6_2_11_6_4(pdf, fd, result)


def _apply_rule_6_2_11_6_1(
    pdf: Pdf,
    fd: pikepdf.Object,
    result: dict[str, int],
) -> None:
    """Rule 6.2.11.6-1: Non-symbolic TrueType must have non-(3,0)-only cmap.

    If the font program has ONLY a (3,0) cmap subtable, a (3,1) Microsoft
    Unicode subtable is added, mapping codes from the 0xF000 range to their
    plain Unicode equivalents.

    Args:
        pdf: The PDF object.
        fd: Resolved FontDescriptor Dictionary.
        result: Result counters dict (modified in place).
    """
    tt_font = _load_tt_font(fd)
    if tt_font is None:
        return

    try:
        cmap_table = tt_font.get("cmap")
        if cmap_table is None:
            return

        subtables = cmap_table.tables

        # Check if any subtable is NOT (3,0) — if so, already compliant
        for st in subtables:
            if not (st.platformID == 3 and st.platEncID == 0):
                return  # At least one non-(3,0) subtable → compliant

        # All subtables are (3,0) — find the (3,0) to derive (3,1) from
        source_30 = None
        for st in subtables:
            if st.platformID == 3 and st.platEncID == 0 and st.cmap:
                source_30 = st
                break

        if source_30 is None:
            return  # No usable (3,0) source

        # Build (3,1) mapping: strip 0xF000 prefix for symbol-range codes
        new_mapping: dict[int, str] = {}
        for code, glyph_name in source_30.cmap.items():
            if 0xF000 <= code <= 0xF0FF:
                new_mapping[code & 0xFF] = glyph_name
            else:
                new_mapping[code] = glyph_name

        new_subtable = cmap_format_4(4)
        new_subtable.platformID = 3
        new_subtable.platEncID = 1
        new_subtable.language = 0
        new_subtable.cmap = new_mapping
        cmap_table.tables.append(new_subtable)

        _save_tt_font(pdf, fd, tt_font)
        result["tt_nonsymbolic_cmap_added"] += 1
        logger.info("Added (3,1) cmap to non-symbolic TrueType font (rule 6.2.11.6-1)")

    finally:
        tt_font.close()


def _apply_rule_6_2_11_6_2(
    pdf: Pdf,
    font: pikepdf.Object,
    fd: pikepdf.Object,
    result: dict[str, int],
) -> None:
    """Rule 6.2.11.6-2: Non-symbolic TrueType /Encoding must be compliant.

    - If no /Encoding: add /WinAnsiEncoding Name
    - If /Encoding is a Name: must be WinAnsiEncoding or MacRomanEncoding
    - If /Encoding is a Dictionary: BaseEncoding must be compliant; Differences
      must use only AGL glyph names.  Fonts that keep valid Differences also
      get a Microsoft Unicode (3,1) cmap ensured in the font program so the
      AGL names can be mapped reliably.

    Args:
        pdf: The PDF object.
        font: Resolved font Dictionary.
        fd: Resolved FontDescriptor Dictionary.
        result: Result counters dict (modified in place).
    """
    encoding_obj = font.get("/Encoding")

    if encoding_obj is None:
        # No /Encoding — add WinAnsiEncoding
        font[Name.Encoding] = Name.WinAnsiEncoding
        result["tt_nonsymbolic_encoding_fixed"] += 1
        return

    try:
        encoding_obj = _resolve(encoding_obj)
    except Exception:
        return

    if isinstance(encoding_obj, Name):
        enc_str = _safe_str(encoding_obj)
        if enc_str in _VALID_ENCODINGS:
            return  # Already compliant
        # Wrong encoding name — replace with WinAnsiEncoding
        font[Name.Encoding] = Name.WinAnsiEncoding
        result["tt_nonsymbolic_encoding_fixed"] += 1
        return

    if isinstance(encoding_obj, Dictionary):
        changed = False

        # Check /BaseEncoding
        base_enc = encoding_obj.get("/BaseEncoding")
        if base_enc is None:
            encoding_obj[Name.BaseEncoding] = Name.WinAnsiEncoding
            changed = True
        else:
            try:
                base_enc = _resolve(base_enc)
                base_str = _safe_str(base_enc)
                if base_str not in _VALID_ENCODINGS:
                    encoding_obj[Name.BaseEncoding] = Name.WinAnsiEncoding
                    changed = True
            except Exception:
                encoding_obj[Name.BaseEncoding] = Name.WinAnsiEncoding
                changed = True

        # Check /Differences — remove if any non-AGL names
        differences = encoding_obj.get("/Differences")
        if differences is not None:
            keep_differences = False
            try:
                differences = _resolve(differences)
                if _has_non_agl_differences(differences):
                    del encoding_obj["/Differences"]
                    changed = True
                else:
                    keep_differences = True
            except Exception:
                del encoding_obj["/Differences"]
                changed = True

            # Valid AGL Differences stay — ensure the font program has a
            # Microsoft Unicode (3,1) cmap so the names map reliably.
            if keep_differences and _ensure_ms_unicode_cmap(pdf, fd):
                result["tt_nonsymbolic_cmap_added"] += 1

        if changed:
            result["tt_nonsymbolic_encoding_fixed"] += 1


def _ensure_ms_unicode_cmap(pdf: Pdf, fd: pikepdf.Object) -> bool:
    """Ensures the font program has a Microsoft Unicode (3,1) cmap subtable.

    Builds the (3,1) subtable from the best available source cmap.  Mac
    Roman byte codes >= 0x80 are not Unicode code points and are translated
    before being used as (3,1) keys.

    Args:
        pdf: The PDF object.
        fd: Resolved FontDescriptor Dictionary.

    Returns:
        True if the font program was modified, False otherwise.
    """
    tt_font = _load_tt_font(fd)
    if tt_font is None:
        return False

    try:
        cmap_table = tt_font.get("cmap")
        if cmap_table is None:
            return False

        # Check if (3,1) already exists
        for st in cmap_table.tables:
            if st.platformID == 3 and st.platEncID == 1:
                return False

        source = _find_best_cmap_source(cmap_table.tables)
        if source is None:
            return False

        new_subtable = cmap_format_4(4)
        new_subtable.platformID = 3
        new_subtable.platEncID = 1
        new_subtable.language = 0
        if source.platformID == 1 and source.platEncID == 0:
            new_mapping: dict[int, str] = {}
            for code, glyph_name in source.cmap.items():
                if 0 <= code <= 0xFF:
                    code = ord(bytes([code]).decode("mac_roman"))
                new_mapping[code] = glyph_name
            new_subtable.cmap = new_mapping
        else:
            new_subtable.cmap = dict(source.cmap)

        cmap_table.tables.append(new_subtable)

        _save_tt_font(pdf, fd, tt_font)
        logger.info(
            "Added Microsoft Unicode (3,1) cmap for TrueType font with "
            "Differences (rule 6.2.11.6-2)"
        )
        return True

    except Exception as e:
        log_suppressed_error(logger, e, "Error ensuring MS Unicode (3,1) cmap: %s", e)
        return False

    finally:
        tt_font.close()


def _apply_rule_6_2_11_6_3(
    font: pikepdf.Object,
    fd: pikepdf.Object,
    result: dict[str, int],
) -> None:
    """Rule 6.2.11.6-3: Symbolic TrueType must not have /Encoding; Symbolic bit set.

    Args:
        font: Resolved font Dictionary.
        fd: Resolved FontDescriptor Dictionary.
        result: Result counters dict (modified in place).
    """
    # Remove /Encoding if present
    if font.get("/Encoding") is not None:
        del font["/Encoding"]
        result["tt_symbolic_encoding_removed"] += 1

    # Ensure Symbolic bit (bit 3 = value 4) is set in /Flags
    flags_obj = fd.get("/Flags")
    try:
        flags = int(flags_obj) if flags_obj is not None else 0
    except (TypeError, ValueError):
        flags = 0

    if not (flags & 4):
        fd[Name.Flags] = flags | 4
        result["tt_symbolic_flag_set"] += 1


def _apply_rule_6_2_11_6_4(
    pdf: Pdf,
    fd: pikepdf.Object,
    result: dict[str, int],
) -> None:
    """Rule 6.2.11.6-4: Symbolic TrueType with multiple cmaps must have (3,0).

    If the font has exactly one cmap subtable, it is already compliant.
    If it has multiple but already has a non-empty (3,0), it is compliant.
    Otherwise, a (3,0) subtable is added/repaired in the 0xF000 range.

    Args:
        pdf: The PDF object.
        fd: Resolved FontDescriptor Dictionary.
        result: Result counters dict (modified in place).
    """
    tt_font = _load_tt_font(fd)
    if tt_font is None:
        return

    try:
        cmap_table = tt_font.get("cmap")
        if cmap_table is None:
            return

        subtables = cmap_table.tables

        # Exactly one subtable → compliant
        if len(subtables) == 1:
            return

        # Check if (3,0) already exists and is non-empty
        existing_30 = None
        for st in subtables:
            if st.platformID == 3 and st.platEncID == 0:
                existing_30 = st
                break

        if existing_30 is not None and existing_30.cmap:
            return  # Already compliant

        # Find best source subtable (prefer (1,0), then (3,1), then first)
        source_subtables = [
            st
            for st in subtables
            if st.cmap and not (st.platformID == 3 and st.platEncID == 0)
        ]
        source = _find_best_cmap_source(source_subtables)
        if source is None:
            return

        # Build (3,0) mapping in 0xF000 range
        new_mapping: dict[int, str] = {}
        for code, glyph_name in source.cmap.items():
            sym_code = (code & 0xFF) | 0xF000
            new_mapping[sym_code] = glyph_name

        if existing_30 is not None:
            # Repair empty (3,0) in-place
            existing_30.cmap = new_mapping
        else:
            new_subtable = cmap_format_4(4)
            new_subtable.platformID = 3
            new_subtable.platEncID = 0
            new_subtable.language = 0
            new_subtable.cmap = new_mapping
            cmap_table.tables.append(new_subtable)

        _save_tt_font(pdf, fd, tt_font)
        result["tt_symbolic_cmap_added"] += 1
        logger.info(
            "%s (3,0) cmap for symbolic TrueType font (rule 6.2.11.6-4)",
            "Repaired" if existing_30 is not None else "Added",
        )

    finally:
        tt_font.close()


def _find_best_cmap_source(subtables: list) -> object | None:
    """Finds the best source cmap subtable.

    Priority: (1,0) Mac Roman → (3,1) MS Unicode → first available.

    Args:
        subtables: List of cmap subtables with non-empty .cmap dicts.

    Returns:
        Best subtable, or None if no subtables are available.
    """
    mac_roman = None
    ms_unicode = None
    for st in subtables:
        if st.platformID == 1 and st.platEncID == 0:
            mac_roman = st
        elif st.platformID == 3 and st.platEncID == 1:
            ms_unicode = st

    if mac_roman is not None:
        return mac_roman
    if ms_unicode is not None:
        return ms_unicode
    if subtables:
        return subtables[0]
    return None


def _has_non_agl_differences(differences: pikepdf.Object) -> bool:
    """Checks if a /Differences array contains non-AGL glyph names.

    Args:
        differences: The /Differences array from an Encoding dict.

    Returns:
        True if any glyph name is not in the Adobe Glyph List.
    """
    for item in differences:
        if isinstance(item, pikepdf.Name):
            glyph_name = _safe_str(item).lstrip("/")
            if not _is_agl_glyph_name(glyph_name):
                return True
    return False
