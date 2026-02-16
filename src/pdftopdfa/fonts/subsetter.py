# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font subsetting for embedded replacement fonts.

Reduces file size by removing unused glyphs from embedded TrueType
and CFF/OpenType fonts. Uses fontTools.subset with retain_gids=True
to keep GIDs stable, so content streams and CIDToGIDMap do not need
modification.
"""

import functools
import logging
import random
import string
from dataclasses import dataclass, field
from io import BytesIO

import pikepdf
from pikepdf import Name, Stream

from ..utils import resolve_indirect as _resolve_indirect
from .analysis import (
    _is_subset_font,
    get_base_font_name,
    get_font_name,
    get_font_type,
)
from .encodings import STANDARD_ENCODING
from .glyph_usage import collect_font_usage
from .tounicode import (
    generate_cidfont_tounicode_cmap,
    generate_tounicode_cmap_data,
    generate_tounicode_for_macroman,
    generate_tounicode_for_winansi,
    parse_tounicode_cmap,
    resolve_glyph_to_unicode,
)
from .traversal import iter_all_page_fonts
from .utils import check_fstype_restrictions, get_fstype
from .utils import safe_str as _safe_str

logger = logging.getLogger(__name__)


def _generate_subset_prefix() -> str:
    """Generates a random 6-letter uppercase subset prefix.

    Returns:
        String like "ABCDEF+" for use as a font subset tag.
    """
    letters = "".join(random.choices(string.ascii_uppercase, k=6))
    return f"{letters}+"


@dataclass
class SubsettingResult:
    """Result of font subsetting.

    Attributes:
        fonts_subsetted: List of font names that were subsetted.
        fonts_skipped: List of font names that were skipped (with reason).
        warnings: List of warnings during subsetting.
        bytes_saved: Total bytes saved by subsetting.
    """

    fonts_subsetted: list[str] = field(default_factory=list)
    fonts_skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bytes_saved: int = 0


class FontSubsetter:
    """Subsets embedded fonts to reduce file size.

    Subsets TrueType (FontFile2) and CFF/OpenType (FontFile3 with
    /Subtype /OpenType) fonts. Type3 fonts, fonts without embedded
    data, already-subsetted fonts (from the original PDF), bare CFF
    programs (CIDFontType0C, Type1C), and FontFile (Type1) are skipped.
    """

    def __init__(self, pdf: pikepdf.Pdf) -> None:
        """Initializes the FontSubsetter.

        Args:
            pdf: Opened pikepdf PDF object.
        """
        self.pdf = pdf

    def subset_all_fonts(self) -> SubsettingResult:
        """Subsets all eligible embedded fonts in the PDF.

        Returns:
            SubsettingResult with subsetting status.
        """
        result = SubsettingResult()

        # Collect glyph usage across all content streams
        font_usage = collect_font_usage(self.pdf)

        # Iterate all fonts and subset eligible ones
        processed_ids: set[tuple[int, int]] = set()

        for page in self.pdf.pages:
            for font_key, font_obj in iter_all_page_fonts(page):
                try:
                    obj_key = font_obj.objgen
                    if obj_key == (0, 0):
                        continue
                    if obj_key in processed_ids:
                        continue
                    processed_ids.add(obj_key)

                    self._subset_font(font_obj, obj_key, font_usage, result)
                except Exception as e:
                    font_name = _safe_font_name(font_obj)
                    warning = f"Error processing font '{font_name}': {e}"
                    result.warnings.append(warning)
                    logger.debug(warning)

        return result

    def _subset_font(
        self,
        font_obj: pikepdf.Object,
        obj_key: tuple[int, int],
        font_usage: dict[tuple[int, int], set[int]],
        result: SubsettingResult,
    ) -> None:
        """Attempts to subset a single font.

        Args:
            font_obj: The font object.
            obj_key: The font's objgen tuple.
            font_usage: Character code usage map.
            result: Result accumulator.
        """
        font_name = _safe_font_name(font_obj)
        font_type = get_font_type(font_obj)

        # Skip Type3 fonts (procedurally defined, no font program)
        if font_type == "Type3":
            result.fonts_skipped.append(f"{font_name} (Type3)")
            return

        # Skip already-subsetted fonts (from the original PDF)
        if _is_subset_font(font_name):
            result.fonts_skipped.append(f"{font_name} (already subsetted)")
            return

        # Route to CIDFont or simple font handler
        if font_type == "CIDFont":
            self._subset_cidfont(font_obj, obj_key, font_usage, font_name, result)
        else:
            self._subset_simple_font(font_obj, obj_key, font_usage, font_name, result)

    def _subset_cidfont(
        self,
        font_obj: pikepdf.Object,
        obj_key: tuple[int, int],
        font_usage: dict[tuple[int, int], set[int]],
        font_name: str,
        result: SubsettingResult,
    ) -> None:
        """Subsets a CIDFont (Type0) with FontFile2 or FontFile3.

        For CIDFonts with Identity-H/V encoding, character codes
        directly correspond to GIDs. With retain_gids=True, the
        subsetter preserves all GID slots, just clears unused ones.

        Args:
            font_obj: The Type0 font object.
            obj_key: The font's objgen tuple.
            font_usage: Character code usage map.
            font_name: Font name for logging.
            result: Result accumulator.
        """
        # Get DescendantFonts
        descendants = font_obj.get("/DescendantFonts")
        if descendants is None or len(descendants) == 0:
            result.fonts_skipped.append(f"{font_name} (no DescendantFonts)")
            return

        desc_font = _resolve_indirect(descendants[0])

        # Get FontDescriptor
        font_descriptor = desc_font.get("/FontDescriptor")
        if font_descriptor is None:
            result.fonts_skipped.append(f"{font_name} (no FontDescriptor)")
            return
        font_descriptor = _resolve_indirect(font_descriptor)

        # Find FontFile2 (TrueType) or FontFile3 (CFF/OpenType)
        font_file_info = _find_font_file(font_descriptor)
        if font_file_info is None:
            result.fonts_skipped.append(f"{font_name} (no FontFile2 or FontFile3)")
            return
        font_file = font_file_info.stream

        # Get used character codes (= GIDs for Identity-H/V)
        used_codes = font_usage.get(obj_key, set())

        # Perform subsetting
        try:
            original_data = bytes(font_file.read_bytes())
            original_size = len(original_data)

            # Check fsType embedding restrictions
            if not _check_subsetting_allowed(original_data, font_name, result):
                return

            subsetted_data = _subset_font_data(original_data, used_codes, is_cid=True)

            if subsetted_data is None:
                result.fonts_skipped.append(f"{font_name} (subsetting failed)")
                return

            new_size = len(subsetted_data)
            saved = original_size - new_size

            if saved <= 0:
                result.fonts_skipped.append(f"{font_name} (no size reduction)")
                return

            # Write subsetted data back
            new_stream = Stream(self.pdf, subsetted_data)
            if font_file_info.is_fontfile3:
                # Preserve /Subtype from original FontFile3 stream
                original_subtype = font_file_info.stream.get("/Subtype")
                if original_subtype is not None:
                    new_stream[Name.Subtype] = original_subtype
            else:
                new_stream[Name.Length1] = new_size
            font_descriptor[font_file_info.descriptor_key] = self.pdf.make_indirect(
                new_stream
            )

            # Add subset prefix
            prefix = _generate_subset_prefix()
            base_name = get_base_font_name(font_name)
            new_name = f"{prefix}{base_name}"

            font_obj[Name.BaseFont] = Name(f"/{new_name}")
            desc_font[Name.BaseFont] = Name(f"/{new_name}")
            font_descriptor[Name.FontName] = Name(f"/{new_name}")

            result.fonts_subsetted.append(font_name)
            result.bytes_saved += saved
            logger.info(
                "Subsetted CIDFont '%s' -> '%s' (saved %d bytes)",
                font_name,
                new_name,
                saved,
            )

            # Clean stale ToUnicode entries
            _clean_tounicode(font_obj, used_codes, is_cid=True, pdf=self.pdf)

        except Exception as e:
            warning = f"Error subsetting CIDFont '{font_name}': {e}"
            result.warnings.append(warning)
            logger.debug(warning)

    def _subset_simple_font(
        self,
        font_obj: pikepdf.Object,
        obj_key: tuple[int, int],
        font_usage: dict[tuple[int, int], set[int]],
        font_name: str,
        result: SubsettingResult,
    ) -> None:
        """Subsets a simple font (Type1, TrueType) with FontFile2 or FontFile3.

        For simple fonts, character codes map to glyphs through the
        font's Encoding. With retain_gids=True, GIDs stay stable.

        Args:
            font_obj: The font object.
            obj_key: The font's objgen tuple.
            font_usage: Character code usage map.
            font_name: Font name for logging.
            result: Result accumulator.
        """
        # Get FontDescriptor
        font_descriptor = font_obj.get("/FontDescriptor")
        if font_descriptor is None:
            result.fonts_skipped.append(f"{font_name} (no FontDescriptor)")
            return
        font_descriptor = _resolve_indirect(font_descriptor)

        # Find FontFile2 (TrueType) or FontFile3 (CFF/OpenType)
        font_file_info = _find_font_file(font_descriptor)
        if font_file_info is None:
            result.fonts_skipped.append(f"{font_name} (no FontFile2 or FontFile3)")
            return
        font_file = font_file_info.stream

        # Get used character codes
        used_codes = font_usage.get(obj_key, set())

        # Resolve encoding for precise glyph selection
        encoding_map = _resolve_simple_font_encoding(font_obj)

        # Perform subsetting
        try:
            original_data = bytes(font_file.read_bytes())
            original_size = len(original_data)

            # For symbolic TrueType fonts without encoding, build a
            # code-to-glyph-name mapping from the font's own cmap so
            # the subsetter retains the correct glyphs.
            is_symbolic = False
            if encoding_map is None:
                encoding_map = _build_symbolic_truetype_encoding(
                    font_obj, original_data
                )
                if encoding_map is not None:
                    is_symbolic = True

            # Check fsType embedding restrictions
            if not _check_subsetting_allowed(original_data, font_name, result):
                return

            subsetted_data = _subset_font_data(
                original_data,
                used_codes,
                is_cid=False,
                code_to_glyphname=encoding_map,
            )

            # For symbolic TrueType fonts, the cmap subtable gets
            # dropped by fontTools (it only preserves cmaps when
            # subsetting by unicode).  Rebuild the (3,0) cmap from
            # the original font's cmap data.
            if is_symbolic and subsetted_data is not None:
                subsetted_data = _rebuild_symbolic_cmap(original_data, subsetted_data)

            if subsetted_data is None:
                result.fonts_skipped.append(f"{font_name} (subsetting failed)")
                return

            new_size = len(subsetted_data)
            saved = original_size - new_size

            if saved <= 0:
                result.fonts_skipped.append(f"{font_name} (no size reduction)")
                return

            # Write subsetted data back
            new_stream = Stream(self.pdf, subsetted_data)
            if font_file_info.is_fontfile3:
                # Preserve /Subtype from original FontFile3 stream
                original_subtype = font_file_info.stream.get("/Subtype")
                if original_subtype is not None:
                    new_stream[Name.Subtype] = original_subtype
            else:
                new_stream[Name.Length1] = new_size
            font_descriptor[font_file_info.descriptor_key] = self.pdf.make_indirect(
                new_stream
            )

            # Add subset prefix
            prefix = _generate_subset_prefix()
            base_name = get_base_font_name(font_name)
            new_name = f"{prefix}{base_name}"

            font_obj[Name.BaseFont] = Name(f"/{new_name}")
            font_descriptor[Name.FontName] = Name(f"/{new_name}")

            result.fonts_subsetted.append(font_name)
            result.bytes_saved += saved
            logger.info(
                "Subsetted font '%s' -> '%s' (saved %d bytes)",
                font_name,
                new_name,
                saved,
            )

            # Clean stale ToUnicode entries
            _clean_tounicode(font_obj, used_codes, is_cid=False, pdf=self.pdf)

        except Exception as e:
            warning = f"Error subsetting font '{font_name}': {e}"
            result.warnings.append(warning)
            logger.debug(warning)


def _check_subsetting_allowed(
    font_data: bytes,
    font_name: str,
    result: SubsettingResult,
) -> bool:
    """Checks if font subsetting is allowed by the font's fsType field.

    Reads the OS/2 fsType field and checks for the No Subsetting bit
    (0x0100) and Restricted License bit (0x0002). If subsetting is
    prohibited, the font is added to the skipped list. Other
    restrictions are logged as warnings.

    Args:
        font_data: Raw font bytes.
        font_name: Font name for logging and result tracking.
        result: Result accumulator to add warnings/skipped entries to.

    Returns:
        True if subsetting is allowed, False if it should be skipped.
    """
    fstype = get_fstype(font_data)
    if fstype is None:
        return True

    embedding_allowed, subsetting_allowed, warnings = check_fstype_restrictions(fstype)

    for warning in warnings:
        msg = f"Font '{font_name}': {warning}"
        result.warnings.append(msg)
        logger.warning(msg)

    if not embedding_allowed:
        result.fonts_skipped.append(f"{font_name} (fsType: embedding not allowed)")
        return False

    if not subsetting_allowed:
        result.fonts_skipped.append(f"{font_name} (fsType: no subsetting allowed)")
        return False

    return True


def _safe_font_name(font_obj: pikepdf.Object) -> str:
    """Gets font name safely, handling errors.

    Args:
        font_obj: pikepdf font object.

    Returns:
        Font name string, or "Unknown" on error.
    """
    try:
        return get_font_name(font_obj)
    except Exception:
        return "Unknown"


def _clean_tounicode(
    font_obj: pikepdf.Object,
    used_codes: set[int],
    *,
    is_cid: bool,
    pdf: pikepdf.Pdf,
) -> None:
    """Removes stale entries from a font's ToUnicode CMap after subsetting.

    Filters the CMap to retain only entries for character codes that are
    actually used. If nothing was removed, the stream is left unchanged.

    Args:
        font_obj: The font dictionary (Type0 or simple font).
        used_codes: Set of character codes used in content streams.
        is_cid: True if the font is a CIDFont (16-bit codes).
        pdf: The pikepdf Pdf object (needed to create new streams).
    """
    tounicode = font_obj.get("/ToUnicode")
    if tounicode is None:
        return

    tounicode = _resolve_indirect(tounicode)

    try:
        raw_data = bytes(tounicode.read_bytes())
    except Exception:
        return

    parsed = parse_tounicode_cmap(raw_data)
    if not parsed:
        return

    # Filter to only used codes
    filtered = {code: uni for code, uni in parsed.items() if code in used_codes}

    if len(filtered) == len(parsed):
        return

    # Regenerate CMap
    if is_cid:
        new_data = generate_cidfont_tounicode_cmap(filtered)
    else:
        new_data = generate_tounicode_cmap_data(filtered)

    new_stream = Stream(pdf, new_data)
    font_obj[pikepdf.Name.ToUnicode] = pdf.make_indirect(new_stream)


@dataclass
class _FontFileInfo:
    """Information about a font file stream in a FontDescriptor.

    Attributes:
        stream: The resolved font file stream object.
        descriptor_key: The pikepdf Name key (/FontFile2 or /FontFile3).
        is_fontfile3: True if the stream is a FontFile3 entry.
    """

    stream: pikepdf.Object
    descriptor_key: Name
    is_fontfile3: bool


def _find_font_file(
    font_descriptor: pikepdf.Object,
) -> _FontFileInfo | None:
    """Finds the font file stream in a FontDescriptor.

    Checks /FontFile2 first (TrueType), then /FontFile3 (CFF/OpenType).
    For FontFile3, only /OpenType subtype is eligible for subsetting.

    Args:
        font_descriptor: Resolved FontDescriptor dictionary.

    Returns:
        _FontFileInfo or None if no eligible font file found.
    """
    # Check FontFile2 first (TrueType outlines)
    font_file = font_descriptor.get("/FontFile2")
    if font_file is not None:
        return _FontFileInfo(
            stream=_resolve_indirect(font_file),
            descriptor_key=Name.FontFile2,
            is_fontfile3=False,
        )

    # Check FontFile3 (CFF/OpenType)
    font_file = font_descriptor.get("/FontFile3")
    if font_file is not None:
        font_file = _resolve_indirect(font_file)
        subtype = font_file.get("/Subtype")
        if subtype is not None and str(subtype) == "/OpenType":
            return _FontFileInfo(
                stream=font_file,
                descriptor_key=Name.FontFile3,
                is_fontfile3=True,
            )

    return None


@functools.cache
def _get_uv2agl() -> dict[int, str]:
    """Returns a reverse Adobe Glyph List mapping (Unicode -> glyph name).

    Lazily built from fontTools AGL2UV on first call. When multiple
    glyph names map to the same Unicode value, the first encountered
    name wins (which is typically the canonical name).

    Returns:
        Dictionary mapping Unicode codepoints to glyph names.
    """
    from fontTools.agl import AGL2UV

    uv2agl: dict[int, str] = {}
    for name, uv in AGL2UV.items():
        if uv not in uv2agl:
            uv2agl[uv] = name
    return uv2agl


def _resolve_simple_font_encoding(
    font_obj: pikepdf.Object,
) -> dict[int, str] | None:
    """Builds a code-to-glyph-name mapping from a simple font's encoding.

    Resolves the font's /Encoding entry (Name or Dictionary with
    /BaseEncoding and /Differences) to map character codes to Adobe
    glyph names. This enables precise glyph selection during subsetting,
    avoiding over-retention from treating raw codes as Unicode values.

    For WinAnsiEncoding and MacRomanEncoding, glyph names are derived
    by resolving through Unicode (code -> Unicode -> reverse AGL).

    Args:
        font_obj: pikepdf simple font object.

    Returns:
        Dictionary mapping character codes to glyph names,
        or None if no encoding can be resolved.
    """
    encoding = font_obj.get("/Encoding")
    if encoding is None:
        return None

    if isinstance(encoding, pikepdf.Name):
        enc_name = _safe_str(encoding)
        if enc_name == "/WinAnsiEncoding":
            return _build_glyphnames_from_unicode(generate_tounicode_for_winansi())
        elif enc_name == "/MacRomanEncoding":
            return _build_glyphnames_from_unicode(generate_tounicode_for_macroman())
        elif enc_name == "/StandardEncoding":
            return dict(STANDARD_ENCODING)
        return None

    # Encoding dictionary with BaseEncoding + Differences
    try:
        enc_dict = _resolve_indirect(encoding)
        base = enc_dict.get("/BaseEncoding")
        if base is not None:
            base_name = _safe_str(base)
            if base_name == "/WinAnsiEncoding":
                code_to_glyphname = _build_glyphnames_from_unicode(
                    generate_tounicode_for_winansi()
                )
            elif base_name == "/MacRomanEncoding":
                code_to_glyphname = _build_glyphnames_from_unicode(
                    generate_tounicode_for_macroman()
                )
            elif base_name == "/StandardEncoding":
                code_to_glyphname = dict(STANDARD_ENCODING)
            else:
                code_to_glyphname = dict(STANDARD_ENCODING)
        else:
            # Default base encoding is StandardEncoding (PDF spec)
            code_to_glyphname = dict(STANDARD_ENCODING)

        # Apply /Differences array (overrides base encoding entries)
        differences = enc_dict.get("/Differences")
        if differences is not None:
            current_code = 0
            for item in differences:
                try:
                    current_code = int(item)
                    continue
                except (TypeError, ValueError):
                    pass
                if isinstance(item, pikepdf.Name):
                    glyph_name = _safe_str(item)[1:]  # Remove leading "/"
                    code_to_glyphname[current_code] = glyph_name
                    current_code += 1

        return code_to_glyphname if code_to_glyphname else None
    except Exception:
        return None


def _build_glyphnames_from_unicode(
    code_to_unicode: dict[int, int],
) -> dict[int, str]:
    """Derives code-to-glyph-name mapping from a code-to-Unicode mapping.

    Uses the reverse Adobe Glyph List to find glyph names for Unicode
    values. Codes whose Unicode values have no AGL entry are omitted.

    Args:
        code_to_unicode: Mapping from character codes to Unicode codepoints.

    Returns:
        Dictionary mapping character codes to glyph names.
    """
    uv2agl = _get_uv2agl()
    result: dict[int, str] = {}
    for code, uv in code_to_unicode.items():
        glyph_name = uv2agl.get(uv)
        if glyph_name is not None:
            result[code] = glyph_name
    return result


def _build_symbolic_truetype_encoding(
    font_obj: pikepdf.Object,
    font_data: bytes,
) -> dict[int, str] | None:
    """Builds code-to-glyph-name mapping for symbolic TrueType fonts.

    For symbolic TrueType fonts without an explicit /Encoding, character
    codes are mapped to glyphs through the font's own cmap tables:
    - (1,0) Mac Roman cmap: codes map directly
    - (3,0) Microsoft Symbol cmap: codes map via 0xF000 + code

    Args:
        font_obj: pikepdf font object.
        font_data: Raw font file bytes.

    Returns:
        Dictionary mapping character codes to glyph names,
        or None if the font is not a symbolic TrueType font or no
        suitable cmap is found.
    """
    fd = font_obj.get("/FontDescriptor")
    if fd is None:
        return None
    fd = _resolve_indirect(fd)
    flags = int(fd.get("/Flags", 0))
    if not (flags & 4):  # Not symbolic
        return None

    try:
        from fontTools.ttLib import TTFont

        tt = TTFont(BytesIO(font_data))
    except Exception:
        return None

    try:
        cmap_table = tt.get("cmap")
        if cmap_table is None:
            return None

        # Prefer (1,0) Mac cmap — direct code mapping
        for table in cmap_table.tables:
            if table.platformID == 1 and table.platEncID == 0:
                return dict(table.cmap)

        # Fall back to (3,0) Microsoft Symbol cmap — strip 0xF000 prefix
        for table in cmap_table.tables:
            if table.platformID == 3 and table.platEncID == 0:
                result = {}
                for unicode_val, glyph_name in table.cmap.items():
                    code = unicode_val & 0xFF
                    result[code] = glyph_name
                return result

        return None
    finally:
        tt.close()


def _rebuild_symbolic_cmap(
    original_data: bytes,
    subsetted_data: bytes,
) -> bytes:
    """Rebuilds cmap subtables for a symbolic TrueType font after subsetting.

    fontTools strips cmap subtables when subsetting by glyph names
    (no unicodes specified). This function restores the (3,0) Microsoft
    Symbol cmap from the original font, filtered to only include entries
    for glyphs that survived subsetting.

    Args:
        original_data: Original font data (before subsetting).
        subsetted_data: Subsetted font data (with missing cmap).

    Returns:
        Modified subsetted font data with restored cmap, or the
        original subsetted data unchanged on error.
    """
    try:
        from fontTools.ttLib import TTFont
        from fontTools.ttLib.tables._c_m_a_p import (
            cmap_format_4,
            table__c_m_a_p,
        )

        original_tt = TTFont(BytesIO(original_data))
        subsetted_tt = TTFont(BytesIO(subsetted_data))

        try:
            # Get original cmap (3,0)
            orig_cmap_30 = None
            orig_cmap = original_tt.get("cmap")
            if orig_cmap is not None:
                for table in orig_cmap.tables:
                    if table.platformID == 3 and table.platEncID == 0:
                        orig_cmap_30 = table.cmap
                        break

            if orig_cmap_30 is None:
                return subsetted_data

            # Build unicode→GID mapping using original font
            orig_glyph_order = original_tt.getGlyphOrder()
            orig_name_to_gid = {n: i for i, n in enumerate(orig_glyph_order)}

            # Map GIDs to new glyph names in subsetted font
            sub_glyph_order = subsetted_tt.getGlyphOrder()
            num_sub_glyphs = len(sub_glyph_order)

            new_cmap_30: dict[int, str] = {}
            for unicode_val, orig_name in orig_cmap_30.items():
                gid = orig_name_to_gid.get(orig_name)
                if gid is not None and gid < num_sub_glyphs:
                    new_cmap_30[unicode_val] = sub_glyph_order[gid]

            if not new_cmap_30:
                return subsetted_data

            # Build new cmap table with (3,0) subtable
            cmap_table = table__c_m_a_p()
            cmap_table.tableVersion = 0

            subtable = cmap_format_4(4)
            subtable.platformID = 3
            subtable.platEncID = 0
            subtable.format = 4
            subtable.reserved = 0
            subtable.length = 0
            subtable.language = 0
            subtable.cmap = new_cmap_30
            cmap_table.tables = [subtable]

            subsetted_tt["cmap"] = cmap_table

            # Save
            output = BytesIO()
            subsetted_tt.save(output)
            return output.getvalue()
        finally:
            original_tt.close()
            subsetted_tt.close()
    except Exception as e:
        logger.debug("Error rebuilding symbolic cmap: %s", e)
        return subsetted_data


def _subset_font_data(
    font_data: bytes,
    used_codes: set[int],
    *,
    is_cid: bool,
    code_to_glyphname: dict[int, str] | None = None,
) -> bytes | None:
    """Subsets TrueType or CFF/OpenType font data using fontTools.

    Args:
        font_data: Original font bytes.
        used_codes: Set of character codes / GIDs used.
        is_cid: True if the font is a CIDFont (codes = GIDs).
        code_to_glyphname: Optional mapping from character codes to
            Adobe glyph names (from the PDF font's /Encoding). When
            provided, enables precise glyph selection instead of
            treating raw codes as Unicode values.

    Returns:
        Subsetted font bytes, or None on error.
    """
    try:
        from fontTools.subset import Options, Subsetter
        from fontTools.ttLib import TTFont
    except ImportError:
        logger.warning("fontTools not available, skipping subsetting")
        return None

    try:
        tt_font = TTFont(BytesIO(font_data))

        # fontTools requires a cmap table for subsetting — fonts
        # without one (bare CFF wrapped in OpenType) cannot be subset.
        cmap_table = tt_font.get("cmap")
        if cmap_table is None or not any(t.cmap for t in cmap_table.tables):
            logger.debug("Font has no cmap table, skipping subsetting")
            tt_font.close()
            return None

        options = Options()
        options.retain_gids = True
        options.notdef_outline = True
        options.name_legacy = True
        options.name_IDs = ["*"]
        options.name_languages = ["*"]

        subsetter = Subsetter(options=options)

        if is_cid:
            # For CIDFonts, codes are GIDs directly
            # Convert GIDs to glyph names for fontTools
            glyph_order = tt_font.getGlyphOrder()
            glyph_names = set()
            for gid in used_codes:
                if 0 <= gid < len(glyph_order):
                    glyph_names.add(glyph_order[gid])
            # Always keep .notdef
            if glyph_order:
                glyph_names.add(glyph_order[0])
            subsetter.populate(glyphs=glyph_names)
        else:
            # For simple fonts, map character codes to glyphs through
            # the PDF encoding when available
            if code_to_glyphname:
                _populate_from_encoding(
                    subsetter, tt_font, used_codes, code_to_glyphname
                )
            else:
                # No encoding info — treat codes directly as Unicode
                # values (works for fonts without explicit /Encoding
                # where codes approximate Unicode)
                subsetter.populate(unicodes=used_codes)

        subsetter.subset(tt_font)

        # Write subsetted font to bytes
        output = BytesIO()
        tt_font.save(output)
        tt_font.close()

        return output.getvalue()

    except Exception as e:
        logger.debug("fontTools subsetting error: %s", e)
        try:
            tt_font.close()
        except Exception:
            pass
        return None


def _populate_from_encoding(
    subsetter: object,
    tt_font: object,
    used_codes: set[int],
    code_to_glyphname: dict[int, str],
) -> None:
    """Populates the fontTools subsetter using PDF encoding mappings.

    Maps used character codes to glyph names via the PDF encoding,
    then finds those glyphs in the font program. Uses direct glyph
    name lookup first (works even without cmap), then falls back to
    AGL-based Unicode resolution for glyph names not found in the font.

    Args:
        subsetter: fontTools Subsetter instance.
        tt_font: fontTools TTFont instance.
        used_codes: Character codes used in content streams.
        code_to_glyphname: Mapping from character codes to Adobe
            glyph names, derived from the font's PDF /Encoding.
    """
    glyph_order_set = set(tt_font.getGlyphOrder())

    target_glyphs: set[str] = set()
    target_unicodes: set[int] = set()

    for code in used_codes:
        glyph_name = code_to_glyphname.get(code)
        if glyph_name is not None:
            # Try direct glyph name lookup (works without cmap)
            if glyph_name in glyph_order_set:
                target_glyphs.add(glyph_name)
                continue
            # Glyph name not in font; resolve via AGL to Unicode
            uval = resolve_glyph_to_unicode(glyph_name)
            if uval is not None:
                target_unicodes.add(uval)
                continue
        # Code not in encoding or unresolvable; use as-is
        target_unicodes.add(code)

    if target_glyphs:
        subsetter.populate(glyphs=target_glyphs)
    if target_unicodes:
        subsetter.populate(unicodes=target_unicodes)
