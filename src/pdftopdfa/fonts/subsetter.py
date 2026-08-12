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
import hashlib
import logging
import string
from dataclasses import dataclass, field
from io import BytesIO

import pikepdf
from fontTools.agl import AGL2UV, LEGACY_AGL2UV
from pikepdf import Name, Stream

from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve_indirect
from .analysis import (
    _is_subset_font,
    get_base_font_name,
    get_font_name,
    get_font_type,
    is_symbolic_font,
)
from .encodings import STANDARD_ENCODING
from .glyph_mapping import (
    SYMBOL_GLYPH_TO_UNICODE,
    ZAPFDINGBATS_GLYPH_TO_UNICODE,
    resolve_glyph_name,
)
from .glyph_usage import CharacterCode, collect_font_usage
from .tounicode import (
    generate_tounicode_for_macroman,
    generate_tounicode_for_winansi,
    map_type0_character_codes_to_cids,
    parse_cidtogidmap_stream,
    parse_tounicode_cmap_sequences,
    resolve_glyph_to_unicode,
)
from .traversal import iter_all_page_fonts
from .utils import (
    check_fstype_restrictions,
    get_fstype,
    get_truetype_byte_encoding,
    is_permitted_fstype_notice,
    resolve_symbol_cmap_glyph,
)
from .utils import safe_str as _safe_str

logger = logging.getLogger(__name__)

_UNICODE_GLYPH_NAME_FALLBACKS: dict[int, str] = {
    0x00A0: "nbspace",
    0x00AD: "sfthyphen",
    0x00A6: "brokenbar",
    0x00AC: "logicalnot",
    0x00AE: "registered",
    0x00AF: "macron",
    0x00B0: "degree",
    0x00B1: "plusminus",
    0x00B2: "twosuperior",
    0x00B3: "threesuperior",
    0x00B4: "acute",
    0x00B5: "mu",
    0x00B6: "paragraph",
    0x00B7: "periodcentered",
    0x00B8: "cedilla",
    0x00B9: "onesuperior",
    0x00BA: "ordmasculine",
    0x00BC: "onequarter",
    0x00BD: "onehalf",
    0x00BE: "threequarters",
}

_WINANSI_GLYPH_NAME_OVERRIDES: dict[int, str] = {
    0xA0: "space",
    0xAD: "hyphen",
}


def _generate_subset_prefix(font_data: bytes) -> str:
    """Generates a 6-letter uppercase subset prefix from the font data.

    The prefix is derived from a hash of the subsetted font program so
    that repeated conversions of the same input produce byte-identical
    output (reproducible builds with ``deterministic_id=True``).

    Args:
        font_data: The subsetted font program bytes.

    Returns:
        String like "ABCDEF+" for use as a font subset tag.
    """
    digest = hashlib.sha256(font_data).digest()
    letters = "".join(string.ascii_uppercase[b % 26] for b in digest[:6])
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


def _resolve_cidfont_used_gids(
    desc_font: pikepdf.Object,
    used_codes: set[int],
) -> set[int]:
    """Resolves used CID character codes to glyph IDs for subsetting.

    CIDFonts with ``/CIDToGIDMap /Identity`` can subset directly from the
    character codes seen in content streams. When the descendant font carries
    an explicit CIDToGIDMap stream, those CIDs must be translated to the
    mapped GIDs first; otherwise the subsetter preserves the wrong glyph slots
    and rendered text can become corrupted.

    Args:
        desc_font: The descendant CIDFont dictionary.
        used_codes: CIDs extracted from the content streams.

    Returns:
        The set of GIDs that must be preserved in the embedded font program.
    """
    cidtogidmap = desc_font.get("/CIDToGIDMap")
    if cidtogidmap is None or isinstance(cidtogidmap, pikepdf.Name):
        return set(used_codes)

    try:
        cidtogidmap = _resolve_indirect(cidtogidmap)
        cid_to_gid = parse_cidtogidmap_stream(bytes(cidtogidmap.read_bytes()))
    except Exception as exc:
        log_suppressed_error(
            logger, exc, "Error parsing CIDToGIDMap during subsetting: %s", exc
        )
        return set(used_codes)

    return {cid_to_gid[cid] for cid in used_codes if cid in cid_to_gid}


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
        font_usage: dict[tuple[int, int], set[CharacterCode]],
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

        # Skip fonts without any recorded glyph usage. Usage collection
        # has blind spots (e.g. unparsable content streams, text drawn
        # with a font inherited from the parent context without a local
        # Tf); subsetting such a font would strip it down to .notdef
        # and lose glyphs in the rendered output.
        if obj_key not in font_usage:
            result.fonts_skipped.append(f"{font_name} (no glyph usage found)")
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
        font_usage: dict[tuple[int, int], set[CharacterCode]],
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

        # Translate content-stream character codes through the Type 0 CMap,
        # then through an explicit descendant CIDToGIDMap when present.
        used_codes = font_usage.get(obj_key, set())
        used_cids = map_type0_character_codes_to_cids(font_obj, used_codes)
        if used_cids is None:
            result.fonts_skipped.append(f"{font_name} (unresolved Type0 CMap)")
            return
        used_cids = {cid if 0 <= cid <= 65_535 else 0 for cid in used_cids}
        used_gids = _resolve_cidfont_used_gids(desc_font, used_cids)

        # Perform subsetting
        try:
            original_data = bytes(font_file.read_bytes())
            original_size = len(original_data)

            # Check fsType embedding restrictions
            if not _check_subsetting_allowed(original_data, font_name, result):
                return

            if 0 in used_gids:
                used_gids.update(
                    _semantic_gids_for_notdef_codes(
                        font_obj,
                        used_codes,
                        original_data,
                    )
                )

            subsetted_data = _subset_font_data(original_data, used_gids, is_cid=True)

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
            prefix = _generate_subset_prefix(subsetted_data)
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

        except Exception as e:
            warning = f"Error subsetting CIDFont '{font_name}': {e}"
            result.warnings.append(warning)
            logger.debug(warning)

    def _subset_simple_font(
        self,
        font_obj: pikepdf.Object,
        obj_key: tuple[int, int],
        font_usage: dict[tuple[int, int], set[CharacterCode]],
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

        # Perform subsetting
        try:
            original_data = bytes(font_file.read_bytes())
            original_size = len(original_data)

            is_symbolic = is_symbolic_font(font_obj)
            program_glyph_names = False
            if get_font_type(font_obj) == "TrueType":
                encoding_map, program_glyph_names = (
                    _resolve_truetype_font_encoding_with_source(
                        font_obj,
                        original_data,
                        pdfa_normalized=True,
                    )
                )
                if is_symbolic and encoding_map is None:
                    result.fonts_skipped.append(
                        f"{font_name} (symbolic encoding unresolved)"
                    )
                    return
            else:
                # Resolve the PDF encoding for precise glyph selection.
                encoding_map = _resolve_simple_font_encoding(font_obj)

            # Check fsType embedding restrictions
            if not _check_subsetting_allowed(original_data, font_name, result):
                return

            subsetted_data = _subset_font_data(
                original_data,
                used_codes,
                is_cid=False,
                code_to_glyphname=encoding_map,
                glyph_names_are_program=program_glyph_names,
            )

            # For symbolic TrueType fonts, the cmap subtable gets
            # dropped by fontTools (it only preserves cmaps when
            # subsetting by unicode).  Rebuild the (3,0) cmap from
            # the original font's cmap data.
            if (is_symbolic or program_glyph_names) and subsetted_data is not None:
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
            prefix = _generate_subset_prefix(subsetted_data)
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
        if is_permitted_fstype_notice(warning):
            logger.info(msg)
        else:
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

    # fontTools' reverse AGL table omits some WinAnsi glyph names
    # (for example twosuperior / onesuperior / nbsp variants). Fill only the
    # missing Unicode values with explicit Adobe glyph-name fallbacks.
    for uv, glyph_name in _UNICODE_GLYPH_NAME_FALLBACKS.items():
        if uv not in uv2agl:
            uv2agl[uv] = glyph_name
    return uv2agl


def _resolve_simple_font_encoding(
    font_obj: pikepdf.Object,
    *,
    pdfa_normalized: bool = False,
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
        pdfa_normalized: Resolve invalid or missing non-symbolic TrueType
            encodings as the WinAnsi encoding installed by the sanitizer.

    Returns:
        Dictionary mapping character codes to glyph names, or None if no
        encoding can be resolved.
    """
    encoding = font_obj.get("/Encoding")
    if encoding is None:
        return _build_winansi_glyphnames() if pdfa_normalized else None

    if isinstance(encoding, pikepdf.Name):
        enc_name = _safe_str(encoding)
        if enc_name == "/WinAnsiEncoding":
            return _build_winansi_glyphnames()
        elif enc_name == "/MacRomanEncoding":
            return _build_glyphnames_from_unicode(generate_tounicode_for_macroman())
        elif enc_name == "/StandardEncoding":
            if not pdfa_normalized:
                return dict(STANDARD_ENCODING)
        return _build_winansi_glyphnames() if pdfa_normalized else None

    # Encoding dictionary with BaseEncoding + Differences
    try:
        enc_dict = _resolve_indirect(encoding)
        base = enc_dict.get("/BaseEncoding")
        if base is not None:
            base_name = _safe_str(base)
            if base_name == "/WinAnsiEncoding":
                code_to_glyphname = _build_winansi_glyphnames()
            elif base_name == "/MacRomanEncoding":
                code_to_glyphname = _build_glyphnames_from_unicode(
                    generate_tounicode_for_macroman()
                )
            elif base_name == "/StandardEncoding":
                code_to_glyphname = (
                    _build_winansi_glyphnames()
                    if pdfa_normalized
                    else dict(STANDARD_ENCODING)
                )
            else:
                code_to_glyphname = (
                    _build_winansi_glyphnames()
                    if pdfa_normalized
                    else dict(STANDARD_ENCODING)
                )
        else:
            code_to_glyphname = (
                _build_winansi_glyphnames()
                if pdfa_normalized
                else dict(STANDARD_ENCODING)
            )

        # Apply /Differences only when the sanitizer will retain the complete
        # array. A single non-AGL name makes rule 6.2.11.6-2 remove it.
        differences = enc_dict.get("/Differences")
        if differences is not None:
            overrides: dict[int, str] = {}
            all_agl = True
            current_code = 0
            for item in differences:
                try:
                    current_code = int(item)
                    continue
                except (TypeError, ValueError):
                    pass
                if isinstance(item, pikepdf.Name):
                    glyph_name = _safe_str(item)[1:]  # Remove leading "/"
                    if (
                        glyph_name != ".notdef"
                        and glyph_name not in AGL2UV
                        and glyph_name not in LEGACY_AGL2UV
                    ):
                        all_agl = False
                    overrides[current_code] = glyph_name
                    current_code += 1
            if not pdfa_normalized or all_agl:
                code_to_glyphname.update(overrides)

        return code_to_glyphname or None
    except Exception:
        return None


def _resolve_truetype_font_encoding(
    font_obj: pikepdf.Object,
    font_data: bytes,
    *,
    pdfa_normalized: bool = False,
) -> dict[int, str] | None:
    """Resolve the effective byte-to-glyph mapping for a simple TrueType font."""
    return _resolve_truetype_font_encoding_with_source(
        font_obj,
        font_data,
        pdfa_normalized=pdfa_normalized,
    )[0]


def _resolve_truetype_font_encoding_with_source(
    font_obj: pikepdf.Object,
    font_data: bytes,
    *,
    pdfa_normalized: bool = False,
) -> tuple[dict[int, str] | None, bool]:
    """Resolve a TrueType encoding and whether names came from its byte cmap."""
    symbolic = is_symbolic_font(font_obj)
    if symbolic or font_obj.get("/Encoding") is None:
        try:
            from fontTools.ttLib import TTFont

            tt_font = TTFont(BytesIO(font_data))
            try:
                byte_encoding = get_truetype_byte_encoding(tt_font)
            finally:
                tt_font.close()
        except Exception:
            byte_encoding = None

        if byte_encoding is not None:
            return byte_encoding[2], True
        if symbolic:
            return _build_symbolic_truetype_encoding(font_obj, font_data), False

    return (
        _resolve_simple_font_encoding(
            font_obj,
            pdfa_normalized=pdfa_normalized,
        ),
        False,
    )


def _build_winansi_glyphnames() -> dict[int, str]:
    """Builds WinAnsiEncoding's PDF glyph-name mapping."""
    code_to_glyphname = _build_glyphnames_from_unicode(generate_tounicode_for_winansi())
    code_to_glyphname.update(_WINANSI_GLYPH_NAME_OVERRIDES)
    return code_to_glyphname


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


def _resolve_explicit_encoding_differences(
    font_obj: pikepdf.Object,
) -> dict[int, str]:
    """Return only explicitly declared PDF encoding differences."""
    encoding = font_obj.get("/Encoding")
    if encoding is None or isinstance(encoding, pikepdf.Name):
        return {}

    try:
        encoding = _resolve_indirect(encoding)
        differences = encoding.get("/Differences")
        if differences is None:
            return {}
    except Exception:
        return {}

    result: dict[int, str] = {}
    current_code = 0
    for item in differences:
        try:
            current_code = int(item)
            continue
        except (TypeError, ValueError):
            pass
        if isinstance(item, pikepdf.Name):
            if 0 <= current_code <= 0xFF:
                result[current_code] = _safe_str(item)[1:]
            current_code += 1
    return result


def _get_unicode_cmap(tt_font) -> dict[int, str]:
    """Return a Unicode-capable cmap without treating legacy bytes as Unicode."""
    cmap_table = tt_font.get("cmap")
    if cmap_table is None:
        return {}

    preferences = (
        (3, 10),
        (0, 6),
        (0, 4),
        (3, 1),
        (0, 3),
        (0, 2),
        (0, 1),
        (0, 0),
    )
    result: dict[int, str] = {}
    visited: set[int] = set()
    for platform_id, encoding_id in preferences:
        for table in cmap_table.tables:
            mapping = getattr(table, "cmap", None)
            if (
                table.platformID == platform_id
                and table.platEncID == encoding_id
                and mapping
            ):
                visited.add(id(table))
                for codepoint, glyph_name in mapping.items():
                    result.setdefault(codepoint, glyph_name)
    for table in cmap_table.tables:
        mapping = getattr(table, "cmap", None)
        if id(table) not in visited and table.platformID == 0 and mapping:
            for codepoint, glyph_name in mapping.items():
                result.setdefault(codepoint, glyph_name)
    return result


def _resolve_encoded_truetype_glyph(
    tt_font,
    code: int,
    glyph_name: str,
) -> str | None:
    """Resolve one PDF byte and glyph name to an embedded TrueType glyph."""
    glyph_names = set(tt_font.getGlyphOrder())
    try:
        metrics = tt_font["hmtx"].metrics
    except Exception:
        metrics = {name: (0, 0) for name in glyph_names}

    unicode_cmap = _get_unicode_cmap(tt_font)
    for custom_mapping in (
        ZAPFDINGBATS_GLYPH_TO_UNICODE,
        SYMBOL_GLYPH_TO_UNICODE,
    ):
        resolved = resolve_glyph_name(
            glyph_name,
            unicode_cmap,
            metrics,
            custom_mapping,
        )
        if resolved is not None:
            return resolved

    cmap_table = tt_font.get("cmap")
    if cmap_table is None:
        return None

    # PDF symbolic TrueType first uses a Microsoft Symbol cmap, then the
    # legacy Mac cmap. These are byte mappings, not Unicode mappings.
    for table in cmap_table.tables:
        if table.platformID == 3 and table.platEncID == 0:
            mapping = getattr(table, "cmap", None) or {}
            resolved = resolve_symbol_cmap_glyph(mapping, code)
            if resolved in glyph_names:
                return resolved
    for table in cmap_table.tables:
        if table.platformID == 1 and table.platEncID == 0:
            mapping = getattr(table, "cmap", None) or {}
            resolved = mapping.get(code)
            if resolved in glyph_names:
                return resolved
    return None


def _build_symbolic_truetype_encoding(
    font_obj: pikepdf.Object,
    font_data: bytes,
) -> dict[int, str] | None:
    """Builds code-to-glyph-name mapping for symbolic TrueType fonts.

    Character codes are mapped through the font program before considering
    malformed PDF Encoding data:
    - (1,0) Mac Roman cmap: codes map directly
    - (3,0) Microsoft Symbol cmap: codes use the 00/F0/F1/F2 ranges

    A usable program byte cmap is authoritative and PDF /Encoding is ignored.
    Explicit Differences are used only as a narrow recovery path when neither
    byte cmap exists.

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
        byte_encoding = get_truetype_byte_encoding(tt)
        if byte_encoding is not None:
            return byte_encoding[2]

        result: dict[int, str] = {}
        for code, glyph_name in _resolve_explicit_encoding_differences(
            font_obj
        ).items():
            resolved = _resolve_encoded_truetype_glyph(
                tt,
                code,
                glyph_name,
            )
            if resolved is not None:
                result[code] = resolved

        return result or None
    finally:
        tt.close()


def _rebuild_symbolic_cmap(
    original_data: bytes,
    subsetted_data: bytes,
) -> bytes:
    """Rebuilds cmap subtables for a symbolic TrueType font after subsetting.

    fontTools strips legacy cmap subtables when subsetting by glyph names
    (no unicodes specified). This function restores the authoritative (3,0)
    Microsoft Symbol cmap, or a (1,0) Mac cmap when no (3,0) exists, filtered
    to glyphs that survived subsetting.

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
            # Prefer the original Microsoft Symbol cmap, then the Mac byte cmap.
            source_cmap = None
            source_platform = None
            orig_cmap = original_tt.get("cmap")
            if orig_cmap is not None:
                for platform_id, encoding_id in ((3, 0), (1, 0)):
                    source = next(
                        (
                            table
                            for table in orig_cmap.tables
                            if table.platformID == platform_id
                            and table.platEncID == encoding_id
                            and table.cmap
                        ),
                        None,
                    )
                    if source is not None:
                        source_cmap = source.cmap
                        source_platform = (platform_id, encoding_id)
                        break

            if source_cmap is None or source_platform is None:
                return subsetted_data

            # Build unicode→GID mapping using original font
            orig_glyph_order = original_tt.getGlyphOrder()
            orig_name_to_gid = {n: i for i, n in enumerate(orig_glyph_order)}

            # Map GIDs to new glyph names in subsetted font
            sub_glyph_order = subsetted_tt.getGlyphOrder()
            num_sub_glyphs = len(sub_glyph_order)

            new_mapping: dict[int, str] = {}
            for character_code, orig_name in source_cmap.items():
                gid = orig_name_to_gid.get(orig_name)
                if gid is not None and gid < num_sub_glyphs:
                    new_mapping[character_code] = sub_glyph_order[gid]

            if not new_mapping:
                return subsetted_data

            subtable = cmap_format_4(4)
            subtable.platformID, subtable.platEncID = source_platform
            subtable.format = 4
            subtable.reserved = 0
            subtable.length = 0
            subtable.language = 0
            subtable.cmap = new_mapping

            # Preserve Unicode cmaps retained by fontTools. They are needed
            # later to resolve explicit PDF glyph-name Differences. Replace
            # only Microsoft Symbol subtables with the rebuilt mapping.
            cmap_table = subsetted_tt.get("cmap")
            if cmap_table is None:
                cmap_table = table__c_m_a_p()
                cmap_table.tableVersion = 0
                cmap_table.tables = []
                subsetted_tt["cmap"] = cmap_table
            cmap_table.tables = [
                table
                for table in cmap_table.tables
                if (table.platformID, table.platEncID) != source_platform
            ]
            cmap_table.tables.append(subtable)

            # Save
            output = BytesIO()
            subsetted_tt.save(output)
            return output.getvalue()
        finally:
            original_tt.close()
            subsetted_tt.close()
    except Exception as e:
        log_suppressed_error(logger, e, "Error rebuilding symbolic cmap: %s", e)
        return subsetted_data


def _semantic_gids_for_notdef_codes(
    font_obj: pikepdf.Object,
    used_codes: set[CharacterCode],
    font_data: bytes,
) -> set[int]:
    """Find real glyphs described by ToUnicode for codes mapped to GID 0."""
    tounicode = _resolve_indirect(font_obj.get("/ToUnicode"))
    if not isinstance(tounicode, Stream):
        return set()
    try:
        unicode_map = parse_tounicode_cmap_sequences(tounicode.read_bytes())
    except Exception:
        return set()

    normalized_codes = {
        code if isinstance(code, bytes) else code.to_bytes(2, "big")
        for code in used_codes
        if isinstance(code, bytes) or 0 <= code <= 65_535
    }
    codepoints = {
        sequence[0] for code in normalized_codes if (sequence := unicode_map.get(code))
    }
    if not codepoints:
        return set()

    from fontTools.ttLib import TTFont

    tt_font = TTFont(BytesIO(font_data))
    try:
        cmap = _get_unicode_cmap(tt_font)
        glyph_order = tt_font.getGlyphOrder()
        glyph_ids = {name: gid for gid, name in enumerate(glyph_order)}
        return {
            glyph_ids[name]
            for codepoint in codepoints
            if (name := cmap.get(codepoint)) in glyph_ids and glyph_ids[name] != 0
        }
    finally:
        tt_font.close()


def _subset_font_data(
    font_data: bytes,
    used_codes: set[int],
    *,
    is_cid: bool,
    code_to_glyphname: dict[int, str] | None = None,
    glyph_names_are_program: bool = False,
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
        glyph_names_are_program: The mapping values are authoritative glyph
            names from a TrueType byte cmap and need no AGL resolution.

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
        if cmap_table is None or not any(
            getattr(table, "cmap", None) for table in cmap_table.tables
        ):
            logger.debug("Font has no cmap table, skipping subsetting")
            tt_font.close()
            return None

        options = Options()
        options.retain_gids = True
        # Preserve glyph names for simple fonts so PDF encodings can still
        # resolve character codes via standard glyph names after subsetting.
        options.glyph_names = True
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
                    subsetter,
                    tt_font,
                    used_codes,
                    code_to_glyphname,
                    glyph_names_are_program=glyph_names_are_program,
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
        log_suppressed_error(logger, e, "fontTools subsetting error: %s", e)
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
    *,
    glyph_names_are_program: bool = False,
) -> None:
    """Populates the fontTools subsetter using PDF encoding mappings.

    Maps used character codes to glyphs through explicit PDF names,
    Unicode cmaps, and symbolic byte cmaps.

    Args:
        subsetter: fontTools Subsetter instance.
        tt_font: fontTools TTFont instance.
        used_codes: Character codes used in content streams.
        code_to_glyphname: Mapping from character codes to Adobe
            glyph names, derived from the font's PDF /Encoding.
        glyph_names_are_program: Values are already resolved program glyphs.
    """
    unicode_cmap = _get_unicode_cmap(tt_font)
    unicodes_by_glyph: dict[str, set[int]] = {}
    for unicode_value, mapped_glyph in unicode_cmap.items():
        unicodes_by_glyph.setdefault(mapped_glyph, set()).add(unicode_value)

    target_glyphs: set[str] = set()
    target_unicodes: set[int] = set()

    for code in used_codes:
        glyph_name = code_to_glyphname.get(code)
        if glyph_name is not None:
            if glyph_names_are_program and glyph_name in tt_font.getGlyphOrder():
                resolved = glyph_name
            else:
                resolved = _resolve_encoded_truetype_glyph(
                    tt_font,
                    code,
                    glyph_name,
                )
            if resolved is not None:
                target_glyphs.add(resolved)
                # Retaining the corresponding Unicode entries preserves the
                # cmap needed to migrate explicit symbolic Differences later.
                target_unicodes.update(unicodes_by_glyph.get(resolved, ()))
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
