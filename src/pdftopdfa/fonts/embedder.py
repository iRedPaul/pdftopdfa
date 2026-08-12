# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font embedding for PDF/A compliance."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

import pikepdf
from fontTools.agl import AGL2UV, LEGACY_AGL2UV, UV2AGL
from pikepdf import Array, Dictionary, Name, Stream

from ..exceptions import FontEmbeddingError
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve_indirect
from .analysis import (
    _is_subset_font,
    get_base_font_name,
    get_font_name,
    get_font_type,
    has_tounicode_cmap,
    is_font_embedded,
    is_symbolic_font,
)
from .cid_unicode import get_cid_to_unicode
from .cidfont import CIDFontBuilder
from .constants import FONT_REPLACEMENTS, SYMBOL_FONTS, resolve_standard14_alias
from .constants import UTF16_ENCODING_NAMES as _UTF16_ENCODING_NAMES
from .encodings import SYMBOL_ENCODING, ZAPFDINGBATS_ENCODING
from .glyph_mapping import SYMBOL_GLYPH_TO_UNICODE, ZAPFDINGBATS_GLYPH_TO_UNICODE
from .glyph_usage import CharacterCode, collect_font_usage
from .loader import FontLoader
from .metrics import FontMetricsExtractor
from .subsetter import (
    FontSubsetter,
    SubsettingResult,
    _get_unicode_cmap,
    _resolve_truetype_font_encoding,
)
from .tounicode import (
    build_identity_unicode_mapping,
    fill_tounicode_gaps_with_pua,
    filter_invalid_unicode_values,
    generate_cidfont_tounicode_cmap,
    generate_to_unicode_for_simple_font,
    generate_tounicode_cmap_data,
    generate_tounicode_for_macroman,
    generate_tounicode_for_standard_encoding,
    generate_tounicode_for_type3_font,
    generate_tounicode_for_winansi,
    generate_tounicode_from_cff_program,
    generate_tounicode_from_encoding_dict,
    generate_tounicode_from_type1_program,
    get_font_code_space_ranges,
    get_type0_cid_encoding_map,
    parse_cidtogidmap_stream,
    parse_tounicode_cmap,
    parse_tounicode_cmap_sequences,
    resolve_glyph_to_unicode,
    resolve_symbol_glyph_to_unicode,
    validate_tounicode_cmap,
)
from .traversal import iter_acroform_dr_fonts, iter_all_page_fonts
from .utils import get_encoding_name as _get_encoding_name
from .utils import safe_str as _safe_str

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont

logger = logging.getLogger(__name__)

_LATIN_CIDFONT_SYSTEM_ALIASES = {
    "Arial": "ArialMT",
}

_CJK_CIDFONT_NAME_HINTS = (
    "AdobeMing",
    "AdobeMyungjo",
    "AdobeSong",
    "Batang",
    "CJK",
    "FangSong",
    "Gothic",
    "Gulim",
    "Hei",
    "Heisei",
    "Kai",
    "Koz",
    "Malgun",
    "Meiryo",
    "Ming",
    "Mincho",
    "MSGothic",
    "MSMincho",
    "NotoSansCJK",
    "SimHei",
    "SimKai",
    "SimSun",
    "Song",
    "YuGothic",
    "YuMincho",
)


def _is_agl_glyph_name(glyph_name: str) -> bool:
    """Return True when a glyph name is explicitly listed in Adobe Glyph List."""
    return (
        glyph_name == ".notdef" or glyph_name in AGL2UV or glyph_name in LEGACY_AGL2UV
    )


def _is_non_pua_unicode_scalar(value: int) -> bool:
    """Return whether a value is a usable non-private Unicode scalar."""
    return (
        0 <= value <= 0x10FFFF
        and not 0xD800 <= value <= 0xDFFF
        and not 0xE000 <= value <= 0xF8FF
        and not 0xF0000 <= value <= 0xFFFFD
        and not 0x100000 <= value <= 0x10FFFD
    )


def _cidfont_replacement_warning(base_name: str) -> str:
    """Build the warning issued when a non-embedded CIDFont is replaced.

    A substitute font can preserve the source character mapping but cannot
    reproduce a missing proprietary font program's exact glyph design.
    """
    return (
        f"Non-embedded CIDFont '{base_name}' was replaced with a substitute "
        "font; its rendered glyph design and metrics may differ from the "
        "unavailable original font"
    )


def _is_utf16_encoding(encoding_name: str) -> bool:
    """Checks if the encoding name indicates a UTF-16/UCS-2 CMap.

    For these encodings, character codes are already Unicode values,
    so ToUnicode should map each code to itself.

    Args:
        encoding_name: The encoding CMap name (e.g. "UniJIS-UTF16-H").

    Returns:
        True if the encoding is a UTF-16/UCS-2 CMap.
    """
    return encoding_name in _UTF16_ENCODING_NAMES


def _find_unicode_cmap_name(encoding: pikepdf.Object) -> str:
    """Find a predefined Unicode CMap in an encoding's /UseCMap chain."""
    current = _resolve_indirect(encoding)
    seen: set[tuple[int, int] | tuple[str, int]] = set()

    while True:
        if isinstance(current, pikepdf.Name):
            name = _get_encoding_name(current)
            return name if _is_utf16_encoding(name) else ""
        if not isinstance(current, pikepdf.Stream):
            return ""

        objgen = current.objgen
        key: tuple[int, int] | tuple[str, int] = (
            objgen if objgen != (0, 0) else ("direct", id(current))
        )
        if key in seen:
            return ""
        seen.add(key)

        name = _get_encoding_name(current)
        if _is_utf16_encoding(name):
            return name

        usecmap = current.get("/UseCMap")
        if usecmap is not None:
            current = _resolve_indirect(usecmap)
            continue

        try:
            data = bytes(current.read_bytes())
        except Exception:
            return ""
        data = re.sub(rb"%[^\r\n]*", b"", data)
        match = re.search(rb"/([^\s/]+)\s+usecmap\b", data)
        if match is None:
            return ""
        name = match.group(1).decode("latin-1")
        return name if _is_utf16_encoding(name) else ""


def _generate_unicode_type0_tounicode(
    code_to_unicode: dict[bytes, int],
    code_space_ranges: tuple[tuple[bytes, bytes], ...],
) -> bytes:
    """Generate a ToUnicode CMap without collapsing variable-width codes."""
    filtered = filter_invalid_unicode_values(code_to_unicode)
    mappings = {
        code: unicode_value
        for code, unicode_value in filtered.items()
        if isinstance(code, bytes)
        and isinstance(unicode_value, int)
        and code
        and 0 <= unicode_value <= 0x10FFFF
    }
    ranges = [
        (lower, upper)
        for lower, upper in code_space_ranges
        if lower and len(lower) == len(upper) and lower <= upper
    ]
    for code in mappings:
        if not any(
            len(lower) == len(code) and lower <= code <= upper
            for lower, upper in ranges
        ):
            ranges.append((code, code))
    ranges = list(dict.fromkeys(ranges))

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
    ]
    for offset in range(0, len(ranges), 100):
        chunk = ranges[offset : offset + 100]
        lines.append(f"{len(chunk)} begincodespacerange")
        lines.extend(
            f"<{lower.hex().upper()}> <{upper.hex().upper()}>" for lower, upper in chunk
        )
        lines.append("endcodespacerange")

    codes = sorted(mappings, key=lambda code: (len(code), code))
    for offset in range(0, len(codes), 100):
        chunk = codes[offset : offset + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for code in chunk:
            destination = chr(mappings[code]).encode("utf-16-be")
            lines.append(f"<{code.hex().upper()}> <{destination.hex().upper()}>")
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


@dataclass
class EmbeddingResult:
    """Result of font embedding.

    Attributes:
        fonts_embedded: List of successfully embedded fonts.
        fonts_failed: List of fonts that could not be embedded.
        fonts_preserved: List of already embedded fonts (were not modified).
        warnings: List of warnings during embedding.
    """

    fonts_embedded: list[str] = field(default_factory=list)
    fonts_failed: list[str] = field(default_factory=list)
    fonts_preserved: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FontProgramDeduplicationResult:
    """Summary of embedded font-program deduplication."""

    programs_deduplicated: int = 0
    bytes_saved_estimate: int = 0


class FontEmbedder:
    """Embeds missing fonts in a PDF.

    Replaces missing fonts with policy-approved system fonts or bundled replacements.
    """

    def __init__(self, pdf: pikepdf.Pdf) -> None:
        """Initializes the FontEmbedder.

        Args:
            pdf: Opened pikepdf PDF object.
        """
        self.pdf = pdf
        self._font_cache: dict[str, tuple[bytes, TTFont]] = {}
        self._metrics = FontMetricsExtractor()
        self._loader = FontLoader(self._font_cache)
        self._cidfont_builder = CIDFontBuilder(pdf, self._metrics)

    def close(self) -> None:
        """Close all cached TTFont objects to release file handles."""
        for _data, tt_font in self._font_cache.values():
            try:
                tt_font.close()
            except Exception:
                pass
        self._font_cache.clear()

    def __enter__(self) -> "FontEmbedder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def embed_missing_fonts(self) -> EmbeddingResult:
        """Embeds all missing fonts.

        Scans page-level Resources, Form XObjects, Tiling Patterns,
        Annotation Appearance Streams (recursively), and AcroForm /DR
        Default Resources.

        Returns:
            EmbeddingResult with embedding status.
        """
        result = EmbeddingResult()
        processed_fonts: set[str] = set()
        preserved_fonts: set[str] = set()
        processed_font_ids: set[tuple[int, int]] = set()
        font_usage: dict[tuple[int, int], set[CharacterCode]] | None = None

        for font_key, font_obj in self._iter_unique_fonts(processed_font_ids):
            try:
                font_name = get_font_name(font_obj)
                base_name = get_base_font_name(font_name)

                # Track already embedded fonts (without duplicates)
                if is_font_embedded(font_obj):
                    if (
                        base_name not in preserved_fonts
                        and base_name not in processed_fonts
                    ):
                        preserved_fonts.add(base_name)
                        logger.debug(
                            "Font already embedded, preserving: %s",
                            base_name,
                        )
                    continue

                font_type = get_font_type(font_obj)
                use_fallback = False
                if font_type == "CIDFont":
                    if font_usage is None:
                        font_usage = collect_font_usage(self.pdf)
                    # Preserve encoding (Identity-H or Identity-V)
                    encoding = self._get_cidfont_encoding(font_obj)
                    success = self._embed_cidfont(
                        font_obj,
                        base_name,
                        encoding=encoding,
                        used_codes=font_usage.get(font_obj.objgen, set()),
                    )
                else:
                    # Replace font (use fallback for unknown fonts)
                    use_fallback = (
                        resolve_standard14_alias(base_name) not in FONT_REPLACEMENTS
                    )
                    success, used_fallback = self._replace_simple_font(
                        font_obj,
                        base_name,
                        use_fallback=use_fallback,
                    )

                if base_name in processed_fonts:
                    continue
                processed_fonts.add(base_name)

                if not success:
                    result.fonts_failed.append(base_name)
                    continue

                result.fonts_embedded.append(base_name)
                if font_type == "CIDFont":
                    warning = _cidfont_replacement_warning(base_name)
                    result.warnings.append(warning)
                    logger.warning(warning)
                elif used_fallback:
                    warning = (
                        f"No specific replacement for font '{base_name}',"
                        f" using LiberationSans as fallback"
                    )
                    result.warnings.append(warning)
                    logger.warning(warning)
                else:
                    logger.info("Font embedded: %s", base_name)

            except UnicodeDecodeError:
                logger.debug(
                    "Skipping font %s: non-UTF-8 bytes in font data",
                    font_key,
                )
                continue
            except Exception as e:
                log_suppressed_error(
                    logger,
                    e,
                    "Error with font %s: %s",
                    font_key,
                    e,
                    level=logging.INFO,
                )
                continue

        # Add preserved fonts to result
        result.fonts_preserved = sorted(preserved_fonts)

        return result

    def subset_embedded_fonts(self) -> SubsettingResult:
        """Subsets all eligible embedded fonts to reduce file size.

        Subsets TrueType (FontFile2) and CFF/OpenType (FontFile3 with
        /Subtype /OpenType) fonts. Skips Type3 fonts, non-embedded
        fonts, already-subsetted fonts, and bare CFF programs.

        Returns:
            SubsettingResult with subsetting status.
        """
        subsetter = FontSubsetter(self.pdf)
        return subsetter.subset_all_fonts()

    def deduplicate_embedded_font_programs(self) -> FontProgramDeduplicationResult:
        """Reuse identical embedded font program streams across font descriptors."""
        result = FontProgramDeduplicationResult()
        seen_streams: dict[
            tuple[str, int, str],
            pikepdf.Object,
        ] = {}
        processed_font_ids: set[tuple[int, int]] = set()

        for font_obj in self._iter_unique_embedded_fonts(processed_font_ids):
            try:
                descriptor, _desc_font = self._get_font_descriptor_for_font(font_obj)
                if descriptor is None:
                    continue
                font_file_info = self._find_any_font_file(descriptor)
                if font_file_info is None:
                    continue

                font_stream = font_file_info[1]
                font_bytes = bytes(font_stream.read_bytes())
                signature = (
                    hashlib.sha256(font_bytes).hexdigest(),
                    len(font_bytes),
                    str(font_stream.get("/Subtype", "")),
                )
                existing_stream = seen_streams.get(signature)
                if existing_stream is None:
                    seen_streams[signature] = font_stream
                    continue

                descriptor[font_file_info[0]] = existing_stream
                result.programs_deduplicated += 1
                result.bytes_saved_estimate += len(font_bytes)
            except Exception:
                continue

        return result

    def collect_subsetted_standard14_font_ids(self) -> set[tuple[int, int]]:
        """Collect embedded subsetted Standard-14 font object IDs.

        This is intended to run before our own subsetting step so later refreshes
        only target problematic pre-existing subsets instead of undoing the size
        savings from fonts we subset ourselves.
        """
        collected_ids: set[tuple[int, int]] = set()
        processed_font_ids: set[tuple[int, int]] = set()

        for page in self.pdf.pages:
            for _font_key, font_obj in iter_all_page_fonts(page):
                try:
                    obj_key = font_obj.objgen
                    if obj_key == (0, 0) or obj_key in processed_font_ids:
                        continue
                    processed_font_ids.add(obj_key)

                    if not is_font_embedded(font_obj):
                        continue

                    font_name = get_font_name(font_obj)
                    if not _is_subset_font(font_name):
                        continue

                    base_name = get_base_font_name(font_name)
                    if resolve_standard14_alias(base_name) not in FONT_REPLACEMENTS:
                        continue

                    if is_symbolic_font(font_obj):
                        continue

                    font_type = get_font_type(font_obj)
                    if font_type not in {"TrueType", "Type1", "MMType1"}:
                        continue

                    collected_ids.add(obj_key)
                except Exception:
                    continue

        return collected_ids

    def replace_subsetted_standard14_fonts(
        self,
        target_font_ids: set[tuple[int, int]] | None = None,
    ) -> EmbeddingResult:
        """Replace embedded subsetted Standard-14 fonts with full replacements.

        Some generators embed incomplete subsets of the Standard 14 fonts.
        veraPDF may still flag these subsets for missing rendered glyphs even
        though the PDF technically embeds a font program. Replacing them with
        the bundled metrically compatible full fonts gives later sanitizers a
        stable, complete font program to work with.

        Returns:
            EmbeddingResult describing which fonts were refreshed.
        """
        result = EmbeddingResult()
        processed_font_ids: set[tuple[int, int]] = set()
        refreshed_names: set[str] = set()

        for page in self.pdf.pages:
            for font_key, font_obj in iter_all_page_fonts(page):
                try:
                    obj_key = font_obj.objgen
                    if obj_key != (0, 0):
                        if obj_key in processed_font_ids:
                            continue
                        processed_font_ids.add(obj_key)
                        if (
                            target_font_ids is not None
                            and obj_key not in target_font_ids
                        ):
                            continue

                    if not is_font_embedded(font_obj):
                        continue

                    font_name = get_font_name(font_obj)
                    if not _is_subset_font(font_name):
                        continue

                    base_name = get_base_font_name(font_name)
                    if resolve_standard14_alias(base_name) not in FONT_REPLACEMENTS:
                        continue

                    if is_symbolic_font(font_obj):
                        logger.info(
                            "Skipping refresh for symbolic subsetted "
                            "Standard-14 font '%s'",
                            font_name,
                        )
                        continue

                    font_type = get_font_type(font_obj)
                    if font_type not in {"TrueType", "Type1", "MMType1"}:
                        continue

                    success, _used_fallback = self._replace_simple_font(
                        font_obj,
                        base_name,
                        preserve_existing_encoding=True,
                    )
                    if not success:
                        if base_name not in result.fonts_failed:
                            result.fonts_failed.append(base_name)
                        continue

                    font_obj[Name.BaseFont] = Name(f"/{base_name}")
                    font_descriptor = font_obj.get("/FontDescriptor")
                    if font_descriptor is not None:
                        font_descriptor = _resolve_indirect(font_descriptor)
                        font_descriptor[Name.FontName] = Name(f"/{base_name}")

                    if base_name not in refreshed_names:
                        refreshed_names.add(base_name)
                        result.fonts_embedded.append(base_name)

                    logger.info(
                        "Replaced subsetted Standard-14 font '%s' with full '%s'",
                        font_name,
                        base_name,
                    )

                except Exception as e:
                    log_suppressed_error(
                        logger,
                        e,
                        "Error refreshing subsetted Standard-14 font %s: %s",
                        font_key,
                        e,
                    )
                    continue

        return result

    def _iter_unique_fonts(
        self,
        processed_font_ids: set[tuple[int, int]],
        *,
        skip_direct: bool = False,
    ):
        """Yield unique (font_key, font_obj) pairs from pages and AcroForm /DR.

        Indirect fonts are deduplicated via ``processed_font_ids``. Direct
        (inline) font objects cannot be deduplicated; they are yielded
        as-is unless ``skip_direct`` is True.
        """

        def _fonts():
            for page in self.pdf.pages:
                yield from iter_all_page_fonts(page)
            yield from iter_acroform_dr_fonts(self.pdf)

        for font_key, font_obj in _fonts():
            obj_key = font_obj.objgen
            if obj_key == (0, 0):
                if skip_direct:
                    continue
            else:
                if obj_key in processed_font_ids:
                    continue
                processed_font_ids.add(obj_key)
            yield font_key, font_obj

    def _iter_unique_embedded_fonts(
        self,
        processed_font_ids: set[tuple[int, int]],
    ):
        """Yield unique indirect page and AcroForm fonts."""
        for _font_key, font_obj in self._iter_unique_fonts(
            processed_font_ids, skip_direct=True
        ):
            yield font_obj

    def _get_font_descriptor_for_font(
        self,
        font_obj: pikepdf.Object,
    ) -> tuple[pikepdf.Object | None, pikepdf.Object | None]:
        """Resolve the FontDescriptor for a simple or CID font."""
        font_descriptor = font_obj.get("/FontDescriptor")
        descendant = None
        if font_descriptor is None:
            descendants = font_obj.get("/DescendantFonts")
            if descendants is not None and len(descendants) > 0:
                descendant = _resolve_indirect(descendants[0])
                font_descriptor = descendant.get("/FontDescriptor")
        if font_descriptor is None:
            return (None, descendant)
        return (_resolve_indirect(font_descriptor), descendant)

    @staticmethod
    def _find_any_font_file(
        font_descriptor: pikepdf.Object,
    ) -> tuple[Name, pikepdf.Object] | None:
        """Find any embedded font program stream in a FontDescriptor."""
        for key in (Name("/FontFile"), Name("/FontFile2"), Name("/FontFile3")):
            stream = font_descriptor.get(str(key))
            if stream is None:
                continue
            return (key, _resolve_indirect(stream))
        return None

    def _build_encoding_dictionary(
        self,
        encoding: dict[int, str],
        *,
        base_encoding: Name | None = None,
    ) -> Dictionary:
        """Creates PDF Encoding with Differences array.

        For Symbol/ZapfDingbats fonts that don't use WinAnsiEncoding.

        Args:
            encoding: Dictionary with code -> glyph name mapping.
            base_encoding: Optional BaseEncoding name to include.

        Returns:
            pikepdf Dictionary with Type=Encoding and Differences array.
        """
        differences = []
        base_glyphs = self._get_base_encoding_glyph_names(base_encoding)
        prev = -2

        for code in sorted(encoding.keys()):
            glyph_name = encoding[code]
            if base_encoding is not None and not _is_agl_glyph_name(glyph_name):
                continue
            if base_glyphs is not None and base_glyphs.get(code) == glyph_name:
                continue
            if code != prev + 1:
                # Start new sequence
                differences.append(code)
            differences.append(Name(f"/{glyph_name}"))
            prev = code

        encoding_dict = Dictionary(Type=Name.Encoding, Differences=Array(differences))
        if base_encoding is not None:
            encoding_dict[Name("/BaseEncoding")] = base_encoding
        return encoding_dict

    @staticmethod
    def _get_base_encoding_glyph_names(
        base_encoding: Name | None,
    ) -> dict[int, str] | None:
        """Return Adobe glyph names for a supported base encoding."""
        if base_encoding is None:
            return None

        if base_encoding == Name.WinAnsiEncoding:
            code_to_unicode = generate_tounicode_for_winansi()
        elif base_encoding == Name.MacRomanEncoding:
            code_to_unicode = generate_tounicode_for_macroman()
        elif str(base_encoding) == "/StandardEncoding":
            code_to_unicode = generate_tounicode_for_standard_encoding()
        else:
            return None

        return {
            code: glyph_name
            for code, unicode_val in code_to_unicode.items()
            if (glyph_name := UV2AGL.get(unicode_val)) is not None
        }

    def _create_font_stream(self, font_data: bytes) -> Stream:
        """Creates a FontFile2 stream for TrueType fonts.

        Args:
            font_data: Raw font data as bytes.

        Returns:
            pikepdf Stream object with font data.
        """
        font_stream = Stream(self.pdf, font_data)
        font_stream[Name.Length1] = len(font_data)
        return font_stream

    def _get_cidfont_encoding(self, font_obj: pikepdf.Object) -> str:
        """Extracts the encoding from a CIDFont (Type0).

        CJK text can be written horizontally (Identity-H) or vertically (Identity-V).
        This method detects the original encoding and returns it so it is preserved
        during embedding.

        Args:
            font_obj: pikepdf font object (Type0/CIDFont).

        Returns:
            'Identity-H' for horizontal writing direction (default) or
            'Identity-V' for vertical writing direction.
        """
        encoding = font_obj.get("/Encoding")
        if encoding is not None:
            encoding_name = _find_unicode_cmap_name(encoding) or _get_encoding_name(
                encoding
            )
            if encoding_name.endswith("-V"):
                return "Identity-V"
            encoding = _resolve_indirect(encoding)
            if isinstance(encoding, pikepdf.Stream):
                try:
                    if int(encoding.get("/WMode", 0)) == 1:
                        return "Identity-V"
                except (TypeError, ValueError):
                    pass
                try:
                    data = re.sub(rb"%[^\r\n]*", b"", bytes(encoding.read_bytes()))
                    if re.search(rb"/WMode\s+1\b", data):
                        return "Identity-V"
                except Exception:
                    pass
        return "Identity-H"

    def _get_cidfont_ordering(self, font_obj: pikepdf.Object) -> str:
        """Extracts the CIDSystemInfo Ordering from a CIDFont (Type0).

        Args:
            font_obj: pikepdf font object (Type0/CIDFont).

        Returns:
            Ordering string (e.g. "Japan1", "GB1") or "Identity" as default.
        """
        try:
            descendants = font_obj.get("/DescendantFonts")
            if descendants is not None and len(descendants) > 0:
                desc_font = _resolve_indirect(descendants[0])
                cid_sys = desc_font.get("/CIDSystemInfo")
                if cid_sys is not None:
                    cid_sys = _resolve_indirect(cid_sys)
                    ordering = cid_sys.get("/Ordering")
                    if ordering is not None:
                        return _safe_str(ordering)
        except Exception:
            pass
        return "Identity"

    @staticmethod
    def _looks_like_cjk_cidfont(font_name: str) -> bool:
        """Return True when a CIDFont name likely belongs to a CJK font."""
        normalized = font_name.replace(" ", "").replace("-", "")
        return any(hint in normalized for hint in _CJK_CIDFONT_NAME_HINTS)

    def _load_identity_cidfont_replacement(
        self, font_name: str
    ) -> tuple[bytes, "TTFont"]:
        """Load a Latin replacement font for an Identity-H/V CIDFont."""
        lookup_name = _LATIN_CIDFONT_SYSTEM_ALIASES.get(font_name, font_name)
        use_fallback = resolve_standard14_alias(lookup_name) not in FONT_REPLACEMENTS
        try:
            return self._loader.load_replacement_font(
                lookup_name,
                use_fallback=use_fallback,
            )
        except FontEmbeddingError:
            if lookup_name == font_name:
                raise
            use_fallback = resolve_standard14_alias(font_name) not in FONT_REPLACEMENTS
            return self._loader.load_replacement_font(
                font_name,
                use_fallback=use_fallback,
            )

    @staticmethod
    def _character_code_bytes(
        code: CharacterCode,
        code_space_ranges: tuple[tuple[bytes, bytes], ...],
    ) -> bytes | None:
        """Return the original byte representation of a Type 0 character code."""
        if isinstance(code, bytes):
            return code
        for lower, upper in code_space_ranges:
            try:
                candidate = code.to_bytes(len(lower), "big")
            except OverflowError:
                continue
            if lower <= candidate <= upper:
                return candidate
        return None

    @staticmethod
    def _unicode_cidfont_replacement_maps(
        font_obj: pikepdf.Object,
        tt_font: "TTFont",
        encoding_name: str,
        used_codes: set[CharacterCode],
        authoritative_unicode: dict[bytes, tuple[int, ...]] | None = None,
        collection_unicode: dict[int, int] | None = None,
    ) -> (
        tuple[
            dict[int, int],
            dict[bytes, int],
            tuple[tuple[bytes, bytes], ...],
        ]
        | None
    ):
        """Map Unicode character codes through the source CMap to replacement GIDs."""
        cid_encoding = get_type0_cid_encoding_map(font_obj)
        if cid_encoding is None or not used_codes:
            return None

        try:
            cmap = tt_font.getBestCmap() or {}
        except KeyError:
            cmap = {}

        glyph_name_to_gid = {
            glyph_name: gid for gid, glyph_name in enumerate(tt_font.getGlyphOrder())
        }
        fallback_gid = 0
        for fallback_unicode in (0xFFFD, 0x25A1, 0x003F):
            fallback_name = cmap.get(fallback_unicode)
            candidate = glyph_name_to_gid.get(fallback_name) if fallback_name else None
            if candidate:
                fallback_gid = candidate
                break
        code_space_ranges = get_font_code_space_ranges(font_obj)
        cid_to_gid: dict[int, int] = {}
        exact_cids: set[int] = set()
        code_to_unicode: dict[bytes, int] = {}
        has_semantic_mapping = False

        for code in sorted(
            used_codes,
            key=lambda value: (
                0 if isinstance(value, bytes) else 1,
                value if isinstance(value, bytes) else str(value).encode("ascii"),
            ),
        ):
            raw = FontEmbedder._character_code_bytes(code, code_space_ranges)
            if raw is None:
                continue
            cid = cid_encoding.map_code(raw)
            if not 0 <= cid <= 0xFFFF:
                continue

            unicode_value = None
            sequence = (
                authoritative_unicode.get(raw)
                if authoritative_unicode is not None
                else None
            )
            if sequence is not None:
                has_semantic_mapping = True
                if len(sequence) == 1:
                    unicode_value = sequence[0]
            elif _is_utf16_encoding(encoding_name):
                try:
                    if "-UCS2-" in encoding_name:
                        unicode_value = (
                            int.from_bytes(raw, "big") if len(raw) == 2 else 0xFFFD
                        )
                        if 0xD800 <= unicode_value <= 0xDFFF:
                            unicode_value = 0xFFFD
                    else:
                        decoded = raw.decode("utf-16-be")
                        unicode_value = ord(decoded) if len(decoded) == 1 else 0xFFFD
                    has_semantic_mapping = True
                except UnicodeDecodeError:
                    unicode_value = 0xFFFD
                    has_semantic_mapping = True
            elif collection_unicode is not None:
                unicode_value = collection_unicode.get(cid, 0xFFFD)
                has_semantic_mapping = True

            gid = fallback_gid
            exact = False
            if unicode_value is not None:
                code_to_unicode[raw] = unicode_value
                glyph_name = cmap.get(unicode_value)
                exact_gid = glyph_name_to_gid.get(glyph_name) if glyph_name else None
                if exact_gid:
                    gid = exact_gid
                    exact = True

            previous_gid = cid_to_gid.get(cid)
            if previous_gid is None or (exact and cid not in exact_cids):
                cid_to_gid[cid] = gid
                if exact:
                    exact_cids.add(cid)
            elif exact and previous_gid != gid:
                logger.debug(
                    "Unicode CMap maps multiple used characters to CID %d; "
                    "keeping replacement GID %d",
                    cid,
                    previous_gid,
                )

        if not cid_to_gid or not has_semantic_mapping:
            return None
        return cid_to_gid, code_to_unicode, code_space_ranges

    def _embed_cidfont(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
        *,
        encoding: str = "Identity-H",
        used_codes: set[CharacterCode] | None = None,
    ) -> bool:
        """Embeds CIDFont with Noto Sans CJK.

        The font object is rewritten in place, so all pages and resources
        referencing it pick up the replacement.

        Args:
            font_obj: The font object.
            font_name: Base name of the font (without subset prefix).
            encoding: CIDFont encoding ('Identity-H' or 'Identity-V').
            used_codes: Character codes shown with this font in content streams.

        Returns:
            True if successful, False on errors.
        """
        try:
            ordering = self._get_cidfont_ordering(font_obj)
            original_encoding = font_obj.get("/Encoding")
            encoding_name = (
                _find_unicode_cmap_name(original_encoding)
                or _get_encoding_name(original_encoding)
                if original_encoding is not None
                else ""
            )
            original_tounicode = font_obj.get("/ToUnicode")
            authoritative_unicode = None
            if original_tounicode is not None:
                try:
                    tounicode_stream = _resolve_indirect(original_tounicode)
                    if isinstance(tounicode_stream, pikepdf.Stream):
                        authoritative_unicode = parse_tounicode_cmap_sequences(
                            bytes(tounicode_stream.read_bytes())
                        )
                except Exception:
                    pass
            original_cid_system_info = None
            original_w2 = None
            original_dw2 = None
            original_descendants = font_obj.get("/DescendantFonts")
            if original_descendants is not None and len(original_descendants) > 0:
                original_descendant = _resolve_indirect(original_descendants[0])
                original_cid_system_info = original_descendant.get("/CIDSystemInfo")
                original_w2 = original_descendant.get("/W2")
                original_dw2 = original_descendant.get("/DW2")
            collection_unicode = None
            if isinstance(original_cid_system_info, pikepdf.Dictionary):
                try:
                    registry = _safe_str(original_cid_system_info.get("/Registry"))
                    supplement = int(original_cid_system_info.get("/Supplement", -1))
                    if registry == "Adobe" and supplement >= 0:
                        collection_unicode = get_cid_to_unicode(ordering)
                except (TypeError, ValueError):
                    pass
            use_cjk_replacement = (
                ordering != "Identity" or self._looks_like_cjk_cidfont(font_name)
            )

            if use_cjk_replacement:
                # Load CJK CIDFont replacement font with script-specific selection.
                font_data, tt_font = self._loader.load_cidfont_replacement_by_ordering(
                    ordering
                )
            else:
                # Malformed Latin Type0 fonts can still encode glyph IDs through
                # Identity-H/V. Keep that model and embed a Latin TrueType program
                # as the descendant CIDFont instead of substituting a CJK font.
                font_data, tt_font = self._load_identity_cidfont_replacement(font_name)

            unicode_maps = None
            if (
                _is_utf16_encoding(encoding_name)
                or authoritative_unicode
                or collection_unicode is not None
            ):
                unicode_maps = self._unicode_cidfont_replacement_maps(
                    font_obj,
                    tt_font,
                    encoding_name,
                    used_codes or set(),
                    authoritative_unicode,
                    collection_unicode,
                )

            # Build complete CIDFont structure
            new_font = self._cidfont_builder.build_structure(
                font_name, tt_font, font_data, encoding=encoding
            )
            if unicode_maps is not None and original_encoding is not None:
                cid_to_gid, code_to_unicode, code_space_ranges = unicode_maps
                new_font[Name.Encoding] = original_encoding
                new_descendant = _resolve_indirect(new_font["/DescendantFonts"][0])
                if original_cid_system_info is not None:
                    new_descendant[Name.CIDSystemInfo] = original_cid_system_info
                if original_w2 is not None:
                    new_descendant[Name.W2] = original_w2
                if original_dw2 is not None:
                    new_descendant[Name.DW2] = original_dw2

                max_cid = max(cid_to_gid)
                cidtogid_data = bytearray((max_cid + 1) * 2)
                for cid, gid in cid_to_gid.items():
                    offset = cid * 2
                    cidtogid_data[offset : offset + 2] = gid.to_bytes(2, "big")
                new_descendant[Name.CIDToGIDMap] = self.pdf.make_indirect(
                    Stream(self.pdf, bytes(cidtogid_data))
                )

                w_array = self._metrics.build_cidfont_w_array(
                    tt_font,
                    cid_to_gid,
                )
                new_descendant[Name.W] = Array(
                    Array(item) if isinstance(item, list) else item for item in w_array
                )

                if original_tounicode is not None:
                    new_font[Name.ToUnicode] = original_tounicode
                else:
                    tounicode_data = _generate_unicode_type0_tounicode(
                        code_to_unicode,
                        code_space_ranges,
                    )
                    new_font[Name.ToUnicode] = self.pdf.make_indirect(
                        Stream(self.pdf, tounicode_data)
                    )
            elif original_tounicode is not None:
                new_font[Name.ToUnicode] = original_tounicode

            # Update the font object with the new structure
            # Delete old entries
            keys_to_remove = [k for k in font_obj.keys()]
            for key in keys_to_remove:
                del font_obj[key]

            # Copy new entries
            for key, value in new_font.items():
                font_obj[key] = value

            return True

        except FontEmbeddingError as e:
            logger.error("Error embedding CIDFont '%s': %s", font_name, e)
            return False
        except Exception as e:
            logger.error(
                "Unexpected error embedding CIDFont '%s': %s",
                font_name,
                e,
            )
            return False

    def _create_font_descriptor(
        self,
        font_name: str,
        metrics: dict,
        font_stream: Stream,
    ) -> Dictionary:
        """Creates a FontDescriptor for the font.

        Args:
            font_name: Name of the font (without leading /).
            metrics: Font metrics from _extract_font_metrics().
            font_stream: FontFile2 stream.

        Returns:
            pikepdf Dictionary for the FontDescriptor.
        """
        return Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name(f"/{font_name}"),
            Flags=metrics["Flags"],
            FontBBox=Array(metrics["FontBBox"]),
            ItalicAngle=metrics["ItalicAngle"],
            Ascent=metrics["Ascent"],
            Descent=metrics["Descent"],
            CapHeight=metrics["CapHeight"],
            StemV=metrics["StemV"],
            FontFile2=self.pdf.make_indirect(font_stream),
        )

    def _generate_to_unicode_for_simple_font(self, font_name: str) -> bytes:
        """Generates ToUnicode CMap for Simple Fonts (Standard-14 replacements).

        This enables text extraction and copy/paste for PDF/A-2b compliance.
        Simple fonts use 8-bit encoding (codes 0-255) unlike CIDFonts which use
        16-bit encoding.

        Args:
            font_name: Name of the Standard-14 font being replaced.

        Returns:
            CMap data in PostScript format as bytes.
        """
        return generate_to_unicode_for_simple_font(font_name)

    def _resolve_symbol_glyph_to_unicode(self, glyph_name: str) -> int | None:
        """Resolves a Symbol font glyph name to its Unicode codepoint.

        Checks SYMBOL_GLYPH_TO_UNICODE first (for special/variant glyphs),
        then falls back to the standard Adobe Glyph List (AGL2UV).

        Args:
            glyph_name: Adobe glyph name from SYMBOL_ENCODING.

        Returns:
            Unicode codepoint, or None if the glyph has no Unicode equivalent.
        """
        return resolve_symbol_glyph_to_unicode(glyph_name)

    def _replace_simple_font(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
        *,
        use_fallback: bool = False,
        preserve_existing_encoding: bool = False,
    ) -> tuple[bool, bool]:
        """Replaces a non-embedded simple font with an embedded one.

        The font object is rewritten in place, so all pages and resources
        referencing it pick up the replacement.

        Args:
            font_obj: The font object.
            font_name: Base name of the font (without subset prefix).
            use_fallback: If True, use the fallback font (LiberationSans)
                instead of looking up font_name in FONT_REPLACEMENTS.
            preserve_existing_encoding: If True, keep the font's current
                code-to-Unicode mapping (used when refreshing subset fonts).

        Returns:
            A pair of success and whether the bundled fallback font was used.
        """
        try:
            # Load replacement font
            font_data, tt_font = self._loader.load_replacement_font(
                font_name,
                use_fallback=use_fallback,
            )
            used_fallback = self._loader.is_fallback_font(tt_font)

            # Check if symbol font
            is_symbol = font_name in SYMBOL_FONTS

            # Extract metrics (with correct Flags value)
            metrics = self._metrics.extract_metrics(tt_font, is_symbol=is_symbol)
            if metrics is None:
                logger.error("Font '%s' missing head/OS2 tables", font_name)
                return False, used_fallback

            encoding_name: Name | None = None
            encoding_dict: Dictionary | None = None

            preserved_encoding = None
            if preserve_existing_encoding and not is_symbol:
                preserved_encoding = self._build_preserved_simple_font_encoding(
                    font_obj,
                    tt_font,
                )

            # Encoding-specific width extraction and encoding object
            if preserved_encoding is not None:
                widths, encoding_dict, to_unicode_data = preserved_encoding
            elif font_name == "Symbol":
                widths = self._metrics.extract_widths_for_encoding(
                    tt_font, SYMBOL_ENCODING, SYMBOL_GLYPH_TO_UNICODE
                )
                encoding_dict = self._build_encoding_dictionary(SYMBOL_ENCODING)
                to_unicode_data = self._generate_to_unicode_for_simple_font(font_name)
            elif font_name == "ZapfDingbats":
                widths = self._metrics.extract_widths_for_encoding(
                    tt_font, ZAPFDINGBATS_ENCODING, ZAPFDINGBATS_GLYPH_TO_UNICODE
                )
                encoding_dict = self._build_encoding_dictionary(ZAPFDINGBATS_ENCODING)
                to_unicode_data = self._generate_to_unicode_for_simple_font(font_name)
            else:
                widths = self._metrics.extract_widths(tt_font)
                encoding_name = Name.WinAnsiEncoding
                to_unicode_data = self._generate_to_unicode_for_simple_font(font_name)

            # Create font stream and descriptor
            font_stream = self._create_font_stream(font_data)
            font_descriptor = self._create_font_descriptor(
                font_name, metrics, font_stream
            )

            # Update the font object
            font_obj[Name.Subtype] = Name.TrueType
            font_obj[Name.FontDescriptor] = self.pdf.make_indirect(font_descriptor)
            font_obj[Name.FirstChar] = 0
            font_obj[Name.LastChar] = 255
            font_obj[Name.Widths] = Array(widths)

            # Set encoding
            if encoding_dict is not None:
                font_obj[Name.Encoding] = self.pdf.make_indirect(encoding_dict)
            elif encoding_name is not None:
                font_obj[Name.Encoding] = encoding_name
            else:
                font_obj.pop(Name.Encoding, None)

            # Generate and attach ToUnicode CMap for text extraction
            to_unicode_stream = Stream(self.pdf, to_unicode_data)
            font_obj[Name.ToUnicode] = self.pdf.make_indirect(to_unicode_stream)

            return True, used_fallback

        except FontEmbeddingError as e:
            logger.error("Error embedding font '%s': %s", font_name, e)
            return False, False
        except Exception as e:
            logger.error(
                "Unexpected error embedding font '%s': %s",
                font_name,
                e,
            )
            return False, False

    def _build_preserved_simple_font_encoding(
        self,
        font_obj: pikepdf.Object,
        tt_font: "TTFont",
    ) -> tuple[list[int], Dictionary, bytes] | None:
        """Preserve a simple font's visible code mapping during refresh.

        Embedded subset fonts sometimes use a custom byte-to-glyph layout
        together with a ToUnicode CMap. Refreshing the font program must keep
        those byte codes mapped to the same Unicode text, otherwise rendered
        output changes even if the replacement font is metrically compatible.
        """
        code_to_unicode = self._get_existing_simple_font_mapping(font_obj)
        if not code_to_unicode:
            return None

        code_to_glyph_name = self._build_glyph_names_from_unicode_map(code_to_unicode)
        if not code_to_glyph_name:
            return None

        widths = self._build_widths_from_unicode_map(tt_font, code_to_unicode)
        if widths is None:
            return None

        encoding = self._build_encoding_dictionary(
            code_to_glyph_name,
            base_encoding=Name.WinAnsiEncoding,
        )
        to_unicode_data = generate_tounicode_cmap_data(code_to_unicode)
        return widths, encoding, to_unicode_data

    def _get_existing_simple_font_mapping(
        self,
        font_obj: pikepdf.Object,
    ) -> dict[int, int]:
        """Return the current simple font's code-to-Unicode mapping."""
        tounicode = font_obj.get("/ToUnicode")
        if tounicode is not None:
            try:
                tounicode = _resolve_indirect(tounicode)
                mapping = parse_tounicode_cmap(bytes(tounicode.read_bytes()))
                if mapping:
                    return {
                        code: unicode_val
                        for code, unicode_val in mapping.items()
                        if 0 <= code <= 255
                    }
            except Exception:
                logger.debug("Could not parse existing ToUnicode for font refresh")

        encoding = font_obj.get("/Encoding")
        if encoding is None:
            return {}

        if isinstance(encoding, pikepdf.Name):
            enc_name = _safe_str(encoding)
            if enc_name == "/WinAnsiEncoding":
                return generate_tounicode_for_winansi()
            if enc_name == "/MacRomanEncoding":
                return generate_tounicode_for_macroman()
            if enc_name == "/StandardEncoding":
                return generate_tounicode_for_standard_encoding()
            return {}

        try:
            return generate_tounicode_from_encoding_dict(encoding)
        except Exception:
            logger.debug("Could not derive encoding mapping for font refresh")
            return {}

    @staticmethod
    def _build_glyph_names_from_unicode_map(
        code_to_unicode: dict[int, int],
    ) -> dict[int, str]:
        """Build a Differences map from Unicode values using AGL names."""
        code_to_glyph_name: dict[int, str] = {}
        for code, unicode_val in code_to_unicode.items():
            glyph_name = UV2AGL.get(unicode_val)
            if glyph_name is not None:
                code_to_glyph_name[code] = glyph_name
        return code_to_glyph_name

    def _build_widths_from_unicode_map(
        self,
        tt_font: "TTFont",
        code_to_unicode: dict[int, int],
    ) -> list[int] | None:
        """Build a 256-entry width array for a preserved code mapping."""
        width_map = self._metrics.compute_widths_for_encoding(tt_font, code_to_unicode)
        if not width_map:
            return None

        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em
        notdef_width = hmtx.metrics.get(".notdef", (500, 0))[0]
        fallback_width = round(notdef_width * scale)

        return [width_map.get(code, fallback_width) for code in range(256)]

    def add_tounicode_to_embedded_fonts(self) -> EmbeddingResult:
        """Adds ToUnicode CMaps to embedded fonts lacking them.

        For PDF/A-2/3 compliance (all levels, rule 6.2.11.7.2), all
        fonts must have Unicode mappings. This method adds ToUnicode
        CMaps to embedded fonts that don't have them.

        Scans page-level Resources, Form XObjects, Tiling Patterns, and
        Annotation Appearance Streams recursively.

        Returns:
            EmbeddingResult with processing status.
        """
        result = EmbeddingResult()
        processed_fonts: set[str] = set()
        processed_font_ids: set[tuple[int, int]] = set()

        for page in self.pdf.pages:
            for font_key, font_obj in iter_all_page_fonts(page):
                try:
                    # Skip same indirect object already processed
                    obj_key = font_obj.objgen
                    if obj_key != (0, 0):
                        if obj_key in processed_font_ids:
                            continue
                        processed_font_ids.add(obj_key)

                    font_name = get_font_name(font_obj)
                    base_name = get_base_font_name(font_name)

                    # Skip if not embedded
                    if not is_font_embedded(font_obj):
                        continue

                    # Skip if already has ToUnicode
                    if has_tounicode_cmap(font_obj):
                        continue

                    # Try to add ToUnicode
                    success = self._add_tounicode_to_font(font_obj, base_name)
                    if base_name not in processed_fonts:
                        processed_fonts.add(base_name)
                        if success:
                            result.fonts_embedded.append(base_name)
                            logger.info("ToUnicode added to font: %s", base_name)
                        else:
                            result.fonts_failed.append(base_name)
                            result.warnings.append(
                                f"Could not add ToUnicode to font '{base_name}'"
                            )

                except UnicodeDecodeError:
                    logger.debug(
                        "Skipping font %s: non-UTF-8 bytes in font data",
                        font_key,
                    )
                    continue
                except Exception as e:
                    log_suppressed_error(
                        logger,
                        e,
                        "Error processing font %s: %s",
                        font_key,
                        e,
                        level=logging.INFO,
                    )
                    continue

        return result

    def _add_tounicode_to_font(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Adds ToUnicode CMap to a single font.

        Args:
            font_obj: The font object.
            font_name: Base name of the font.

        Returns:
            True if successful, False otherwise.
        """
        try:
            font_type = get_font_type(font_obj)

            if font_type == "CIDFont":
                # CIDFont/Type0: generate ToUnicode from CIDToGIDMap
                return self._add_tounicode_to_cidfont(font_obj, font_name)
            elif font_type == "Type3":
                # Type3: custom encoding with PUA fallback
                return self._add_tounicode_to_type3_font(font_obj, font_name)
            else:
                # Simple font (Type1, TrueType): generate from encoding
                return self._add_tounicode_to_simple_font(font_obj, font_name)

        except Exception as e:
            log_suppressed_error(
                logger,
                e,
                "Error adding ToUnicode to font '%s': %s",
                font_name,
                e,
                level=logging.INFO,
            )
            return False

    def _add_tounicode_to_simple_font(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Adds ToUnicode CMap to a simple font (Type1, TrueType, MMType1).

        Handles WinAnsiEncoding, MacRomanEncoding, and custom encodings
        with Differences arrays. Type3 fonts are handled separately by
        _add_tounicode_to_type3_font().

        Args:
            font_obj: The font object.
            font_name: Base name of the font.

        Returns:
            True if successful, False otherwise.
        """
        encoding = font_obj.get("/Encoding")
        code_to_unicode: dict[int, int | tuple[int, ...]] = {}
        subtype = font_obj.get("/Subtype")
        subtype_name = _safe_str(subtype) if subtype is not None else ""

        if subtype_name == "/TrueType" and (
            encoding is None or is_symbolic_font(font_obj)
        ):
            code_to_unicode = self._tounicode_from_truetype_program(font_obj)
            first_char = 0
            last_char = 255
            try:
                first_char = int(font_obj.get("/FirstChar", first_char))
            except (TypeError, ValueError):
                pass
            try:
                last_char = int(font_obj.get("/LastChar", last_char))
            except (TypeError, ValueError):
                pass
            code_to_unicode = fill_tounicode_gaps_with_pua(
                code_to_unicode,
                first_char,
                last_char,
            )
        elif encoding is None:
            # Encoding-less Type1 fonts use the Encoding in their embedded
            # program. StandardEncoding remains the fallback for fonts whose
            # program is unavailable or cannot be parsed. Fill any remaining
            # gaps with PUA codepoints for complete PDF/A coverage.
            if subtype_name in ("/Type1", "/MMType1"):
                descriptor = font_obj.get("/FontDescriptor")
                if descriptor is not None:
                    descriptor = _resolve_indirect(descriptor)
                    if isinstance(descriptor, Dictionary):
                        font_file = descriptor.get("/FontFile")
                        if font_file is not None:
                            font_file = _resolve_indirect(font_file)
                            if isinstance(font_file, Stream):
                                try:
                                    code_to_unicode = (
                                        generate_tounicode_from_type1_program(
                                            bytes(font_file.read_bytes())
                                        )
                                    )
                                except Exception:
                                    pass
                        font_file3 = descriptor.get("/FontFile3")
                        if not code_to_unicode and font_file3 is not None:
                            font_file3 = _resolve_indirect(font_file3)
                            if (
                                isinstance(font_file3, Stream)
                                and _safe_str(font_file3.get("/Subtype")) == "/Type1C"
                            ):
                                try:
                                    cff_font_name = descriptor.get("/FontName")
                                    if cff_font_name is None:
                                        cff_font_name = font_obj.get("/BaseFont")
                                    selected_name = (
                                        _safe_str(cff_font_name).removeprefix("/")
                                        if cff_font_name is not None
                                        else None
                                    )
                                    code_to_unicode = (
                                        generate_tounicode_from_cff_program(
                                            bytes(font_file3.read_bytes()),
                                            selected_name,
                                        )
                                    )
                                except Exception:
                                    pass
            if not code_to_unicode:
                code_to_unicode = generate_tounicode_for_standard_encoding()
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
            code_to_unicode = fill_tounicode_gaps_with_pua(
                code_to_unicode, first_char, last_char
            )
        elif isinstance(encoding, pikepdf.Name):
            enc_name = _safe_str(encoding)
            if enc_name == "/WinAnsiEncoding":
                code_to_unicode = generate_tounicode_for_winansi()
            elif enc_name == "/MacRomanEncoding":
                code_to_unicode = generate_tounicode_for_macroman()
            elif enc_name == "/StandardEncoding":
                code_to_unicode = generate_tounicode_for_standard_encoding()
            else:
                # Unknown encoding, try WinAnsi as fallback
                logger.debug(
                    "Unknown encoding %s for %s, using WinAnsi", enc_name, font_name
                )
                code_to_unicode = generate_tounicode_for_winansi()
        elif isinstance(encoding, pikepdf.Dictionary):
            # Encoding dictionary with potential Differences
            code_to_unicode = generate_tounicode_from_encoding_dict(encoding)
        else:
            # Try dict-like access for indirect/wrapped pikepdf objects
            try:
                code_to_unicode = generate_tounicode_from_encoding_dict(encoding)
            except Exception:
                # Can't determine encoding
                logger.info("Cannot determine encoding for font '%s'", font_name)
                return False

        if not code_to_unicode:
            return False

        # Generate and attach ToUnicode CMap
        tounicode_data = generate_tounicode_cmap_data(code_to_unicode)
        tounicode_stream = Stream(self.pdf, tounicode_data)
        font_obj[Name.ToUnicode] = self.pdf.make_indirect(tounicode_stream)

        return True

    def _tounicode_from_truetype_program(
        self,
        font_obj: pikepdf.Object,
    ) -> dict[int, int]:
        """Resolve an encoding-less or symbolic TrueType font through its cmap."""
        descriptor = font_obj.get("/FontDescriptor")
        if descriptor is None:
            return {}
        descriptor = _resolve_indirect(descriptor)
        if not isinstance(descriptor, Dictionary):
            return {}

        font_file = descriptor.get("/FontFile2")
        if font_file is None:
            font_file = descriptor.get("/FontFile3")
        if font_file is None:
            return {}
        font_file = _resolve_indirect(font_file)
        if not isinstance(font_file, Stream):
            return {}
        font_data = bytes(font_file.read_bytes())

        byte_to_glyph = _resolve_truetype_font_encoding(
            font_obj,
            font_data,
            pdfa_normalized=True,
        )
        if not byte_to_glyph:
            return {}

        from fontTools.ttLib import TTFont

        tt_font = TTFont(BytesIO(font_data))
        try:
            unicode_cmap = _get_unicode_cmap(tt_font)
        finally:
            tt_font.close()

        unicode_by_glyph: dict[str, set[int]] = {}
        for unicode_value, glyph_name in unicode_cmap.items():
            if _is_non_pua_unicode_scalar(unicode_value):
                unicode_by_glyph.setdefault(glyph_name, set()).add(unicode_value)

        base_name = get_base_font_name(get_font_name(font_obj))
        result: dict[int, int] = {}
        for code, glyph_name in byte_to_glyph.items():
            candidates = unicode_by_glyph.get(glyph_name, set())
            if len(candidates) == 1:
                result[code] = next(iter(candidates))
                continue

            preferred = resolve_symbol_glyph_to_unicode(glyph_name)
            if preferred is None:
                preferred = ZAPFDINGBATS_GLYPH_TO_UNICODE.get(glyph_name)
            if preferred is None:
                preferred = resolve_glyph_to_unicode(glyph_name)
            if (
                preferred is not None
                and _is_non_pua_unicode_scalar(preferred)
                and (not candidates or preferred in candidates)
            ):
                result[code] = preferred
                continue

            if candidates:
                continue

            if base_name == "Symbol":
                encoded_name = SYMBOL_ENCODING.get(code)
                if encoded_name is not None:
                    preferred = resolve_symbol_glyph_to_unicode(encoded_name)
            elif base_name == "ZapfDingbats":
                encoded_name = ZAPFDINGBATS_ENCODING.get(code)
                if encoded_name is not None:
                    preferred = ZAPFDINGBATS_GLYPH_TO_UNICODE.get(encoded_name)
            if preferred is not None and _is_non_pua_unicode_scalar(preferred):
                result[code] = preferred

        return result

    def _add_tounicode_to_type3_font(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Adds ToUnicode CMap to a Type3 font.

        Type3 fonts often have custom encodings with non-AGL glyph names.
        Unresolvable glyph names are mapped to the Unicode Private Use Area
        (U+E000-U+F8FF) to satisfy PDF/A-2/3 requirements (all levels).

        Args:
            font_obj: The Type3 font object.
            font_name: Base name of the font.

        Returns:
            True if successful, False otherwise.
        """
        code_to_unicode = generate_tounicode_for_type3_font(font_obj)

        if not code_to_unicode:
            logger.info("No character codes found for Type3 font '%s'", font_name)
            return False

        tounicode_data = generate_tounicode_cmap_data(code_to_unicode)
        tounicode_stream = Stream(self.pdf, tounicode_data)
        font_obj[Name.ToUnicode] = self.pdf.make_indirect(tounicode_stream)

        return True

    def _add_tounicode_to_cidfont(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Adds ToUnicode CMap to a CIDFont (Type0).

        For CIDFonts with embedded TrueType data, extracts the cmap
        table to build Unicode mappings. Handles three scenarios:

        1. UTF-16/UCS-2 encoding: character codes are already Unicode,
           so ToUnicode maps each code to itself.
        2. Identity CIDToGIDMap (or absent): CID=GID, use GID->Unicode
           directly from font's cmap.
        3. Stream CIDToGIDMap: parse CID->GID mapping, then compose
           CID->GID->Unicode.

        Args:
            font_obj: The font object (Type0).
            font_name: Base name of the font.

        Returns:
            True if successful, False otherwise.
        """
        # Get DescendantFonts to access embedded font data
        descendants = font_obj.get("/DescendantFonts")
        if descendants is None:
            return False

        # Get the first (usually only) descendant
        if len(descendants) == 0:
            return False

        desc_font = _resolve_indirect(descendants[0])

        # Check encoding on the Type0 font
        encoding = font_obj.get("/Encoding")
        encoding_name = _get_encoding_name(encoding) if encoding else ""

        # Get FontDescriptor to access embedded font
        font_descriptor = desc_font.get("/FontDescriptor")
        if font_descriptor is None:
            return False

        font_descriptor = _resolve_indirect(font_descriptor)

        # Try to extract font data from FontFile2 (TrueType)
        font_file = font_descriptor.get("/FontFile2")
        if font_file is None:
            # Try FontFile3 (CFF/OpenType)
            font_file = font_descriptor.get("/FontFile3")

        if font_file is None:
            logger.debug("No embedded font data found for %s", font_name)
            return False

        font_file = _resolve_indirect(font_file)

        # Detect bare CFF CID-keyed font (not loadable by TTFont)
        ff3_subtype = font_file.get("/Subtype")
        is_bare_cff_cid = (
            ff3_subtype is not None and str(ff3_subtype) == "/CIDFontType0C"
        )
        if is_bare_cff_cid:
            return self._add_tounicode_from_cid_collection(
                font_obj, encoding_name, font_name
            )

        try:
            # Extract and parse font data
            from io import BytesIO

            from fontTools.ttLib import TTFont

            font_data = bytes(font_file.read_bytes())
            tt_font = TTFont(BytesIO(font_data))
            try:
                # Get font's cmap table — getBestCmap() raises KeyError
                # when the cmap table is entirely absent from the font.
                cmap = None
                try:
                    cmap = tt_font.getBestCmap()
                except KeyError:
                    pass

                # Fallback: try symbol font cmap (platform 3, encoding 0)
                if cmap is None and "cmap" in tt_font:
                    for subtable in tt_font["cmap"].tables:
                        if subtable.platformID == 3 and subtable.platEncID == 0:
                            cmap = subtable.cmap
                            break

                if cmap is not None:
                    if _is_utf16_encoding(encoding_name):
                        # UTF-16/UCS-2: character codes ARE Unicode values
                        code_to_unicode = build_identity_unicode_mapping(cmap)
                    else:
                        # Build GID -> Unicode mapping from font's cmap
                        glyph_order = tt_font.getGlyphOrder()
                        glyph_name_to_gid = {
                            name: i for i, name in enumerate(glyph_order)
                        }
                        gid_to_unicode: dict[int, int] = {}

                        for unicode_val, glyph_name in cmap.items():
                            gid = glyph_name_to_gid.get(glyph_name)
                            if gid is not None:
                                if gid not in gid_to_unicode:
                                    gid_to_unicode[gid] = unicode_val

                        # Check CIDToGIDMap on the descendant CIDFont
                        cidtogidmap = desc_font.get("/CIDToGIDMap")
                        if cidtogidmap is not None and not isinstance(
                            cidtogidmap, pikepdf.Name
                        ):
                            # Stream-based CIDToGIDMap: CID != GID
                            cidtogidmap = _resolve_indirect(cidtogidmap)
                            stream_data = bytes(cidtogidmap.read_bytes())
                            cid_to_gid = parse_cidtogidmap_stream(stream_data)

                            # Compose CID -> GID -> Unicode
                            code_to_unicode = {}
                            for cid, gid in cid_to_gid.items():
                                if gid in gid_to_unicode:
                                    code_to_unicode[cid] = gid_to_unicode[gid]
                        else:
                            # Identity or absent: CID = GID
                            code_to_unicode = gid_to_unicode
                else:
                    # No cmap at all — generate PUA-based fallback mapping.
                    # Each GID gets a unique PUA codepoint.  This satisfies
                    # the formal ToUnicode requirement even though the
                    # mappings carry no semantic meaning.
                    logger.debug(
                        "No cmap table in font %s, using PUA fallback",
                        font_name,
                    )
                    num_glyphs = len(tt_font.getGlyphOrder())
                    code_to_unicode = {}
                    pua = 0xE000
                    for gid in range(1, num_glyphs):  # skip .notdef at 0
                        code_to_unicode[gid] = pua
                        pua += 1
                        if pua > 0xF8FF:
                            pua = 0xF0000  # Supplementary PUA-A

                if not code_to_unicode:
                    return False

                # Generate ToUnicode CMap (16-bit for CIDFonts)
                tounicode_data = generate_cidfont_tounicode_cmap(code_to_unicode)
                tounicode_stream = Stream(self.pdf, tounicode_data)
                font_obj[Name.ToUnicode] = self.pdf.make_indirect(tounicode_stream)

                return True
            finally:
                tt_font.close()

        except Exception as e:
            log_suppressed_error(
                logger, e, "Error parsing embedded font %s: %s", font_name, e
            )
            return False

    def _add_tounicode_from_cid_collection(
        self,
        font_obj: pikepdf.Object,
        encoding_name: str,
        font_name: str,
    ) -> bool:
        """Adds ToUnicode CMap using Adobe CID collection mapping data.

        For bare CFF CID-keyed fonts (CIDFontType0C) that lack a cmap
        table, uses pre-built CID-to-Unicode mappings derived from Adobe's
        cmap-resources.

        Args:
            font_obj: The Type0 font object.
            encoding_name: Encoding name (e.g. "Identity-H").
            font_name: Base name of the font.

        Returns:
            True if a ToUnicode CMap was successfully added.
        """
        ordering = self._get_cidfont_ordering(font_obj)
        if ordering == "Identity":
            logger.debug(
                "Cannot derive CID->Unicode for Identity ordering: %s",
                font_name,
            )
            return False

        cid_to_unicode = get_cid_to_unicode(ordering)
        if cid_to_unicode is None:
            logger.debug(
                "No CID->Unicode data for ordering '%s': %s",
                ordering,
                font_name,
            )
            return False

        if _is_utf16_encoding(encoding_name):
            # UTF-16 encoding: character codes are Unicode values
            code_to_unicode = {u: u for u in cid_to_unicode.values()}
        else:
            # CID-keyed: use CID->Unicode directly
            code_to_unicode = cid_to_unicode

        if not code_to_unicode:
            return False

        tounicode_data = generate_cidfont_tounicode_cmap(code_to_unicode)
        tounicode_stream = Stream(self.pdf, tounicode_data)
        font_obj[Name.ToUnicode] = self.pdf.make_indirect(tounicode_stream)

        logger.debug(
            "Added ToUnicode from %s collection for %s",
            ordering,
            font_name,
        )
        return True
