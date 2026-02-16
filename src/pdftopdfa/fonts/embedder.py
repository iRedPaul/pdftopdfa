# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font embedding for PDF/A compliance."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pikepdf
from pikepdf import Array, Dictionary, Name, Stream

from ..exceptions import FontEmbeddingError
from ..utils import resolve_indirect as _resolve_indirect
from .analysis import (
    get_base_font_name,
    get_font_name,
    get_font_type,
    has_tounicode_cmap,
    is_font_embedded,
    is_symbolic_font,
)
from .cid_unicode import get_cid_to_unicode
from .cidfont import CIDFontBuilder
from .constants import FONT_REPLACEMENTS, SYMBOL_FONTS
from .constants import UTF16_ENCODING_NAMES as _UTF16_ENCODING_NAMES
from .encodings import SYMBOL_ENCODING, ZAPFDINGBATS_ENCODING
from .glyph_mapping import SYMBOL_GLYPH_TO_UNICODE, ZAPFDINGBATS_GLYPH_TO_UNICODE
from .loader import FontLoader
from .metrics import FontMetricsExtractor
from .subsetter import FontSubsetter, SubsettingResult
from .tounicode import (
    build_identity_unicode_mapping,
    fill_tounicode_gaps_with_pua,
    generate_cidfont_tounicode_cmap,
    generate_to_unicode_for_simple_font,
    generate_tounicode_cmap_data,
    generate_tounicode_for_macroman,
    generate_tounicode_for_standard_encoding,
    generate_tounicode_for_type3_font,
    generate_tounicode_for_winansi,
    generate_tounicode_from_encoding_dict,
    parse_cidtogidmap_stream,
    resolve_glyph_to_unicode,
    resolve_symbol_glyph_to_unicode,
)
from .traversal import iter_all_page_fonts
from .utils import get_encoding_name as _get_encoding_name
from .utils import safe_str as _safe_str

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont

logger = logging.getLogger(__name__)


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


class FontEmbedder:
    """Embeds missing fonts in a PDF.

    Replaces Standard-14 fonts with metrically compatible Liberation fonts.
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

        Scans page-level Resources, Form XObjects, Tiling Patterns, and
        Annotation Appearance Streams recursively.

        Returns:
            EmbeddingResult with embedding status.
        """
        result = EmbeddingResult()
        processed_fonts: set[str] = set()
        preserved_fonts: set[str] = set()
        processed_font_ids: set[tuple[int, int]] = set()

        for page in self.pdf.pages:
            for font_key, font_obj in iter_all_page_fonts(page):
                try:
                    # Skip same indirect object seen on another page
                    obj_key = font_obj.objgen
                    if obj_key != (0, 0):
                        if obj_key in processed_font_ids:
                            continue
                        processed_font_ids.add(obj_key)

                    font_name = get_font_name(font_obj)
                    base_name = get_base_font_name(font_name)

                    # Check if already embedded
                    if is_font_embedded(font_obj):
                        # Track already embedded fonts (without duplicates)
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

                    # Check for CIDFont/Type0 (CJK fonts)
                    font_type = get_font_type(font_obj)
                    if font_type == "CIDFont":
                        # Extract encoding (Identity-H or Identity-V)
                        encoding = self._get_cidfont_encoding(font_obj)
                        success = self._embed_cidfont(
                            page, font_key, font_obj, base_name, encoding=encoding
                        )
                        if base_name not in processed_fonts:
                            processed_fonts.add(base_name)
                            if success:
                                result.fonts_embedded.append(base_name)
                                logger.info(
                                    "CIDFont embedded: %s (Encoding: %s)",
                                    base_name,
                                    encoding,
                                )
                            else:
                                result.fonts_failed.append(base_name)
                        continue

                    # Replace font (use fallback for unknown fonts)
                    use_fallback = base_name not in FONT_REPLACEMENTS
                    success = self._replace_font_in_page(
                        page,
                        font_key,
                        font_obj,
                        base_name,
                        use_fallback=use_fallback,
                    )
                    if base_name not in processed_fonts:
                        processed_fonts.add(base_name)
                        if success:
                            result.fonts_embedded.append(base_name)
                            if use_fallback:
                                warning = (
                                    f"No specific replacement for"
                                    f" font '{base_name}',"
                                    f" using LiberationSans as"
                                    f" fallback"
                                )
                                result.warnings.append(warning)
                                logger.warning(warning)
                            else:
                                logger.info("Font embedded: %s", base_name)
                        else:
                            result.fonts_failed.append(base_name)

                except UnicodeDecodeError:
                    logger.debug(
                        "Skipping font %s: non-UTF-8 bytes in font data",
                        font_key,
                    )
                    continue
                except Exception as e:
                    logger.info("Error with font %s: %s", font_key, e)
                    continue

        # Embed non-embedded fonts in AcroForm DR (Default Resources)
        try:
            root = self.pdf.Root
            if root is not None and "/AcroForm" in root:
                acroform = _resolve_indirect(root.AcroForm)
                dr = acroform.get("/DR")
                if dr is not None:
                    dr = _resolve_indirect(dr)
                    font_dict = dr.get("/Font")
                    if font_dict is not None:
                        font_dict = _resolve_indirect(font_dict)
                        for font_key in list(font_dict.keys()):
                            try:
                                font_obj = _resolve_indirect(font_dict[font_key])
                                obj_key = font_obj.objgen
                                if obj_key != (0, 0):
                                    if obj_key in processed_font_ids:
                                        continue
                                    processed_font_ids.add(obj_key)

                                font_name = get_font_name(font_obj)
                                base_name = get_base_font_name(font_name)

                                if is_font_embedded(font_obj):
                                    if (
                                        base_name not in preserved_fonts
                                        and base_name not in processed_fonts
                                    ):
                                        preserved_fonts.add(base_name)
                                    continue

                                font_type = get_font_type(font_obj)
                                if font_type == "CIDFont":
                                    encoding = self._get_cidfont_encoding(font_obj)
                                    success = self._embed_cidfont(
                                        self.pdf.pages[0],
                                        str(font_key),
                                        font_obj,
                                        base_name,
                                        encoding=encoding,
                                    )
                                else:
                                    use_fallback = base_name not in FONT_REPLACEMENTS
                                    success = self._replace_font_in_page(
                                        self.pdf.pages[0],
                                        str(font_key),
                                        font_obj,
                                        base_name,
                                        use_fallback=use_fallback,
                                    )
                                if base_name not in processed_fonts:
                                    processed_fonts.add(base_name)
                                    if success:
                                        result.fonts_embedded.append(base_name)
                                        logger.info(
                                            "AcroForm DR font embedded: %s",
                                            base_name,
                                        )
                                    else:
                                        result.fonts_failed.append(base_name)
                            except Exception as e:
                                logger.info(
                                    "Error with AcroForm DR font %s: %s",
                                    font_key,
                                    e,
                                )
                                continue
        except Exception:
            pass

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

    def _build_encoding_dictionary(self, encoding: dict[int, str]) -> Dictionary:
        """Creates PDF Encoding with Differences array.

        For Symbol/ZapfDingbats fonts that don't use WinAnsiEncoding.

        Args:
            encoding: Dictionary with code -> glyph name mapping.

        Returns:
            pikepdf Dictionary with Type=Encoding and Differences array.
        """
        differences = []
        prev = -2

        for code in sorted(encoding.keys()):
            if code != prev + 1:
                # Start new sequence
                differences.append(code)
            differences.append(Name(f"/{encoding[code]}"))
            prev = code

        return Dictionary(Type=Name.Encoding, Differences=Array(differences))

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
            enc_str = _safe_str(encoding)
            if "Identity-V" in enc_str:
                return "Identity-V"
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

    def _embed_cidfont(
        self,
        page: pikepdf.Page,
        font_key: str,
        font_obj: pikepdf.Object,
        font_name: str,
        *,
        encoding: str = "Identity-H",
    ) -> bool:
        """Embeds CIDFont with Noto Sans CJK.

        Args:
            page: Page where the font is used.
            font_key: Key of the font in the font dictionary.
            font_obj: The font object.
            font_name: Base name of the font (without subset prefix).
            encoding: CIDFont encoding ('Identity-H' or 'Identity-V').

        Returns:
            True if successful, False on errors.
        """
        try:
            # Load CIDFont replacement font with script-specific selection
            ordering = self._get_cidfont_ordering(font_obj)
            font_data, tt_font = self._loader.load_cidfont_replacement_by_ordering(
                ordering
            )

            # Build complete CIDFont structure
            new_font = self._cidfont_builder.build_structure(
                font_name, tt_font, font_data, encoding=encoding
            )

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

    def _replace_font_in_page(
        self,
        page: pikepdf.Page,
        font_key: str,
        font_obj: pikepdf.Object,
        font_name: str,
        *,
        use_fallback: bool = False,
    ) -> bool:
        """Replaces a non-embedded font with an embedded one.

        Args:
            page: Page where the font is used.
            font_key: Key of the font in the font dictionary.
            font_obj: The font object.
            font_name: Base name of the font (without subset prefix).
            use_fallback: If True, use the fallback font (LiberationSans)
                instead of looking up font_name in FONT_REPLACEMENTS.

        Returns:
            True if successful, False on errors.
        """
        try:
            # Load replacement font
            if use_fallback:
                font_data, tt_font = self._loader.load_fallback_font()
            else:
                font_data, tt_font = self._loader.load_standard14_font(font_name)

            # Check if symbol font
            is_symbol = font_name in SYMBOL_FONTS

            # Extract metrics (with correct Flags value)
            metrics = self._metrics.extract_metrics(tt_font, is_symbol=is_symbol)
            if metrics is None:
                logger.error("Font '%s' missing head/OS2 tables", font_name)
                return False

            # Encoding-specific width extraction and encoding object
            if font_name == "Symbol":
                widths = self._metrics.extract_widths_for_encoding(
                    tt_font, SYMBOL_ENCODING, SYMBOL_GLYPH_TO_UNICODE
                )
                encoding = self._build_encoding_dictionary(SYMBOL_ENCODING)
            elif font_name == "ZapfDingbats":
                widths = self._metrics.extract_widths_for_encoding(
                    tt_font, ZAPFDINGBATS_ENCODING, ZAPFDINGBATS_GLYPH_TO_UNICODE
                )
                encoding = self._build_encoding_dictionary(ZAPFDINGBATS_ENCODING)
            else:
                widths = self._metrics.extract_widths(tt_font)
                encoding = None  # WinAnsiEncoding as Name

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
            if encoding is not None:
                # Symbol font: Encoding dictionary with Differences
                font_obj[Name.Encoding] = self.pdf.make_indirect(encoding)
            else:
                # Standard font: WinAnsiEncoding
                font_obj[Name.Encoding] = Name.WinAnsiEncoding

            # Generate and attach ToUnicode CMap for text extraction
            to_unicode_data = self._generate_to_unicode_for_simple_font(font_name)
            to_unicode_stream = Stream(self.pdf, to_unicode_data)
            font_obj[Name.ToUnicode] = self.pdf.make_indirect(to_unicode_stream)

            return True

        except FontEmbeddingError as e:
            logger.error("Error embedding font '%s': %s", font_name, e)
            return False
        except Exception as e:
            logger.error(
                "Unexpected error embedding font '%s': %s",
                font_name,
                e,
            )
            return False

    def fix_font_encodings(self) -> int:
        """Fixes encoding issues on embedded simple fonts for PDF/A rule 6.2.11.6.

        For TrueType fonts:
        - Non-symbolic: ensures encoding is WinAnsi or MacRoman
        - Symbolic: removes /Encoding, ensures MS Symbol (3,0) cmap

        For Type1/MMType1:
        - Adds /WinAnsiEncoding if missing (non-symbolic only)

        Returns:
            Number of fonts modified.
        """
        modified = 0
        processed_font_ids: set[tuple[int, int]] = set()

        for page in self.pdf.pages:
            for font_key, font_obj in iter_all_page_fonts(page):
                try:
                    obj_key = font_obj.objgen
                    if obj_key != (0, 0):
                        if obj_key in processed_font_ids:
                            continue
                        processed_font_ids.add(obj_key)

                    if not is_font_embedded(font_obj):
                        continue

                    font_name = get_font_name(font_obj)
                    base_name = get_base_font_name(font_name)

                    if self._fix_truetype_encoding(font_obj, base_name):
                        modified += 1

                except Exception as e:
                    logger.debug("Error fixing encoding for %s: %s", font_key, e)
                    continue

        return modified

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
                    logger.info("Error processing font %s: %s", font_key, e)
                    continue

        return result

    def _fix_truetype_encoding(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Fixes encoding issues on simple fonts for PDF/A rule 6.2.11.6.

        For TrueType fonts:
        - Non-symbolic: ensures encoding is WinAnsi or MacRoman
        - Symbolic: removes /Encoding and ensures MS Symbol (3,0) cmap

        For Type1/MMType1 fonts:
        - Adds /WinAnsiEncoding if missing (non-symbolic only)

        Args:
            font_obj: The font object.
            font_name: Base name of the font (for logging).

        Returns:
            True if any changes were made, False otherwise.
        """
        font_type = get_font_type(font_obj)
        if font_type == "TrueType":
            if is_symbolic_font(font_obj):
                changed = self._fix_symbolic_truetype_encoding(font_obj, font_name)
                changed |= self._fix_symbolic_truetype_cmap(font_obj, font_name)
                return changed
            else:
                return self._fix_nonsymbolic_truetype_encoding(font_obj, font_name)
        elif font_type in ("Type1", "MMType1"):
            if font_obj.get("/Encoding") is not None:
                return False
            if is_symbolic_font(font_obj):
                return False
            font_obj[Name.Encoding] = Name.WinAnsiEncoding
            logger.info(
                "Added /WinAnsiEncoding to font: %s (ISO 19005-2, 6.2.11.6)",
                font_name,
            )
            return True
        return False

    def _fix_nonsymbolic_truetype_encoding(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Fixes encoding on non-symbolic TrueType fonts (FM1).

        Ensures encoding is WinAnsiEncoding or MacRomanEncoding.
        Handles Name encodings, Dictionary encodings with wrong BaseEncoding,
        and Differences arrays with non-AGL glyph names.

        Args:
            font_obj: The font object.
            font_name: Base name of the font (for logging).

        Returns:
            True if changes were made, False otherwise.
        """
        encoding = font_obj.get("/Encoding")

        if encoding is None:
            font_obj[Name.Encoding] = Name.WinAnsiEncoding
            logger.info("Added /WinAnsiEncoding to TrueType font: %s", font_name)
            return True

        # Resolve indirect reference
        encoding = _resolve_indirect(encoding)

        if isinstance(encoding, pikepdf.Name):
            enc_name = _safe_str(encoding)
            if enc_name in ("/WinAnsiEncoding", "/MacRomanEncoding"):
                return False
            # Wrong encoding (e.g. StandardEncoding) → replace
            font_obj[Name.Encoding] = Name.WinAnsiEncoding
            logger.info(
                "Replaced %s with /WinAnsiEncoding on TrueType font: %s",
                enc_name,
                font_name,
            )
            return True

        # Dictionary encoding
        try:
            encoding.get  # noqa: B018 — test for dict-like access
        except AttributeError:
            return False

        changed = False

        # Check BaseEncoding
        base_enc = encoding.get("/BaseEncoding")
        if base_enc is not None:
            base_name = _safe_str(base_enc)
            if base_name not in ("/WinAnsiEncoding", "/MacRomanEncoding"):
                encoding[Name("/BaseEncoding")] = Name.WinAnsiEncoding
                logger.info(
                    "Fixed BaseEncoding %s → /WinAnsiEncoding on TrueType font: %s",
                    base_name,
                    font_name,
                )
                changed = True
        else:
            # No BaseEncoding → add WinAnsiEncoding
            encoding[Name("/BaseEncoding")] = Name.WinAnsiEncoding
            changed = True

        # Check Differences for non-AGL glyph names
        differences = encoding.get("/Differences")
        if differences is not None:
            if self._has_non_agl_differences(differences):
                del encoding[Name("/Differences")]
                logger.info(
                    "Removed non-AGL /Differences from TrueType font: %s",
                    font_name,
                )
                changed = True
            else:
                # Valid AGL differences — ensure font has MS Unicode cmap
                if self._ensure_microsoft_unicode_cmap(font_obj, font_name):
                    changed = True

        return changed

    def _fix_symbolic_truetype_encoding(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Removes /Encoding from symbolic TrueType fonts (FM2).

        PDF/A rule 6.2.11.6 forbids /Encoding on symbolic TrueType fonts.

        Args:
            font_obj: The font object.
            font_name: Base name of the font (for logging).

        Returns:
            True if /Encoding was removed, False otherwise.
        """
        if font_obj.get("/Encoding") is not None:
            del font_obj[Name.Encoding]
            logger.info(
                "Removed /Encoding from symbolic TrueType font: %s",
                font_name,
            )
            return True
        return False

    def _fix_symbolic_truetype_cmap(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Ensures symbolic TrueType has Microsoft Symbol (3,0) cmap (FM3).

        If the font program has exactly one cmap subtable or already has
        a non-empty (3,0) subtable, no changes are needed. Otherwise,
        creates or repairs the (3,0) subtable with mappings in the
        0xF000 range derived from a suitable source cmap.

        Args:
            font_obj: The font object.
            font_name: Base name of the font (for logging).

        Returns:
            True if the font was modified, False otherwise.
        """
        try:
            font_descriptor = font_obj.get("/FontDescriptor")
            if font_descriptor is None:
                return False
            font_descriptor = _resolve_indirect(font_descriptor)

            font_file = font_descriptor.get("/FontFile2")
            if font_file is None:
                return False
            font_file = _resolve_indirect(font_file)

            from io import BytesIO

            from fontTools.ttLib import TTFont

            font_data = bytes(font_file.read_bytes())
            tt_font = TTFont(BytesIO(font_data))
            try:
                cmap_table = tt_font.get("cmap")
                if cmap_table is None:
                    return False

                subtables = cmap_table.tables

                # If exactly one subtable, veraPDF accepts it
                if len(subtables) == 1:
                    return False

                # Check if (3,0) MS Symbol already exists and is non-empty
                existing_30 = None
                for st in subtables:
                    if st.platformID == 3 and st.platEncID == 0:
                        existing_30 = st
                        break

                if existing_30 is not None and existing_30.cmap:
                    return False

                # Find a source cmap to build (3,0) from (skip empty (3,0))
                source_subtables = [
                    st
                    for st in subtables
                    if st.cmap and not (st.platformID == 3 and st.platEncID == 0)
                ]
                source = self._find_best_cmap_source(source_subtables)
                if source is None:
                    return False

                # Build mapping in 0xF000 range
                new_mapping: dict[int, str] = {}
                for code, glyph_name in source.cmap.items():
                    sym_code = (code & 0xFF) | 0xF000
                    new_mapping[sym_code] = glyph_name

                from fontTools.ttLib.tables._c_m_a_p import (
                    cmap_format_4,
                )

                if existing_30 is not None:
                    # Repair empty (3,0) in-place
                    existing_30.cmap = new_mapping
                else:
                    # Create new (3,0) subtable
                    new_subtable = cmap_format_4(4)
                    new_subtable.platformID = 3
                    new_subtable.platEncID = 0
                    new_subtable.language = 0
                    new_subtable.cmap = new_mapping
                    cmap_table.tables.append(new_subtable)

                # Write modified font back
                out = BytesIO()
                tt_font.save(out)
                new_font_data = out.getvalue()
            finally:
                tt_font.close()

            new_stream = Stream(self.pdf, new_font_data)
            new_stream[Name.Length1] = len(new_font_data)
            font_descriptor[Name("/FontFile2")] = self.pdf.make_indirect(new_stream)

            logger.info(
                "%s Microsoft Symbol (3,0) cmap for font: %s",
                "Repaired" if existing_30 is not None else "Added",
                font_name,
            )
            return True

        except Exception as e:
            logger.debug("Error fixing cmap for font %s: %s", font_name, e)
            return False

    @staticmethod
    def _find_best_cmap_source(subtables: list) -> object | None:
        """Finds the best source cmap subtable for building (3,0).

        Priority: (1,0) Mac Roman → (3,1) MS Unicode → first available.

        Args:
            subtables: List of cmap subtables.

        Returns:
            Best subtable, or None if no subtables available.
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

    @staticmethod
    def _has_non_agl_differences(differences: pikepdf.Object) -> bool:
        """Checks if a Differences array contains non-AGL glyph names.

        Args:
            differences: The /Differences array from an Encoding dict.

        Returns:
            True if any glyph name is not in the Adobe Glyph List.
        """
        for item in differences:
            if isinstance(item, pikepdf.Name):
                glyph_name = _safe_str(item).lstrip("/")
                if glyph_name == ".notdef":
                    continue
                if resolve_glyph_to_unicode(glyph_name) is None:
                    return True
        return False

    def _ensure_microsoft_unicode_cmap(
        self,
        font_obj: pikepdf.Object,
        font_name: str,
    ) -> bool:
        """Ensures non-symbolic TrueType with Differences has MS Unicode cmap.

        Adds a (3,1) Microsoft Unicode cmap subtable if not already present.

        Args:
            font_obj: The font object.
            font_name: Base name of the font (for logging).

        Returns:
            True if the font was modified, False otherwise.
        """
        try:
            font_descriptor = font_obj.get("/FontDescriptor")
            if font_descriptor is None:
                return False
            font_descriptor = _resolve_indirect(font_descriptor)

            font_file = font_descriptor.get("/FontFile2")
            if font_file is None:
                return False
            font_file = _resolve_indirect(font_file)

            from io import BytesIO

            from fontTools.ttLib import TTFont

            font_data = bytes(font_file.read_bytes())
            tt_font = TTFont(BytesIO(font_data))

            cmap_table = tt_font.get("cmap")
            if cmap_table is None:
                return False

            # Check if (3,1) already exists
            for st in cmap_table.tables:
                if st.platformID == 3 and st.platEncID == 1:
                    tt_font.close()
                    return False

            # Build (3,1) from best available source
            source = self._find_best_cmap_source(cmap_table.tables)
            if source is None:
                tt_font.close()
                return False

            from fontTools.ttLib.tables._c_m_a_p import (
                cmap_format_4,
            )

            new_subtable = cmap_format_4(4)
            new_subtable.platformID = 3
            new_subtable.platEncID = 1
            new_subtable.language = 0
            new_subtable.cmap = dict(source.cmap)

            cmap_table.tables.append(new_subtable)

            out = BytesIO()
            tt_font.save(out)
            new_font_data = out.getvalue()
            tt_font.close()

            new_stream = Stream(self.pdf, new_font_data)
            new_stream[Name.Length1] = len(new_font_data)
            font_descriptor[Name("/FontFile2")] = self.pdf.make_indirect(new_stream)

            logger.info(
                "Added Microsoft Unicode (3,1) cmap to font: %s",
                font_name,
            )
            return True

        except Exception as e:
            logger.debug(
                "Error ensuring MS Unicode cmap for font %s: %s",
                font_name,
                e,
            )
            return False

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
            logger.info("Error adding ToUnicode to font '%s': %s", font_name, e)
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
        code_to_unicode: dict[int, int] = {}

        if encoding is None:
            # No encoding specified — per PDF spec, the implicit encoding
            # for non-symbolic Type1 fonts is StandardEncoding.
            # For symbolic fonts, StandardEncoding has gaps (codes 0-31,
            # 128-160, etc.), so fill gaps with PUA codepoints to ensure
            # complete ToUnicode coverage for PDF/A compliance.
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
                encoding.get  # noqa: B018
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
            logger.debug("Error parsing embedded font %s: %s", font_name, e)
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
