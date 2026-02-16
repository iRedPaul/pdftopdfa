# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""CIDFont building for PDF/A compliance."""

from typing import TYPE_CHECKING

import pikepdf
from pikepdf import Array, Dictionary, Name, Stream

from .metrics import FontMetricsExtractor
from .tounicode import generate_cidfont_tounicode_cmap

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont


class CIDFontBuilder:
    """Builds CIDFont/Type0 PDF structures.

    This helper class handles creating the complete font hierarchy
    for CIDFonts including CIDSystemInfo, FontDescriptor, ToUnicode CMap,
    and W array.
    """

    def __init__(
        self,
        pdf: pikepdf.Pdf,
        metrics_extractor: FontMetricsExtractor,
    ) -> None:
        """Initializes the CIDFontBuilder.

        Args:
            pdf: Opened pikepdf PDF object.
            metrics_extractor: FontMetricsExtractor instance for metrics extraction.
        """
        self._pdf = pdf
        self._metrics = metrics_extractor

    def build_structure(
        self,
        font_name: str,
        tt_font: "TTFont",
        font_data: bytes,
        *,
        encoding: str = "Identity-H",
    ) -> Dictionary:
        """Creates complete Type0/CIDFont structure.

        Builds the complete font hierarchy for CIDFonts:
        - Type0 Dictionary (main font)
        - CIDFont Dictionary (descendant)
        - CIDSystemInfo
        - FontDescriptor with FontFile2
        - ToUnicode CMap stream
        - /W array

        Args:
            font_name: Name of the font to replace.
            tt_font: fonttools TTFont object of the replacement font.
            font_data: Raw font data as bytes.
            encoding: CIDFont encoding, either 'Identity-H' (horizontal, default)
                     or 'Identity-V' (vertical for CJK text).

        Returns:
            pikepdf Dictionary for the Type0 font.
        """
        # Extract metrics
        metrics = self._metrics.extract_metrics(tt_font, is_symbol=False)
        if metrics is None:
            msg = f"Font '{font_name}' missing head/OS2 tables"
            raise ValueError(msg)

        # Create font stream (FontFile2 for TrueType-based CIDFonts)
        font_stream = Stream(self._pdf, font_data)
        font_stream[Name.Length1] = len(font_data)

        # Create FontDescriptor
        font_descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name(f"/{font_name}"),
            Flags=metrics["Flags"],
            FontBBox=Array(metrics["FontBBox"]),
            ItalicAngle=metrics["ItalicAngle"],
            Ascent=metrics["Ascent"],
            Descent=metrics["Descent"],
            CapHeight=metrics["CapHeight"],
            StemV=metrics["StemV"],
            FontFile2=self._pdf.make_indirect(font_stream),
        )

        # W array for character widths
        w_array = self._metrics.build_cidfont_w_array(tt_font)

        # Default width (for CIDs not explicitly specified)
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        scale = 1000.0 / head.unitsPerEm
        default_width = int(hmtx.metrics.get(".notdef", (500, 0))[0] * scale)

        # CIDSystemInfo
        cid_system_info = Dictionary(
            Registry=pikepdf.String("Adobe"),
            Ordering=pikepdf.String("Identity"),
            Supplement=0,
        )

        # CIDFont Dictionary (DescendantFont)
        cid_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name(f"/{font_name}"),
            CIDSystemInfo=cid_system_info,
            FontDescriptor=self._pdf.make_indirect(font_descriptor),
            DW=default_width,
            W=Array(self._convert_w_array_to_pikepdf(w_array)),
            CIDToGIDMap=Name.Identity,
        )

        # ToUnicode CMap stream
        to_unicode_data = self._generate_to_unicode_cmap(tt_font)
        to_unicode_stream = Stream(self._pdf, to_unicode_data)

        # Type0 (main font) Dictionary
        type0_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type0,
            BaseFont=Name(f"/{font_name}"),
            Encoding=Name(f"/{encoding}"),
            DescendantFonts=Array([self._pdf.make_indirect(cid_font)]),
            ToUnicode=self._pdf.make_indirect(to_unicode_stream),
        )

        return type0_font

    def _generate_to_unicode_cmap(self, tt_font: "TTFont") -> bytes:
        """Generates ToUnicode CMap for PDF/A text extraction.

        The ToUnicode CMap enables mapping CIDs to Unicode characters,
        which is required for copy/paste and text extraction from the PDF.

        Args:
            tt_font: fonttools TTFont object.

        Returns:
            CMap data in PostScript format as bytes.
        """
        try:
            cmap = tt_font.getBestCmap()
        except KeyError:
            cmap = None
        if cmap is None:
            cmap = {}

        # Create reverse mapping: GID -> Unicode
        # For Identity encoding: CID = GID
        glyph_order = tt_font.getGlyphOrder()
        glyph_name_to_gid = {name: i for i, name in enumerate(glyph_order)}
        gid_to_unicode: dict[int, int] = {}

        for unicode_val, glyph_name in cmap.items():
            gid = glyph_name_to_gid.get(glyph_name)
            if gid is not None:
                # Only store the first Unicode value per GID
                if gid not in gid_to_unicode:
                    gid_to_unicode[gid] = unicode_val

        return generate_cidfont_tounicode_cmap(gid_to_unicode)

    def _convert_w_array_to_pikepdf(self, w_array: list) -> list:
        """Converts the W array to pikepdf-compatible format.

        Args:
            w_array: W array in Python format [cid, [widths], ...].

        Returns:
            List with pikepdf-compatible objects.
        """
        result = []
        for item in w_array:
            if isinstance(item, list):
                result.append(Array(item))
            else:
                result.append(item)
        return result
