# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font metrics extraction for PDF/A compliance."""

from typing import TYPE_CHECKING

from .glyph_mapping import resolve_glyph_name

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont


class FontMetricsExtractor:
    """Extracts font metrics from TTFont objects.

    This is a stateless helper class that handles all font metrics extraction
    for both Standard-14 fonts and CIDFonts.
    """

    def _compute_font_flags(self, tt_font: "TTFont", *, is_symbol: bool = False) -> int:
        """Computes PDF font flags from TrueType font data.

        PDF Font Flags (ISO 32000):
        - Bit 1 (1): FixedPitch - Monospace font
        - Bit 2 (2): Serif - Font has serifs
        - Bit 3 (4): Symbolic - Non-standard character set
        - Bit 4 (8): Script - Cursive/handwriting style
        - Bit 6 (32): Nonsymbolic - Standard Latin character set
        - Bit 7 (64): Italic - Slanted glyphs

        Args:
            tt_font: fonttools TTFont object.
            is_symbol: True for symbol fonts (sets Symbolic flag).

        Returns:
            Integer with combined font flags.
        """
        flags = 0

        # Symbolic vs Nonsymbolic (mutually exclusive)
        if is_symbol:
            flags |= 4  # Symbolic
        else:
            flags |= 32  # Nonsymbolic

        os2 = tt_font.get("OS/2")

        # FixedPitch: from post.isFixedPitch
        if "post" in tt_font:
            if getattr(tt_font["post"], "isFixedPitch", 0):
                flags |= 1  # FixedPitch

        # Serif, Script, Italic detection (only for non-symbol fonts)
        if not is_symbol and os2 is not None:
            family_class = getattr(os2, "sFamilyClass", 0) >> 8

            # Serif: sFamilyClass 1-7 are serif families
            if 1 <= family_class <= 7:
                flags |= 2  # Serif

            # Script: sFamilyClass 10 is Scripts
            if family_class == 10:
                flags |= 8  # Script

            # Italic: fsSelection bit 0
            if getattr(os2, "fsSelection", 0) & 0x0001:
                flags |= 64  # Italic

        # Italic also from post.italicAngle (non-zero = italic)
        if not is_symbol and "post" in tt_font:
            if getattr(tt_font["post"], "italicAngle", 0) != 0:
                flags |= 64  # Italic

        return flags

    def extract_metrics(
        self, tt_font: "TTFont", *, is_symbol: bool = False
    ) -> dict | None:
        """Extracts font metrics from a TrueType font.

        Args:
            tt_font: fonttools TTFont object.
            is_symbol: True for symbol fonts (sets Flags=4 instead of 32).

        Returns:
            Dictionary with font metrics (FontBBox, Ascent, Descent, etc.).
        """
        if "head" not in tt_font or "OS/2" not in tt_font:
            return None
        head = tt_font["head"]
        os2 = tt_font["OS/2"]
        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em

        # FontBBox (scaled to 1000 units)
        font_bbox = [
            int(head.xMin * scale),
            int(head.yMin * scale),
            int(head.xMax * scale),
            int(head.yMax * scale),
        ]

        # Ascent/Descent (scaled)
        ascent = int(os2.sTypoAscender * scale)
        descent = int(os2.sTypoDescender * scale)

        # CapHeight and StemV
        cap_height = int(getattr(os2, "sCapHeight", 700) * scale)
        # Estimate StemV from usWeightClass: 10 + 220 * (weight/1000)^2
        weight = getattr(os2, "usWeightClass", 400)
        stem_v = int(10 + 220 * (weight / 1000) ** 2)

        # ItalicAngle from post table
        italic_angle = 0
        if "post" in tt_font:
            italic_angle = tt_font["post"].italicAngle

        # Compute comprehensive font flags from TrueType data
        flags = self._compute_font_flags(tt_font, is_symbol=is_symbol)

        return {
            "FontBBox": font_bbox,
            "Ascent": ascent,
            "Descent": descent,
            "CapHeight": cap_height,
            "StemV": stem_v,
            "ItalicAngle": italic_angle,
            "Flags": flags,
        }

    def extract_widths(self, tt_font: "TTFont") -> list[int]:
        """Extracts character widths for Latin-1 (0-255).

        Args:
            tt_font: fonttools TTFont object.

        Returns:
            List with 256 width values (for characters 0-255).
        """
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        try:
            cmap = tt_font.getBestCmap()
        except KeyError:
            cmap = None
        if cmap is None:
            cmap = self._get_any_cmap(tt_font)
        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em

        if cmap is None:
            notdef_width = hmtx.metrics.get(".notdef", (500, 0))[0]
            return [round(notdef_width * scale)] * 256

        widths = []
        for char_code in range(256):
            glyph_name = cmap.get(char_code)
            if glyph_name and glyph_name in hmtx.metrics:
                width = round(hmtx.metrics[glyph_name][0] * scale)
            else:
                # Fallback: width of .notdef or space
                notdef_width = hmtx.metrics.get(".notdef", (500, 0))[0]
                width = round(notdef_width * scale)
            widths.append(width)

        return widths

    def extract_widths_for_encoding(
        self,
        tt_font: "TTFont",
        encoding: dict[int, str],
        glyph_mapping: dict[str, int | None] | None = None,
    ) -> list[int]:
        """Extracts widths based on encoding glyph names.

        For Symbol/ZapfDingbats fonts that use special encodings.

        Args:
            tt_font: fonttools TTFont object.
            encoding: Dictionary with code -> glyph name mapping.
            glyph_mapping: Optional mapping from Adobe glyph names to Unicode
                codepoints for fonts that use Unicode-based glyph names.

        Returns:
            List with 256 width values.
        """
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        try:
            cmap = tt_font.getBestCmap()
        except KeyError:
            cmap = None
        if cmap is None:
            cmap = self._get_any_cmap(tt_font)
        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em
        notdef_width = hmtx.metrics.get(".notdef", (500, 0))[0]

        widths = []
        for code in range(256):
            adobe_name = encoding.get(code)
            if adobe_name:
                # Resolve Adobe glyph name to actual font glyph name
                glyph = resolve_glyph_name(
                    adobe_name, cmap, hmtx.metrics, glyph_mapping
                )
                if glyph:
                    width = round(hmtx.metrics[glyph][0] * scale)
                else:
                    # Glyph not found - use .notdef width
                    width = round(notdef_width * scale)
            else:
                # No glyph defined for this code
                width = round(notdef_width * scale)
            widths.append(width)

        return widths

    def compute_widths_for_encoding(
        self,
        tt_font: "TTFont",
        code_to_unicode: dict[int, int],
    ) -> dict[int, int]:
        """Computes glyph widths for a code-to-Unicode mapping.

        For each character code, maps to Unicode, then to a glyph via cmap,
        and reads the width from hmtx. Useful for validating existing font
        widths against the embedded font program.

        Args:
            tt_font: fonttools TTFont object.
            code_to_unicode: Mapping from character codes to Unicode codepoints.

        Returns:
            Dictionary mapping char_code to width (scaled to 1000 units).
            Only includes codes where a glyph was found in the font.
        """
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        try:
            cmap = tt_font.getBestCmap()
        except KeyError:
            cmap = None
        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em

        # If getBestCmap() returns None (e.g. symbol fonts like Wingdings),
        # try to get a cmap dict from any available subtable.
        if cmap is None:
            cmap = self._get_any_cmap(tt_font)
        if cmap is None:
            return {}

        result: dict[int, int] = {}
        for code, unicode_val in code_to_unicode.items():
            glyph_name = cmap.get(unicode_val)
            if glyph_name and glyph_name in hmtx.metrics:
                result[code] = round(hmtx.metrics[glyph_name][0] * scale)

        # Fallback for symbol fonts (e.g. Wingdings, Symbol): try the
        # Microsoft Symbol encoding convention where codepoints are
        # mapped at 0xF000 + charcode.
        if not result:
            for code, unicode_val in code_to_unicode.items():
                sym_val = 0xF000 + unicode_val
                glyph_name = cmap.get(sym_val)
                if glyph_name and glyph_name in hmtx.metrics:
                    result[code] = round(hmtx.metrics[glyph_name][0] * scale)
            # Also try direct charcode lookup (some symbol fonts map
            # directly from charcode without Unicode indirection)
            if not result:
                for code, _unicode_val in code_to_unicode.items():
                    glyph_name = cmap.get(code)
                    if glyph_name and glyph_name in hmtx.metrics:
                        result[code] = round(hmtx.metrics[glyph_name][0] * scale)

        return result

    @staticmethod
    def _get_any_cmap(tt_font: "TTFont") -> dict[int, str] | None:
        """Gets a cmap dict from any available subtable.

        Used as fallback when getBestCmap() returns None, which happens
        for symbol fonts (platform 3, encoding 0) and Mac-only fonts.
        """
        if "cmap" not in tt_font:
            return None
        cmap_table = tt_font["cmap"]
        # Prefer Windows Symbol (3,0), then Mac Roman (1,0)
        for subtable in cmap_table.tables:
            if subtable.platformID == 3 and subtable.platEncID == 0:
                return subtable.cmap
        for subtable in cmap_table.tables:
            if subtable.platformID == 1 and subtable.platEncID == 0:
                return subtable.cmap
        # Return any available subtable
        for subtable in cmap_table.tables:
            if subtable.cmap:
                return subtable.cmap
        return None

    def compute_widths_for_gids(
        self,
        tt_font: "TTFont",
        gids: set[int],
    ) -> dict[int, int]:
        """Computes glyph widths for a set of GIDs.

        Args:
            tt_font: fonttools TTFont object.
            gids: Set of glyph IDs.

        Returns:
            Dictionary mapping GID to width (scaled to 1000 units).
        """
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em
        glyph_order = tt_font.getGlyphOrder()

        result: dict[int, int] = {}
        for gid in gids:
            if gid < len(glyph_order):
                glyph_name = glyph_order[gid]
                if glyph_name in hmtx.metrics:
                    result[gid] = round(hmtx.metrics[glyph_name][0] * scale)

        return result

    def build_cidfont_w_array(self, tt_font: "TTFont") -> list:
        """Creates /W array for CIDFont (sparse format).

        The /W array contains character widths in CIDFont-specific format:
        [cid [w1 w2 ...]] for consecutive CIDs or
        [cid_start cid_end width] for equal widths.

        Args:
            tt_font: fonttools TTFont object.

        Returns:
            List in sparse format for the /W array.
        """
        head = tt_font["head"]
        hmtx = tt_font["hmtx"]
        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em

        glyph_order = tt_font.getGlyphOrder()

        # Collect all widths
        widths: list[tuple[int, int]] = []
        for gid, glyph_name in enumerate(glyph_order):
            if glyph_name in hmtx.metrics:
                width = round(hmtx.metrics[glyph_name][0] * scale)
            else:
                width = round(hmtx.metrics.get(".notdef", (500, 0))[0] * scale)
            widths.append((gid, width))

        # Create W array using both PDF formats:
        #   Format 1: start_cid [w1 w2 ...] — individual widths
        #   Format 2: cid_first cid_last width — range of same width
        w_array: list = []
        i = 0
        while i < len(widths):
            start_gid = widths[i][0]

            # Find the full run of consecutive GIDs
            j = i + 1
            while j < len(widths) and widths[j][0] == start_gid + (j - i):
                j += 1
            run = widths[i:j]

            # Split run into sub-runs: same-width ranges vs mixed sequences
            k = 0
            while k < len(run):
                w = run[k][1]
                # Count consecutive entries with the same width
                m = k + 1
                while m < len(run) and run[m][1] == w:
                    m += 1
                same_count = m - k

                if same_count >= 4:
                    # Range format: cid_first cid_last width
                    w_array.append(run[k][0])
                    w_array.append(run[m - 1][0])
                    w_array.append(w)
                    k = m
                else:
                    # Individual format: collect until next same-width run (≥4)
                    end = m
                    while end < len(run):
                        w2 = run[end][1]
                        lookahead = end + 1
                        while lookahead < len(run) and run[lookahead][1] == w2:
                            lookahead += 1
                        if lookahead - end >= 4:
                            break
                        end = lookahead
                    seq_widths = [run[n][1] for n in range(k, end)]
                    w_array.append(run[k][0])
                    w_array.append(seq_widths)
                    k = end

            i = j

        return w_array
