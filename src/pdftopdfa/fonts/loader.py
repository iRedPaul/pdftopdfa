# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font loading for PDF/A compliance."""

from importlib import resources
from typing import TYPE_CHECKING

from ..exceptions import FontEmbeddingError
from .constants import (
    CIDFONT_REPLACEMENT,
    CJK_FONT_INDEX,
    FALLBACK_FONT,
    FONT_REPLACEMENTS,
)

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont


class FontLoader:
    """Loads and caches font files.

    This helper class handles loading replacement fonts from resources
    and caching them for reuse.
    """

    def __init__(self, font_cache: dict[str, tuple[bytes, "TTFont"]]) -> None:
        """Initializes the FontLoader.

        Args:
            font_cache: Shared cache dictionary for loaded fonts.
        """
        self._font_cache = font_cache

    def load_standard14_font(self, font_name: str) -> tuple[bytes, "TTFont"]:
        """Loads a replacement font for Standard-14 fonts from resources.

        Args:
            font_name: Name of the Standard-14 font.

        Returns:
            Tuple of (font data as bytes, TTFont object).

        Raises:
            FontEmbeddingError: If the font cannot be loaded.
        """
        from fontTools.ttLib import TTFont

        if font_name in self._font_cache:
            return self._font_cache[font_name]

        replacement_file = FONT_REPLACEMENTS.get(font_name)
        if replacement_file is None:
            raise FontEmbeddingError(f"No replacement defined for font '{font_name}'")

        # Load font from resources
        try:
            font_ref = (
                resources.files("pdftopdfa") / "resources" / "fonts" / replacement_file
            )
            font_data = font_ref.read_bytes()
        except Exception as e:
            raise FontEmbeddingError(
                f"Could not load replacement font '{replacement_file}': {e}"
            ) from e

        # Parse font with fonttools
        from io import BytesIO

        tt_font = TTFont(BytesIO(font_data))
        self._font_cache[font_name] = (font_data, tt_font)
        return font_data, tt_font

    def load_fallback_font(self) -> tuple[bytes, "TTFont"]:
        """Loads the fallback font (LiberationSans) for unknown fonts.

        Used when a non-embedded font has no specific replacement
        in FONT_REPLACEMENTS.

        Returns:
            Tuple of (font data as bytes, TTFont object).

        Raises:
            FontEmbeddingError: If the font cannot be loaded.
        """
        from fontTools.ttLib import TTFont

        cache_key = "__fallback__"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        try:
            font_ref = (
                resources.files("pdftopdfa") / "resources" / "fonts" / FALLBACK_FONT
            )
            font_data = font_ref.read_bytes()
        except Exception as e:
            raise FontEmbeddingError(
                f"Could not load fallback font '{FALLBACK_FONT}': {e}"
            ) from e

        from io import BytesIO

        tt_font = TTFont(BytesIO(font_data))
        self._font_cache[cache_key] = (font_data, tt_font)
        return font_data, tt_font

    def load_cidfont_replacement_by_ordering(
        self, ordering: str
    ) -> tuple[bytes, "TTFont"]:
        """Loads the CIDFont replacement font for a specific CJK ordering.

        Selects the correct font from the TTC based on CJK_FONT_INDEX.
        Falls back to index 0 (Simplified Chinese) for unknown orderings.

        Args:
            ordering: CIDSystemInfo Ordering value (e.g. "Japan1", "GB1").

        Returns:
            Tuple of (font data as bytes, TTFont object).

        Raises:
            FontEmbeddingError: If the font cannot be loaded.
        """
        from fontTools.ttLib import TTCollection, TTFont

        font_index = CJK_FONT_INDEX.get(ordering, 0)
        cache_key = f"__cidfont_{font_index}__"

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        # Load font from resources
        try:
            font_ref = (
                resources.files("pdftopdfa")
                / "resources"
                / "fonts"
                / CIDFONT_REPLACEMENT
            )
            font_data = font_ref.read_bytes()
        except Exception as e:
            raise FontEmbeddingError(
                f"Could not load CIDFont replacement font '{CIDFONT_REPLACEMENT}': {e}"
            ) from e

        # Parse font with fonttools (TTC = TrueType Collection)
        from io import BytesIO

        if CIDFONT_REPLACEMENT.endswith(".ttc"):
            ttc = TTCollection(BytesIO(font_data))
            try:
                if font_index >= len(ttc.fonts):
                    font_index = 0
                # Serialize the single font for FontFile2
                buf = BytesIO()
                ttc.fonts[font_index].save(buf)
                font_data = buf.getvalue()
            finally:
                ttc.close()
            tt_font = TTFont(BytesIO(font_data))
        else:
            tt_font = TTFont(BytesIO(font_data))

        self._font_cache[cache_key] = (font_data, tt_font)
        return font_data, tt_font
