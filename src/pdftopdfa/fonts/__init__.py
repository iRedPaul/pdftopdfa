# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font analysis and embedding for PDF/A compliance."""

from ..exceptions import FontEmbeddingError
from .analysis import (
    FontInfo,
    analyze_fonts,
    can_derive_unicode,
    check_font_compliance,
    check_unicode_compliance,
    get_fonts_missing_tounicode,
    get_missing_fonts,
    has_tounicode_cmap,
    is_font_embedded,
    is_symbolic_font,
)
from .cidfont import CIDFontBuilder
from .constants import FONT_REPLACEMENTS, STANDARD_14_FONTS
from .embedder import EmbeddingResult, FontEmbedder
from .loader import FontLoader
from .metrics import FontMetricsExtractor
from .subsetter import SubsettingResult

__all__ = [
    # Exceptions
    "FontEmbeddingError",
    # Analysis
    "FontInfo",
    "analyze_fonts",
    "can_derive_unicode",
    "check_font_compliance",
    "check_unicode_compliance",
    "get_fonts_missing_tounicode",
    "get_missing_fonts",
    "has_tounicode_cmap",
    "is_font_embedded",
    "is_symbolic_font",
    # Constants
    "FONT_REPLACEMENTS",
    "STANDARD_14_FONTS",
    # Embedding
    "EmbeddingResult",
    "FontEmbedder",
    # Subsetting
    "SubsettingResult",
    # Helper classes
    "FontMetricsExtractor",
    "FontLoader",
    "CIDFontBuilder",
]
