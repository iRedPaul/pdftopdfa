# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font constants and replacement mappings."""

# Mapping from Standard-14 fonts to Liberation font replacements
FONT_REPLACEMENTS = {
    # Helvetica -> LiberationSans
    "Helvetica": "LiberationSans-Regular.ttf",
    "Helvetica-Bold": "LiberationSans-Bold.ttf",
    "Helvetica-Oblique": "LiberationSans-Italic.ttf",
    "Helvetica-BoldOblique": "LiberationSans-BoldItalic.ttf",
    # Times -> LiberationSerif
    "Times-Roman": "LiberationSerif-Regular.ttf",
    "Times-Bold": "LiberationSerif-Bold.ttf",
    "Times-Italic": "LiberationSerif-Italic.ttf",
    "Times-BoldItalic": "LiberationSerif-BoldItalic.ttf",
    # Courier -> LiberationMono
    "Courier": "LiberationMono-Regular.ttf",
    "Courier-Bold": "LiberationMono-Bold.ttf",
    "Courier-Oblique": "LiberationMono-Italic.ttf",
    "Courier-BoldOblique": "LiberationMono-BoldItalic.ttf",
    # Symbol fonts (SIL OFL licensed)
    "Symbol": "STIXTwoMath-Regular.ttf",
    "ZapfDingbats": "NotoSansSymbols2-Regular.ttf",
}

# Symbol fonts (require special encoding handling)
SYMBOL_FONTS = frozenset({"Symbol", "ZapfDingbats"})

# Fallback font for non-Standard-14 fonts without specific replacement
FALLBACK_FONT = "LiberationSans-Regular.ttf"

# CIDFont replacement for CJK fonts (Chinese, Japanese, Korean)
CIDFONT_REPLACEMENT = "NotoSansCJK-Regular.ttc"

# CIDFont ordering → TTC font index mapping
# NotoSansCJK TTC contains multiple fonts for different scripts
CJK_FONT_INDEX: dict[str, int] = {
    "Japan1": 2,  # Japanese
    "CNS1": 1,  # Traditional Chinese
    "GB1": 0,  # Simplified Chinese
    "Korea1": 3,  # Korean
    "Identity": 0,  # Default to SC
}

# UTF-16/UCS-2 CMap encoding names where character codes are already Unicode
UTF16_ENCODING_NAMES = frozenset(
    {
        "UniJIS-UTF16-H",
        "UniJIS-UTF16-V",
        "UniGB-UTF16-H",
        "UniGB-UTF16-V",
        "UniCNS-UTF16-H",
        "UniCNS-UTF16-V",
        "UniKS-UTF16-H",
        "UniKS-UTF16-V",
        "UniJIS-UCS2-H",
        "UniJIS-UCS2-V",
        "UniGB-UCS2-H",
        "UniGB-UCS2-V",
        "UniCNS-UCS2-H",
        "UniCNS-UCS2-V",
        "UniKS-UCS2-H",
        "UniKS-UCS2-V",
    }
)

# Standard 14 PDF fonts (not embedded in standard PDFs)
STANDARD_14_FONTS = frozenset(
    {
        "Courier",
        "Courier-Bold",
        "Courier-BoldOblique",
        "Courier-Oblique",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-BoldOblique",
        "Helvetica-Oblique",
        "Times-Roman",
        "Times-Bold",
        "Times-BoldItalic",
        "Times-Italic",
        "Symbol",
        "ZapfDingbats",
    }
)
