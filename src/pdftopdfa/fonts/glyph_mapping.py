# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Glyph name mapping for Symbol and ZapfDingbats fonts.

This module provides mappings from Adobe glyph names to Unicode codepoints
for fonts that use non-standard naming conventions. The Noto Sans Symbols 2
font uses Unicode-based glyph names (e.g., 'uni2701') rather than Adobe names
(e.g., 'a1'), so we need to map between them.
"""

from fontTools.agl import AGL2UV, LEGACY_AGL2UV, toUnicode

from .encodings import ZAPFDINGBATS_ENCODING

ADOBE_GLYPH_NAME_FALLBACKS: dict[str, int] = {
    "nbspace": 0x00A0,
    "sfthyphen": 0x00AD,
}

# ZapfDingbats glyph names are interpreted using fontTools' authoritative table.
ZAPFDINGBATS_GLYPH_TO_UNICODE: dict[str, int | None] = {
    glyph_name: ord(toUnicode(glyph_name, isZapfDingbats=True))
    for glyph_name in ZAPFDINGBATS_ENCODING.values()
}

# Symbol font: Exceptions for glyphs not in standard AGL2UV
# Only for glyphs where STIX Two Math uses different names or lacks the glyph
SYMBOL_GLYPH_TO_UNICODE: dict[str, int | None] = {
    # Construction glyphs (used to build large brackets/radicals)
    # These have no standalone Unicode equivalent
    "radicalex": None,  # Radical extender - construction glyph
    "arrowvertex": None,  # Vertical arrow extender - construction glyph
    "arrowhorizex": 0x23AF,  # HORIZONTAL LINE EXTENSION (if available)
    # Greek variant forms
    "theta1": 0x03D1,  # GREEK THETA SYMBOL (variant)
    "phi1": 0x03D5,  # GREEK PHI SYMBOL (variant)
    "omega1": 0x03D6,  # GREEK PI SYMBOL (variant omega)
    "sigma1": 0x03C2,  # GREEK SMALL LETTER FINAL SIGMA
    "Upsilon1": 0x03D2,  # GREEK UPSILON WITH HOOK SYMBOL
    # Other special glyphs
    "suchthat": 0x220B,  # CONTAINS AS MEMBER
    "universal": 0x2200,  # FOR ALL
    "existential": 0x2203,  # THERE EXISTS
    "asteriskmath": 0x2217,  # ASTERISK OPERATOR
    "perpendicular": 0x22A5,  # UP TACK
    "similar": 0x223C,  # TILDE OPERATOR
    "congruent": 0x2245,  # APPROXIMATELY EQUAL TO
    "propersuperset": 0x2283,  # SUPERSET OF
    "reflexsuperset": 0x2287,  # SUPERSET OF OR EQUAL TO
    "notsubset": 0x2284,  # NOT A SUBSET OF
    "propersubset": 0x2282,  # SUBSET OF
    "reflexsubset": 0x2286,  # SUBSET OF OR EQUAL TO
    "element": 0x2208,  # ELEMENT OF
    "notelement": 0x2209,  # NOT AN ELEMENT OF
    "registerserif": 0x00AE,  # REGISTERED SIGN
    "copyrightserif": 0x00A9,  # COPYRIGHT SIGN
    "trademarkserif": 0x2122,  # TRADE MARK SIGN
    "registersans": 0x00AE,  # REGISTERED SIGN
    "copyrightsans": 0x00A9,  # COPYRIGHT SIGN
    "trademarksans": 0x2122,  # TRADE MARK SIGN
    "weierstrass": 0x2118,  # SCRIPT CAPITAL P (Weierstrass p)
    "Ifraktur": 0x2111,  # BLACK-LETTER CAPITAL I
    "Rfraktur": 0x211C,  # BLACK-LETTER CAPITAL R
    "aleph": 0x2135,  # ALEF SYMBOL
    "minute": 0x2032,  # PRIME
    "second": 0x2033,  # DOUBLE PRIME
    "dotmath": 0x22C5,  # DOT OPERATOR
    "circlemultiply": 0x2297,  # CIRCLED TIMES
    "circleplus": 0x2295,  # CIRCLED PLUS
    "emptyset": 0x2205,  # EMPTY SET
    "lozenge": 0x25CA,  # LOZENGE
    "angleleft": 0x2329,  # LEFT-POINTING ANGLE BRACKET
    "angleright": 0x232A,  # RIGHT-POINTING ANGLE BRACKET
    "gradient": 0x2207,  # NABLA
    "integraltp": 0x2320,  # TOP HALF INTEGRAL
    "integralbt": 0x2321,  # BOTTOM HALF INTEGRAL
    "integralex": None,  # Integral extender - construction glyph
    # Bracket parts (construction glyphs)
    "parenlefttp": 0x239B,  # LEFT PARENTHESIS UPPER HOOK
    "parenleftex": 0x239C,  # LEFT PARENTHESIS EXTENSION
    "parenleftbt": 0x239D,  # LEFT PARENTHESIS LOWER HOOK
    "parenrighttp": 0x239E,  # RIGHT PARENTHESIS UPPER HOOK
    "parenrightex": 0x239F,  # RIGHT PARENTHESIS EXTENSION
    "parenrightbt": 0x23A0,  # RIGHT PARENTHESIS LOWER HOOK
    "bracketlefttp": 0x23A1,  # LEFT SQUARE BRACKET UPPER CORNER
    "bracketleftex": 0x23A2,  # LEFT SQUARE BRACKET EXTENSION
    "bracketleftbt": 0x23A3,  # LEFT SQUARE BRACKET LOWER CORNER
    "bracketrighttp": 0x23A4,  # RIGHT SQUARE BRACKET UPPER CORNER
    "bracketrightex": 0x23A5,  # RIGHT SQUARE BRACKET EXTENSION
    "bracketrightbt": 0x23A6,  # RIGHT SQUARE BRACKET LOWER CORNER
    "bracelefttp": 0x23A7,  # LEFT CURLY BRACKET UPPER HOOK
    "braceleftmid": 0x23A8,  # LEFT CURLY BRACKET MIDDLE PIECE
    "braceleftbt": 0x23A9,  # LEFT CURLY BRACKET LOWER HOOK
    "bracerighttp": 0x23AB,  # RIGHT CURLY BRACKET UPPER HOOK
    "bracerightmid": 0x23AC,  # RIGHT CURLY BRACKET MIDDLE PIECE
    "bracerightbt": 0x23AD,  # RIGHT CURLY BRACKET LOWER HOOK
    "braceex": 0x23AA,  # CURLY BRACKET EXTENSION
    "arrowboth": 0x2194,  # LEFT RIGHT ARROW
    "arrowleft": 0x2190,  # LEFTWARDS ARROW
    "arrowup": 0x2191,  # UPWARDS ARROW
    "arrowright": 0x2192,  # RIGHTWARDS ARROW
    "arrowdown": 0x2193,  # DOWNWARDS ARROW
    "arrowdblboth": 0x21D4,  # LEFT RIGHT DOUBLE ARROW
    "arrowdblleft": 0x21D0,  # LEFTWARDS DOUBLE ARROW
    "arrowdblup": 0x21D1,  # UPWARDS DOUBLE ARROW
    "arrowdblright": 0x21D2,  # RIGHTWARDS DOUBLE ARROW
    "arrowdbldown": 0x21D3,  # DOWNWARDS DOUBLE ARROW
    "carriagereturn": 0x21B5,  # DOWNWARDS ARROW WITH CORNER LEFTWARDS
}


def resolve_glyph_name(
    adobe_name: str,
    cmap: dict[int, str],
    hmtx_metrics: dict[str, tuple[int, int]],
    custom_mapping: dict[str, int | None] | None = None,
) -> str | None:
    """Resolve an Adobe glyph name to an actual font glyph name.

    This function tries the TrueType lookup order used for PDF glyph names:
    map the name to Unicode and use the font cmap first, then fall back to a
    direct glyph-name lookup in the font program.

    Args:
        adobe_name: The Adobe glyph name to resolve (e.g., 'a1', 'Alpha').
        cmap: The font's character map (codepoint -> glyph name).
        hmtx_metrics: The font's horizontal metrics (glyph name -> (width, lsb)).
        custom_mapping: Optional custom mapping dict (glyph name -> Unicode).

    Returns:
        The actual glyph name in the font, or None if not found.
    """
    # Custom mapping (for ZapfDingbats a1-a206 and Symbol exceptions)
    if custom_mapping and adobe_name in custom_mapping:
        unicode_val = custom_mapping[adobe_name]
        if unicode_val is not None:
            glyph_name = cmap.get(unicode_val)
            if glyph_name and glyph_name in hmtx_metrics:
                return glyph_name

    # Standard AGL2UV mapping
    if adobe_name in AGL2UV:
        unicode_val = AGL2UV[adobe_name]
        glyph_name = cmap.get(unicode_val)
        if glyph_name and glyph_name in hmtx_metrics:
            return glyph_name

    legacy_values = LEGACY_AGL2UV.get(adobe_name)
    if legacy_values is not None and len(legacy_values) == 1:
        glyph_name = cmap.get(legacy_values[0])
        if glyph_name and glyph_name in hmtx_metrics:
            return glyph_name

    if adobe_name in ADOBE_GLYPH_NAME_FALLBACKS:
        unicode_val = ADOBE_GLYPH_NAME_FALLBACKS[adobe_name]
        glyph_name = cmap.get(unicode_val)
        if glyph_name and glyph_name in hmtx_metrics:
            return glyph_name

    if adobe_name.startswith("uni") and len(adobe_name) == 7:
        try:
            unicode_val = int(adobe_name[3:], 16)
        except ValueError:
            unicode_val = None
        if unicode_val is not None:
            glyph_name = cmap.get(unicode_val)
            if glyph_name and glyph_name in hmtx_metrics:
                return glyph_name

    # Fonts without a usable Unicode cmap can still expose post glyph names.
    if adobe_name in hmtx_metrics:
        return adobe_name

    return None
