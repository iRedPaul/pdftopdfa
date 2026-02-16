# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Glyph name mapping for Symbol and ZapfDingbats fonts.

This module provides mappings from Adobe glyph names to Unicode codepoints
for fonts that use non-standard naming conventions. The Noto Sans Symbols 2
font uses Unicode-based glyph names (e.g., 'uni2701') rather than Adobe names
(e.g., 'a1'), so we need to map between them.
"""

from fontTools.agl import AGL2UV

# ZapfDingbats: Adobe glyph names (a1-a206) -> Unicode codepoints
# Based on Adobe ZapfDingbats encoding specification
ZAPFDINGBATS_GLYPH_TO_UNICODE: dict[str, int | None] = {
    "space": 0x0020,
    "a1": 0x2701,  # UPPER BLADE SCISSORS
    "a2": 0x2702,  # BLACK SCISSORS
    "a3": 0x2704,  # WHITE SCISSORS
    "a4": 0x260E,  # BLACK TELEPHONE
    "a5": 0x2706,  # TELEPHONE LOCATION SIGN
    "a6": 0x2709,  # ENVELOPE
    "a7": 0x275B,  # HEAVY SINGLE TURNED COMMA QUOTATION MARK ORNAMENT
    "a8": 0x275C,  # HEAVY SINGLE COMMA QUOTATION MARK ORNAMENT
    "a9": 0x275D,  # HEAVY DOUBLE TURNED COMMA QUOTATION MARK ORNAMENT
    "a10": 0x275E,  # HEAVY DOUBLE COMMA QUOTATION MARK ORNAMENT
    "a11": 0x2761,  # CURVED STEM PARAGRAPH SIGN ORNAMENT
    "a12": 0x2762,  # HEAVY EXCLAMATION MARK ORNAMENT
    "a13": 0x2763,  # HEAVY HEART EXCLAMATION MARK ORNAMENT
    "a14": 0x2764,  # HEAVY BLACK HEART
    "a15": 0x2765,  # ROTATED HEAVY BLACK HEART BULLET
    "a16": 0x2766,  # FLORAL HEART
    "a17": 0x2767,  # ROTATED FLORAL HEART BULLET
    "a18": 0x2663,  # BLACK CLUB SUIT
    "a19": 0x2666,  # BLACK DIAMOND SUIT
    "a20": 0x2665,  # BLACK HEART SUIT
    "a21": 0x2660,  # BLACK SPADE SUIT
    "a22": 0x2460,  # CIRCLED DIGIT ONE
    "a23": 0x2461,  # CIRCLED DIGIT TWO
    "a24": 0x2462,  # CIRCLED DIGIT THREE
    "a25": 0x2463,  # CIRCLED DIGIT FOUR
    "a26": 0x2464,  # CIRCLED DIGIT FIVE
    "a27": 0x2465,  # CIRCLED DIGIT SIX
    "a28": 0x2466,  # CIRCLED DIGIT SEVEN
    "a29": 0x2467,  # CIRCLED DIGIT EIGHT
    "a30": 0x2468,  # CIRCLED DIGIT NINE
    "a31": 0x2469,  # CIRCLED NUMBER TEN
    "a32": 0x2776,  # DINGBAT NEGATIVE CIRCLED DIGIT ONE
    "a33": 0x2777,  # DINGBAT NEGATIVE CIRCLED DIGIT TWO
    "a34": 0x2778,  # DINGBAT NEGATIVE CIRCLED DIGIT THREE
    "a35": 0x2779,  # DINGBAT NEGATIVE CIRCLED DIGIT FOUR
    "a36": 0x277A,  # DINGBAT NEGATIVE CIRCLED DIGIT FIVE
    "a37": 0x277B,  # DINGBAT NEGATIVE CIRCLED DIGIT SIX
    "a38": 0x277C,  # DINGBAT NEGATIVE CIRCLED DIGIT SEVEN
    "a39": 0x277D,  # DINGBAT NEGATIVE CIRCLED DIGIT EIGHT
    "a40": 0x277E,  # DINGBAT NEGATIVE CIRCLED DIGIT NINE
    "a41": 0x277F,  # DINGBAT NEGATIVE CIRCLED NUMBER TEN
    "a42": 0x2780,  # DINGBAT CIRCLED SANS-SERIF DIGIT ONE
    "a43": 0x2781,  # DINGBAT CIRCLED SANS-SERIF DIGIT TWO
    "a44": 0x2782,  # DINGBAT CIRCLED SANS-SERIF DIGIT THREE
    "a45": 0x2783,  # DINGBAT CIRCLED SANS-SERIF DIGIT FOUR
    "a46": 0x2784,  # DINGBAT CIRCLED SANS-SERIF DIGIT FIVE
    "a47": 0x2785,  # DINGBAT CIRCLED SANS-SERIF DIGIT SIX
    "a48": 0x2786,  # DINGBAT CIRCLED SANS-SERIF DIGIT SEVEN
    "a49": 0x2787,  # DINGBAT CIRCLED SANS-SERIF DIGIT EIGHT
    "a50": 0x2788,  # DINGBAT CIRCLED SANS-SERIF DIGIT NINE
    "a51": 0x2789,  # DINGBAT CIRCLED SANS-SERIF NUMBER TEN
    "a52": 0x278A,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT ONE
    "a53": 0x278B,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT TWO
    "a54": 0x278C,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT THREE
    "a55": 0x278D,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT FOUR
    "a56": 0x278E,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT FIVE
    "a57": 0x278F,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT SIX
    "a58": 0x2790,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT SEVEN
    "a59": 0x2791,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT EIGHT
    "a60": 0x2792,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF DIGIT NINE
    "a61": 0x2793,  # DINGBAT NEGATIVE CIRCLED SANS-SERIF NUMBER TEN
    "a62": 0x2794,  # HEAVY WIDE-HEADED RIGHTWARDS ARROW
    "a63": 0x2192,  # RIGHTWARDS ARROW
    "a64": 0x27A3,  # THREE-D TOP-LIGHTED RIGHTWARDS ARROWHEAD
    "a65": 0x2195,  # UP DOWN ARROW (placeholder - actual varies)
    "a66": 0x2799,  # HEAVY RIGHTWARDS ARROW
    "a67": 0x279B,  # DRAFTING POINT RIGHTWARDS ARROW
    "a68": 0x279C,  # HEAVY ROUND-TIPPED RIGHTWARDS ARROW
    "a69": 0x279D,  # TRIANGLE-HEADED RIGHTWARDS ARROW
    "a70": 0x279E,  # HEAVY TRIANGLE-HEADED RIGHTWARDS ARROW
    "a71": 0x279F,  # DASHED TRIANGLE-HEADED RIGHTWARDS ARROW
    "a72": 0x27A0,  # HEAVY DASHED TRIANGLE-HEADED RIGHTWARDS ARROW
    "a73": 0x27A1,  # BLACK RIGHTWARDS ARROW
    "a74": 0x27A2,  # THREE-D TOP-LIGHTED RIGHTWARDS ARROWHEAD
    "a75": 0x27A4,  # BLACK RIGHTWARDS ARROWHEAD
    "a76": 0x27A5,  # HEAVY BLACK CURVED DOWNWARDS AND RIGHTWARDS ARROW
    "a77": 0x27A6,  # HEAVY BLACK CURVED UPWARDS AND RIGHTWARDS ARROW
    "a78": 0x27A7,  # SQUAT BLACK RIGHTWARDS ARROW
    "a79": 0x27A8,  # HEAVY CONCAVE-POINTED BLACK RIGHTWARDS ARROW
    "a81": 0x27A9,  # RIGHT-SHADED WHITE RIGHTWARDS ARROW
    "a82": 0x27AA,  # LEFT-SHADED WHITE RIGHTWARDS ARROW
    "a83": 0x27AB,  # BACK-TILTED SHADOWED WHITE RIGHTWARDS ARROW
    "a84": 0x27AC,  # FRONT-TILTED SHADOWED WHITE RIGHTWARDS ARROW
    "a85": 0x27AD,  # HEAVY LOWER RIGHT-SHADOWED WHITE RIGHTWARDS ARROW
    "a86": 0x27AE,  # HEAVY UPPER RIGHT-SHADOWED WHITE RIGHTWARDS ARROW
    "a87": 0x27AF,  # NOTCHED LOWER RIGHT-SHADOWED WHITE RIGHTWARDS ARROW
    "a88": 0x27B1,  # NOTCHED UPPER RIGHT-SHADOWED WHITE RIGHTWARDS ARROW
    "a89": 0x27B2,  # CIRCLED HEAVY WHITE RIGHTWARDS ARROW
    "a90": 0x27B3,  # WHITE-FEATHERED RIGHTWARDS ARROW
    "a91": 0x27B4,  # BLACK-FEATHERED SOUTH EAST ARROW
    "a92": 0x27B5,  # BLACK-FEATHERED RIGHTWARDS ARROW
    "a93": 0x27B6,  # BLACK-FEATHERED NORTH EAST ARROW
    "a94": 0x27B7,  # HEAVY BLACK-FEATHERED SOUTH EAST ARROW
    "a95": 0x27B8,  # HEAVY BLACK-FEATHERED RIGHTWARDS ARROW
    "a96": 0x27B9,  # HEAVY BLACK-FEATHERED NORTH EAST ARROW
    "a97": 0x27BA,  # TEARDROP-BARBED RIGHTWARDS ARROW
    "a98": 0x27BB,  # HEAVY TEARDROP-SHANKED RIGHTWARDS ARROW
    "a99": 0x27BC,  # WEDGE-TAILED RIGHTWARDS ARROW
    "a100": 0x27BD,  # HEAVY WEDGE-TAILED RIGHTWARDS ARROW
    "a101": 0x27BE,  # OPEN-OUTLINED RIGHTWARDS ARROW
    "a102": 0x279A,  # HEAVY RIGHTWARDS ARROW WITH EQUILATERAL ARROWHEAD
    "a103": 0x27B0,  # CURLY LOOP (placeholder)
    "a104": 0x27BF,  # DOUBLE CURLY LOOP
    "a105": 0x2768,  # MEDIUM LEFT PARENTHESIS ORNAMENT
    "a106": 0x2769,  # MEDIUM RIGHT PARENTHESIS ORNAMENT
    "a107": 0x276A,  # MEDIUM FLATTENED LEFT PARENTHESIS ORNAMENT
    "a108": 0x276B,  # MEDIUM FLATTENED RIGHT PARENTHESIS ORNAMENT
    "a109": 0x276C,  # MEDIUM LEFT-POINTING ANGLE BRACKET ORNAMENT
    "a110": 0x276D,  # MEDIUM RIGHT-POINTING ANGLE BRACKET ORNAMENT
    "a111": 0x276E,  # HEAVY LEFT-POINTING ANGLE QUOTATION MARK ORNAMENT
    "a112": 0x276F,  # HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT
    "a117": 0x2770,  # HEAVY LEFT-POINTING ANGLE BRACKET ORNAMENT
    "a118": 0x2771,  # HEAVY RIGHT-POINTING ANGLE BRACKET ORNAMENT
    "a119": 0x2772,  # LIGHT LEFT TORTOISE SHELL BRACKET ORNAMENT
    "a120": 0x2773,  # LIGHT RIGHT TORTOISE SHELL BRACKET ORNAMENT
    "a121": 0x2774,  # MEDIUM LEFT CURLY BRACKET ORNAMENT
    "a122": 0x2775,  # MEDIUM RIGHT CURLY BRACKET ORNAMENT
    "a123": 0x2761,  # CURVED STEM PARAGRAPH SIGN ORNAMENT (dup check)
    "a124": 0x2022,  # BULLET
    "a125": 0x25CF,  # BLACK CIRCLE
    "a126": 0x274D,  # SHADOWED WHITE CIRCLE
    "a127": 0x25A0,  # BLACK SQUARE
    "a128": 0x274F,  # LOWER RIGHT DROP-SHADOWED WHITE SQUARE
    "a129": 0x2750,  # UPPER RIGHT DROP-SHADOWED WHITE SQUARE
    "a130": 0x2751,  # LOWER RIGHT SHADOWED WHITE SQUARE
    "a131": 0x2752,  # UPPER RIGHT SHADOWED WHITE SQUARE
    "a132": 0x25B2,  # BLACK UP-POINTING TRIANGLE
    "a133": 0x25BC,  # BLACK DOWN-POINTING TRIANGLE
    "a134": 0x25C6,  # BLACK DIAMOND
    "a135": 0x2756,  # BLACK DIAMOND MINUS WHITE X
    "a136": 0x25D7,  # RIGHT HALF BLACK CIRCLE
    "a137": 0x2758,  # LIGHT VERTICAL BAR
    "a138": 0x2759,  # MEDIUM VERTICAL BAR
    "a139": 0x275A,  # HEAVY VERTICAL BAR
    "a140": 0x2762,  # HEAVY EXCLAMATION MARK ORNAMENT (dup check)
    "a141": 0x2767,  # ROTATED FLORAL HEART BULLET (dup check)
    "a142": 0x2639,  # WHITE FROWNING FACE
    "a143": 0x263A,  # WHITE SMILING FACE
    "a144": 0x263B,  # BLACK SMILING FACE
    "a145": 0x2620,  # SKULL AND CROSSBONES
    "a146": 0x2625,  # ANKH
    "a147": 0x262F,  # YIN YANG
    "a148": 0x2638,  # WHEEL OF DHARMA
    "a149": 0x2648,  # ARIES
    "a150": 0x2649,  # TAURUS
    "a151": 0x264A,  # GEMINI
    "a152": 0x264B,  # CANCER
    "a153": 0x264C,  # LEO
    "a154": 0x264D,  # VIRGO
    "a155": 0x264E,  # LIBRA
    "a156": 0x264F,  # SCORPIO
    "a157": 0x2650,  # SAGITTARIUS
    "a158": 0x2651,  # CAPRICORN
    "a159": 0x2652,  # AQUARIUS
    "a160": 0x2653,  # PISCES
    "a161": 0x2660,  # BLACK SPADE SUIT (dup check)
    "a162": 0x2663,  # BLACK CLUB SUIT (dup check)
    "a163": 0x2665,  # BLACK HEART SUIT (dup check)
    "a164": 0x2666,  # BLACK DIAMOND SUIT (dup check)
    "a165": 0x2667,  # WHITE CLUB SUIT
    "a166": 0x2664,  # WHITE SPADE SUIT
    "a167": 0x2661,  # WHITE HEART SUIT
    "a168": 0x2662,  # WHITE DIAMOND SUIT
    "a169": 0x2721,  # STAR OF DAVID
    "a170": 0x261B,  # BLACK RIGHT POINTING INDEX
    "a171": 0x261E,  # WHITE RIGHT POINTING INDEX
    "a172": 0x270C,  # VICTORY HAND
    "a173": 0x270D,  # WRITING HAND
    "a174": 0x270E,  # LOWER RIGHT PENCIL
    "a175": 0x270F,  # PENCIL
    "a176": 0x2710,  # UPPER RIGHT PENCIL
    "a177": 0x2711,  # WHITE NIB
    "a178": 0x2712,  # BLACK NIB
    "a179": 0x2713,  # CHECK MARK
    "a180": 0x2714,  # HEAVY CHECK MARK
    "a181": 0x2715,  # MULTIPLICATION X
    "a182": 0x2716,  # HEAVY MULTIPLICATION X
    "a183": 0x2717,  # BALLOT X
    "a184": 0x2718,  # HEAVY BALLOT X
    "a185": 0x2719,  # OUTLINED GREEK CROSS
    "a186": 0x271A,  # HEAVY GREEK CROSS
    "a187": 0x271B,  # OPEN CENTRE CROSS
    "a188": 0x271C,  # HEAVY OPEN CENTRE CROSS
    "a189": 0x271D,  # LATIN CROSS
    "a190": 0x271E,  # SHADOWED WHITE LATIN CROSS
    "a191": 0x271F,  # OUTLINED LATIN CROSS
    "a192": 0x2720,  # MALTESE CROSS
    "a193": 0x2721,  # STAR OF DAVID (dup check)
    "a194": 0x2722,  # FOUR TEARDROP-SPOKED ASTERISK
    "a195": 0x2723,  # FOUR BALLOON-SPOKED ASTERISK
    "a196": 0x2724,  # HEAVY FOUR BALLOON-SPOKED ASTERISK
    "a197": 0x2725,  # FOUR CLUB-SPOKED ASTERISK
    "a198": 0x2726,  # BLACK FOUR POINTED STAR
    "a199": 0x2727,  # WHITE FOUR POINTED STAR
    "a200": 0x2605,  # BLACK STAR
    "a201": 0x2729,  # STRESS OUTLINED WHITE STAR
    "a202": 0x2703,  # LOWER BLADE SCISSORS
    "a203": 0x272A,  # CIRCLED WHITE STAR
    "a204": 0x272B,  # OPEN CENTRE BLACK STAR
    "a205": 0x272C,  # BLACK CENTRE WHITE STAR
    "a206": 0x272D,  # OUTLINED BLACK STAR
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

    This function tries multiple strategies to find the correct glyph:
    1. Direct lookup in hmtx (font has the Adobe name)
    2. Custom mapping to Unicode, then find glyph via cmap
    3. AGL2UV standard mapping to Unicode, then find glyph via cmap

    Args:
        adobe_name: The Adobe glyph name to resolve (e.g., 'a1', 'Alpha').
        cmap: The font's character map (codepoint -> glyph name).
        hmtx_metrics: The font's horizontal metrics (glyph name -> (width, lsb)).
        custom_mapping: Optional custom mapping dict (glyph name -> Unicode).

    Returns:
        The actual glyph name in the font, or None if not found.
    """
    # Strategy 1: Direct lookup - font uses Adobe glyph names
    if adobe_name in hmtx_metrics:
        return adobe_name

    # Strategy 2: Custom mapping (for ZapfDingbats a1-a206, Symbol exceptions)
    if custom_mapping and adobe_name in custom_mapping:
        unicode_val = custom_mapping[adobe_name]
        if unicode_val is None:
            # Construction glyph with no Unicode equivalent
            return None
        # Look up the Unicode codepoint in cmap
        glyph_name = cmap.get(unicode_val)
        if glyph_name and glyph_name in hmtx_metrics:
            return glyph_name

    # Strategy 3: Standard AGL2UV mapping
    if adobe_name in AGL2UV:
        unicode_val = AGL2UV[adobe_name]
        glyph_name = cmap.get(unicode_val)
        if glyph_name and glyph_name in hmtx_metrics:
            return glyph_name

    # Not found
    return None
