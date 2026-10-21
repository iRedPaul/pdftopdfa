# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared helper functions for font-related tests."""


def _liberation_fonts_available() -> bool:
    """Checks if Liberation fonts are available."""
    try:
        from importlib import resources

        font_ref = (
            resources.files("pdftopdfa")
            / "resources"
            / "fonts"
            / "LiberationSans-Regular.ttf"
        )
        font_ref.read_bytes()
        return True
    except Exception:
        return False


def _noto_cjk_font_available() -> bool:
    """Checks if Noto Sans CJK Font is available."""
    try:
        from importlib import resources

        font_ref = (
            resources.files("pdftopdfa")
            / "resources"
            / "fonts"
            / "NotoSansCJK-Regular.ttc"
        )
        font_ref.read_bytes()
        return True
    except Exception:
        return False


def make_mac_roman_only_truetype(mac_codes: dict[int, str]) -> bytes:
    """Build a TrueType font whose glyphs are reachable only via (1,0) cmap.

    Mirrors legacy Mac fonts: glyphs carry ``uniF0xx`` names, a Symbol
    (3,0) subtable maps the F0xx range, there is no (3,1) subtable, and
    the post table stores no glyph names.
    """
    import io

    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._c_m_a_p import (
        cmap_format_0,
        cmap_format_4,
        table__c_m_a_p,
    )
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    names = [".notdef", *mac_codes.values()]
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({})
    builder.setupGlyf({name: Glyph() for name in names})
    builder.setupHorizontalMetrics({name: (500, 0) for name in names})
    builder.setupHorizontalHeader()
    builder.setupNameTable({"familyName": "MacOnly", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost(keepGlyphNames=False)
    builder.setupHead(unitsPerEm=1000)

    mac = cmap_format_0(0)
    mac.platformID, mac.platEncID, mac.language = 1, 0, 0
    mac.cmap = {code: mac_codes.get(code, ".notdef") for code in range(256)}
    symbol = cmap_format_4(4)
    symbol.platformID, symbol.platEncID, symbol.language = 3, 0, 0
    symbol.cmap = {0xF000 + code: name for code, name in mac_codes.items()}
    table = table__c_m_a_p()
    table.tableVersion = 0
    table.tables = [mac, symbol]
    builder.font["cmap"] = table

    output = io.BytesIO()
    builder.font.save(output)
    return output.getvalue()
