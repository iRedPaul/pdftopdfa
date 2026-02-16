# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font metrics for PDF form field appearance stream generation.

Provides glyph width tables for the 14 Standard PDF fonts (from Adobe AFM
data) and helpers to read metrics from embedded font programs.  All widths
are in the standard PDF unit of 1/1000 of the font size.
"""

from __future__ import annotations

import logging
from io import BytesIO

from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WinAnsiEncoding <-> Unicode mapping (positions 128-159 differ from Latin-1)
# ---------------------------------------------------------------------------

_WIN_ANSI_TO_UNICODE: dict[int, int] = {
    128: 0x20AC,
    130: 0x201A,
    131: 0x0192,
    132: 0x201E,
    133: 0x2026,
    134: 0x2020,
    135: 0x2021,
    136: 0x02C6,
    137: 0x2030,
    138: 0x0160,
    139: 0x2039,
    140: 0x0152,
    142: 0x017D,
    145: 0x2018,
    146: 0x2019,
    147: 0x201C,
    148: 0x201D,
    149: 0x2022,
    150: 0x2013,
    151: 0x2014,
    152: 0x02DC,
    153: 0x2122,
    154: 0x0161,
    155: 0x203A,
    156: 0x0153,
    158: 0x017E,
    159: 0x0178,
}

# Positions 0-127 and 160-255 map to their Unicode code point directly.
_UNICODE_TO_WIN_ANSI: dict[int, int] = {v: k for k, v in _WIN_ANSI_TO_UNICODE.items()}


def _winansi_to_unicode(code: int) -> int:
    """Map a WinAnsiEncoding byte to a Unicode code point."""
    return _WIN_ANSI_TO_UNICODE.get(code, code)


def _unicode_to_winansi(cp: int) -> int | None:
    """Map a Unicode code point to WinAnsiEncoding byte (None if unmappable)."""
    if cp < 128 or 160 <= cp <= 255:
        return cp
    return _UNICODE_TO_WIN_ANSI.get(cp)


# ===================================================================
# Standard-14 font glyph widths (WinAnsiEncoding char code -> width)
# ===================================================================
# Source: Adobe Font Metrics (AFM) files.  Widths in 1/1000 of font size.
# Only codes that map to a glyph in WinAnsiEncoding are present;
# missing codes fall back to the font's default width.

_HELVETICA_WIDTHS: dict[int, int] = {
    32: 278,
    33: 278,
    34: 355,
    35: 556,
    36: 556,
    37: 889,
    38: 667,
    39: 191,
    40: 333,
    41: 333,
    42: 389,
    43: 584,
    44: 278,
    45: 333,
    46: 278,
    47: 278,
    48: 556,
    49: 556,
    50: 556,
    51: 556,
    52: 556,
    53: 556,
    54: 556,
    55: 556,
    56: 556,
    57: 556,
    58: 278,
    59: 278,
    60: 584,
    61: 584,
    62: 584,
    63: 556,
    64: 1015,
    65: 667,
    66: 667,
    67: 722,
    68: 722,
    69: 611,
    70: 556,
    71: 778,
    72: 722,
    73: 278,
    74: 500,
    75: 667,
    76: 556,
    77: 833,
    78: 722,
    79: 778,
    80: 667,
    81: 778,
    82: 722,
    83: 667,
    84: 611,
    85: 722,
    86: 667,
    87: 944,
    88: 667,
    89: 667,
    90: 611,
    91: 278,
    92: 278,
    93: 278,
    94: 469,
    95: 556,
    96: 333,
    97: 556,
    98: 556,
    99: 500,
    100: 556,
    101: 556,
    102: 278,
    103: 556,
    104: 556,
    105: 222,
    106: 222,
    107: 500,
    108: 222,
    109: 833,
    110: 556,
    111: 556,
    112: 556,
    113: 556,
    114: 333,
    115: 500,
    116: 278,
    117: 556,
    118: 500,
    119: 722,
    120: 500,
    121: 500,
    122: 500,
    123: 334,
    124: 260,
    125: 334,
    126: 584,
    128: 556,
    130: 222,
    131: 556,
    132: 333,
    133: 1000,
    134: 556,
    135: 556,
    136: 333,
    137: 1000,
    138: 667,
    139: 333,
    140: 1000,
    142: 611,
    145: 222,
    146: 222,
    147: 333,
    148: 333,
    149: 350,
    150: 556,
    151: 1000,
    152: 333,
    153: 1000,
    154: 500,
    155: 333,
    156: 944,
    158: 500,
    159: 667,
    160: 278,
    161: 333,
    162: 556,
    163: 556,
    164: 556,
    165: 556,
    166: 260,
    167: 556,
    168: 333,
    169: 737,
    170: 370,
    171: 556,
    172: 584,
    173: 333,
    174: 737,
    175: 333,
    176: 400,
    177: 584,
    178: 333,
    179: 333,
    180: 333,
    181: 556,
    182: 537,
    183: 278,
    184: 333,
    185: 333,
    186: 365,
    187: 556,
    188: 834,
    189: 834,
    190: 834,
    191: 611,
    192: 667,
    193: 667,
    194: 667,
    195: 667,
    196: 667,
    197: 667,
    198: 1000,
    199: 722,
    200: 611,
    201: 611,
    202: 611,
    203: 611,
    204: 278,
    205: 278,
    206: 278,
    207: 278,
    208: 722,
    209: 722,
    210: 778,
    211: 778,
    212: 778,
    213: 778,
    214: 778,
    215: 584,
    216: 778,
    217: 722,
    218: 722,
    219: 722,
    220: 722,
    221: 667,
    222: 667,
    223: 611,
    224: 556,
    225: 556,
    226: 556,
    227: 556,
    228: 556,
    229: 556,
    230: 889,
    231: 500,
    232: 556,
    233: 556,
    234: 556,
    235: 556,
    236: 278,
    237: 278,
    238: 278,
    239: 278,
    240: 556,
    241: 556,
    242: 556,
    243: 556,
    244: 556,
    245: 556,
    246: 556,
    247: 584,
    248: 611,
    249: 556,
    250: 556,
    251: 556,
    252: 556,
    253: 500,
    254: 556,
    255: 500,
}

_HELVETICA_BOLD_WIDTHS: dict[int, int] = {
    32: 278,
    33: 333,
    34: 474,
    35: 556,
    36: 556,
    37: 889,
    38: 722,
    39: 238,
    40: 333,
    41: 333,
    42: 389,
    43: 584,
    44: 278,
    45: 333,
    46: 278,
    47: 278,
    48: 556,
    49: 556,
    50: 556,
    51: 556,
    52: 556,
    53: 556,
    54: 556,
    55: 556,
    56: 556,
    57: 556,
    58: 333,
    59: 333,
    60: 584,
    61: 584,
    62: 584,
    63: 611,
    64: 975,
    65: 722,
    66: 722,
    67: 722,
    68: 722,
    69: 667,
    70: 611,
    71: 778,
    72: 722,
    73: 278,
    74: 556,
    75: 722,
    76: 611,
    77: 833,
    78: 722,
    79: 778,
    80: 667,
    81: 778,
    82: 722,
    83: 667,
    84: 611,
    85: 722,
    86: 667,
    87: 944,
    88: 667,
    89: 667,
    90: 611,
    91: 333,
    92: 278,
    93: 333,
    94: 584,
    95: 556,
    96: 333,
    97: 556,
    98: 611,
    99: 556,
    100: 611,
    101: 556,
    102: 333,
    103: 611,
    104: 611,
    105: 278,
    106: 278,
    107: 556,
    108: 278,
    109: 889,
    110: 611,
    111: 611,
    112: 611,
    113: 611,
    114: 389,
    115: 556,
    116: 333,
    117: 611,
    118: 556,
    119: 778,
    120: 556,
    121: 556,
    122: 500,
    123: 389,
    124: 280,
    125: 389,
    126: 584,
    128: 556,
    130: 278,
    131: 556,
    132: 500,
    133: 1000,
    134: 556,
    135: 556,
    136: 333,
    137: 1000,
    138: 667,
    139: 333,
    140: 1000,
    142: 611,
    145: 278,
    146: 278,
    147: 500,
    148: 500,
    149: 350,
    150: 556,
    151: 1000,
    152: 333,
    153: 1000,
    154: 556,
    155: 333,
    156: 944,
    158: 500,
    159: 667,
    160: 278,
    161: 333,
    162: 556,
    163: 556,
    164: 556,
    165: 556,
    166: 280,
    167: 556,
    168: 333,
    169: 737,
    170: 370,
    171: 556,
    172: 584,
    173: 333,
    174: 737,
    175: 333,
    176: 400,
    177: 584,
    178: 333,
    179: 333,
    180: 333,
    181: 611,
    182: 556,
    183: 278,
    184: 333,
    185: 333,
    186: 365,
    187: 556,
    188: 834,
    189: 834,
    190: 834,
    191: 611,
    192: 722,
    193: 722,
    194: 722,
    195: 722,
    196: 722,
    197: 722,
    198: 1000,
    199: 722,
    200: 667,
    201: 667,
    202: 667,
    203: 667,
    204: 278,
    205: 278,
    206: 278,
    207: 278,
    208: 722,
    209: 722,
    210: 778,
    211: 778,
    212: 778,
    213: 778,
    214: 778,
    215: 584,
    216: 778,
    217: 722,
    218: 722,
    219: 722,
    220: 722,
    221: 667,
    222: 667,
    223: 611,
    224: 556,
    225: 556,
    226: 556,
    227: 556,
    228: 556,
    229: 556,
    230: 889,
    231: 556,
    232: 556,
    233: 556,
    234: 556,
    235: 556,
    236: 278,
    237: 278,
    238: 278,
    239: 278,
    240: 611,
    241: 611,
    242: 611,
    243: 611,
    244: 611,
    245: 611,
    246: 611,
    247: 584,
    248: 611,
    249: 611,
    250: 611,
    251: 611,
    252: 611,
    253: 556,
    254: 611,
    255: 556,
}

_TIMES_ROMAN_WIDTHS: dict[int, int] = {
    32: 250,
    33: 333,
    34: 408,
    35: 500,
    36: 500,
    37: 833,
    38: 778,
    39: 180,
    40: 333,
    41: 333,
    42: 500,
    43: 564,
    44: 250,
    45: 333,
    46: 250,
    47: 278,
    48: 500,
    49: 500,
    50: 500,
    51: 500,
    52: 500,
    53: 500,
    54: 500,
    55: 500,
    56: 500,
    57: 500,
    58: 278,
    59: 278,
    60: 564,
    61: 564,
    62: 564,
    63: 444,
    64: 921,
    65: 722,
    66: 667,
    67: 667,
    68: 722,
    69: 611,
    70: 556,
    71: 722,
    72: 722,
    73: 333,
    74: 389,
    75: 722,
    76: 611,
    77: 889,
    78: 722,
    79: 722,
    80: 556,
    81: 722,
    82: 667,
    83: 556,
    84: 611,
    85: 722,
    86: 722,
    87: 944,
    88: 722,
    89: 722,
    90: 611,
    91: 333,
    92: 278,
    93: 333,
    94: 469,
    95: 500,
    96: 333,
    97: 444,
    98: 500,
    99: 444,
    100: 500,
    101: 444,
    102: 333,
    103: 500,
    104: 500,
    105: 278,
    106: 278,
    107: 500,
    108: 278,
    109: 778,
    110: 500,
    111: 500,
    112: 500,
    113: 500,
    114: 333,
    115: 389,
    116: 278,
    117: 500,
    118: 500,
    119: 722,
    120: 500,
    121: 500,
    122: 444,
    123: 480,
    124: 200,
    125: 480,
    126: 541,
    128: 500,
    130: 333,
    131: 500,
    132: 444,
    133: 1000,
    134: 500,
    135: 500,
    136: 333,
    137: 1000,
    138: 556,
    139: 333,
    140: 889,
    142: 611,
    145: 333,
    146: 333,
    147: 444,
    148: 444,
    149: 350,
    150: 500,
    151: 1000,
    152: 333,
    153: 980,
    154: 389,
    155: 333,
    156: 722,
    158: 444,
    159: 722,
    160: 250,
    161: 333,
    162: 500,
    163: 500,
    164: 500,
    165: 500,
    166: 200,
    167: 500,
    168: 333,
    169: 760,
    170: 276,
    171: 500,
    172: 564,
    173: 333,
    174: 760,
    175: 333,
    176: 400,
    177: 564,
    178: 300,
    179: 300,
    180: 333,
    181: 500,
    182: 453,
    183: 250,
    184: 333,
    185: 300,
    186: 310,
    187: 500,
    188: 750,
    189: 750,
    190: 750,
    191: 444,
    192: 722,
    193: 722,
    194: 722,
    195: 722,
    196: 722,
    197: 722,
    198: 889,
    199: 667,
    200: 611,
    201: 611,
    202: 611,
    203: 611,
    204: 333,
    205: 333,
    206: 333,
    207: 333,
    208: 722,
    209: 722,
    210: 722,
    211: 722,
    212: 722,
    213: 722,
    214: 722,
    215: 564,
    216: 722,
    217: 722,
    218: 722,
    219: 722,
    220: 722,
    221: 722,
    222: 556,
    223: 500,
    224: 444,
    225: 444,
    226: 444,
    227: 444,
    228: 444,
    229: 444,
    230: 667,
    231: 444,
    232: 444,
    233: 444,
    234: 444,
    235: 444,
    236: 278,
    237: 278,
    238: 278,
    239: 278,
    240: 500,
    241: 500,
    242: 500,
    243: 500,
    244: 500,
    245: 500,
    246: 500,
    247: 564,
    248: 500,
    249: 500,
    250: 500,
    251: 500,
    252: 500,
    253: 500,
    254: 500,
    255: 500,
}

_TIMES_BOLD_WIDTHS: dict[int, int] = {
    32: 250,
    33: 333,
    34: 555,
    35: 500,
    36: 500,
    37: 1000,
    38: 833,
    39: 278,
    40: 333,
    41: 333,
    42: 500,
    43: 570,
    44: 250,
    45: 333,
    46: 250,
    47: 278,
    48: 500,
    49: 500,
    50: 500,
    51: 500,
    52: 500,
    53: 500,
    54: 500,
    55: 500,
    56: 500,
    57: 500,
    58: 333,
    59: 333,
    60: 570,
    61: 570,
    62: 570,
    63: 500,
    64: 930,
    65: 722,
    66: 667,
    67: 722,
    68: 722,
    69: 667,
    70: 611,
    71: 778,
    72: 778,
    73: 389,
    74: 500,
    75: 778,
    76: 667,
    77: 944,
    78: 722,
    79: 778,
    80: 611,
    81: 778,
    82: 722,
    83: 556,
    84: 667,
    85: 722,
    86: 722,
    87: 1000,
    88: 722,
    89: 722,
    90: 667,
    91: 333,
    92: 278,
    93: 333,
    94: 581,
    95: 500,
    96: 333,
    97: 500,
    98: 556,
    99: 444,
    100: 556,
    101: 444,
    102: 333,
    103: 500,
    104: 556,
    105: 278,
    106: 333,
    107: 556,
    108: 278,
    109: 833,
    110: 556,
    111: 500,
    112: 556,
    113: 556,
    114: 444,
    115: 389,
    116: 333,
    117: 556,
    118: 500,
    119: 722,
    120: 500,
    121: 500,
    122: 444,
    123: 394,
    124: 220,
    125: 394,
    126: 520,
    128: 500,
    130: 333,
    131: 500,
    132: 500,
    133: 1000,
    134: 500,
    135: 500,
    136: 333,
    137: 1000,
    138: 556,
    139: 333,
    140: 1000,
    142: 667,
    145: 333,
    146: 333,
    147: 500,
    148: 500,
    149: 350,
    150: 500,
    151: 1000,
    152: 333,
    153: 1000,
    154: 389,
    155: 333,
    156: 722,
    158: 444,
    159: 722,
    160: 250,
    161: 333,
    162: 500,
    163: 500,
    164: 500,
    165: 500,
    166: 220,
    167: 500,
    168: 333,
    169: 747,
    170: 300,
    171: 500,
    172: 570,
    173: 333,
    174: 747,
    175: 333,
    176: 400,
    177: 570,
    178: 300,
    179: 300,
    180: 333,
    181: 556,
    182: 540,
    183: 250,
    184: 333,
    185: 300,
    186: 330,
    187: 500,
    188: 750,
    189: 750,
    190: 750,
    191: 500,
    192: 722,
    193: 722,
    194: 722,
    195: 722,
    196: 722,
    197: 722,
    198: 1000,
    199: 722,
    200: 667,
    201: 667,
    202: 667,
    203: 667,
    204: 389,
    205: 389,
    206: 389,
    207: 389,
    208: 722,
    209: 722,
    210: 778,
    211: 778,
    212: 778,
    213: 778,
    214: 778,
    215: 570,
    216: 778,
    217: 722,
    218: 722,
    219: 722,
    220: 722,
    221: 722,
    222: 611,
    223: 556,
    224: 500,
    225: 500,
    226: 500,
    227: 500,
    228: 500,
    229: 500,
    230: 722,
    231: 444,
    232: 444,
    233: 444,
    234: 444,
    235: 444,
    236: 278,
    237: 278,
    238: 278,
    239: 278,
    240: 500,
    241: 556,
    242: 500,
    243: 500,
    244: 500,
    245: 500,
    246: 500,
    247: 570,
    248: 500,
    249: 556,
    250: 556,
    251: 556,
    252: 556,
    253: 500,
    254: 556,
    255: 500,
}

_TIMES_ITALIC_WIDTHS: dict[int, int] = {
    32: 250,
    33: 333,
    34: 420,
    35: 500,
    36: 500,
    37: 833,
    38: 778,
    39: 214,
    40: 333,
    41: 333,
    42: 500,
    43: 675,
    44: 250,
    45: 333,
    46: 250,
    47: 278,
    48: 500,
    49: 500,
    50: 500,
    51: 500,
    52: 500,
    53: 500,
    54: 500,
    55: 500,
    56: 500,
    57: 500,
    58: 333,
    59: 333,
    60: 675,
    61: 675,
    62: 675,
    63: 500,
    64: 920,
    65: 611,
    66: 611,
    67: 667,
    68: 722,
    69: 611,
    70: 611,
    71: 722,
    72: 722,
    73: 333,
    74: 444,
    75: 667,
    76: 556,
    77: 833,
    78: 667,
    79: 722,
    80: 611,
    81: 722,
    82: 611,
    83: 500,
    84: 556,
    85: 722,
    86: 611,
    87: 833,
    88: 611,
    89: 556,
    90: 556,
    91: 389,
    92: 278,
    93: 389,
    94: 422,
    95: 500,
    96: 333,
    97: 500,
    98: 500,
    99: 444,
    100: 500,
    101: 444,
    102: 278,
    103: 500,
    104: 500,
    105: 278,
    106: 278,
    107: 444,
    108: 278,
    109: 722,
    110: 500,
    111: 500,
    112: 500,
    113: 500,
    114: 389,
    115: 389,
    116: 278,
    117: 500,
    118: 444,
    119: 667,
    120: 444,
    121: 444,
    122: 389,
    123: 400,
    124: 275,
    125: 400,
    126: 541,
    128: 500,
    130: 333,
    131: 500,
    132: 556,
    133: 889,
    134: 500,
    135: 500,
    136: 333,
    137: 1000,
    138: 500,
    139: 333,
    140: 944,
    142: 556,
    145: 333,
    146: 333,
    147: 556,
    148: 556,
    149: 350,
    150: 500,
    151: 889,
    152: 333,
    153: 980,
    154: 389,
    155: 333,
    156: 722,
    158: 389,
    159: 556,
    160: 250,
    161: 389,
    162: 500,
    163: 500,
    164: 500,
    165: 500,
    166: 275,
    167: 500,
    168: 333,
    169: 760,
    170: 276,
    171: 500,
    172: 675,
    173: 333,
    174: 760,
    175: 333,
    176: 400,
    177: 675,
    178: 300,
    179: 300,
    180: 333,
    181: 500,
    182: 523,
    183: 250,
    184: 333,
    185: 300,
    186: 310,
    187: 500,
    188: 750,
    189: 750,
    190: 750,
    191: 500,
    192: 611,
    193: 611,
    194: 611,
    195: 611,
    196: 611,
    197: 611,
    198: 889,
    199: 667,
    200: 611,
    201: 611,
    202: 611,
    203: 611,
    204: 333,
    205: 333,
    206: 333,
    207: 333,
    208: 722,
    209: 667,
    210: 722,
    211: 722,
    212: 722,
    213: 722,
    214: 722,
    215: 675,
    216: 722,
    217: 722,
    218: 722,
    219: 722,
    220: 722,
    221: 556,
    222: 611,
    223: 500,
    224: 500,
    225: 500,
    226: 500,
    227: 500,
    228: 500,
    229: 500,
    230: 667,
    231: 444,
    232: 444,
    233: 444,
    234: 444,
    235: 444,
    236: 278,
    237: 278,
    238: 278,
    239: 278,
    240: 500,
    241: 500,
    242: 500,
    243: 500,
    244: 500,
    245: 500,
    246: 500,
    247: 675,
    248: 500,
    249: 500,
    250: 500,
    251: 500,
    252: 500,
    253: 444,
    254: 500,
    255: 444,
}

_TIMES_BOLD_ITALIC_WIDTHS: dict[int, int] = {
    32: 250,
    33: 389,
    34: 555,
    35: 500,
    36: 500,
    37: 833,
    38: 778,
    39: 278,
    40: 333,
    41: 333,
    42: 500,
    43: 570,
    44: 250,
    45: 333,
    46: 250,
    47: 278,
    48: 500,
    49: 500,
    50: 500,
    51: 500,
    52: 500,
    53: 500,
    54: 500,
    55: 500,
    56: 500,
    57: 500,
    58: 333,
    59: 333,
    60: 570,
    61: 570,
    62: 570,
    63: 500,
    64: 832,
    65: 667,
    66: 667,
    67: 667,
    68: 722,
    69: 667,
    70: 667,
    71: 722,
    72: 778,
    73: 389,
    74: 500,
    75: 667,
    76: 611,
    77: 889,
    78: 722,
    79: 722,
    80: 611,
    81: 722,
    82: 667,
    83: 556,
    84: 611,
    85: 722,
    86: 667,
    87: 889,
    88: 667,
    89: 611,
    90: 611,
    91: 333,
    92: 278,
    93: 333,
    94: 570,
    95: 500,
    96: 333,
    97: 500,
    98: 500,
    99: 444,
    100: 500,
    101: 444,
    102: 333,
    103: 500,
    104: 556,
    105: 278,
    106: 278,
    107: 500,
    108: 278,
    109: 778,
    110: 556,
    111: 500,
    112: 556,
    113: 500,
    114: 389,
    115: 389,
    116: 278,
    117: 556,
    118: 444,
    119: 667,
    120: 500,
    121: 444,
    122: 389,
    123: 348,
    124: 220,
    125: 348,
    126: 570,
    128: 500,
    130: 333,
    131: 500,
    132: 500,
    133: 1000,
    134: 500,
    135: 500,
    136: 333,
    137: 1000,
    138: 556,
    139: 333,
    140: 944,
    142: 611,
    145: 333,
    146: 333,
    147: 500,
    148: 500,
    149: 350,
    150: 500,
    151: 1000,
    152: 333,
    153: 1000,
    154: 389,
    155: 333,
    156: 722,
    158: 389,
    159: 611,
    160: 250,
    161: 389,
    162: 500,
    163: 500,
    164: 500,
    165: 500,
    166: 220,
    167: 500,
    168: 333,
    169: 747,
    170: 266,
    171: 500,
    172: 606,
    173: 333,
    174: 747,
    175: 333,
    176: 400,
    177: 570,
    178: 300,
    179: 300,
    180: 333,
    181: 576,
    182: 500,
    183: 250,
    184: 333,
    185: 300,
    186: 300,
    187: 500,
    188: 750,
    189: 750,
    190: 750,
    191: 500,
    192: 667,
    193: 667,
    194: 667,
    195: 667,
    196: 667,
    197: 667,
    198: 944,
    199: 667,
    200: 667,
    201: 667,
    202: 667,
    203: 667,
    204: 389,
    205: 389,
    206: 389,
    207: 389,
    208: 722,
    209: 722,
    210: 722,
    211: 722,
    212: 722,
    213: 722,
    214: 722,
    215: 570,
    216: 722,
    217: 722,
    218: 722,
    219: 722,
    220: 722,
    221: 611,
    222: 611,
    223: 500,
    224: 500,
    225: 500,
    226: 500,
    227: 500,
    228: 500,
    229: 500,
    230: 722,
    231: 444,
    232: 444,
    233: 444,
    234: 444,
    235: 444,
    236: 278,
    237: 278,
    238: 278,
    239: 278,
    240: 500,
    241: 556,
    242: 500,
    243: 500,
    244: 500,
    245: 500,
    246: 500,
    247: 570,
    248: 500,
    249: 556,
    250: 556,
    251: 556,
    252: 556,
    253: 444,
    254: 500,
    255: 444,
}

# Courier family: all glyphs are 600 units wide (monospaced).
_COURIER_WIDTHS: dict[int, int] = {
    i: 600 for i in range(32, 256) if i not in (127, 129, 141, 143, 144, 157)
}

# Symbol font (Symbol encoding, not WinAnsiEncoding).
# Minimal table for the rare case it appears in a form field.
_SYMBOL_WIDTHS: dict[int, int] = {
    32: 250,
    33: 333,
    34: 713,
    35: 500,
    36: 549,
    37: 833,
    38: 778,
    39: 439,
    40: 333,
    41: 333,
    42: 500,
    43: 549,
    44: 250,
    45: 549,
    46: 250,
    47: 278,
    48: 500,
    49: 500,
    50: 500,
    51: 500,
    52: 500,
    53: 500,
    54: 500,
    55: 500,
    56: 500,
    57: 500,
    58: 278,
    59: 278,
    60: 549,
    61: 549,
    62: 549,
    63: 444,
    65: 722,
    66: 667,
    67: 722,
    68: 612,
    69: 611,
    70: 763,
    71: 603,
    72: 722,
    73: 333,
    74: 631,
    75: 722,
    76: 686,
    77: 889,
    78: 722,
    79: 722,
    80: 768,
    81: 741,
    82: 556,
    83: 592,
    84: 611,
    85: 690,
    86: 439,
    87: 768,
    88: 645,
    89: 795,
    90: 611,
    97: 611,
    98: 611,
    99: 549,
    100: 611,
    101: 549,
    102: 611,
    103: 556,
    104: 603,
    105: 329,
    106: 603,
    107: 549,
    108: 549,
    109: 576,
    110: 521,
    111: 549,
    112: 549,
    113: 521,
    114: 549,
    115: 603,
    116: 439,
    117: 576,
    118: 713,
    119: 686,
    120: 493,
    121: 686,
    122: 494,
}

# ZapfDingbats: rarely used in form fields, provide basic table.
_ZAPFDINGBATS_WIDTHS: dict[int, int] = {
    32: 278,
    33: 974,
    34: 961,
    35: 974,
    36: 980,
    37: 719,
    38: 789,
    39: 790,
    40: 791,
    41: 690,
    42: 960,
    43: 939,
    44: 549,
    45: 855,
    46: 911,
    47: 933,
    48: 911,
    49: 945,
    50: 974,
    51: 755,
    52: 846,
    53: 762,
    54: 761,
    55: 571,
    56: 677,
    57: 763,
    58: 760,
    59: 759,
    60: 754,
    61: 494,
    62: 552,
    63: 537,
    64: 577,
    65: 692,
    66: 786,
    67: 788,
    68: 788,
    69: 790,
    70: 793,
    71: 794,
    72: 816,
    73: 823,
    74: 789,
    75: 841,
    76: 823,
    77: 833,
    78: 816,
    79: 831,
    80: 923,
    81: 744,
    82: 723,
    83: 749,
    84: 790,
    85: 792,
    86: 695,
    87: 776,
    88: 768,
    89: 792,
    90: 759,
    91: 707,
    92: 708,
    93: 682,
    94: 701,
    95: 826,
    96: 815,
    97: 789,
    98: 789,
    99: 707,
    100: 687,
    101: 696,
    102: 689,
    103: 786,
    104: 787,
    105: 713,
    106: 791,
    107: 785,
    108: 791,
    109: 873,
    110: 761,
    111: 762,
    112: 762,
    113: 759,
    114: 759,
    115: 892,
    116: 892,
    117: 788,
    118: 784,
    119: 438,
    120: 138,
    121: 277,
    122: 415,
}

# ---------------------------------------------------------------------------
# Standard-14 font metadata
# ---------------------------------------------------------------------------

_STANDARD_14: dict[str, dict] = {
    "Helvetica": {
        "widths": _HELVETICA_WIDTHS,
        "ascent": 718,
        "descent": -207,
        "cap_height": 718,
        "bbox": (-166, -225, 1000, 931),
        "default_width": 278,
    },
    "Helvetica-Bold": {
        "widths": _HELVETICA_BOLD_WIDTHS,
        "ascent": 718,
        "descent": -207,
        "cap_height": 718,
        "bbox": (-170, -228, 1003, 962),
        "default_width": 278,
    },
    "Helvetica-Oblique": {
        "widths": _HELVETICA_WIDTHS,  # same as Helvetica
        "ascent": 718,
        "descent": -207,
        "cap_height": 718,
        "bbox": (-170, -225, 1116, 931),
        "default_width": 278,
    },
    "Helvetica-BoldOblique": {
        "widths": _HELVETICA_BOLD_WIDTHS,  # same as Helvetica-Bold
        "ascent": 718,
        "descent": -207,
        "cap_height": 718,
        "bbox": (-174, -228, 1114, 962),
        "default_width": 278,
    },
    "Times-Roman": {
        "widths": _TIMES_ROMAN_WIDTHS,
        "ascent": 683,
        "descent": -217,
        "cap_height": 662,
        "bbox": (-168, -218, 1000, 898),
        "default_width": 250,
    },
    "Times-Bold": {
        "widths": _TIMES_BOLD_WIDTHS,
        "ascent": 683,
        "descent": -217,
        "cap_height": 676,
        "bbox": (-168, -218, 1000, 935),
        "default_width": 250,
    },
    "Times-Italic": {
        "widths": _TIMES_ITALIC_WIDTHS,
        "ascent": 683,
        "descent": -217,
        "cap_height": 653,
        "bbox": (-169, -217, 1010, 883),
        "default_width": 250,
    },
    "Times-BoldItalic": {
        "widths": _TIMES_BOLD_ITALIC_WIDTHS,
        "ascent": 683,
        "descent": -217,
        "cap_height": 669,
        "bbox": (-200, -218, 996, 921),
        "default_width": 250,
    },
    "Courier": {
        "widths": _COURIER_WIDTHS,
        "ascent": 629,
        "descent": -157,
        "cap_height": 562,
        "bbox": (-23, -250, 715, 805),
        "default_width": 600,
    },
    "Courier-Bold": {
        "widths": _COURIER_WIDTHS,
        "ascent": 629,
        "descent": -157,
        "cap_height": 562,
        "bbox": (-113, -250, 749, 801),
        "default_width": 600,
    },
    "Courier-Oblique": {
        "widths": _COURIER_WIDTHS,
        "ascent": 629,
        "descent": -157,
        "cap_height": 562,
        "bbox": (-27, -250, 849, 805),
        "default_width": 600,
    },
    "Courier-BoldOblique": {
        "widths": _COURIER_WIDTHS,
        "ascent": 629,
        "descent": -157,
        "cap_height": 562,
        "bbox": (-57, -250, 869, 801),
        "default_width": 600,
    },
    "Symbol": {
        "widths": _SYMBOL_WIDTHS,
        "ascent": 800,
        "descent": -200,
        "cap_height": 700,
        "bbox": (-180, -293, 1090, 1010),
        "default_width": 500,
    },
    "ZapfDingbats": {
        "widths": _ZAPFDINGBATS_WIDTHS,
        "ascent": 800,
        "descent": -200,
        "cap_height": 700,
        "bbox": (-1, -143, 981, 820),
        "default_width": 278,
    },
}

# Common aliases used in PDF form fields
_FONT_ALIASES: dict[str, str] = {
    "Helv": "Helvetica",
    "HeBo": "Helvetica-Bold",
    "HeOb": "Helvetica-Oblique",
    "HeBO": "Helvetica-BoldOblique",
    "TiRo": "Times-Roman",
    "TiBo": "Times-Bold",
    "TiIt": "Times-Italic",
    "TiBI": "Times-BoldItalic",
    "Cour": "Courier",
    "CoBo": "Courier-Bold",
    "CoOb": "Courier-Oblique",
    "CoBO": "Courier-BoldOblique",
    "Symb": "Symbol",
    "ZaDb": "ZapfDingbats",
    # Common substitution fonts
    "Arial": "Helvetica",
    "ArialMT": "Helvetica",
    "Arial,Bold": "Helvetica-Bold",
    "Arial,Italic": "Helvetica-Oblique",
    "Arial,BoldItalic": "Helvetica-BoldOblique",
    "Arial-BoldMT": "Helvetica-Bold",
    "Arial-ItalicMT": "Helvetica-Oblique",
    "Arial-BoldItalicMT": "Helvetica-BoldOblique",
    "TimesNewRoman": "Times-Roman",
    "TimesNewRomanPSMT": "Times-Roman",
    "TimesNewRomanPS-BoldMT": "Times-Bold",
    "TimesNewRomanPS-ItalicMT": "Times-Italic",
    "TimesNewRomanPS-BoldItalicMT": "Times-BoldItalic",
    "CourierNew": "Courier",
    "CourierNewPSMT": "Courier",
    "CourierNew-Bold": "Courier-Bold",
    "CourierNew-BoldItalic": "Courier-BoldOblique",
}


# ---------------------------------------------------------------------------
# Internal helpers — font metric extraction
# ---------------------------------------------------------------------------


def _get_standard14_metrics(font_name: str) -> dict | None:
    """Look up Standard-14 font metrics by name, checking aliases."""
    if font_name in _STANDARD_14:
        return _STANDARD_14[font_name]
    canonical = _FONT_ALIASES.get(font_name)
    if canonical and canonical in _STANDARD_14:
        return _STANDARD_14[canonical]
    return None


def _base_font_name(font_dict) -> str | None:
    """Extract the /BaseFont name from a font dictionary."""
    if font_dict is None:
        return None
    try:
        bf = font_dict.get("/BaseFont")
        if bf is not None:
            name = str(bf)
            return name.lstrip("/")
    except Exception:
        pass
    return None


def _read_widths_from_font_dict(font_dict) -> dict[int, int]:
    """Read the /Widths array from a Type1 or TrueType font dictionary.

    Returns a dict mapping character codes to widths in 1/1000 units.
    """
    result: dict[int, int] = {}
    if font_dict is None:
        return result
    try:
        widths_arr = font_dict.get("/Widths")
        first_char = font_dict.get("/FirstChar")
        last_char = font_dict.get("/LastChar")
        if widths_arr is None or first_char is None or last_char is None:
            return result
        widths_arr = _resolve(widths_arr)
        fc = int(first_char)
        lc = int(last_char)
        for i, w in enumerate(widths_arr):
            code = fc + i
            if code > lc:
                break
            result[code] = int(w)
    except Exception:
        pass
    return result


def _read_widths_from_ttfont(font_stream_data: bytes) -> dict[int, int]:
    """Read glyph widths from an embedded TrueType font program using fonttools.

    Returns a dict mapping Unicode code points to widths in 1/1000 units.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return {}

    tt = None
    try:
        tt = TTFont(BytesIO(font_stream_data))
        hmtx = tt["hmtx"]
        try:
            cmap = tt.getBestCmap()
        except KeyError:
            cmap = None
        if cmap is None:
            return {}
        units_per_em = tt["head"].unitsPerEm
        scale = 1000.0 / units_per_em if units_per_em != 1000 else 1.0
        widths: dict[int, int] = {}
        for code, name in cmap.items():
            if name in hmtx.metrics:
                widths[code] = int(hmtx.metrics[name][0] * scale)
        return widths
    except Exception:
        return {}
    finally:
        if tt is not None:
            tt.close()


def _get_font_descriptor(font_dict):
    """Get the /FontDescriptor dictionary from a font dict."""
    if font_dict is None:
        return None
    try:
        fd = font_dict.get("/FontDescriptor")
        if fd is not None:
            return _resolve(fd)
    except Exception:
        pass
    return None


def _get_widths_for_font(font_dict, font_name: str | None = None) -> dict[int, int]:
    """Build a complete width table for a font.

    Priority:
    1. /Widths array from the font dictionary (most reliable)
    2. Embedded TrueType font program via fonttools
    3. Standard-14 metrics by base font name
    4. Empty dict (caller uses fallback)
    """
    # 1. Try /Widths array
    widths = _read_widths_from_font_dict(font_dict)
    if widths:
        return widths

    # 2. Try embedded font program (TrueType / OpenType)
    if font_dict is not None:
        try:
            fd = _get_font_descriptor(font_dict)
            if fd is not None:
                for key in ("/FontFile2", "/FontFile3"):
                    ff = fd.get(key)
                    if ff is not None:
                        ff = _resolve(ff)
                        data = bytes(ff.read_bytes())
                        if data:
                            widths = _read_widths_from_ttfont(data)
                            if widths:
                                return widths
        except Exception:
            pass

    # 3. Try Standard-14 metrics
    name = font_name or _base_font_name(font_dict)
    if name:
        metrics = _get_standard14_metrics(name)
        if metrics:
            return metrics["widths"]

    return {}


def _get_default_width(font_dict, font_name: str | None = None) -> int:
    """Get the default glyph width for missing characters."""
    if font_dict is not None:
        try:
            dw = font_dict.get("/DW")
            if dw is not None:
                return int(dw)
        except Exception:
            pass
        fd = _get_font_descriptor(font_dict)
        if fd is not None:
            try:
                mw = fd.get("/MissingWidth")
                if mw is not None:
                    return int(mw)
            except Exception:
                pass

    name = font_name or _base_font_name(font_dict)
    if name:
        metrics = _get_standard14_metrics(name)
        if metrics:
            return metrics["default_width"]

    return 600  # Courier-width fallback


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _get_encoding_map(font_dict) -> dict[int, int] | None:
    """Build a char-code -> Unicode mapping from the font's /Encoding.

    Returns None if no custom encoding is found (use WinAnsiEncoding default).
    """
    if font_dict is None:
        return None
    try:
        enc = font_dict.get("/Encoding")
        if enc is None:
            return None
        enc = _resolve(enc)
        from pikepdf import Dictionary as _Dictionary

        enc_str = str(enc) if not isinstance(enc, _Dictionary) else None

        if enc_str in ("/WinAnsiEncoding", "/MacRomanEncoding", "/StandardEncoding"):
            return None  # handled by default path

        # Dictionary encoding with /Differences
        if isinstance(enc, _Dictionary):
            diffs = enc.get("/Differences")
            if diffs is None:
                return None
            return _apply_differences(diffs)
    except Exception:
        pass
    return None


def _apply_differences(differences_array) -> dict[int, int]:
    """Parse a /Differences array into a char-code -> Unicode mapping.

    The /Differences array alternates between integer codes and glyph names:
    [32 /space /exclam 65 /A /B ...]
    """
    from pikepdf import Name as _Name

    result: dict[int, int] = {}
    code = 0
    try:
        for item in differences_array:
            item = _resolve(item)
            if isinstance(item, (int, float)):
                code = int(item)
            elif isinstance(item, _Name):
                glyph_name = str(item).lstrip("/")
                cp = _glyph_name_to_unicode(glyph_name)
                if cp is not None:
                    result[code] = cp
                code += 1
            else:
                try:
                    code = int(item)
                except (TypeError, ValueError):
                    code += 1
    except Exception:
        pass
    return result


# Common Adobe glyph names -> Unicode code points
_GLYPH_TO_UNICODE: dict[str, int] = {
    "space": 0x0020,
    "exclam": 0x0021,
    "quotedbl": 0x0022,
    "numbersign": 0x0023,
    "dollar": 0x0024,
    "percent": 0x0025,
    "ampersand": 0x0026,
    "quotesingle": 0x0027,
    "parenleft": 0x0028,
    "parenright": 0x0029,
    "asterisk": 0x002A,
    "plus": 0x002B,
    "comma": 0x002C,
    "hyphen": 0x002D,
    "period": 0x002E,
    "slash": 0x002F,
    "zero": 0x0030,
    "one": 0x0031,
    "two": 0x0032,
    "three": 0x0033,
    "four": 0x0034,
    "five": 0x0035,
    "six": 0x0036,
    "seven": 0x0037,
    "eight": 0x0038,
    "nine": 0x0039,
    "colon": 0x003A,
    "semicolon": 0x003B,
    "less": 0x003C,
    "equal": 0x003D,
    "greater": 0x003E,
    "question": 0x003F,
    "at": 0x0040,
    "A": 0x0041,
    "B": 0x0042,
    "C": 0x0043,
    "D": 0x0044,
    "E": 0x0045,
    "F": 0x0046,
    "G": 0x0047,
    "H": 0x0048,
    "I": 0x0049,
    "J": 0x004A,
    "K": 0x004B,
    "L": 0x004C,
    "M": 0x004D,
    "N": 0x004E,
    "O": 0x004F,
    "P": 0x0050,
    "Q": 0x0051,
    "R": 0x0052,
    "S": 0x0053,
    "T": 0x0054,
    "U": 0x0055,
    "V": 0x0056,
    "W": 0x0057,
    "X": 0x0058,
    "Y": 0x0059,
    "Z": 0x005A,
    "bracketleft": 0x005B,
    "backslash": 0x005C,
    "bracketright": 0x005D,
    "asciicircum": 0x005E,
    "underscore": 0x005F,
    "grave": 0x0060,
    "a": 0x0061,
    "b": 0x0062,
    "c": 0x0063,
    "d": 0x0064,
    "e": 0x0065,
    "f": 0x0066,
    "g": 0x0067,
    "h": 0x0068,
    "i": 0x0069,
    "j": 0x006A,
    "k": 0x006B,
    "l": 0x006C,
    "m": 0x006D,
    "n": 0x006E,
    "o": 0x006F,
    "p": 0x0070,
    "q": 0x0071,
    "r": 0x0072,
    "s": 0x0073,
    "t": 0x0074,
    "u": 0x0075,
    "v": 0x0076,
    "w": 0x0077,
    "x": 0x0078,
    "y": 0x0079,
    "z": 0x007A,
    "braceleft": 0x007B,
    "bar": 0x007C,
    "braceright": 0x007D,
    "asciitilde": 0x007E,
    "bullet": 0x2022,
    "endash": 0x2013,
    "emdash": 0x2014,
    "quoteleft": 0x2018,
    "quoteright": 0x2019,
    "quotedblleft": 0x201C,
    "quotedblright": 0x201D,
    "quotesinglbase": 0x201A,
    "quotedblbase": 0x201E,
    "dagger": 0x2020,
    "daggerdbl": 0x2021,
    "ellipsis": 0x2026,
    "perthousand": 0x2030,
    "guilsinglleft": 0x2039,
    "guilsinglright": 0x203A,
    "trademark": 0x2122,
    "fi": 0xFB01,
    "fl": 0xFB02,
    "Euro": 0x20AC,
    "florin": 0x0192,
    "Scaron": 0x0160,
    "scaron": 0x0161,
    "Zcaron": 0x017D,
    "zcaron": 0x017E,
    "OE": 0x0152,
    "oe": 0x0153,
    "Ydieresis": 0x0178,
    "circumflex": 0x02C6,
    "tilde": 0x02DC,
    "exclamdown": 0x00A1,
    "cent": 0x00A2,
    "sterling": 0x00A3,
    "currency": 0x00A4,
    "yen": 0x00A5,
    "brokenbar": 0x00A6,
    "section": 0x00A7,
    "dieresis": 0x00A8,
    "copyright": 0x00A9,
    "ordfeminine": 0x00AA,
    "guillemotleft": 0x00AB,
    "logicalnot": 0x00AC,
    "registered": 0x00AE,
    "macron": 0x00AF,
    "degree": 0x00B0,
    "plusminus": 0x00B1,
    "twosuperior": 0x00B2,
    "threesuperior": 0x00B3,
    "acute": 0x00B4,
    "mu": 0x00B5,
    "paragraph": 0x00B6,
    "periodcentered": 0x00B7,
    "cedilla": 0x00B8,
    "onesuperior": 0x00B9,
    "ordmasculine": 0x00BA,
    "guillemotright": 0x00BB,
    "onequarter": 0x00BC,
    "onehalf": 0x00BD,
    "threequarters": 0x00BE,
    "questiondown": 0x00BF,
    "Agrave": 0x00C0,
    "Aacute": 0x00C1,
    "Acircumflex": 0x00C2,
    "Atilde": 0x00C3,
    "Adieresis": 0x00C4,
    "Aring": 0x00C5,
    "AE": 0x00C6,
    "Ccedilla": 0x00C7,
    "Egrave": 0x00C8,
    "Eacute": 0x00C9,
    "Ecircumflex": 0x00CA,
    "Edieresis": 0x00CB,
    "Igrave": 0x00CC,
    "Iacute": 0x00CD,
    "Icircumflex": 0x00CE,
    "Idieresis": 0x00CF,
    "Eth": 0x00D0,
    "Ntilde": 0x00D1,
    "Ograve": 0x00D2,
    "Oacute": 0x00D3,
    "Ocircumflex": 0x00D4,
    "Otilde": 0x00D5,
    "Odieresis": 0x00D6,
    "multiply": 0x00D7,
    "Oslash": 0x00D8,
    "Ugrave": 0x00D9,
    "Uacute": 0x00DA,
    "Ucircumflex": 0x00DB,
    "Udieresis": 0x00DC,
    "Yacute": 0x00DD,
    "Thorn": 0x00DE,
    "germandbls": 0x00DF,
    "agrave": 0x00E0,
    "aacute": 0x00E1,
    "acircumflex": 0x00E2,
    "atilde": 0x00E3,
    "adieresis": 0x00E4,
    "aring": 0x00E5,
    "ae": 0x00E6,
    "ccedilla": 0x00E7,
    "egrave": 0x00E8,
    "eacute": 0x00E9,
    "ecircumflex": 0x00EA,
    "edieresis": 0x00EB,
    "igrave": 0x00EC,
    "iacute": 0x00ED,
    "icircumflex": 0x00EE,
    "idieresis": 0x00EF,
    "eth": 0x00F0,
    "ntilde": 0x00F1,
    "ograve": 0x00F2,
    "oacute": 0x00F3,
    "ocircumflex": 0x00F4,
    "otilde": 0x00F5,
    "odieresis": 0x00F6,
    "divide": 0x00F7,
    "oslash": 0x00F8,
    "ugrave": 0x00F9,
    "uacute": 0x00FA,
    "ucircumflex": 0x00FB,
    "udieresis": 0x00FC,
    "yacute": 0x00FD,
    "thorn": 0x00FE,
    "ydieresis": 0x00FF,
    "sfthyphen": 0x00AD,
    "nbspace": 0x00A0,
}


def _glyph_name_to_unicode(name: str) -> int | None:
    """Map an Adobe glyph name to a Unicode code point."""
    if name in _GLYPH_TO_UNICODE:
        return _GLYPH_TO_UNICODE[name]
    # Try "uniXXXX" format
    if name.startswith("uni") and len(name) == 7:
        try:
            return int(name[3:], 16)
        except ValueError:
            pass
    # Single ASCII character
    if len(name) == 1:
        return ord(name)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_text_width(
    text: str, font_dict=None, font_size: float = 12.0, font_name: str | None = None
) -> float:
    """Calculate the width of a text string in PDF points.

    Args:
        text: The text string to measure.
        font_dict: pikepdf font dictionary (optional).
        font_size: Font size in points.
        font_name: Font resource name (e.g. "Helv") for Standard-14 lookup.

    Returns:
        Width in PDF points (font_size * sum_of_glyph_widths / 1000).
    """
    if not text:
        return 0.0

    widths = _get_widths_for_font(font_dict, font_name)
    default_w = _get_default_width(font_dict, font_name)

    total = 0
    for ch in text:
        cp = ord(ch)
        # For WinAnsiEncoding fonts, map Unicode to WinAnsi code first
        wa = _unicode_to_winansi(cp)
        if wa is not None and wa in widths:
            total += widths[wa]
        elif cp in widths:
            total += widths[cp]
        else:
            total += default_w

    return total * font_size / 1000.0


def get_font_bbox(font_dict) -> tuple[float, float, float, float]:
    """Return the font bounding box (llx, lly, urx, ury) in 1/1000 units.

    Falls back to Standard-14 values, then to a generic default.
    """
    fd = _get_font_descriptor(font_dict)
    if fd is not None:
        try:
            bbox = fd.get("/FontBBox")
            if bbox is not None and len(bbox) == 4:
                return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            pass

    name = _base_font_name(font_dict)
    if name:
        metrics = _get_standard14_metrics(name)
        if metrics:
            return metrics["bbox"]

    return (-200, -250, 1100, 950)


def get_ascent_descent(
    font_dict=None, font_name: str | None = None
) -> tuple[float, float]:
    """Return (ascent, descent) in 1/1000 units.

    Ascent is positive (above baseline), descent is negative (below).
    Falls back to Standard-14 values, then to a generic default.
    """
    fd = _get_font_descriptor(font_dict)
    if fd is not None:
        try:
            asc = fd.get("/Ascent")
            desc = fd.get("/Descent")
            if asc is not None and desc is not None:
                return float(asc), float(desc)
        except Exception:
            pass

    name = font_name or _base_font_name(font_dict)
    if name:
        metrics = _get_standard14_metrics(name)
        if metrics:
            return metrics["ascent"], metrics["descent"]

    return 750.0, -250.0


def compute_auto_font_size(
    text: str,
    font_dict=None,
    field_width: float = 100.0,
    field_height: float = 20.0,
    font_name: str | None = None,
    multiline: bool = False,
) -> float:
    """Compute the optimal font size when DA specifies size 0 (auto-size).

    Args:
        text: The text to fit.
        font_dict: pikepdf font dictionary (optional).
        field_width: Available width in points.
        field_height: Available height in points.
        font_name: Font resource name for Standard-14 lookup.
        multiline: If True, consider word-wrapping.

    Returns:
        Optimal font size in points (minimum 4, maximum field_height - 4).
    """
    if not text:
        return min(12.0, max(4.0, field_height - 4))

    margin = 2.0
    avail_w = field_width - 2 * margin
    avail_h = field_height - 2 * margin

    if avail_w <= 0 or avail_h <= 0:
        return 4.0

    max_size = min(avail_h, 96.0)
    min_size = 4.0

    if not multiline:
        # Binary search for largest size that fits width and height
        lo, hi = min_size, max_size
        for _ in range(20):  # ~20 iterations gives sub-0.01pt precision
            mid = (lo + hi) / 2.0
            tw = get_text_width(text, font_dict, mid, font_name)
            if tw <= avail_w and mid <= avail_h:
                lo = mid
            else:
                hi = mid
        return max(min_size, lo)

    # Multiline: try decreasing sizes until wrapped text fits
    ascent, descent = get_ascent_descent(font_dict, font_name)
    line_height_factor = (ascent - descent) / 1000.0 * 1.2

    size = max_size
    while size >= min_size:
        lh = size * line_height_factor
        lines = _wrap_text(text, font_dict, size, avail_w, font_name)
        total_h = len(lines) * lh
        if total_h <= avail_h:
            return size
        size -= max(0.5, size * 0.1)

    return min_size


def _wrap_text(
    text: str,
    font_dict,
    font_size: float,
    max_width: float,
    font_name: str | None = None,
) -> list[str]:
    """Word-wrap text to fit within max_width at the given font size.

    Handles explicit line breaks (\\r, \\n, \\r\\n) and word-wrapping.
    """
    if max_width <= 0:
        return text.splitlines() or [""]

    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    result: list[str] = []

    for para in paragraphs:
        if not para:
            result.append("")
            continue

        words = para.split(" ")
        current_line = ""
        for word in words:
            if not current_line:
                candidate = word
            else:
                candidate = current_line + " " + word

            w = get_text_width(candidate, font_dict, font_size, font_name)
            if w <= max_width:
                current_line = candidate
            else:
                if current_line:
                    result.append(current_line)
                # Check if single word is wider than max_width
                ww = get_text_width(word, font_dict, font_size, font_name)
                if ww > max_width:
                    # Character-level wrapping
                    current_line = ""
                    for ch in word:
                        test = current_line + ch
                        cw = get_text_width(test, font_dict, font_size, font_name)
                        if cw > max_width and current_line:
                            result.append(current_line)
                            current_line = ch
                        else:
                            current_line = test
                else:
                    current_line = word

        if current_line:
            result.append(current_line)

    return result or [""]


# ---------------------------------------------------------------------------
# PDF string decode / encode
# ---------------------------------------------------------------------------


def decode_pdf_string(value, font_dict=None) -> str:
    """Decode a pikepdf String value to a Python str.

    Handles UTF-16BE (BOM \\xfe\\xff), PDFDocEncoding, and custom encodings.
    """
    if value is None:
        return ""

    # pikepdf.String — str() usually does the right thing for BMP text
    text = str(value)

    # pikepdf already handles UTF-16BE BOM and PDFDocEncoding in most cases.
    # For custom /Encoding with /Differences, we'd need byte-level access,
    # but for form field values (/V), str() is reliable.
    return text


def encode_for_content_stream(text: str, font_dict=None) -> bytes:
    """Encode a Python string for a PDF content stream Tj operator.

    Maps Unicode characters to WinAnsiEncoding bytes, escaping
    parentheses and backslashes.
    """
    out = bytearray()
    for ch in text:
        cp = ord(ch)
        wa = _unicode_to_winansi(cp)
        if wa is not None and 0 <= wa <= 255:
            byte = wa
        elif 0 <= cp <= 255:
            byte = cp
        else:
            byte = 0x3F  # '?' for unmappable characters

        if byte == 0x5C:  # backslash
            out.extend(b"\\\\")
        elif byte == 0x28:  # (
            out.extend(b"\\(")
        elif byte == 0x29:  # )
            out.extend(b"\\)")
        else:
            out.append(byte)

    return bytes(out)
