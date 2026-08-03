# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Remove .notdef glyph references from content streams.

ISO 19005-2, Rule 6.2.11.8 forbids references to the .notdef glyph
from any text-showing operator.  This module strips character codes
that resolve to .notdef from Tj, TJ, ' and " operators in all content
streams (page contents, Form XObjects, Tiling Patterns, annotation AP
streams, Type3 CharProcs).
"""

import io
import logging
import re
import struct
from dataclasses import dataclass, field

import pikepdf
from pikepdf import Array, Dictionary, Name, Operator, Pdf, Stream, String

from ..exceptions import ConversionError
from ..fonts.glyph_mapping import resolve_glyph_name
from ..fonts.glyph_usage import (
    _iter_content_streams_with_resources,
    find_ambiguous_resource_context_streams,
)
from ..fonts.subsetter import (
    _get_unicode_cmap,
    _resolve_simple_font_encoding,
)
from ..fonts.tounicode import (
    CIDEncodingMap,
    get_type0_cid_encoding_map,
    parse_tounicode_cmap_sequences,
    split_cmap_codes,
)
from ..fonts.utils import get_any_cmap as _get_any_cmap
from ..fonts.utils import safe_str as _safe_str
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Text-showing operators whose string operands may reference .notdef
_TEXT_OPERATORS = frozenset({"Tj", "'", '"'})
_SAVE_GRAPHICS_STATE = "q"
_RESTORE_GRAPHICS_STATE = "Q"


class _NotdefCodes:
    """Set-like object for .notdef character codes.

    Supports both explicit codes and a max_valid_code threshold so that
    CIDFonts with Identity mapping can represent "every CID >= numGlyphs"
    without building a 65k-entry frozenset.
    """

    __slots__ = ("_cid_map", "_explicit", "_max_valid_code", "_valid_codes")

    def __init__(
        self,
        explicit: frozenset[int] = frozenset(),
        max_valid_code: int | None = None,
        valid_codes: frozenset[int] | None = None,
        cid_map: CIDEncodingMap | None = None,
    ) -> None:
        self._explicit = explicit
        self._max_valid_code = max_valid_code
        self._valid_codes = valid_codes
        self._cid_map = cid_map

    def __contains__(self, code: int | bytes) -> bool:
        if self._cid_map is not None:
            code = self._cid_map.map_code(code)
        elif isinstance(code, bytes):
            code = int.from_bytes(code, "big")
        if code in self._explicit:
            return True
        if self._valid_codes is not None:
            return code not in self._valid_codes
        return self._max_valid_code is not None and code > self._max_valid_code

    def __bool__(self) -> bool:
        return (
            bool(self._explicit)
            or self._max_valid_code is not None
            or self._valid_codes is not None
        )

    @property
    def code_space_ranges(self) -> tuple[tuple[bytes, bytes], ...]:
        """Return the embedded CMap code-space ranges, when available."""
        if self._cid_map is None:
            return ()
        return self._cid_map.code_space_ranges

    def mapped_code(self, code: int | bytes) -> int:
        """Return the descendant CID, or the simple-font character code."""
        if self._cid_map is not None:
            return self._cid_map.map_code(code)
        if isinstance(code, bytes):
            return int.from_bytes(code, "big")
        return code


def sanitize_notdef_usage(pdf: Pdf) -> dict[str, int]:
    """Removes .notdef glyph references from content streams.

    Scans all content streams for text-showing operators whose character
    codes resolve to .notdef for the active font, and strips those bytes
    from the operand strings.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"notdef_usage_fixed": N}``.
    """
    total_fixed = 0
    # Cache notdef codes per font objgen to avoid recomputation
    notdef_cache: dict[tuple[int, int], _NotdefCodes] = {}
    ambiguous_streams = find_ambiguous_resource_context_streams(pdf)

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            for owner, resources in _iter_content_streams_with_resources(page):
                font_map = _build_font_map(resources)
                if isinstance(owner, Stream):
                    objgen = owner.objgen
                    if objgen != (0, 0) and objgen in ambiguous_streams:
                        continue
                    total_fixed += _fix_notdef_in_stream(owner, font_map, notdef_cache)
                else:
                    total_fixed += _fix_notdef_in_page_contents(
                        owner,
                        font_map,
                        notdef_cache,
                        ambiguous_streams,
                    )

        except ConversionError:
            raise
        except Exception as e:
            log_suppressed_error(
                logger, e, "Error fixing .notdef usage on page %d: %s", page_num, e
            )

    if total_fixed > 0:
        logger.info("Notdef usage: %d text operators fixed", total_fixed)

    return {"notdef_usage_fixed": total_fixed}


# ---------------------------------------------------------------------------
# Font map building
# ---------------------------------------------------------------------------


def _build_font_map(
    resources: pikepdf.Object,
) -> dict[str, pikepdf.Object]:
    """Builds a mapping from font resource name to font dictionary.

    Args:
        resources: A resolved Resources dictionary.

    Returns:
        Dictionary mapping font name (e.g. "/F1") to resolved font dict.
    """
    font_map: dict[str, pikepdf.Object] = {}
    resources = _resolve(resources)
    if not isinstance(resources, Dictionary):
        return font_map
    font_dict = resources.get("/Font")
    if font_dict is None:
        return font_map
    font_dict = _resolve(font_dict)
    if not isinstance(font_dict, Dictionary):
        return font_map
    for key in list(font_dict.keys()):
        try:
            font_obj = _resolve(font_dict[key])
            font_map[str(key)] = font_obj
        except Exception:
            continue
    return font_map


# ---------------------------------------------------------------------------
# Notdef code computation (cached per font)
# ---------------------------------------------------------------------------


def _get_notdef_codes(
    font_obj: pikepdf.Object,
    cache: dict[tuple[int, int], _NotdefCodes],
) -> _NotdefCodes:
    """Returns the set of character codes that resolve to .notdef for a font.

    Results are cached by font objgen.

    Args:
        font_obj: Resolved font dictionary.
        cache: Shared cache dict.

    Returns:
        _NotdefCodes instance for .notdef character/CID codes.
    """
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        return _NotdefCodes()

    # Cache key — only indirect objects can be stably cached
    objgen = font_obj.objgen
    cache_key: tuple[int, int] | None
    if objgen != (0, 0):
        cache_key = objgen
        if cache_key in cache:
            return cache[cache_key]
    else:
        cache_key = None

    subtype = font_obj.get("/Subtype")
    if subtype is None:
        result = _NotdefCodes()
        if cache_key is not None:
            cache[cache_key] = result
        return result

    subtype_str = _safe_str(subtype)

    if subtype_str in ("/TrueType", "/Type1", "/MMType1"):
        result = _get_simple_font_notdef_codes(font_obj)
    elif subtype_str == "/Type0":
        result = _get_cidfont_notdef_codes(font_obj)
    else:
        result = _NotdefCodes()

    if cache_key is not None:
        cache[cache_key] = result
    return result


def _find_missing_glyphs_in_simple_font(font_obj: pikepdf.Object) -> set[int]:
    """Finds character codes (0-255) whose glyph is missing.

    Parses the embedded font program with fontTools, resolves the font's
    encoding to map codes to glyph names, and returns codes whose glyph
    name is absent from the font's glyph order.

    Args:
        font_obj: Resolved simple font dictionary.

    Returns:
        Set of character codes whose encoded glyph is missing.
    """
    try:
        fd = font_obj.get("/FontDescriptor")
        if fd is None:
            return set()
        fd = _resolve(fd)

        # Find embedded font data
        font_data = None
        font_file_key = None
        for key in ("/FontFile2", "/FontFile3", "/FontFile"):
            stream = fd.get(key)
            if stream is not None:
                stream = _resolve(stream)
                font_data = bytes(stream.read_bytes())
                font_file_key = key
                break

        if font_data is None:
            return set()

        subtype = _safe_str(font_obj.get("/Subtype") or b"")
        if subtype == "/TrueType":
            from ..fonts.subsetter import _resolve_truetype_font_encoding

            encoding = _resolve_truetype_font_encoding(
                font_obj,
                font_data,
                pdfa_normalized=True,
            )
        else:
            encoding = _resolve_simple_font_encoding(font_obj)
        if not encoding:
            return set()

        from fontTools.ttLib import TTFont

        tt_font = None
        try:
            try:
                tt_font = TTFont(io.BytesIO(font_data))
            except Exception:
                if font_file_key == "/FontFile3":
                    from .glyph_coverage import _wrap_cff_in_otf

                    otf_data = _wrap_cff_in_otf(font_data)
                    tt_font = TTFont(io.BytesIO(otf_data))
                else:
                    return set()

            glyph_set = set(tt_font.getGlyphOrder())
            hmtx_metrics = tt_font["hmtx"].metrics if "hmtx" in tt_font else {}
            cmap = _get_any_cmap(tt_font)
            missing = set()
            for code in range(256):
                name = encoding.get(code)
                if name is None or name == ".notdef":
                    # No encoding entry or explicit .notdef → maps to .notdef
                    missing.add(code)
                elif (
                    name not in glyph_set
                    and resolve_glyph_name(name, cmap, hmtx_metrics) is None
                ):
                    missing.add(code)

            return missing
        finally:
            if tt_font is not None:
                tt_font.close()
    except Exception:
        logger.debug("Error analyzing simple font glyphs", exc_info=True)
        return set()


def _get_simple_font_notdef_codes(font_obj: pikepdf.Object) -> _NotdefCodes:
    """Computes character codes that resolve to .notdef for simple fonts.

    The encoding selects the glyph for every code regardless of the
    [FirstChar, LastChar] range (that range only affects widths, which
    fall back to MissingWidth outside it), so a code is only .notdef
    when its encoded glyph is actually absent from the embedded font
    program.  Fonts without an embedded program are left untouched.

    Args:
        font_obj: Resolved simple font dictionary.

    Returns:
        _NotdefCodes for byte values (0-255) that are .notdef.
    """
    return _NotdefCodes(frozenset(_find_missing_glyphs_in_simple_font(font_obj)))


def _get_cidfont_num_glyphs(cidfont: pikepdf.Object) -> int | None:
    """Returns the number of glyphs in a CIDFont's embedded font program.

    Args:
        cidfont: Resolved CIDFont dictionary (descendant font).

    Returns:
        Number of glyphs, or None if the font program cannot be parsed.
    """
    try:
        fd = cidfont.get("/FontDescriptor")
        if fd is None:
            return None
        fd = _resolve(fd)

        font_data = None
        font_file_key = None
        for key in ("/FontFile2", "/FontFile3", "/FontFile"):
            stream = fd.get(key)
            if stream is not None:
                stream = _resolve(stream)
                font_data = bytes(stream.read_bytes())
                font_file_key = key
                break

        if font_data is None:
            return None

        from fontTools.ttLib import TTFont

        tt_font = None
        try:
            try:
                tt_font = TTFont(io.BytesIO(font_data))
            except Exception:
                if font_file_key == "/FontFile3":
                    from .glyph_coverage import _wrap_cff_in_otf

                    otf_data = _wrap_cff_in_otf(font_data)
                    tt_font = TTFont(io.BytesIO(otf_data))
                else:
                    return None

            return len(tt_font.getGlyphOrder())
        finally:
            if tt_font is not None:
                tt_font.close()
    except Exception:
        return None


def _get_cidfonttype0_valid_cids(cidfont: pikepdf.Object) -> frozenset[int] | None:
    """Return valid CIDs for CIDFontType0 fonts from the CFF charset."""
    try:
        fd = cidfont.get("/FontDescriptor")
        if fd is None:
            return None
        fd = _resolve(fd)

        font_data = None
        font_file_key = None
        for key in ("/FontFile3", "/FontFile2", "/FontFile"):
            stream = fd.get(key)
            if stream is not None:
                stream = _resolve(stream)
                font_data = bytes(stream.read_bytes())
                font_file_key = key
                break

        if font_data is None:
            return None

        from fontTools.ttLib import TTFont

        tt_font = None
        try:
            try:
                tt_font = TTFont(io.BytesIO(font_data))
            except Exception:
                if font_file_key == "/FontFile3":
                    from .glyph_coverage import _wrap_cff_in_otf

                    tt_font = TTFont(io.BytesIO(_wrap_cff_in_otf(font_data)))
                else:
                    return None

            if "CFF " not in tt_font:
                return None

            valid_cids: set[int] = set()
            char_strings = tt_font["CFF "].cff.topDictIndex[0].CharStrings
            for name in char_strings.keys():
                if name == ".notdef":
                    valid_cids.add(0)
                elif name.startswith("cid"):
                    try:
                        valid_cids.add(int(name[3:]))
                    except ValueError:
                        continue

            return frozenset(valid_cids) if valid_cids else None
        finally:
            if tt_font is not None:
                tt_font.close()
    except Exception:
        return None


def _get_cidfont_notdef_codes(font_obj: pikepdf.Object) -> _NotdefCodes:
    """Computes CIDs that resolve to .notdef for CIDFonts (Type0).

    For CIDFonts with Identity CIDToGIDMap, CID 0 is always .notdef
    and any CID >= numGlyphs is also .notdef.  For stream CIDToGIDMap,
    CIDs mapping to GID 0 or to GID >= numGlyphs are .notdef.

    Args:
        font_obj: Resolved Type0 font dictionary.

    Returns:
        _NotdefCodes for CID values that are .notdef.
    """
    notdef: set[int] = set()
    cid_map = get_type0_cid_encoding_map(font_obj)
    if cid_map is None:
        return _NotdefCodes()

    descendants = font_obj.get("/DescendantFonts")
    if descendants is None:
        return _NotdefCodes()
    descendants = _resolve(descendants)
    if not isinstance(descendants, Array) or len(descendants) == 0:
        return _NotdefCodes()

    cidfont = _resolve(descendants[0])
    if not isinstance(cidfont, Dictionary):
        return _NotdefCodes()

    cidfont_subtype = _safe_str(cidfont.get("/Subtype") or "")
    if cidfont_subtype == "/CIDFontType0":
        valid_cids = _get_cidfonttype0_valid_cids(cidfont)
        if valid_cids is not None:
            return _NotdefCodes(frozenset({0}), valid_codes=valid_cids, cid_map=cid_map)
        return _NotdefCodes(frozenset({0}), cid_map=cid_map)

    num_glyphs = _get_cidfont_num_glyphs(cidfont)

    cidtogidmap = cidfont.get("/CIDToGIDMap")
    if cidtogidmap is None:
        # No mapping — CID 0 is .notdef by convention
        notdef.add(0)
        max_valid = (num_glyphs - 1) if num_glyphs is not None else None
        return _NotdefCodes(frozenset(notdef), max_valid, cid_map=cid_map)

    cidtogidmap = _resolve(cidtogidmap)

    if isinstance(cidtogidmap, Name) and str(cidtogidmap) == "/Identity":
        # Identity mapping: CID = GID, so CID 0 → GID 0 → .notdef
        # and CID >= numGlyphs → beyond font program
        notdef.add(0)
        max_valid = (num_glyphs - 1) if num_glyphs is not None else None
        return _NotdefCodes(frozenset(notdef), max_valid, cid_map=cid_map)
    elif isinstance(cidtogidmap, Stream):
        # Stream mapping: parse to find CIDs that map to GID 0
        # or to GID >= numGlyphs
        notdef.add(0)
        max_valid: int | None = None
        try:
            stream_data = bytes(cidtogidmap.read_bytes())
            num_entries = len(stream_data) // 2
            max_valid = num_entries - 1
            for cid in range(num_entries):
                gid = struct.unpack_from(">H", stream_data, cid * 2)[0]
                if gid == 0:
                    notdef.add(cid)
                elif num_glyphs is not None and gid >= num_glyphs:
                    notdef.add(cid)
        except Exception:
            # If we can't parse, conservatively only flag CID 0
            notdef.add(0)

        return _NotdefCodes(
            frozenset(notdef),
            max_valid_code=max_valid,
            cid_map=cid_map,
        )

    return _NotdefCodes(frozenset(notdef), cid_map=cid_map)


# ---------------------------------------------------------------------------
# Content stream fixing
# ---------------------------------------------------------------------------


@dataclass
class _ContentState:
    """Font-related graphics state shared by consecutive content streams."""

    current_font_name: str | None = None
    font_size: float = 0.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    text_rendering_mode: int = 0
    graphics_state_stack: list[tuple[str | None, float, float, float, int]] = field(
        default_factory=list
    )
    width_cache: dict[tuple[tuple[int, int], bytes], float | None] = field(
        default_factory=dict
    )


def _is_cidfont(font_obj: pikepdf.Object) -> bool:
    """Checks if a font is a CIDFont (Type0).

    Args:
        font_obj: pikepdf font object.

    Returns:
        True if the font is Type0 (CIDFont).
    """
    try:
        subtype = font_obj.get("/Subtype")
        if subtype is not None and str(subtype) == "/Type0":
            return True
    except Exception:
        pass
    return False


def _font_identity(font_obj: pikepdf.Object) -> tuple[int, int] | None:
    """Return a stable cache key for an indirect font."""
    objgen = font_obj.objgen
    return objgen if objgen != (0, 0) else None


def _type0_wmode(font_obj: pikepdf.Object) -> int:
    """Return the Type 0 font writing mode."""
    encoding = _resolve(font_obj.get("/Encoding"))
    if isinstance(encoding, Name):
        return int(str(encoding).endswith("-V"))
    if isinstance(encoding, Stream):
        try:
            wmode = encoding.get("/WMode")
            if wmode is not None:
                return int(wmode)
            match = re.search(rb"/WMode\s+([01])\s+def", encoding.read_bytes())
            if match is not None:
                return int(match.group(1))
        except Exception:
            pass
    return 0


def _cid_width(cidfont: Dictionary, cid: int, vertical: bool) -> float:
    """Return a CID's horizontal or vertical displacement in glyph units."""
    if vertical:
        default = -1000.0
        dw2 = _resolve(cidfont.get("/DW2"))
        if isinstance(dw2, Array) and len(dw2) >= 2:
            try:
                default = float(dw2[1])
            except (TypeError, ValueError):
                pass
        widths = _resolve(cidfont.get("/W2"))
        stride = 3
    else:
        try:
            default = float(cidfont.get("/DW", 1000))
        except (TypeError, ValueError):
            default = 1000.0
        widths = _resolve(cidfont.get("/W"))
        stride = 1

    if not isinstance(widths, Array):
        return default

    items = list(widths)
    index = 0
    while index < len(items):
        try:
            start = int(items[index])
        except (TypeError, ValueError):
            break
        index += 1
        if index >= len(items):
            break
        next_item = _resolve(items[index])
        if isinstance(next_item, Array):
            offset = (cid - start) * stride
            if 0 <= offset < len(next_item):
                try:
                    return float(next_item[offset])
                except (TypeError, ValueError):
                    return default
            index += 1
            continue
        try:
            end = int(next_item)
        except (TypeError, ValueError):
            break
        index += 1
        if index + stride > len(items):
            break
        if start <= cid <= end:
            try:
                return float(items[index])
            except (TypeError, ValueError):
                return default
        index += stride
    return default


def _simple_notdef_width(font_obj: Dictionary, code: int) -> float | None:
    """Return the exact advance used for a simple-font .notdef glyph."""
    try:
        first_char = int(font_obj.get("/FirstChar", 0))
        widths = _resolve(font_obj.get("/Widths"))
        offset = code - first_char
        if isinstance(widths, Array) and 0 <= offset < len(widths):
            return float(widths[offset])
    except (TypeError, ValueError):
        pass

    descriptor = _resolve(font_obj.get("/FontDescriptor"))
    if isinstance(descriptor, Dictionary):
        try:
            missing_width = descriptor.get("/MissingWidth")
            if missing_width is not None:
                return float(missing_width)
        except (TypeError, ValueError):
            pass

        for key in ("/FontFile2", "/FontFile3"):
            font_stream = _resolve(descriptor.get(key))
            if not isinstance(font_stream, Stream):
                continue
            try:
                from fontTools.ttLib import TTFont

                tt_font = TTFont(io.BytesIO(bytes(font_stream.read_bytes())))
                try:
                    glyph_order = tt_font.getGlyphOrder()
                    if (
                        glyph_order
                        and "hmtx" in tt_font
                        and glyph_order[0] in tt_font["hmtx"].metrics
                    ):
                        width = tt_font["hmtx"].metrics[glyph_order[0]][0]
                        units_per_em = tt_font["head"].unitsPerEm
                        return width * 1000.0 / units_per_em
                finally:
                    tt_font.close()
            except Exception:
                continue
    return None


def _code_advance_width(
    font_obj: pikepdf.Object,
    encoded: bytes,
    notdef_codes: _NotdefCodes,
    state: _ContentState,
) -> float | None:
    """Return the text displacement of one encoded .notdef character."""
    identity = _font_identity(font_obj)
    cache_key = (identity, encoded) if identity is not None else None
    if cache_key is not None and cache_key in state.width_cache:
        return state.width_cache[cache_key]

    width: float | None = None
    font_obj = _resolve(font_obj)
    if isinstance(font_obj, Dictionary):
        if _is_cidfont(font_obj):
            descendants = _resolve(font_obj.get("/DescendantFonts"))
            if isinstance(descendants, Array) and descendants:
                cidfont = _resolve(descendants[0])
                if isinstance(cidfont, Dictionary):
                    cid = notdef_codes.mapped_code(encoded)
                    width = _cid_width(
                        cidfont,
                        cid,
                        vertical=_type0_wmode(font_obj) == 1,
                    )
        elif len(encoded) == 1:
            width = _simple_notdef_width(font_obj, encoded[0])

    if cache_key is not None:
        state.width_cache[cache_key] = width
    return width


def _cid_notdef_width(font_obj: pikepdf.Object) -> float | None:
    """Return the descendant font's .notdef displacement."""
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary) or not _is_cidfont(font_obj):
        return None
    descendants = _resolve(font_obj.get("/DescendantFonts"))
    if not isinstance(descendants, Array) or not descendants:
        return None
    cidfont = _resolve(descendants[0])
    if not isinstance(cidfont, Dictionary):
        return None
    return _cid_width(
        cidfont,
        0,
        vertical=_type0_wmode(font_obj) == 1,
    )


def _semantic_cid_replacement(
    font_obj: pikepdf.Object,
    encoded: bytes,
    unicode_sequence: tuple[int, ...],
) -> tuple[bytes, float]:
    """Map invisible OCR text to a real nonzero glyph with the same Unicode."""
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        raise ConversionError("Invisible OCR font dictionary is invalid")

    encoding = _resolve(font_obj.get("/Encoding"))
    if not isinstance(encoding, Name) or str(encoding) != "/Identity-H":
        raise ConversionError(
            "Invisible .notdef OCR text can only be safely remapped for "
            "Identity-H CID fonts"
        )
    if len(encoded) != 2 or not unicode_sequence:
        raise ConversionError("Invisible OCR character mapping is not remappable")

    descendants = _resolve(font_obj.get("/DescendantFonts"))
    if not isinstance(descendants, Array) or not descendants:
        raise ConversionError("Invisible OCR CID font has no descendant")
    cidfont = _resolve(descendants[0])
    if (
        not isinstance(cidfont, Dictionary)
        or str(cidfont.get("/Subtype")) != "/CIDFontType2"
    ):
        raise ConversionError(
            "Invisible .notdef OCR text requires a CIDFontType2 descendant"
        )

    cidtogid = _resolve(cidfont.get("/CIDToGIDMap"))
    if not isinstance(cidtogid, Name) or str(cidtogid) != "/Identity":
        raise ConversionError(
            "Invisible .notdef OCR text requires an Identity CIDToGIDMap"
        )

    descriptor = _resolve(cidfont.get("/FontDescriptor"))
    if not isinstance(descriptor, Dictionary):
        raise ConversionError("Invisible OCR font has no descriptor")
    font_stream = None
    for key in ("/FontFile2", "/FontFile3"):
        candidate = _resolve(descriptor.get(key))
        if isinstance(candidate, Stream):
            font_stream = candidate
            break
    if font_stream is None:
        raise ConversionError("Invisible OCR font program is unavailable")

    from fontTools.ttLib import TTFont

    tt_font = TTFont(io.BytesIO(font_stream.read_bytes()))
    try:
        cmap = _get_unicode_cmap(tt_font)
        glyph_name = cmap.get(unicode_sequence[0])
        if glyph_name is None:
            raise ConversionError(
                "Invisible OCR Unicode value has no real glyph in the font"
            )
        glyph_id = tt_font.getGlyphID(glyph_name)
        if glyph_id <= 0 or glyph_id > 65_535:
            raise ConversionError(
                "Invisible OCR Unicode value does not resolve to a valid CID"
            )
        if "hmtx" not in tt_font or "head" not in tt_font:
            raise ConversionError("Invisible OCR font has no horizontal metrics")
        advance = tt_font["hmtx"].metrics[glyph_name][0]
        width = advance * 1000.0 / tt_font["head"].unitsPerEm
    finally:
        tt_font.close()

    replacement = glyph_id.to_bytes(2, "big")
    _set_cid_width(cidfont, glyph_id, round(width))
    _add_tounicode_mapping(font_obj, replacement, unicode_sequence)
    return replacement, width


def _set_cid_width(cidfont: Dictionary, cid: int, width: int) -> None:
    """Set one horizontal CID width without disturbing surrounding ranges."""
    widths = _resolve(cidfont.get("/W"))
    if not isinstance(widths, Array):
        cidfont[Name.W] = Array([cid, Array([width])])
        return

    items = list(widths)
    rewritten: list = []
    found = False
    index = 0
    while index < len(items):
        try:
            start = int(items[index])
        except (TypeError, ValueError):
            rewritten.extend(items[index:])
            break
        if index + 1 >= len(items):
            rewritten.append(items[index])
            break

        next_item = _resolve(items[index + 1])
        if isinstance(next_item, Array):
            values = list(next_item)
            offset = cid - start
            if 0 <= offset < len(values):
                values[offset] = width
                found = True
            rewritten.extend([start, Array(values)])
            index += 2
            continue

        if index + 2 >= len(items):
            rewritten.extend(items[index:])
            break
        try:
            end = int(next_item)
            range_width = items[index + 2]
        except (TypeError, ValueError):
            rewritten.extend(items[index : index + 3])
            index += 3
            continue
        if start <= cid <= end:
            if start < cid:
                rewritten.extend([start, cid - 1, range_width])
            rewritten.extend([cid, Array([width])])
            if cid < end:
                rewritten.extend([cid + 1, end, range_width])
            found = True
        else:
            rewritten.extend(items[index : index + 3])
        index += 3

    if not found:
        rewritten.extend([cid, Array([width])])
    cidfont[Name.W] = Array(rewritten)


def _add_tounicode_mapping(
    font_obj: Dictionary,
    encoded: bytes,
    unicode_sequence: tuple[int, ...],
) -> None:
    """Add a non-conflicting source mapping to an existing ToUnicode CMap."""
    tounicode = _resolve(font_obj.get("/ToUnicode"))
    if not isinstance(tounicode, Stream):
        raise ConversionError("Invisible OCR font has no ToUnicode CMap")
    data = tounicode.read_bytes()
    mappings = parse_tounicode_cmap_sequences(data)
    existing = mappings.get(encoded)
    if existing == unicode_sequence:
        return
    if existing is not None:
        raise ConversionError(
            "Invisible OCR remap would overwrite an existing ToUnicode mapping"
        )

    marker = re.search(rb"\bendcmap\b", data)
    if marker is None:
        raise ConversionError("Invisible OCR ToUnicode CMap is malformed")
    try:
        destination = "".join(chr(value) for value in unicode_sequence).encode(
            "utf-16-be"
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise ConversionError("Invisible OCR Unicode mapping is invalid") from exc
    entry = (
        b"1 beginbfchar\n<"
        + encoded.hex().upper().encode()
        + b"> <"
        + destination.hex().upper().encode()
        + b">\nendbfchar\n"
    )
    tounicode.write(data[: marker.start()] + entry + data[marker.start() :])


def _remap_invisible_tj_items(
    operand: pikepdf.Object,
    font_obj: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    state: _ContentState,
) -> list | None:
    """Replace invisible .notdef codes while preserving Unicode and advance."""
    try:
        raw = bytes(operand)
    except Exception:
        return None

    ranges = notdef_codes.code_space_ranges or ((b"\x00\x00", b"\xff\xff"),)
    codes = split_cmap_codes(raw, ranges)
    tounicode = _resolve(font_obj.get("/ToUnicode"))
    unicode_map = (
        parse_tounicode_cmap_sequences(tounicode.read_bytes())
        if isinstance(tounicode, Stream)
        else {}
    )

    items: list = []
    changed = False
    consumed = 0
    for encoded in codes:
        consumed += len(encoded)
        if encoded not in notdef_codes:
            _append_tj_item(items, String(encoded))
            continue

        old_width = _code_advance_width(font_obj, encoded, notdef_codes, state)
        if old_width is None:
            raise ConversionError(
                "Invisible .notdef OCR text has no reliable advance width"
            )
        unicode_sequence = unicode_map.get(encoded)
        if not unicode_sequence:
            adjustment = -old_width
            if state.char_spacing:
                if state.font_size == 0:
                    raise ConversionError(
                        "Invisible .notdef OCR text has no reliable advance width"
                    )
                adjustment -= 1000.0 * state.char_spacing / state.font_size
            _append_tj_item(items, adjustment)
            changed = True
            continue
        replacement, new_width = _semantic_cid_replacement(
            font_obj,
            encoded,
            unicode_sequence,
        )
        if replacement in notdef_codes:
            raise ConversionError("Invisible OCR replacement still resolves to .notdef")
        _append_tj_item(items, String(replacement))
        adjustment = new_width - old_width
        if abs(adjustment) > 0.001:
            _append_tj_item(items, adjustment)
        changed = True

    if consumed < len(raw):
        _append_tj_item(items, String(raw[consumed:]))
    return items if changed else None


def _append_tj_item(items: list, item: pikepdf.Object | float) -> None:
    """Append a TJ item while coalescing adjacent strings or adjustments."""
    if isinstance(item, String) and items and isinstance(items[-1], String):
        items[-1] = String(bytes(items[-1]) + bytes(item))
    elif (
        isinstance(item, (int, float)) and items and isinstance(items[-1], (int, float))
    ):
        items[-1] = float(items[-1]) + float(item)
    else:
        items.append(item)


def _replacement_tj_items(
    operand: pikepdf.Object,
    font_obj: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    state: _ContentState,
    *,
    is_cid: bool,
) -> list | None:
    """Replace .notdef codes with equivalent TJ cursor adjustments."""
    try:
        raw = bytes(operand)
    except Exception:
        return None

    if is_cid:
        ranges = notdef_codes.code_space_ranges or ((b"\x00\x00", b"\xff\xff"),)
        encoded_codes = split_cmap_codes(raw, ranges)
    else:
        encoded_codes = [bytes([value]) for value in raw]

    items: list = []
    changed = False
    consumed = 0
    for encoded in encoded_codes:
        consumed += len(encoded)
        if encoded not in notdef_codes:
            _append_tj_item(items, String(encoded))
            continue

        width = _code_advance_width(font_obj, encoded, notdef_codes, state)
        if width is None:
            return None
        spacing = state.char_spacing
        if not is_cid and encoded == b"\x20":
            spacing += state.word_spacing
        if spacing and state.font_size == 0:
            return None
        adjustment = -width
        if spacing:
            adjustment -= 1000.0 * spacing / state.font_size
        _append_tj_item(items, adjustment)
        changed = True

    if consumed < len(raw):
        if not is_cid:
            _append_tj_item(items, String(raw[consumed:]))
        else:
            width = _cid_notdef_width(font_obj)
            if width is None:
                return None
            adjustment = -width
            if state.char_spacing:
                if state.font_size == 0:
                    return None
                adjustment -= 1000.0 * state.char_spacing / state.font_size
            _append_tj_item(items, adjustment)
            changed = True
    return items if changed else None


def _operand_uses_notdef(
    operand: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    *,
    is_cid: bool,
) -> bool:
    """Return whether a string operand contains a .notdef character."""
    try:
        raw = bytes(operand)
    except Exception:
        return False
    if is_cid:
        ranges = notdef_codes.code_space_ranges or ((b"\x00\x00", b"\xff\xff"),)
        return any(code in notdef_codes for code in split_cmap_codes(raw, ranges))
    return any(bytes([value]) in notdef_codes for value in raw)


def _fix_notdef_in_stream(
    stream_obj: Stream,
    font_map: dict[str, pikepdf.Object],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
    state: _ContentState | None = None,
) -> int:
    """Parses a content stream and removes .notdef references from text ops.

    Args:
        stream_obj: A pikepdf Stream whose content may contain text operators.
        font_map: Mapping of font resource names to font dictionaries.
        notdef_cache: Shared cache for notdef code computation.

    Returns:
        Number of text operators modified.
    """
    try:
        instructions = list(pikepdf.parse_content_stream(stream_obj))
    except Exception:
        return 0

    state = state or _ContentState()
    fixed = 0
    new_instructions = []
    current_font_name = state.current_font_name
    font_size = state.font_size
    char_spacing = state.char_spacing
    word_spacing = state.word_spacing
    text_rendering_mode = state.text_rendering_mode
    graphics_state_stack = state.graphics_state_stack

    for item in instructions:
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            new_instructions.append(item)
            continue

        operands, operator = item.operands, item.operator
        op_str = str(operator)

        if op_str == _SAVE_GRAPHICS_STATE:
            graphics_state_stack.append(
                (
                    current_font_name,
                    font_size,
                    char_spacing,
                    word_spacing,
                    text_rendering_mode,
                )
            )
            new_instructions.append(item)
            continue

        if op_str == _RESTORE_GRAPHICS_STATE:
            if graphics_state_stack:
                (
                    current_font_name,
                    font_size,
                    char_spacing,
                    word_spacing,
                    text_rendering_mode,
                ) = graphics_state_stack.pop()
            else:
                current_font_name = None
                font_size = 0.0
                char_spacing = 0.0
                word_spacing = 0.0
                text_rendering_mode = 0
            new_instructions.append(item)
            continue

        # Track font changes via Tf operator
        if op_str == "Tf" and len(operands) >= 1:
            try:
                current_font_name = str(operands[0])
            except Exception:
                current_font_name = None
            if len(operands) >= 2:
                try:
                    font_size = float(operands[1])
                except (TypeError, ValueError):
                    font_size = 0.0
            new_instructions.append(item)
            continue

        if op_str == "Tc" and operands:
            try:
                char_spacing = float(operands[0])
            except (TypeError, ValueError):
                pass
            new_instructions.append(item)
            continue

        if op_str == "Tw" and operands:
            try:
                word_spacing = float(operands[0])
            except (TypeError, ValueError):
                pass
            new_instructions.append(item)
            continue

        if op_str == "Tr" and operands:
            try:
                text_rendering_mode = int(operands[0])
            except (TypeError, ValueError):
                pass
            new_instructions.append(item)
            continue

        # Handle single-string text operators: Tj, ', "
        if op_str in _TEXT_OPERATORS and current_font_name is not None:
            if op_str == '"' and len(operands) >= 2:
                try:
                    word_spacing = float(operands[0])
                    char_spacing = float(operands[1])
                except (TypeError, ValueError):
                    pass
            font_obj = font_map.get(current_font_name)
            if font_obj is not None:
                notdef_codes = _get_notdef_codes(font_obj, notdef_cache)
                is_cid = _is_cidfont(font_obj)
                string_index = 2 if op_str == '"' else 0
                if (
                    notdef_codes
                    and text_rendering_mode == 3
                    and len(operands) > string_index
                    and isinstance(operands[string_index], String)
                    and _operand_uses_notdef(
                        operands[string_index],
                        notdef_codes,
                        is_cid=is_cid,
                    )
                ):
                    if not is_cid:
                        raise ConversionError(
                            "Invisible simple-font OCR text references .notdef"
                        )
                    state.font_size = font_size
                    state.char_spacing = char_spacing
                    state.word_spacing = word_spacing
                    replacement = _fix_invisible_single_string_op(
                        operands,
                        operator,
                        op_str,
                        font_obj,
                        notdef_codes,
                        state,
                    )
                    if replacement is None:
                        raise ConversionError(
                            "Invisible .notdef OCR text could not be remapped"
                        )
                    fixed += 1
                    new_instructions.extend(replacement)
                    continue
                if notdef_codes:
                    state.font_size = font_size
                    state.char_spacing = char_spacing
                    state.word_spacing = word_spacing
                    replacement = _fix_single_string_op(
                        operands,
                        op_str,
                        font_obj,
                        notdef_codes,
                        state,
                        is_cid=is_cid,
                    )
                    if replacement is not None:
                        fixed += 1
                        # Empty list → operator removed entirely
                        new_instructions.extend(replacement)
                        continue

            new_instructions.append(item)
            continue

        # Handle TJ (array of strings and adjustments)
        if op_str == "TJ" and current_font_name is not None:
            font_obj = font_map.get(current_font_name)
            if font_obj is not None:
                notdef_codes = _get_notdef_codes(font_obj, notdef_cache)
                is_cid = _is_cidfont(font_obj)
                if (
                    notdef_codes
                    and text_rendering_mode == 3
                    and operands
                    and isinstance(operands[0], Array)
                    and any(
                        isinstance(element, String)
                        and _operand_uses_notdef(
                            element,
                            notdef_codes,
                            is_cid=is_cid,
                        )
                        for element in operands[0]
                    )
                ):
                    if not is_cid:
                        raise ConversionError(
                            "Invisible simple-font OCR text references .notdef"
                        )
                    state.font_size = font_size
                    state.char_spacing = char_spacing
                    state.word_spacing = word_spacing
                    replacement_tj = _fix_invisible_tj_array_op(
                        operands,
                        operator,
                        font_obj,
                        notdef_codes,
                        state,
                    )
                    if replacement_tj is None:
                        raise ConversionError(
                            "Invisible .notdef OCR text could not be remapped"
                        )
                    fixed += 1
                    new_instructions.extend(replacement_tj)
                    continue
                if notdef_codes:
                    state.font_size = font_size
                    state.char_spacing = char_spacing
                    state.word_spacing = word_spacing
                    replacement_tj = _fix_tj_array_op(
                        operands,
                        operator,
                        font_obj,
                        notdef_codes,
                        state,
                        is_cid=is_cid,
                    )
                    if replacement_tj is not None:
                        fixed += 1
                        # Empty list → operator removed entirely
                        new_instructions.extend(replacement_tj)
                        continue

            new_instructions.append(item)
            continue

        new_instructions.append(item)

    if fixed > 0:
        stream_obj.write(pikepdf.unparse_content_stream(new_instructions))

    state.current_font_name = current_font_name
    state.font_size = font_size
    state.char_spacing = char_spacing
    state.word_spacing = word_spacing
    state.text_rendering_mode = text_rendering_mode
    return fixed


def _fix_invisible_single_string_op(
    operands: list,
    operator: pikepdf.Operator,
    op_str: str,
    font_obj: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    state: _ContentState,
) -> list[pikepdf.ContentStreamInstruction] | None:
    """Remap invisible CID text without removing its searchable content."""
    string_index = 2 if op_str == '"' else 0
    if len(operands) <= string_index or not isinstance(operands[string_index], String):
        return None
    items = _remap_invisible_tj_items(
        operands[string_index],
        font_obj,
        notdef_codes,
        state,
    )
    if items is None:
        return None

    if op_str == "Tj" and len(items) == 1 and isinstance(items[0], String):
        replacement_operands = list(operands)
        replacement_operands[string_index] = items[0]
        return [pikepdf.ContentStreamInstruction(replacement_operands, operator)]

    replacement: list[pikepdf.ContentStreamInstruction] = []
    if op_str == "'":
        replacement.append(pikepdf.ContentStreamInstruction([], Operator("T*")))
    elif op_str == '"':
        replacement.extend(
            [
                pikepdf.ContentStreamInstruction([operands[0]], Operator("Tw")),
                pikepdf.ContentStreamInstruction([operands[1]], Operator("Tc")),
                pikepdf.ContentStreamInstruction([], Operator("T*")),
            ]
        )
    replacement.append(pikepdf.ContentStreamInstruction([Array(items)], Operator("TJ")))
    return replacement


def _fix_invisible_tj_array_op(
    operands: list,
    operator: pikepdf.Operator,
    font_obj: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    state: _ContentState,
) -> list[pikepdf.ContentStreamInstruction] | None:
    """Remap invisible .notdef strings inside a TJ array."""
    if not operands or not isinstance(operands[0], Array):
        return None

    changed = False
    items: list = []
    for value in operands[0]:
        if not isinstance(value, String):
            _append_tj_item(items, value)
            continue
        replacement = _remap_invisible_tj_items(
            value,
            font_obj,
            notdef_codes,
            state,
        )
        if replacement is None:
            _append_tj_item(items, value)
            continue
        changed = True
        for item in replacement:
            _append_tj_item(items, item)

    if not changed:
        return None
    return [pikepdf.ContentStreamInstruction([Array(items)], operator)]


def _fix_single_string_op(
    operands: list,
    op_str: str,
    font_obj: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    state: _ContentState,
    *,
    is_cid: bool = False,
) -> list[pikepdf.ContentStreamInstruction] | None:
    """Filters .notdef codes from a single-string text operator.

    Args:
        operands: The instruction operands.
        operator: The operator.
        op_str: String form of the operator.
        notdef_codes: Set of .notdef byte values.
        is_cid: If True, treat operands as 2-byte CID pairs.

    Returns:
        - None if no change needed
        - A list of replacement instructions otherwise (empty if the
          operator can be dropped without side effects)
    """
    if not operands:
        return None

    # For " operator: operands are [aw, ac, string]
    if op_str == '"':
        if len(operands) < 3:
            return None
        string_idx = 2
    else:
        string_idx = 0

    operand = operands[string_idx]
    if not isinstance(operand, String):
        return None

    replacement_items = _replacement_tj_items(
        operand,
        font_obj,
        notdef_codes,
        state,
        is_cid=is_cid,
    )
    if replacement_items is None:
        return None

    replacement: list[pikepdf.ContentStreamInstruction] = []
    if op_str == "'":
        replacement.append(pikepdf.ContentStreamInstruction([], Operator("T*")))
    elif op_str == '"':
        replacement.extend(
            [
                pikepdf.ContentStreamInstruction([operands[0]], Operator("Tw")),
                pikepdf.ContentStreamInstruction([operands[1]], Operator("Tc")),
                pikepdf.ContentStreamInstruction([], Operator("T*")),
            ]
        )
    for replacement_item in replacement_items:
        if isinstance(replacement_item, String):
            replacement.append(
                pikepdf.ContentStreamInstruction(
                    [replacement_item],
                    Operator("Tj"),
                )
            )
        else:
            replacement.append(
                pikepdf.ContentStreamInstruction(
                    [Array([replacement_item])],
                    Operator("TJ"),
                )
            )
    return replacement


def _fix_tj_array_op(
    operands: list,
    operator: pikepdf.Operator,
    font_obj: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    state: _ContentState,
    *,
    is_cid: bool = False,
) -> list[pikepdf.ContentStreamInstruction] | None:
    """Filters .notdef codes from a TJ array operator.

    Args:
        operands: The instruction operands (should contain an Array).
        operator: The TJ operator.
        notdef_codes: Set of .notdef byte values.
        is_cid: If True, treat string elements as 2-byte CID pairs.

    Returns:
        - None if no change needed
        - A list with the filtered instruction otherwise (empty if all
          strings became empty and the operator is removed)
    """
    if not operands or not isinstance(operands[0], Array):
        return None

    arr = operands[0]
    changed = False
    new_items = []

    for elem in arr:
        if isinstance(elem, String):
            replacement_items = _replacement_tj_items(
                elem,
                font_obj,
                notdef_codes,
                state,
                is_cid=is_cid,
            )
            if replacement_items is not None:
                changed = True
                for replacement_item in replacement_items:
                    _append_tj_item(new_items, replacement_item)
            else:
                new_items.append(elem)
        else:
            # Numeric adjustment — keep it
            new_items.append(elem)

    if not changed:
        return None

    new_arr = Array(new_items)
    return [pikepdf.ContentStreamInstruction([new_arr], operator)]


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------


def _fix_notdef_in_page_contents(
    page_dict: Dictionary,
    font_map: dict[str, pikepdf.Object],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
    ambiguous_streams: set[object] | None = None,
) -> int:
    """Fixes .notdef references in page Contents.

    Args:
        page_dict: A resolved page dictionary.
        font_map: Font name to font dict mapping.
        notdef_cache: Shared notdef code cache.

    Returns:
        Number of text operators fixed.
    """
    contents = page_dict.get("/Contents")
    if contents is None:
        return 0

    contents = _resolve(contents)
    fixed = 0
    ambiguous_streams = ambiguous_streams or set()

    if isinstance(contents, Stream):
        objgen = contents.objgen
        if objgen == (0, 0) or objgen not in ambiguous_streams:
            fixed += _fix_notdef_in_stream(contents, font_map, notdef_cache)
    elif isinstance(contents, Array):
        state = _ContentState()
        for item in contents:
            item = _resolve(item)
            if isinstance(item, Stream):
                objgen = item.objgen
                if objgen != (0, 0) and objgen in ambiguous_streams:
                    continue
                fixed += _fix_notdef_in_stream(
                    item,
                    font_map,
                    notdef_cache,
                    state,
                )

    return fixed
