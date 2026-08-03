# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Wrap PUA-mapped characters in /ActualText marked-content sequences.

ISO 19005 Rule 6.2.11.7.3-1 requires that any character mapped to a
Unicode Private Use Area (PUA) code point be wrapped in a marked-content
sequence with /ActualText.  This module inserts the required BDC/EMC
wrappers around text-showing operators whose character codes resolve
to PUA via the font's ToUnicode CMap.
"""

import logging
from dataclasses import dataclass, field
from io import BytesIO

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream, String

from ..fonts.cid_unicode import get_cid_to_unicode
from ..fonts.encodings import SYMBOL_ENCODING
from ..fonts.glyph_mapping import ZAPFDINGBATS_GLYPH_TO_UNICODE
from ..fonts.glyph_usage import _iter_content_streams_with_resources
from ..fonts.subsetter import _get_unicode_cmap, _resolve_simple_font_encoding
from ..fonts.tounicode import (
    CIDEncodingMap,
    get_font_code_space_ranges,
    get_type0_cid_encoding_map,
    parse_cidtogidmap_stream,
    parse_tounicode_cmap_sequences,
    resolve_glyph_to_unicode,
    resolve_symbol_glyph_to_unicode,
    split_cmap_codes,
)
from ..fonts.utils import (
    get_truetype_byte_encoding,
    safe_str,
    symbol_cmap_code_to_byte,
)
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Text-showing operators whose string operands may reference PUA
_TEXT_OPERATORS = frozenset({"Tj", "'", '"'})

_CodeSpaceRanges = tuple[tuple[bytes, bytes], ...]
_ToUnicodeMap = dict[bytes, tuple[int, ...]]
_ToUnicodeInfo = tuple[_ToUnicodeMap, _CodeSpaceRanges]
_ObjectKey = tuple[int, int]
_Type0Resolver = tuple[CIDEncodingMap, dict[int, int] | None, dict[int, int]]
_SimpleSymbolResolver = dict[int, int]
_StructuralActualTextReferences = frozenset[tuple[_ObjectKey, int]]

# Unicode WG2 N4363 mappings for Wingdings slots observed in office documents.
_WINGDINGS_UNICODE = {
    0x28: 0x1F57F,
    0x2A: 0x1F582,
    0x6E: 0x25FC,
}


@dataclass
class _ContentState:
    current_font_name: str | None = None
    font_stack: list[str | None] = field(default_factory=list)
    actualtext_stack: list[bool] = field(default_factory=list)
    actualtext_depth: int = 0


# ---------------------------------------------------------------------------
# PUA detection
# ---------------------------------------------------------------------------


def _is_pua(code_point: int) -> bool:
    """Check if a Unicode code point falls in any Private Use Area range.

    PUA ranges: U+E000..U+F8FF (BMP), U+F0000..U+FFFFD (Supplementary A),
    U+100000..U+10FFFD (Supplementary B).
    """
    return (
        0xE000 <= code_point <= 0xF8FF
        or 0xF0000 <= code_point <= 0xFFFFD
        or 0x100000 <= code_point <= 0x10FFFD
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sanitize_pua_actualtext(pdf: Pdf) -> dict[str, int]:
    """Wraps PUA-mapped characters in /ActualText marked-content sequences.

    Scans all content streams for text-showing operators whose character
    codes resolve to PUA Unicode values via the font's ToUnicode CMap,
    and wraps them in ``/Span <</ActualText ...>> BDC ... EMC``.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"pua_actualtext_added": N,
        "pua_actualtext_warnings": N}``.
    """
    total_added = 0
    total_warnings = 0
    tounicode_cache: dict[tuple[int, int], _ToUnicodeInfo] = {}
    encoding_cache: dict[tuple[int, int], dict[int, str] | None] = {}
    type0_cache: dict[_ObjectKey, _Type0Resolver | None] = {}
    simple_symbol_cache: dict[_ObjectKey, _SimpleSymbolResolver] = {}
    from ..tagging import get_structural_actualtext_references

    structural_actualtext_references = get_structural_actualtext_references(pdf)

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            for owner, resources in _iter_content_streams_with_resources(page):
                font_map = _build_font_map(resources)
                if isinstance(owner, Stream):
                    added, warnings = _fix_pua_in_stream(
                        owner,
                        font_map,
                        tounicode_cache,
                        encoding_cache,
                        type0_cache,
                        simple_symbol_cache,
                        resources=resources,
                        allow_unset_font=True,
                        structural_actualtext_references=(
                            structural_actualtext_references
                        ),
                    )
                else:
                    added, warnings = _fix_pua_in_page_contents(
                        owner,
                        font_map,
                        tounicode_cache,
                        encoding_cache,
                        type0_cache,
                        simple_symbol_cache,
                        resources,
                        structural_actualtext_references,
                    )
                total_added += added
                total_warnings += warnings

        except Exception as e:
            log_suppressed_error(
                logger,
                e,
                "Error fixing PUA ActualText on page %d: %s",
                page_num,
                e,
            )

    if total_added > 0:
        logger.info("PUA ActualText: %d text operators wrapped", total_added)
    if total_warnings > 0:
        logger.warning(
            "PUA ActualText: %d unresolvable PUA characters",
            total_warnings,
        )

    return {
        "pua_actualtext_added": total_added,
        "pua_actualtext_warnings": total_warnings,
    }


# ---------------------------------------------------------------------------
# Font map building
# ---------------------------------------------------------------------------


def _build_font_map(
    resources: pikepdf.Object,
) -> dict[str, pikepdf.Object]:
    """Builds a mapping from font resource name to font dictionary."""
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


def _is_cidfont(font_obj: pikepdf.Object) -> bool:
    """Checks if a font is a CIDFont (Type0)."""
    try:
        subtype = font_obj.get("/Subtype")
        if subtype is not None and str(subtype) == "/Type0":
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# ToUnicode and encoding helpers
# ---------------------------------------------------------------------------


def _get_tounicode_info(
    font_obj: pikepdf.Object,
    cache: dict[tuple[int, int], _ToUnicodeInfo],
) -> _ToUnicodeInfo:
    """Return a font's width-preserving ToUnicode data, cached by objgen."""
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        return {}, ()

    objgen = font_obj.objgen
    if objgen != (0, 0) and objgen in cache:
        return cache[objgen]

    tounicode = font_obj.get("/ToUnicode")
    if tounicode is None:
        data = None
        mapping: _ToUnicodeMap = {}
    else:
        tounicode = _resolve(tounicode)
        try:
            data = bytes(tounicode.read_bytes())
            mapping = parse_tounicode_cmap_sequences(data)
        except Exception:
            data = None
            mapping = {}
    result = mapping, get_font_code_space_ranges(font_obj, data)

    if objgen != (0, 0):
        cache[objgen] = result
    return result


def _resolve_pua_to_actual_unicode(
    code_bytes: bytes,
    pua_value: int,
    font_obj: pikepdf.Object,
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
    type0_cache: dict[_ObjectKey, _Type0Resolver | None],
    simple_symbol_cache: dict[_ObjectKey, _SimpleSymbolResolver],
) -> int | None:
    """Try to find real Unicode for a PUA character code via encoding.

    Simple fonts resolve code -> glyph name -> Unicode via AGL. Type 0 fonts
    resolve character code -> CID -> GID -> embedded-font Unicode.
    """
    if _is_cidfont(font_obj):
        resolved = _resolve_type0_unicode(code_bytes, font_obj, type0_cache)
        if resolved is not None:
            return resolved
        return _resolve_named_type0_pua(font_obj, pua_value)

    font_obj = _resolve(font_obj)
    objgen = font_obj.objgen
    code = int.from_bytes(code_bytes, "big")

    if objgen != (0, 0) and objgen in encoding_cache:
        encoding = encoding_cache[objgen]
    else:
        encoding = _resolve_simple_font_encoding(font_obj)
        if objgen != (0, 0):
            encoding_cache[objgen] = encoding

    if encoding is not None:
        glyph_name = encoding.get(code)
        if glyph_name is not None:
            unicode_val = resolve_glyph_to_unicode(glyph_name)
            if unicode_val is not None and not _is_pua(unicode_val):
                return unicode_val
    if len(code_bytes) == 1:
        return _resolve_windows_symbol_code(
            font_obj,
            code_bytes[0],
            simple_symbol_cache,
        )
    return None


def _object_key(obj: pikepdf.Object) -> _ObjectKey | None:
    """Return a stable cache key for an indirect PDF object."""
    objgen = obj.objgen
    return objgen if objgen != (0, 0) else None


def _font_names(font_obj: pikepdf.Object) -> list[str]:
    """Collect normalized font names used for narrow font fallbacks."""
    names: list[str] = []

    def add_name(value: object) -> None:
        if value is None:
            return
        normalized = str(value).lstrip("/").split("+")[-1].lower()
        normalized = normalized.replace(" ", "").replace("-", "").replace("_", "")
        if normalized:
            names.append(normalized)

    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        return names
    add_name(font_obj.get("/BaseFont"))

    descendants = _resolve(font_obj.get("/DescendantFonts"))
    descendant = (
        _resolve(descendants[0])
        if isinstance(descendants, Array) and descendants
        else None
    )
    if isinstance(descendant, Dictionary):
        add_name(descendant.get("/BaseFont"))
        descriptor = _resolve(descendant.get("/FontDescriptor"))
    else:
        descriptor = _resolve(font_obj.get("/FontDescriptor"))
    if isinstance(descriptor, Dictionary):
        add_name(descriptor.get("/FontName"))
    return names


def _resolve_named_type0_pua(
    font_obj: pikepdf.Object,
    pua_value: int,
) -> int | None:
    """Resolve legacy Type 0 PUA conventions for known symbol font families."""
    names = _font_names(font_obj)
    slot = symbol_cmap_code_to_byte(pua_value)
    if slot is None:
        return None
    if 0x20 <= slot <= 0x7E and any(
        "code39" in name or "barcode39" in name for name in names
    ):
        return slot

    if any("wingdings" in name for name in names):
        return _WINGDINGS_UNICODE.get(slot)

    if any(name in {"symbol", "symbolmt"} for name in names):
        glyph_name = SYMBOL_ENCODING.get(slot)
        if glyph_name is not None:
            unicode_val = resolve_symbol_glyph_to_unicode(glyph_name)
            if unicode_val is not None and not _is_pua(unicode_val):
                return unicode_val
    return None


def _build_simple_symbol_resolver(
    font_obj: pikepdf.Object,
) -> _SimpleSymbolResolver:
    """Resolve simple-font bytes from an authoritative program byte cmap."""
    names = _font_names(font_obj)
    font_obj = _resolve(font_obj)
    descriptor = (
        _resolve(font_obj.get("/FontDescriptor"))
        if isinstance(font_obj, Dictionary)
        else None
    )
    if not isinstance(descriptor, Dictionary):
        return {}
    font_file = descriptor.get("/FontFile2")
    if font_file is None:
        font_file = descriptor.get("/FontFile3")
    font_file = _resolve(font_file)
    if not isinstance(font_file, Stream):
        return {}

    try:
        from fontTools.ttLib import TTFont

        tt_font = TTFont(BytesIO(bytes(font_file.read_bytes())))
        try:
            byte_encoding = get_truetype_byte_encoding(tt_font)
            if byte_encoding is None:
                return {}
            byte_mapping = byte_encoding[2]
            unicode_cmap = _get_unicode_cmap(tt_font)
        finally:
            tt_font.close()
    except Exception:
        return {}

    result: _SimpleSymbolResolver = {}
    unicode_by_glyph: dict[str, set[int]] = {}
    for unicode_value, glyph_name in unicode_cmap.items():
        if not _is_pua(unicode_value):
            unicode_by_glyph.setdefault(glyph_name, set()).add(unicode_value)

    for code, glyph_name in byte_mapping.items():
        candidates = unicode_by_glyph.get(glyph_name, set())
        if len(candidates) == 1:
            result[code] = next(iter(candidates))
            continue

        unicode_value = resolve_symbol_glyph_to_unicode(glyph_name)
        if (
            unicode_value is not None
            and not _is_pua(unicode_value)
            and (not candidates or unicode_value in candidates)
        ):
            result[code] = unicode_value
            continue

        zapf_value = ZAPFDINGBATS_GLYPH_TO_UNICODE.get(glyph_name)
        if zapf_value is not None and (not candidates or zapf_value in candidates):
            result[code] = zapf_value
            continue

        if candidates:
            continue

        if any("wingdings" in name for name in names):
            unicode_value = _WINGDINGS_UNICODE.get(code)
        elif any(name in {"symbol", "symbolmt"} for name in names):
            encoded_name = SYMBOL_ENCODING.get(code)
            unicode_value = (
                resolve_symbol_glyph_to_unicode(encoded_name)
                if encoded_name is not None
                else None
            )
        elif any("code39" in name or "barcode39" in name for name in names):
            unicode_value = code if 0x20 <= code <= 0x7E else None
        else:
            unicode_value = None
        if unicode_value is not None and not _is_pua(unicode_value):
            result[code] = unicode_value
    return result


def _resolve_windows_symbol_code(
    font_obj: pikepdf.Object,
    code: int,
    cache: dict[_ObjectKey, _SimpleSymbolResolver],
) -> int | None:
    """Resolve a simple-font byte through its authoritative program cmap."""
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        return None
    key = _object_key(font_obj)
    if key is None:
        return _build_simple_symbol_resolver(font_obj).get(code)
    if key not in cache:
        cache[key] = _build_simple_symbol_resolver(font_obj)
    return cache[key].get(code)


def _build_type0_resolver(
    font_obj: pikepdf.Object,
) -> _Type0Resolver | None:
    """Build the data needed to resolve Type 0 character codes to Unicode."""
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        return None

    encoding = get_type0_cid_encoding_map(font_obj)
    if encoding is None:
        return None

    descendants = _resolve(font_obj.get("/DescendantFonts"))
    if not isinstance(descendants, Array) or not descendants:
        return None
    descendant = _resolve(descendants[0])
    if not isinstance(descendant, Dictionary):
        return None
    descendant_subtype = str(descendant.get("/Subtype") or "")

    cid_to_gid: dict[int, int] | None
    if descendant_subtype == "/CIDFontType0":
        # CID-keyed CFF fonts select glyphs through their CFF charset. A
        # CIDToGIDMap entry has no meaning for this subtype.
        cid_to_gid = None
    elif descendant_subtype == "/CIDFontType2":
        cid_to_gid_obj = descendant.get("/CIDToGIDMap")
        if cid_to_gid_obj is None:
            cid_to_gid = None
        else:
            cid_to_gid_obj = _resolve(cid_to_gid_obj)
            if isinstance(cid_to_gid_obj, Name):
                if str(cid_to_gid_obj) != "/Identity":
                    return None
                cid_to_gid = None
            elif isinstance(cid_to_gid_obj, Stream):
                try:
                    cid_to_gid = parse_cidtogidmap_stream(
                        bytes(cid_to_gid_obj.read_bytes())
                    )
                except Exception:
                    return None
            else:
                return None
    else:
        return None

    descriptor = _resolve(descendant.get("/FontDescriptor"))
    if not isinstance(descriptor, Dictionary):
        return None
    font_file = descriptor.get("/FontFile2")
    if font_file is None:
        font_file = descriptor.get("/FontFile3")
    font_file = _resolve(font_file)
    if not isinstance(font_file, Stream):
        return None

    try:
        from fontTools.ttLib import TTFont

        font_data = bytes(font_file.read_bytes())
        try:
            tt_font = TTFont(BytesIO(font_data))
        except Exception:
            if descendant_subtype != "/CIDFontType0":
                raise
            from .glyph_coverage import _wrap_cff_in_otf

            tt_font = TTFont(BytesIO(_wrap_cff_in_otf(font_data)))
        try:
            glyph_order = tt_font.getGlyphOrder()
            glyph_to_gid = {name: gid for gid, name in enumerate(glyph_order)}
            unicode_by_gid: dict[int, set[int]] = {}
            for unicode_value, glyph_name in sorted(_get_unicode_cmap(tt_font).items()):
                if _is_pua(unicode_value):
                    continue
                gid = glyph_to_gid.get(glyph_name)
                if gid is not None:
                    unicode_by_gid.setdefault(gid, set()).add(unicode_value)

            if descendant_subtype == "/CIDFontType0":
                cid_to_gid = {}
                for gid, glyph_name in enumerate(glyph_order):
                    if glyph_name == ".notdef":
                        cid_to_gid[0] = gid
                    elif glyph_name.startswith("cid"):
                        try:
                            cid_to_gid[int(glyph_name[3:])] = gid
                        except ValueError:
                            continue
                if not cid_to_gid:
                    return None

                cid_system_info = _resolve(descendant.get("/CIDSystemInfo"))
                if isinstance(cid_system_info, Dictionary):
                    ordering_obj = cid_system_info.get("/Ordering")
                    ordering = (
                        safe_str(ordering_obj).lstrip("/")
                        if ordering_obj is not None
                        else ""
                    )
                    collection_mapping = get_cid_to_unicode(ordering)
                    if collection_mapping:
                        for cid, gid in cid_to_gid.items():
                            unicode_value = collection_mapping.get(cid)
                            if unicode_value is not None and not _is_pua(unicode_value):
                                unicode_by_gid.setdefault(gid, set()).add(unicode_value)

            gid_to_unicode: dict[int, int] = {}
            for gid, glyph_name in enumerate(glyph_order):
                candidates = unicode_by_gid.get(gid, set())
                if len(candidates) == 1:
                    gid_to_unicode[gid] = next(iter(candidates))
                    continue
                unicode_value = resolve_symbol_glyph_to_unicode(glyph_name)
                if (
                    unicode_value is not None
                    and not _is_pua(unicode_value)
                    and (not candidates or unicode_value in candidates)
                ):
                    gid_to_unicode[gid] = unicode_value
        finally:
            tt_font.close()
    except Exception:
        return None

    return encoding, cid_to_gid, gid_to_unicode


def _resolve_type0_unicode(
    code_bytes: bytes,
    font_obj: pikepdf.Object,
    cache: dict[_ObjectKey, _Type0Resolver | None],
) -> int | None:
    """Resolve one encoded Type 0 character through its embedded font."""
    font_obj = _resolve(font_obj)
    if not isinstance(font_obj, Dictionary):
        return None
    key = _object_key(font_obj)
    if key is None:
        resolver = _build_type0_resolver(font_obj)
    else:
        if key not in cache:
            cache[key] = _build_type0_resolver(font_obj)
        resolver = cache[key]
    if resolver is None:
        return None

    encoding, cid_to_gid, gid_to_unicode = resolver
    cid = encoding.map_code(code_bytes)
    gid = cid if cid_to_gid is None else cid_to_gid.get(cid, 0)
    return gid_to_unicode.get(gid)


# ---------------------------------------------------------------------------
# ActualText construction
# ---------------------------------------------------------------------------


def _encode_actualtext(text: str) -> bytes:
    """Encode a string as UTF-16BE with BOM for PDF ActualText."""
    return b"\xfe\xff" + text.encode("utf-16-be")


def _has_pua_codes(
    raw: bytes,
    tounicode: _ToUnicodeMap,
    code_space_ranges: _CodeSpaceRanges,
) -> bool:
    """Check if any character codes in raw bytes map to PUA."""
    return any(
        any(_is_pua(value) for value in tounicode.get(code, ()))
        for code in split_cmap_codes(raw, code_space_ranges)
    )


def _extract_text_bytes(op_str: str, operands: list) -> bytes | None:
    """Extract raw text bytes from a text operator's operands."""
    if op_str == "TJ":
        if not operands or not isinstance(operands[0], Array):
            return None
        parts = bytearray()
        for elem in operands[0]:
            if isinstance(elem, String):
                parts.extend(bytes(elem))
        return bytes(parts) if parts else None
    elif op_str == '"':
        if len(operands) >= 3 and isinstance(operands[2], String):
            return bytes(operands[2])
        return None
    else:
        # Tj or '
        if operands and isinstance(operands[0], String):
            return bytes(operands[0])
        return None


def _get_bdc_properties(
    operands: list,
    resources: pikepdf.Object | None,
) -> Dictionary | None:
    """Resolve a BDC property dictionary, including named resources."""
    if len(operands) < 2:
        return None
    properties = _resolve(operands[1])
    if isinstance(properties, Name):
        resources = _resolve(resources)
        if not isinstance(resources, Dictionary):
            return None
        property_dict = _resolve(resources.get("/Properties"))
        if not isinstance(property_dict, Dictionary):
            return None
        properties = _resolve(property_dict.get(str(properties)))
    if not isinstance(properties, Dictionary):
        return None
    return properties


def _has_actualtext_property(
    operands: list,
    resources: pikepdf.Object | None,
) -> bool:
    """Return whether BDC operands resolve to an ActualText string."""
    properties = _get_bdc_properties(operands, resources)
    if properties is None:
        return False
    try:
        return isinstance(_resolve(properties.get("/ActualText")), String)
    except Exception:
        return False


def _get_mcid_property(
    operands: list,
    resources: pikepdf.Object | None,
) -> int | None:
    """Return a BDC marked-content identifier when it is well formed."""
    properties = _get_bdc_properties(operands, resources)
    if properties is None:
        return None
    mcid = _resolve(properties.get("/MCID"))
    if isinstance(mcid, int) and not isinstance(mcid, bool) and mcid >= 0:
        return mcid
    return None


def _build_actualtext_value(
    raw: bytes,
    tounicode: _ToUnicodeMap,
    code_space_ranges: _CodeSpaceRanges,
    font_obj: pikepdf.Object,
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
    type0_cache: dict[_ObjectKey, _Type0Resolver | None],
    simple_symbol_cache: dict[_ObjectKey, _SimpleSymbolResolver],
) -> tuple[str | None, int]:
    """Build the ActualText string for a text operand.

    For each character code: if non-PUA Unicode, use it; if PUA, try
    to resolve via the font program. If a PUA value has no authoritative
    replacement, preserve that value in ActualText instead of inventing text
    or changing the document's existing extraction semantics. A character
    code without any Unicode mapping still prevents a safe replacement.

    Returns:
        Tuple of (actualtext_string, num_unresolvable_pua).
    """
    chars: list[str] = []
    warnings = 0

    for code_bytes in split_cmap_codes(raw, code_space_ranges):
        unicode_values = tounicode.get(code_bytes)
        if not unicode_values:
            warnings += 1
            continue
        for unicode_val in unicode_values:
            if _is_pua(unicode_val):
                resolved = _resolve_pua_to_actual_unicode(
                    code_bytes,
                    unicode_val,
                    font_obj,
                    encoding_cache,
                    type0_cache,
                    simple_symbol_cache,
                )
                if resolved is not None:
                    chars.append(chr(resolved))
                else:
                    chars.append(chr(unicode_val))
            else:
                chars.append(chr(unicode_val))

    if warnings or not chars:
        return None, max(warnings, 1)
    return "".join(chars), 0


# ---------------------------------------------------------------------------
# Content stream fixing
# ---------------------------------------------------------------------------


def _fix_pua_in_stream(
    stream_obj: Stream,
    font_map: dict[str, pikepdf.Object],
    tounicode_cache: dict[tuple[int, int], _ToUnicodeInfo],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
    type0_cache: dict[_ObjectKey, _Type0Resolver | None],
    simple_symbol_cache: dict[_ObjectKey, _SimpleSymbolResolver],
    state: _ContentState | None = None,
    resources: pikepdf.Object | None = None,
    allow_unset_font: bool = False,
    structural_actualtext_references: _StructuralActualTextReferences = frozenset(),
    structural_container_key: _ObjectKey | None = None,
) -> tuple[int, int]:
    """Core stream processor.

    Parses a content stream, identifies text operators with PUA codes,
    and wraps them in BDC /Span <</ActualText ...>> ... EMC.

    Returns:
        Tuple of (wrapped_count, warning_count).
    """
    try:
        instructions = list(pikepdf.parse_content_stream(stream_obj))
    except Exception:
        return 0, 0

    if state is None:
        state = _ContentState()
    container_key = structural_container_key or _object_key(stream_obj)
    new_instructions: list = []
    wrapped_count = 0
    warning_count = 0

    for item in instructions:
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            new_instructions.append(item)
            continue

        operands, operator = item.operands, item.operator
        op_str = str(operator)

        if op_str == "BDC":
            has_actualtext = _has_actualtext_property(operands, resources)
            if not has_actualtext:
                mcid = _get_mcid_property(operands, resources)
                has_actualtext = (
                    container_key is not None
                    and mcid is not None
                    and (container_key, mcid) in structural_actualtext_references
                )
            state.actualtext_stack.append(has_actualtext)
            state.actualtext_depth += int(has_actualtext)
            new_instructions.append(item)
            continue
        if op_str == "BMC":
            state.actualtext_stack.append(False)
            new_instructions.append(item)
            continue
        if op_str == "EMC":
            if state.actualtext_stack and state.actualtext_stack.pop():
                state.actualtext_depth -= 1
            new_instructions.append(item)
            continue
        if op_str == "q":
            state.font_stack.append(state.current_font_name)
            new_instructions.append(item)
            continue
        if op_str == "Q":
            state.current_font_name = (
                state.font_stack.pop() if state.font_stack else None
            )
            new_instructions.append(item)
            continue
        if op_str == "Tf" and len(operands) >= 1:
            try:
                state.current_font_name = str(operands[0])
            except Exception:
                state.current_font_name = None
            new_instructions.append(item)
            continue

        if op_str not in _TEXT_OPERATORS and op_str != "TJ":
            new_instructions.append(item)
            continue
        if state.actualtext_depth > 0:
            new_instructions.append(item)
            continue

        raw = _extract_text_bytes(op_str, operands)
        if raw is None:
            new_instructions.append(item)
            continue

        font_obj = (
            font_map.get(state.current_font_name)
            if state.current_font_name is not None
            else None
        )
        if font_obj is None and state.current_font_name is None and allow_unset_font:
            unique_fonts: list[pikepdf.Object] = []
            seen_indirect: set[_ObjectKey] = set()
            for candidate in font_map.values():
                key = _object_key(candidate)
                if key is not None:
                    if key in seen_indirect:
                        continue
                    seen_indirect.add(key)
                unique_fonts.append(candidate)
            if len(unique_fonts) == 1:
                font_obj = unique_fonts[0]
        if font_obj is None:
            for candidate in font_map.values():
                tounicode, code_space_ranges = _get_tounicode_info(
                    candidate,
                    tounicode_cache,
                )
                if tounicode and _has_pua_codes(raw, tounicode, code_space_ranges):
                    warning_count += 1
                    break
            new_instructions.append(item)
            continue

        tounicode, code_space_ranges = _get_tounicode_info(font_obj, tounicode_cache)
        if not tounicode:
            new_instructions.append(item)
            continue

        if not _has_pua_codes(raw, tounicode, code_space_ranges):
            new_instructions.append(item)
            continue

        text, warnings = _build_actualtext_value(
            raw,
            tounicode,
            code_space_ranges,
            font_obj,
            encoding_cache,
            type0_cache,
            simple_symbol_cache,
        )
        warning_count += warnings
        if text is None:
            new_instructions.append(item)
            continue

        actualtext_bytes = _encode_actualtext(text)
        props = Dictionary()
        props[Name("/ActualText")] = String(actualtext_bytes)
        bdc = pikepdf.ContentStreamInstruction(
            [Name("/Span"), props], pikepdf.Operator("BDC")
        )
        emc = pikepdf.ContentStreamInstruction([], pikepdf.Operator("EMC"))

        new_instructions.append(bdc)
        new_instructions.append(item)
        new_instructions.append(emc)
        wrapped_count += 1

    if wrapped_count > 0:
        stream_obj.write(pikepdf.unparse_content_stream(new_instructions))

    return wrapped_count, warning_count


def _fix_pua_in_page_contents(
    page_dict: Dictionary,
    font_map: dict[str, pikepdf.Object],
    tounicode_cache: dict[tuple[int, int], _ToUnicodeInfo],
    encoding_cache: dict[tuple[int, int], dict[int, str] | None],
    type0_cache: dict[_ObjectKey, _Type0Resolver | None],
    simple_symbol_cache: dict[_ObjectKey, _SimpleSymbolResolver],
    resources: pikepdf.Object,
    structural_actualtext_references: _StructuralActualTextReferences = frozenset(),
) -> tuple[int, int]:
    """Fixes PUA references in page Contents."""
    contents = page_dict.get("/Contents")
    if contents is None:
        return 0, 0

    contents = _resolve(contents)
    total_added = 0
    total_warnings = 0
    state = _ContentState()
    container_key = _object_key(page_dict)

    if isinstance(contents, Stream):
        a, w = _fix_pua_in_stream(
            contents,
            font_map,
            tounicode_cache,
            encoding_cache,
            type0_cache,
            simple_symbol_cache,
            state,
            resources,
            structural_actualtext_references=structural_actualtext_references,
            structural_container_key=container_key,
        )
        total_added += a
        total_warnings += w
    elif isinstance(contents, Array):
        for item in contents:
            item = _resolve(item)
            if isinstance(item, Stream):
                a, w = _fix_pua_in_stream(
                    item,
                    font_map,
                    tounicode_cache,
                    encoding_cache,
                    type0_cache,
                    simple_symbol_cache,
                    state,
                    resources,
                    structural_actualtext_references=(structural_actualtext_references),
                    structural_container_key=container_key,
                )
                total_added += a
                total_warnings += w

    return total_added, total_warnings
