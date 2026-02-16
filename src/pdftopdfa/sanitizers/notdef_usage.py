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
import struct

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream, String

from ..fonts.subsetter import (
    _resolve_simple_font_encoding,
)
from ..fonts.utils import safe_str as _safe_str
from ..utils import iter_type3_fonts as _iter_type3_fonts
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Text-showing operators whose string operands may reference .notdef
_TEXT_OPERATORS = frozenset({"Tj", "'", '"'})


class _NotdefCodes:
    """Set-like object for .notdef character codes.

    Supports both explicit codes and a max_valid_code threshold so that
    CIDFonts with Identity mapping can represent "every CID >= numGlyphs"
    without building a 65k-entry frozenset.
    """

    __slots__ = ("_explicit", "_max_valid_code")

    def __init__(
        self,
        explicit: frozenset[int] = frozenset(),
        max_valid_code: int | None = None,
    ) -> None:
        self._explicit = explicit
        self._max_valid_code = max_valid_code

    def __contains__(self, code: int) -> bool:
        return code in self._explicit or (
            self._max_valid_code is not None and code > self._max_valid_code
        )

    def __bool__(self) -> bool:
        return bool(self._explicit) or self._max_valid_code is not None


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
    visited: set[tuple[int, int]] = set()
    # Cache notdef codes per font objgen to avoid recomputation
    notdef_cache: dict[tuple[int, int], _NotdefCodes] = {}

    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve(page.obj)

            # Build font map from page resources
            resources = page_dict.get("/Resources")
            if resources is not None:
                resources = _resolve(resources)
            font_map = _build_font_map(resources) if resources else {}

            # 1. Page Contents
            total_fixed += _fix_notdef_in_page_contents(
                page_dict, font_map, notdef_cache
            )

            # 2. Form XObjects (recursive)
            if resources is not None:
                total_fixed += _fix_notdef_in_form_xobjects(
                    resources, visited, notdef_cache
                )

                # 3. Tiling Patterns (recursive)
                total_fixed += _fix_notdef_in_patterns(resources, visited, notdef_cache)

                # 4. Type3 CharProcs
                total_fixed += _fix_notdef_in_type3_charprocs(
                    resources, visited, notdef_cache
                )

            # 5. Annotation AP streams
            annots = page_dict.get("/Annots")
            if annots:
                annots = _resolve(annots)
                for annot_ref in annots:
                    annot = _resolve(annot_ref)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = annot.get("/AP")
                    if not ap:
                        continue
                    ap = _resolve(ap)
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry:
                            total_fixed += _fix_notdef_in_ap_stream(
                                ap_entry, visited, notdef_cache
                            )

        except Exception as e:
            logger.debug("Error fixing .notdef usage on page %d: %s", page_num, e)

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


def _find_missing_glyphs_in_simple_font(
    font_obj: pikepdf.Object, first_char: int, last_char: int
) -> set[int]:
    """Finds codes in [first_char, last_char] whose glyph is missing.

    Parses the embedded font program with fontTools, resolves the font's
    encoding to map codes to glyph names, and returns codes whose glyph
    name is absent from the font's glyph order.

    Args:
        font_obj: Resolved simple font dictionary.
        first_char: First valid character code.
        last_char: Last valid character code.

    Returns:
        Set of character codes whose encoded glyph is missing.
    """
    try:
        encoding = _resolve_simple_font_encoding(font_obj)
        if not encoding:
            return set()

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
            missing = set()
            for code in range(first_char, last_char + 1):
                name = encoding.get(code)
                if name is None or name == ".notdef":
                    # No encoding entry or explicit .notdef → maps to .notdef
                    missing.add(code)
                elif name not in glyph_set:
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

    For simple fonts (TrueType/Type1/MMType1), codes outside the
    [FirstChar, LastChar] range always map to .notdef.  Additionally,
    codes within the range whose encoded glyph name is absent from the
    embedded font program are also flagged.

    Args:
        font_obj: Resolved simple font dictionary.

    Returns:
        _NotdefCodes for byte values (0-255) that are .notdef.
    """
    try:
        first_char = int(font_obj.get("/FirstChar", 0))
    except (TypeError, ValueError):
        first_char = 0
    try:
        last_char = int(font_obj.get("/LastChar", 255))
    except (TypeError, ValueError):
        last_char = 255

    # Codes outside [FirstChar, LastChar] are always .notdef
    notdef = set(range(0, first_char)) | set(range(last_char + 1, 256))

    # Also check for codes within range whose glyph is missing
    notdef |= _find_missing_glyphs_in_simple_font(font_obj, first_char, last_char)

    return _NotdefCodes(frozenset(notdef))


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

    descendants = font_obj.get("/DescendantFonts")
    if descendants is None:
        return _NotdefCodes()
    descendants = _resolve(descendants)
    if not isinstance(descendants, Array) or len(descendants) == 0:
        return _NotdefCodes()

    cidfont = _resolve(descendants[0])
    if not isinstance(cidfont, Dictionary):
        return _NotdefCodes()

    num_glyphs = _get_cidfont_num_glyphs(cidfont)

    cidtogidmap = cidfont.get("/CIDToGIDMap")
    if cidtogidmap is None:
        # No mapping — CID 0 is .notdef by convention
        notdef.add(0)
        max_valid = (num_glyphs - 1) if num_glyphs is not None else None
        return _NotdefCodes(frozenset(notdef), max_valid)

    cidtogidmap = _resolve(cidtogidmap)

    if isinstance(cidtogidmap, Name) and str(cidtogidmap) == "/Identity":
        # Identity mapping: CID = GID, so CID 0 → GID 0 → .notdef
        # and CID >= numGlyphs → beyond font program
        notdef.add(0)
        max_valid = (num_glyphs - 1) if num_glyphs is not None else None
        return _NotdefCodes(frozenset(notdef), max_valid)
    elif isinstance(cidtogidmap, Stream):
        # Stream mapping: parse to find CIDs that map to GID 0
        # or to GID >= numGlyphs
        try:
            stream_data = bytes(cidtogidmap.read_bytes())
            num_entries = len(stream_data) // 2
            for cid in range(num_entries):
                gid = struct.unpack_from(">H", stream_data, cid * 2)[0]
                if gid == 0:
                    notdef.add(cid)
                elif num_glyphs is not None and gid >= num_glyphs:
                    notdef.add(cid)
        except Exception:
            # If we can't parse, conservatively only flag CID 0
            notdef.add(0)

    return _NotdefCodes(frozenset(notdef))


# ---------------------------------------------------------------------------
# Content stream fixing
# ---------------------------------------------------------------------------


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


def _filter_text_operand(
    operand: pikepdf.Object,
    notdef_codes: _NotdefCodes,
    *,
    is_cid: bool = False,
) -> pikepdf.Object | None:
    """Filters .notdef bytes from a text string operand.

    Args:
        operand: A pikepdf String operand from a text operator.
        notdef_codes: Set of byte values that are .notdef.
        is_cid: If True, treat operand as 2-byte CID pairs instead of
            single bytes.

    Returns:
        Filtered String, or None if the string becomes empty.
    """
    try:
        raw = bytes(operand)
    except Exception:
        return operand

    if is_cid:
        # Filter 2-byte CID pairs
        filtered = bytearray()
        for i in range(0, len(raw) - 1, 2):
            cid = (raw[i] << 8) | raw[i + 1]
            if cid not in notdef_codes:
                filtered.extend(raw[i : i + 2])
        filtered = bytes(filtered)
    else:
        filtered = bytes(b for b in raw if b not in notdef_codes)
    if filtered == raw:
        return operand
    if not filtered:
        return None
    return String(filtered)


def _fix_notdef_in_stream(
    stream_obj: Stream,
    font_map: dict[str, pikepdf.Object],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
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

    fixed = 0
    new_instructions = []
    current_font_name: str | None = None

    for item in instructions:
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            new_instructions.append(item)
            continue

        operands, operator = item.operands, item.operator
        op_str = str(operator)

        # Track font changes via Tf operator
        if op_str == "Tf" and len(operands) >= 1:
            try:
                current_font_name = str(operands[0])
            except Exception:
                current_font_name = None
            new_instructions.append(item)
            continue

        # Handle single-string text operators: Tj, ', "
        if op_str in _TEXT_OPERATORS and current_font_name is not None:
            font_obj = font_map.get(current_font_name)
            if font_obj is not None:
                notdef_codes = _get_notdef_codes(font_obj, notdef_cache)
                if notdef_codes:
                    is_cid = _is_cidfont(font_obj)
                    modified = _fix_single_string_op(
                        operands,
                        operator,
                        op_str,
                        notdef_codes,
                        is_cid=is_cid,
                    )
                    if modified is not None:
                        fixed += 1
                        if modified:
                            new_instructions.append(modified)
                        # modified is empty list → operator removed
                        continue

            new_instructions.append(item)
            continue

        # Handle TJ (array of strings and adjustments)
        if op_str == "TJ" and current_font_name is not None:
            font_obj = font_map.get(current_font_name)
            if font_obj is not None:
                notdef_codes = _get_notdef_codes(font_obj, notdef_cache)
                if notdef_codes:
                    is_cid = _is_cidfont(font_obj)
                    modified_tj = _fix_tj_array_op(
                        operands,
                        operator,
                        notdef_codes,
                        is_cid=is_cid,
                    )
                    if modified_tj is not None:
                        fixed += 1
                        if modified_tj:
                            new_instructions.append(modified_tj)
                        continue

            new_instructions.append(item)
            continue

        new_instructions.append(item)

    if fixed > 0:
        stream_obj.write(pikepdf.unparse_content_stream(new_instructions))

    return fixed


def _fix_single_string_op(
    operands: list,
    operator: pikepdf.Operator,
    op_str: str,
    notdef_codes: _NotdefCodes,
    *,
    is_cid: bool = False,
) -> pikepdf.ContentStreamInstruction | None:
    """Filters .notdef codes from a single-string text operator.

    Args:
        operands: The instruction operands.
        operator: The operator.
        op_str: String form of the operator.
        notdef_codes: Set of .notdef byte values.
        is_cid: If True, treat operands as 2-byte CID pairs.

    Returns:
        - None if no change needed
        - A new ContentStreamInstruction if the string was filtered
        - Empty list if the operator should be removed entirely
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

    filtered = _filter_text_operand(operand, notdef_codes, is_cid=is_cid)
    if filtered is operand:
        return None  # No change

    if filtered is None:
        # String became empty → remove operator
        return []

    new_operands = list(operands)
    new_operands[string_idx] = filtered
    return pikepdf.ContentStreamInstruction(new_operands, operator)


def _fix_tj_array_op(
    operands: list,
    operator: pikepdf.Operator,
    notdef_codes: _NotdefCodes,
    *,
    is_cid: bool = False,
) -> pikepdf.ContentStreamInstruction | None:
    """Filters .notdef codes from a TJ array operator.

    Args:
        operands: The instruction operands (should contain an Array).
        operator: The TJ operator.
        notdef_codes: Set of .notdef byte values.
        is_cid: If True, treat string elements as 2-byte CID pairs.

    Returns:
        - None if no change needed
        - A new ContentStreamInstruction if strings were filtered
        - Empty list if all strings became empty
    """
    if not operands or not isinstance(operands[0], Array):
        return None

    arr = operands[0]
    changed = False
    new_items = []

    for elem in arr:
        if isinstance(elem, String):
            filtered = _filter_text_operand(elem, notdef_codes, is_cid=is_cid)
            if filtered is not elem:
                changed = True
                if filtered is not None:
                    new_items.append(filtered)
                # else: skip empty string
            else:
                new_items.append(elem)
        else:
            # Numeric adjustment — keep it
            new_items.append(elem)

    if not changed:
        return None

    # Check if any strings remain
    has_strings = any(isinstance(item, String) for item in new_items)
    if not has_strings:
        return []  # Remove operator entirely

    new_arr = Array(new_items)
    return pikepdf.ContentStreamInstruction([new_arr], operator)


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------


def _fix_notdef_in_page_contents(
    page_dict: Dictionary,
    font_map: dict[str, pikepdf.Object],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
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

    if isinstance(contents, Stream):
        fixed += _fix_notdef_in_stream(contents, font_map, notdef_cache)
    elif isinstance(contents, Array):
        for item in contents:
            item = _resolve(item)
            if isinstance(item, Stream):
                fixed += _fix_notdef_in_stream(item, font_map, notdef_cache)

    return fixed


def _fix_notdef_in_form_xobjects(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
) -> int:
    """Recurses into Form XObjects to fix .notdef references.

    Args:
        resources: A resolved Resources dictionary.
        visited: Set of (objnum, gen) tuples for cycle detection.
        notdef_cache: Shared notdef code cache.

    Returns:
        Number of text operators fixed.
    """
    fixed = 0
    resources = _resolve(resources)
    if not isinstance(resources, Dictionary):
        return 0

    xobjects = resources.get("/XObject")
    if not xobjects:
        return 0
    xobjects = _resolve(xobjects)
    if not isinstance(xobjects, Dictionary):
        return 0

    for xobj_name in list(xobjects.keys()):
        xobj = _resolve(xobjects[xobj_name])
        if not isinstance(xobj, Stream):
            continue

        subtype = xobj.get("/Subtype")
        if subtype is None or str(subtype) != "/Form":
            continue

        objgen = xobj.objgen
        if objgen != (0, 0):
            if objgen in visited:
                continue
            visited.add(objgen)

        # Build font map from Form XObject's own resources
        form_resources = xobj.get("/Resources")
        if form_resources:
            form_resources = _resolve(form_resources)
            form_font_map = _build_font_map(form_resources)
        else:
            form_font_map = {}

        fixed += _fix_notdef_in_stream(xobj, form_font_map, notdef_cache)

        # Recurse into nested Form XObjects and Patterns
        if form_resources:
            fixed += _fix_notdef_in_form_xobjects(form_resources, visited, notdef_cache)
            fixed += _fix_notdef_in_patterns(form_resources, visited, notdef_cache)

    return fixed


def _fix_notdef_in_patterns(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
) -> int:
    """Recurses into Tiling Patterns to fix .notdef references.

    Args:
        resources: A resolved Resources dictionary.
        visited: Set of (objnum, gen) tuples for cycle detection.
        notdef_cache: Shared notdef code cache.

    Returns:
        Number of text operators fixed.
    """
    fixed = 0
    resources = _resolve(resources)
    if not isinstance(resources, Dictionary):
        return 0

    patterns = resources.get("/Pattern")
    if not patterns:
        return 0
    patterns = _resolve(patterns)
    if not isinstance(patterns, Dictionary):
        return 0

    for pat_name in list(patterns.keys()):
        try:
            pattern = _resolve(patterns[pat_name])
            if not isinstance(pattern, Stream):
                continue

            # Only process Tiling Patterns (PatternType 1)
            pattern_type = pattern.get("/PatternType")
            if pattern_type is None or int(pattern_type) != 1:
                continue

            objgen = pattern.objgen
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            # Build font map from pattern's own resources
            pat_resources = pattern.get("/Resources")
            if pat_resources:
                pat_resources = _resolve(pat_resources)
                pat_font_map = _build_font_map(pat_resources)
            else:
                pat_font_map = {}

            fixed += _fix_notdef_in_stream(pattern, pat_font_map, notdef_cache)

            # Recurse into nested Form XObjects and Patterns
            if pat_resources:
                fixed += _fix_notdef_in_form_xobjects(
                    pat_resources, visited, notdef_cache
                )
                fixed += _fix_notdef_in_patterns(pat_resources, visited, notdef_cache)
        except Exception:
            continue

    return fixed


def _fix_notdef_in_ap_stream(
    ap_entry: pikepdf.Object,
    visited: set[tuple[int, int]],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
) -> int:
    """Fixes .notdef references in an annotation appearance stream entry.

    Args:
        ap_entry: An appearance entry (N, R, or D value).
        visited: Set of (objnum, gen) tuples for cycle detection.
        notdef_cache: Shared notdef code cache.

    Returns:
        Number of text operators fixed.
    """
    fixed = 0
    ap_entry = _resolve(ap_entry)

    if isinstance(ap_entry, Stream):
        ap_resources = ap_entry.get("/Resources")
        ap_font_map = _build_font_map(ap_resources) if ap_resources else {}
        fixed += _fix_notdef_in_stream(ap_entry, ap_font_map, notdef_cache)
        if ap_resources:
            ap_resources = _resolve(ap_resources)
            fixed += _fix_notdef_in_form_xobjects(ap_resources, visited, notdef_cache)
            fixed += _fix_notdef_in_patterns(ap_resources, visited, notdef_cache)
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                st_resources = state_stream.get("/Resources")
                st_font_map = _build_font_map(st_resources) if st_resources else {}
                fixed += _fix_notdef_in_stream(state_stream, st_font_map, notdef_cache)
                if st_resources:
                    st_resources = _resolve(st_resources)
                    fixed += _fix_notdef_in_form_xobjects(
                        st_resources, visited, notdef_cache
                    )
                    fixed += _fix_notdef_in_patterns(
                        st_resources, visited, notdef_cache
                    )

    return fixed


def _fix_notdef_in_type3_charprocs(
    resources: pikepdf.Object,
    visited: set[tuple[int, int]],
    notdef_cache: dict[tuple[int, int], _NotdefCodes],
) -> int:
    """Fixes .notdef references in Type3 font CharProcs.

    Args:
        resources: A resolved Resources dictionary.
        visited: Set of (objnum, gen) tuples for cycle detection.
        notdef_cache: Shared notdef code cache.

    Returns:
        Number of text operators fixed.
    """
    fixed = 0

    for _font_name, font in _iter_type3_fonts(resources, visited):
        charprocs = font.get("/CharProcs")
        if charprocs is None:
            continue
        charprocs = _resolve(charprocs)
        if not isinstance(charprocs, Dictionary):
            continue

        # Type3 CharProcs may reference fonts from the font's own resources
        font_resources = font.get("/Resources")
        cp_font_map = _build_font_map(font_resources) if font_resources else {}

        for cp_name in list(charprocs.keys()):
            cp_stream = _resolve(charprocs[cp_name])
            if isinstance(cp_stream, Stream):
                fixed += _fix_notdef_in_stream(cp_stream, cp_font_map, notdef_cache)

    return fixed
