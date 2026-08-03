# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Glyph usage collection from PDF content streams.

Parses all content streams (pages, Form XObjects, Tiling Patterns,
soft-mask groups, Type3 CharProcs, Annotation Appearance Streams)
and collects character codes used with each font. This is needed
for font subsetting — only glyphs that are actually used need to
be kept in the font program.
"""

from collections import defaultdict
from collections.abc import Iterator

import pikepdf

from ..utils import resolve_indirect as _resolve_indirect
from .tounicode import get_font_code_space_ranges, split_cmap_codes
from .traversal import get_page_resources

CharacterCode = int | bytes
_ObjectKey = tuple[int, int] | tuple[str, bytes]
_ContextKey = tuple[_ObjectKey, _ObjectKey]

# Text-showing operators that contain character strings
_TEXT_OPERATORS = frozenset(
    {
        pikepdf.Operator("Tj"),
        pikepdf.Operator("TJ"),
        pikepdf.Operator("'"),
        pikepdf.Operator('"'),
    }
)

_TF_OPERATOR = pikepdf.Operator("Tf")
_Q_OPERATOR = pikepdf.Operator("q")
_RESTORE_OPERATOR = pikepdf.Operator("Q")


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


def _extract_char_codes(
    string_operand: pikepdf.Object,
    is_cid: bool,
    code_space_ranges: tuple[tuple[bytes, bytes], ...] | None = None,
) -> set[int]:
    """Extracts character codes from a text string operand.

    For simple fonts, each byte is one character code (0-255).
    CIDFont codes are decoded with their CMap codespace ranges.

    Args:
        string_operand: pikepdf String object from a text operator.
        is_cid: True if the current font is a CIDFont.
        code_space_ranges: Effective CMap codespace ranges for a CIDFont.

    Returns:
        Set of character codes found in the string.
    """
    codes: set[int] = set()
    try:
        raw = bytes(string_operand)
    except Exception:
        return codes

    if is_cid:
        ranges = code_space_ranges or ((b"\x00\x00", b"\xff\xff"),)
        codes.update(
            int.from_bytes(code, "big") for code in split_cmap_codes(raw, ranges)
        )
    else:
        # 1-byte encoding
        for b in raw:
            codes.add(b)

    return codes


def _iter_content_streams_with_resources(
    page: pikepdf.Page,
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object]]:
    """Yields (content_stream_owner, resources) for all nested structures on a page.

    Traverses page-level content, Form XObjects, Tiling Patterns,
    soft-mask groups, and Annotation Appearance Streams recursively.

    Args:
        page: A pikepdf Page object.

    Yields:
        Tuples of (stream_owner, resources_dict).
    """
    processed: set[_ContextKey] = set()
    active: set[_ObjectKey] = set()

    # Page-level
    resources = get_page_resources(page)

    if resources is not None:
        yield (page.obj, resources)
        yield from _iter_nested_streams(resources, processed, active)

    # Annotation Appearance Streams
    annots = page.get("/Annots")
    if annots is None:
        return

    try:
        annots = _resolve_indirect(annots)
    except Exception:
        return

    for annot_ref in annots:
        try:
            annot = _resolve_indirect(annot_ref)
            ap = annot.get("/AP")
            if ap is None:
                continue
            ap = _resolve_indirect(ap)

            for ap_key in ("/N", "/R", "/D"):
                ap_entry = ap.get(ap_key)
                if ap_entry is None:
                    continue

                try:
                    ap_entry = _resolve_indirect(ap_entry)
                except Exception:
                    continue

                if isinstance(ap_entry, pikepdf.Stream):
                    res = ap_entry.get("/Resources")
                    if res is not None:
                        res = _resolve_indirect(res)
                    else:
                        res = resources
                    if isinstance(res, pikepdf.Dictionary):
                        yield from _iter_stream_context(
                            ap_entry,
                            res,
                            processed,
                            active,
                        )
                elif isinstance(ap_entry, pikepdf.Dictionary):
                    for sub_key in list(ap_entry.keys()):
                        try:
                            sub = _resolve_indirect(ap_entry[sub_key])
                            if isinstance(sub, pikepdf.Stream):
                                res = sub.get("/Resources")
                                if res is not None:
                                    res = _resolve_indirect(res)
                                else:
                                    res = resources
                                if isinstance(res, pikepdf.Dictionary):
                                    yield from _iter_stream_context(
                                        sub,
                                        res,
                                        processed,
                                        active,
                                    )
                        except Exception:
                            continue
        except Exception:
            continue


def _object_identity(obj: pikepdf.Object) -> _ObjectKey:
    """Return a stable identity for traversal cycle detection."""
    key = obj.objgen
    return key if key != (0, 0) else ("direct", obj.unparse())


def find_ambiguous_resource_context_streams(
    pdf: pikepdf.Pdf,
) -> set[_ObjectKey]:
    """Return content streams reused with different effective resources."""
    contexts: dict[
        _ObjectKey,
        set[_ObjectKey],
    ] = defaultdict(set)
    for page in pdf.pages:
        page_resources = get_page_resources(page)
        if isinstance(page_resources, pikepdf.Dictionary):
            resource_key = _object_identity(page_resources)
            contents = _resolve_indirect(page.obj.get("/Contents"))
            content_streams = (
                list(contents) if isinstance(contents, pikepdf.Array) else [contents]
            )
            for content in content_streams:
                content = _resolve_indirect(content)
                if isinstance(content, pikepdf.Stream):
                    contexts[_object_identity(content)].add(resource_key)

        for owner, resources in _iter_content_streams_with_resources(page):
            if isinstance(owner, pikepdf.Stream):
                contexts[_object_identity(owner)].add(_object_identity(resources))

    return {
        stream_key
        for stream_key, resource_keys in contexts.items()
        if len(resource_keys) > 1
    }


def _iter_stream_context(
    stream: pikepdf.Stream,
    resources: pikepdf.Dictionary,
    processed: set[_ContextKey],
    active: set[_ObjectKey],
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object]]:
    """Yield one stream/resource context and its graph without recursion."""
    yield from _iter_resource_graph(
        [("stream", stream, resources)],
        processed,
        active,
    )


def _iter_nested_streams(
    resources: pikepdf.Object,
    processed: set[_ContextKey],
    active: set[_ObjectKey],
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object]]:
    """Yield nested stream/resource contexts without using Python recursion."""
    yield from _iter_resource_graph(
        [("resources", resources, None)],
        processed,
        active,
    )


def _iter_resource_graph(
    initial_tasks: list[tuple[str, pikepdf.Object, pikepdf.Object | None]],
    processed: set[_ContextKey],
    active: set[_ObjectKey],
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object]]:
    """Walk content-bearing resource graphs with an explicit work stack."""
    del active  # Context-pair deduplication is sufficient for finite PDF graphs.
    tasks = list(reversed(initial_tasks))

    while tasks:
        kind, obj, context_resources = tasks.pop()
        if kind == "stream":
            stream = _resolve_indirect(obj)
            resources = _resolve_indirect(context_resources)
            if not isinstance(stream, pikepdf.Stream) or not isinstance(
                resources, pikepdf.Dictionary
            ):
                continue
            context = (_object_identity(stream), _object_identity(resources))
            if context in processed:
                continue
            processed.add(context)
            yield stream, resources
            tasks.append(("resources", resources, None))
            continue

        resources = _resolve_indirect(obj)
        if not isinstance(resources, pikepdf.Dictionary):
            continue
        discovered: list[tuple[str, pikepdf.Object, pikepdf.Object | None]] = []

        xobjects = _resolve_indirect(resources.get("/XObject"))
        if isinstance(xobjects, pikepdf.Dictionary):
            for name in list(xobjects.keys()):
                try:
                    stream = _resolve_indirect(xobjects[name])
                    if (
                        isinstance(stream, pikepdf.Stream)
                        and str(stream.get("/Subtype")) == "/Form"
                    ):
                        nested = _resolve_indirect(stream.get("/Resources"))
                        if not isinstance(nested, pikepdf.Dictionary):
                            nested = resources
                        discovered.append(("stream", stream, nested))
                except Exception:
                    continue

        patterns = _resolve_indirect(resources.get("/Pattern"))
        if isinstance(patterns, pikepdf.Dictionary):
            for name in list(patterns.keys()):
                try:
                    stream = _resolve_indirect(patterns[name])
                    if (
                        isinstance(stream, pikepdf.Stream)
                        and int(stream.get("/PatternType", 0)) == 1
                    ):
                        nested = _resolve_indirect(stream.get("/Resources"))
                        if not isinstance(nested, pikepdf.Dictionary):
                            nested = resources
                        discovered.append(("stream", stream, nested))
                except Exception:
                    continue

        extgstates = _resolve_indirect(resources.get("/ExtGState"))
        if isinstance(extgstates, pikepdf.Dictionary):
            for name in list(extgstates.keys()):
                try:
                    extgstate = _resolve_indirect(extgstates[name])
                    if not isinstance(extgstate, pikepdf.Dictionary):
                        continue
                    soft_mask = _resolve_indirect(extgstate.get("/SMask"))
                    if not isinstance(soft_mask, pikepdf.Dictionary):
                        continue
                    stream = _resolve_indirect(soft_mask.get("/G"))
                    if not isinstance(stream, pikepdf.Stream):
                        continue
                    subtype = stream.get("/Subtype")
                    if subtype is not None and str(subtype) != "/Form":
                        continue
                    nested = _resolve_indirect(stream.get("/Resources"))
                    if not isinstance(nested, pikepdf.Dictionary):
                        nested = resources
                    discovered.append(("stream", stream, nested))
                except Exception:
                    continue

        fonts = _resolve_indirect(resources.get("/Font"))
        if isinstance(fonts, pikepdf.Dictionary):
            for name in list(fonts.keys()):
                try:
                    font = _resolve_indirect(fonts[name])
                    if (
                        not isinstance(font, pikepdf.Dictionary)
                        or str(font.get("/Subtype")) != "/Type3"
                    ):
                        continue
                    t3_resources = _resolve_indirect(font.get("/Resources"))
                    if not isinstance(t3_resources, pikepdf.Dictionary):
                        t3_resources = resources
                    font_context = (
                        _object_identity(font),
                        _object_identity(t3_resources),
                    )
                    if font_context in processed:
                        continue
                    processed.add(font_context)
                    charprocs = _resolve_indirect(font.get("/CharProcs"))
                    if isinstance(charprocs, pikepdf.Dictionary):
                        for proc_name in list(charprocs.keys()):
                            proc = _resolve_indirect(charprocs[proc_name])
                            if isinstance(proc, pikepdf.Stream):
                                discovered.append(("stream", proc, t3_resources))
                    if _object_identity(t3_resources) != _object_identity(resources):
                        discovered.append(("resources", t3_resources, None))
                except Exception:
                    continue

        tasks.extend(reversed(discovered))


def _resolve_font_object(
    font_name_in_stream: str,
    resources: pikepdf.Object,
) -> pikepdf.Object | None:
    """Resolves a font name from a content stream to its font object.

    Args:
        font_name_in_stream: Font name as used in Tf operator (e.g. "/F1").
        resources: Resources dictionary containing the Font sub-dictionary.

    Returns:
        The resolved font object, or None if not found.
    """
    font_dict = resources.get("/Font")
    if font_dict is None:
        return None

    try:
        font_dict = _resolve_indirect(font_dict)
    except Exception:
        return None

    # The font name in the stream includes the leading "/" — use it as a key
    font_ref = font_dict.get(font_name_in_stream)
    if font_ref is None:
        return None

    try:
        return _resolve_indirect(font_ref)
    except Exception:
        return None


class FontUsageCache:
    """Lazily computed, invalidatable cache around collect_font_usage().

    collect_font_usage() parses every content stream in the document,
    which is expensive. Passes that only read glyph usage can share a
    single collection through this cache; passes that rewrite content
    streams (and thereby may change which codes are used) must call
    invalidate() so the next consumer sees fresh data.
    """

    def __init__(self, pdf: pikepdf.Pdf) -> None:
        """Initializes the cache for a specific PDF.

        Args:
            pdf: Opened pikepdf PDF object.
        """
        self._pdf = pdf
        self._usage: dict[tuple[int, int], set[CharacterCode]] | None = None

    def get(self) -> dict[tuple[int, int], set[CharacterCode]]:
        """Returns the font usage map, collecting it on first access."""
        if self._usage is None:
            self._usage = collect_font_usage(self._pdf)
        return self._usage

    def invalidate(self) -> None:
        """Drops the cached usage map after content streams changed."""
        self._usage = None


def collect_font_usage(
    pdf: pikepdf.Pdf,
) -> dict[tuple[int, int], set[CharacterCode]]:
    """Collects character codes used with each font across the entire PDF.

    Iterates all pages and their nested structures (Form XObjects,
    Tiling Patterns, soft masks, Annotation APs), parses content
    streams, and records which character codes are used with each font.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        Dictionary mapping font objgen (object_number, generation)
        to the set of character codes used with that font.
        Only fonts with objgen != (0,0) are included.
    """
    usage: dict[tuple[int, int], set[CharacterCode]] = {}

    for page in pdf.pages:
        for stream_owner, resources in _iter_content_streams_with_resources(page):
            _process_content_stream(stream_owner, resources, usage)

    return usage


def _process_content_stream(
    stream_owner: pikepdf.Object,
    resources: pikepdf.Object,
    usage: dict[tuple[int, int], set[CharacterCode]],
) -> None:
    """Parses a content stream and records character code usage.

    Args:
        stream_owner: Object that owns the content stream (page or XObject).
        resources: Resources dictionary for font resolution.
        usage: Accumulator mapping font objgen -> used character codes.
    """
    try:
        instructions = pikepdf.parse_content_stream(stream_owner)
    except Exception:
        return

    current_font: pikepdf.Object | None = None
    current_font_is_cid = False
    current_code_space_ranges: tuple[tuple[bytes, bytes], ...] | None = None
    graphics_state_stack: list[
        tuple[
            pikepdf.Object | None,
            bool,
            tuple[tuple[bytes, bytes], ...] | None,
        ]
    ] = []

    def record_codes(operand: pikepdf.Object, objgen: tuple[int, int]) -> None:
        if current_font_is_cid:
            try:
                raw = bytes(operand)
            except Exception:
                return
            ranges = current_code_space_ranges or ((b"\x00\x00", b"\xff\xff"),)
            codes = split_cmap_codes(raw, ranges)
            if codes:
                usage.setdefault(objgen, set()).update(codes)
            return

        codes = _extract_char_codes(operand, False)
        if codes:
            usage.setdefault(objgen, set()).update(codes)

    def text_operands(
        operands: list[pikepdf.Object],
        operator: pikepdf.Operator,
    ) -> Iterator[pikepdf.Object]:
        if operator == pikepdf.Operator("TJ"):
            if operands and isinstance(operands[0], pikepdf.Array):
                yield from (
                    item for item in operands[0] if isinstance(item, pikepdf.String)
                )
        elif operator == pikepdf.Operator('"'):
            if len(operands) >= 3:
                yield operands[2]
        elif operands:
            yield operands[0]

    def record_unresolved_font_state(
        operands: list[pikepdf.Object],
        operator: pikepdf.Operator,
    ) -> None:
        """Conservatively preserve shown codes for every effective font."""
        font_dict = _resolve_indirect(resources.get("/Font"))
        if not isinstance(font_dict, pikepdf.Dictionary):
            return
        strings = list(text_operands(operands, operator))
        for font_key in list(font_dict.keys()):
            try:
                font_obj = _resolve_indirect(font_dict[font_key])
                objgen = font_obj.objgen
            except Exception:
                continue
            if objgen == (0, 0):
                continue
            is_cid = _is_cidfont(font_obj)
            ranges = get_font_code_space_ranges(font_obj) if is_cid else None
            for operand in strings:
                try:
                    raw = bytes(operand)
                except Exception:
                    continue
                codes: set[CharacterCode]
                if is_cid:
                    codes = set(
                        split_cmap_codes(
                            raw,
                            ranges or ((b"\x00\x00", b"\xff\xff"),),
                        )
                    )
                else:
                    codes = set(raw)
                if codes:
                    usage.setdefault(objgen, set()).update(codes)

    for operands, operator in instructions:
        if operator == _Q_OPERATOR:
            graphics_state_stack.append(
                (current_font, current_font_is_cid, current_code_space_ranges)
            )
        elif operator == _RESTORE_OPERATOR:
            if graphics_state_stack:
                (
                    current_font,
                    current_font_is_cid,
                    current_code_space_ranges,
                ) = graphics_state_stack.pop()
            else:
                current_font = None
                current_font_is_cid = False
                current_code_space_ranges = None
        elif operator == _TF_OPERATOR:
            # Tf: set current font
            if operands:
                font_name = str(operands[0])
                font_obj = _resolve_font_object(font_name, resources)
                if font_obj is not None:
                    current_font = font_obj
                    current_font_is_cid = _is_cidfont(font_obj)
                    current_code_space_ranges = get_font_code_space_ranges(font_obj)
                else:
                    current_font = None
                    current_font_is_cid = False
                    current_code_space_ranges = None

        elif operator in _TEXT_OPERATORS:
            if current_font is None:
                # A Form XObject may legally inherit the caller's current
                # text font.  The traversal does not execute Do operators,
                # so preserve the shown codes for every effective font.
                record_unresolved_font_state(operands, operator)
                continue

            # Get objgen for the current font
            try:
                objgen = current_font.objgen
            except Exception:
                continue

            if objgen == (0, 0):
                continue

            for operand in text_operands(operands, operator):
                record_codes(operand, objgen)
