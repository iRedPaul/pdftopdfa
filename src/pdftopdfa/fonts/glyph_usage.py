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
from collections.abc import Iterable, Iterator

import pikepdf

from ..utils import iter_type3_fonts
from ..utils import resolve_indirect as _resolve_indirect
from .tounicode import get_font_code_space_ranges, split_cmap_codes
from .traversal import get_page_resources, iter_all_page_fonts

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
# Operators _process_content_stream acts on (graphics state, font, text).
_USAGE_OPERATORS = "q Q gs Tf Tj TJ ' \""

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
    processed: set[_ContextKey] | None = None,
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object]]:
    """Yields (content_stream_owner, resources) for all nested structures.

    See :func:`iter_content_streams_with_resource_keys`, which this wraps.
    """
    for owner, resources, _key, _inherited in iter_content_streams_with_resource_info(
        page, processed
    ):
        yield owner, resources


def iter_content_streams_with_resource_keys(
    page: pikepdf.Page,
    processed: set[_ContextKey] | None = None,
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object, _ObjectKey]]:
    """Like :func:`iter_content_streams_with_resource_info`, without the flag."""
    for owner, resources, key, _inherited in iter_content_streams_with_resource_info(
        page, processed
    ):
        yield owner, resources, key


def iter_content_streams_with_resource_info(
    page: pikepdf.Page,
    processed: set[_ContextKey] | None = None,
    *,
    resources_only: bool = False,
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object, _ObjectKey, bool]]:
    """Yields (content_stream_owner, resources) for all nested structures on a page.

    Traverses page-level content, Form XObjects, Tiling Patterns,
    soft-mask groups, and Annotation Appearance Streams recursively.

    Args:
        page: A pikepdf Page object.

        processed: Optional context set shared across pages. Pass one set
            for a whole-document walk to visit each (stream, resources)
            context once instead of once per page; the page itself is
            always yielded. Only for callers that do not need per-page
            results for shared streams.

    Yields:
        Tuples of (stream_owner, resources_dict, resources_identity,
        inherited). The identity lets read-only callers skip resources they
        have already examined. ``inherited`` is True when the owner uses the
        very resources object already yielded for its parent (resourceless
        forms, CharProcs of a Type3 font without /Resources), so callers that
        only act on resources can skip it, even ones that modify them.
        With ``resources_only=True`` such owners are not visited at all
        (their resources are the parent's, and their children are queued
        from the parent already), which saves walking every shared glyph.
    """
    if processed is None:
        processed = set()

    # Page-level
    resources = get_page_resources(page)

    if resources is not None:
        resources_key = _object_identity(_resolve_indirect(resources))
        yield (page.obj, resources, resources_key, False)
        yield from _iter_resource_graph(
            [("resources", resources, None, resources_key)],
            processed,
            skip_inherited=resources_only,
        )

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
                            skip_inherited=resources_only,
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
                                        skip_inherited=resources_only,
                                    )
                        except Exception:
                            continue
        except Exception:
            continue


def _object_identity(obj: pikepdf.Object) -> _ObjectKey:
    """Return a stable identity for traversal cycle detection."""
    key = obj.objgen
    return key if key != (0, 0) else ("direct", obj.unparse())


_DEVICE_COLOR_SPACES = frozenset(
    {
        "/DeviceGray",
        "/DeviceRGB",
        "/DeviceCMYK",
        "/Pattern",
    }
)
_DEFAULT_COLOR_SPACES = {
    "/DeviceGray": "/DefaultGray",
    "/DeviceRGB": "/DefaultRGB",
    "/DeviceCMYK": "/DefaultCMYK",
}
_DEVICE_COLOR_OPERATORS = {
    "g": "/DefaultGray",
    "G": "/DefaultGray",
    "rg": "/DefaultRGB",
    "RG": "/DefaultRGB",
    "k": "/DefaultCMYK",
    "K": "/DefaultCMYK",
}
_INLINE_COLOR_SPACE_ALIASES = {
    "/G": "/DeviceGray",
    "/RGB": "/DeviceRGB",
    "/CMYK": "/DeviceCMYK",
}
_NAMED_RESOURCE_OPERATORS = frozenset({"Tf", "Do", "gs", "sh"})


def stream_uses_named_resources(
    stream: pikepdf.Stream, cache: dict[_ObjectKey, bool] | None = None
) -> bool:
    """Return whether a content stream depends on its resource context.

    Streams that never do (plain path glyphs, most Type3 CharProcs) render
    the same in every resource context, so they never need per-context
    copies. Device color selections also look up the implicit DefaultGray,
    DefaultRGB or DefaultCMYK resource. Any doubt (unreadable or unparsable
    data) answers True.
    """
    key = _object_identity(stream)
    if cache is not None and key in cache:
        return cache[key]
    result = _scan_named_resources(stream)
    if cache is not None:
        cache[key] = result
    return result


def _scan_named_resources(stream: pikepdf.Stream) -> bool:
    try:
        for instruction in pikepdf.parse_content_stream(stream):
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                image = instruction.iimage.obj
                space = image.get("/CS", image.get("/ColorSpace"))
                if space is not None:
                    # Even device spaces depend on contextual Default entries.
                    return True
                continue
            operator = str(instruction.operator)
            operands = instruction.operands
            if operator in _NAMED_RESOURCE_OPERATORS:
                return True
            if operator in _DEVICE_COLOR_OPERATORS:
                return True
            if operator in ("cs", "CS"):
                if not operands or str(operands[0]) != "/Pattern":
                    return True
            elif operator in ("scn", "SCN"):
                if operands and isinstance(operands[-1], pikepdf.Name):
                    return True
            elif operator in ("BDC", "DP"):
                if len(operands) > 1 and isinstance(operands[1], pikepdf.Name):
                    return True
        return False
    except Exception:
        return True


_NAME_OPERATOR_CATEGORIES = {
    "Tf": "/Font",
    "Do": "/XObject",
    "gs": "/ExtGState",
    "sh": "/Shading",
    "cs": "/ColorSpace",
    "CS": "/ColorSpace",
}


def used_resource_names(
    streams: Iterable[pikepdf.Stream],
) -> dict[str, set[str]] | None:
    """Return the resource names the given content streams look up.

    Maps each resource category (``/Font``, ``/XObject``, ...) to the names
    used from it. Returns None when any stream cannot be read or parsed, so
    callers fall back to keeping every inherited resource.
    """
    used: dict[str, set[str]] = defaultdict(set)
    for stream in streams:
        try:
            instructions = list(pikepdf.parse_content_stream(stream))
        except Exception:
            return None
        for instruction in instructions:
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                image = instruction.iimage.obj
                space = image.get("/CS", image.get("/ColorSpace"))
                if isinstance(space, pikepdf.Name):
                    name = _INLINE_COLOR_SPACE_ALIASES.get(str(space), str(space))
                    used["/ColorSpace"].add(_DEFAULT_COLOR_SPACES.get(name, name))
                elif space is not None:  # e.g. [/Indexed /CS0 ...]
                    return None
                continue
            operator = str(instruction.operator)
            operands = instruction.operands
            if operator in _DEVICE_COLOR_OPERATORS:
                used["/ColorSpace"].add(_DEVICE_COLOR_OPERATORS[operator])
            category = _NAME_OPERATOR_CATEGORIES.get(operator)
            if category is not None:
                if operands and isinstance(operands[0], pikepdf.Name):
                    name = str(operands[0])
                    if category == "/ColorSpace" and name in _DEFAULT_COLOR_SPACES:
                        used[category].add(_DEFAULT_COLOR_SPACES[name])
                    elif category != "/ColorSpace" or name not in _DEVICE_COLOR_SPACES:
                        used[category].add(name)
            elif operator in ("scn", "SCN"):
                if operands and isinstance(operands[-1], pikepdf.Name):
                    used["/Pattern"].add(str(operands[-1]))
            elif operator in ("BDC", "DP"):
                if len(operands) > 1 and isinstance(operands[1], pikepdf.Name):
                    used["/Properties"].add(str(operands[1]))
    return dict(used)


def find_ambiguous_resource_context_streams(
    pdf: pikepdf.Pdf,
) -> set[_ObjectKey]:
    """Return content streams reused with different effective resources.

    A stream without its own ``/Resources`` that never looks up a resource
    name renders identically in every context and is not reported.
    """
    uses_names: dict[_ObjectKey, bool] = {}
    contexts: dict[
        _ObjectKey,
        set[_ObjectKey],
    ] = defaultdict(set)
    walked: set[_ContextKey] = set()  # contexts are aggregated document-wide
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

        for owner, resources in _iter_content_streams_with_resources(page, walked):
            if isinstance(owner, pikepdf.Stream):
                contexts[_object_identity(owner)].add(_object_identity(resources))

    ambiguous = set()
    for stream_key, resource_keys in contexts.items():
        if len(resource_keys) < 2:
            continue
        stream = pdf.get_object(stream_key) if stream_key[0] != "direct" else None
        if (
            isinstance(stream, pikepdf.Stream)
            and "/Resources" not in stream
            and not stream_uses_named_resources(stream, uses_names)
        ):
            continue
        ambiguous.add(stream_key)
    return ambiguous


def _iter_stream_context(
    stream: pikepdf.Stream,
    resources: pikepdf.Dictionary,
    processed: set[_ContextKey],
    *,
    skip_inherited: bool = False,
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object, _ObjectKey, bool]]:
    """Yield one stream/resource context and its graph without recursion."""
    yield from _iter_resource_graph(
        [("stream", stream, resources, None)],
        processed,
        skip_inherited=skip_inherited,
    )


def _iter_nested_streams(
    resources: pikepdf.Object,
    processed: set[_ContextKey],
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object, _ObjectKey, bool]]:
    """Yield nested stream/resource contexts without using Python recursion."""
    yield from _iter_resource_graph(
        [("resources", resources, None, None)],
        processed,
    )


def _iter_resource_graph(
    initial_tasks: list[
        tuple[str, pikepdf.Object, pikepdf.Object | None, _ObjectKey | None]
    ],
    processed: set[_ContextKey],
    *,
    skip_inherited: bool = False,
) -> Iterator[tuple[pikepdf.Object, pikepdf.Object, _ObjectKey, bool]]:
    """Walk content-bearing resource graphs with an explicit work stack.

    Each task may carry the precomputed identity of its resources so that a
    direct dictionary is serialized once per expansion rather than once per
    child, and each resources identity is expanded only once. Without this, a
    direct dictionary with N resourceless Form XObjects was re-expanded for
    every child and re-serialized for every task: O(N^3).
    """
    tasks = list(reversed(initial_tasks))

    while tasks:
        kind, obj, context_resources, known_key = tasks.pop()
        if kind in ("stream", "istream"):
            stream = _resolve_indirect(obj)
            resources = _resolve_indirect(context_resources)
            if not isinstance(stream, pikepdf.Stream) or not isinstance(
                resources, pikepdf.Dictionary
            ):
                continue
            resources_key = (
                known_key if known_key is not None else _object_identity(resources)
            )
            context = (_object_identity(stream), resources_key)
            if context in processed:
                continue
            processed.add(context)
            # "istream": the stream inherits the very resources object that
            # was yielded (and expanded) for its parent.
            yield stream, resources, resources_key, kind == "istream"
            # Most streams (all CharProcs, resourceless forms) use the parent
            # resources, which are already expanded; skip the no-op task.
            if ("expanded", resources_key) not in processed:
                tasks.append(("resources", resources, None, resources_key))
            continue

        resources = _resolve_indirect(obj)
        if not isinstance(resources, pikepdf.Dictionary):
            continue
        key = known_key if known_key is not None else _object_identity(resources)
        expanded_marker = ("expanded", key)
        if expanded_marker in processed:
            continue
        processed.add(expanded_marker)
        discovered: list[
            tuple[str, pikepdf.Object, pikepdf.Object | None, _ObjectKey | None]
        ] = []

        xobjects = _resolve_indirect(resources.get("/XObject"))
        if isinstance(xobjects, pikepdf.Dictionary):
            for name in list(xobjects.keys()):
                try:
                    stream = _resolve_indirect(xobjects[name])
                    if (
                        isinstance(stream, pikepdf.Stream)
                        and str(stream.get("/Subtype")) == "/Form"
                    ):
                        discovered.append(_stream_task(stream, resources, key))
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
                        discovered.append(_stream_task(stream, resources, key))
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
                    discovered.append(_stream_task(stream, resources, key))
                except Exception:
                    continue

        for _name, font in iter_type3_fonts(resources, set()):
            try:
                t3_resources = _resolve_indirect(font.get("/Resources"))
                if isinstance(t3_resources, pikepdf.Dictionary):
                    t3_key = _object_identity(t3_resources)
                    proc_kind = "stream"
                else:
                    t3_resources, t3_key = resources, key
                    proc_kind = "istream"
                font_context = (_object_identity(font), t3_key)
                if font_context in processed:
                    continue
                processed.add(font_context)
                charprocs = _resolve_indirect(font.get("/CharProcs"))
                if skip_inherited and proc_kind == "istream":
                    charprocs = None  # glyphs reuse these resources
                if isinstance(charprocs, pikepdf.Dictionary):
                    # All CharProcs use the same t3_resources object. With the
                    # font's own /Resources, the first glyph introduces it and
                    # the others reuse it ("istream"), so resource-only passes
                    # examine it once rather than once per glyph.
                    procs = [
                        proc
                        for _proc_name, proc in charprocs.items()
                        if isinstance(proc, pikepdf.Stream)
                    ]
                    for index, proc in enumerate(procs):
                        kind_here = proc_kind if index == 0 else "istream"
                        if skip_inherited and kind_here == "istream":
                            break
                        discovered.append((kind_here, proc, t3_resources, t3_key))
                if t3_key != key:
                    discovered.append(("resources", t3_resources, None, t3_key))
            except Exception:
                continue

        if skip_inherited:
            discovered = [task for task in discovered if task[0] != "istream"]
        tasks.extend(reversed(discovered))


def _stream_task(
    stream: pikepdf.Stream,
    parent: pikepdf.Dictionary,
    parent_key: _ObjectKey,
) -> tuple[str, pikepdf.Object, pikepdf.Object, _ObjectKey | None]:
    """Build a stream task using own resources, else the parent's (keyed)."""
    nested = _resolve_indirect(stream.get("/Resources"))
    if isinstance(nested, pikepdf.Dictionary):
        return ("stream", stream, nested, None)
    return ("istream", stream, parent, parent_key)


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
        self._usage: dict[bool, dict[_ObjectKey, set[CharacterCode]]] = {}
        self._raw: _RawFontUsage | None = None

    def get(
        self, *, require_resolved_font: bool = False
    ) -> dict[_ObjectKey, set[CharacterCode]]:
        """Returns the requested usage map, collecting it on first access."""
        if require_resolved_font not in self._usage:
            # Both variants come from the same walk; collect it only once.
            if self._raw is None:
                self._raw = _collect_raw_font_usage(self._pdf)
            self._usage[require_resolved_font] = _merge_font_usage(
                self._raw, require_resolved_font
            )
        return self._usage[require_resolved_font]

    def invalidate(self) -> None:
        """Drops the cached usage map after content streams changed."""
        self._usage.clear()
        self._raw = None


def collect_font_usage(
    pdf: pikepdf.Pdf,
    *,
    require_resolved_font: bool = False,
) -> dict[_ObjectKey, set[CharacterCode]]:
    """Collects character codes used with each font across the entire PDF.

    Iterates all pages and their nested structures (Form XObjects,
    Tiling Patterns, soft masks, Annotation APs), parses content
    streams, and records which character codes are used with each font.

    Args:
        pdf: Opened pikepdf PDF object.
        require_resolved_font: Exclude fonts matched only by the conservative
            fallback, so font replacement and subsetting retain the no-usage
            safety check.

    Returns:
        Dictionary mapping indirect font objgens, or serialized direct Type0
        font identities, to the character codes used with each font.
    """
    return _merge_font_usage(_collect_raw_font_usage(pdf), require_resolved_font)


_RawFontUsage = tuple[
    dict[_ObjectKey, set[CharacterCode]], dict[_ObjectKey, set[CharacterCode]]
]


def _merge_font_usage(
    raw: _RawFontUsage, require_resolved_font: bool
) -> dict[_ObjectKey, set[CharacterCode]]:
    """Combine resolved and unresolved usage into one fresh usage map."""
    usage_raw, unresolved_usage = raw
    usage = {key: set(codes) for key, codes in usage_raw.items()}
    for font_key, codes in unresolved_usage.items():
        if not require_resolved_font or font_key in usage_raw:
            usage.setdefault(font_key, set()).update(codes)
    return usage


def _collect_raw_font_usage(pdf: pikepdf.Pdf) -> _RawFontUsage:
    """Collect (resolved, unresolved) font usage for the whole document.

    Resolved usage depends only on each (stream, resources) context, so a
    document-wide walk that visits every context once gives the same result
    as walking every page. Unresolved text (shown before any font is set)
    is attributed to the fonts of each calling page instead; if any occurs,
    the collection is repeated page by page so every page's fonts get it.
    """
    parse_cache: dict[_ObjectKey, list] = {}
    for document_wide in (True, False):
        usage: dict[_ObjectKey, set[CharacterCode]] = {}
        unresolved_usage: dict[_ObjectKey, set[CharacterCode]] = {}
        saw_unresolved: list[bool] = []
        walked: set[_ContextKey] | None = set() if document_wide else None
        for page in pdf.pages:
            # A nested stream can inherit a font even with its own empty
            # Resources. Keep its unresolved text in every possible calling
            # font on this page.
            # (Only the page-by-page pass needs them; the first pass only
            # detects unresolved text.)
            page_fonts = (
                ()
                if document_wide
                else tuple(font for _name, font in iter_all_page_fonts(page))
            )
            for stream_owner, resources in _iter_content_streams_with_resources(
                page, walked
            ):
                _process_content_stream(
                    stream_owner,
                    resources,
                    usage,
                    page_fonts,
                    unresolved_usage,
                    parse_cache,
                    saw_unresolved,
                )
        if not saw_unresolved:
            break
    return usage, unresolved_usage


def _process_content_stream(
    stream_owner: pikepdf.Object,
    resources: pikepdf.Object,
    usage: dict[_ObjectKey, set[CharacterCode]],
    page_fonts: tuple[pikepdf.Object, ...],
    unresolved_usage: dict[_ObjectKey, set[CharacterCode]],
    parse_cache: dict[_ObjectKey, list] | None = None,
    saw_unresolved: list[bool] | None = None,
) -> None:
    """Parses a content stream and records character code usage.

    Args:
        stream_owner: Object that owns the content stream (page or XObject).
        resources: Resources dictionary for font resolution.
        usage: Accumulator mapping font objgen -> used character codes.
        parse_cache: Optional per-collection cache of parsed streams.
    """
    cache_key = None
    if parse_cache is not None and isinstance(stream_owner, pikepdf.Stream):
        cache_key = _object_identity(stream_owner)
    if cache_key is not None and cache_key in parse_cache:
        instructions = parse_cache[cache_key]
    else:
        try:
            # Only these operators affect glyph usage; the whitelist also
            # spares pikepdf building objects for inline image data.
            instructions = list(
                pikepdf.parse_content_stream(stream_owner, _USAGE_OPERATORS)
            )
        except Exception:
            instructions = []
        if not any(operator in _TEXT_OPERATORS for _, operator in instructions):
            instructions = []
        if cache_key is not None:
            parse_cache[cache_key] = instructions
    if not instructions:
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

    def record_codes(operand: pikepdf.Object, font_key: _ObjectKey) -> None:
        if current_font_is_cid:
            try:
                raw = bytes(operand)
            except Exception:
                return
            ranges = current_code_space_ranges or ((b"\x00\x00", b"\xff\xff"),)
            codes = split_cmap_codes(raw, ranges)
            if codes:
                usage.setdefault(font_key, set()).update(codes)
            return

        codes = _extract_char_codes(operand, False)
        if codes:
            usage.setdefault(font_key, set()).update(codes)

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
        if saw_unresolved is not None:
            saw_unresolved.append(True)
        strings = list(text_operands(operands, operator))
        for font_obj in page_fonts:
            try:
                font_obj = _resolve_indirect(font_obj)
                objgen = font_obj.objgen
            except Exception:
                continue
            is_cid = _is_cidfont(font_obj)
            if objgen == (0, 0) and not is_cid:
                continue
            font_key = _object_identity(font_obj)
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
                    unresolved_usage.setdefault(font_key, set()).update(codes)

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
        elif operator == pikepdf.Operator("gs"):
            states = resources.get("/ExtGState")
            if operands and isinstance(states, pikepdf.Dictionary):
                state = states.get(str(operands[0]))
                if isinstance(state, pikepdf.Dictionary):
                    font = state.get("/Font")
                    if isinstance(font, pikepdf.Array) and len(font) == 2:
                        font_obj = font[0]
                        if isinstance(font_obj, pikepdf.Dictionary):
                            current_font = font_obj
                            current_font_is_cid = _is_cidfont(font_obj)
                            current_code_space_ranges = get_font_code_space_ranges(
                                font_obj
                            )
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

            if objgen == (0, 0) and not current_font_is_cid:
                continue

            font_key = _object_identity(current_font)
            for operand in text_operands(operands, operator):
                record_codes(operand, font_key)
