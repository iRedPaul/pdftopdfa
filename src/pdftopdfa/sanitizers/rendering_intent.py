# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Content stream sanitization for PDF/A compliance.

This module handles three related requirements:
1. Replace invalid rendering intent operands for ``ri`` operators.
2. Remove undefined content stream operators (ISO 32000-1 only).
3. Ensure content streams use explicit associated ``/Resources`` dictionaries
   instead of inherited resource names.
"""

import logging
import re
import warnings

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..exceptions import ConversionError
from ..fonts.glyph_usage import find_ambiguous_resource_context_streams
from ..utils import log_suppressed_error
from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

VALID_RENDERING_INTENTS: frozenset[str] = frozenset(
    {
        "/RelativeColorimetric",
        "/AbsoluteColorimetric",
        "/Perceptual",
        "/Saturation",
    }
)

# Operators defined in ISO 32000-1 for page/form/pattern/text content streams.
VALID_CONTENT_STREAM_OPERATORS: frozenset[str] = frozenset(
    {
        # General graphics state
        "w",
        "J",
        "j",
        "M",
        "d",
        "ri",
        "i",
        "gs",
        "q",
        "Q",
        "cm",
        # Special graphics state
        "BX",
        "EX",
        # Path construction
        "m",
        "l",
        "c",
        "v",
        "y",
        "h",
        "re",
        # Path painting
        "S",
        "s",
        "f",
        "F",
        "f*",
        "B",
        "B*",
        "b",
        "b*",
        "n",
        # Clipping paths
        "W",
        "W*",
        # Text objects/state/positioning/showing
        "BT",
        "ET",
        "Tc",
        "Tw",
        "Tz",
        "TL",
        "Tf",
        "Tr",
        "Ts",
        "Td",
        "TD",
        "Tm",
        "T*",
        "Tj",
        "TJ",
        "'",
        '"',
        # Type3 fonts
        "d0",
        "d1",
        # Color
        "CS",
        "cs",
        "SC",
        "SCN",
        "sc",
        "scn",
        "G",
        "g",
        "RG",
        "rg",
        "K",
        "k",
        # Shadings and external objects
        "sh",
        "Do",
        # Marked content
        "MP",
        "DP",
        "BMC",
        "BDC",
        "EMC",
    }
)

_DEFAULT_INTENT = Name.RelativeColorimetric
type _ObjectIdentity = tuple[int, int] | tuple[str, bytes]

# Expected operand counts for critical operators (veraPDF checks these).
# value is (count, validator) where validator checks operand types.
_OPERATOR_ARG_COUNTS: dict[str, tuple[int, str]] = {
    "m": (2, "numeric"),  # moveto: x y
    "l": (2, "numeric"),  # lineto: x y
    "re": (4, "numeric"),  # rectangle: x y w h
    "rg": (3, "numeric"),  # setrgbcolor (nonstroking): r g b
    "RG": (3, "numeric"),  # setrgbcolor (stroking): r g b
    "k": (4, "numeric"),  # setcmykcolor (nonstroking): c m y k
    "K": (4, "numeric"),  # setcmykcolor (stroking): c m y k
    "g": (1, "numeric"),  # setgraycolor (nonstroking): gray
    "G": (1, "numeric"),  # setgraycolor (stroking): gray
    "cm": (6, "numeric"),  # concat matrix: a b c d e f
    "d": (2, "mixed"),  # setdash: array phase
}


def _is_numeric(operand) -> bool:
    """Return True if operand is a numeric type (int or float)."""
    if isinstance(operand, (int, float)):
        return True
    # pikepdf may wrap numerics as objects
    try:
        float(operand)
        return True
    except (TypeError, ValueError):
        return False


def _check_operator_args(operator_name: str, operands: list) -> bool:
    """Return True if operator has the correct number and type of arguments."""
    spec = _OPERATOR_ARG_COUNTS.get(operator_name)
    if spec is None:
        return True  # no validation rule for this operator

    expected_count, arg_type = spec

    if len(operands) != expected_count:
        return False

    if arg_type == "numeric":
        return all(_is_numeric(op) for op in operands)
    elif arg_type == "mixed":
        # `d` operator: [array] number
        if expected_count == 2:
            return isinstance(operands[0], pikepdf.Array) and _is_numeric(operands[1])
    return True


def _visit_once(obj, visited: set[tuple[int, int]]) -> bool:
    """Return True once per object, for both indirect and direct objects."""
    try:
        objgen = obj.objgen
    except Exception:
        return True  # direct object, always process
    if objgen == (0, 0):
        return True  # direct object, always process
    if objgen in visited:
        return False
    visited.add(objgen)
    return True


def _stream_identity(stream: Stream) -> _ObjectIdentity:
    """Return the identity used by resource-context alias detection."""
    return _object_identity(stream)


def _object_identity(obj) -> _ObjectIdentity:
    """Return a stable resource-context equivalence key.

    Equal direct objects intentionally share a key. Mutating traversals must
    instead process direct objects independently and deduplicate only objgens.
    """
    objgen = obj.objgen
    return objgen if objgen != (0, 0) else ("direct", obj.unparse())


def _clone_stream(pdf: Pdf, source: Stream) -> Stream:
    """Clone a stream after decoding it, preserving its non-filter entries."""
    clone = pdf.make_stream(source.read_bytes())
    omitted = {"/Length", "/Filter", "/DecodeParms"}
    for key in list(source.keys()):
        if str(key) not in omitted:
            value = source[key]
            if str(key) == "/Resources":
                resources = _resolve_indirect(value)
                if isinstance(resources, Dictionary):
                    value = _clone_resources_for_context(resources)
            clone[key] = value
    return clone


def _clone_resource_context_streams(pdf: Pdf) -> int:
    """Clone streams that are reused under different resource dictionaries."""
    first_context: dict[
        _ObjectIdentity,
        _ObjectIdentity,
    ] = {}
    processed: set[tuple[_ObjectIdentity, _ObjectIdentity]] = set()
    active: set[_ObjectIdentity] = set()
    processed_resources: set[tuple[int, int]] = set()
    processed_type3: set[tuple[_ObjectIdentity, _ObjectIdentity]] = set()
    cloned = 0

    def resource_tasks(resources) -> list[tuple]:
        resources = _resolve_indirect(resources)
        if not isinstance(resources, Dictionary):
            return []
        if not _visit_once(resources, processed_resources):
            return []
        discovered: list[tuple] = []

        xobjects = _resolve_indirect(resources.get("/XObject"))
        if isinstance(xobjects, Dictionary):
            for name in list(xobjects.keys()):
                candidate = _resolve_indirect(xobjects[name])
                if (
                    isinstance(candidate, Stream)
                    and str(candidate.get("/Subtype")) == "/Form"
                ):
                    discovered.append(("stream", xobjects, name, resources))

        patterns = _resolve_indirect(resources.get("/Pattern"))
        if isinstance(patterns, Dictionary):
            for name in list(patterns.keys()):
                candidate = _resolve_indirect(patterns[name])
                if (
                    isinstance(candidate, Stream)
                    and int(candidate.get("/PatternType", 0)) == 1
                ):
                    discovered.append(("stream", patterns, name, resources))

        extgstates = _resolve_indirect(resources.get("/ExtGState"))
        if isinstance(extgstates, Dictionary):
            for name in list(extgstates.keys()):
                extgstate = _resolve_indirect(extgstates[name])
                if not isinstance(extgstate, Dictionary):
                    continue
                smask = _resolve_indirect(extgstate.get("/SMask"))
                if isinstance(smask, Dictionary) and isinstance(
                    _resolve_indirect(smask.get("/G")), Stream
                ):
                    discovered.append(("stream", smask, Name.G, resources))

        fonts = _resolve_indirect(resources.get("/Font"))
        if isinstance(fonts, Dictionary):
            for name in list(fonts.keys()):
                font = _resolve_indirect(fonts[name])
                if (
                    not isinstance(font, Dictionary)
                    or str(font.get("/Subtype")) != "/Type3"
                ):
                    continue
                font_resources = _resolve_indirect(font.get("/Resources"))
                if not isinstance(font_resources, Dictionary):
                    font_resources = resources
                font_context = (
                    _object_identity(font),
                    _object_identity(font_resources),
                )
                if font_context in processed_type3:
                    continue
                processed_type3.add(font_context)
                charprocs = _resolve_indirect(font.get("/CharProcs"))
                if isinstance(charprocs, Dictionary):
                    discovered.extend(
                        ("stream", charprocs, char_name, font_resources)
                        for char_name in list(charprocs.keys())
                        if isinstance(_resolve_indirect(charprocs[char_name]), Stream)
                    )
                if font_resources is not resources:
                    discovered.append(("resources", font_resources, None, None))
        return discovered

    def appearance_tasks(container, key, page_resources) -> list[tuple]:
        entry = _resolve_indirect(container[key])
        if isinstance(entry, Stream):
            return [("stream", container, key, page_resources)]
        if not isinstance(entry, Dictionary):
            return []
        return [
            ("stream", entry, state, page_resources)
            for state in list(entry.keys())
            if isinstance(_resolve_indirect(entry[state]), Stream)
        ]

    for page in pdf.pages:
        page_dict = _resolve_indirect(page.obj)
        page_resources = _resolve_indirect(page_dict.get("/Resources"))
        if not isinstance(page_resources, Dictionary):
            page_resources = _get_inherited_page_resources(page_dict)
        if not isinstance(page_resources, Dictionary):
            page_resources = Dictionary()

        tasks: list[tuple] = []
        contents = _resolve_indirect(page_dict.get("/Contents"))
        if isinstance(contents, Stream):
            tasks.append(("stream", page_dict, Name.Contents, page_resources))
        elif isinstance(contents, Array):
            tasks.extend(
                ("stream", contents, index, page_resources)
                for index in range(len(contents))
                if isinstance(_resolve_indirect(contents[index]), Stream)
            )

        tasks.append(("resources", page_resources, None, None))

        annots = _resolve_indirect(page_dict.get("/Annots"))
        if isinstance(annots, Array):
            for annot in annots:
                annot = _resolve_indirect(annot)
                if not isinstance(annot, Dictionary):
                    continue
                ap = _resolve_indirect(annot.get("/AP"))
                if not isinstance(ap, Dictionary):
                    continue
                for ap_key in (Name.N, Name.R, Name.D):
                    if ap_key in ap:
                        tasks.extend(appearance_tasks(ap, ap_key, page_resources))

        tasks.reverse()
        while tasks:
            kind, container, key, parent_resources = tasks.pop()
            if kind == "exit":
                active.discard(container)
                continue
            if kind == "resources":
                discovered = resource_tasks(container)
                tasks.extend(reversed(discovered))
                continue

            stream = _resolve_indirect(container[key])
            if not isinstance(stream, Stream):
                continue
            stream_key = _stream_identity(stream)
            context_key = _object_identity(parent_resources)
            if stream_key in active or (stream_key, context_key) in processed:
                continue

            prior_context = first_context.setdefault(stream_key, context_key)
            if prior_context != context_key:
                stream = _clone_stream(pdf, stream)
                container[key] = stream
                stream_key = _stream_identity(stream)
                first_context[stream_key] = context_key
                cloned += 1

            processed.add((stream_key, context_key))
            active.add(stream_key)
            tasks.append(("exit", stream_key, None, None))
            own_resources = _resolve_indirect(stream.get("/Resources"))
            nested_resources = (
                own_resources
                if isinstance(own_resources, Dictionary)
                else parent_resources
            )
            tasks.append(("resources", nested_resources, None, None))

    return cloned


def _clone_resources_shallow(
    resources: Dictionary, excluded_keys: frozenset[str] = frozenset()
) -> Dictionary:
    """Create a shallow clone of a resources dictionary."""
    cloned = Dictionary()
    for key in list(resources.keys()):
        if str(key) in excluded_keys:
            continue
        cloned[key] = resources[key]
    return cloned


def _clone_resources_for_context(resources: Dictionary) -> Dictionary:
    """Clone mutable resource containers while sharing their leaf objects."""
    cloned = _clone_resources_shallow(resources)
    for category_name in ("/XObject", "/Pattern", "/Font", "/ExtGState"):
        category = _resolve_indirect(cloned.get(category_name))
        if not isinstance(category, Dictionary):
            continue
        category_clone = _clone_resources_shallow(category)
        cloned[category_name] = category_clone
        if category_name != "/ExtGState":
            continue
        for name in list(category_clone.keys()):
            extgstate = _resolve_indirect(category_clone[name])
            if not isinstance(extgstate, Dictionary):
                continue
            extgstate_clone = _clone_resources_shallow(extgstate)
            smask = _resolve_indirect(extgstate_clone.get("/SMask"))
            if isinstance(smask, Dictionary):
                extgstate_clone[Name.SMask] = _clone_resources_shallow(smask)
            category_clone[name] = extgstate_clone
    return cloned


def _merge_resource_dictionaries(
    target: Dictionary,
    parent: Dictionary,
    excluded_keys: frozenset[str] = frozenset(),
) -> int:
    """Merge missing resource categories/names from parent into target."""
    merged = 0
    for key in list(parent.keys()):
        if str(key) in excluded_keys:
            continue
        if key not in target:
            target[key] = parent[key]
            merged += 1
            continue

        tgt_val = _resolve_indirect(target[key])
        par_val = _resolve_indirect(parent[key])
        if isinstance(tgt_val, Dictionary) and isinstance(par_val, Dictionary):
            for name in list(par_val.keys()):
                if name not in tgt_val:
                    tgt_val[name] = par_val[name]
                    merged += 1
    return merged


def _ensure_associated_resources(
    owner: Dictionary | Stream,
    parent_resources,
    excluded_keys: frozenset[str] = frozenset(),
) -> tuple[Dictionary | None, int, int]:
    """Ensure owner has explicit /Resources and merge inherited entries."""
    parent_resources = _resolve_indirect(parent_resources)
    if not isinstance(parent_resources, Dictionary):
        parent_resources = None

    resources = owner.get("/Resources")
    resources = _resolve_indirect(resources) if resources is not None else None

    if not isinstance(resources, Dictionary):
        if isinstance(parent_resources, Dictionary):
            owner[Name.Resources] = _clone_resources_shallow(
                parent_resources, excluded_keys=excluded_keys
            )
        else:
            owner[Name.Resources] = Dictionary()
        resources = _resolve_indirect(owner.get("/Resources"))
        return resources if isinstance(resources, Dictionary) else None, 1, 0

    merged = 0
    if isinstance(parent_resources, Dictionary):
        merged = _merge_resource_dictionaries(
            resources, parent_resources, excluded_keys=excluded_keys
        )
    return resources, 0, merged


def _get_inherited_page_resources(page_dict: Dictionary):
    """Return inherited page resources from the page tree, if present."""
    seen: set[tuple[int, int]] = set()
    parent = _resolve_indirect(page_dict.get("/Parent"))
    while isinstance(parent, Dictionary):
        objgen = parent.objgen
        if objgen != (0, 0):
            if objgen in seen:
                break
            seen.add(objgen)
        parent_resources = parent.get("/Resources")
        if parent_resources is not None:
            return _resolve_indirect(parent_resources)
        parent = _resolve_indirect(parent.get("/Parent"))
    return None


def _sanitize_stream_operators(
    stream_obj: Stream,
) -> tuple[int, int, int, int]:
    """Replace invalid ``ri`` operands, remove undefined operators,
    validate operator argument counts, and fix invalid ``/Intent`` in
    inline images.

    Returns:
        Tuple of (ri_fixed, undefined_removed, inline_intents_fixed,
        bad_args_removed).
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unexpected end of stream", category=UserWarning
            )
            instructions = list(pikepdf.parse_content_stream(stream_obj))
    except Exception:
        return 0, 0, 0, 0

    ri_fixed = 0
    undefined_removed = 0
    bad_args_removed = 0
    inline_intents: dict[str, int] = {}
    new_instructions = []

    for item in instructions:
        # ContentStreamInlineImage items are not regular instructions
        if isinstance(item, pikepdf.ContentStreamInlineImage):
            try:
                intent = item.iimage.obj.get("/Intent")
                if isinstance(intent, Name) and (
                    str(intent) not in VALID_RENDERING_INTENTS
                ):
                    key = str(intent)
                    inline_intents[key] = inline_intents.get(key, 0) + 1
            except Exception:
                pass
            new_instructions.append(item)
            continue

        operands, operator = item.operands, item.operator
        operator_name = str(operator)

        if operator_name not in VALID_CONTENT_STREAM_OPERATORS:
            undefined_removed += 1
            logger.debug(
                "Removed undefined content stream operator: %s",
                operator_name,
            )
            continue

        # Validate argument counts for critical operators
        if not _check_operator_args(operator_name, operands):
            bad_args_removed += 1
            logger.warning(
                "Removed operator '%s' with %d operand(s) (expected %d)",
                operator_name,
                len(operands),
                _OPERATOR_ARG_COUNTS[operator_name][0],
            )
            continue

        if operator_name == "ri" and operands:
            operand = operands[0]
            if isinstance(operand, Name) and (
                str(operand) not in VALID_RENDERING_INTENTS
            ):
                new_instructions.append(
                    pikepdf.ContentStreamInstruction([_DEFAULT_INTENT], operator)
                )
                ri_fixed += 1
                logger.debug(
                    "Replaced invalid ri operand %s with /RelativeColorimetric",
                    operand,
                )
                continue

        new_instructions.append(item)

    inline_fixed = sum(inline_intents.values())
    has_changes = (
        ri_fixed > 0
        or undefined_removed > 0
        or inline_fixed > 0
        or bad_args_removed > 0
    )

    if has_changes:
        # Always use parse-and-unparse cycle for structured output
        data = pikepdf.unparse_content_stream(new_instructions)

        if inline_fixed > 0:
            # Replace invalid intents only within inline image headers
            # (between BI and ID markers) to avoid false matches elsewhere.
            intents_to_fix = inline_intents

            def _fix_inline_header(m: re.Match[bytes]) -> bytes:
                header = m.group(0)
                for intent_str in intents_to_fix:
                    old = f"/Intent {intent_str}".encode()
                    header = header.replace(old, b"/Intent /RelativeColorimetric")
                return header

            data = re.sub(
                rb"\bBI\b(.*?)\bID\b",
                _fix_inline_header,
                data,
                flags=re.DOTALL,
            )
            for intent_str in inline_intents:
                logger.debug(
                    "Replaced invalid inline image /Intent %s "
                    "with /RelativeColorimetric",
                    intent_str,
                )

        stream_obj.write(data)

    return ri_fixed, undefined_removed, inline_fixed, bad_args_removed


def _sanitize_page_contents(
    page_dict: Dictionary,
) -> tuple[int, int, int, int]:
    """Sanitize operators in page ``/Contents`` (stream or array)."""
    contents = page_dict.get("/Contents")
    if contents is None:
        return 0, 0, 0, 0

    contents = _resolve_indirect(contents)
    ri_fixed = 0
    undefined_removed = 0
    inline_fixed = 0
    bad_args_removed = 0

    if isinstance(contents, Stream):
        ri, undef, inl, bad = _sanitize_stream_operators(contents)
        ri_fixed += ri
        undefined_removed += undef
        inline_fixed += inl
        bad_args_removed += bad
    elif isinstance(contents, Array):
        for item in contents:
            item = _resolve_indirect(item)
            if isinstance(item, Stream):
                ri, undef, inl, bad = _sanitize_stream_operators(item)
                ri_fixed += ri
                undefined_removed += undef
                inline_fixed += inl
                bad_args_removed += bad

    return ri_fixed, undefined_removed, inline_fixed, bad_args_removed


def _iter_form_xobjects(resources, visited: set[tuple[int, int]]):
    """Yield Form XObjects from a resources dictionary with cycle detection."""
    resources = _resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return

    xobjects = resources.get("/XObject")
    xobjects = _resolve_indirect(xobjects) if xobjects else None
    if not isinstance(xobjects, Dictionary):
        return

    for xobj_name in list(xobjects.keys()):
        xobj = _resolve_indirect(xobjects[xobj_name])
        if not isinstance(xobj, Stream):
            continue
        if str(xobj.get("/Subtype")) != "/Form":
            continue

        if not _visit_once(xobj, visited):
            continue
        yield xobj


def _iter_tiling_patterns(resources, visited: set[tuple[int, int]]):
    """Yield tiling pattern streams (PatternType 1) with cycle detection."""
    resources = _resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return

    patterns = resources.get("/Pattern")
    patterns = _resolve_indirect(patterns) if patterns else None
    if not isinstance(patterns, Dictionary):
        return

    for pattern_name in list(patterns.keys()):
        pattern = _resolve_indirect(patterns[pattern_name])
        if not isinstance(pattern, Stream):
            continue
        if int(pattern.get("/PatternType", 0)) != 1:
            continue

        if not _visit_once(pattern, visited):
            continue
        yield pattern


def _iter_soft_mask_groups(resources, visited: set[tuple[int, int]]):
    """Yield transparency-group streams referenced by soft masks."""
    resources = _resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return

    extgstates = _resolve_indirect(resources.get("/ExtGState"))
    if not isinstance(extgstates, Dictionary):
        return

    for name in list(extgstates.keys()):
        extgstate = _resolve_indirect(extgstates[name])
        if not isinstance(extgstate, Dictionary):
            continue
        smask = _resolve_indirect(extgstate.get("/SMask"))
        if not isinstance(smask, Dictionary):
            continue
        group = _resolve_indirect(smask.get("/G"))
        if not isinstance(group, Stream) or not _visit_once(group, visited):
            continue
        yield group


def _iter_type3_fonts(resources, visited: set[tuple[int, int]]):
    """Yield Type3 fonts from a resources dictionary with cycle detection."""
    resources = _resolve_indirect(resources)
    if not isinstance(resources, Dictionary):
        return

    fonts = resources.get("/Font")
    fonts = _resolve_indirect(fonts) if fonts else None
    if not isinstance(fonts, Dictionary):
        return

    for font_name in list(fonts.keys()):
        font = _resolve_indirect(fonts[font_name])
        if not isinstance(font, Dictionary):
            continue
        if str(font.get("/Subtype")) != "/Type3":
            continue
        if not _visit_once(font, visited):
            continue
        yield font_name, font


def _ensure_explicit_resources_in_resource_graph(
    resources,
    visited_forms: set[tuple[int, int]],
    visited_fonts: set[tuple[int, int]],
    visited_patterns: set[tuple[int, int]],
    ambiguous_streams: set[_ObjectIdentity],
) -> tuple[int, int]:
    """Ensure explicit resources for nested content stream containers."""
    resources_added = 0
    resources_merged = 0
    pending = [resources]
    processed_resources: set[tuple[int, int]] = set()
    processed_owners: set[tuple[int, int]] = set()

    while pending:
        parent = _resolve_indirect(pending.pop())
        if not isinstance(parent, Dictionary):
            continue
        if not _visit_once(parent, processed_resources):
            continue

        for form in _iter_form_xobjects(parent, visited_forms):
            if not _visit_once(form, processed_owners):
                continue
            form_key = _stream_identity(form)
            if form_key in ambiguous_streams:
                form_resources = _resolve_indirect(form.get("/Resources"))
                if not isinstance(form_resources, Dictionary):
                    continue
                added = merged = 0
            else:
                form_resources, added, merged = _ensure_associated_resources(
                    form, parent
                )
            resources_added += added
            resources_merged += merged
            if isinstance(form_resources, Dictionary):
                pending.append(form_resources)

        for _font_name, font in _iter_type3_fonts(parent, visited_fonts):
            if not _visit_once(font, processed_owners):
                continue
            font_resources, added, merged = _ensure_associated_resources(
                font,
                parent,
                excluded_keys=frozenset({"/Font"}),
            )
            resources_added += added
            resources_merged += merged
            if isinstance(font_resources, Dictionary):
                pending.append(font_resources)

        for pattern in _iter_tiling_patterns(parent, visited_patterns):
            if not _visit_once(pattern, processed_owners):
                continue
            pattern_key = _stream_identity(pattern)
            if pattern_key in ambiguous_streams:
                pattern_resources = _resolve_indirect(pattern.get("/Resources"))
                if not isinstance(pattern_resources, Dictionary):
                    continue
                added = merged = 0
            else:
                pattern_resources, added, merged = _ensure_associated_resources(
                    pattern, parent
                )
            resources_added += added
            resources_merged += merged
            if isinstance(pattern_resources, Dictionary):
                pending.append(pattern_resources)

        for group in _iter_soft_mask_groups(parent, visited_forms):
            if not _visit_once(group, processed_owners):
                continue
            group_resources, added, merged = _ensure_associated_resources(group, parent)
            resources_added += added
            resources_merged += merged
            if isinstance(group_resources, Dictionary):
                pending.append(group_resources)

    return resources_added, resources_merged


def _sanitize_operators_in_resource_graph(
    resources,
    visited_forms: set[tuple[int, int]],
    visited_fonts: set[tuple[int, int]],
    visited_patterns: set[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Sanitize operators in nested forms, Type3 CharProcs and patterns."""
    ri_fixed = 0
    undefined_removed = 0
    inline_fixed = 0
    bad_args_removed = 0
    pending = [resources]
    processed_resources: set[tuple[int, int]] = set()
    processed_streams: set[_ObjectIdentity] = set()

    def sanitize_stream(stream: Stream) -> None:
        nonlocal ri_fixed, undefined_removed, inline_fixed, bad_args_removed
        stream_key = _stream_identity(stream)
        if stream_key in processed_streams:
            return
        processed_streams.add(stream_key)
        ri, undefined, inline, bad_args = _sanitize_stream_operators(stream)
        ri_fixed += ri
        undefined_removed += undefined
        inline_fixed += inline
        bad_args_removed += bad_args

    while pending:
        parent = _resolve_indirect(pending.pop())
        if not isinstance(parent, Dictionary):
            continue
        if not _visit_once(parent, processed_resources):
            continue

        for form in _iter_form_xobjects(parent, visited_forms):
            sanitize_stream(form)
            pending.append(_resolve_indirect(form.get("/Resources")))

        for _font_name, font in _iter_type3_fonts(parent, visited_fonts):
            charprocs = _resolve_indirect(font.get("/CharProcs"))
            if isinstance(charprocs, Dictionary):
                for name in list(charprocs.keys()):
                    stream = _resolve_indirect(charprocs[name])
                    if isinstance(stream, Stream):
                        sanitize_stream(stream)
            pending.append(_resolve_indirect(font.get("/Resources")))

        for pattern in _iter_tiling_patterns(parent, visited_patterns):
            sanitize_stream(pattern)
            pending.append(_resolve_indirect(pattern.get("/Resources")))

        for group in _iter_soft_mask_groups(parent, visited_forms):
            sanitize_stream(group)
            pending.append(_resolve_indirect(group.get("/Resources")))

    return ri_fixed, undefined_removed, inline_fixed, bad_args_removed


def _ensure_resources_in_ap_stream(
    ap_entry,
    page_resources,
    visited_forms: set[tuple[int, int]],
    visited_fonts: set[tuple[int, int]],
    visited_patterns: set[tuple[int, int]],
    ambiguous_streams: set[_ObjectIdentity],
) -> tuple[int, int]:
    """Ensure explicit resources on AP streams and nested content streams."""
    ap_entry = _resolve_indirect(ap_entry)
    resources_added = 0
    resources_merged = 0

    if isinstance(ap_entry, Stream):
        if _stream_identity(ap_entry) in ambiguous_streams:
            ap_resources = _resolve_indirect(ap_entry.get("/Resources"))
            if not isinstance(ap_resources, Dictionary):
                return 0, 0
            added = merged = 0
        else:
            ap_resources, added, merged = _ensure_associated_resources(
                ap_entry, page_resources
            )
        resources_added += added
        resources_merged += merged
        add2, merge2 = _ensure_explicit_resources_in_resource_graph(
            ap_resources,
            visited_forms,
            visited_fonts,
            visited_patterns,
            ambiguous_streams,
        )
        resources_added += add2
        resources_merged += merge2
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                if _stream_identity(state_stream) in ambiguous_streams:
                    st_resources = _resolve_indirect(state_stream.get("/Resources"))
                    if not isinstance(st_resources, Dictionary):
                        continue
                    added = merged = 0
                else:
                    st_resources, added, merged = _ensure_associated_resources(
                        state_stream, page_resources
                    )
                resources_added += added
                resources_merged += merged
                add2, merge2 = _ensure_explicit_resources_in_resource_graph(
                    st_resources,
                    visited_forms,
                    visited_fonts,
                    visited_patterns,
                    ambiguous_streams,
                )
                resources_added += add2
                resources_merged += merge2

    return resources_added, resources_merged


def _sanitize_ap_stream(
    ap_entry,
    visited_forms: set[tuple[int, int]],
    visited_fonts: set[tuple[int, int]],
    visited_patterns: set[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Sanitize operators in AP streams and nested content streams."""
    ap_entry = _resolve_indirect(ap_entry)
    ri_fixed = 0
    undefined_removed = 0
    inline_fixed = 0
    bad_args_removed = 0

    if isinstance(ap_entry, Stream):
        ri, undef, inl, bad = _sanitize_stream_operators(ap_entry)
        ri_fixed += ri
        undefined_removed += undef
        inline_fixed += inl
        bad_args_removed += bad
        ap_resources = _resolve_indirect(ap_entry.get("/Resources"))
        ri2, undef2, inl2, bad2 = _sanitize_operators_in_resource_graph(
            ap_resources, visited_forms, visited_fonts, visited_patterns
        )
        ri_fixed += ri2
        undefined_removed += undef2
        inline_fixed += inl2
        bad_args_removed += bad2
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                ri, undef, inl, bad = _sanitize_stream_operators(state_stream)
                ri_fixed += ri
                undefined_removed += undef
                inline_fixed += inl
                bad_args_removed += bad

                st_resources = _resolve_indirect(state_stream.get("/Resources"))
                ri2, undef2, inl2, bad2 = _sanitize_operators_in_resource_graph(
                    st_resources,
                    visited_forms,
                    visited_fonts,
                    visited_patterns,
                )
                ri_fixed += ri2
                undefined_removed += undef2
                inline_fixed += inl2
                bad_args_removed += bad2

    return ri_fixed, undefined_removed, inline_fixed, bad_args_removed


def _sanitize_image_intents_in_resource_graph(
    resources, visited: set[tuple[int, int]]
) -> int:
    """Fix invalid ``/Intent`` on Image XObjects throughout a resource graph.

    Recursively traverses XObjects, tiling patterns and Type3 fonts to find
    Image XObjects with invalid ``/Intent`` and replaces with the default
    ``/RelativeColorimetric``.
    """
    fixed = 0
    pending = [resources]
    processed_resources: set[tuple[int, int]] = set()

    while pending:
        parent = _resolve_indirect(pending.pop())
        if not isinstance(parent, Dictionary):
            continue
        if not _visit_once(parent, processed_resources):
            continue

        xobjects = _resolve_indirect(parent.get("/XObject"))
        if isinstance(xobjects, Dictionary):
            for xobj_name in list(xobjects.keys()):
                xobj = _resolve_indirect(xobjects[xobj_name])
                if not isinstance(xobj, Stream):
                    continue
                subtype = str(xobj.get("/Subtype", ""))
                if subtype == "/Image":
                    intent = _resolve_indirect(xobj.get("/Intent"))
                    if isinstance(intent, Name) and (
                        str(intent) not in VALID_RENDERING_INTENTS
                    ):
                        xobj[Name.Intent] = _DEFAULT_INTENT
                        fixed += 1
                        logger.debug(
                            "Replaced invalid /Intent %s on Image XObject %s",
                            intent,
                            xobj_name,
                        )
                elif subtype == "/Form" and _visit_once(xobj, visited):
                    pending.append(_resolve_indirect(xobj.get("/Resources")))

        patterns = _resolve_indirect(parent.get("/Pattern"))
        if isinstance(patterns, Dictionary):
            for name in list(patterns.keys()):
                pattern = _resolve_indirect(patterns[name])
                if (
                    isinstance(pattern, Stream)
                    and int(pattern.get("/PatternType", 0)) == 1
                    and _visit_once(pattern, visited)
                ):
                    pending.append(_resolve_indirect(pattern.get("/Resources")))

        pending.extend(
            _resolve_indirect(group.get("/Resources"))
            for group in _iter_soft_mask_groups(parent, visited)
        )

    return fixed


def _sanitize_image_intents_in_ap(ap_entry, visited: set[tuple[int, int]]) -> int:
    """Fix invalid ``/Intent`` on Image XObjects in AP streams."""
    ap_entry = _resolve_indirect(ap_entry)
    fixed = 0

    if isinstance(ap_entry, Stream):
        ap_resources = _resolve_indirect(ap_entry.get("/Resources"))
        fixed += _sanitize_image_intents_in_resource_graph(ap_resources, visited)
    elif isinstance(ap_entry, Dictionary):
        for state_name in list(ap_entry.keys()):
            state_stream = _resolve_indirect(ap_entry[state_name])
            if isinstance(state_stream, Stream):
                st_resources = _resolve_indirect(state_stream.get("/Resources"))
                fixed += _sanitize_image_intents_in_resource_graph(
                    st_resources, visited
                )

    return fixed


def sanitize_rendering_intent(pdf: Pdf) -> dict[str, int]:
    """Sanitize content streams for rule 6.2.2 + rendering intents.

    Traverses:
    - Page contents
    - Form XObjects (recursive)
    - Type3 CharProcs (recursive resources)
    - Tiling patterns (recursive resources)
    - Annotation AP streams

    Returns:
        Dictionary with:
        - ``ri_operators_fixed``: invalid ``ri`` operands replaced
        - ``undefined_operators_removed``: unknown operators removed
        - ``resources_dictionaries_added``: explicit ``/Resources`` added
        - ``resources_entries_merged``: inherited resource names copied
        - ``image_intents_fixed``: invalid ``/Intent`` on images replaced
        - ``bad_args_operators_removed``: operators with wrong arg counts
    """
    ri_total = 0
    undefined_total = 0
    inline_total = 0
    bad_args_total = 0
    image_intents_total = 0
    resources_added_total = 0
    resources_merged_total = 0

    context_streams_cloned = _clone_resource_context_streams(pdf)
    ambiguous_streams = find_ambiguous_resource_context_streams(pdf)
    if ambiguous_streams:
        raise ConversionError(
            "Cannot safely associate reused content streams with their "
            "resource contexts"
        )

    ensure_forms_visited: set[tuple[int, int]] = set()
    ensure_fonts_visited: set[tuple[int, int]] = set()
    ensure_patterns_visited: set[tuple[int, int]] = set()

    sanitize_forms_visited: set[tuple[int, int]] = set()
    sanitize_fonts_visited: set[tuple[int, int]] = set()
    sanitize_patterns_visited: set[tuple[int, int]] = set()

    image_intents_visited: set[tuple[int, int]] = set()
    for page_num, page in enumerate(pdf.pages, start=1):
        try:
            page_dict = _resolve_indirect(page.obj)
            parent_resources = _get_inherited_page_resources(page_dict)

            # 1) Ensure page /Resources is explicit and self-contained
            page_resources, added, merged = _ensure_associated_resources(
                page_dict, parent_resources
            )
            resources_added_total += added
            resources_merged_total += merged

            add2, merge2 = _ensure_explicit_resources_in_resource_graph(
                page_resources,
                ensure_forms_visited,
                ensure_fonts_visited,
                ensure_patterns_visited,
                ambiguous_streams,
            )
            resources_added_total += add2
            resources_merged_total += merge2

            # Annotation AP streams are outside page /Resources graph
            annots = page_dict.get("/Annots")
            annots = _resolve_indirect(annots) if annots else None
            if isinstance(annots, Array):
                for annot in annots:
                    annot = _resolve_indirect(annot)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = _resolve_indirect(annot.get("/AP"))
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry:
                            add_ap, merge_ap = _ensure_resources_in_ap_stream(
                                ap_entry,
                                page_resources,
                                ensure_forms_visited,
                                ensure_fonts_visited,
                                ensure_patterns_visited,
                                ambiguous_streams,
                            )
                            resources_added_total += add_ap
                            resources_merged_total += merge_ap

            # 2) Sanitize operators in page contents + nested content streams
            ri, undef, inl, bad = _sanitize_page_contents(page_dict)
            ri_total += ri
            undefined_total += undef
            inline_total += inl
            bad_args_total += bad

            ri2, undef2, inl2, bad2 = _sanitize_operators_in_resource_graph(
                page_resources,
                sanitize_forms_visited,
                sanitize_fonts_visited,
                sanitize_patterns_visited,
            )
            ri_total += ri2
            undefined_total += undef2
            inline_total += inl2
            bad_args_total += bad2

            if isinstance(annots, Array):
                for annot in annots:
                    annot = _resolve_indirect(annot)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = _resolve_indirect(annot.get("/AP"))
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry:
                            ri_ap, undef_ap, inl_ap, bad_ap = _sanitize_ap_stream(
                                ap_entry,
                                sanitize_forms_visited,
                                sanitize_fonts_visited,
                                sanitize_patterns_visited,
                            )
                            ri_total += ri_ap
                            undefined_total += undef_ap
                            inline_total += inl_ap
                            bad_args_total += bad_ap

            # 3) Fix invalid /Intent on Image XObjects
            image_intents_total += _sanitize_image_intents_in_resource_graph(
                page_resources, image_intents_visited
            )

            if isinstance(annots, Array):
                for annot in annots:
                    annot = _resolve_indirect(annot)
                    if not isinstance(annot, Dictionary):
                        continue
                    ap = _resolve_indirect(annot.get("/AP"))
                    if not isinstance(ap, Dictionary):
                        continue
                    for ap_key in ("/N", "/R", "/D"):
                        ap_entry = ap.get(ap_key)
                        if ap_entry:
                            image_intents_total += _sanitize_image_intents_in_ap(
                                ap_entry, image_intents_visited
                            )

        except Exception as e:
            log_suppressed_error(
                logger,
                e,
                "Error sanitizing content streams on page %d: %s",
                page_num,
                e,
            )

    # Combine inline image intents with image XObject intents
    image_intents_total += inline_total

    if (
        ri_total > 0
        or undefined_total > 0
        or bad_args_total > 0
        or resources_added_total > 0
        or resources_merged_total > 0
        or context_streams_cloned > 0
        or image_intents_total > 0
    ):
        logger.info(
            "Content streams sanitized: %d ri fixed, %d undefined operators "
            "removed, %d bad-args operators removed, "
            "%d resources dictionaries added, %d resource entries "
            "merged, %d resource-context streams cloned, "
            "%d image intents fixed",
            ri_total,
            undefined_total,
            bad_args_total,
            resources_added_total,
            resources_merged_total,
            context_streams_cloned,
            image_intents_total,
        )

    return {
        "ri_operators_fixed": ri_total,
        "undefined_operators_removed": undefined_total,
        "bad_args_operators_removed": bad_args_total,
        "resources_dictionaries_added": resources_added_total,
        "resources_entries_merged": resources_merged_total,
        "image_intents_fixed": image_intents_total,
    }
