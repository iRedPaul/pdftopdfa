# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Preserve, repair, or create logical structure for PDF/A level A."""

from __future__ import annotations

import math
import re
import unicodedata
import warnings
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING

import pikepdf
from pikepdf import (
    Array,
    Dictionary,
    Name,
    NameTree,
    NumberTree,
    Operator,
    Stream,
    String,
)

from .exceptions import ConversionError
from .fonts.glyph_usage import _iter_content_streams_with_resources
from .sanitizers.catalog import _is_valid_bcp47
from .sanitizers.optional_content import (
    _default_optional_content_visibility,
    _DefaultOCVisibility,
)
from .utils import resolve_indirect

if TYPE_CHECKING:
    from .digital_layout import ClipPolygon, InvocationPaintState

_BDC = Operator("BDC")
_BMC = Operator("BMC")
_BT = Operator("BT")
_DP = Operator("DP")
_EMC = Operator("EMC")
_ET = Operator("ET")
_MP = Operator("MP")
_MAX_ARRAY_ITEMS = 8_191
_MAX_STRING_BYTES = 32_767
_MAX_FORM_SCAN_DEPTH = 64
_PREFLIGHT_MEMORY_LIMIT = 16 * 1024 * 1024
_TEXT_SHOW_OPERAND_COUNTS = {"Tj": 1, "TJ": 1, "'": 1, '"': 3}
_ObjectKey = tuple[int, int]
_PAINTING_OPERATORS = frozenset(
    {"Tj", "TJ", "'", '"', "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "sh"}
)
_WRAPPER_PREFIX = re.compile(rb"/Div\s*<<\s*/MCID\s+(\d+)\s*>>\s*BDC\s*")
_WRAPPER_SUFFIX = re.compile(rb"\s*EMC\s*")
# PDF 32000-1, Table 333: grouping elements may nest inside one another, so a
# /Document below one of them is as valid as a /Document below the root.
_GROUPING_STRUCTURE_TYPES = frozenset(
    {
        "/Document",
        "/Part",
        "/Art",
        "/Sect",
        "/Div",
        "/BlockQuote",
        "/Caption",
        "/TOC",
        "/TOCI",
        "/Index",
        "/NonStruct",
        "/Private",
    }
)
_STANDARD_STRUCTURE_TYPES = frozenset(
    {
        "/Document",
        "/Part",
        "/Art",
        "/Sect",
        "/Div",
        "/BlockQuote",
        "/Caption",
        "/TOC",
        "/TOCI",
        "/Index",
        "/NonStruct",
        "/Private",
        "/H",
        "/H1",
        "/H2",
        "/H3",
        "/H4",
        "/H5",
        "/H6",
        "/P",
        "/L",
        "/LI",
        "/Lbl",
        "/LBody",
        "/Table",
        "/TR",
        "/TH",
        "/TD",
        "/THead",
        "/TBody",
        "/TFoot",
        "/Span",
        "/Quote",
        "/Note",
        "/Reference",
        "/BibEntry",
        "/Code",
        "/Link",
        "/Annot",
        "/Ruby",
        "/RB",
        "/RT",
        "/RP",
        "/Warichu",
        "/WT",
        "/WP",
        "/Figure",
        "/Formula",
        "/Form",
    }
)


def _ensure_mark_info(pdf: pikepdf.Pdf) -> bool:
    mark_info = resolve_indirect(pdf.Root.get("/MarkInfo"))
    if isinstance(mark_info, Dictionary):
        updated = False
        if mark_info.get("/Marked") is not True:
            mark_info["/Marked"] = True
            updated = True
        if mark_info.get("/Suspects") is True:
            del mark_info["/Suspects"]
            updated = True
        return updated

    pdf.Root["/MarkInfo"] = pdf.make_indirect(Dictionary(Marked=True))
    return True


def _resolvable_roles(role_map: Dictionary | None) -> set[str] | None:
    roles = set(_STANDARD_STRUCTURE_TYPES)
    if role_map is None:
        return roles

    mappings: dict[str, str] = {}
    for raw_key in role_map:
        key = str(raw_key)
        mapped = resolve_indirect(role_map.get(raw_key))
        if not isinstance(mapped, Name):
            return None
        target = str(mapped)
        if key in _STANDARD_STRUCTURE_TYPES and target not in _STANDARD_STRUCTURE_TYPES:
            return None
        if key == target and key in _STANDARD_STRUCTURE_TYPES:
            continue
        mappings[key] = target

    checked: set[str] = set()
    for role in mappings:
        current = role
        path: set[str] = set()
        while current in mappings and current not in checked:
            if current in path:
                return None
            path.add(current)
            current = mappings[current]
        checked.update(path)

    for role in mappings:
        if role in roles:
            continue
        current = role
        path: list[str] = []
        while current not in roles:
            target = mappings.get(current)
            if target is None:
                return None
            path.append(current)
            current = target
        roles.update(path)
    return roles


def _effective_structure_role(
    role: object,
    role_map: Dictionary | None,
) -> str | None:
    role = resolve_indirect(role)
    if not isinstance(role, Name):
        return None
    current = str(role)
    seen: set[str] = set()
    while isinstance(role_map, Dictionary) and current in role_map:
        if current in seen:
            return None
        seen.add(current)
        mapped = resolve_indirect(role_map.get(current))
        if not isinstance(mapped, Name):
            return None
        mapped_name = str(mapped)
        if mapped_name == current:
            return current if current in _STANDARD_STRUCTURE_TYPES else None
        current = mapped_name
    return current if current in _STANDARD_STRUCTURE_TYPES else None


def _valid_structure_hierarchy(
    root: Dictionary,
    role_map: Dictionary | None,
    elements: list[Dictionary],
) -> bool:
    root_key = _object_key(root)
    if root_key is None:
        return False
    roles: dict[_ObjectKey, str] = {}
    children: dict[_ObjectKey, list[str]] = {}
    parents: dict[_ObjectKey, _ObjectKey] = {}
    for element in elements:
        key = _object_key(element)
        parent = resolve_indirect(element.get("/P"))
        parent_key = _object_key(parent)
        role = _effective_structure_role(element.get("/S"), role_map)
        if key is None or parent_key is None or role is None:
            return False
        roles[key] = role
        parents[key] = parent_key
    for key, parent_key in parents.items():
        children.setdefault(parent_key, []).append(roles[key])

    required_parents = {
        "/Document": {
            root_key,
            *(key for key, role in roles.items() if role in _GROUPING_STRUCTURE_TYPES),
        },
        "/THead": {key for key, role in roles.items() if role == "/Table"},
        "/TBody": {key for key, role in roles.items() if role == "/Table"},
        "/TFoot": {key for key, role in roles.items() if role == "/Table"},
        "/TR": {
            key
            for key, role in roles.items()
            if role in {"/Table", "/THead", "/TBody", "/TFoot"}
        },
        "/TH": {key for key, role in roles.items() if role == "/TR"},
        "/TD": {key for key, role in roles.items() if role == "/TR"},
        "/LI": {key for key, role in roles.items() if role == "/L"},
        # PDF 32000-1, Table 336: /Lbl labels list items and table-of-contents
        # items alike.
        "/Lbl": {key for key, role in roles.items() if role in {"/LI", "/TOCI"}},
        "/LBody": {key for key, role in roles.items() if role == "/LI"},
    }
    for key, role in roles.items():
        allowed_parents = required_parents.get(role)
        if allowed_parents is not None and parents[key] not in allowed_parents:
            return False

    for key, role in roles.items():
        child_roles = children.get(key, [])
        if role == "/Table":
            if not child_roles or any(
                child not in {"/Caption", "/THead", "/TBody", "/TFoot", "/TR"}
                for child in child_roles
            ):
                return False
        elif role in {"/THead", "/TBody", "/TFoot"}:
            if not child_roles or any(child != "/TR" for child in child_roles):
                return False
        elif role == "/TR":
            if not child_roles or any(
                child not in {"/TH", "/TD"} for child in child_roles
            ):
                return False
        elif role == "/L":
            if (
                not child_roles
                or any(child not in {"/Caption", "/LI"} for child in child_roles)
                or not any(child == "/LI" for child in child_roles)
            ):
                return False
        elif role == "/LI" and (
            not child_roles
            or any(child not in {"/Lbl", "/LBody"} for child in child_roles)
        ):
            return False
    return True


def _object_key(value: object) -> _ObjectKey | None:
    value = resolve_indirect(value)
    if isinstance(value, pikepdf.Page):
        value = value.obj
    if isinstance(value, pikepdf.Object) and value.is_indirect:
        return value.objgen
    return None


def _same_object(left: object, right: object) -> bool:
    left_key = _object_key(left)
    return left_key is not None and left_key == _object_key(right)


def _optional_content_is_visible(
    visibility: _DefaultOCVisibility | None,
    membership: object,
    context: str,
) -> bool:
    if visibility is None:
        return True
    try:
        return visibility.is_visible(membership)
    except ValueError as exc:
        raise ConversionError(
            f"Cannot create semantic structure: {context} has malformed "
            "optional content"
        ) from exc


def _walk_structure_k(
    value: object,
    resolvable_roles: set[str],
    elements: list[Dictionary],
    seen: set[_ObjectKey],
    expected_parent: Dictionary,
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
    object_owners: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[Dictionary | Stream, _ObjectKey, Dictionary],
    ],
    owner: Dictionary | None = None,
    inherited_page: Dictionary | None = None,
) -> bool:
    pending: list[tuple[object, Dictionary, Dictionary | None, Dictionary | None]] = [
        (value, expected_parent, owner, inherited_page)
    ]
    while pending:
        value, expected_parent, owner, inherited_page = pending.pop()
        value = resolve_indirect(value)
        if isinstance(value, int) and not isinstance(value, bool):
            if value < 0 or owner is None or inherited_page is None:
                return False
            page_key = _object_key(inherited_page)
            if page_key is None or _object_key(owner) is None:
                return False
            reference_key = (page_key, value)
            if reference_key in content_references:
                return False
            content_references[reference_key] = (
                inherited_page,
                owner,
                inherited_page,
                None,
            )
            continue
        if isinstance(value, Array):
            if len(value) > _MAX_ARRAY_ITEMS:
                return False
            pending.extend(
                (item, expected_parent, owner, inherited_page)
                for item in reversed(value)
            )
            continue
        if not isinstance(value, Dictionary):
            return False

        type_name = resolve_indirect(value.get("/Type"))
        if type_name == Name.MCR:
            mcid = resolve_indirect(value.get("/MCID"))
            if (
                owner is None
                or not isinstance(mcid, int)
                or isinstance(mcid, bool)
                or mcid < 0
            ):
                return False
            raw_stream = value.get("/Stm")
            stream = resolve_indirect(raw_stream)
            stream_subtype = (
                resolve_indirect(stream.get("/Subtype"))
                if isinstance(stream, Stream)
                else None
            )
            if raw_stream is not None and (
                not isinstance(stream, Stream)
                or stream_subtype not in (None, Name.Form)
            ):
                return False
            raw_stream_owner = value.get("/StmOwn")
            stream_owner = resolve_indirect(raw_stream_owner)
            if raw_stream_owner is not None and (
                raw_stream is None
                or not isinstance(stream_owner, (Dictionary, Stream))
                or not stream_owner.is_indirect
            ):
                return False
            page = resolve_indirect(value.get("/Pg"))
            if page is None:
                page = inherited_page
            if not isinstance(page, Dictionary):
                return False
            container = stream if isinstance(stream, Stream) else page
            container_key = _object_key(container)
            if container_key is None or _object_key(owner) is None:
                return False
            reference_key = (container_key, mcid)
            if reference_key in content_references:
                return False
            content_references[reference_key] = (
                container,
                owner,
                page,
                (
                    stream_owner
                    if isinstance(stream_owner, (Dictionary, Stream))
                    else None
                ),
            )
            continue
        if type_name == Name.OBJR:
            referenced_object = resolve_indirect(value.get("/Obj"))
            page = resolve_indirect(value.get("/Pg"))
            if page is None:
                page = inherited_page
            if (
                owner is None
                or not isinstance(referenced_object, (Dictionary, Stream))
                or not isinstance(page, Dictionary)
            ):
                return False
            object_key = _object_key(referenced_object)
            page_key = _object_key(page)
            owner_key = _object_key(owner)
            if object_key is None or page_key is None or owner_key is None:
                return False
            referenced_key = (object_key, page_key)
            if referenced_key in object_owners:
                return False
            object_owners[referenced_key] = (
                referenced_object,
                owner_key,
                page,
            )
            continue
        if type_name is not None and type_name != Name.StructElem:
            return False

        object_key = _object_key(value)
        if object_key is None or object_key in seen:
            return False
        seen.add(object_key)

        parent = resolve_indirect(value.get("/P"))
        if not isinstance(parent, Dictionary) or not _same_object(
            parent, expected_parent
        ):
            return False

        role = resolve_indirect(value.get("/S"))
        if not isinstance(role, Name) or str(role) not in resolvable_roles:
            return False
        elements.append(value)

        page = resolve_indirect(value.get("/Pg"))
        if page is not None and not isinstance(page, Dictionary):
            return False

        kids = value.get("/K")
        if kids is not None:
            pending.append(
                (
                    kids,
                    value,
                    value,
                    page if isinstance(page, Dictionary) else inherited_page,
                )
            )
    return True


def _existing_structure_elements(
    pdf: pikepdf.Pdf,
    *,
    content_references_out: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ]
    | None = None,
    object_owners_out: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[Dictionary | Stream, _ObjectKey, Dictionary],
    ]
    | None = None,
) -> list[Dictionary] | None:
    root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    if not isinstance(root, Dictionary):
        return None
    if resolve_indirect(root.get("/Type")) != Name.StructTreeRoot:
        return None

    raw_parent_tree = root.get("/ParentTree")
    parent_tree = resolve_indirect(raw_parent_tree)
    if raw_parent_tree is not None and not isinstance(parent_tree, Dictionary):
        return None

    kids = resolve_indirect(root.get("/K"))
    if kids is not None and not isinstance(kids, (Array, Dictionary)):
        return None

    raw_role_map = root.get("/RoleMap")
    role_map = resolve_indirect(raw_role_map)
    if raw_role_map is not None and not isinstance(role_map, Dictionary):
        return None
    resolvable_roles = _resolvable_roles(role_map)
    if resolvable_roles is None:
        return None

    elements: list[Dictionary] = []
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ] = {}
    object_owners: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[Dictionary | Stream, _ObjectKey, Dictionary],
    ] = {}
    if kids is not None and not _walk_structure_k(
        kids,
        resolvable_roles,
        elements,
        set(),
        root,
        content_references,
        object_owners,
    ):
        return None
    page_keys = {_object_key(page.obj) for page in pdf.pages}
    if None in page_keys:
        return None
    if any(
        _object_key(page) not in page_keys
        for element in elements
        if isinstance(
            page := resolve_indirect(element.get("/Pg")),
            Dictionary,
        )
    ):
        return None
    if not _valid_class_map(root, elements):
        return None
    if not _valid_id_tree(root, elements):
        return None
    if not _valid_structure_hierarchy(root, role_map, elements):
        return None
    if not _valid_parent_tree(
        pdf,
        parent_tree,
        content_references,
        object_owners,
        root,
    ) and not _repair_parent_tree(
        pdf,
        root,
        elements,
        content_references,
        object_owners,
    ):
        return None
    if content_references_out is not None:
        content_references_out.update(content_references)
    if object_owners_out is not None:
        object_owners_out.update(object_owners)
    return elements


def _has_actualtext(element: Dictionary) -> bool:
    """Return whether a structure element or ancestor defines ActualText."""
    visited: set[_ObjectKey] = set()
    current: object = element
    while isinstance(current := resolve_indirect(current), Dictionary):
        current_key = _object_key(current)
        if current_key is None or current_key in visited:
            return False
        visited.add(current_key)

        if resolve_indirect(current.get("/Type")) == Name.StructTreeRoot:
            return False
        actualtext = resolve_indirect(current.get("/ActualText"))
        if isinstance(actualtext, String):
            return True
        current = current.get("/P")
    return False


def get_structural_actualtext_references(
    pdf: pikepdf.Pdf,
) -> frozenset[tuple[_ObjectKey, int]]:
    """Return valid content references covered by structure ActualText."""
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ] = {}
    try:
        elements = _existing_structure_elements(
            pdf,
            content_references_out=content_references,
        )
    except Exception:
        return frozenset()
    if elements is None:
        return frozenset()
    return frozenset(
        reference
        for reference, (_, owner, _, _) in content_references.items()
        if _has_actualtext(owner)
    )


def _normalize_structure_languages(elements: list[Dictionary]) -> int:
    normalized = 0
    for element in elements:
        if "/Lang" not in element:
            continue
        value = resolve_indirect(element.get("/Lang"))
        language = str(value) if isinstance(value, String) else None
        if language is not None and _is_valid_bcp47(language):
            continue
        element["/Lang"] = String("und")
        normalized += 1
    return normalized


def _valid_structure_text(value: object) -> bool:
    value = resolve_indirect(value)
    return isinstance(value, String) and bool(str(value).strip())


def _bounded_pdf_string(value: str) -> String:
    """Return a PDF text string within ISO 32000's byte limit."""
    result = String(value)
    encoded = bytes(result)
    if len(encoded) <= _MAX_STRING_BYTES:
        return result
    if not encoded.startswith(b"\xfe\xff"):
        return String(value[:_MAX_STRING_BYTES])

    remaining = _MAX_STRING_BYTES - 2
    length = 0
    end = 0
    for end, character in enumerate(value, start=1):
        character_length = len(character.encode("utf-16-be"))
        if length + character_length > remaining:
            end -= 1
            break
        length += character_length
    return String(value[:end])


def _structure_elements_with_content(
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
    references: frozenset[tuple[_ObjectKey, int]] | None = None,
) -> frozenset[_ObjectKey]:
    elements: set[_ObjectKey] = set()
    for reference, (_, owner, _, _) in content_references.items():
        if references is not None and reference not in references:
            continue
        current: object = owner
        visited: set[_ObjectKey] = set()
        while isinstance(current := resolve_indirect(current), Dictionary):
            if resolve_indirect(current.get("/Type")) != Name.StructElem:
                break
            key = _object_key(current)
            if key is None or key in visited:
                break
            visited.add(key)
            elements.add(key)
            current = current.get("/P")
    return frozenset(elements)


def _text_show_has_content(
    instruction: pikepdf.ContentStreamInstruction,
) -> bool:
    operator = str(instruction.operator)
    operands = list(instruction.operands)
    if operator == "TJ" and operands:
        values = resolve_indirect(operands[-1])
        return isinstance(values, Array) and any(
            _valid_structure_text(value) for value in values
        )
    if operator in {"Tj", "'", '"'} and operands:
        return _valid_structure_text(operands[-1])
    return False


def _marked_content_properties(
    operands: list[object],
    resources: Dictionary | None,
) -> Dictionary | None:
    if len(operands) < 2:
        return None
    properties = resolve_indirect(operands[1])
    if isinstance(properties, Name):
        properties = _named_property(resources, properties)
    return properties if isinstance(properties, Dictionary) else None


def _scan_content_description_evidence(
    owner: pikepdf.Page | Stream,
    resources: Dictionary | None,
    description: str,
    *,
    visited_forms: frozenset[_ObjectKey] = frozenset(),
    depth: int = 0,
) -> tuple[frozenset[int], frozenset[int], bool, bool]:
    """Find textual and ActualText evidence for marked-content references."""
    if depth > _MAX_FORM_SCAN_DEPTH:
        raise ConversionError(
            f"Cannot inspect semantic alternatives: {description} is nested "
            f"deeper than {_MAX_FORM_SCAN_DEPTH} Form XObject levels"
        )
    try:
        instructions = pikepdf.parse_content_stream(owner)
    except Exception as exc:
        raise ConversionError(
            f"Cannot inspect semantic alternatives: {description} cannot be parsed"
        ) from exc

    nesting: list[tuple[int | None, str | None, bool]] = []
    text_mcids: set[int] = set()
    actual_text_mcids: set[int] = set()
    has_text = False
    has_actual_text = False

    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            is_paint = True
            instruction_has_text = False
            instruction_has_actual_text = False
        else:
            operator = instruction.operator
            operands = list(instruction.operands)
            if operator in {_BMC, _BDC}:
                tag = resolve_indirect(operands[0]) if operands else None
                if not isinstance(tag, Name):
                    raise ConversionError(
                        f"Cannot inspect semantic alternatives: {description} has "
                        "malformed marked content"
                    )
                properties = (
                    _marked_content_properties(operands, resources)
                    if operator == _BDC
                    else None
                )
                mcid = resolve_indirect(properties.get("/MCID")) if properties else None
                if not isinstance(mcid, int) or isinstance(mcid, bool) or mcid < 0:
                    mcid = None
                actual_text = (
                    str(resolve_indirect(properties.get("/ActualText")))
                    if properties
                    and _valid_structure_text(properties.get("/ActualText"))
                    else None
                )
                nesting.append((mcid, actual_text, tag == Name.Artifact))
                continue
            if operator == _EMC:
                if not nesting:
                    raise ConversionError(
                        f"Cannot inspect semantic alternatives: {description} has "
                        "unbalanced marked content"
                    )
                nesting.pop()
                continue

            operator_name = str(operator)
            is_paint = operator_name in _PAINTING_OPERATORS or operator_name == "Do"
            instruction_has_text = _text_show_has_content(instruction)
            instruction_has_actual_text = False
            if operator_name == "Do" and operands:
                resource_name = resolve_indirect(operands[0])
                xobjects = (
                    resolve_indirect(resources.get("/XObject"))
                    if resources is not None
                    else None
                )
                xobject = (
                    resolve_indirect(xobjects.get(resource_name))
                    if isinstance(resource_name, Name)
                    and isinstance(xobjects, Dictionary)
                    else None
                )
                if (
                    isinstance(xobject, Stream)
                    and resolve_indirect(xobject.get("/Subtype")) == Name.Form
                ):
                    form_key = _object_key(xobject)
                    if form_key is not None and form_key in visited_forms:
                        raise ConversionError(
                            "Cannot inspect semantic alternatives: recursive Form "
                            f"XObject in {description}"
                        )
                    form_resources = resolve_indirect(xobject.get("/Resources"))
                    (
                        _form_text_mcids,
                        _form_actual_text_mcids,
                        instruction_has_text,
                        instruction_has_actual_text,
                    ) = _scan_content_description_evidence(
                        xobject,
                        (
                            form_resources
                            if isinstance(form_resources, Dictionary) and form_resources
                            else resources
                        ),
                        f"Form XObject {resource_name} invoked by {description}",
                        visited_forms=(
                            visited_forms | frozenset({form_key})
                            if form_key is not None
                            else visited_forms
                        ),
                        depth=depth + 1,
                    )

        if not is_paint or any(artifact for _mcid, _actual, artifact in nesting):
            continue
        active_mcids = tuple(
            mcid for mcid, _actual, _artifact in nesting if mcid is not None
        )
        active_actual_text = any(actual for _mcid, actual, _artifact in nesting)
        if instruction_has_text:
            has_text = True
            text_mcids.update(active_mcids)
        if instruction_has_actual_text or active_actual_text:
            has_actual_text = True
            actual_text_mcids.update(active_mcids)

    if nesting:
        raise ConversionError(
            f"Cannot inspect semantic alternatives: {description} has unbalanced "
            "marked content"
        )
    return (
        frozenset(text_mcids),
        frozenset(actual_text_mcids),
        has_text,
        has_actual_text,
    )


def _content_reference_description_evidence(
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
) -> tuple[
    frozenset[tuple[_ObjectKey, int]],
    frozenset[tuple[_ObjectKey, int]],
]:
    text_references: set[tuple[_ObjectKey, int]] = set()
    actual_text_references: set[tuple[_ObjectKey, int]] = set()
    cache: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[frozenset[int], frozenset[int]],
    ] = {}
    for reference, (
        container,
        _owner,
        page,
        stream_owner,
    ) in content_references.items():
        container_key = _object_key(container)
        page_key = _object_key(page)
        if container_key is None or page_key is None:
            continue
        cache_key = (container_key, page_key)
        evidence = cache.get(cache_key)
        if evidence is None:
            page_resources = _page_resources(pikepdf.Page(page))
            resources = (
                resolve_indirect(container.get("/Resources"))
                if isinstance(container, Stream)
                else page_resources
            )
            if not isinstance(resources, Dictionary) and isinstance(
                stream_owner, (Dictionary, Stream)
            ):
                resources = resolve_indirect(stream_owner.get("/Resources"))
            if not isinstance(resources, Dictionary):
                resources = page_resources
            text_mcids, actual_text_mcids, _has_text, _has_actual_text = (
                _scan_content_description_evidence(
                    pikepdf.Page(container)
                    if isinstance(container, Dictionary)
                    else container,
                    resources if isinstance(resources, Dictionary) else None,
                    f"content container {container_key[0]} {container_key[1]}",
                    visited_forms=(
                        frozenset({container_key})
                        if isinstance(container, Stream)
                        else frozenset()
                    ),
                )
            )
            evidence = (text_mcids, actual_text_mcids)
            cache[cache_key] = evidence
        if reference[1] in evidence[0]:
            text_references.add(reference)
        if reference[1] in evidence[1]:
            actual_text_references.add(reference)
    return frozenset(text_references), frozenset(actual_text_references)


def _marked_text_evidence_by_mcid(
    owner: pikepdf.Page | Stream,
    resources: Dictionary | None,
    description: str,
) -> dict[int, tuple[str | None, str | None]]:
    try:
        instructions = pikepdf.parse_content_stream(owner)
    except Exception as exc:
        raise ConversionError(
            f"Cannot inspect marked text evidence: {description} cannot be parsed"
        ) from exc

    nesting: list[tuple[int | None, str | None, str | None, bool]] = []
    evidence: dict[int, tuple[set[str], set[str]]] = {}
    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            is_paint = True
        else:
            operator = instruction.operator
            operands = list(instruction.operands)
            if operator in {_BMC, _BDC}:
                tag = resolve_indirect(operands[0]) if operands else None
                if not isinstance(tag, Name):
                    raise ConversionError(
                        f"Cannot inspect marked text evidence: {description} has "
                        "malformed marked content"
                    )
                properties = (
                    _marked_content_properties(operands, resources)
                    if operator == _BDC
                    else None
                )
                mcid = resolve_indirect(properties.get("/MCID")) if properties else None
                if not isinstance(mcid, int) or isinstance(mcid, bool) or mcid < 0:
                    mcid = None
                actual_text = (
                    str(resolve_indirect(properties.get("/ActualText")))
                    if properties
                    and _valid_structure_text(properties.get("/ActualText"))
                    else None
                )
                alt_text = (
                    str(resolve_indirect(properties.get("/Alt")))
                    if properties and _valid_structure_text(properties.get("/Alt"))
                    else None
                )
                nesting.append((mcid, actual_text, alt_text, tag == Name.Artifact))
                continue
            if operator == _EMC:
                if not nesting:
                    raise ConversionError(
                        f"Cannot inspect marked text evidence: {description} has "
                        "unbalanced marked content"
                    )
                nesting.pop()
                continue
            is_paint = str(operator) in _PAINTING_OPERATORS or str(operator) == "Do"

        if not is_paint or any(artifact for _mcid, _actual, _alt, artifact in nesting):
            continue
        actual_text = next(
            (actual for _mcid, actual, _alt, _artifact in reversed(nesting) if actual),
            None,
        )
        alt_text = next(
            (alt for _mcid, _actual, alt, _artifact in reversed(nesting) if alt),
            None,
        )
        for mcid, _actual, _alt, _artifact in nesting:
            if mcid is None:
                continue
            actual_values, alt_values = evidence.setdefault(mcid, (set(), set()))
            if actual_text is not None:
                actual_values.add(actual_text)
            if alt_text is not None:
                alt_values.add(alt_text)
    if nesting:
        raise ConversionError(
            f"Cannot inspect marked text evidence: {description} has unbalanced "
            "marked content"
        )

    resolved = {}
    for mcid, (actual_values, alt_values) in evidence.items():
        resolved[mcid] = (
            next(iter(actual_values)) if len(actual_values) == 1 else None,
            next(iter(alt_values)) if len(alt_values) == 1 else None,
        )
    return resolved


def _propagate_existing_marked_text_evidence(
    pdf: pikepdf.Pdf,
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
) -> int:
    root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    role_map = (
        resolve_indirect(root.get("/RoleMap")) if isinstance(root, Dictionary) else None
    )
    if not isinstance(role_map, Dictionary):
        role_map = None
    cache: dict[
        tuple[_ObjectKey, _ObjectKey], dict[int, tuple[str | None, str | None]]
    ] = {}
    owners: dict[_ObjectKey, tuple[Dictionary, set[str], set[str]]] = {}
    for (container_key, mcid), (
        container,
        owner,
        page,
        stream_owner,
    ) in content_references.items():
        if _effective_structure_role(owner.get("/S"), role_map) not in {
            "/Figure",
            "/Formula",
        }:
            continue
        owner_key = _object_key(owner)
        page_key = _object_key(page)
        if owner_key is None or page_key is None:
            continue
        cache_key = (container_key, page_key)
        container_evidence = cache.get(cache_key)
        if container_evidence is None:
            page_resources = _page_resources(pikepdf.Page(page))
            resources = (
                resolve_indirect(container.get("/Resources"))
                if isinstance(container, Stream)
                else page_resources
            )
            if not isinstance(resources, Dictionary) and isinstance(
                stream_owner, (Dictionary, Stream)
            ):
                resources = resolve_indirect(stream_owner.get("/Resources"))
            if not isinstance(resources, Dictionary):
                resources = page_resources
            container_evidence = _marked_text_evidence_by_mcid(
                pikepdf.Page(container)
                if isinstance(container, Dictionary)
                else container,
                resources if isinstance(resources, Dictionary) else None,
                f"content container {container_key[0]} {container_key[1]}",
            )
            cache[cache_key] = container_evidence
        actual_text, alt_text = container_evidence.get(mcid, (None, None))
        candidate, actual_values, alt_values = owners.setdefault(
            owner_key,
            (owner, set(), set()),
        )
        assert _same_object(candidate, owner)
        if actual_text is not None:
            actual_values.add(actual_text)
        if alt_text is not None:
            alt_values.add(alt_text)

    repaired = 0
    for owner, actual_values, alt_values in owners.values():
        if "/ActualText" not in owner and len(actual_values) == 1:
            owner["/ActualText"] = _bounded_pdf_string(next(iter(actual_values)))
            repaired += 1
        if "/Alt" not in owner and len(alt_values) == 1:
            owner["/Alt"] = _bounded_pdf_string(next(iter(alt_values)))
            repaired += 1
    return repaired


def _missing_structure_alternatives(
    pdf: pikepdf.Pdf,
    elements: list[Dictionary],
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
) -> int:
    root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    role_map = (
        resolve_indirect(root.get("/RoleMap")) if isinstance(root, Dictionary) else None
    )
    if not isinstance(role_map, Dictionary):
        role_map = None
    elements_with_content = _structure_elements_with_content(content_references)
    candidates = [
        element
        for element in elements
        if _effective_structure_role(element.get("/S"), role_map)
        in {"/Figure", "/Formula"}
        and _object_key(element) in elements_with_content
        and not any(
            _valid_structure_text(element.get(key)) for key in ("/Alt", "/ActualText")
        )
        and not any(
            _effective_structure_role(child.get("/S"), role_map) == "/Caption"
            and any(
                _valid_structure_text(child.get(key)) for key in ("/Alt", "/ActualText")
            )
            for child in _structure_element_children(element)
        )
    ]
    if not candidates:
        return 0
    text_references, actual_text_references = _content_reference_description_evidence(
        content_references
    )
    elements_with_text = _structure_elements_with_content(
        content_references,
        text_references,
    )
    elements_with_actual_text = _structure_elements_with_content(
        content_references,
        actual_text_references,
    )
    missing = 0
    for element in candidates:
        element_key = _object_key(element)
        if element_key in elements_with_actual_text:
            continue
        if any(
            _effective_structure_role(child.get("/S"), role_map) == "/Caption"
            and (
                _object_key(child) in elements_with_text
                or _object_key(child) in elements_with_actual_text
            )
            for child in _structure_element_children(element)
        ):
            continue
        missing += 1
    return missing


def _valid_table_header_association(
    element: Dictionary,
    header_ids: frozenset[str],
) -> bool:
    attributes = resolve_indirect(element.get("/A"))
    candidates = list(attributes) if isinstance(attributes, Array) else [attributes]
    for candidate in candidates:
        candidate = resolve_indirect(candidate)
        if not isinstance(candidate, Dictionary) or (
            resolve_indirect(candidate.get("/O")) != Name.Table
        ):
            continue
        scope = resolve_indirect(candidate.get("/Scope"))
        if scope in {Name.Row, Name.Column, Name.Both}:
            return True
        headers = resolve_indirect(candidate.get("/Headers"))
        if (
            isinstance(headers, Array)
            and headers
            and all(
                _valid_structure_text(header)
                and str(resolve_indirect(header)) in header_ids
                for header in headers
            )
        ):
            return True
    return False


def _repair_table_header_association(
    element: Dictionary,
    header_ids: frozenset[str],
    scope: Name,
) -> int:
    attributes = resolve_indirect(element.get("/A"))
    candidates = list(attributes) if isinstance(attributes, Array) else [attributes]
    repaired = 0
    for candidate in candidates:
        candidate = resolve_indirect(candidate)
        if not isinstance(candidate, Dictionary) or (
            resolve_indirect(candidate.get("/O")) != Name.Table
        ):
            continue
        if "/Scope" in candidate and resolve_indirect(candidate.get("/Scope")) not in {
            Name.Row,
            Name.Column,
            Name.Both,
        }:
            del candidate["/Scope"]
            repaired += 1
        headers = resolve_indirect(candidate.get("/Headers"))
        if "/Headers" in candidate and not (
            isinstance(headers, Array)
            and headers
            and all(
                _valid_structure_text(header)
                and str(resolve_indirect(header)) in header_ids
                for header in headers
            )
        ):
            del candidate["/Headers"]
            repaired += 1
    if not _valid_table_header_association(element, header_ids):
        _append_structure_attribute(
            element,
            Dictionary(O=Name.Table, Scope=scope),
        )
        repaired += 1
    return repaired


def _append_structure_attribute(element: Dictionary, attribute: Dictionary) -> None:
    current = resolve_indirect(element.get("/A"))
    if current is None:
        element["/A"] = attribute
    elif isinstance(current, Array):
        current.append(attribute)
    else:
        element["/A"] = Array([current, attribute])


def _replace_structure_kid(
    parent: Dictionary,
    old_kid: Dictionary,
    new_kid: Dictionary,
) -> bool:
    kids = resolve_indirect(parent.get("/K"))
    if isinstance(kids, Array):
        for index, item in enumerate(kids):
            candidate = resolve_indirect(item)
            if not isinstance(candidate, Dictionary):
                continue
            if candidate is old_kid or _same_object(candidate, old_kid):
                kids[index] = new_kid
                return True
        return False
    if kids is old_kid or _same_object(kids, old_kid):
        parent["/K"] = new_kid
        return True
    return False


def _repair_annotation_structure_role(
    pdf: pikepdf.Pdf,
    element: Dictionary,
    object_reference: Dictionary,
    annotation: Dictionary,
    expected_role: Name,
) -> Dictionary | None:
    if resolve_indirect(element.get("/S")) == expected_role:
        return None
    struct_parent = resolve_indirect(annotation.get("/StructParent"))
    root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    raw_parent_tree = root.get("/ParentTree") if isinstance(root, Dictionary) else None
    parent_tree_object = resolve_indirect(raw_parent_tree)
    if (
        not isinstance(struct_parent, int)
        or isinstance(struct_parent, bool)
        or struct_parent < 0
        or not isinstance(parent_tree_object, Dictionary)
    ):
        return None
    parent_tree = NumberTree(parent_tree_object)
    try:
        current_owner = resolve_indirect(parent_tree[struct_parent])
    except (KeyError, ValueError, TypeError):
        return None
    if not _same_object(current_owner, element):
        return None

    properties: dict[str, object] = {
        "Type": Name.StructElem,
        "S": expected_role,
        "P": element,
        "K": object_reference,
    }
    page = resolve_indirect(
        object_reference.get("/Pg", annotation.get("/P", element.get("/Pg")))
    )
    if isinstance(page, Dictionary):
        properties["Pg"] = page
    wrapper = pdf.make_indirect(Dictionary(**properties))
    if not _replace_structure_kid(element, object_reference, wrapper):
        return None
    parent_tree[struct_parent] = wrapper
    return wrapper


def _repair_numbered_heading_sequence(elements: list[Dictionary]) -> int:
    repaired = 0
    previous_level = 0
    for element in elements:
        role = str(resolve_indirect(element.get("/S")))
        if not re.fullmatch(r"/H[1-6]", role):
            continue
        raw_level = int(role[2:])
        level = 1 if previous_level == 0 else min(raw_level, previous_level + 1)
        if level != raw_level:
            element["/S"] = Name(f"/H{level}")
            repaired += 1
        previous_level = level
    return repaired


_GENERIC_WIDGET_TOOLTIP = "Form field"


def _widget_tooltip_evidence(
    annotation: Dictionary,
) -> tuple[str | None, bool, bool]:
    current: object = annotation
    visited: set[_ObjectKey | int] = set()
    field_name: str | None = None
    has_generic_tooltip = False
    is_annotation = True
    while isinstance(current := resolve_indirect(current), Dictionary):
        key: _ObjectKey | int = _object_key(current) or id(current)
        if key in visited:
            break
        visited.add(key)
        tooltip = resolve_indirect(current.get("/TU"))
        if isinstance(tooltip, String) and str(tooltip).strip():
            tooltip_text = str(tooltip)
            if tooltip_text == _GENERIC_WIDGET_TOOLTIP:
                has_generic_tooltip = True
            else:
                return tooltip_text, is_annotation, has_generic_tooltip
        name = resolve_indirect(current.get("/T"))
        if field_name is None and isinstance(name, String) and str(name).strip():
            field_name = str(name)
        current = current.get("/Parent")
        is_annotation = False
    return field_name, False, has_generic_tooltip


def _ensure_widget_tooltip(annotation: Dictionary) -> bool:
    tooltip, is_direct, has_generic_tooltip = _widget_tooltip_evidence(annotation)
    if is_direct:
        return False
    if tooltip is not None:
        annotation["/TU"] = _bounded_pdf_string(tooltip)
        return True
    if has_generic_tooltip:
        return False
    annotation["/TU"] = _bounded_pdf_string(_GENERIC_WIDGET_TOOLTIP)
    return True


def _widgets_requiring_name_review(
    pdf: pikepdf.Pdf,
    optional_content: _DefaultOCVisibility | None = None,
) -> int:
    widgets: set[_ObjectKey | int] = set()
    for page in pdf.pages:
        annotations = resolve_indirect(page.obj.get("/Annots"))
        if not isinstance(annotations, Array):
            continue
        for value in annotations:
            annotation = resolve_indirect(value)
            if not isinstance(annotation, Dictionary) or (
                resolve_indirect(annotation.get("/Subtype")) != Name.Widget
            ):
                continue
            if "/OC" in annotation and not _optional_content_is_visible(
                optional_content,
                annotation.get("/OC"),
                "Widget annotation",
            ):
                continue
            tooltip, _is_direct, _has_generic = _widget_tooltip_evidence(annotation)
            if tooltip is None:
                widgets.add(_object_key(annotation) or id(annotation))
    return len(widgets)


def _structure_element_children(element: Dictionary) -> list[Dictionary]:
    kids = resolve_indirect(element.get("/K"))
    candidates = list(kids) if isinstance(kids, Array) else [kids]
    return [
        candidate
        for item in candidates
        if isinstance((candidate := resolve_indirect(item)), Dictionary)
        and resolve_indirect(candidate.get("/Type")) == Name.StructElem
    ]


def _table_header_scope(
    element: Dictionary,
    role_map: Dictionary | None,
) -> Name:
    row = resolve_indirect(element.get("/P"))
    if not isinstance(row, Dictionary) or (
        _effective_structure_role(row.get("/S"), role_map) != "/TR"
    ):
        return Name.Column
    cells = [
        child
        for child in _structure_element_children(row)
        if _effective_structure_role(child.get("/S"), role_map) in {"/TH", "/TD"}
    ]
    cell_index = next(
        (index for index, cell in enumerate(cells) if _same_object(cell, element)),
        None,
    )
    container = resolve_indirect(row.get("/P"))
    if not isinstance(container, Dictionary):
        return Name.Column
    container_role = _effective_structure_role(container.get("/S"), role_map)
    table = (
        container
        if container_role == "/Table"
        else resolve_indirect(container.get("/P"))
    )
    if not isinstance(table, Dictionary) or (
        _effective_structure_role(table.get("/S"), role_map) != "/Table"
    ):
        return Name.Column
    rows: list[Dictionary] = []
    for child in _structure_element_children(table):
        child_role = _effective_structure_role(child.get("/S"), role_map)
        if child_role == "/TR":
            rows.append(child)
        elif child_role in {"/THead", "/TBody", "/TFoot"}:
            rows.extend(
                row_child
                for row_child in _structure_element_children(child)
                if _effective_structure_role(row_child.get("/S"), role_map) == "/TR"
            )
    row_index = next(
        (index for index, candidate in enumerate(rows) if _same_object(candidate, row)),
        None,
    )
    column_header = row_index == 0
    row_header = cell_index == 0 and len(rows) > 1
    if column_header and row_header:
        return Name.Both
    if row_header:
        return Name.Row
    return Name.Column


def _repair_existing_semantics(
    pdf: pikepdf.Pdf,
    elements: list[Dictionary],
) -> int:
    """Repair local role requirements without replacing author structure."""
    repaired = _repair_numbered_heading_sequence(elements)
    root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    role_map = (
        resolve_indirect(root.get("/RoleMap")) if isinstance(root, Dictionary) else None
    )
    if not isinstance(role_map, Dictionary):
        role_map = None
    header_ids = frozenset(
        str(identifier)
        for element in elements
        if _effective_structure_role(element.get("/S"), role_map) == "/TH"
        and _valid_structure_text(identifier := resolve_indirect(element.get("/ID")))
    )
    for element in elements:
        role = _effective_structure_role(element.get("/S"), role_map)
        if role in {"/Figure", "/Formula"}:
            for key in ("/Alt", "/ActualText"):
                if key in element and not _valid_structure_text(element.get(key)):
                    del element[key]
                    repaired += 1
        if role == "/TH":
            repaired += _repair_table_header_association(
                element,
                header_ids,
                _table_header_scope(element, role_map),
            )
        kids = resolve_indirect(element.get("/K"))
        candidates = list(kids) if isinstance(kids, Array) else [kids]
        for candidate in candidates:
            candidate = resolve_indirect(candidate)
            annotation = (
                resolve_indirect(candidate.get("/Obj"))
                if isinstance(candidate, Dictionary)
                and resolve_indirect(candidate.get("/Type")) == Name.OBJR
                else None
            )
            if not isinstance(annotation, Dictionary):
                continue
            subtype = resolve_indirect(annotation.get("/Subtype"))
            expected_role = (
                Name.Form
                if subtype == Name.Widget
                else Name.Link
                if subtype == Name.Link
                else None
            )
            if expected_role is not None:
                wrapper = _repair_annotation_structure_role(
                    pdf,
                    element,
                    candidate,
                    annotation,
                    expected_role,
                )
                if wrapper is not None:
                    elements.append(wrapper)
                    repaired += 1
            if subtype != Name.Widget:
                continue
            repaired += int(_ensure_widget_tooltip(annotation))
    return repaired


_PATH_PAINTING_OPERATORS = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})


def _has_semantic_structure_roles(pdf: pikepdf.Pdf) -> bool:
    """Return whether an existing tree contains roles richer than containers."""
    root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
    if not isinstance(root, Dictionary):
        return False
    role_map = resolve_indirect(root.get("/RoleMap"))
    if not isinstance(role_map, Dictionary):
        role_map = None
    pending = [root.get("/K")]
    seen: set[_ObjectKey] = set()
    while pending:
        value = resolve_indirect(pending.pop())
        if isinstance(value, Array):
            pending.extend(value)
            continue
        if not isinstance(value, Dictionary):
            continue
        key = _object_key(value)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        role = _effective_structure_role(value.get("/S"), role_map)
        if role is not None and role not in {
            "/Document",
            "/Part",
            "/Art",
            "/Sect",
            "/Div",
            "/NonStruct",
            "/Private",
            "/Link",
            "/Form",
            "/Annot",
        }:
            return True
        pending.append(value.get("/K"))
    return False


def _artifact_untagged_path_painting(
    pdf: pikepdf.Pdf,
) -> tuple[list[Dictionary] | None, int]:
    """Preserve a rich tree when only decorative vector paths are untagged."""
    page_contents = [page.obj.get("/Contents") for page in pdf.pages]
    form_bytes: dict[_ObjectKey, tuple[Stream, bytes]] = {}
    forms: dict[_ObjectKey, Stream] = {}
    for item in pdf.objects:
        if not isinstance(item, Stream):
            continue
        if resolve_indirect(item.get("/Subtype")) != Name.Form:
            continue
        key = _object_key(item)
        if key is not None:
            forms[key] = item

    def rewrite(
        owner: pikepdf.Page | Stream,
        *,
        page: pikepdf.Page | None = None,
        resources: Dictionary | None = None,
    ) -> int:
        instructions = list(pikepdf.parse_content_stream(owner))
        rewritten: list[
            pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage
        ] = []
        protected: list[bool] = []
        changed = 0
        path_start: int | None = None
        for instruction in instructions:
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                rewritten.append(instruction)
                continue
            operator = instruction.operator
            if operator in (_BMC, _BDC):
                operands = instruction.operands
                tag = resolve_indirect(operands[0]) if operands else None
                covered = isinstance(tag, Name) and tag == Name.Artifact
                if operator == _BDC and len(operands) >= 2:
                    properties = resolve_indirect(operands[1])
                    if isinstance(properties, Name):
                        properties = _named_property(resources, properties)
                    if isinstance(properties, Dictionary):
                        mcid = resolve_indirect(properties.get("/MCID"))
                        covered = covered or (
                            isinstance(mcid, int)
                            and not isinstance(mcid, bool)
                            and mcid >= 0
                        )
                        covered = covered or any(
                            _valid_structure_text(properties.get(key))
                            for key in ("/ActualText", "/Alt")
                        )
                protected.append(covered)
            elif operator == _EMC:
                if protected:
                    protected.pop()
            operator_name = str(operator)
            if operator_name in {"m", "re"} and path_start is None:
                path_start = len(rewritten)
            path_painting = operator_name in _PATH_PAINTING_OPERATORS
            if (path_painting or operator_name == "sh") and not any(protected):
                marker = pikepdf.ContentStreamInstruction(
                    [Name.Artifact],
                    _BMC,
                )
                if path_painting and path_start is not None:
                    rewritten.insert(path_start, marker)
                else:
                    rewritten.append(marker)
                rewritten.extend(
                    (instruction, pikepdf.ContentStreamInstruction([], _EMC))
                )
                changed += 1
            else:
                rewritten.append(instruction)
            if path_painting or operator_name == "n":
                path_start = None
        if changed:
            content = pikepdf.unparse_content_stream(rewritten)
            if page is None:
                assert isinstance(owner, Stream)
                owner.write(content)
            else:
                page.obj["/Contents"] = pdf.make_stream(content)
        return changed

    changed = 0
    try:
        for page in pdf.pages:
            changed += rewrite(page, page=page, resources=_page_resources(page))
        for key, form in forms.items():
            form_bytes[key] = (form, bytes(form.read_bytes()))
            resources = resolve_indirect(form.get("/Resources"))
            changed += rewrite(
                form,
                resources=resources if isinstance(resources, Dictionary) else None,
            )
        if not changed:
            return None, 0
        elements = _existing_structure_elements(pdf)
        if elements is not None:
            return elements, changed
    except Exception:
        pass

    for page, contents in zip(pdf.pages, page_contents, strict=True):
        if contents is None:
            page.obj.pop("/Contents", None)
        else:
            page.obj["/Contents"] = contents
    for form, content in form_bytes.values():
        form.write(content)
    return None, 0


def _valid_structure_attribute(value: object) -> bool:
    value = resolve_indirect(value)
    return isinstance(value, (Dictionary, Stream)) and isinstance(
        resolve_indirect(value.get("/O")),
        Name,
    )


def _valid_structure_attribute_array(value: object) -> bool:
    value = resolve_indirect(value)
    if _valid_structure_attribute(value):
        return True
    if not isinstance(value, Array):
        return False
    index = 0
    while index < len(value):
        if not _valid_structure_attribute(value[index]):
            return False
        index += 1
        if index < len(value):
            revision = resolve_indirect(value[index])
            if isinstance(revision, int) and not isinstance(revision, bool):
                index += 1
    return True


def _valid_class_map(root: Dictionary, elements: list[Dictionary]) -> bool:
    raw_class_map = root.get("/ClassMap")
    class_map = resolve_indirect(raw_class_map)
    if raw_class_map is not None:
        if not isinstance(class_map, Dictionary):
            return False
        for value in class_map.values():
            value = resolve_indirect(value)
            if _valid_structure_attribute(value):
                continue
            if not isinstance(value, Array):
                return False
            if any(not _valid_structure_attribute(attribute) for attribute in value):
                return False

    for element in elements:
        revision = resolve_indirect(element.get("/R"))
        if "/R" in element and (
            not isinstance(revision, int) or isinstance(revision, bool) or revision < 0
        ):
            return False
        if "/A" in element and not _valid_structure_attribute_array(element.get("/A")):
            return False
        raw_classes = element.get("/C")
        if raw_classes is None:
            continue
        classes = resolve_indirect(raw_classes)
        if isinstance(classes, Name):
            class_names = [classes]
        elif isinstance(classes, Array):
            class_names = []
            index = 0
            while index < len(classes):
                class_name = resolve_indirect(classes[index])
                if not isinstance(class_name, Name):
                    return False
                class_names.append(class_name)
                index += 1
                if index < len(classes):
                    revision = resolve_indirect(classes[index])
                    if isinstance(revision, int) and not isinstance(revision, bool):
                        index += 1
        else:
            return False
        if class_names and (
            not isinstance(class_map, Dictionary)
            or any(str(class_name) not in class_map for class_name in class_names)
        ):
            return False
    return True


def _content_streams(page: pikepdf.Page, page_number: int) -> list[Stream]:
    contents = resolve_indirect(page.obj.get("/Contents"))
    if contents is None:
        return []
    if isinstance(contents, Stream):
        return [contents]
    if not isinstance(contents, Array):
        raise ConversionError(
            f"Cannot create logical structure: page {page_number} has a "
            "malformed /Contents entry"
        )

    streams: list[Stream] = []
    for item in contents:
        stream = resolve_indirect(item)
        if not isinstance(stream, Stream):
            raise ConversionError(
                f"Cannot create logical structure: page {page_number} has a "
                "malformed /Contents entry"
            )
        streams.append(stream)
    return streams


def _remove_generated_wrapper(
    streams: list[Stream], page_number: int
) -> tuple[list[Stream], int | None]:
    if len(streams) < 2:
        return streams, None
    try:
        prefix = bytes(streams[0].read_bytes())
        suffix = bytes(streams[-1].read_bytes())
    except Exception as exc:
        raise ConversionError(
            f"Cannot create logical structure: page {page_number} content "
            "cannot be read"
        ) from exc

    match = _WRAPPER_PREFIX.fullmatch(prefix)
    if match is None or _WRAPPER_SUFFIX.fullmatch(suffix) is None:
        return streams, None
    return streams[1:-1], int(match.group(1))


def _page_resources(page: pikepdf.Page) -> Dictionary | None:
    current = resolve_indirect(page.obj)
    seen: set[_ObjectKey] = set()
    while isinstance(current, Dictionary):
        key = _object_key(current)
        if key is not None:
            if key in seen:
                return None
            seen.add(key)
        resources = resolve_indirect(current.get("/Resources"))
        if isinstance(resources, Dictionary):
            return resources
        current = resolve_indirect(current.get("/Parent"))
    return None


def _named_property(
    resources: Dictionary | None,
    name: Name,
) -> object | None:
    if resources is None:
        return None
    properties = resolve_indirect(resources.get("/Properties"))
    if not isinstance(properties, Dictionary):
        return None
    return resolve_indirect(properties.get(name))


def _collect_content_mcids(
    owner: pikepdf.Page | Stream,
    resources: Dictionary | None,
    description: str,
) -> tuple[set[int], bool]:
    mcids: set[int] = set()
    invalid = False
    paint_modes: dict[int, set[str]] = {}

    def record_paint(active_mcids: list[int], artifact: bool) -> None:
        mode = "artifact" if artifact else "semantic"
        for mcid in active_mcids:
            paint_modes.setdefault(mcid, set()).add(mode)

    def scan(
        current_owner: pikepdf.Page | Stream,
        current_resources: Dictionary | None,
        active_mcids: tuple[int, ...],
        inherited_artifact: bool,
        active_forms: frozenset[_ObjectKey],
        *,
        collect_mcids: bool,
    ) -> None:
        nonlocal invalid
        try:
            instructions = pikepdf.parse_content_stream(current_owner)
        except Exception as exc:
            raise ConversionError(
                f"Cannot create logical structure: {description} cannot be parsed"
            ) from exc

        local_mcids = list(active_mcids)
        nesting: list[tuple[str, bool, int | None]] = []
        for instruction in instructions:
            artifact = inherited_artifact or any(
                item_artifact
                for kind, item_artifact, _mcid in nesting
                if kind == "marked"
            )
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                record_paint(local_mcids, artifact)
                continue

            operator = instruction.operator
            operands = instruction.operands
            if operator in {_BMC, _BDC}:
                tag = resolve_indirect(operands[0]) if operands else None
                item_artifact = isinstance(tag, Name) and tag == Name.Artifact
                owned_mcid = None
                if operator == _BDC and len(operands) >= 2:
                    properties = resolve_indirect(operands[1])
                    if isinstance(properties, Name):
                        properties = _named_property(current_resources, properties)
                    if isinstance(properties, Dictionary):
                        mcid = resolve_indirect(properties.get("/MCID"))
                        if (
                            collect_mcids
                            and isinstance(mcid, int)
                            and not isinstance(mcid, bool)
                            and mcid >= 0
                        ):
                            invalid = (
                                invalid
                                or item_artifact
                                or any(item[2] is not None for item in nesting)
                                or mcid in mcids
                            )
                            mcids.add(mcid)
                            local_mcids.append(mcid)
                            owned_mcid = mcid
                nesting.append(("marked", item_artifact, owned_mcid))
                continue
            if operator == _BT:
                if any(kind == "text" for kind, _artifact, _mcid in nesting):
                    invalid = True
                nesting.append(("text", False, None))
                continue
            if operator == _EMC:
                if not nesting or nesting[-1][0] != "marked":
                    invalid = True
                else:
                    _kind, _artifact, owned_mcid = nesting.pop()
                    if owned_mcid is not None:
                        if not local_mcids or local_mcids[-1] != owned_mcid:
                            invalid = True
                        else:
                            local_mcids.pop()
                continue
            if operator == _ET:
                if not nesting or nesting[-1][0] != "text":
                    invalid = True
                else:
                    nesting.pop()
                continue

            artifact = inherited_artifact or any(
                item_artifact
                for kind, item_artifact, _mcid in nesting
                if kind == "marked"
            )
            operator_name = str(operator)
            if operator_name == "Do" and operands:
                xobjects = (
                    resolve_indirect(current_resources.get("/XObject"))
                    if current_resources is not None
                    else None
                )
                xobject = (
                    resolve_indirect(xobjects.get(operands[0]))
                    if isinstance(xobjects, Dictionary)
                    else None
                )
                if (
                    isinstance(xobject, Stream)
                    and resolve_indirect(xobject.get("/Subtype")) == Name.Form
                    and local_mcids
                ):
                    xobject_key = _object_key(xobject)
                    if xobject_key is None or xobject_key in active_forms:
                        invalid = True
                        continue
                    nested_resources = resolve_indirect(xobject.get("/Resources"))
                    scan(
                        xobject,
                        (
                            nested_resources
                            if isinstance(nested_resources, Dictionary)
                            and nested_resources
                            else current_resources
                        ),
                        tuple(local_mcids),
                        artifact,
                        active_forms | frozenset({xobject_key}),
                        collect_mcids=False,
                    )
                    continue
                record_paint(local_mcids, artifact)
                continue
            if operator_name in _PAINTING_OPERATORS:
                record_paint(local_mcids, artifact)
        invalid = invalid or bool(nesting)

    owner_key = _object_key(owner)
    scan(
        owner,
        resources,
        (),
        False,
        (
            frozenset({owner_key})
            if isinstance(owner, Stream) and owner_key is not None
            else frozenset()
        ),
        collect_mcids=True,
    )
    artifact_only_mcids = {
        mcid for mcid, modes in paint_modes.items() if modes == {"artifact"}
    }
    return mcids, invalid or bool(artifact_only_mcids)


def _collect_page_mcids(
    page: pikepdf.Page,
    page_number: int,
) -> tuple[set[int], bool]:
    if page.obj.get("/Contents") is None:
        return set(), False
    return _collect_content_mcids(
        page,
        _page_resources(page),
        f"page {page_number} content",
    )


def _has_untagged_painting(
    owner: pikepdf.Page | Stream,
    resources: Dictionary | None,
    structured_xobjects: set[_ObjectKey],
    active_forms: set[_ObjectKey] | None = None,
) -> bool:
    active = set() if active_forms is None else set(active_forms)
    frames = [
        (
            iter(pikepdf.parse_content_stream(owner)),
            resources,
            [],
            None,
        )
    ]
    while frames:
        instructions, current_resources, protected, active_key = frames[-1]
        try:
            instruction = next(instructions)
        except StopIteration:
            frames.pop()
            if active_key is not None:
                active.remove(active_key)
            continue

        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            if not any(protected):
                return True
            continue

        operator = instruction.operator
        operands = instruction.operands
        if operator in (_BMC, _BDC):
            tag = resolve_indirect(operands[0]) if operands else None
            is_protected = isinstance(tag, Name) and str(tag) == "/Artifact"
            if operator == _BDC and len(operands) >= 2:
                properties = resolve_indirect(operands[1])
                if isinstance(properties, Name):
                    properties = _named_property(current_resources, properties)
                if isinstance(properties, Dictionary):
                    mcid = resolve_indirect(properties.get("/MCID"))
                    is_protected = is_protected or (
                        isinstance(mcid, int)
                        and not isinstance(mcid, bool)
                        and mcid >= 0
                    )
            protected.append(is_protected)
            continue
        if operator == _EMC:
            if protected:
                protected.pop()
            continue
        if any(protected):
            continue

        operator_name = str(operator)
        if operator_name == "Do" and operands:
            xobjects = (
                resolve_indirect(current_resources.get("/XObject"))
                if current_resources is not None
                else None
            )
            xobject = (
                resolve_indirect(xobjects.get(operands[0]))
                if isinstance(xobjects, Dictionary)
                else None
            )
            if (
                isinstance(xobject, Stream)
                and (xobject_key := _object_key(xobject)) is not None
                and xobject_key in structured_xobjects
            ):
                continue
            if (
                isinstance(xobject, Stream)
                and resolve_indirect(xobject.get("/Subtype")) == Name.Form
            ):
                xobject_key = _object_key(xobject)
                if xobject_key is None or xobject_key in active:
                    return True
                nested_resources = resolve_indirect(xobject.get("/Resources"))
                active.add(xobject_key)
                frames.append(
                    (
                        iter(pikepdf.parse_content_stream(xobject)),
                        (
                            nested_resources
                            if isinstance(nested_resources, Dictionary)
                            and nested_resources
                            else current_resources
                        ),
                        [],
                        xobject_key,
                    )
                )
                continue
            return True
        if operator_name in _PAINTING_OPERATORS:
            return True
    return False


def _append_resource_context(
    contexts: list[Dictionary | None],
    candidate: Dictionary | None,
) -> None:
    """Append a context unless the same indirect dictionary is already present."""
    if candidate is None:
        if None not in contexts:
            contexts.append(None)
        return
    candidate_key = _object_key(candidate)
    if candidate_key is None or all(
        _object_key(context) != candidate_key for context in contexts
    ):
        contexts.append(candidate)


def _rendered_xobjects(
    owner: pikepdf.Page | Stream,
    resources: Dictionary | None,
    description: str,
    *,
    resource_contexts: dict[
        _ObjectKey,
        tuple[Stream, list[Dictionary | None]],
    ]
    | None = None,
    usage_modes: dict[_ObjectKey, list[str]] | None = None,
    active: set[_ObjectKey] | None = None,
    inherited_mode: str = "unmarked",
) -> set[_ObjectKey]:
    try:
        instructions = iter(pikepdf.parse_content_stream(owner))
    except Exception as exc:
        raise ConversionError(
            f"Cannot create logical structure: {description} cannot be parsed"
        ) from exc

    active_objects = set() if active is None else set(active)
    rendered: set[_ObjectKey] = set()
    frames = [(instructions, resources, [], inherited_mode, None)]
    while frames:
        (
            current_instructions,
            current_resources,
            marked_content,
            current_inherited_mode,
            active_key,
        ) = frames[-1]
        try:
            instruction = next(current_instructions)
        except StopIteration:
            frames.pop()
            if active_key is not None:
                active_objects.remove(active_key)
            continue

        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        operator = instruction.operator
        operands = instruction.operands
        if operator in (_BMC, _BDC):
            tag = resolve_indirect(operands[0]) if operands else None
            mode = "artifact" if str(tag) == "/Artifact" else "unmarked"
            if operator == _BDC and len(operands) >= 2:
                properties = resolve_indirect(operands[1])
                if isinstance(properties, Name):
                    properties = _named_property(current_resources, properties)
                if isinstance(properties, Dictionary):
                    mcid = resolve_indirect(properties.get("/MCID"))
                    if (
                        isinstance(mcid, int)
                        and not isinstance(mcid, bool)
                        and mcid >= 0
                    ):
                        mode = "mcid"
            marked_content.append(mode)
            continue
        if operator == _EMC:
            if marked_content:
                marked_content.pop()
            continue
        if not operands:
            continue

        operator_name = str(operator)
        xobject = None
        auxiliary = False
        if operator_name == "Do":
            xobjects = (
                resolve_indirect(current_resources.get("/XObject"))
                if current_resources is not None
                else None
            )
            xobject = (
                resolve_indirect(xobjects.get(operands[0]))
                if isinstance(xobjects, Dictionary)
                else None
            )
        elif operator_name == "gs":
            extgstates = (
                resolve_indirect(current_resources.get("/ExtGState"))
                if current_resources is not None
                else None
            )
            extgstate = (
                resolve_indirect(extgstates.get(operands[0]))
                if isinstance(extgstates, Dictionary)
                else None
            )
            soft_mask = (
                resolve_indirect(extgstate.get("/SMask"))
                if isinstance(extgstate, Dictionary)
                else None
            )
            xobject = (
                resolve_indirect(soft_mask.get("/G"))
                if isinstance(soft_mask, Dictionary)
                else None
            )
        elif operator_name in {"scn", "SCN"}:
            patterns = (
                resolve_indirect(current_resources.get("/Pattern"))
                if current_resources is not None
                else None
            )
            if isinstance(patterns, Dictionary):
                for operand in reversed(operands):
                    if not isinstance(operand, Name):
                        continue
                    candidate = resolve_indirect(patterns.get(operand))
                    if (
                        isinstance(candidate, Stream)
                        and resolve_indirect(candidate.get("/PatternType")) == 1
                    ):
                        xobject = candidate
                        auxiliary = True
                        break
        if not isinstance(xobject, Stream):
            continue

        object_key = _object_key(xobject)
        if object_key is None:
            raise ConversionError(
                "Cannot create logical structure: a rendered stream is direct"
            )
        rendered.add(object_key)
        if auxiliary:
            usage_mode = "auxiliary"
        elif any(mode == "artifact" for mode in marked_content):
            usage_mode = "artifact"
        elif any(mode == "mcid" for mode in marked_content):
            usage_mode = "mcid"
        else:
            usage_mode = current_inherited_mode
        if usage_modes is not None:
            usage_modes.setdefault(object_key, []).append(usage_mode)

        if (
            resolve_indirect(xobject.get("/Subtype")) != Name.Form
            and resolve_indirect(xobject.get("/PatternType")) != 1
        ):
            continue
        nested_resources = resolve_indirect(xobject.get("/Resources"))
        effective_resources = (
            nested_resources
            if isinstance(nested_resources, Dictionary) and nested_resources
            else current_resources
        )
        if resource_contexts is not None:
            _, contexts = resource_contexts.setdefault(
                object_key,
                (xobject, []),
            )
            _append_resource_context(contexts, effective_resources)

        if object_key in active_objects:
            continue
        try:
            nested_instructions = iter(pikepdf.parse_content_stream(xobject))
        except Exception as exc:
            raise ConversionError(
                "Cannot create logical structure: Form XObject or tiling "
                "pattern content cannot be parsed"
            ) from exc
        active_objects.add(object_key)
        frames.append(
            (
                nested_instructions,
                effective_resources,
                [],
                usage_mode,
                object_key,
            )
        )
    return rendered


def _annotation_appearance_streams(
    annotation: Dictionary,
    page_resources: Dictionary | None,
) -> list[tuple[Stream, Dictionary | None]]:
    appearance = resolve_indirect(annotation.get("/AP"))
    if not isinstance(appearance, Dictionary):
        return []

    streams: dict[
        _ObjectKey,
        tuple[Stream, Dictionary | None],
    ] = {}
    for key in ("/N", "/R", "/D"):
        entry = resolve_indirect(appearance.get(key))
        candidates: list[object]
        if isinstance(entry, Stream):
            candidates = [entry]
        elif isinstance(entry, Dictionary):
            candidates = [resolve_indirect(value) for value in entry.values()]
        else:
            continue
        for candidate in candidates:
            if not isinstance(candidate, Stream):
                continue
            resources = resolve_indirect(candidate.get("/Resources"))
            effective_resources = (
                resources
                if isinstance(resources, Dictionary) and resources
                else page_resources
            )
            candidate_key = _object_key(candidate)
            if candidate_key is None:
                raise ConversionError(
                    "Cannot create logical structure: an appearance stream is direct"
                )
            streams[candidate_key] = (candidate, effective_resources)
    return list(streams.values())


def _type3_charproc_streams(
    pdf: pikepdf.Pdf,
) -> dict[
    _ObjectKey,
    tuple[Stream, list[Dictionary | None]],
]:
    streams: dict[
        _ObjectKey,
        tuple[Stream, list[Dictionary | None]],
    ] = {}
    visited: set[tuple[int, int]] = set()

    def walk(obj: object) -> None:
        pending = [obj]
        while pending:
            try:
                resolved = resolve_indirect(pending.pop())
            except Exception:
                continue
            if not isinstance(resolved, pikepdf.Object):
                continue

            objgen = resolved.objgen
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            if isinstance(resolved, (Dictionary, Stream)):
                if resolve_indirect(resolved.get("/Subtype")) == Name.Type3:
                    charprocs = resolve_indirect(resolved.get("/CharProcs"))
                    resources = resolve_indirect(resolved.get("/Resources"))
                    context = resources if isinstance(resources, Dictionary) else None
                    if isinstance(charprocs, Dictionary):
                        for value in charprocs.values():
                            stream = resolve_indirect(value)
                            if isinstance(stream, Stream):
                                stream_key = _object_key(stream)
                                if stream_key is None:
                                    continue
                                _, contexts = streams.setdefault(
                                    stream_key,
                                    (stream, []),
                                )
                                _append_resource_context(contexts, context)
                pending.extend(reversed(list(resolved.values())))
            elif isinstance(resolved, Array):
                pending.extend(reversed(resolved))

    walk(pdf.Root)
    for obj in pdf.objects:
        walk(obj)
    return streams


def _name_tree_entries(root: Dictionary) -> list[tuple[bytes, object]] | None:
    seen: set[_ObjectKey] = set()
    entries_by_node: dict[
        _ObjectKey,
        list[tuple[bytes, object]],
    ] = {}
    root_entries: list[tuple[bytes, object]] | None = None
    pending: list[tuple[Dictionary, bool, bool]] = [(root, True, False)]
    while pending:
        node, is_root, expanded = pending.pop()
        node_key = _object_key(node)
        if not expanded:
            if node_key is not None:
                if node_key in seen:
                    return None
                seen.add(node_key)
            elif not is_root:
                return None

            names = resolve_indirect(node.get("/Names"))
            kids = resolve_indirect(node.get("/Kids"))
            if names is not None and kids is not None:
                return None

            if names is not None:
                if (
                    not isinstance(names, Array)
                    or len(names) > _MAX_ARRAY_ITEMS
                    or len(names) % 2
                ):
                    return None
                entries: list[tuple[bytes, object]] = []
                for index in range(0, len(names), 2):
                    key = resolve_indirect(names[index])
                    if (
                        not isinstance(key, String)
                        or len(bytes(key)) > _MAX_STRING_BYTES
                    ):
                        return None
                    key_bytes = bytes(key)
                    if entries and key_bytes <= entries[-1][0]:
                        return None
                    entries.append((key_bytes, resolve_indirect(names[index + 1])))
            elif isinstance(kids, Array) and kids and len(kids) <= _MAX_ARRAY_ITEMS:
                children: list[Dictionary] = []
                for item in kids:
                    child = resolve_indirect(item)
                    if not isinstance(child, Dictionary) or not child.is_indirect:
                        return None
                    children.append(child)
                pending.append((node, is_root, True))
                pending.extend((child, False, False) for child in reversed(children))
                continue
            elif is_root and kids is None:
                entries = []
            else:
                return None
        else:
            kids = resolve_indirect(node.get("/Kids"))
            if not isinstance(kids, Array):
                return None
            entries = []
            for item in kids:
                child = resolve_indirect(item)
                if not isinstance(child, Dictionary):
                    return None
                child_entries = entries_by_node.get(_object_key(child))
                if child_entries is None or (
                    entries and child_entries and child_entries[0][0] <= entries[-1][0]
                ):
                    return None
                entries.extend(child_entries)

        if node_key is None:
            if not is_root:
                return None
            root_entries = entries
        else:
            entries_by_node[node_key] = entries
        limits = resolve_indirect(node.get("/Limits"))
        if is_root:
            if limits is not None:
                return None
        elif limits is None:
            return None
        else:
            if not isinstance(limits, Array) or len(limits) != 2 or not entries:
                return None
            lower = resolve_indirect(limits[0])
            upper = resolve_indirect(limits[1])
            if (
                not isinstance(lower, String)
                or not isinstance(upper, String)
                or len(bytes(lower)) > _MAX_STRING_BYTES
                or len(bytes(upper)) > _MAX_STRING_BYTES
                or bytes(lower) != entries[0][0]
                or bytes(upper) != entries[-1][0]
            ):
                return None
    root_key = _object_key(root)
    return root_entries if root_key is None else entries_by_node.get(root_key)


def _valid_id_tree(structure_root: Dictionary, elements: list[Dictionary]) -> bool:
    identifiers: dict[bytes, Dictionary] = {}
    for element in elements:
        identifier = resolve_indirect(element.get("/ID"))
        if identifier is None:
            continue
        if not isinstance(identifier, String):
            return False
        key = bytes(identifier)
        if len(key) > _MAX_STRING_BYTES or key in identifiers:
            return False
        identifiers[key] = element

    raw_id_tree = structure_root.get("/IDTree")
    id_tree = resolve_indirect(raw_id_tree)
    if raw_id_tree is not None and not isinstance(id_tree, Dictionary):
        return False
    if id_tree is None:
        return not identifiers

    entries = _name_tree_entries(id_tree)
    if entries is None or len(entries) != len(identifiers):
        return False
    try:
        name_tree = NameTree(id_tree)
    except Exception:
        return False
    if len(name_tree) != len(entries):
        return False
    for key, value in entries:
        expected = identifiers.get(key)
        if (
            expected is None
            or not isinstance(value, Dictionary)
            or not value.is_indirect
            or not _same_object(value, expected)
        ):
            return False
    return True


def _number_tree_keys(root: Dictionary) -> list[int] | None:
    seen: set[_ObjectKey] = set()
    keys_by_node: dict[_ObjectKey, list[int]] = {}
    root_keys: list[int] | None = None
    pending: list[tuple[Dictionary, bool, bool]] = [(root, True, False)]
    while pending:
        node, is_root, expanded = pending.pop()
        node_key = _object_key(node)
        if not expanded:
            if node_key is not None:
                if node_key in seen:
                    return None
                seen.add(node_key)
            elif not is_root:
                return None

            nums = resolve_indirect(node.get("/Nums"))
            kids = resolve_indirect(node.get("/Kids"))
            if nums is not None and kids is not None:
                return None

            if nums is not None:
                if (
                    not isinstance(nums, Array)
                    or len(nums) > _MAX_ARRAY_ITEMS
                    or len(nums) % 2
                ):
                    return None
                keys: list[int] = []
                for index in range(0, len(nums), 2):
                    key = resolve_indirect(nums[index])
                    if (
                        not isinstance(key, int)
                        or isinstance(key, bool)
                        or key < 0
                        or (keys and key <= keys[-1])
                    ):
                        return None
                    keys.append(key)
            elif isinstance(kids, Array) and kids and len(kids) <= _MAX_ARRAY_ITEMS:
                children: list[Dictionary] = []
                for item in kids:
                    child = resolve_indirect(item)
                    if not isinstance(child, Dictionary) or not child.is_indirect:
                        return None
                    children.append(child)
                pending.append((node, is_root, True))
                pending.extend((child, False, False) for child in reversed(children))
                continue
            elif is_root and kids is None:
                keys = []
            else:
                return None
        else:
            kids = resolve_indirect(node.get("/Kids"))
            if not isinstance(kids, Array):
                return None
            keys = []
            for item in kids:
                child = resolve_indirect(item)
                if not isinstance(child, Dictionary):
                    return None
                child_keys = keys_by_node.get(_object_key(child))
                if child_keys is None or (
                    keys and child_keys and child_keys[0] <= keys[-1]
                ):
                    return None
                keys.extend(child_keys)

        if node_key is None:
            if not is_root:
                return None
            root_keys = keys
        else:
            keys_by_node[node_key] = keys
        limits = resolve_indirect(node.get("/Limits"))
        if is_root:
            if limits is not None:
                return None
        elif limits is None:
            return None
        else:
            if not isinstance(limits, Array) or len(limits) != 2 or not keys:
                return None
            lower = resolve_indirect(limits[0])
            upper = resolve_indirect(limits[1])
            if (
                not isinstance(lower, int)
                or isinstance(lower, bool)
                or not isinstance(upper, int)
                or isinstance(upper, bool)
                or lower != keys[0]
                or upper != keys[-1]
            ):
                return None
    root_key = _object_key(root)
    return root_keys if root_key is None else keys_by_node.get(root_key)


def _valid_parent_tree(
    pdf: pikepdf.Pdf,
    parent_tree: Dictionary | None,
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
    object_owners: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[Dictionary | Stream, _ObjectKey, Dictionary],
    ],
    structure_root: Dictionary,
) -> bool:
    number_tree = None
    number_tree_keys: set[int] = set()
    if parent_tree is not None:
        raw_number_tree_keys = _number_tree_keys(parent_tree)
        if raw_number_tree_keys is None:
            return False
        try:
            number_tree = NumberTree(parent_tree)
            number_tree_keys.update(raw_number_tree_keys)
        except Exception:
            return False
    if number_tree is None and (content_references or object_owners):
        return False

    page_keys = {_object_key(page.obj) for page in pdf.pages}
    if None in page_keys:
        return False
    page_containers: dict[_ObjectKey, set[_ObjectKey]] = {}
    rendered_by_page: dict[_ObjectKey, set[_ObjectKey]] = {}
    stream_contexts: dict[
        _ObjectKey,
        list[tuple[_ObjectKey, Dictionary | None]],
    ] = {}
    xobject_usage_modes_by_page: dict[
        _ObjectKey,
        dict[_ObjectKey, list[str]],
    ] = {}
    xobject_usage_modes: dict[_ObjectKey, list[str]] = {}
    annotation_pages: dict[_ObjectKey, _ObjectKey] = {}
    appearance_owners: dict[
        tuple[_ObjectKey, _ObjectKey],
        set[_ObjectKey],
    ] = {}
    page_annotations: dict[_ObjectKey, list[Dictionary]] = {}
    page_content_streams: dict[_ObjectKey, list[Stream]] = {}
    page_content_stream_keys: set[_ObjectKey] = set()
    type3_charprocs = _type3_charproc_streams(pdf)
    for page_number, page in enumerate(pdf.pages, start=1):
        page_key = _object_key(page.obj)
        if page_key is None:
            return False
        resources = _page_resources(page)
        form_contexts: dict[
            _ObjectKey,
            tuple[Stream, list[Dictionary | None]],
        ] = {}
        page_usage_modes: dict[_ObjectKey, list[str]] = {}
        rendered = _rendered_xobjects(
            page,
            resources,
            f"page {page_number} content",
            resource_contexts=form_contexts,
            usage_modes=page_usage_modes,
        )
        containers = {page_key, *rendered}
        content_streams = _content_streams(page, page_number)
        page_content_streams[page_key] = content_streams
        for stream in content_streams:
            stream_key = _object_key(stream)
            if stream_key is None:
                return False
            containers.add(stream_key)
            page_content_stream_keys.add(stream_key)
            stream_contexts.setdefault(stream_key, []).append((page_key, resources))

        annots = resolve_indirect(page.obj.get("/Annots"))
        if annots is None:
            annotations = []
        elif not isinstance(annots, Array):
            return False
        else:
            annotations = []
            annotation_names: set[bytes] = set()
            for item in annots:
                annotation = resolve_indirect(item)
                if not isinstance(annotation, Dictionary):
                    return False
                annotation_page = resolve_indirect(annotation.get("/P"))
                if annotation_page is not None and (
                    not isinstance(annotation_page, Dictionary)
                    or not _same_object(annotation_page, page.obj)
                ):
                    return False
                annotation_name = resolve_indirect(annotation.get("/NM"))
                if annotation_name is not None:
                    if not isinstance(annotation_name, String):
                        return False
                    name_key = bytes(annotation_name)
                    if name_key in annotation_names:
                        return False
                    annotation_names.add(name_key)
                annotation_key = _object_key(annotation)
                if annotation_key is None or annotation_key in annotation_pages:
                    return False
                annotation_pages[annotation_key] = page_key
                annotations.append(annotation)

                for appearance, appearance_resources in _annotation_appearance_streams(
                    annotation, resources
                ):
                    appearance_key = _object_key(appearance)
                    if appearance_key is None:
                        return False
                    containers.add(appearance_key)
                    rendered.add(appearance_key)
                    appearance_owners.setdefault(
                        (appearance_key, page_key),
                        set(),
                    ).add(annotation_key)
                    _, contexts = form_contexts.setdefault(
                        appearance_key,
                        (appearance, []),
                    )
                    _append_resource_context(contexts, appearance_resources)
                    page_usage_modes.setdefault(appearance_key, []).append("unmarked")
                    nested_rendered = _rendered_xobjects(
                        appearance,
                        appearance_resources,
                        "annotation appearance content",
                        resource_contexts=form_contexts,
                        usage_modes=page_usage_modes,
                    )
                    rendered.update(nested_rendered)
                    containers.update(nested_rendered)

        for owner, nested_resources in _iter_content_streams_with_resources(page):
            owner = resolve_indirect(owner)
            owner_key = _object_key(owner)
            if (
                not isinstance(owner, Stream)
                or owner_key is None
                or owner_key not in type3_charprocs
            ):
                continue
            resources = resolve_indirect(nested_resources)
            _, contexts = form_contexts.setdefault(
                owner_key,
                (owner, []),
            )
            _append_resource_context(
                contexts,
                resources if isinstance(resources, Dictionary) else None,
            )

        page_annotations[page_key] = annotations
        for stream_key, (_, contexts) in form_contexts.items():
            stream_contexts.setdefault(stream_key, []).extend(
                (page_key, context) for context in contexts
            )
        xobject_usage_modes_by_page[page_key] = page_usage_modes
        for object_key, modes in page_usage_modes.items():
            xobject_usage_modes.setdefault(object_key, []).extend(modes)
        rendered_by_page[page_key] = rendered
        page_containers[page_key] = containers

    for (container_key, _), (
        container,
        _,
        page,
        stream_owner,
    ) in content_references.items():
        page_key = _object_key(page)
        if page_key is None or page_key not in page_keys:
            return False
        if isinstance(container, Stream):
            if container_key not in page_containers[page_key]:
                return False
            owners = appearance_owners.get((container_key, page_key))
            if stream_owner is not None:
                if owners is None or _object_key(stream_owner) not in owners:
                    return False
        elif container_key != page_key:
            return False
        elif stream_owner is not None:
            return False

    for (object_key, _), (referenced_object, _, page) in object_owners.items():
        page_key = _object_key(page)
        if page_key is None or page_key not in page_keys:
            return False
        annotation_page = annotation_pages.get(object_key)
        if annotation_page is not None:
            if annotation_page != page_key:
                return False
        elif (
            not isinstance(referenced_object, Stream)
            or object_key not in rendered_by_page[page_key]
            or any(
                mode != "unmarked"
                for mode in xobject_usage_modes_by_page[page_key].get(
                    object_key,
                    [],
                )
            )
        ):
            return False

    structured_xobjects_by_page: dict[_ObjectKey, set[_ObjectKey]] = {
        page_key: set() for page_key in page_keys if page_key is not None
    }
    for (container_key, _), (
        container,
        _,
        reference_page,
        _,
    ) in content_references.items():
        if isinstance(container, Stream):
            page_key = _object_key(reference_page)
            if page_key is None:
                return False
            structured_xobjects_by_page[page_key].add(container_key)
    for (object_key, _), (
        referenced_object,
        _,
        reference_page,
    ) in object_owners.items():
        if isinstance(referenced_object, Stream):
            page_key = _object_key(reference_page)
            if page_key is None:
                return False
            structured_xobjects_by_page[page_key].add(object_key)

    reference_owners_by_container: dict[
        _ObjectKey,
        dict[int, _ObjectKey],
    ] = {}
    for (container_key, mcid), (_, owner, _, _) in content_references.items():
        owner_key = _object_key(owner)
        if owner_key is None:
            return False
        reference_owners_by_container.setdefault(container_key, {})[mcid] = owner_key
    used_parent_keys: set[int] = set()
    validated_containers: set[_ObjectKey] = set()

    def validate_content_container(
        container: Dictionary | Stream,
        actual_mcids: set[int],
        duplicate_mcid: bool,
    ) -> bool:
        container_key = _object_key(container)
        if container_key is None:
            return False
        if "/StructParent" in container and "/StructParents" in container:
            return False
        expected_owners = reference_owners_by_container.get(container_key, {})
        if duplicate_mcid or actual_mcids != set(expected_owners):
            return False
        if not actual_mcids:
            return "/StructParents" not in container

        parent_key = resolve_indirect(container.get("/StructParents"))
        if (
            not isinstance(parent_key, int)
            or isinstance(parent_key, bool)
            or parent_key < 0
            or parent_key in used_parent_keys
        ):
            return False
        used_parent_keys.add(parent_key)
        if number_tree is None:
            return False
        try:
            parent_array = resolve_indirect(number_tree.get(parent_key))
        except Exception:
            return False
        if not isinstance(parent_array, Array) or len(parent_array) > _MAX_ARRAY_ITEMS:
            return False

        for mcid, mapped in enumerate(parent_array):
            mapped = resolve_indirect(mapped)
            expected_owner = expected_owners.get(mcid)
            if mapped is None:
                if expected_owner is not None:
                    return False
                continue
            if expected_owner is None or not isinstance(mapped, Dictionary):
                return False
            if _object_key(mapped) != expected_owner:
                return False
        return all(mcid < len(parent_array) for mcid in expected_owners)

    streams: dict[_ObjectKey, Stream] = {}
    for container, _, _, _ in content_references.values():
        if not isinstance(container, Stream):
            continue
        container_key = _object_key(container)
        if container_key is None:
            return False
        streams[container_key] = container
    for item in pdf.objects:
        if isinstance(item, Stream) and (
            resolve_indirect(item.get("/Subtype")) == Name.Form
            or resolve_indirect(item.get("/PatternType")) == 1
        ):
            item_key = _object_key(item)
            if item_key is None:
                return False
            streams[item_key] = item
    streams.update(
        (stream_key, stream) for stream_key, (stream, _) in type3_charprocs.items()
    )
    for stream in streams.values():
        stream_key = _object_key(stream)
        if stream_key is None:
            return False
        contexts = stream_contexts.get(stream_key, [])
        if not contexts:
            if stream_key in type3_charprocs:
                contexts = [
                    (None, resources) for resources in type3_charprocs[stream_key][1]
                ]
            else:
                resources = resolve_indirect(stream.get("/Resources"))
                contexts = [
                    (
                        None,
                        resources if isinstance(resources, Dictionary) else None,
                    )
                ]

        actual_mcids: set[int] | None = None
        for page_key, resources in contexts:
            mcids, invalid = _collect_content_mcids(
                stream,
                resources,
                "Form XObject content",
            )
            if invalid or (actual_mcids is not None and mcids != actual_mcids):
                return False
            actual_mcids = mcids
            if (
                mcids
                and page_key is not None
                and _has_untagged_painting(
                    stream,
                    resources,
                    structured_xobjects_by_page[page_key],
                )
            ):
                return False

        actual_mcids = actual_mcids or set()
        if not validate_content_container(stream, actual_mcids, False):
            return False
        validated_containers.add(stream_key)
        if (
            actual_mcids
            and stream_key not in type3_charprocs
            and stream_key not in page_content_stream_keys
            and xobject_usage_modes.get(stream_key) != ["unmarked"]
        ):
            return False

    for page_number, page in enumerate(pdf.pages, start=1):
        page_key = _object_key(page.obj)
        if page_key is None:
            return False
        if "/StructParent" in page.obj:
            return False
        content_streams = page_content_streams[page_key]
        if any(
            _object_key(stream) in validated_containers for stream in content_streams
        ):
            mcids: set[int] = set()
            duplicate = False
            for stream in content_streams:
                if _object_key(stream) in validated_containers:
                    continue
                stream_mcids, invalid = _collect_content_mcids(
                    stream,
                    _page_resources(page),
                    f"page {page_number} content",
                )
                duplicate = duplicate or invalid or bool(mcids & stream_mcids)
                mcids.update(stream_mcids)
        else:
            mcids, duplicate = _collect_page_mcids(page, page_number)
        if not validate_content_container(page.obj, mcids, duplicate):
            return False
        validated_containers.add(page_key)
        if _has_untagged_painting(
            page,
            _page_resources(page),
            structured_xobjects_by_page[page_key],
        ):
            return False

        for annotation in page_annotations[page_key]:
            if resolve_indirect(annotation.get("/Subtype")) == Name.Popup:
                continue
            annotation_key = _object_key(annotation)
            if (
                annotation_key is None
                or (annotation_key, page_key) not in object_owners
            ):
                return False

    if any(
        _object_key(container) not in validated_containers
        for container, _, _, _ in content_references.values()
    ):
        return False

    object_parent_owners: dict[
        _ObjectKey,
        tuple[Dictionary | Stream, _ObjectKey],
    ] = {}
    for (object_key, _), (
        referenced_object,
        expected_owner,
        _,
    ) in object_owners.items():
        existing = object_parent_owners.get(object_key)
        if existing is not None and existing[1] != expected_owner:
            return False
        object_parent_owners[object_key] = (referenced_object, expected_owner)

    for referenced_object, expected_owner in object_parent_owners.values():
        if (
            "/StructParent" in referenced_object
            and "/StructParents" in referenced_object
        ):
            return False
        parent_key = resolve_indirect(referenced_object.get("/StructParent"))
        if (
            not isinstance(parent_key, int)
            or isinstance(parent_key, bool)
            or parent_key < 0
            or parent_key in used_parent_keys
        ):
            return False
        used_parent_keys.add(parent_key)
        if number_tree is None:
            return False
        try:
            mapped = resolve_indirect(number_tree.get(parent_key))
        except Exception:
            return False
        if not isinstance(mapped, Dictionary) or _object_key(mapped) != expected_owner:
            return False

    for item in pdf.objects:
        if not isinstance(item, Stream):
            continue
        object_key = _object_key(item)
        has_struct_parent = "/StructParent" in item
        has_struct_parents = "/StructParents" in item
        if has_struct_parent and has_struct_parents:
            return False
        if has_struct_parent and object_key not in object_parent_owners:
            return False
        if has_struct_parents and object_key not in validated_containers:
            return False

    if number_tree_keys != used_parent_keys:
        return False

    if "/ParentTreeNextKey" in structure_root:
        next_key = resolve_indirect(structure_root.get("/ParentTreeNextKey"))
        if (
            not isinstance(next_key, int)
            or isinstance(next_key, bool)
            or next_key < 0
            or (number_tree_keys and next_key <= max(number_tree_keys))
        ):
            return False
    return True


def _repair_parent_tree(
    pdf: pikepdf.Pdf,
    structure_root: Dictionary,
    elements: list[Dictionary],
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
    object_owners: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[Dictionary | Stream, _ObjectKey, Dictionary],
    ],
) -> bool:
    """Replace only a malformed parent tree whose reachable mappings are valid."""
    entries: dict[int, pikepdf.Object] = {}
    references_by_container: dict[
        _ObjectKey,
        tuple[Dictionary | Stream, dict[int, Dictionary]],
    ] = {}
    for (container_key, mcid), (container, owner, _, _) in content_references.items():
        if mcid >= _MAX_ARRAY_ITEMS:
            return False
        grouped = references_by_container.setdefault(container_key, (container, {}))
        grouped[1][mcid] = owner

    used_parent_keys: set[int] = set()
    for container, owners in references_by_container.values():
        parent_key = resolve_indirect(container.get("/StructParents"))
        if (
            not isinstance(parent_key, int)
            or isinstance(parent_key, bool)
            or parent_key < 0
            or parent_key in used_parent_keys
        ):
            return False
        used_parent_keys.add(parent_key)
        parent_array = Array([None] * (max(owners) + 1))
        for mcid, owner in owners.items():
            parent_array[mcid] = owner
        entries[parent_key] = parent_array

    elements_by_key = {
        key: element
        for element in elements
        if (key := _object_key(element)) is not None
    }
    referenced_objects: dict[_ObjectKey, tuple[Dictionary | Stream, _ObjectKey]] = {}
    for (object_key, _), (referenced_object, owner_key, _) in object_owners.items():
        existing = referenced_objects.get(object_key)
        if existing is not None:
            if existing[1] != owner_key:
                return False
            continue
        referenced_objects[object_key] = (referenced_object, owner_key)

    for referenced_object, owner_key in referenced_objects.values():
        parent_key = resolve_indirect(referenced_object.get("/StructParent"))
        owner = elements_by_key.get(owner_key)
        if (
            not isinstance(parent_key, int)
            or isinstance(parent_key, bool)
            or parent_key < 0
            or parent_key in used_parent_keys
            or owner is None
        ):
            return False
        used_parent_keys.add(parent_key)
        entries[parent_key] = owner

    if len(entries) * 2 > _MAX_ARRAY_ITEMS:
        return False

    had_parent_tree = "/ParentTree" in structure_root
    old_parent_tree = structure_root.get("/ParentTree")
    had_next_key = "/ParentTreeNextKey" in structure_root
    old_next_key = structure_root.get("/ParentTreeNextKey")
    candidate = NumberTree.new(pdf)
    for key, value in sorted(entries.items()):
        candidate[key] = value
    structure_root["/ParentTree"] = candidate.obj
    structure_root["/ParentTreeNextKey"] = max(entries, default=-1) + 1

    try:
        if _valid_parent_tree(
            pdf,
            candidate.obj,
            content_references,
            object_owners,
            structure_root,
        ):
            return True
    except Exception:
        pass

    if had_parent_tree:
        structure_root["/ParentTree"] = old_parent_tree
    else:
        del structure_root["/ParentTree"]
    if had_next_key:
        structure_root["/ParentTreeNextKey"] = old_next_key
    else:
        del structure_root["/ParentTreeNextKey"]
    return False


def _normalize_language(properties: Dictionary) -> bool:
    if "/Lang" not in properties:
        return False
    value = resolve_indirect(properties.get("/Lang"))
    language = str(value) if isinstance(value, String) else None
    if language is not None and _is_valid_bcp47(language):
        return False
    properties["/Lang"] = String("und")
    return True


def _sanitize_content_stream(
    owner: pikepdf.Page | Stream,
    resources: Dictionary | None,
    *,
    remove_mcids: bool,
    description: str,
    rewrite_streams: list[Stream] | None = None,
    pdf: pikepdf.Pdf | None = None,
) -> tuple[int, int]:
    try:
        instructions = list(pikepdf.parse_content_stream(owner))
    except Exception as exc:
        raise ConversionError(
            f"Cannot create logical structure: {description} cannot be parsed"
        ) from exc

    languages_normalized = 0
    mcids_removed = 0
    stream_changed = False
    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        operands = instruction.operands
        if instruction.operator not in (_BDC, _DP) or len(operands) < 2:
            continue

        raw_properties = resolve_indirect(operands[1])
        inline = isinstance(raw_properties, Dictionary)
        properties = raw_properties
        if isinstance(properties, Name):
            properties = _named_property(resources, properties)
        if not isinstance(properties, Dictionary):
            continue

        language_changed = _normalize_language(properties)
        languages_normalized += int(language_changed)
        if remove_mcids and "/MCID" in properties:
            del properties["/MCID"]
            mcids_removed += 1
            stream_changed = stream_changed or inline
        stream_changed = stream_changed or (inline and language_changed)

    if stream_changed:
        try:
            content = pikepdf.unparse_content_stream(instructions)
            if rewrite_streams is None:
                if not isinstance(owner, Stream):
                    raise TypeError("page content streams were not provided")
                owner.write(content)
            elif rewrite_streams:
                if not isinstance(owner, pikepdf.Page) or pdf is None:
                    raise TypeError("PDF object was not provided for page content")
                owner.obj["/Contents"] = pdf.make_stream(content)
        except Exception as exc:
            raise ConversionError(
                f"Cannot create logical structure: {description} cannot be rewritten"
            ) from exc
    return languages_normalized, mcids_removed


def _repair_marked_content(
    streams: list[Stream],
    description: str,
    *,
    pdf: pikepdf.Pdf | None = None,
) -> list[Stream]:
    parsed_streams: list[
        tuple[
            Stream,
            list[pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage],
        ]
    ] = []
    flat: list[
        list[int | pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage]
    ] = []
    for stream_index, stream in enumerate(streams):
        try:
            instructions = list(pikepdf.parse_content_stream(stream))
        except Exception as exc:
            raise ConversionError(
                f"Cannot create logical structure: {description} cannot be parsed"
            ) from exc
        parsed_streams.append((stream, instructions))
        flat.extend([stream_index, instruction] for instruction in instructions)

    changed_streams: set[int] = set()
    nesting: list[str] = []
    index = 0
    while index < len(flat):
        instruction = flat[index][1]
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            index += 1
            continue

        operator = instruction.operator
        if operator in (_BMC, _BDC):
            nesting.append("marked")
            index += 1
            continue
        if operator == _BT:
            if "text" in nesting:
                raise ConversionError(
                    f"Cannot create logical structure: {description} contains "
                    "nested text objects"
                )
            nesting.append("text")
            index += 1
            continue
        if operator not in (_EMC, _ET):
            index += 1
            continue

        expected = "marked" if operator == _EMC else "text"
        if nesting and nesting[-1] == expected:
            nesting.pop()
            index += 1
            continue
        if expected not in nesting:
            changed_streams.add(int(flat[index][0]))
            del flat[index]
            continue

        if index + 1 >= len(flat):
            raise ConversionError(
                f"Cannot create logical structure: {description} has "
                "crossed marked-content and text-object boundaries"
            )
        next_instruction = flat[index + 1][1]
        next_expected = None
        if not isinstance(next_instruction, pikepdf.ContentStreamInlineImage):
            if next_instruction.operator == _EMC:
                next_expected = "marked"
            elif next_instruction.operator == _ET:
                next_expected = "text"
        if len(nesting) < 2 or next_expected != nesting[-1] or expected != nesting[-2]:
            raise ConversionError(
                f"Cannot create logical structure: {description} has "
                "crossed marked-content and text-object boundaries"
            )
        first_stream = int(flat[index][0])
        second_stream = int(flat[index + 1][0])
        flat[index][1], flat[index + 1][1] = (
            flat[index + 1][1],
            flat[index][1],
        )
        changed_streams.update((first_stream, second_stream))

    if nesting and parsed_streams:
        last_stream = len(parsed_streams) - 1
        for item in reversed(nesting):
            operator = _EMC if item == "marked" else _ET
            flat.append(
                [
                    last_stream,
                    pikepdf.ContentStreamInstruction([], operator),
                ]
            )
        changed_streams.add(last_stream)

    if changed_streams and pdf is not None:
        try:
            content = pikepdf.unparse_content_stream(
                [instruction for _, instruction in flat]
            )
            return [pdf.make_stream(content)]
        except Exception as exc:
            raise ConversionError(
                f"Cannot create logical structure: {description} cannot be rewritten"
            ) from exc

    rewritten: list[
        list[pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage]
    ] = [[] for _ in parsed_streams]
    for stream_index, instruction in flat:
        rewritten[int(stream_index)].append(instruction)

    for stream_index in changed_streams:
        stream = parsed_streams[stream_index][0]
        try:
            stream.write(pikepdf.unparse_content_stream(rewritten[stream_index]))
        except Exception as exc:
            raise ConversionError(
                f"Cannot create logical structure: {description} cannot be rewritten"
            ) from exc
    return streams


def _sanitize_marked_content(
    pdf: pikepdf.Pdf,
    *,
    remove_mcids: bool,
    include_page_streams: bool,
    preserve_stream_keys: frozenset[_ObjectKey] = frozenset(),
) -> tuple[int, int, int]:
    processed: set[tuple[_ObjectKey, _ObjectKey | None]] = set()
    languages_normalized = 0
    mcids_removed = 0
    structure_keys_removed = 0

    def sanitize(
        stream: Stream,
        resources: Dictionary | None,
        description: str,
    ) -> None:
        nonlocal languages_normalized, mcids_removed, structure_keys_removed
        stream_key = _object_key(stream)
        if stream_key is None:
            raise ConversionError(
                "Cannot create logical structure: a content stream is direct"
            )
        if stream_key in preserve_stream_keys:
            return
        resource_key = _object_key(resources)
        deduplicate = resources is None or resource_key is not None
        key = (stream_key, resource_key)
        if deduplicate:
            if key in processed:
                return
            processed.add(key)

        normalized, removed = _sanitize_content_stream(
            stream,
            resources,
            remove_mcids=remove_mcids,
            description=description,
        )
        languages_normalized += normalized
        mcids_removed += removed
        if remove_mcids:
            _repair_marked_content([stream], description)
        if remove_mcids:
            for structure_key in ("/StructParent", "/StructParents"):
                if structure_key in stream:
                    del stream[structure_key]
                    structure_keys_removed += 1

    for page_number, page in enumerate(pdf.pages, start=1):
        resources = _page_resources(page)
        page_streams = _content_streams(page, page_number)
        form_contexts: dict[
            _ObjectKey,
            tuple[Stream, list[Dictionary | None]],
        ] = {}
        _rendered_xobjects(
            page,
            resources,
            f"page {page_number} content",
            resource_contexts=form_contexts,
        )
        for form, contexts in form_contexts.values():
            for context in contexts:
                sanitize(
                    form,
                    context,
                    f"Form XObject content on page {page_number}",
                )

        annots = resolve_indirect(page.obj.get("/Annots"))
        if isinstance(annots, Array):
            for item in annots:
                annotation = resolve_indirect(item)
                if not isinstance(annotation, Dictionary):
                    continue
                for appearance, appearance_resources in _annotation_appearance_streams(
                    annotation, resources
                ):
                    sanitize(
                        appearance,
                        appearance_resources,
                        f"annotation appearance on page {page_number}",
                    )
                    appearance_contexts: dict[
                        _ObjectKey,
                        tuple[Stream, list[Dictionary | None]],
                    ] = {}
                    _rendered_xobjects(
                        appearance,
                        appearance_resources,
                        f"annotation appearance on page {page_number}",
                        resource_contexts=appearance_contexts,
                    )
                    for form, contexts in appearance_contexts.values():
                        for context in contexts:
                            sanitize(
                                form,
                                context,
                                (
                                    "nested annotation appearance content on "
                                    f"page {page_number}"
                                ),
                            )

        if include_page_streams:
            normalized, removed = _sanitize_content_stream(
                page,
                resources,
                remove_mcids=remove_mcids,
                description=f"page {page_number} content",
                rewrite_streams=page_streams,
                pdf=pdf,
            )
            languages_normalized += normalized
            mcids_removed += removed

        for owner, nested_resources in _iter_content_streams_with_resources(page):
            if not isinstance(owner, Stream):
                continue
            resolved_resources = resolve_indirect(nested_resources)
            sanitize(
                owner,
                resolved_resources
                if isinstance(resolved_resources, Dictionary)
                else None,
                f"nested content on page {page_number}",
            )

    for item in pdf.objects:
        if not isinstance(item, Stream):
            continue
        subtype = resolve_indirect(item.get("/Subtype"))
        pattern_type = resolve_indirect(item.get("/PatternType"))
        if subtype != Name.Form and pattern_type != 1:
            continue
        resources = resolve_indirect(item.get("/Resources"))
        sanitize(
            item,
            resources if isinstance(resources, Dictionary) else None,
            "Form XObject or tiling pattern content",
        )

    for charproc, contexts in _type3_charproc_streams(pdf).values():
        for resources in contexts:
            sanitize(
                charproc,
                resources,
                "Type 3 CharProc content",
            )

    return languages_normalized, mcids_removed, structure_keys_removed


def _remove_stream_structure_keys(pdf: pikepdf.Pdf) -> int:
    removed = 0
    for item in pdf.objects:
        if not isinstance(item, Stream):
            continue
        for key in ("/StructParent", "/StructParents"):
            if key in item:
                del item[key]
                removed += 1
    return removed


def _semantic_rectangle(
    value: object,
    context: str,
) -> tuple[float, float, float, float]:
    value = resolve_indirect(value)
    if not isinstance(value, Array) or len(value) != 4:
        raise ConversionError(
            f"Cannot create semantic structure: {context} is malformed"
        )
    try:
        raw_x0, raw_y0, raw_x1, raw_y1 = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(
            f"Cannot create semantic structure: {context} is malformed"
        ) from exc
    if not all(math.isfinite(item) for item in (raw_x0, raw_y0, raw_x1, raw_y1)):
        raise ConversionError(
            f"Cannot create semantic structure: {context} is malformed"
        )
    x0, x1 = sorted((raw_x0, raw_x1))
    y0, y1 = sorted((raw_y0, raw_y1))
    if x1 <= x0 or y1 <= y0:
        raise ConversionError(f"Cannot create semantic structure: {context} is empty")
    return x0, y0, x1, y1


def _make_annotation_elements(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    parent: Dictionary,
    parent_tree: NumberTree,
    next_key: int,
    page_number: int,
    seen_annotations: set[_ObjectKey],
    prelinked_annotations: dict[_ObjectKey, Dictionary] | None = None,
    optional_content: _DefaultOCVisibility | None = None,
) -> tuple[list[Dictionary], int]:
    annots = resolve_indirect(page.obj.get("/Annots"))
    if annots is None:
        return [], next_key
    if not isinstance(annots, Array):
        raise ConversionError(
            f"Cannot create logical structure: page {page_number} has a "
            "malformed /Annots entry"
        )

    elements: list[Dictionary] = []
    annotation_names: set[bytes] = set()
    for index, item in enumerate(annots):
        annotation = resolve_indirect(item)
        if not isinstance(annotation, Dictionary):
            raise ConversionError(
                f"Cannot create logical structure: page {page_number} has a "
                "malformed annotation"
            )
        if "/OC" in annotation and not _optional_content_is_visible(
            optional_content,
            annotation.get("/OC"),
            f"page {page_number} annotation",
        ):
            for key in ("/StructParent", "/StructParents"):
                if key in annotation:
                    del annotation[key]
            continue
        annotation_key = _object_key(annotation)
        if annotation_key is not None and annotation_key in seen_annotations:
            if str(resolve_indirect(annotation.get("/Subtype"))) == "/Widget":
                raise ConversionError(
                    "Cannot create logical structure: a Widget annotation is "
                    "reused in multiple /Annots entries and cannot be cloned "
                    "without corrupting the AcroForm field tree"
                )
            relationship_keys = [
                key for key in ("/Popup", "/Parent", "/IRT") if key in annotation
            ]
            if relationship_keys:
                raise ConversionError(
                    "Cannot create logical structure: an annotation with "
                    f"{', '.join(relationship_keys)} is reused in multiple "
                    "/Annots entries and cannot be cloned without corrupting "
                    "annotation relationships"
                )
            annotation = pdf.make_indirect(Dictionary(annotation))
            annots[index] = annotation
        elif not annotation.is_indirect:
            annotation = pdf.make_indirect(annotation)
            annots[index] = annotation
        if annotation_key is not None:
            seen_annotations.add(annotation_key)
        indirect_key = _object_key(annotation)
        if indirect_key is None:
            raise ConversionError(
                f"Cannot create logical structure: page {page_number} annotation "
                "could not be made indirect"
            )
        seen_annotations.add(indirect_key)

        annotation_name = resolve_indirect(annotation.get("/NM"))
        if isinstance(annotation_name, String):
            name_key = bytes(annotation_name)
            if name_key in annotation_names:
                del annotation["/NM"]
            else:
                annotation_names.add(name_key)
        elif annotation_name is not None:
            del annotation["/NM"]
        if "/P" in annotation:
            annotation["/P"] = page.obj
        if "/StructParents" in annotation:
            del annotation["/StructParents"]
        annotation["/StructParent"] = next_key
        prelinked = (
            prelinked_annotations.get(indirect_key)
            if prelinked_annotations is not None
            else None
        )
        if prelinked is not None:
            parent_tree[next_key] = prelinked
            next_key += 1
            continue
        annotation_bbox = (
            _semantic_rectangle(
                annotation.get("/Rect"),
                f"page {page_number} annotation /Rect",
            )
            if prelinked_annotations is not None
            else None
        )
        object_reference = pdf.make_indirect(
            Dictionary(Type=Name.OBJR, Obj=annotation, Pg=page.obj)
        )
        subtype = resolve_indirect(annotation.get("/Subtype"))
        if subtype == Name.Widget:
            _ensure_widget_tooltip(annotation)
        role = (
            Name.Link
            if subtype == Name.Link
            else Name.Form
            if subtype == Name.Widget
            else Name.Annot
        )
        element_properties = Dictionary(
            Type=Name.StructElem,
            S=role,
            P=parent,
            Pg=page.obj,
            K=object_reference,
        )
        if annotation_bbox is not None:
            element_properties["/A"] = Dictionary(
                O=Name.Layout,
                BBox=Array(annotation_bbox),
            )
        element = pdf.make_indirect(element_properties)
        parent_tree[next_key] = element
        elements.append(element)
        next_key += 1
    return elements, next_key


def _bounded_structure_children(
    pdf: pikepdf.Pdf,
    parent: Dictionary,
    children: list[Dictionary],
    *,
    parent_capacity: int = _MAX_ARRAY_ITEMS,
) -> list[Dictionary]:
    current = children
    while len(current) > parent_capacity:
        groups: list[Dictionary] = []
        for offset in range(0, len(current), _MAX_ARRAY_ITEMS):
            chunk = current[offset : offset + _MAX_ARRAY_ITEMS]
            group = pdf.make_indirect(
                Dictionary(
                    Type=Name.StructElem,
                    S=Name.Part,
                    K=Array(chunk),
                )
            )
            for child in chunk:
                child["/P"] = group
            groups.append(group)
        current = groups

    for child in current:
        child["/P"] = parent
    return current


@dataclass(slots=True)
class _SemanticBinding:
    page_number: int
    container: Dictionary | Stream
    stream: Stream | None
    source: str
    source_index: int
    bbox: tuple[float, float, float, float]
    mcid: int | None = None


@dataclass(slots=True)
class _SemanticFormInvocation:
    page_number: int
    page_xobject_index: int
    resource_name: Name
    source: Stream
    span_ids: frozenset[str]
    expected_xobjects: dict[int, tuple[str, str | None]]
    source_prefix: str
    children: tuple[_SemanticFormInvocation, ...] = ()
    clone: Stream | None = None
    target_name: Name | None = None


@dataclass(frozen=True, slots=True)
class _MarkedVectorScope:
    marked_content_index: int
    actual_text: str | None
    alt_text: str | None
    bbox: tuple[float, float, float, float]
    mixed_paint: bool


@dataclass(frozen=True, slots=True)
class _FormSemanticSummary:
    text: str
    actual_text: str | None
    alt_text: str | None
    bbox: tuple[float, float, float, float] | None
    font_size: float | None
    font_name: str | None
    invisible: bool
    image_visibility_uncertain: bool
    has_semantic_paint: bool
    has_artifact_paint: bool
    unclassified_vector_paint: bool
    ambiguous_unclassified_vector_paint: bool
    described_vector_scopes: int
    semantic_leaf_count: int
    bind_as_figure: bool
    vector_review_required: bool


@dataclass(frozen=True, slots=True)
class _PageGeometry:
    media_box: tuple[float, float, float, float]
    crop_box: tuple[float, float, float, float]
    rotation: int

    @property
    def media_size(self) -> tuple[float, float]:
        left, bottom, right, top = self.media_box
        width, height = right - left, top - bottom
        return (height, width) if self.rotation in {90, 270} else (width, height)

    @property
    def visual_size(self) -> tuple[float, float]:
        left, bottom, right, top = self.crop_box
        width, height = right - left, top - bottom
        return (height, width) if self.rotation in {90, 270} else (width, height)

    def validate_layout_size(self, width: float, height: float) -> None:
        expected_width, expected_height = self.media_size
        if not (
            math.isclose(width, expected_width, rel_tol=1e-9, abs_tol=1e-6)
            and math.isclose(height, expected_height, rel_tol=1e-9, abs_tol=1e-6)
        ):
            raise ConversionError(
                "Cannot create semantic structure: digital layout page geometry "
                "does not match the PDF MediaBox"
            )

    def default_to_visual_bbox(
        self,
        bbox: tuple[float, float, float, float],
        *,
        frame: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:
        x0, x1 = sorted((bbox[0], bbox[2]))
        y0, y1 = sorted((bbox[1], bbox[3]))
        left, bottom, right, top = frame or self.crop_box
        if self.rotation == 90:
            return y0 - bottom, x0 - left, y1 - bottom, x1 - left
        if self.rotation == 180:
            return right - x1, y0 - bottom, right - x0, y1 - bottom
        if self.rotation == 270:
            return top - y1, right - x1, top - y0, right - x0
        return x0 - left, top - y1, x1 - left, top - y0

    def visual_to_default_bbox(
        self,
        bbox: tuple[float, float, float, float],
        *,
        frame: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:
        bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
        left, bottom, right, top = frame or self.crop_box
        if self.rotation == 90:
            return (
                left + bbox_top,
                bottom + bbox_left,
                left + bbox_bottom,
                bottom + bbox_right,
            )
        if self.rotation == 180:
            return (
                right - bbox_right,
                bottom + bbox_top,
                right - bbox_left,
                bottom + bbox_bottom,
            )
        if self.rotation == 270:
            return (
                right - bbox_bottom,
                top - bbox_right,
                right - bbox_top,
                top - bbox_left,
            )
        return (
            left + bbox_left,
            top - bbox_bottom,
            left + bbox_right,
            top - bbox_top,
        )

    def pdfminer_to_visual_bbox(
        self,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float] | None:
        left, bottom, right, top = bbox
        if right <= left or top <= bottom:
            return None
        _, media_height = self.media_size
        media_top_bbox = (
            left,
            media_height - top,
            right,
            media_height - bottom,
        )
        default_bbox = self.visual_to_default_bbox(
            media_top_bbox,
            frame=self.media_box,
        )
        clipped_bbox = _bbox_intersection(default_bbox, self.crop_box)
        return (
            self.default_to_visual_bbox(clipped_bbox)
            if clipped_bbox is not None
            else None
        )


def _semantic_span_id(page_number: int, source: str, index: int) -> str:
    return f"page-{page_number}:{source}-{index}"


def _page_box_coordinates(
    box: object,
    name: str,
) -> tuple[float, float, float, float]:
    if not isinstance(box, Array) or len(box) != 4:
        raise ConversionError(
            f"Cannot create semantic structure: effective {name} is missing"
        )
    try:
        left, bottom, right, top = (float(value) for value in box)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(
            f"Cannot create semantic structure: effective {name} is malformed"
        ) from exc
    if not all(math.isfinite(value) for value in (left, bottom, right, top)):
        raise ConversionError(
            f"Cannot create semantic structure: effective {name} is malformed"
        )
    if right <= left or top <= bottom:
        raise ConversionError(
            f"Cannot create semantic structure: effective {name} is empty"
        )
    return left, bottom, right, top


def _page_geometry(page: pikepdf.Page) -> _PageGeometry:
    try:
        media_box = _page_box_coordinates(page.mediabox, "MediaBox")
        crop_box = _page_box_coordinates(page.cropbox, "CropBox")
        rotation = page.rotation
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(
            "Cannot create semantic structure: effective page geometry is malformed"
        ) from exc
    if rotation not in {0, 90, 180, 270}:
        raise ConversionError(
            "Cannot create semantic structure: page rotation is not a multiple of 90"
        )
    return _PageGeometry(media_box, crop_box, rotation)


def _ocr_bbox_to_default(
    value: object,
    x_scale: float,
    y_scale: float,
    geometry: _PageGeometry,
) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        raise ConversionError(
            "Cannot create semantic structure: invalid OCR line geometry"
        )
    try:
        media_visual_bbox = (
            float(value["left"]) * x_scale,
            float(value["top"]) * y_scale,
            float(value["right"]) * x_scale,
            float(value["bottom"]) * y_scale,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(
            "Cannot create semantic structure: invalid OCR line geometry"
        ) from exc
    if (
        not all(math.isfinite(item) for item in media_visual_bbox)
        or media_visual_bbox[0] >= media_visual_bbox[2]
        or media_visual_bbox[1] >= media_visual_bbox[3]
    ):
        raise ConversionError(
            "Cannot create semantic structure: invalid OCR line geometry"
        )
    return geometry.visual_to_default_bbox(
        media_visual_bbox,
        frame=geometry.media_box,
    )


def _bbox_intersection(
    bbox: tuple[float, float, float, float],
    frame: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    intersection = (
        max(bbox[0], frame[0]),
        max(bbox[1], frame[1]),
        min(bbox[2], frame[2]),
        min(bbox[3], frame[3]),
    )
    if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
        return None
    return intersection


def _ocr_semantic_inputs(
    pdf: pikepdf.Pdf,
    manifest: dict[str, object],
) -> tuple[
    dict[int, tuple[object, ...]],
    dict[str, _SemanticBinding],
    dict[int, tuple[Name, Stream]],
    frozenset[_ObjectKey],
    dict[str, object],
    dict[str, str],
    dict[int, tuple[float, ...]],
]:
    from .semantics import (
        ArtifactKind,
        ArtifactReference,
        BoundingBox,
        SemanticSpan,
    )

    if (
        manifest.get("schema_version") != 1
        or manifest.get("type") != "pdftopdfa-ocr-document"
        or manifest.get("page_count") != len(pdf.pages)
    ):
        raise ConversionError("Cannot create semantic structure: invalid OCR manifest")
    raw_pages = manifest.get("pages")
    if not isinstance(raw_pages, list):
        raise ConversionError("Cannot create semantic structure: invalid OCR pages")

    spans_by_page: dict[int, tuple[object, ...]] = {}
    bindings: dict[str, _SemanticBinding] = {}
    form_targets: dict[int, tuple[Name, Stream]] = {}
    preserved_forms: set[_ObjectKey] = set()
    forced_artifacts: dict[str, object] = {}
    actual_text_overrides: dict[str, str] = {}
    column_gutters_by_page: dict[int, tuple[float, ...]] = {}
    seen_pages: set[int] = set()
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            raise ConversionError(
                "Cannot create semantic structure: invalid OCR page manifest"
            )
        page_index = raw_page.get("page_index")
        form_name = raw_page.get("form_name")
        coordinates = raw_page.get("coordinates")
        lines = raw_page.get("lines")
        if (
            isinstance(page_index, bool)
            or not isinstance(page_index, int)
            or page_index < 0
            or page_index >= len(pdf.pages)
            or page_index in seen_pages
            or not isinstance(form_name, str)
            or not form_name.startswith("/OCR-")
            or not isinstance(coordinates, dict)
            or not isinstance(lines, list)
        ):
            raise ConversionError(
                "Cannot create semantic structure: invalid OCR page manifest"
            )
        seen_pages.add(page_index)
        page = pdf.pages[page_index]
        resources = _page_resources(page)
        xobjects = (
            resolve_indirect(resources.get("/XObject"))
            if resources is not None
            else None
        )
        form = (
            resolve_indirect(xobjects.get(form_name))
            if isinstance(xobjects, Dictionary)
            else None
        )
        if (
            not isinstance(form, Stream)
            or resolve_indirect(form.get("/Subtype")) != Name.Form
        ):
            raise ConversionError(
                f"Cannot create semantic structure: OCR Form {form_name} is missing"
            )
        form_key = _object_key(form)
        if form_key is None or form_key in preserved_forms:
            raise ConversionError(
                "Cannot create semantic structure: OCR Form is not page-unique"
            )
        preserved_forms.add(form_key)
        form_resources = resolve_indirect(form.get("/Resources"))
        actual_mcids, invalid = _collect_content_mcids(
            form,
            (form_resources if isinstance(form_resources, Dictionary) else resources),
            f"OCR Form content on page {page_index + 1}",
        )
        if invalid:
            raise ConversionError(
                "Cannot create semantic structure: OCR Form MCIDs are malformed"
            )

        geometry = _page_geometry(page)
        media_width, media_height = geometry.media_size
        try:
            coordinate_width = float(coordinates["width"])
            coordinate_height = float(coordinates["height"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ConversionError(
                "Cannot create semantic structure: invalid OCR coordinates"
            ) from exc
        if (
            not math.isfinite(coordinate_width)
            or not math.isfinite(coordinate_height)
            or coordinate_width <= 0
            or coordinate_height <= 0
        ):
            raise ConversionError(
                "Cannot create semantic structure: invalid OCR coordinates"
            )
        x_scale = media_width / coordinate_width
        y_scale = media_height / coordinate_height
        raw_layout = raw_page.get("layout")
        raw_columns = (
            raw_layout.get("selected_columns") if isinstance(raw_layout, dict) else None
        )
        if raw_columns is not None:
            if not isinstance(raw_columns, list):
                raise ConversionError(
                    "Cannot create semantic structure: invalid OCR layout columns"
                )
            visual_columns = []
            for raw_column in raw_columns:
                if not isinstance(raw_column, dict):
                    raise ConversionError(
                        "Cannot create semantic structure: invalid OCR layout column"
                    )
                default_column = _ocr_bbox_to_default(
                    raw_column,
                    x_scale,
                    y_scale,
                    geometry,
                )
                clipped_column = _bbox_intersection(
                    default_column,
                    geometry.crop_box,
                )
                if clipped_column is not None:
                    visual_columns.append(
                        geometry.default_to_visual_bbox(clipped_column)
                    )
            visual_columns.sort(key=lambda column: (column[0], column[2]))
            gutters = tuple(
                (left[2] + right[0]) / 2
                for left, right in zip(
                    visual_columns,
                    visual_columns[1:],
                    strict=False,
                )
                if 0 < (left[2] + right[0]) / 2 < geometry.visual_size[0]
            )
            if gutters:
                column_gutters_by_page[page_index] = tuple(dict.fromkeys(gutters))
        declared_mcids: set[int] = set()
        page_spans = []
        for raw_line in lines:
            if not isinstance(raw_line, dict):
                raise ConversionError(
                    "Cannot create semantic structure: invalid OCR line"
                )
            mcid = raw_line.get("mcid")
            text = raw_line.get("text")
            raw_bbox = raw_line.get("bbox")
            confidence = raw_line.get("confidence")
            if (
                isinstance(mcid, bool)
                or not isinstance(mcid, int)
                or mcid < 0
                or mcid in declared_mcids
                or not isinstance(text, str)
                or not text
                or not isinstance(raw_bbox, dict)
            ):
                raise ConversionError(
                    "Cannot create semantic structure: invalid OCR line"
                )
            declared_mcids.add(mcid)
            if mcid not in actual_mcids:
                continue
            default_bbox = _ocr_bbox_to_default(
                raw_bbox,
                x_scale,
                y_scale,
                geometry,
            )
            span_id = _semantic_span_id(page_index + 1, "ocr", mcid)
            clipped_default_bbox = _bbox_intersection(
                default_bbox,
                geometry.crop_box,
            )
            span_text = text
            if clipped_default_bbox is not None and any(
                not math.isclose(
                    clipped,
                    original,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
                for clipped, original in zip(
                    clipped_default_bbox,
                    default_bbox,
                    strict=True,
                )
            ):
                raw_words = raw_line.get("words")
                if not isinstance(raw_words, list) or not raw_words:
                    raise ConversionError(
                        "Cannot create semantic structure: partially clipped OCR "
                        "line has no word geometry"
                    )
                visible_words = []
                for raw_word in raw_words:
                    if not isinstance(raw_word, dict):
                        raise ConversionError(
                            "Cannot create semantic structure: invalid OCR word"
                        )
                    word_text = raw_word.get("text")
                    if not isinstance(word_text, str) or not word_text:
                        raise ConversionError(
                            "Cannot create semantic structure: invalid OCR word"
                        )
                    word_default_bbox = _ocr_bbox_to_default(
                        raw_word.get("bbox"),
                        x_scale,
                        y_scale,
                        geometry,
                    )
                    if _bbox_intersection(word_default_bbox, geometry.crop_box):
                        visible_words.append(word_text)
                if visible_words:
                    span_text = " ".join(visible_words)
                    actual_text_overrides[span_id] = span_text
                else:
                    clipped_default_bbox = None

            if clipped_default_bbox is not None:
                visual_bbox = geometry.default_to_visual_bbox(clipped_default_bbox)
                try:
                    box = BoundingBox(*visual_bbox)
                except (TypeError, ValueError) as exc:
                    raise ConversionError(
                        "Cannot create semantic structure: invalid OCR line geometry"
                    ) from exc
                page_spans.append(
                    SemanticSpan(
                        span_id,
                        span_text,
                        box,
                        font_size=max(0.1, box.height * 0.8),
                        confidence=(
                            float(confidence)
                            if isinstance(confidence, (int, float))
                            and not isinstance(confidence, bool)
                            else None
                        ),
                        invisible=True,
                    )
                )
                binding_bbox = (box.left, box.top, box.right, box.bottom)
            else:
                forced_artifacts[span_id] = ArtifactReference(
                    span_id,
                    page_index + 1,
                    ArtifactKind.LAYOUT,
                    "Layout",
                    None,
                )
                binding_bbox = geometry.default_to_visual_bbox(default_bbox)
            bindings[span_id] = _SemanticBinding(
                page_index + 1,
                form,
                form,
                "ocr",
                mcid,
                binding_bbox,
                mcid,
            )
        if actual_mcids != declared_mcids:
            details = []
            missing = sorted(declared_mcids - actual_mcids)
            if missing:
                details.append(f"manifest lines without marked content: {missing}")
            extra = sorted(actual_mcids - declared_mcids)
            if extra:
                details.append(f"marked content without manifest line: {extra}")
            raise ConversionError(
                "Cannot create semantic structure: OCR Form MCIDs do not match "
                f"manifest on page {page_index + 1} ({'; '.join(details)})"
            )
        spans_by_page[page_index] = tuple(page_spans)
        form_targets[page_index] = (Name(form_name), form)
    return (
        spans_by_page,
        bindings,
        form_targets,
        frozenset(preserved_forms),
        forced_artifacts,
        actual_text_overrides,
        column_gutters_by_page,
    )


def _normalized_text_positions(value: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if normalized.isalnum():
                characters.append(normalized)
                positions.append(index)
    return "".join(characters), tuple(positions)


def _native_text_fragments(*values: str | None) -> frozenset[str]:
    fragments: set[str] = set()
    for value in values:
        if not value or "(cid:" in value:
            continue
        normalized, _positions = _normalized_text_positions(value)
        if normalized:
            fragments.add(normalized)
        for token in re.findall(r"[^\W_]+", value, flags=re.UNICODE):
            normalized_token, _positions = _normalized_text_positions(token)
            if normalized_token:
                fragments.add(normalized_token)
    return frozenset(fragments)


def _remove_native_text_from_ocr_word(
    text: str,
    bbox: tuple[float, float, float, float],
    native_evidence: tuple[
        tuple[tuple[float, float, float, float], frozenset[str]], ...
    ],
) -> tuple[str, tuple[float, float, float, float], bool] | None:
    normalized, positions = _normalized_text_positions(text)
    if not normalized:
        return text, bbox, False
    ranges: set[tuple[int, int]] = set()
    width = bbox[2] - bbox[0]
    for native_bbox, fragments in native_evidence:
        for fragment in fragments:
            required_overlap = 0.35
            if fragment == normalized:
                match = (0, len(normalized))
            elif (
                len(fragment) >= 3
                and any(character.isalpha() for character in fragment)
                and normalized.startswith(fragment)
            ):
                match = (0, len(fragment))
            elif (
                len(fragment) >= 3
                and any(character.isalpha() for character in fragment)
                and normalized.endswith(fragment)
            ):
                match = (len(normalized) - len(fragment), len(normalized))
            elif (
                min(len(fragment), len(normalized)) >= 4
                and abs(len(fragment) - len(normalized))
                <= max(2, round(max(len(fragment), len(normalized)) * 0.2))
                and SequenceMatcher(
                    None,
                    fragment,
                    normalized,
                    autojunk=False,
                ).ratio()
                >= 0.75
            ):
                match = (0, len(normalized))
                required_overlap = 0.65
            else:
                continue
            segment = (
                bbox[0] + width * match[0] / len(normalized),
                bbox[1],
                bbox[0] + width * match[1] / len(normalized),
                bbox[3],
            )
            intersection = _bbox_intersection(segment, native_bbox)
            segment_area = (segment[2] - segment[0]) * (segment[3] - segment[1])
            if (
                intersection is not None
                and segment_area > 0
                and (intersection[2] - intersection[0])
                * (intersection[3] - intersection[1])
                >= segment_area * required_overlap
            ):
                ranges.add(match)
    if not ranges:
        return text, bbox, False
    longest_length = max(end - start for start, end in ranges)
    longest = {(start, end) for start, end in ranges if end - start == longest_length}
    if len(longest) != 1:
        raise ConversionError(
            "Cannot create semantic structure: ambiguous native/OCR text overlap"
        )
    start, end = next(iter(longest))
    if start == 0 and end == len(normalized):
        return None
    if start == 0:
        cut = positions[end - 1] + 1
        remaining = text[cut:].strip()
        remaining_bbox = (
            bbox[0] + width * end / len(normalized),
            bbox[1],
            bbox[2],
            bbox[3],
        )
    elif end == len(normalized):
        cut = positions[start]
        remaining = text[:cut].strip()
        remaining_bbox = (
            bbox[0],
            bbox[1],
            bbox[0] + width * start / len(normalized),
            bbox[3],
        )
    else:
        raise ConversionError(
            "Cannot create semantic structure: ambiguous native/OCR text overlap"
        )
    remaining_normalized, _positions = _normalized_text_positions(remaining)
    return (remaining, remaining_bbox, True) if remaining_normalized else None


def _deduplicate_ocr_native_text(
    pdf: pikepdf.Pdf,
    manifest: dict[str, object],
    ocr_spans: dict[int, tuple[object, ...]],
    ocr_bindings: dict[str, _SemanticBinding],
    forced_artifacts: dict[str, object],
    actual_text_overrides: dict[str, str],
    digital_spans: dict[int, tuple[object, ...]],
    source_actual_texts: dict[str, str],
) -> None:
    from .semantics import (
        ArtifactKind,
        ArtifactReference,
        BoundingBox,
        SemanticSpan,
        SpanKind,
    )

    raw_pages = manifest.get("pages")
    if not isinstance(raw_pages, list):
        raise ConversionError("Cannot create semantic structure: invalid OCR pages")
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            raise ConversionError(
                "Cannot create semantic structure: invalid OCR page manifest"
            )
        page_index = raw_page.get("page_index")
        if isinstance(page_index, bool) or not isinstance(page_index, int):
            raise ConversionError(
                "Cannot create semantic structure: invalid OCR page manifest"
            )
        native_evidence = []
        for span in digital_spans.get(page_index, ()):
            if (
                not isinstance(span, SemanticSpan)
                or span.kind is not SpanKind.TEXT
                or span.invisible
            ):
                continue
            fragments = _native_text_fragments(
                span.text,
                source_actual_texts.get(span.id),
            )
            if fragments:
                native_evidence.append(
                    (
                        (
                            span.bbox.left,
                            span.bbox.top,
                            span.bbox.right,
                            span.bbox.bottom,
                        ),
                        fragments,
                    )
                )
        if not native_evidence:
            continue

        coordinates = raw_page.get("coordinates")
        lines = raw_page.get("lines")
        if not isinstance(coordinates, dict) or not isinstance(lines, list):
            raise ConversionError("Cannot create semantic structure: invalid OCR page")
        geometry = _page_geometry(pdf.pages[page_index])
        media_width, media_height = geometry.media_size
        try:
            x_scale = media_width / float(coordinates["width"])
            y_scale = media_height / float(coordinates["height"])
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            ZeroDivisionError,
        ) as exc:
            raise ConversionError(
                "Cannot create semantic structure: invalid OCR coordinates"
            ) from exc
        spans = {
            span.id: span
            for span in ocr_spans.get(page_index, ())
            if isinstance(span, SemanticSpan)
        }
        replacements: dict[str, SemanticSpan | None] = {}
        for raw_line in lines:
            if not isinstance(raw_line, dict):
                raise ConversionError(
                    "Cannot create semantic structure: invalid OCR line"
                )
            mcid = raw_line.get("mcid")
            if isinstance(mcid, bool) or not isinstance(mcid, int):
                raise ConversionError(
                    "Cannot create semantic structure: invalid OCR line"
                )
            span_id = _semantic_span_id(page_index + 1, "ocr", mcid)
            span = spans.get(span_id)
            raw_words = raw_line.get("words")
            if span is None or raw_words is None:
                continue
            if not isinstance(raw_words, list) or not raw_words:
                raise ConversionError(
                    "Cannot create semantic structure: invalid OCR words"
                )
            changed = False
            remaining_words: list[tuple[str, tuple[float, float, float, float]]] = []
            for raw_word in raw_words:
                if not isinstance(raw_word, dict):
                    raise ConversionError(
                        "Cannot create semantic structure: invalid OCR word"
                    )
                word_text = raw_word.get("text")
                if not isinstance(word_text, str) or not word_text:
                    raise ConversionError(
                        "Cannot create semantic structure: invalid OCR word"
                    )
                word_default_bbox = _ocr_bbox_to_default(
                    raw_word.get("bbox"),
                    x_scale,
                    y_scale,
                    geometry,
                )
                clipped_word_bbox = _bbox_intersection(
                    word_default_bbox,
                    geometry.crop_box,
                )
                if clipped_word_bbox is None:
                    continue
                word_bbox = geometry.default_to_visual_bbox(clipped_word_bbox)
                result = _remove_native_text_from_ocr_word(
                    word_text,
                    word_bbox,
                    tuple(native_evidence),
                )
                if result is None:
                    changed = True
                    continue
                remaining_text, remaining_bbox, word_changed = result
                changed = changed or word_changed
                remaining_words.append((remaining_text, remaining_bbox))
            if not changed:
                continue
            if not remaining_words:
                replacements[span_id] = None
                forced_artifacts[span_id] = ArtifactReference(
                    span_id,
                    page_index + 1,
                    ArtifactKind.LAYOUT,
                    "Layout",
                    None,
                )
                actual_text_overrides.pop(span_id, None)
                continue
            remaining_text = " ".join(text for text, _bbox in remaining_words)
            remaining_bbox = (
                min(bbox[0] for _text, bbox in remaining_words),
                min(bbox[1] for _text, bbox in remaining_words),
                max(bbox[2] for _text, bbox in remaining_words),
                max(bbox[3] for _text, bbox in remaining_words),
            )
            box = BoundingBox(*remaining_bbox)
            replacements[span_id] = replace(
                span,
                text=remaining_text,
                bbox=box,
            )
            binding = ocr_bindings.get(span_id)
            if binding is None:
                raise ConversionError(
                    "Cannot create semantic structure: OCR binding is missing"
                )
            binding.bbox = (box.left, box.top, box.right, box.bottom)
            actual_text_overrides[span_id] = remaining_text
        if replacements:
            ocr_spans[page_index] = tuple(
                replacement
                for span in ocr_spans.get(page_index, ())
                if (replacement := replacements.get(span.id, span)) is not None
            )


def _deduplicate_invisible_digital_text(
    digital_spans: dict[int, tuple[object, ...]],
    forced_artifacts: dict[str, object],
    source_actual_texts: dict[str, str],
    source_alt_texts: dict[str, str],
) -> None:
    """Artifact exact invisible overlays of equivalent visible digital text."""
    from .semantics import (
        ArtifactKind,
        ArtifactReference,
        SemanticSpan,
        SpanKind,
    )

    def normalized(value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split())

    def equivalent(left: SemanticSpan, right: SemanticSpan) -> bool:
        if normalized(left.text) != normalized(right.text):
            return False
        if normalized(source_actual_texts.get(left.id, left.text)) != normalized(
            source_actual_texts.get(right.id, right.text)
        ):
            return False
        if normalized(source_alt_texts.get(left.id)) != normalized(
            source_alt_texts.get(right.id)
        ):
            return False
        intersection_width = max(
            0.0,
            min(left.bbox.right, right.bbox.right)
            - max(left.bbox.left, right.bbox.left),
        )
        intersection_height = max(
            0.0,
            min(left.bbox.bottom, right.bbox.bottom)
            - max(left.bbox.top, right.bbox.top),
        )
        intersection = intersection_width * intersection_height
        left_area = left.bbox.width * left.bbox.height
        right_area = right.bbox.width * right.bbox.height
        union = left_area + right_area - intersection
        return union > 0 and intersection / union >= 0.8

    for page_index, spans in tuple(digital_spans.items()):
        visible = tuple(
            span
            for span in spans
            if isinstance(span, SemanticSpan)
            and span.kind is SpanKind.TEXT
            and not span.invisible
            and normalized(span.text)
        )
        duplicate_ids = {
            span.id
            for span in spans
            if isinstance(span, SemanticSpan)
            and span.kind is SpanKind.TEXT
            and span.invisible
            and normalized(span.text)
            and any(equivalent(span, candidate) for candidate in visible)
        }
        if not duplicate_ids:
            continue
        digital_spans[page_index] = tuple(
            span
            for span in spans
            if not isinstance(span, SemanticSpan) or span.id not in duplicate_ids
        )
        for span_id in duplicate_ids:
            forced_artifacts[span_id] = ArtifactReference(
                span_id,
                page_index + 1,
                ArtifactKind.LAYOUT,
                "Layout",
                None,
            )


def _direct_artifact_items(
    owner: pikepdf.Page | Stream,
    description: str,
    inherited_resources: Dictionary | None = None,
    *,
    mcid_items_out: dict[tuple[str, int], tuple[int, ...]] | None = None,
    optional_content: _DefaultOCVisibility | None = None,
    initial_state: InvocationPaintState | None = None,
    initial_clip_polygon: ClipPolygon | None = None,
) -> tuple[
    frozenset[tuple[str, int]],
    dict[tuple[str, int], str],
    dict[tuple[str, int], str],
    bool,
    bool,
    bool,
    bool,
    tuple[float, float, float, float] | None,
    tuple[_MarkedVectorScope, ...],
    frozenset[tuple[str, int]],
]:
    """Return direct marked-content provenance and semantic vector evidence."""
    from .digital_layout import (
        InvocationPaintState,
        _clip_bbox_to_polygon,
        _normalize_polygon,
        _parallelogram_polygon,
        _polygon_bbox,
        _polygon_intersection,
        _polygon_tolerance,
        _rect_polygon,
    )

    try:
        instructions = pikepdf.parse_content_stream(owner)
    except Exception as exc:
        raise ConversionError(
            f"Cannot create semantic structure: {description} cannot be parsed"
        ) from exc

    resources = (
        _page_resources(owner)
        if isinstance(owner, pikepdf.Page)
        else resolve_indirect(owner.get("/Resources"))
    )
    if not isinstance(resources, Dictionary) or not resources:
        resources = inherited_resources
    nesting: list[tuple[str, bool, bool, str | None, str | None, int | None]] = []
    marked_mcids: list[int | None] = []
    artifact_items: set[tuple[str, int]] = set()
    hidden_items: set[tuple[str, int]] = set()
    actual_text_items: dict[tuple[str, int], str] = {}
    alt_text_items: dict[tuple[str, int], str] = {}
    has_semantic_vector_paint = False
    has_unclassified_vector_paint = False
    has_ambiguous_unclassified_vector_paint = False
    unclassified_vector_bbox: tuple[float, float, float, float] | None = None
    has_artifact_vector_paint = False
    vector_scope_states: dict[
        int,
        dict[str, object],
    ] = {}
    marked_content_index = 0
    text_index = 0
    xobject_index = 0
    current_matrix = pikepdf.Matrix()
    clip_bbox: tuple[float, float, float, float] | None = None
    clip_polygon = initial_clip_polygon
    if initial_clip_polygon is not None:
        clip_bbox = _polygon_bbox(initial_clip_polygon)
    elif isinstance(owner, pikepdf.Page):
        clip_bbox = _page_geometry(owner).crop_box
    elif (
        isinstance(owner, Stream)
        and resolve_indirect(owner.get("/Subtype")) == Name.Form
    ):
        raw_bbox = resolve_indirect(owner.get("/BBox"))
        if isinstance(raw_bbox, Array) and len(raw_bbox) == 4:
            try:
                left, bottom, right, top = (float(value) for value in raw_bbox)
                clip_bbox = (
                    min(left, right),
                    min(bottom, top),
                    max(left, right),
                    max(bottom, top),
                )
            except (TypeError, ValueError):
                clip_bbox = None
    if initial_clip_polygon is None and clip_bbox is not None:
        clip_polygon = _rect_polygon(
            (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            clip_bbox,
        )
    graphics_stack: list[
        tuple[
            pikepdf.Matrix,
            tuple[float, float, float, float] | None,
            ClipPolygon | None,
            bool,
            InvocationPaintState,
        ]
    ] = []
    path_points: list[tuple[float, float]] = []
    path_operators: list[str] = []
    pending_clip = False
    text_clip_pending = False
    paint_state = initial_state or InvocationPaintState()
    clip_is_exact = clip_polygon == () or not paint_state.clip_visibility_uncertain

    def numeric_operands(operands) -> tuple[float, ...] | None:
        try:
            values = tuple(float(resolve_indirect(value)) for value in operands)
        except (TypeError, ValueError):
            return None
        return values if all(math.isfinite(value) for value in values) else None

    def current_path_bbox(*, stroke: bool) -> tuple[float, float, float, float] | None:
        if not path_points:
            return None
        left = min(point[0] for point in path_points)
        bottom = min(point[1] for point in path_points)
        right = max(point[0] for point in path_points)
        top = max(point[1] for point in path_points)
        if stroke:
            radius = paint_state.line_width / 2.0
            row_x_radius = radius * math.hypot(
                current_matrix.a,
                current_matrix.c,
            )
            row_y_radius = radius * math.hypot(
                current_matrix.b,
                current_matrix.d,
            )
            if path_operators == ["m", "l"] and len(path_points) == 2:
                start, end = path_points
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.hypot(dx, dy)
                determinant = (
                    current_matrix.a * current_matrix.d
                    - current_matrix.b * current_matrix.c
                )
                inverse_tangent_x = current_matrix.d * dx - current_matrix.c * dy
                inverse_tangent_y = -current_matrix.b * dx + current_matrix.a * dy
                inverse_length = math.hypot(inverse_tangent_x, inverse_tangent_y)
                if length == 0 or determinant == 0 or inverse_length == 0:
                    cap_factor = math.sqrt(2.0) if paint_state.line_cap == 2 else 1.0
                    x_expansion = row_x_radius * cap_factor
                    y_expansion = row_y_radius * cap_factor
                else:
                    user_tangent_x = inverse_tangent_x / inverse_length
                    user_tangent_y = inverse_tangent_y / inverse_length
                    user_normal_x = -user_tangent_y
                    user_normal_y = user_tangent_x
                    tangent_x = (
                        current_matrix.a * user_tangent_x
                        + current_matrix.c * user_tangent_y
                    )
                    tangent_y = (
                        current_matrix.b * user_tangent_x
                        + current_matrix.d * user_tangent_y
                    )
                    normal_x = (
                        current_matrix.a * user_normal_x
                        + current_matrix.c * user_normal_y
                    )
                    normal_y = (
                        current_matrix.b * user_normal_x
                        + current_matrix.d * user_normal_y
                    )
                    if paint_state.line_cap == 1:
                        x_expansion = row_x_radius
                        y_expansion = row_y_radius
                    else:
                        cap_extension = 1.0 if paint_state.line_cap == 2 else 0.0
                        x_expansion = radius * (
                            abs(normal_x) + abs(tangent_x) * cap_extension
                        )
                        y_expansion = radius * (
                            abs(normal_y) + abs(tangent_y) * cap_extension
                        )
                left -= x_expansion
                bottom -= y_expansion
                right += x_expansion
                top += y_expansion
            else:
                miter_factor = max(1.0, paint_state.miter_limit)
                left -= row_x_radius * miter_factor
                bottom -= row_y_radius * miter_factor
                right += row_x_radius * miter_factor
                top += row_y_radius * miter_factor
        return left, bottom, right, top

    def rectangular_path_bbox() -> tuple[float, float, float, float] | None:
        points = list(path_points)
        explicitly_closed = False
        if path_operators in (["re"], ["re", "h"]):
            pass
        elif path_operators in (
            ["m", "l", "l", "l"],
            ["m", "l", "l", "l", "h"],
        ):
            pass
        elif path_operators == ["m", "l", "l", "l", "l"]:
            explicitly_closed = True
        else:
            return None
        if len(points) != (5 if explicitly_closed else 4):
            return None
        coordinates = tuple(value for point in points for value in point)
        extent = max(
            max(point[0] for point in points) - min(point[0] for point in points),
            max(point[1] for point in points) - min(point[1] for point in points),
            1.0,
        )
        tolerance = max(
            1e-7,
            extent * 1e-9,
            max(math.ulp(abs(value)) for value in coordinates) * 4,
        )

        def close(first: float, second: float) -> bool:
            return abs(first - second) <= tolerance

        if explicitly_closed:
            if not all(
                close(first, last) for first, last in zip(points[0], points[-1])
            ):
                return None
            points.pop()

        left = min(point[0] for point in points)
        bottom = min(point[1] for point in points)
        right = max(point[0] for point in points)
        top = max(point[1] for point in points)
        if close(left, right) or close(bottom, top):
            return (0.0, 0.0, 0.0, 0.0)
        corners = ((left, bottom), (left, top), (right, bottom), (right, top))
        if any(
            sum(all(close(a, b) for a, b in zip(point, corner)) for point in points)
            != 1
            for corner in corners
        ):
            return None
        for first, second in zip(points, (*points[1:], points[0])):
            if not (close(first[0], second[0]) ^ close(first[1], second[1])):
                return None
        return left, bottom, right, top

    def rectangular_path_polygon() -> ClipPolygon | None:
        if path_operators in (["re"], ["re", "h"]) and len(path_points) == 4:
            return _normalize_polygon(tuple(path_points))
        bbox = rectangular_path_bbox()
        if bbox is not None:
            return _rect_polygon((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), bbox)
        if path_operators not in (
            ["m", "l", "l", "l"],
            ["m", "l", "l", "l", "h"],
            ["m", "l", "l", "l", "l"],
            ["m", "l", "l", "l", "l", "h"],
        ):
            return None
        points = list(path_points)
        if len(points) == 5:
            tolerance = _polygon_tolerance(tuple(points))
            if any(
                abs(first - last) > tolerance
                for first, last in zip(points[0], points[-1])
            ):
                return None
            points.pop()
        return _parallelogram_polygon(tuple(points))

    def described_stroke_geometry_status() -> bool:
        has_segment = any(
            operator in {"l", "c", "v", "y", "re"} for operator in path_operators
        )
        unique_points = set(path_points)
        return has_segment and (
            len(unique_points) > 1 or paint_state.line_cap in {1, 2}
        )

    def described_fill_geometry_status() -> bool | None:
        if not path_points:
            return False
        subpaths = sum(operator in {"m", "re"} for operator in path_operators)
        if subpaths != 1 or any(
            operator in {"c", "v", "y"} for operator in path_operators
        ):
            return None
        points = list(path_points)
        tolerance = _polygon_tolerance(tuple(points))

        def points_close(
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> bool:
            return math.hypot(first[0] - second[0], first[1] - second[1]) <= tolerance

        if points_close(points[0], points[-1]):
            points.pop()
        if len(points) < 3:
            return False
        if any(
            points_close(first, second)
            for index, first in enumerate(points)
            for second in points[index + 1 :]
        ):
            return None

        extent = max(
            max(point[0] for point in points) - min(point[0] for point in points),
            max(point[1] for point in points) - min(point[1] for point in points),
            1.0,
        )
        cross_tolerance = tolerance * extent

        def cross(
            start: tuple[float, float],
            end: tuple[float, float],
            point: tuple[float, float],
        ) -> float:
            return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
                point[0] - start[0]
            )

        def on_segment(
            point: tuple[float, float],
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> bool:
            return (
                abs(cross(start, end, point)) <= cross_tolerance
                and min(start[0], end[0]) - tolerance
                <= point[0]
                <= max(start[0], end[0]) + tolerance
                and min(start[1], end[1]) - tolerance
                <= point[1]
                <= max(start[1], end[1]) + tolerance
            )

        def segments_intersect(
            first_start: tuple[float, float],
            first_end: tuple[float, float],
            second_start: tuple[float, float],
            second_end: tuple[float, float],
        ) -> bool:
            first_side = cross(first_start, first_end, second_start)
            second_side = cross(first_start, first_end, second_end)
            third_side = cross(second_start, second_end, first_start)
            fourth_side = cross(second_start, second_end, first_end)
            if (
                first_side > cross_tolerance
                and second_side < -cross_tolerance
                or first_side < -cross_tolerance
                and second_side > cross_tolerance
            ) and (
                third_side > cross_tolerance
                and fourth_side < -cross_tolerance
                or third_side < -cross_tolerance
                and fourth_side > cross_tolerance
            ):
                return True
            return (
                on_segment(second_start, first_start, first_end)
                or on_segment(second_end, first_start, first_end)
                or on_segment(first_start, second_start, second_end)
                or on_segment(first_end, second_start, second_end)
            )

        for index, current in enumerate(points):
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            if abs(cross(previous, current, following)) <= cross_tolerance:
                incoming = (
                    previous[0] - current[0],
                    previous[1] - current[1],
                )
                outgoing = (
                    following[0] - current[0],
                    following[1] - current[1],
                )
                if incoming[0] * outgoing[0] + incoming[1] * outgoing[1] > 0:
                    return None

        edges = tuple(zip(points, (*points[1:], points[0])))
        for first_index, (first_start, first_end) in enumerate(edges):
            for second_index in range(first_index + 1, len(edges)):
                if second_index in {
                    first_index + 1,
                    (first_index - 1) % len(edges),
                }:
                    continue
                second_start, second_end = edges[second_index]
                if segments_intersect(
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                ):
                    return None

        origin_x, origin_y = points[0]
        points = [(point[0] - origin_x, point[1] - origin_y) for point in points]
        twice_area = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(points, (*points[1:], points[0]))
        )
        coordinate_scale = max(
            max(abs(value) for point in points for value in point),
            1.0,
        )
        if abs(twice_area) <= max(
            1e-10,
            coordinate_scale * coordinate_scale * 1e-12,
        ):
            return False if len(points) < 4 else None
        return True

    def union_bboxes(
        boxes: tuple[tuple[float, float, float, float] | None, ...],
    ) -> tuple[float, float, float, float] | None:
        concrete = tuple(box for box in boxes if box is not None)
        if not concrete:
            return None
        return (
            min(box[0] for box in concrete),
            min(box[1] for box in concrete),
            max(box[2] for box in concrete),
            max(box[3] for box in concrete),
        )

    def point_in_clip(point: tuple[float, float]) -> bool:
        if clip_polygon is None:
            return True
        if not clip_polygon:
            return False
        tolerance = _polygon_tolerance((*clip_polygon, point))
        return all(
            (second[0] - first[0]) * (point[1] - first[1])
            - (second[1] - first[1]) * (point[0] - first[0])
            >= -tolerance
            for first, second in zip(
                clip_polygon,
                (*clip_polygon[1:], clip_polygon[0]),
            )
        )

    def straight_stroke_is_outside_clip() -> bool:
        if (
            path_operators != ["m", "l"]
            or len(path_points) != 2
            or clip_polygon is None
            or not clip_polygon
            or paint_state.line_width == 0
            or paint_state.stroke_adjust
        ):
            return False
        start, end = path_points
        if point_in_clip(start) or point_in_clip(end):
            return False
        tolerance = _polygon_tolerance((*clip_polygon, start, end))
        extent = max(
            max(point[0] for point in (*clip_polygon, start, end))
            - min(point[0] for point in (*clip_polygon, start, end)),
            max(point[1] for point in (*clip_polygon, start, end))
            - min(point[1] for point in (*clip_polygon, start, end)),
            1.0,
        )
        cross_tolerance = tolerance * extent

        def cross(
            first: tuple[float, float],
            second: tuple[float, float],
            point: tuple[float, float],
        ) -> float:
            return (second[0] - first[0]) * (point[1] - first[1]) - (
                second[1] - first[1]
            ) * (point[0] - first[0])

        def on_segment(
            point: tuple[float, float],
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> bool:
            return (
                abs(cross(first, second, point)) <= cross_tolerance
                and min(first[0], second[0]) - tolerance
                <= point[0]
                <= max(first[0], second[0]) + tolerance
                and min(first[1], second[1]) - tolerance
                <= point[1]
                <= max(first[1], second[1]) + tolerance
            )

        def segments_intersect(
            first_start: tuple[float, float],
            first_end: tuple[float, float],
            second_start: tuple[float, float],
            second_end: tuple[float, float],
        ) -> bool:
            first_side = cross(first_start, first_end, second_start)
            second_side = cross(first_start, first_end, second_end)
            third_side = cross(second_start, second_end, first_start)
            fourth_side = cross(second_start, second_end, first_end)
            if (
                first_side > cross_tolerance
                and second_side < -cross_tolerance
                or first_side < -cross_tolerance
                and second_side > cross_tolerance
            ) and (
                third_side > cross_tolerance
                and fourth_side < -cross_tolerance
                or third_side < -cross_tolerance
                and fourth_side > cross_tolerance
            ):
                return True
            return (
                on_segment(second_start, first_start, first_end)
                or on_segment(second_end, first_start, first_end)
                or on_segment(first_start, second_start, second_end)
                or on_segment(first_end, second_start, second_end)
            )

        def point_segment_distance(
            point: tuple[float, float],
            segment_start: tuple[float, float],
            segment_end: tuple[float, float],
        ) -> float:
            dx = segment_end[0] - segment_start[0]
            dy = segment_end[1] - segment_start[1]
            squared_length = dx * dx + dy * dy
            if squared_length == 0:
                return math.hypot(
                    point[0] - segment_start[0],
                    point[1] - segment_start[1],
                )
            position = max(
                0.0,
                min(
                    1.0,
                    (
                        (point[0] - segment_start[0]) * dx
                        + (point[1] - segment_start[1]) * dy
                    )
                    / squared_length,
                ),
            )
            closest = (
                segment_start[0] + position * dx,
                segment_start[1] + position * dy,
            )
            return math.hypot(point[0] - closest[0], point[1] - closest[1])

        distance = math.inf
        for clip_start, clip_end in zip(
            clip_polygon,
            (*clip_polygon[1:], clip_polygon[0]),
        ):
            if segments_intersect(start, end, clip_start, clip_end):
                return False
            distance = min(
                distance,
                point_segment_distance(start, clip_start, clip_end),
                point_segment_distance(end, clip_start, clip_end),
                point_segment_distance(clip_start, start, end),
                point_segment_distance(clip_end, start, end),
            )

        squared_column_difference = (
            current_matrix.a * current_matrix.a
            + current_matrix.b * current_matrix.b
            - current_matrix.c * current_matrix.c
            - current_matrix.d * current_matrix.d
        )
        column_product = (
            current_matrix.a * current_matrix.c + current_matrix.b * current_matrix.d
        )
        squared_frobenius_norm = (
            current_matrix.a * current_matrix.a
            + current_matrix.b * current_matrix.b
            + current_matrix.c * current_matrix.c
            + current_matrix.d * current_matrix.d
        )
        operator_norm = math.sqrt(
            max(
                0.0,
                (
                    squared_frobenius_norm
                    + math.hypot(squared_column_difference, 2.0 * column_product)
                )
                / 2.0,
            )
        )
        radius = paint_state.line_width * operator_norm / 2.0
        if paint_state.line_cap == 2:
            radius *= math.sqrt(2.0)
        return distance > radius + tolerance

    def paint_clip_relation(
        painted_bbox: tuple[float, float, float, float] | None,
    ) -> str:
        if clip_polygon == () or (
            clip_bbox is not None
            and (clip_bbox[0] >= clip_bbox[2] or clip_bbox[1] >= clip_bbox[3])
        ):
            return "hidden"
        if painted_bbox is None or clip_polygon is None:
            return "full"
        if _clip_bbox_to_polygon(painted_bbox, clip_polygon) is None:
            return "hidden"
        corners = (
            (painted_bbox[0], painted_bbox[1]),
            (painted_bbox[2], painted_bbox[1]),
            (painted_bbox[2], painted_bbox[3]),
            (painted_bbox[0], painted_bbox[3]),
        )
        return "full" if all(point_in_clip(point) for point in corners) else "partial"

    def combined_clip_relation(
        channels: tuple[tuple[float, float, float, float] | None, ...],
    ) -> str:
        relations = tuple(paint_clip_relation(bbox) for bbox in channels)
        if not relations or all(relation == "hidden" for relation in relations):
            return "hidden"
        return (
            "full" if all(relation == "full" for relation in relations) else "partial"
        )

    def visible_paint_bbox(
        painted_bbox: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if clip_bbox is not None and (
            clip_bbox[0] >= clip_bbox[2] or clip_bbox[1] >= clip_bbox[3]
        ):
            return None
        if painted_bbox is None:
            return clip_bbox
        if clip_is_exact:
            return _clip_bbox_to_polygon(painted_bbox, clip_polygon)
        if clip_bbox is None:
            return painted_bbox
        return _bbox_intersection(painted_bbox, clip_bbox)

    def shading_paint_bbox(
        operands: list[object],
    ) -> tuple[float, float, float, float] | None:
        shading_name = resolve_indirect(operands[0]) if len(operands) == 1 else None
        shadings = (
            resolve_indirect(resources.get("/Shading"))
            if isinstance(resources, Dictionary)
            else None
        )
        shading = (
            resolve_indirect(shadings.get(shading_name))
            if isinstance(shading_name, Name) and isinstance(shadings, Dictionary)
            else None
        )
        if not isinstance(shading, (Dictionary, Stream)):
            raise ConversionError(
                f"Cannot create semantic structure: {description} has malformed "
                "shading provenance"
            )
        raw_bbox = resolve_indirect(shading.get("/BBox"))
        if raw_bbox is None:
            return None
        values = numeric_operands(raw_bbox) if isinstance(raw_bbox, Array) else None
        if values is None or len(values) != 4:
            raise ConversionError(
                f"Cannot create semantic structure: {description} has malformed "
                "shading BBox"
            )
        left, bottom, right, top = values
        points = tuple(
            current_matrix.transform(point)
            for point in (
                (left, bottom),
                (right, bottom),
                (right, top),
                (left, top),
            )
        )
        return (
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        )

    def complex_color_space(operand: object) -> bool:
        color_space = resolve_indirect(operand)
        if not isinstance(color_space, Name):
            return True
        simple_names = {
            Name.DeviceGray,
            Name.DeviceRGB,
            Name.DeviceCMYK,
            Name("/G"),
            Name("/RGB"),
            Name("/CMYK"),
        }
        if color_space in simple_names:
            return False
        color_spaces = (
            resolve_indirect(resources.get("/ColorSpace"))
            if isinstance(resources, Dictionary)
            else None
        )
        resolved = (
            resolve_indirect(color_spaces.get(color_space))
            if isinstance(color_spaces, Dictionary)
            else None
        )
        return resolved not in simple_names

    def text_show_has_codes(operator_name: str, operands: list[object]) -> bool:
        value = resolve_indirect(operands[-1])
        if operator_name != "TJ":
            if not isinstance(value, String):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has malformed "
                    "direct text provenance"
                )
            return bool(bytes(value))
        if not isinstance(value, Array):
            raise ConversionError(
                f"Cannot create semantic structure: {description} has malformed "
                "direct text provenance"
            )
        has_codes = False
        for item in value:
            item = resolve_indirect(item)
            if isinstance(item, String):
                has_codes = has_codes or bool(bytes(item))
                continue
            try:
                numeric = float(item)
            except (TypeError, ValueError) as exc:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has malformed "
                    "direct text provenance"
                ) from exc
            if isinstance(item, bool) or not math.isfinite(numeric):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has malformed "
                    "direct text provenance"
                )
        return has_codes

    def commit_pending_clip() -> None:
        nonlocal clip_bbox, clip_is_exact, clip_polygon, pending_clip
        if not pending_clip:
            return
        path_bbox = current_path_bbox(stroke=False)
        path_polygon = rectangular_path_polygon()
        if path_bbox is None:
            clip_bbox = (0.0, 0.0, 0.0, 0.0)
            clip_polygon = ()
            clip_is_exact = True
        elif path_polygon is not None and clip_is_exact:
            clip_polygon = _polygon_intersection(clip_polygon, path_polygon)
            clip_bbox = _polygon_bbox(clip_polygon)
            clip_is_exact = True
        else:
            clip_bbox = (
                path_bbox
                if clip_bbox is None
                else _bbox_intersection(clip_bbox, path_bbox) or (0.0, 0.0, 0.0, 0.0)
            )
            clip_is_exact = bool(
                clip_bbox[0] >= clip_bbox[2] or clip_bbox[1] >= clip_bbox[3]
            )
            clip_polygon = () if clip_is_exact else None
        pending_clip = False

    def optional_content_hidden() -> bool:
        return any(
            kind == "marked" and hidden
            for kind, _artifact, hidden, _actual, _alt, _scope in nesting
        )

    def record(source: str, index: int, *, force_hidden: bool = False) -> None:
        item = (source, index)
        if any(
            kind == "marked" and artifact
            for kind, artifact, _hidden, _actual, _alt, _scope in nesting
        ):
            artifact_items.add(item)
            return
        if force_hidden or optional_content_hidden():
            hidden_items.add(item)
            return
        if mcid_items_out is not None:
            active_mcids = tuple(mcid for mcid in marked_mcids if mcid is not None)
            if active_mcids:
                mcid_items_out[item] = active_mcids
        scope_index = next(
            (
                scope_index
                for kind, _artifact, _hidden, _actual, _alt, scope_index in reversed(
                    nesting
                )
                if kind == "marked" and scope_index is not None
            ),
            None,
        )
        if scope_index is not None:
            vector_scope_states[scope_index]["has_other_paint"] = True
        actual_text = next(
            (
                actual_text
                for kind, _artifact, _hidden, actual_text, _alt, _scope in reversed(
                    nesting
                )
                if kind == "marked" and actual_text is not None
            ),
            None,
        )
        if actual_text is not None:
            actual_text_items[item] = actual_text
        alt_text = next(
            (
                alt_text
                for kind, _artifact, _hidden, _actual, alt_text, _scope in reversed(
                    nesting
                )
                if kind == "marked" and alt_text is not None
            ),
            None,
        )
        if alt_text is not None:
            alt_text_items[item] = alt_text

    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            record("xobject", xobject_index)
            xobject_index += 1
            continue
        operator = instruction.operator
        operands = instruction.operands
        if operator in {_BMC, _BDC}:
            current_marked_content_index = marked_content_index
            marked_content_index += 1
            tag = resolve_indirect(operands[0]) if operands else None
            if not isinstance(tag, Name):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "malformed marked content"
                )
            properties = (
                _marked_content_properties(list(operands), resources)
                if operator == _BDC
                else None
            )
            optional_hidden = False
            if tag == Name.OC:
                if operator != _BDC or properties is None:
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "malformed optional-content marked content"
                    )
                optional_hidden = not _optional_content_is_visible(
                    optional_content,
                    properties,
                    f"{description} marked content",
                )
            mcid = resolve_indirect(properties.get("/MCID")) if properties else None
            if not isinstance(mcid, int) or isinstance(mcid, bool) or mcid < 0:
                mcid = None
            marked_mcids.append(mcid)
            actual_text = (
                str(resolve_indirect(properties.get("/ActualText")))
                if properties and _valid_structure_text(properties.get("/ActualText"))
                else None
            )
            alt_text = (
                str(resolve_indirect(properties.get("/Alt")))
                if properties and _valid_structure_text(properties.get("/Alt"))
                else None
            )
            vector_scope_index = None
            if (
                (actual_text is not None or alt_text is not None)
                and tag != Name.Artifact
                and not (optional_hidden or optional_content_hidden())
            ):
                vector_scope_index = current_marked_content_index
                vector_scope_states[vector_scope_index] = {
                    "actual_text": actual_text,
                    "alt_text": alt_text,
                    "bbox": None,
                    "has_vector_paint": False,
                    "has_other_paint": False,
                }
            nesting.append(
                (
                    "marked",
                    tag == Name.Artifact,
                    optional_hidden,
                    actual_text,
                    alt_text,
                    vector_scope_index,
                )
            )
            continue
        if operator == _EMC:
            if not nesting or nesting[-1][0] != "marked":
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "unbalanced marked content"
                )
            nesting.pop()
            marked_mcids.pop()
            continue
        if operator == _BT:
            if any(
                kind == "text"
                for kind, _artifact, _hidden, _actual, _alt, _scope in nesting
            ):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "nested text objects"
                )
            text_clip_pending = False
            nesting.append(("text", False, False, None, None, None))
            continue
        if operator == _ET:
            if not nesting or nesting[-1][0] != "text":
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "crossed text and marked-content boundaries"
                )
            nesting.pop()
            if text_clip_pending and clip_polygon != ():
                clip_is_exact = False
                clip_polygon = None
            text_clip_pending = False
            continue

        operator_name = str(operator)
        expected_operands = _TEXT_SHOW_OPERAND_COUNTS.get(operator_name)
        if expected_operands is not None and len(operands) != expected_operands:
            raise ConversionError(
                f"Cannot create semantic structure: {description} has malformed "
                "direct text provenance"
            )
        if operator_name == "Do" and len(operands) != 1:
            raise ConversionError(
                f"Cannot create semantic structure: {description} has malformed "
                "direct XObject provenance"
            )
        if operator_name == "Tr":
            values = numeric_operands(operands)
            if (
                values is None
                or len(values) != 1
                or not values[0].is_integer()
                or not 0 <= values[0] <= 7
            ):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has malformed "
                    "text rendering mode"
                )
            paint_state = replace(paint_state, text_render_mode=int(values[0]))
            continue
        if operator_name == "q":
            graphics_stack.append(
                (
                    current_matrix,
                    clip_bbox,
                    clip_polygon,
                    clip_is_exact,
                    paint_state,
                )
            )
            continue
        if operator_name == "Q":
            if graphics_stack:
                (
                    current_matrix,
                    clip_bbox,
                    clip_polygon,
                    clip_is_exact,
                    paint_state,
                ) = graphics_stack.pop()
            continue
        if operator_name == "cm":
            values = numeric_operands(operands)
            if values is not None and len(values) == 6:
                current_matrix = pikepdf.Matrix(*values) @ current_matrix
            continue
        if operator_name in {"CS", "cs"}:
            if len(operands) != 1:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has malformed "
                    "color-space provenance"
                )
            field_name = (
                "stroke_color_complex"
                if operator_name == "CS"
                else "fill_color_complex"
            )
            paint_state = replace(
                paint_state,
                **{field_name: complex_color_space(operands[0])},
            )
            continue
        if operator_name in {"G", "RG", "K", "g", "rg", "k"}:
            field_name = (
                "stroke_color_complex"
                if operator_name in {"G", "RG", "K"}
                else "fill_color_complex"
            )
            paint_state = replace(paint_state, **{field_name: False})
            continue
        if operator_name in {"SCN", "scn"} and any(
            isinstance(resolve_indirect(value), Name) for value in operands
        ):
            field_name = (
                "stroke_color_complex"
                if operator_name == "SCN"
                else "fill_color_complex"
            )
            paint_state = replace(paint_state, **{field_name: True})
            continue
        if operator_name == "w":
            values = numeric_operands(operands)
            if values is not None and len(values) == 1:
                paint_state = replace(paint_state, line_width=abs(values[0]))
            continue
        if operator_name == "M":
            values = numeric_operands(operands)
            if values is not None and len(values) == 1 and values[0] >= 1:
                paint_state = replace(paint_state, miter_limit=values[0])
            continue
        if operator_name == "J":
            values = numeric_operands(operands)
            if values is not None and len(values) == 1 and values[0] in {0, 1, 2}:
                paint_state = replace(paint_state, line_cap=int(values[0]))
            continue
        if operator_name == "j":
            values = numeric_operands(operands)
            if values is not None and len(values) == 1 and values[0] in {0, 1, 2}:
                paint_state = replace(paint_state, line_join=int(values[0]))
            continue
        if operator_name == "d":
            if len(operands) == 2:
                raw_dash = resolve_indirect(operands[0])
                values = (
                    numeric_operands(raw_dash) if isinstance(raw_dash, Array) else None
                )
                phase = numeric_operands(operands[1:])
                if (
                    values is not None
                    and all(value >= 0 for value in values)
                    and (not values or any(values))
                    and phase is not None
                    and len(phase) == 1
                ):
                    paint_state = replace(
                        paint_state,
                        dash_array=values,
                        dash_phase=phase[0],
                    )
            continue
        if operator_name == "gs":
            extgstates = (
                resolve_indirect(resources.get("/ExtGState"))
                if isinstance(resources, Dictionary)
                else None
            )
            parameters = (
                resolve_indirect(extgstates.get(resolve_indirect(operands[0])))
                if len(operands) == 1 and isinstance(extgstates, Dictionary)
                else None
            )
            if not isinstance(parameters, Dictionary):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "malformed ExtGState"
                )

            def extgstate_number(
                key: str,
                *,
                minimum: float,
                maximum: float | None = None,
            ) -> float | None:
                if key not in parameters:
                    return None
                try:
                    value = float(resolve_indirect(parameters[key]))
                except (TypeError, ValueError) as exc:
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        f"malformed ExtGState {key}"
                    ) from exc
                if (
                    not math.isfinite(value)
                    or value < minimum
                    or (maximum is not None and value > maximum)
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        f"malformed ExtGState {key}"
                    )
                return value

            for key, field_name, minimum, maximum in (
                ("/LW", "line_width", 0.0, None),
                ("/ML", "miter_limit", 1.0, None),
                ("/CA", "stroke_alpha", 0.0, 1.0),
                ("/ca", "fill_alpha", 0.0, 1.0),
            ):
                value = extgstate_number(
                    key,
                    minimum=minimum,
                    maximum=maximum,
                )
                if value is not None:
                    paint_state = replace(paint_state, **{field_name: value})
            for key, field_name in (("/LC", "line_cap"), ("/LJ", "line_join")):
                value = extgstate_number(key, minimum=0, maximum=2)
                if value is not None:
                    if not value.is_integer():
                        raise ConversionError(
                            f"Cannot create semantic structure: {description} has "
                            f"malformed ExtGState {key}"
                        )
                    paint_state = replace(
                        paint_state,
                        **{field_name: int(value)},
                    )
            if "/D" in parameters:
                raw_dash = resolve_indirect(parameters["/D"])
                dash_array = (
                    resolve_indirect(raw_dash[0])
                    if isinstance(raw_dash, Array) and len(raw_dash) == 2
                    else None
                )
                dash_values = (
                    numeric_operands(dash_array)
                    if isinstance(dash_array, Array)
                    else None
                )
                phase = (
                    numeric_operands(raw_dash[1:])
                    if isinstance(raw_dash, Array) and len(raw_dash) == 2
                    else None
                )
                if (
                    dash_values is None
                    or any(value < 0 for value in dash_values)
                    or (dash_values and not any(dash_values))
                    or phase is None
                    or len(phase) != 1
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "malformed ExtGState /D"
                    )
                paint_state = replace(
                    paint_state,
                    dash_array=dash_values,
                    dash_phase=phase[0],
                )
            if "/SMask" in parameters:
                soft_mask = resolve_indirect(parameters["/SMask"])
                paint_state = replace(
                    paint_state,
                    soft_mask_active=soft_mask != Name("/None"),
                )
            if "/SA" in parameters:
                stroke_adjust = resolve_indirect(parameters["/SA"])
                if not isinstance(stroke_adjust, bool):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "malformed ExtGState /SA"
                    )
                paint_state = replace(
                    paint_state,
                    stroke_adjust=stroke_adjust,
                )
            stroke_overprint = None
            fill_overprint = None
            for key in ("/OP", "/op"):
                if key not in parameters:
                    continue
                value = resolve_indirect(parameters[key])
                if not isinstance(value, bool):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        f"malformed ExtGState {key}"
                    )
                if key == "/OP":
                    stroke_overprint = value
                else:
                    fill_overprint = value
            if stroke_overprint is not None:
                paint_state = replace(
                    paint_state,
                    stroke_overprint=stroke_overprint,
                    fill_overprint=(
                        stroke_overprint if fill_overprint is None else fill_overprint
                    ),
                )
            elif fill_overprint is not None:
                paint_state = replace(
                    paint_state,
                    fill_overprint=fill_overprint,
                )
            if "/OPM" in parameters:
                mode = extgstate_number("/OPM", minimum=0, maximum=1)
                if mode is None or not mode.is_integer():
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "malformed ExtGState /OPM"
                    )
                paint_state = replace(paint_state, overprint_mode=int(mode))
            if "/BM" in parameters:
                raw_blend_mode = resolve_indirect(parameters["/BM"])
                values = (
                    tuple(raw_blend_mode)
                    if isinstance(raw_blend_mode, Array)
                    else (raw_blend_mode,)
                )
                if not values or any(
                    not isinstance(resolve_indirect(value), Name) for value in values
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "malformed ExtGState /BM"
                    )
                paint_state = replace(
                    paint_state,
                    blend_mode_complex=any(
                        resolve_indirect(value)
                        not in {Name.Normal, Name("/Compatible")}
                        for value in values
                    ),
                )
            continue
        if operator_name in {"m", "l", "c", "v", "y", "re"}:
            path_operators.append(operator_name)
            values = numeric_operands(operands)
            if values is not None:
                if operator_name == "re" and len(values) == 4:
                    x, y, width, height = values
                    coordinates = (
                        (x, y),
                        (x + width, y),
                        (x + width, y + height),
                        (x, y + height),
                    )
                elif len(values) % 2 == 0:
                    coordinates = tuple(
                        (values[index], values[index + 1])
                        for index in range(0, len(values), 2)
                    )
                else:
                    coordinates = ()
                path_points.extend(
                    current_matrix.transform(point) for point in coordinates
                )
            continue
        if operator_name == "h":
            path_operators.append(operator_name)
            continue
        if operator_name in {"W", "W*"}:
            pending_clip = True
            continue
        if operator_name == "n":
            commit_pending_clip()
            path_points.clear()
            path_operators.clear()
            continue
        if operator_name in {"Tj", "TJ", "'", '"'}:
            if paint_state.text_render_mode in {4, 5, 6, 7} and text_show_has_codes(
                operator_name, operands
            ):
                text_clip_pending = True
            record("text", text_index)
            text_index += 1
        elif operator_name == "Do":
            resource_name = resolve_indirect(operands[0])
            xobjects = (
                resolve_indirect(resources.get("/XObject"))
                if isinstance(resources, Dictionary)
                else None
            )
            xobject = (
                resolve_indirect(xobjects.get(resource_name))
                if isinstance(resource_name, Name) and isinstance(xobjects, Dictionary)
                else None
            )
            xobject_hidden = bool(
                isinstance(xobject, Stream)
                and "/OC" in xobject
                and not _optional_content_is_visible(
                    optional_content,
                    xobject.get("/OC"),
                    f"{description} XObject {resource_name}",
                )
            )
            record("xobject", xobject_index, force_hidden=xobject_hidden)
            xobject_index += 1
        elif operator_name in _PATH_PAINTING_OPERATORS | {"sh"}:
            if optional_content_hidden():
                has_artifact_vector_paint = True
                commit_pending_clip()
                path_points.clear()
                path_operators.clear()
                continue
            stroke = operator_name in {"S", "s", "B", "B*", "b", "b*"}
            fill = operator_name in {"f", "F", "f*", "B", "B*", "b", "b*", "sh"}
            artifact_paint = any(
                kind == "marked" and artifact
                for kind, artifact, _hidden, _actual, _alt, _scope in nesting
            )
            scope_index = next(
                (
                    scope_index
                    for (
                        kind,
                        _artifact,
                        _hidden,
                        _actual_text,
                        _alt_text,
                        scope_index,
                    ) in reversed(nesting)
                    if kind == "marked" and scope_index is not None
                ),
                None,
            )
            visible_stroke = stroke and paint_state.stroke_alpha > 0
            visible_fill = fill and paint_state.fill_alpha > 0
            if not visible_stroke and not visible_fill:
                commit_pending_clip()
                path_points.clear()
                path_operators.clear()
                continue
            geometry_bbox = (
                shading_paint_bbox(operands)
                if operator_name == "sh"
                else current_path_bbox(stroke=False)
            )
            stroke_geometry_status = (
                described_stroke_geometry_status() if visible_stroke else False
            )
            fill_geometry_status = (
                (
                    geometry_bbox is None
                    or (
                        geometry_bbox[0] < geometry_bbox[2]
                        and geometry_bbox[1] < geometry_bbox[3]
                    )
                )
                if operator_name == "sh" and visible_fill
                else described_fill_geometry_status()
                if visible_fill
                else False
            )
            device_stroke_uncertain = bool(stroke_geometry_status) and (
                paint_state.line_width == 0 or paint_state.stroke_adjust
            )
            stroke_state_uncertain = bool(stroke_geometry_status) and (
                device_stroke_uncertain
                or bool(paint_state.dash_array)
                or paint_state.stroke_overprint
                or paint_state.stroke_color_complex
                or paint_state.blend_mode_complex
            )
            fill_state_uncertain = fill_geometry_status is not False and (
                paint_state.fill_overprint
                or paint_state.fill_color_complex
                or paint_state.blend_mode_complex
            )
            stroke_status: bool | None = (
                None if stroke_state_uncertain else stroke_geometry_status
            )
            fill_status: bool | None = (
                None if fill_state_uncertain else fill_geometry_status
            )
            stroke_bbox = (
                None
                if device_stroke_uncertain
                else current_path_bbox(stroke=True)
                if visible_stroke and stroke_status is not False
                else None
            )
            if stroke_bbox is not None and straight_stroke_is_outside_clip():
                stroke_status = False
                stroke_bbox = None
            fill_bbox = (
                geometry_bbox if visible_fill and fill_status is not False else None
            )
            possible_channels = tuple(
                channel
                for channel, status in (
                    (stroke_bbox, stroke_status),
                    (fill_bbox, fill_status),
                )
                if status is not False
            )
            if not possible_channels:
                commit_pending_clip()
                path_points.clear()
                path_operators.clear()
                continue
            possible_relation = combined_clip_relation(possible_channels)
            if possible_relation == "hidden":
                commit_pending_clip()
                path_points.clear()
                path_operators.clear()
                continue
            if paint_state.soft_mask_active and not artifact_paint:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "semantic vector painting under a nontrivial soft mask"
                )
            described_paint = not artifact_paint and scope_index is not None
            channels = possible_channels
            if described_paint:
                if operator_name == "sh":
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "described shading without exact painted geometry"
                    )
                if stroke_geometry_status is None or fill_geometry_status is None:
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "described vector painting with ambiguous path geometry"
                    )
                proven_channels = tuple(
                    channel
                    for channel, status in (
                        (stroke_bbox, stroke_status),
                        (fill_bbox, fill_status),
                    )
                    if status is True
                )
                if not proven_channels:
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "described vector painting with uncertain final-paint "
                        "visibility"
                    )
                proven_relation = combined_clip_relation(proven_channels)
                if proven_relation == "hidden":
                    if any(status is None for status in (stroke_status, fill_status)):
                        raise ConversionError(
                            f"Cannot create semantic structure: {description} has "
                            "described vector painting with uncertain final-paint "
                            "visibility"
                        )
                    commit_pending_clip()
                    path_points.clear()
                    path_operators.clear()
                    continue
                channels = proven_channels
            clip_relation = combined_clip_relation(channels)
            painted_bbox = union_bboxes(channels)
            visible_bbox = (
                None if clip_relation == "hidden" else visible_paint_bbox(painted_bbox)
            )
            if visible_bbox is not None:
                unclassified_form_paint = (
                    not artifact_paint
                    and scope_index is None
                    and isinstance(owner, Stream)
                    and resolve_indirect(owner.get("/Subtype")) == Name.Form
                )
                ambiguous_unclassified_form_paint = (
                    unclassified_form_paint and fill_geometry_status is None
                )
                if unclassified_form_paint and (
                    stroke_state_uncertain or fill_state_uncertain or not clip_is_exact
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "unclassified Form vector painting with uncertain final-paint "
                        "visibility"
                    )
                if (
                    operator_name == "sh"
                    and not artifact_paint
                    and scope_index is None
                    and isinstance(owner, Stream)
                    and resolve_indirect(owner.get("/Subtype")) == Name.Form
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "unclassified Form shading without exact painted geometry"
                    )
                if not clip_is_exact and not artifact_paint and scope_index is not None:
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "described vector painting under a non-rectangular clip"
                    )
                if (
                    clip_relation != "full"
                    and not artifact_paint
                    and scope_index is not None
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has "
                        "described vector painting that is partially clipped"
                    )
                if artifact_paint:
                    has_artifact_vector_paint = True
                elif ambiguous_unclassified_form_paint:
                    has_artifact_vector_paint = True
                    has_unclassified_vector_paint = True
                    has_ambiguous_unclassified_vector_paint = True
                else:
                    has_semantic_vector_paint = True
                    if scope_index is None:
                        has_unclassified_vector_paint = True
                        if unclassified_vector_bbox is None:
                            unclassified_vector_bbox = visible_bbox
                        else:
                            unclassified_vector_bbox = (
                                min(unclassified_vector_bbox[0], visible_bbox[0]),
                                min(unclassified_vector_bbox[1], visible_bbox[1]),
                                max(unclassified_vector_bbox[2], visible_bbox[2]),
                                max(unclassified_vector_bbox[3], visible_bbox[3]),
                            )
                    else:
                        scope = vector_scope_states[scope_index]
                        scope["has_vector_paint"] = True
                        scope_bbox = scope["bbox"]
                        if scope_bbox is None:
                            scope["bbox"] = visible_bbox
                        else:
                            assert isinstance(scope_bbox, tuple)
                            scope["bbox"] = (
                                min(scope_bbox[0], visible_bbox[0]),
                                min(scope_bbox[1], visible_bbox[1]),
                                max(scope_bbox[2], visible_bbox[2]),
                                max(scope_bbox[3], visible_bbox[3]),
                            )
            commit_pending_clip()
            path_points.clear()
            path_operators.clear()

    if nesting or marked_mcids:
        raise ConversionError(
            f"Cannot create semantic structure: {description} has unbalanced "
            "marked content or text objects"
        )
    vector_scopes = []
    for scope_index, state in vector_scope_states.items():
        if not state["has_vector_paint"]:
            continue
        scope_bbox = state["bbox"]
        actual_text = state["actual_text"]
        alt_text = state["alt_text"]
        if not isinstance(scope_bbox, tuple) or not (
            isinstance(actual_text, str) or isinstance(alt_text, str)
        ):
            raise ConversionError(
                f"Cannot create semantic structure: {description} marked vector "
                "scope has no usable evidence or geometry"
            )
        vector_scopes.append(
            _MarkedVectorScope(
                scope_index,
                actual_text if isinstance(actual_text, str) else None,
                alt_text if isinstance(alt_text, str) else None,
                scope_bbox,
                bool(state["has_other_paint"]),
            )
        )
    return (
        frozenset(artifact_items),
        actual_text_items,
        alt_text_items,
        has_semantic_vector_paint,
        bool(artifact_items or hidden_items) or has_artifact_vector_paint,
        has_unclassified_vector_paint,
        has_ambiguous_unclassified_vector_paint,
        unclassified_vector_bbox,
        tuple(vector_scopes),
        frozenset(hidden_items),
    )


def _digital_semantic_inputs(
    pdf: pikepdf.Pdf,
    ignored_page_xobjects: dict[int, tuple[Name, Stream]],
    optional_content: _DefaultOCVisibility | None = None,
) -> tuple[
    dict[int, tuple[object, ...]],
    dict[str, _SemanticBinding],
    dict[int, tuple[float, float]],
    frozenset[str],
    dict[int, tuple[_SemanticFormInvocation, ...]],
    dict[str, str],
    dict[str, str],
    frozenset[int],
    frozenset[int],
    dict[str, object],
]:
    import statistics

    from .digital_layout import (
        DirectTextSpan,
        DirectXObjectSpan,
        _clip_bbox_to_polygon,
        _inverse_matrix,
        _transform_polygon,
        extract_digital_layout,
    )
    from .semantics import (
        ArtifactKind,
        ArtifactReference,
        BoundingBox,
        SemanticSpan,
        SpanKind,
    )

    if optional_content is None:
        try:
            optional_content = _default_optional_content_visibility(pdf)
        except ValueError as exc:
            raise ConversionError(
                "Cannot create semantic structure: malformed optional-content "
                "default configuration"
            ) from exc
    layouts = extract_digital_layout(pdf)
    spans_by_page: dict[int, tuple[object, ...]] = {}
    bindings: dict[str, _SemanticBinding] = {}
    dimensions: dict[int, tuple[float, float]] = {}
    source_artifact_ids: set[str] = set()
    optional_artifacts: dict[str, object] = {}
    source_actual_texts: dict[str, str] = {}
    source_alt_texts: dict[str, str] = {}
    form_invocations: dict[int, tuple[_SemanticFormInvocation, ...]] = {}
    vector_review_pages: set[int] = set()
    native_reading_pages: set[int] = set()
    form_marked_items: dict[
        tuple[
            _ObjectKey,
            _ObjectKey | int,
            InvocationPaintState,
            ClipPolygon | None,
        ],
        tuple[
            frozenset[tuple[str, int]],
            dict[tuple[str, int], str],
            dict[tuple[str, int], str],
            bool,
            bool,
            bool,
            bool,
            tuple[float, float, float, float] | None,
            tuple[_MarkedVectorScope, ...],
            frozenset[tuple[str, int]],
        ],
    ] = {}
    form_summaries: dict[int, _FormSemanticSummary] = {}
    for layout in layouts:
        page = pdf.pages[layout.page_index]
        page_number = layout.page_index + 1
        (
            page_artifacts,
            page_actual_texts,
            page_alt_texts,
            _page_vectors,
            _page_artifact_paint,
            page_unclassified_vectors,
            _page_ambiguous_vectors,
            _page_unclassified_vector_bbox,
            page_vector_scopes,
            page_hidden_items,
        ) = _direct_artifact_items(
            page,
            f"page {page_number}",
            optional_content=optional_content,
        )
        if any(scope.mixed_paint for scope in page_vector_scopes):
            raise ConversionError(
                f"Cannot create semantic structure: page {page_number} has a "
                "marked text-evidence scope mixing vector and other painting"
            )
        if page_unclassified_vectors:
            vector_review_pages.add(layout.page_index)
        resources = _page_resources(page)
        geometry = _page_geometry(page)
        geometry.validate_layout_size(layout.width, layout.height)
        dimensions[layout.page_index] = geometry.visual_size
        page_spans = []
        page_form_invocations: list[_SemanticFormInvocation] = []
        ignored_xobject_invocations = 0

        def visible_source_bbox(
            span: DirectTextSpan | DirectXObjectSpan,
        ) -> tuple[float, float, float, float] | None:
            if isinstance(span, DirectTextSpan) and span.render_mode == 7:
                return None
            if isinstance(span, DirectXObjectSpan) and span.invisible:
                return None
            clipped_bbox = (
                _clip_bbox_to_polygon(span.bbox, span.clip_polygon)
                if span.clip_polygon is not None
                else _bbox_intersection(span.bbox, span.clip_bbox)
                if span.clip_bbox is not None
                else span.bbox
            )
            if (
                clipped_bbox is None
                or geometry.pdfminer_to_visual_bbox(clipped_bbox) is None
            ):
                return None
            return clipped_bbox

        def visible_form_local_bbox(
            span: DirectXObjectSpan,
            local_bbox: tuple[float, float, float, float],
            description: str,
        ) -> tuple[float, float, float, float]:
            transformed = span.local_to_page_bbox(local_bbox)
            clipped = (
                _clip_bbox_to_polygon(transformed, span.clip_polygon)
                if span.clip_polygon is not None
                else _bbox_intersection(transformed, span.clip_bbox)
                if span.clip_bbox is not None
                else transformed
            )
            if clipped is None or geometry.pdfminer_to_visual_bbox(clipped) is None:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} is outside "
                    "the visible page"
                )
            return clipped

        def visible_form_scope_bbox(
            span: DirectXObjectSpan,
            scope: _MarkedVectorScope,
        ) -> tuple[float, float, float, float]:
            return visible_form_local_bbox(
                span,
                scope.bbox,
                "Form ActualText vector scope",
            )

        def union_source_bboxes(
            boxes: list[tuple[float, float, float, float]],
        ) -> tuple[float, float, float, float] | None:
            if not boxes:
                return None
            return (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )

        def effective_form_resources(
            form: Stream,
            inherited: Dictionary | None,
            description: str,
        ) -> Dictionary | None:
            raw_resources = form.get("/Resources")
            own_resources = resolve_indirect(raw_resources)
            if raw_resources is not None and not isinstance(own_resources, Dictionary):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} resources "
                    "are malformed"
                )
            return (
                own_resources
                if isinstance(own_resources, Dictionary) and own_resources
                else inherited
            )

        def resolve_form_child(
            span: DirectXObjectSpan,
            effective_resources: Dictionary | None,
            description: str,
        ) -> Stream:
            if span.resource_name is None or effective_resources is None:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} Form "
                    "resource is missing"
                )
            resource_name = Name(f"/{span.resource_name}")
            xobjects = resolve_indirect(effective_resources.get("/XObject"))
            form = (
                resolve_indirect(xobjects.get(resource_name))
                if isinstance(xobjects, Dictionary)
                else None
            )
            if (
                not isinstance(form, Stream)
                or resolve_indirect(form.get("/Subtype")) != Name.Form
                or _object_key(form) is None
            ):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} Form "
                    f"resource {resource_name} cannot be resolved safely"
                )
            return form

        def marked_form_content(
            span: DirectXObjectSpan,
            form: Stream,
            effective_resources: Dictionary | None,
            description: str,
        ) -> tuple[
            frozenset[tuple[str, int]],
            dict[tuple[str, int], str],
            dict[tuple[str, int], str],
            bool,
            bool,
            bool,
            bool,
            tuple[float, float, float, float] | None,
            tuple[_MarkedVectorScope, ...],
            frozenset[tuple[str, int]],
        ]:
            form_key = _object_key(form)
            if form_key is None:
                raise ConversionError(
                    "Cannot create semantic structure: Form XObject is direct"
                )
            resource_key: _ObjectKey | int = _object_key(effective_resources) or id(
                effective_resources
            )
            try:
                local_clip = (
                    _transform_polygon(
                        span.clip_polygon,
                        _inverse_matrix(span.matrix),
                    )
                    if span.clip_polygon is not None
                    else None
                )
            except ValueError as exc:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has a "
                    "singular invocation matrix"
                ) from exc
            cache_key = (
                form_key,
                resource_key,
                span.entry_state,
                local_clip,
            )
            marked_items = form_marked_items.get(cache_key)
            if marked_items is None:
                marked_items = _direct_artifact_items(
                    form,
                    description,
                    effective_resources,
                    optional_content=optional_content,
                    initial_state=span.entry_state,
                    initial_clip_polygon=local_clip,
                )
                form_marked_items[cache_key] = marked_items
            return marked_items

        def summarize_form(
            span: DirectXObjectSpan,
            form: Stream,
            inherited_resources: Dictionary | None,
            description: str,
            active_forms: frozenset[_ObjectKey] = frozenset(),
        ) -> _FormSemanticSummary:
            form_key = _object_key(form)
            if form_key is None or form_key in active_forms:
                raise ConversionError(
                    f"Cannot create semantic structure: recursive Form XObject in "
                    f"{description}"
                )
            cached_summary = form_summaries.get(id(span))
            if cached_summary is not None:
                return cached_summary
            effective_resources = effective_form_resources(
                form,
                inherited_resources,
                description,
            )
            (
                artifact_items,
                actual_text_items,
                alt_text_items,
                has_direct_vector,
                has_direct_artifact_paint,
                has_unclassified_direct_vector,
                has_ambiguous_direct_vector,
                unclassified_direct_vector_bbox,
                vector_scopes,
                hidden_items,
            ) = marked_form_content(span, form, effective_resources, description)
            if any(scope.mixed_paint for scope in vector_scopes):
                raise ConversionError(
                    "Cannot create semantic structure: Form marked text-evidence scope "
                    "mixes vector and other painting"
                )
            semantic_text: list[str] = []
            semantic_boxes = [
                visible_form_scope_bbox(span, scope) for scope in vector_scopes
            ]
            semantic_font_sizes: list[float] = []
            semantic_font_names: list[str] = []
            semantic_invisibility: list[bool] = []
            actual_texts = [
                scope.actual_text
                for scope in vector_scopes
                if scope.actual_text is not None
            ]
            alt_texts = [
                scope.alt_text for scope in vector_scopes if scope.alt_text is not None
            ]
            has_semantic_child = False
            semantic_leaf_count = len(vector_scopes)
            has_artifact_paint = has_direct_artifact_paint
            vector_review_required = False
            image_visibility_uncertain = False
            ambiguous_vector_paint = has_ambiguous_direct_vector
            nested_active = active_forms | frozenset({form_key})

            for child in span.children:
                if isinstance(child, DirectTextSpan):
                    child_source = "text"
                    child_index = child.direct_text_index
                else:
                    child_source = "xobject"
                    child_index = child.direct_xobject_index
                item = (child_source, child_index)
                if item in artifact_items:
                    has_artifact_paint = True
                    continue
                if item in hidden_items:
                    has_artifact_paint = True
                    continue
                child_bbox = visible_source_bbox(child)
                direct_actual_text = actual_text_items.get(item)
                direct_alt_text = alt_text_items.get(item)
                if child.final_paint_uncertain:
                    if (
                        isinstance(child, DirectXObjectSpan)
                        and child.intrinsic_visibility_uncertain
                        and not child.non_intrinsic_visibility_uncertain
                    ):
                        if (
                            direct_actual_text is not None
                            or direct_alt_text is not None
                        ):
                            raise ConversionError(
                                "Cannot create semantic structure: described image "
                                f"in {description} has uncertain intrinsic visibility"
                            )
                        image_visibility_uncertain = True
                    else:
                        raise ConversionError(
                            f"Cannot create semantic structure: {description} has "
                            "semantic painting with uncertain final-paint visibility"
                        )
                if child_bbox is None:
                    continue

                if isinstance(child, DirectTextSpan):
                    if not child.text.strip():
                        continue
                    has_semantic_child = True
                    semantic_leaf_count += 1
                    semantic_text.append(child.text)
                    semantic_boxes.append(child_bbox)
                    if child.font_size > 0:
                        semantic_font_sizes.append(child.font_size)
                    if child.font_name is not None:
                        semantic_font_names.append(child.font_name)
                    semantic_invisibility.append(child.invisible)
                    if direct_actual_text is not None:
                        actual_texts.append(direct_actual_text)
                    if direct_alt_text is not None:
                        alt_texts.append(direct_alt_text)
                    continue
                if child.kind != "form":
                    has_semantic_child = True
                    semantic_leaf_count += 1
                    semantic_boxes.append(child_bbox)
                    if direct_actual_text is not None:
                        actual_texts.append(direct_actual_text)
                    if direct_alt_text is not None:
                        alt_texts.append(direct_alt_text)
                    continue

                nested_form = resolve_form_child(
                    child,
                    effective_resources,
                    description,
                )
                nested_summary = summarize_form(
                    child,
                    nested_form,
                    effective_resources,
                    (f"Form XObject /{child.resource_name} invoked by {description}"),
                    nested_active,
                )
                vector_review_required = (
                    vector_review_required or nested_summary.vector_review_required
                )
                image_visibility_uncertain = (
                    image_visibility_uncertain
                    or nested_summary.image_visibility_uncertain
                )
                has_artifact_paint = (
                    has_artifact_paint or nested_summary.has_artifact_paint
                )
                ambiguous_vector_paint = (
                    ambiguous_vector_paint
                    or nested_summary.ambiguous_unclassified_vector_paint
                )
                if nested_summary.ambiguous_unclassified_vector_paint and (
                    direct_actual_text is not None or direct_alt_text is not None
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has a "
                        "described Form with ambiguous path geometry"
                    )
                if not nested_summary.has_semantic_paint:
                    continue
                has_semantic_child = True
                semantic_leaf_count += nested_summary.semantic_leaf_count
                semantic_boxes.append(nested_summary.bbox or child_bbox)
                if nested_summary.text:
                    semantic_text.append(nested_summary.text)
                    if nested_summary.font_size is not None:
                        semantic_font_sizes.append(nested_summary.font_size)
                    if nested_summary.font_name is not None:
                        semantic_font_names.append(nested_summary.font_name)
                    semantic_invisibility.append(nested_summary.invisible)
                nested_actual_text = direct_actual_text or nested_summary.actual_text
                if nested_actual_text is not None:
                    actual_texts.append(nested_actual_text)
                nested_alt_text = direct_alt_text or nested_summary.alt_text
                if nested_alt_text is not None:
                    alt_texts.append(nested_alt_text)

            distinct_actual_texts = tuple(dict.fromkeys(actual_texts))
            distinct_alt_texts = tuple(dict.fromkeys(alt_texts))
            unclassified_bbox = (
                visible_form_local_bbox(
                    span,
                    unclassified_direct_vector_bbox,
                    "Form vector painting",
                )
                if unclassified_direct_vector_bbox is not None
                else None
            )
            has_semantic_direct_vector = bool(vector_scopes) or (
                has_direct_vector and not ambiguous_vector_paint
            )
            has_unclassified_semantic_vector = (
                has_unclassified_direct_vector and not ambiguous_vector_paint
            )
            summary = _FormSemanticSummary(
                "".join(semantic_text),
                " ".join(distinct_actual_texts) or None,
                " ".join(distinct_alt_texts) or None,
                (
                    union_source_bboxes(semantic_boxes)
                    if semantic_boxes
                    else unclassified_bbox
                ),
                (
                    statistics.median(semantic_font_sizes)
                    if semantic_font_sizes
                    else None
                ),
                (
                    semantic_font_names[0]
                    if semantic_font_names and len(set(semantic_font_names)) == 1
                    else None
                ),
                bool(semantic_invisibility) and all(semantic_invisibility),
                image_visibility_uncertain,
                has_semantic_direct_vector or has_semantic_child,
                has_artifact_paint,
                has_unclassified_semantic_vector,
                ambiguous_vector_paint,
                len(vector_scopes),
                semantic_leaf_count or int(has_unclassified_semantic_vector),
                has_unclassified_semantic_vector
                and not has_semantic_child
                and not vector_scopes,
                vector_review_required
                or ambiguous_vector_paint
                or (
                    has_unclassified_semantic_vector
                    and (has_semantic_child or bool(vector_scopes))
                ),
            )
            form_summaries[id(span)] = summary
            return summary

        def add_span(
            span: DirectTextSpan | DirectXObjectSpan,
            *,
            container: Dictionary | Stream,
            stream: Stream | None,
            source: str,
            source_index: int,
            span_id: str,
            source_artifact: bool,
            generated_artifact: bool = False,
            source_actual_text: str | None,
            source_alt_text: str | None = None,
            source_bbox: tuple[float, float, float, float] | None = None,
            text_override: str | None = None,
            kind_override=None,
            style_override: _FormSemanticSummary | None = None,
        ) -> bool:
            if (
                span.final_paint_uncertain
                and not source_artifact
                and not generated_artifact
            ):
                if (
                    isinstance(span, DirectXObjectSpan)
                    and span.intrinsic_visibility_uncertain
                    and not span.non_intrinsic_visibility_uncertain
                ):
                    if source_actual_text is not None or source_alt_text is not None:
                        raise ConversionError(
                            "Cannot create semantic structure: described image on "
                            f"page {page_number} has uncertain intrinsic visibility"
                        )
                else:
                    raise ConversionError(
                        f"Cannot create semantic structure: page {page_number} has "
                        "semantic painting with uncertain final-paint visibility"
                    )
            if (
                style_override is not None
                and style_override.image_visibility_uncertain
                and (source_actual_text is not None or source_alt_text is not None)
            ):
                raise ConversionError(
                    "Cannot create semantic structure: described Form on page "
                    f"{page_number} contains an image with uncertain intrinsic "
                    "visibility"
                )
            clipped_source_bbox = (
                source_bbox if source_bbox is not None else visible_source_bbox(span)
            )
            if clipped_source_bbox is None:
                return False
            visual_bbox = geometry.pdfminer_to_visual_bbox(clipped_source_bbox)
            if visual_bbox is None:
                return False
            box = BoundingBox(*visual_bbox)
            if isinstance(span, DirectTextSpan):
                text = span.text if text_override is None else text_override
                font_size = span.font_size if span.font_size > 0 else None
                font_name = span.font_name
                kind = kind_override or SpanKind.TEXT
                invisible = span.invisible
            else:
                text = span.text if text_override is None else text_override
                if style_override is not None:
                    font_size = style_override.font_size
                    font_name = style_override.font_name
                    invisible = style_override.invisible
                else:
                    text_runs = [run for run in span.text_runs if run.text]
                    font_size = (
                        statistics.median(run.font_size for run in text_runs)
                        if text_runs
                        else None
                    )
                    font_name = text_runs[0].font_name if text_runs else None
                    invisible = bool(text_runs) and all(
                        run.invisible for run in text_runs
                    )
                kind = kind_override or (
                    SpanKind.TEXT if text.strip() else SpanKind.IMAGE
                )
            bindings[span_id] = _SemanticBinding(
                page_number,
                container,
                stream,
                source,
                source_index,
                (box.left, box.top, box.right, box.bottom),
            )
            if generated_artifact:
                optional_artifacts[span_id] = ArtifactReference(
                    span_id,
                    page_number,
                    ArtifactKind.LAYOUT,
                    "Layout",
                    None,
                )
                return True
            if source_artifact:
                source_artifact_ids.add(span_id)
                return True
            if source_actual_text is not None:
                source_actual_texts[span_id] = source_actual_text
            if source_alt_text is not None:
                source_alt_texts[span_id] = source_alt_text
            page_spans.append(
                SemanticSpan(
                    span_id,
                    text,
                    box,
                    font_size=font_size,
                    font_name=font_name,
                    kind=kind,
                    invisible=invisible,
                )
            )
            return True

        def add_vector_scope(
            scope: _MarkedVectorScope,
            *,
            container: Dictionary | Stream,
            stream: Stream | None,
            source: str,
            span_id: str,
            source_bbox: tuple[float, float, float, float] | None = None,
        ) -> None:
            visual_bbox = (
                geometry.pdfminer_to_visual_bbox(source_bbox)
                if source_bbox is not None
                else geometry.default_to_visual_bbox(scope.bbox)
            )
            if visual_bbox is None:
                raise ConversionError(
                    "Cannot create semantic structure: ActualText vector scope "
                    "is outside the visible page"
                )
            box = BoundingBox(*visual_bbox)
            bindings[span_id] = _SemanticBinding(
                page_number,
                container,
                stream,
                source,
                scope.marked_content_index,
                (box.left, box.top, box.right, box.bottom),
            )
            if scope.actual_text is not None:
                source_actual_texts[span_id] = scope.actual_text
            if scope.alt_text is not None:
                source_alt_texts[span_id] = scope.alt_text
            page_spans.append(
                SemanticSpan(
                    span_id,
                    "",
                    box,
                    kind=SpanKind.IMAGE,
                )
            )

        def build_form_invocation(
            span: DirectXObjectSpan,
            form: Stream,
            inherited_resources: Dictionary | None,
            description: str,
            *,
            invocation_index: int,
            resource_name: Name,
            source_prefix: str,
            invocation_actual_text: str | None = None,
            invocation_alt_text: str | None = None,
            active_forms: frozenset[_ObjectKey] = frozenset(),
            summary: _FormSemanticSummary | None = None,
        ) -> _SemanticFormInvocation | None:
            form_key = _object_key(form)
            if form_key is None or form_key in active_forms:
                raise ConversionError(
                    f"Cannot create semantic structure: recursive Form XObject in "
                    f"{description}"
                )
            effective_resources = effective_form_resources(
                form,
                inherited_resources,
                description,
            )
            (
                artifact_items,
                form_actual_texts,
                form_alt_texts,
                _has_direct_vector,
                _has_direct_artifact_paint,
                _has_unclassified_vector,
                _has_ambiguous_vector,
                _unclassified_vector_bbox,
                vector_scopes,
                hidden_items,
            ) = marked_form_content(span, form, effective_resources, description)
            if summary is None:
                summary = summarize_form(
                    span,
                    form,
                    inherited_resources,
                    description,
                    active_forms,
                )
            if summary.vector_review_required:
                vector_review_pages.add(layout.page_index)
            if summary.semantic_leaf_count > 1 and (
                invocation_actual_text is not None or invocation_alt_text is not None
            ):
                raise ConversionError(
                    "Cannot create semantic structure: a Form-level alternative "
                    "covers multiple semantic leaves"
                )

            invocation_span_ids: set[str] = set()
            expected_xobjects: dict[int, tuple[str, str | None]] = {}
            child_invocations: list[_SemanticFormInvocation] = []
            nested_active = active_forms | frozenset({form_key})
            fallback_actual_text = (
                invocation_actual_text if summary.semantic_leaf_count == 1 else None
            )
            fallback_alt_text = (
                invocation_alt_text if summary.semantic_leaf_count == 1 else None
            )

            for child in span.children:
                if isinstance(child, DirectTextSpan):
                    child_source = "text"
                    child_index = child.direct_text_index
                else:
                    child_source = "xobject"
                    child_index = child.direct_xobject_index
                    expected_xobjects[child_index] = (
                        child.kind,
                        child.resource_name,
                    )
                child_item = (child_source, child_index)
                child_binding_source = f"{source_prefix}{child_source}"
                child_span_id = _semantic_span_id(
                    page_number,
                    child_binding_source,
                    child_index,
                )
                child_source_artifact = child_item in artifact_items
                child_hidden = child_item in hidden_items
                child_actual_text = form_actual_texts.get(child_item)
                child_alt_text = form_alt_texts.get(child_item)

                if (
                    isinstance(child, DirectXObjectSpan)
                    and child.kind == "form"
                    and not child_source_artifact
                    and not child_hidden
                ):
                    nested_form = resolve_form_child(
                        child,
                        effective_resources,
                        description,
                    )
                    nested_description = (
                        f"Form XObject /{child.resource_name} invoked by {description}"
                    )
                    nested_summary = summarize_form(
                        child,
                        nested_form,
                        effective_resources,
                        nested_description,
                        nested_active,
                    )
                    if not nested_summary.has_semantic_paint:
                        continue
                    if nested_summary.bind_as_figure:
                        if nested_summary.has_artifact_paint:
                            raise ConversionError(
                                "Cannot create semantic structure: mixed Artifact "
                                f"and semantic painting in {nested_description} "
                                "cannot be bound as one content item"
                            )
                        if add_span(
                            child,
                            container=form,
                            stream=form,
                            source=child_binding_source,
                            source_index=child_index,
                            span_id=child_span_id,
                            source_artifact=False,
                            source_actual_text=(
                                child_actual_text
                                or fallback_actual_text
                                or nested_summary.actual_text
                            ),
                            source_alt_text=(
                                child_alt_text
                                or fallback_alt_text
                                or nested_summary.alt_text
                            ),
                            source_bbox=nested_summary.bbox,
                            text_override="",
                            kind_override=SpanKind.IMAGE,
                            style_override=nested_summary,
                        ):
                            invocation_span_ids.add(child_span_id)
                        continue
                    nested_invocation = build_form_invocation(
                        child,
                        nested_form,
                        effective_resources,
                        nested_description,
                        invocation_index=child_index,
                        resource_name=Name(f"/{child.resource_name}"),
                        source_prefix=(f"{source_prefix}xobject-{child_index}-"),
                        invocation_actual_text=(
                            child_actual_text or fallback_actual_text
                        ),
                        invocation_alt_text=(child_alt_text or fallback_alt_text),
                        active_forms=nested_active,
                        summary=nested_summary,
                    )
                    if nested_invocation is not None:
                        child_invocations.append(nested_invocation)
                    continue

                if add_span(
                    child,
                    container=form,
                    stream=form,
                    source=child_binding_source,
                    source_index=child_index,
                    span_id=child_span_id,
                    source_artifact=child_source_artifact,
                    generated_artifact=child_hidden,
                    source_actual_text=(child_actual_text or fallback_actual_text),
                    source_alt_text=(child_alt_text or fallback_alt_text),
                ):
                    invocation_span_ids.add(child_span_id)

            for scope in vector_scopes:
                scope_source = f"{source_prefix}marked"
                scope_span_id = _semantic_span_id(
                    page_number,
                    scope_source,
                    scope.marked_content_index,
                )
                add_vector_scope(
                    scope,
                    container=form,
                    stream=form,
                    source=scope_source,
                    span_id=scope_span_id,
                    source_bbox=visible_form_scope_bbox(span, scope),
                )
                invocation_span_ids.add(scope_span_id)

            if not invocation_span_ids and not child_invocations:
                return None
            return _SemanticFormInvocation(
                page_number,
                invocation_index,
                resource_name,
                form,
                frozenset(invocation_span_ids),
                expected_xobjects,
                source_prefix,
                tuple(child_invocations),
            )

        for scope in page_vector_scopes:
            add_vector_scope(
                scope,
                container=page.obj,
                stream=None,
                source="marked",
                span_id=_semantic_span_id(
                    page_number,
                    "marked",
                    scope.marked_content_index,
                ),
            )
        if page_vector_scopes:
            native_reading_pages.add(layout.page_index)

        for span in layout.spans:
            if isinstance(span, DirectTextSpan):
                source = "text"
                source_index = span.direct_text_index
            else:
                source = "xobject"
                source_index = span.direct_xobject_index
            span_id = _semantic_span_id(page_number, source, source_index)
            source_artifact = (source, source_index) in page_artifacts
            source_hidden = (source, source_index) in page_hidden_items
            ignored_xobject = ignored_page_xobjects.get(layout.page_index)
            if (
                isinstance(span, DirectXObjectSpan)
                and span.kind == "form"
                and ignored_xobject is not None
                and span.resource_name == str(ignored_xobject[0]).removeprefix("/")
            ):
                if source_hidden:
                    raise ConversionError(
                        f"Cannot create semantic structure: OCR Form on page "
                        f"{page_number} is hidden by optional content"
                    )
                form = resolve_form_child(span, resources, f"page {page_number}")
                if not _same_object(form, ignored_xobject[1]):
                    raise ConversionError(
                        f"Cannot create semantic structure: OCR Form on page "
                        f"{page_number} no longer matches its resource"
                    )
                ignored_xobject_invocations += 1
                continue
            spans_before = len(page_spans)
            if (
                not isinstance(span, DirectXObjectSpan)
                or span.kind != "form"
                or source_artifact
                or source_hidden
            ):
                add_span(
                    span,
                    container=page.obj,
                    stream=None,
                    source=source,
                    source_index=source_index,
                    span_id=span_id,
                    source_artifact=source_artifact,
                    generated_artifact=source_hidden,
                    source_actual_text=page_actual_texts.get((source, source_index)),
                    source_alt_text=page_alt_texts.get((source, source_index)),
                )
                if len(page_spans) > spans_before and (
                    isinstance(span, DirectTextSpan)
                    or (
                        isinstance(span, DirectXObjectSpan)
                        and (
                            span.kind == "form"
                            or source_actual_texts.get(span_id) is not None
                            or source_alt_texts.get(span_id) is not None
                            or (
                                page_spans[-1].bbox.width * page_spans[-1].bbox.height
                                < geometry.visual_size[0]
                                * geometry.visual_size[1]
                                * 0.72
                            )
                        )
                    )
                ):
                    native_reading_pages.add(layout.page_index)
                continue

            if visible_source_bbox(span) is None:
                continue
            form = resolve_form_child(span, resources, f"page {page_number}")
            assert span.resource_name is not None
            resource_name = Name(f"/{span.resource_name}")
            form_description = f"Form XObject {resource_name} on page {page_number}"
            effective_resources = effective_form_resources(
                form,
                resources,
                form_description,
            )
            (
                artifact_items,
                _form_actual_texts,
                _form_alt_texts,
                _has_direct_vector,
                _has_direct_artifact_paint,
                _has_unclassified_form_vector,
                _has_ambiguous_form_vector,
                _unclassified_form_vector_bbox,
                _form_vector_scopes,
                _form_hidden_items,
            ) = marked_form_content(
                span,
                form,
                effective_resources,
                form_description,
            )
            summary = summarize_form(
                span,
                form,
                resources,
                form_description,
            )
            if summary.vector_review_required:
                vector_review_pages.add(layout.page_index)
            invocation_actual_text = page_actual_texts.get((source, source_index))
            invocation_alt_text = page_alt_texts.get((source, source_index))
            if summary.ambiguous_unclassified_vector_paint and (
                invocation_actual_text is not None or invocation_alt_text is not None
            ):
                raise ConversionError(
                    f"Cannot create semantic structure: {form_description} is "
                    "described but has ambiguous path geometry"
                )

            if summary.bind_as_figure:
                if summary.has_artifact_paint:
                    raise ConversionError(
                        "Cannot create semantic structure: mixed Artifact and "
                        f"semantic painting in {form_description} cannot be bound "
                        "as one content item"
                    )
                add_span(
                    span,
                    container=page.obj,
                    stream=None,
                    source=source,
                    source_index=source_index,
                    span_id=span_id,
                    source_artifact=False,
                    source_actual_text=(invocation_actual_text or summary.actual_text),
                    source_alt_text=invocation_alt_text or summary.alt_text,
                    source_bbox=summary.bbox,
                    text_override="",
                    kind_override=SpanKind.IMAGE,
                    style_override=summary,
                )
                if len(page_spans) > spans_before:
                    native_reading_pages.add(layout.page_index)
                continue

            if not summary.has_semantic_paint and not artifact_items:
                continue
            invocation = build_form_invocation(
                span,
                form,
                resources,
                form_description,
                invocation_index=span.direct_xobject_index,
                resource_name=resource_name,
                source_prefix=f"form-{span.direct_xobject_index}-",
                invocation_actual_text=invocation_actual_text,
                invocation_alt_text=invocation_alt_text,
                summary=summary,
            )
            if invocation is not None:
                native_reading_pages.add(layout.page_index)
                page_form_invocations.append(invocation)
        if ignored_xobject_invocations != int(
            layout.page_index in ignored_page_xobjects
        ):
            raise ConversionError(
                f"Cannot create semantic structure: page {page_number} invokes its "
                f"OCR Form {ignored_xobject_invocations} times"
            )
        spans_by_page[layout.page_index] = tuple(page_spans)
        form_invocations[layout.page_index] = tuple(page_form_invocations)
    return (
        spans_by_page,
        bindings,
        dimensions,
        frozenset(source_artifact_ids),
        form_invocations,
        source_actual_texts,
        source_alt_texts,
        frozenset(vector_review_pages),
        frozenset(native_reading_pages),
        optional_artifacts,
    )


def _existing_structure_references_hidden_optional_content(
    pdf: pikepdf.Pdf,
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
    object_owners: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[Dictionary | Stream, _ObjectKey, Dictionary],
    ],
    optional_content: _DefaultOCVisibility,
) -> bool:
    """Return whether existing MCRs or OBJRs resolve to hidden painting."""
    hidden_reference = False
    for referenced_object, _owner_key, _page in object_owners.values():
        if "/OC" in referenced_object and not _optional_content_is_visible(
            optional_content,
            referenced_object.get("/OC"),
            "existing structure object reference",
        ):
            hidden_reference = True

    referenced_mcids = frozenset(content_references)
    visited_forms: set[tuple[_ObjectKey, _ObjectKey | int, bool, bool]] = set()

    def inspect_container(
        owner: pikepdf.Page | Stream,
        inherited_resources: Dictionary | None,
        description: str,
        *,
        inherited_hidden: bool = False,
        referenced_ancestor_mcr: bool = False,
        active_forms: frozenset[_ObjectKey] = frozenset(),
    ) -> bool:
        try:
            instructions = pikepdf.parse_content_stream(owner)
        except Exception as exc:
            raise ConversionError(
                f"Cannot inspect optional content: {description} cannot be parsed"
            ) from exc

        container = owner.obj if isinstance(owner, pikepdf.Page) else owner
        container_key = _object_key(container)
        if container_key is None:
            raise ConversionError(
                "Cannot inspect optional content: direct content container"
            )
        own_resources = (
            _page_resources(owner)
            if isinstance(owner, pikepdf.Page)
            else resolve_indirect(owner.get("/Resources"))
        )
        resources = (
            own_resources
            if isinstance(own_resources, Dictionary) and own_resources
            else inherited_resources
        )
        resource_key: _ObjectKey | int = _object_key(resources) or id(resources)
        if isinstance(owner, Stream):
            visit_key = (
                container_key,
                resource_key,
                inherited_hidden,
                referenced_ancestor_mcr,
            )
            if visit_key in visited_forms:
                return False
            visited_forms.add(visit_key)

        nesting: list[tuple[bool, int | None]] = []
        hidden_reference = False

        def scope_hidden() -> bool:
            return inherited_hidden or any(hidden for hidden, _mcid in nesting)

        def has_referenced_scope() -> bool:
            return referenced_ancestor_mcr or any(
                (container_key, mcid) in referenced_mcids
                for _hidden, mcid in nesting
                if mcid is not None
            )

        for instruction in instructions:
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                if scope_hidden() and has_referenced_scope():
                    hidden_reference = True
                continue
            operator = instruction.operator
            operands = instruction.operands
            if operator in {_BMC, _BDC}:
                tag = resolve_indirect(operands[0]) if operands else None
                if not isinstance(tag, Name):
                    raise ConversionError(
                        f"Cannot inspect optional content: {description} has "
                        "malformed marked content"
                    )
                properties = (
                    _marked_content_properties(list(operands), resources)
                    if operator == _BDC
                    else None
                )
                hidden = False
                if tag == Name.OC:
                    if operator != _BDC or properties is None:
                        raise ConversionError(
                            f"Cannot inspect optional content: {description} has "
                            "malformed optional-content marked content"
                        )
                    hidden = not _optional_content_is_visible(
                        optional_content,
                        properties,
                        f"{description} marked content",
                    )
                mcid = resolve_indirect(properties.get("/MCID")) if properties else None
                if not isinstance(mcid, int) or isinstance(mcid, bool) or mcid < 0:
                    mcid = None
                nesting.append((hidden, mcid))
                continue
            if operator == _EMC:
                if not nesting:
                    raise ConversionError(
                        f"Cannot inspect optional content: {description} has "
                        "unbalanced marked content"
                    )
                nesting.pop()
                continue

            operator_name = str(operator)
            if operator_name == "Do":
                if len(operands) != 1:
                    raise ConversionError(
                        f"Cannot inspect optional content: {description} has "
                        "malformed XObject provenance"
                    )
                resource_name = resolve_indirect(operands[0])
                xobjects = (
                    resolve_indirect(resources.get("/XObject"))
                    if isinstance(resources, Dictionary)
                    else None
                )
                xobject = (
                    resolve_indirect(xobjects.get(resource_name))
                    if isinstance(resource_name, Name)
                    and isinstance(xobjects, Dictionary)
                    else None
                )
                xobject_hidden = bool(
                    isinstance(xobject, Stream)
                    and "/OC" in xobject
                    and not _optional_content_is_visible(
                        optional_content,
                        xobject.get("/OC"),
                        f"{description} XObject {resource_name}",
                    )
                )
                invocation_hidden = scope_hidden() or xobject_hidden
                referenced_scope = has_referenced_scope()
                is_form = isinstance(xobject, Stream) and (
                    resolve_indirect(xobject.get("/Subtype")) == Name.Form
                )
                if not is_form:
                    if invocation_hidden and referenced_scope:
                        hidden_reference = True
                    continue
                assert isinstance(xobject, Stream)
                form_key = _object_key(xobject)
                if form_key is None or form_key in active_forms:
                    raise ConversionError(
                        "Cannot inspect optional content: recursive Form XObject"
                    )
                form_hidden_reference = inspect_container(
                    xobject,
                    resources,
                    f"Form XObject {resource_name} invoked by {description}",
                    inherited_hidden=invocation_hidden,
                    referenced_ancestor_mcr=referenced_scope,
                    active_forms=active_forms | frozenset({form_key}),
                )
                hidden_reference = form_hidden_reference or hidden_reference
                continue
            if (
                operator_name in _PAINTING_OPERATORS
                and scope_hidden()
                and has_referenced_scope()
            ):
                hidden_reference = True

        if nesting:
            raise ConversionError(
                f"Cannot inspect optional content: {description} has unbalanced "
                "marked content"
            )
        return hidden_reference

    for page_number, page in enumerate(pdf.pages, start=1):
        annotations = resolve_indirect(page.obj.get("/Annots"))
        if isinstance(annotations, Array):
            page_key = _object_key(page.obj)
            for value in annotations:
                annotation = resolve_indirect(value)
                if not isinstance(annotation, Dictionary) or "/OC" not in annotation:
                    continue
                visible = _optional_content_is_visible(
                    optional_content,
                    annotation.get("/OC"),
                    f"page {page_number} annotation",
                )
                annotation_key = _object_key(annotation)
                if (
                    not visible
                    and annotation_key is not None
                    and page_key is not None
                    and (annotation_key, page_key) in object_owners
                ):
                    hidden_reference = True
        page_hidden_reference = inspect_container(
            page,
            _page_resources(page),
            f"page {page_number}",
        )
        hidden_reference = page_hidden_reference or hidden_reference
    return hidden_reference


def _requires_existing_image_visibility_rebuild(
    pdf: pikepdf.Pdf,
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
) -> tuple[bool, bool]:
    """Return rebuild and described-uncertainty results for tagged images."""
    from .digital_layout import (
        DirectTextSpan,
        DirectXObjectSpan,
        extract_digital_layout,
    )

    referenced_page_keys = {
        _object_key(page)
        for _container, _owner, page, _stream in content_references.values()
    }
    page_indices = frozenset(
        page_index
        for page_index, page in enumerate(pdf.pages)
        if _object_key(page.obj) in referenced_page_keys
    )
    rebuild = False
    described_uncertainty = False

    def structure_description(owner: Dictionary) -> bool:
        current: object = owner
        visited: set[_ObjectKey] = set()
        while isinstance(current := resolve_indirect(current), Dictionary):
            current_key = _object_key(current)
            if current_key is None or current_key in visited:
                return False
            visited.add(current_key)
            if resolve_indirect(current.get("/Type")) == Name.StructTreeRoot:
                return False
            if _valid_structure_text(
                current.get("/ActualText")
            ) or _valid_structure_text(current.get("/Alt")):
                return True
            current = current.get("/P")
        return False

    def image_state(
        span: DirectTextSpan | DirectXObjectSpan,
    ) -> tuple[bool, bool]:
        if isinstance(span, DirectTextSpan):
            return False, False
        if span.kind in {"image", "inline_image"}:
            return span.invisible, span.intrinsic_visibility_uncertain
        invisible = False
        uncertain = False
        for child in span.children:
            child_invisible, child_uncertain = image_state(child)
            invisible = invisible or child_invisible
            uncertain = uncertain or child_uncertain
        return invisible, uncertain

    def inspect_container(
        owner: pikepdf.Page | Stream,
        spans: tuple[DirectTextSpan | DirectXObjectSpan, ...],
        resources: Dictionary | None,
        description: str,
        active_forms: frozenset[_ObjectKey] = frozenset(),
    ) -> tuple[bool, bool]:
        nonlocal described_uncertainty, rebuild
        container = owner.obj if isinstance(owner, pikepdf.Page) else owner
        container_key = _object_key(container)
        if container_key is None:
            raise ConversionError(
                "Cannot inspect existing image visibility: direct content container"
            )
        mcid_items: dict[tuple[str, int], tuple[int, ...]] = {}
        actual_text_items: dict[tuple[str, int], str] = {}
        alt_text_items: dict[tuple[str, int], str] = {}
        (
            artifact_items,
            actual_text_items,
            alt_text_items,
            *_paint,
        ) = _direct_artifact_items(
            owner,
            description,
            resources,
            mcid_items_out=mcid_items,
        )
        effective_resources = (
            _page_resources(owner)
            if isinstance(owner, pikepdf.Page)
            else resolve_indirect(owner.get("/Resources"))
        )
        if not isinstance(effective_resources, Dictionary) or not effective_resources:
            effective_resources = resources

        container_invisible = False
        container_uncertain = False
        for span in spans:
            if isinstance(span, DirectTextSpan):
                item = ("text", span.direct_text_index)
            else:
                item = ("xobject", span.direct_xobject_index)
            if item in artifact_items:
                continue
            if isinstance(span, DirectXObjectSpan) and span.kind == "form":
                if span.resource_name is None or not isinstance(
                    effective_resources, Dictionary
                ):
                    raise ConversionError(
                        "Cannot inspect existing image visibility: Form provenance "
                        "is invalid"
                    )
                xobjects = resolve_indirect(effective_resources.get("/XObject"))
                form = (
                    resolve_indirect(xobjects.get(Name(f"/{span.resource_name}")))
                    if isinstance(xobjects, Dictionary)
                    else None
                )
                form_key = _object_key(form)
                if (
                    not isinstance(form, Stream)
                    or resolve_indirect(form.get("/Subtype")) != Name.Form
                    or form_key is None
                    or form_key in active_forms
                ):
                    raise ConversionError(
                        "Cannot inspect existing image visibility: Form provenance "
                        "is invalid"
                    )
                invisible, uncertain = inspect_container(
                    form,
                    span.children,
                    effective_resources,
                    f"Form XObject /{span.resource_name} invoked by {description}",
                    active_forms | frozenset({form_key}),
                )
            else:
                invisible, uncertain = image_state(span)
            container_invisible = container_invisible or invisible
            container_uncertain = container_uncertain or uncertain
            for mcid in mcid_items.get(item, ()):
                reference = (container_key, mcid)
                existing = content_references.get(reference)
                if existing is None:
                    continue
                described = (
                    item in actual_text_items
                    or item in alt_text_items
                    or structure_description(existing[1])
                )
                if uncertain and described:
                    described_uncertainty = True
                rebuild = rebuild or invisible
        return container_invisible, container_uncertain

    for layout in extract_digital_layout(pdf, page_indices=page_indices):
        page = pdf.pages[layout.page_index]
        inspect_container(
            page,
            layout.spans,
            _page_resources(page),
            f"page {layout.page_index + 1}",
        )
    return rebuild, described_uncertainty


def _has_unambiguous_existing_reading_order_inversion(
    pdf: pikepdf.Pdf,
    root: Dictionary,
    elements: list[Dictionary],
    content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ],
) -> bool:
    """Return whether visible text proves a single-column sibling inversion."""
    from .digital_layout import (
        DirectTextSpan,
        DirectXObjectSpan,
        extract_digital_layout,
    )

    def direct_structure_children(value: object) -> tuple[list[Dictionary], bool]:
        value = resolve_indirect(value)
        if isinstance(value, Array):
            children: list[Dictionary] = []
            for item in value:
                nested, complete = direct_structure_children(item)
                if not complete:
                    return [], False
                children.extend(nested)
            return children, True
        if (
            isinstance(value, Dictionary)
            and resolve_indirect(value.get("/Type")) == Name.StructElem
        ):
            return [value], True
        return [], value is None

    page_numbers = {
        _object_key(page.obj): page_number
        for page_number, page in enumerate(pdf.pages, start=1)
    }
    element_pages: dict[_ObjectKey, set[int]] = {}
    element_references: dict[_ObjectKey, set[tuple[_ObjectKey, int]]] = {}
    for reference, (
        _container,
        owner,
        page,
        _stream_owner,
    ) in content_references.items():
        page_number = page_numbers.get(_object_key(page))
        if page_number is None:
            continue
        current: object = owner
        visited: set[_ObjectKey] = set()
        while isinstance(current := resolve_indirect(current), Dictionary):
            if resolve_indirect(current.get("/Type")) != Name.StructElem:
                break
            current_key = _object_key(current)
            if current_key is None or current_key in visited:
                break
            visited.add(current_key)
            element_pages.setdefault(current_key, set()).add(page_number)
            element_references.setdefault(current_key, set()).add(reference)
            current = current.get("/P")

    role_map = resolve_indirect(root.get("/RoleMap"))
    if not isinstance(role_map, Dictionary):
        role_map = None
    block_roles = {
        "/Document",
        "/Part",
        "/Art",
        "/Sect",
        "/Div",
        "/BlockQuote",
        "/Caption",
        "/TOC",
        "/TOCI",
        "/Index",
        "/H",
        "/H1",
        "/H2",
        "/H3",
        "/H4",
        "/H5",
        "/H6",
        "/P",
        "/L",
        "/LI",
        "/Table",
        "/THead",
        "/TBody",
        "/TFoot",
        "/TR",
        "/Figure",
        "/Formula",
        "/Form",
    }
    candidates: list[tuple[tuple[_ObjectKey, ...], int]] = []
    relevant_references: set[tuple[_ObjectKey, int]] = set()
    for parent in [root, *elements]:
        children, complete = direct_structure_children(parent.get("/K"))
        if not complete or len(children) < 2:
            continue
        child_keys = tuple(_object_key(child) for child in children)
        if any(key is None for key in child_keys) or any(
            _effective_structure_role(child.get("/S"), role_map) not in block_roles
            for child in children
        ):
            continue
        valid_child_keys = tuple(key for key in child_keys if key is not None)
        pages = [element_pages.get(key, set()) for key in valid_child_keys]
        if any(len(child_pages) != 1 for child_pages in pages):
            continue
        page_number = next(iter(pages[0]))
        if any(next(iter(child_pages)) != page_number for child_pages in pages[1:]):
            continue
        candidates.append((valid_child_keys, page_number))
        for child_key in valid_child_keys:
            relevant_references.update(element_references.get(child_key, set()))
    if not candidates:
        return False

    relevant_container_keys = {
        container_key for container_key, _mcid in relevant_references
    }
    layouts = extract_digital_layout(
        pdf,
        page_indices=frozenset(page_number - 1 for _keys, page_number in candidates),
    )
    reference_boxes: dict[
        tuple[_ObjectKey, int], list[tuple[float, float, float, float]]
    ] = {}

    def union_boxes(
        boxes: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float]:
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    def visible_text_bbox(
        span: DirectTextSpan | DirectXObjectSpan,
        geometry: _PageGeometry,
    ) -> tuple[float, float, float, float] | None:
        if isinstance(span, DirectTextSpan):
            if span.invisible or span.render_mode == 7 or not span.text.strip():
                return None
            source_bbox = (
                _bbox_intersection(span.bbox, span.clip_bbox)
                if span.clip_bbox is not None
                else span.bbox
            )
            return (
                geometry.pdfminer_to_visual_bbox(source_bbox)
                if source_bbox is not None
                else None
            )
        child_boxes = [
            box
            for child in span.children
            if (box := visible_text_bbox(child, geometry)) is not None
        ]
        return union_boxes(child_boxes) if child_boxes else None

    def inspect_container(
        owner: pikepdf.Page | Stream,
        spans: tuple[DirectTextSpan | DirectXObjectSpan, ...],
        resources: Dictionary | None,
        geometry: _PageGeometry,
        page: pikepdf.Page,
        description: str,
        active_forms: frozenset[_ObjectKey] = frozenset(),
    ) -> None:
        container = owner.obj if isinstance(owner, pikepdf.Page) else owner
        container_key = _object_key(container)
        if container_key is None:
            raise ConversionError(
                "Cannot inspect existing reading order: direct content container"
            )
        mcid_items: dict[tuple[str, int], tuple[int, ...]] = {}
        if container_key in relevant_container_keys:
            _direct_artifact_items(
                owner,
                description,
                resources,
                mcid_items_out=mcid_items,
            )
        effective_resources = (
            _page_resources(owner)
            if isinstance(owner, pikepdf.Page)
            else resolve_indirect(owner.get("/Resources"))
        )
        if not isinstance(effective_resources, Dictionary) or not effective_resources:
            effective_resources = resources

        for span in spans:
            if isinstance(span, DirectTextSpan):
                item = ("text", span.direct_text_index)
            else:
                item = ("xobject", span.direct_xobject_index)
            bbox = visible_text_bbox(span, geometry)
            if bbox is not None:
                for mcid in mcid_items.get(item, ()):
                    reference = (container_key, mcid)
                    existing = content_references.get(reference)
                    if (
                        reference in relevant_references
                        and existing is not None
                        and _same_object(existing[2], page.obj)
                    ):
                        reference_boxes.setdefault(reference, []).append(bbox)

            if (
                not isinstance(span, DirectXObjectSpan)
                or span.kind != "form"
                or span.resource_name is None
                or not isinstance(effective_resources, Dictionary)
            ):
                continue
            xobjects = resolve_indirect(effective_resources.get("/XObject"))
            form = (
                resolve_indirect(xobjects.get(Name(f"/{span.resource_name}")))
                if isinstance(xobjects, Dictionary)
                else None
            )
            form_key = _object_key(form)
            if (
                not isinstance(form, Stream)
                or resolve_indirect(form.get("/Subtype")) != Name.Form
                or form_key is None
                or form_key in active_forms
            ):
                if form_key in relevant_container_keys:
                    raise ConversionError(
                        "Cannot inspect existing reading order: Form provenance is "
                        "invalid"
                    )
                continue
            inspect_container(
                form,
                span.children,
                effective_resources,
                geometry,
                page,
                f"Form XObject /{span.resource_name} invoked by {description}",
                active_forms | frozenset({form_key}),
            )

    for layout in layouts:
        page = pdf.pages[layout.page_index]
        inspect_container(
            page,
            layout.spans,
            _page_resources(page),
            _page_geometry(page),
            page,
            f"page {layout.page_index + 1}",
        )

    element_boxes: dict[
        tuple[_ObjectKey, int], list[tuple[float, float, float, float]]
    ] = {}
    for reference, boxes in reference_boxes.items():
        if not boxes:
            continue
        _container, owner, page, _stream_owner = content_references[reference]
        page_number = page_numbers.get(_object_key(page))
        if page_number is None:
            continue
        box = union_boxes(boxes)
        current: object = owner
        visited: set[_ObjectKey] = set()
        while isinstance(current := resolve_indirect(current), Dictionary):
            if resolve_indirect(current.get("/Type")) != Name.StructElem:
                break
            current_key = _object_key(current)
            if current_key is None or current_key in visited:
                break
            visited.add(current_key)
            element_boxes.setdefault((current_key, page_number), []).append(box)
            current = current.get("/P")

    for child_keys, page_number in candidates:
        boxes = [
            union_boxes(element_boxes[(key, page_number)])
            for key in child_keys
            if (key, page_number) in element_boxes
        ]
        if len(boxes) != len(child_keys):
            continue
        same_column = True
        for left_index, left in enumerate(boxes):
            for right in boxes[left_index + 1 :]:
                horizontal_overlap = max(
                    0.0, min(left[2], right[2]) - max(left[0], right[0])
                )
                if (
                    horizontal_overlap / min(left[2] - left[0], right[2] - right[0])
                    < 0.5
                ):
                    same_column = False
                    break
            if not same_column:
                break
        if not same_column:
            continue
        for earlier_index, earlier in enumerate(boxes):
            for later in boxes[earlier_index + 1 :]:
                tolerance = max(
                    2.0, min(earlier[3] - earlier[1], later[3] - later[1]) * 0.2
                )
                if earlier[1] >= later[3] + tolerance:
                    return True
    return False


def _artifact_marker(artifact: object | None) -> pikepdf.ContentStreamInstruction:
    properties = Dictionary()
    pdf_type = getattr(artifact, "pdf_type", "Layout")
    pdf_subtype = getattr(artifact, "pdf_subtype", None)
    properties["/Type"] = Name(f"/{pdf_type}")
    if pdf_subtype is not None:
        properties["/Subtype"] = Name(f"/{pdf_subtype}")
    return pikepdf.ContentStreamInstruction([Name.Artifact, properties], _BDC)


def _content_marker(mcid: int) -> pikepdf.ContentStreamInstruction:
    return pikepdf.ContentStreamInstruction(
        [Name.Span, Dictionary(MCID=mcid)],
        _BDC,
    )


def _wrap_instruction(
    instruction: pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage,
    marker: pikepdf.ContentStreamInstruction,
) -> tuple[
    pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage,
    ...,
]:
    return (
        marker,
        instruction,
        pikepdf.ContentStreamInstruction([], _EMC),
    )


def _rewrite_ocr_form_semantics(
    form: Stream,
    page_number: int,
    artifacts: dict[str, object],
    actual_text_overrides: dict[str, str],
) -> int:
    instructions = list(pikepdf.parse_content_stream(form))
    artifacts_tagged = 0
    changed = False
    for index, instruction in enumerate(instructions):
        if (
            isinstance(instruction, pikepdf.ContentStreamInlineImage)
            or instruction.operator != _BDC
            or len(instruction.operands) < 2
        ):
            continue
        properties = resolve_indirect(instruction.operands[1])
        if not isinstance(properties, Dictionary):
            continue
        mcid = resolve_indirect(properties.get("/MCID"))
        if not isinstance(mcid, int) or isinstance(mcid, bool) or mcid < 0:
            continue
        span_id = _semantic_span_id(page_number, "ocr", mcid)
        artifact = artifacts.get(span_id)
        if artifact is not None:
            instructions[index] = _artifact_marker(artifact)
            artifacts_tagged += 1
            changed = True
            continue
        actual_text = actual_text_overrides.get(span_id)
        if actual_text is None:
            continue
        rewritten_properties = Dictionary()
        for key in properties.keys():
            rewritten_properties[key] = properties[key]
        rewritten_properties["/ActualText"] = _bounded_pdf_string(actual_text)
        operands = list(instruction.operands)
        operands[1] = rewritten_properties
        instructions[index] = pikepdf.ContentStreamInstruction(
            operands,
            instruction.operator,
        )
        changed = True
    if changed:
        form.write(pikepdf.unparse_content_stream(instructions))
    return artifacts_tagged


def _marked_instruction_without_mcid(
    instruction: pikepdf.ContentStreamInstruction,
    resources: Dictionary | None,
    page_number: int,
) -> tuple[
    pikepdf.ContentStreamInstruction,
    bool,
    bool,
    Dictionary | None,
]:
    operands = list(instruction.operands)
    if len(operands) < 2:
        raise ConversionError(
            f"Cannot create semantic structure: page {page_number} has malformed "
            "marked-content properties"
        )
    raw_properties = resolve_indirect(operands[1])
    named_properties = None
    properties = raw_properties
    if isinstance(properties, Name):
        properties = _named_property(resources, properties)
        named_properties = properties if isinstance(properties, Dictionary) else None
    if not isinstance(properties, Dictionary):
        raise ConversionError(
            f"Cannot create semantic structure: page {page_number} has unresolved "
            "marked-content properties"
        )
    if "/MCID" not in properties:
        return instruction, False, bool(properties), None

    cleaned = Dictionary()
    for key in properties.keys():
        if str(key) != "/MCID":
            cleaned[key] = properties[key]
    if named_properties is None:
        operands[1] = cleaned
        instruction = pikepdf.ContentStreamInstruction(operands, instruction.operator)
    return instruction, True, bool(cleaned), named_properties


def _rewrite_semantic_content(
    owner: pikepdf.Page | Stream,
    page_number: int,
    description: str,
    referenced_ids: frozenset[str],
    artifacts: dict[str, object],
    bindings: dict[str, _SemanticBinding],
    source_artifact_ids: frozenset[str],
    resources: Dictionary | None,
    source_prefix: str,
    expected_references: frozenset[str],
    expected_source_artifacts: frozenset[str],
    ocr_target: tuple[Name, Stream] | None,
    form_replacements: dict[int, tuple[Name, Name, Stream]],
    expected_xobjects: dict[int, tuple[str, str | None]] | None,
) -> tuple[bytes, int, int, int, tuple[Dictionary, ...]]:
    try:
        instructions = list(pikepdf.parse_content_stream(owner))
    except Exception as exc:
        raise ConversionError(
            f"Cannot create semantic structure: {description} cannot be parsed"
        ) from exc

    rewritten: list[
        pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage
    ] = []
    next_mcid = 0
    text_index = 0
    xobject_index = 0
    marked_content_index = 0
    target_invocations = 0
    artifacts_tagged = 0
    mcids_removed = 0
    seen_references: set[str] = set()
    seen_source_artifacts: set[str] = set()
    seen_form_replacements: set[int] = set()
    nesting: list[tuple[str, bool, bool, bool]] = []
    named_properties_to_clean: dict[_ObjectKey | int, Dictionary] = {}
    path_start: int | None = None

    def in_source_artifact() -> bool:
        return any(
            kind == "marked" and artifact
            for kind, _emit, artifact, _bound_scope in nesting
        )

    def in_bound_scope() -> bool:
        return any(
            kind == "marked" and bound_scope
            for kind, _emit, _artifact, bound_scope in nesting
        )

    def append_paint(
        instruction: pikepdf.ContentStreamInstruction
        | pikepdf.ContentStreamInlineImage,
        span_id: str | None,
        object_start: int | None = None,
    ) -> None:
        nonlocal next_mcid, artifacts_tagged
        if in_source_artifact():
            binding = bindings.get(span_id) if span_id is not None else None
            if binding is not None:
                if (
                    span_id not in source_artifact_ids
                    or binding.page_number != page_number
                    or not _same_object(binding.container, owner)
                ):
                    raise ConversionError(
                        "Cannot create semantic structure: source Artifact "
                        "provenance no longer matches"
                    )
                seen_source_artifacts.add(span_id)
            rewritten.append(instruction)
            return
        if span_id is not None and span_id in source_artifact_ids:
            raise ConversionError(
                "Cannot create semantic structure: source Artifact scope is missing"
            )
        if in_bound_scope():
            rewritten.append(instruction)
            return
        if span_id is not None and span_id in referenced_ids:
            binding = bindings.get(span_id)
            if (
                binding is None
                or binding.page_number != page_number
                or not _same_object(binding.container, owner)
            ):
                raise ConversionError(
                    "Cannot create semantic structure: content binding is missing"
                )
            if next_mcid >= _MAX_ARRAY_ITEMS:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has too "
                    "many marked-content items"
                )
            binding.mcid = next_mcid
            seen_references.add(span_id)
            marker = _content_marker(next_mcid)
            next_mcid += 1
        else:
            artifact = artifacts.get(span_id) if span_id is not None else None
            marker = _artifact_marker(artifact)
            artifacts_tagged += 1
        if object_start is None:
            rewritten.extend(_wrap_instruction(instruction, marker))
        else:
            rewritten.insert(object_start, marker)
            rewritten.append(instruction)
            rewritten.append(pikepdf.ContentStreamInstruction([], _EMC))

    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            if expected_xobjects is not None and expected_xobjects.get(
                xobject_index
            ) != ("inline_image", None):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} nested "
                    "XObject provenance no longer matches"
                )
            span_id = _semantic_span_id(
                page_number,
                f"{source_prefix}xobject",
                xobject_index,
            )
            xobject_index += 1
            append_paint(instruction, span_id)
            continue
        operator = instruction.operator
        operands = instruction.operands
        if operator in {_BMC, _BDC}:
            current_marked_content_index = marked_content_index
            marked_content_index += 1
            tag = resolve_indirect(operands[0]) if operands else None
            if not isinstance(tag, Name):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "malformed marked content"
                )
            span_id = _semantic_span_id(
                page_number,
                f"{source_prefix}marked",
                current_marked_content_index,
            )
            bound_scope = False
            emit = True
            original_properties = (
                _marked_content_properties(list(operands), resources)
                if operator == _BDC
                else None
            )
            if operator == _BDC:
                (
                    instruction,
                    removed,
                    has_properties,
                    named_properties,
                ) = _marked_instruction_without_mcid(
                    instruction,
                    resources,
                    page_number,
                )
                mcids_removed += int(removed)
                if named_properties is not None:
                    key: _ObjectKey | int = _object_key(named_properties) or id(
                        named_properties
                    )
                    named_properties_to_clean[key] = named_properties
                emit = not (
                    removed
                    and not has_properties
                    and tag not in {Name.Artifact, Name("/OC")}
                )
            if span_id in referenced_ids:
                binding = bindings.get(span_id)
                if (
                    operator != _BDC
                    or tag == Name.Artifact
                    or in_source_artifact()
                    or binding is None
                    or binding.page_number != page_number
                    or not _same_object(binding.container, owner)
                    or not isinstance(original_properties, Dictionary)
                ):
                    raise ConversionError(
                        "Cannot create semantic structure: ActualText vector "
                        "scope provenance no longer matches"
                    )
                if next_mcid >= _MAX_ARRAY_ITEMS:
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} has too "
                        "many marked-content items"
                    )
                rewritten_properties = Dictionary()
                for key in original_properties.keys():
                    if str(key) != "/MCID":
                        rewritten_properties[key] = original_properties[key]
                rewritten_properties["/MCID"] = next_mcid
                instruction = pikepdf.ContentStreamInstruction(
                    [tag, rewritten_properties],
                    _BDC,
                )
                binding.mcid = next_mcid
                next_mcid += 1
                seen_references.add(span_id)
                emit = True
                bound_scope = True
            elif span_id in artifacts:
                instruction = _artifact_marker(artifacts[span_id])
                artifacts_tagged += 1
                emit = True
                tag = Name.Artifact
            nesting.append(("marked", emit, tag == Name.Artifact, bound_scope))
            if emit:
                rewritten.append(instruction)
            continue
        if operator == _EMC:
            if not nesting or nesting[-1][0] != "marked":
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "unbalanced marked content"
                )
            _kind, emit, _artifact, _bound_scope = nesting.pop()
            if emit:
                rewritten.append(instruction)
            continue
        if operator in {_DP, _MP}:
            tag = resolve_indirect(operands[0]) if operands else None
            if not isinstance(tag, Name):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has a "
                    "malformed marked-content point"
                )
            if operator == _DP:
                (
                    instruction,
                    removed,
                    has_properties,
                    named_properties,
                ) = _marked_instruction_without_mcid(
                    instruction,
                    resources,
                    page_number,
                )
                mcids_removed += int(removed)
                if named_properties is not None:
                    key = _object_key(named_properties) or id(named_properties)
                    named_properties_to_clean[key] = named_properties
                if (
                    removed
                    and not has_properties
                    and tag
                    not in {
                        Name.Artifact,
                        Name("/OC"),
                    }
                ):
                    continue
            rewritten.append(instruction)
            continue
        if operator == _BT:
            if any(kind == "text" for kind, _emit, _artifact, _bound_scope in nesting):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "nested text objects"
                )
            nesting.append(("text", True, False, False))
            rewritten.append(instruction)
            continue
        if operator == _ET:
            if not nesting or nesting[-1][0] != "text":
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "crossed text and marked-content boundaries"
                )
            nesting.pop()
            rewritten.append(instruction)
            continue
        operator_name = str(operator)
        if operator_name in {"Tj", "TJ", "'", '"'}:
            if len(operands) != _TEXT_SHOW_OPERAND_COUNTS[operator_name]:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "malformed direct text provenance"
                )
            span_id = _semantic_span_id(
                page_number,
                f"{source_prefix}text",
                text_index,
            )
            text_index += 1
            append_paint(instruction, span_id)
            continue
        if operator_name == "Do":
            if len(operands) != 1:
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has "
                    "malformed direct XObject provenance"
                )
            current_xobject_index = xobject_index
            xobject_index += 1
            if (
                ocr_target is not None
                and instruction.operands
                and resolve_indirect(instruction.operands[0]) == ocr_target[0]
            ):
                if in_source_artifact():
                    raise ConversionError(
                        "Cannot create semantic structure: OCR content is inside "
                        "an Artifact sequence"
                    )
                rewritten.append(instruction)
                target_invocations += 1
                continue
            operand_name = resolve_indirect(operands[0]) if operands else None
            if not isinstance(operand_name, Name):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has a "
                    "malformed XObject invocation"
                )
            xobjects = (
                resolve_indirect(resources.get("/XObject"))
                if resources is not None
                else None
            )
            xobject = (
                resolve_indirect(xobjects.get(operand_name))
                if isinstance(xobjects, Dictionary)
                else None
            )
            replacement = form_replacements.get(current_xobject_index)
            if replacement is not None:
                old_name, new_name, source_form = replacement
                if (
                    in_source_artifact()
                    or operand_name != old_name
                    or not isinstance(xobject, Stream)
                    or not _same_object(xobject, source_form)
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} Form "
                        "invocation provenance no longer matches"
                    )
                rewritten.append(
                    pikepdf.ContentStreamInstruction([new_name], instruction.operator)
                )
                seen_form_replacements.add(current_xobject_index)
                continue
            if expected_xobjects is not None:
                expected = expected_xobjects.get(current_xobject_index)
                actual_kind = (
                    "form"
                    if isinstance(xobject, Stream)
                    and resolve_indirect(xobject.get("/Subtype")) == Name.Form
                    else "image"
                    if isinstance(xobject, Stream)
                    and resolve_indirect(xobject.get("/Subtype")) == Name.Image
                    else None
                )
                if expected != (actual_kind, str(operand_name).removeprefix("/")):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} nested "
                        "XObject provenance no longer matches"
                    )
            span_id = _semantic_span_id(
                page_number,
                f"{source_prefix}xobject",
                current_xobject_index,
            )
            append_paint(instruction, span_id)
            continue
        if operator_name in {"m", "re"} and path_start is None:
            path_start = len(rewritten)
        if operator_name == "n":
            path_start = None
        if operator_name in _PATH_PAINTING_OPERATORS:
            append_paint(instruction, None, path_start)
            path_start = None
            continue
        if operator_name in _PAINTING_OPERATORS:
            append_paint(instruction, None)
            continue
        rewritten.append(instruction)

    if nesting:
        raise ConversionError(
            f"Cannot create semantic structure: {description} has unbalanced "
            "marked content or text objects"
        )
    if seen_references != expected_references:
        raise ConversionError(
            f"Cannot create semantic structure: {description} content "
            "provenance no longer matches"
        )
    if seen_source_artifacts != expected_source_artifacts:
        raise ConversionError(
            f"Cannot create semantic structure: {description} Artifact "
            "provenance no longer matches"
        )
    if seen_form_replacements != set(form_replacements):
        raise ConversionError(
            f"Cannot create semantic structure: {description} Form invocation "
            "provenance no longer matches"
        )
    if expected_xobjects is not None and set(expected_xobjects) != set(
        range(xobject_index)
    ):
        raise ConversionError(
            f"Cannot create semantic structure: {description} nested XObject "
            "provenance no longer matches"
        )
    if ocr_target is not None and target_invocations != 1:
        raise ConversionError(
            f"Cannot create semantic structure: page {page_number} invokes its "
            f"OCR Form {target_invocations} times"
        )
    return (
        pikepdf.unparse_content_stream(rewritten),
        next_mcid,
        artifacts_tagged,
        mcids_removed,
        tuple(named_properties_to_clean.values()),
    )


def _shallow_pdf_dictionary(source: Dictionary) -> Dictionary:
    copied = Dictionary()
    for key in source.keys():
        copied[key] = source[key]
    return copied


def _clone_semantic_form(pdf: pikepdf.Pdf, source: Stream) -> Stream:
    try:
        clone = pdf.make_stream(bytes(source.read_bytes()))
    except Exception as exc:
        raise ConversionError(
            "Cannot create semantic structure: Form XObject cannot be cloned"
        ) from exc
    excluded = {
        "/Length",
        "/DL",
        "/Filter",
        "/DecodeParms",
        "/F",
        "/FFilter",
        "/FDecodeParms",
        "/StructParent",
        "/StructParents",
    }
    for key in source.keys():
        if str(key) not in excluded:
            clone[key] = source[key]
    return clone


def _prepare_semantic_form_invocations(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    page_number: int,
    invocations: tuple[_SemanticFormInvocation, ...],
    referenced_ids: frozenset[str],
    artifacts: dict[str, object],
    bindings: dict[str, _SemanticBinding],
    source_artifact_ids: frozenset[str],
) -> tuple[
    Dictionary | None,
    dict[int, tuple[Name, Name, Stream]],
    frozenset[_ObjectKey],
    int,
    int,
    tuple[Dictionary, ...],
]:
    if not invocations:
        return None, {}, frozenset(), 0, 0, ()

    resources = _page_resources(page)
    xobjects = (
        resolve_indirect(resources.get("/XObject")) if resources is not None else None
    )
    if resources is None or not isinstance(xobjects, Dictionary):
        raise ConversionError(
            f"Cannot create semantic structure: page {page_number} Form resources "
            "cannot be updated safely"
        )

    def direct_xobject_names(
        owner: pikepdf.Page | Stream,
        description: str,
    ) -> tuple[dict[int, Name], dict[str, int]]:
        try:
            instructions = pikepdf.parse_content_stream(owner)
        except Exception as exc:
            raise ConversionError(
                f"Cannot create semantic structure: {description} cannot be parsed"
            ) from exc
        names: dict[int, Name] = {}
        uses: dict[str, int] = {}
        xobject_index = 0
        for instruction in instructions:
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                xobject_index += 1
                continue
            if str(instruction.operator) != "Do":
                continue
            operand = (
                resolve_indirect(instruction.operands[0])
                if instruction.operands
                else None
            )
            if not isinstance(operand, Name):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} has a "
                    "malformed XObject invocation"
                )
            names[xobject_index] = operand
            uses[str(operand)] = uses.get(str(operand), 0) + 1
            xobject_index += 1
        return names, uses

    def target_form_name(
        invocation: _SemanticFormInvocation,
        source_name: Name,
        available_xobjects: Dictionary,
        name_uses: dict[str, int],
        claimed_names: set[str],
    ) -> Name:
        path = re.findall(
            r"(?:form|xobject)-([0-9]+)-",
            invocation.source_prefix,
        )
        if not path:
            raise ConversionError(
                "Cannot create semantic structure: Form invocation path is invalid"
            )
        base_name = f"/PdftopdfaSemanticForm{page_number}_{'_'.join(path)}"
        current_name = str(source_name)
        if name_uses.get(current_name) == 1 and re.fullmatch(
            rf"{re.escape(base_name)}(?:_[1-9][0-9]*)?",
            current_name,
        ):
            target_name = source_name
        else:
            suffix = 0
            while True:
                candidate_text = base_name if suffix == 0 else f"{base_name}_{suffix}"
                candidate = Name(candidate_text)
                if (
                    candidate_text not in claimed_names
                    and candidate not in available_xobjects
                ):
                    target_name = candidate
                    break
                suffix += 1
        claimed_names.add(str(target_name))
        return target_name

    copied_resources = _shallow_pdf_dictionary(resources)
    copied_xobjects = _shallow_pdf_dictionary(xobjects)
    copied_resources["/XObject"] = copied_xobjects
    replacements: dict[int, tuple[Name, Name, Stream]] = {}
    clone_keys: set[_ObjectKey] = set()
    artifacts_tagged = 0
    mcids_removed = 0
    named_properties: dict[_ObjectKey | int, Dictionary] = {}
    binding_updates: list[tuple[str, Stream]] = []

    def prepare_invocation(
        invocation: _SemanticFormInvocation,
        inherited_resources: Dictionary | None,
        target_name: Name,
        description: str,
    ) -> Stream:
        nonlocal artifacts_tagged, mcids_removed
        source_form = invocation.source
        for span_id in invocation.span_ids:
            binding = bindings.get(span_id)
            if binding is None or not _same_object(binding.container, source_form):
                raise ConversionError(
                    "Cannot create semantic structure: Form content binding is missing"
                )

        raw_resources = source_form.get("/Resources")
        own_resources = resolve_indirect(raw_resources)
        if raw_resources is not None and not isinstance(own_resources, Dictionary):
            raise ConversionError(
                f"Cannot create semantic structure: {description} resources are "
                "malformed"
            )
        effective_resources = (
            own_resources
            if isinstance(own_resources, Dictionary) and own_resources
            else inherited_resources
        )
        child_replacements: dict[int, tuple[Name, Name, Stream]] = {}
        cloned_resources: Dictionary | None = None
        if invocation.children:
            effective_xobjects = (
                resolve_indirect(effective_resources.get("/XObject"))
                if effective_resources is not None
                else None
            )
            if not isinstance(effective_xobjects, Dictionary):
                raise ConversionError(
                    f"Cannot create semantic structure: {description} nested Form "
                    "resources cannot be updated safely"
                )
            direct_names, name_uses = direct_xobject_names(source_form, description)
            cloned_resources = _shallow_pdf_dictionary(effective_resources)
            cloned_xobjects = _shallow_pdf_dictionary(effective_xobjects)
            cloned_resources["/XObject"] = cloned_xobjects
            claimed_names: set[str] = set()
            seen_indices: set[int] = set()
            for child in sorted(
                invocation.children,
                key=lambda item: item.page_xobject_index,
            ):
                if child.page_xobject_index in seen_indices:
                    raise ConversionError(
                        "Cannot create semantic structure: duplicate nested Form "
                        "invocation"
                    )
                seen_indices.add(child.page_xobject_index)
                child_source_name = direct_names.get(child.page_xobject_index)
                child_source_form = (
                    resolve_indirect(effective_xobjects.get(child_source_name))
                    if child_source_name is not None
                    else None
                )
                if (
                    child_source_name != child.resource_name
                    or not isinstance(child_source_form, Stream)
                    or not _same_object(child_source_form, child.source)
                ):
                    raise ConversionError(
                        f"Cannot create semantic structure: {description} nested "
                        "Form invocation provenance no longer matches"
                    )
                child_target_name = target_form_name(
                    child,
                    child_source_name,
                    cloned_xobjects,
                    name_uses,
                    claimed_names,
                )
                child_clone = prepare_invocation(
                    child,
                    effective_resources,
                    child_target_name,
                    (
                        f"Form XObject {child_source_name} invocation "
                        f"{child.page_xobject_index} in {description}"
                    ),
                )
                cloned_xobjects[child_target_name] = child_clone
                child_replacements[child.page_xobject_index] = (
                    child_source_name,
                    child_target_name,
                    child_source_form,
                )

        invocation_references = frozenset(invocation.span_ids & referenced_ids)
        invocation_source_artifacts = frozenset(
            invocation.span_ids & source_artifact_ids
        )
        (
            content,
            _next_mcid,
            invocation_artifacts,
            invocation_removed,
            invocation_named_properties,
        ) = _rewrite_semantic_content(
            source_form,
            page_number,
            description,
            referenced_ids,
            artifacts,
            bindings,
            source_artifact_ids,
            effective_resources,
            invocation.source_prefix,
            invocation_references,
            invocation_source_artifacts,
            None,
            child_replacements,
            invocation.expected_xobjects,
        )
        clone = _clone_semantic_form(pdf, source_form)
        clone_key = _object_key(clone)
        if clone_key is None:
            raise ConversionError(
                "Cannot create semantic structure: cloned Form XObject is direct"
            )
        if cloned_resources is not None:
            clone["/Resources"] = cloned_resources
        clone.write(content)
        clone_keys.add(clone_key)
        invocation.clone = clone
        invocation.target_name = target_name
        binding_updates.extend((span_id, clone) for span_id in invocation.span_ids)
        artifacts_tagged += invocation_artifacts
        mcids_removed += invocation_removed
        for properties in invocation_named_properties:
            key: _ObjectKey | int = _object_key(properties) or id(properties)
            named_properties[key] = properties
        return clone

    page_names, page_name_uses = direct_xobject_names(page, f"page {page_number}")
    claimed_page_names: set[str] = set()
    seen_page_indices: set[int] = set()
    for invocation in sorted(
        invocations,
        key=lambda item: item.page_xobject_index,
    ):
        if invocation.page_xobject_index in seen_page_indices:
            raise ConversionError(
                "Cannot create semantic structure: duplicate page Form invocation"
            )
        seen_page_indices.add(invocation.page_xobject_index)
        source_name = page_names.get(invocation.page_xobject_index)
        source_form = (
            resolve_indirect(xobjects.get(source_name))
            if source_name is not None
            else None
        )
        if (
            source_name != invocation.resource_name
            or not isinstance(source_form, Stream)
            or not _same_object(source_form, invocation.source)
        ):
            raise ConversionError(
                f"Cannot create semantic structure: page {page_number} Form "
                "invocation provenance no longer matches"
            )
        target_name = target_form_name(
            invocation,
            source_name,
            copied_xobjects,
            page_name_uses,
            claimed_page_names,
        )
        clone = prepare_invocation(
            invocation,
            resources,
            target_name,
            (
                f"Form XObject {source_name} invocation "
                f"{invocation.page_xobject_index} on page {page_number}"
            ),
        )
        copied_xobjects[target_name] = clone
        replacements[invocation.page_xobject_index] = (
            source_name,
            target_name,
            source_form,
        )

    for span_id, clone in binding_updates:
        binding = bindings[span_id]
        binding.container = clone
        binding.stream = clone

    return (
        copied_resources,
        replacements,
        frozenset(clone_keys),
        artifacts_tagged,
        mcids_removed,
        tuple(named_properties.values()),
    )


def _rewrite_semantic_page(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    page_number: int,
    referenced_ids: frozenset[str],
    artifacts: dict[str, object],
    bindings: dict[str, _SemanticBinding],
    source_artifact_ids: frozenset[str],
    ocr_target: tuple[Name, Stream] | None,
    form_replacements: dict[int, tuple[Name, Name, Stream]],
    replacement_resources: Dictionary | None,
) -> tuple[int, int, int, tuple[Dictionary, ...]]:
    expected_references = frozenset(
        span_id
        for span_id in referenced_ids
        if _same_object(bindings[span_id].container, page.obj)
    )
    expected_source_artifacts = frozenset(
        span_id
        for span_id in source_artifact_ids
        if _same_object(bindings[span_id].container, page.obj)
    )
    (
        content,
        next_mcid,
        artifacts_tagged,
        mcids_removed,
        named_properties,
    ) = _rewrite_semantic_content(
        page,
        page_number,
        f"page {page_number}",
        referenced_ids,
        artifacts,
        bindings,
        source_artifact_ids,
        _page_resources(page),
        "",
        expected_references,
        expected_source_artifacts,
        ocr_target,
        form_replacements,
        None,
    )
    new_contents = pdf.make_stream(content)
    if replacement_resources is not None:
        page.obj["/Resources"] = replacement_resources
    page.obj["/Contents"] = new_contents
    if "/StructParent" in page.obj:
        del page.obj["/StructParent"]
    if "/StructParents" in page.obj:
        del page.obj["/StructParents"]
    page.obj["/Tabs"] = Name.S
    return (
        next_mcid,
        artifacts_tagged,
        mcids_removed,
        named_properties,
    )


def _structure_attribute_value(value: str) -> Name | String:
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", value):
        return Name(f"/{value}")
    return String(value)


def _set_semantic_attributes(
    element: Dictionary,
    node: object,
    pdf_pages: list[pikepdf.Page],
) -> None:
    grouped: dict[str, Dictionary] = {}
    for attribute in getattr(node, "attributes", ()):
        owner = attribute.owner
        properties = grouped.setdefault(owner, Dictionary(O=Name(f"/{owner}")))
        properties[f"/{attribute.name}"] = _structure_attribute_value(attribute.value)

    bbox = getattr(node, "bbox", None)
    page_number = getattr(node, "page_number", None)
    if (
        bbox is not None
        and isinstance(page_number, int)
        and 1 <= page_number <= len(pdf_pages)
    ):
        geometry = _page_geometry(pdf_pages[page_number - 1])
        default_bbox = geometry.visual_to_default_bbox(
            (bbox.left, bbox.top, bbox.right, bbox.bottom)
        )
        layout = grouped.setdefault("Layout", Dictionary(O=Name.Layout))
        layout["/BBox"] = Array(default_bbox)
    if grouped:
        attributes = list(grouped.values())
        element["/A"] = attributes[0] if len(attributes) == 1 else Array(attributes)


def _semantic_node_has_content(node: object) -> bool:
    walk = getattr(node, "walk", None)
    return callable(walk) and any(
        bool(getattr(candidate, "content", ())) for candidate in walk()
    )


def _semantic_node_source_text(
    node: object,
    source_texts: dict[str, str],
) -> str | None:
    walk = getattr(node, "walk", None)
    if not callable(walk):
        return None
    references = [
        reference
        for candidate in walk()
        for reference in getattr(candidate, "content", ())
    ]
    if not references or any(
        reference.span_id not in source_texts for reference in references
    ):
        return None
    values: list[str] = []
    for reference in references:
        value = source_texts[reference.span_id].strip()
        if value and (not values or values[-1] != value):
            values.append(value)
    return " ".join(values) or None


def _missing_plan_alternatives(
    root: object,
    source_actual_texts: dict[str, str],
    source_alt_texts: dict[str, str],
) -> int:
    walk = getattr(root, "walk", None)
    if not callable(walk):
        return 0
    missing = 0
    for node in walk():
        if getattr(node, "role", None) not in {"Figure", "Formula"}:
            continue
        if not _semantic_node_has_content(node):
            continue
        actual_text = getattr(node, "actual_text", None)
        if isinstance(actual_text, str) and actual_text.strip():
            continue
        if _semantic_node_source_text(node, source_actual_texts) is not None:
            continue
        if _semantic_node_source_text(node, source_alt_texts) is not None:
            continue
        if any(
            getattr(child, "role", None) == "Caption"
            and _semantic_node_has_content(child)
            for child in getattr(node, "children", ())
        ):
            continue
        missing += 1
    return missing


def _make_semantic_element(
    pdf: pikepdf.Pdf,
    node: object,
    parent: Dictionary,
    pdf_pages: list[pikepdf.Page],
    bindings: dict[str, _SemanticBinding],
    owners: dict[tuple[_ObjectKey, int], Dictionary],
    page_elements: dict[int, Dictionary],
    source_actual_texts: dict[str, str],
    source_alt_texts: dict[str, str],
) -> Dictionary:
    role = getattr(node, "role", None)
    if not isinstance(role, str) or f"/{role}" not in _STANDARD_STRUCTURE_TYPES:
        raise ConversionError(
            f"Cannot create semantic structure: unsupported role {role!r}"
        )
    page_number = getattr(node, "page_number", None)
    if page_number is not None and not 1 <= page_number <= len(pdf_pages):
        raise ConversionError(
            "Cannot create semantic structure: invalid structure page number"
        )
    properties = Dictionary(
        Type=Name.StructElem,
        S=Name(f"/{role}"),
        P=parent,
    )
    if page_number is not None:
        properties["/Pg"] = pdf_pages[page_number - 1].obj
    element = pdf.make_indirect(properties)
    if role == "Div" and page_number is not None:
        page_elements[page_number] = element

    actual_text = getattr(node, "actual_text", None)
    if (not isinstance(actual_text, str) or not actual_text.strip()) and (
        role in {"Figure", "Formula"} or bool(getattr(node, "content", ()))
    ):
        actual_text = _semantic_node_source_text(node, source_actual_texts)
    if isinstance(actual_text, str):
        element["/ActualText"] = _bounded_pdf_string(actual_text)
    if role in {"Figure", "Formula"}:
        alt_text = _semantic_node_source_text(node, source_alt_texts)
        if alt_text is not None:
            element["/Alt"] = _bounded_pdf_string(alt_text)
    _set_semantic_attributes(element, node, pdf_pages)

    items: list[Dictionary] = []
    for reference in getattr(node, "content", ()):
        span_id = reference.span_id
        binding = bindings.get(span_id)
        if binding is None or binding.mcid is None:
            raise ConversionError(
                f"Cannot create semantic structure: unresolved span {span_id}"
            )
        page = pdf_pages[binding.page_number - 1]
        mcr = Dictionary(Type=Name.MCR, Pg=page.obj, MCID=binding.mcid)
        if binding.stream is not None:
            mcr["/Stm"] = binding.stream
        container_key = _object_key(binding.container)
        owner_key = (container_key, binding.mcid) if container_key is not None else None
        if owner_key is None or owner_key in owners:
            raise ConversionError(
                "Cannot create semantic structure: duplicate content reference"
            )
        owners[owner_key] = element
        items.append(mcr)

    child_elements = [
        _make_semantic_element(
            pdf,
            child,
            element,
            pdf_pages,
            bindings,
            owners,
            page_elements,
            source_actual_texts,
            source_alt_texts,
        )
        for child in getattr(node, "children", ())
    ]
    if not items and len(child_elements) > _MAX_ARRAY_ITEMS:
        child_elements = _bounded_structure_children(pdf, element, child_elements)
    items.extend(child_elements)
    if len(items) > _MAX_ARRAY_ITEMS:
        raise ConversionError(
            "Cannot create semantic structure: structure element has too many children"
        )
    if len(items) == 1:
        element["/K"] = items[0]
    elif items:
        element["/K"] = Array(items)

    return element


def _assign_semantic_parent_tree(
    parent_tree: NumberTree,
    bindings: dict[str, _SemanticBinding],
    referenced_ids: frozenset[str],
    owners: dict[tuple[_ObjectKey, int], Dictionary],
) -> int:
    containers: dict[_ObjectKey, tuple[Dictionary | Stream, dict[int, Dictionary]]] = {}
    for span_id in sorted(
        referenced_ids,
        key=lambda item: (
            bindings[item].page_number,
            bindings[item].source,
            bindings[item].source_index,
        ),
    ):
        binding = bindings[span_id]
        if binding.mcid is None:
            raise ConversionError(
                "Cannot create semantic structure: unresolved marked content"
            )
        container_key = _object_key(binding.container)
        owner = owners.get((container_key, binding.mcid)) if container_key else None
        if container_key is None or owner is None:
            raise ConversionError(
                "Cannot create semantic structure: parent-tree owner is missing"
            )
        grouped = containers.setdefault(container_key, (binding.container, {}))
        previous = grouped[1].setdefault(binding.mcid, owner)
        if not _same_object(previous, owner):
            raise ConversionError(
                "Cannot create semantic structure: conflicting parent-tree owner"
            )

    next_key = 0
    for container, mapped in containers.values():
        highest = max(mapped)
        if highest >= _MAX_ARRAY_ITEMS:
            raise ConversionError(
                "Cannot create semantic structure: parent array is too large"
            )
        parent_array = Array([None] * (highest + 1))
        for mcid, owner in mapped.items():
            parent_array[mcid] = owner
        if "/StructParent" in container:
            del container["/StructParent"]
        container["/StructParents"] = next_key
        parent_tree[next_key] = parent_array
        next_key += 1
    return next_key


def _link_hit_regions(
    annotation: Dictionary,
    geometry: _PageGeometry,
    page_number: int,
) -> tuple[tuple[float, float, float, float], ...]:
    rect = _semantic_rectangle(
        annotation.get("/Rect"),
        f"page {page_number} Link /Rect",
    )
    if "/QuadPoints" not in annotation:
        return (geometry.default_to_visual_bbox(rect),)

    quad_points = resolve_indirect(annotation.get("/QuadPoints"))
    if not isinstance(quad_points, Array) or not quad_points or len(quad_points) % 8:
        raise ConversionError(
            "Cannot create semantic structure: page "
            f"{page_number} Link /QuadPoints is malformed"
        )
    try:
        coordinates = tuple(float(value) for value in quad_points)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(
            "Cannot create semantic structure: page "
            f"{page_number} Link /QuadPoints is malformed"
        ) from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ConversionError(
            "Cannot create semantic structure: page "
            f"{page_number} Link /QuadPoints is malformed"
        )

    regions = []
    for offset in range(0, len(coordinates), 8):
        quad = coordinates[offset : offset + 8]
        xs = quad[0::2]
        ys = quad[1::2]
        left, right = min(xs), max(xs)
        bottom, top = min(ys), max(ys)
        if right <= left or top <= bottom:
            raise ConversionError(
                "Cannot create semantic structure: page "
                f"{page_number} Link /QuadPoints contains an empty quadrilateral"
            )
        regions.append(geometry.default_to_visual_bbox((left, bottom, right, top)))
    return tuple(regions)


def _bbox_hits_regions(
    bbox: tuple[float, float, float, float],
    regions: tuple[tuple[float, float, float, float], ...],
) -> bool:
    span_left, span_top, span_right, span_bottom = bbox
    span_area = (span_right - span_left) * (span_bottom - span_top)
    if span_area <= 0:
        return False
    center_x = (span_left + span_right) / 2
    center_y = (span_top + span_bottom) / 2
    for left, top, right, bottom in regions:
        if left <= center_x <= right and top <= center_y <= bottom:
            return True
        intersection = max(0.0, min(right, span_right) - max(left, span_left)) * max(
            0.0,
            min(bottom, span_bottom) - max(top, span_top),
        )
        if intersection / span_area >= 0.25:
            return True
    return False


def _link_semantic_annotations(
    pdf: pikepdf.Pdf,
    bindings: dict[str, _SemanticBinding],
    referenced_ids: frozenset[str],
    owners: dict[tuple[_ObjectKey, int], Dictionary],
    optional_content: _DefaultOCVisibility,
) -> dict[_ObjectKey, Dictionary]:
    claimed: dict[_ObjectKey, Dictionary] = {}
    for page_number, page in enumerate(pdf.pages, start=1):
        annotations = resolve_indirect(page.obj.get("/Annots"))
        if not isinstance(annotations, Array):
            continue
        geometry = _page_geometry(page)
        page_bindings = [
            (span_id, bindings[span_id])
            for span_id in referenced_ids
            if bindings[span_id].page_number == page_number
        ]
        for item in annotations:
            annotation = resolve_indirect(item)
            annotation_key = _object_key(annotation)
            if (
                not isinstance(annotation, Dictionary)
                or annotation_key is None
                or annotation_key in claimed
                or resolve_indirect(annotation.get("/Subtype")) != Name.Link
            ):
                continue
            if "/OC" in annotation and not _optional_content_is_visible(
                optional_content,
                annotation.get("/OC"),
                f"page {page_number} Link annotation",
            ):
                continue
            hit_regions = _link_hit_regions(annotation, geometry, page_number)

            matched: list[tuple[str, _SemanticBinding, Dictionary]] = []
            for span_id, binding in page_bindings:
                if not _bbox_hits_regions(binding.bbox, hit_regions):
                    continue
                container_key = _object_key(binding.container)
                owner = (
                    owners.get((container_key, binding.mcid))
                    if container_key is not None and binding.mcid is not None
                    else None
                )
                if owner is not None:
                    matched.append((span_id, binding, owner))
            if not matched or any(
                not _same_object(owner, matched[0][2])
                for _span_id, _binding, owner in matched[1:]
            ):
                continue

            parent = matched[0][2]
            raw_kids = resolve_indirect(parent.get("/K"))
            kids = list(raw_kids) if isinstance(raw_kids, Array) else [raw_kids]
            matched_keys = {
                (_object_key(binding.container), binding.mcid)
                for _span_id, binding, _owner in matched
            }
            selected: list[Dictionary] = []
            selected_indexes: list[int] = []
            for index, kid in enumerate(kids):
                kid = resolve_indirect(kid)
                if not isinstance(kid, Dictionary) or kid.get("/Type") != Name.MCR:
                    continue
                stream = resolve_indirect(kid.get("/Stm"))
                container = stream if isinstance(stream, Stream) else page.obj
                mcid = resolve_indirect(kid.get("/MCID"))
                if (_object_key(container), mcid) in matched_keys:
                    selected.append(kid)
                    selected_indexes.append(index)
            if len(selected) != len(matched):
                continue

            object_reference = pdf.make_indirect(
                Dictionary(Type=Name.OBJR, Obj=annotation, Pg=page.obj)
            )
            link = pdf.make_indirect(
                Dictionary(
                    Type=Name.StructElem,
                    S=Name.Link,
                    P=parent,
                    Pg=page.obj,
                    K=Array([*selected, object_reference]),
                )
            )
            selected_index_set = set(selected_indexes)
            first_index = selected_indexes[0]
            replacement = [
                link if index == first_index else kid
                for index, kid in enumerate(kids)
                if index not in selected_index_set or index == first_index
            ]
            parent["/K"] = (
                replacement[0] if len(replacement) == 1 else Array(replacement)
            )
            for _span_id, binding, _owner in matched:
                container_key = _object_key(binding.container)
                assert container_key is not None and binding.mcid is not None
                owners[(container_key, binding.mcid)] = link
            claimed[annotation_key] = link
    return claimed


def _structure_visual_bbox(
    element: Dictionary,
    geometry: _PageGeometry,
    page_number: int,
) -> tuple[float, float, float, float] | None:
    attributes = resolve_indirect(element.get("/A"))
    if attributes is None:
        return None
    if isinstance(attributes, Dictionary):
        candidates = [attributes]
    elif isinstance(attributes, Array):
        candidates = list(attributes)
    else:
        raise ConversionError(
            "Cannot create semantic structure: page "
            f"{page_number} structure attributes are malformed"
        )
    for candidate in candidates:
        candidate = resolve_indirect(candidate)
        if not isinstance(candidate, Dictionary) or (
            resolve_indirect(candidate.get("/O")) != Name.Layout
            or "/BBox" not in candidate
        ):
            continue
        default_bbox = _semantic_rectangle(
            candidate.get("/BBox"),
            f"page {page_number} structure /BBox",
        )
        return geometry.default_to_visual_bbox(default_bbox)
    return None


def _bbox_distance_squared(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    horizontal = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical = max(left[1] - right[3], right[1] - left[3], 0.0)
    return horizontal * horizontal + vertical * vertical


def _merge_semantic_annotations(
    page: pikepdf.Page,
    page_number: int,
    children: list[Dictionary],
    annotations: list[Dictionary],
) -> list[Dictionary]:
    geometry = _page_geometry(page)
    anchors: list[tuple[int, tuple[float, float, float, float]]] = []
    for index, child in enumerate(children):
        resolved = resolve_indirect(child)
        if not isinstance(resolved, Dictionary):
            raise ConversionError(
                "Cannot create semantic structure: page "
                f"{page_number} structure child is malformed"
            )
        bbox = _structure_visual_bbox(resolved, geometry, page_number)
        if bbox is not None:
            anchors.append((index, bbox))

    placements: dict[
        int,
        list[
            tuple[
                tuple[float, float, float, float],
                int,
                Dictionary,
            ]
        ],
    ] = {}
    for annotation_index, annotation in enumerate(annotations):
        bbox = _structure_visual_bbox(annotation, geometry, page_number)
        if bbox is None:
            raise ConversionError(
                "Cannot create semantic structure: page "
                f"{page_number} annotation geometry is missing"
            )
        if anchors:
            anchor_index, anchor_bbox = min(
                anchors,
                key=lambda item: (
                    _bbox_distance_squared(bbox, item[1]),
                    abs((bbox[1] + bbox[3]) - (item[1][1] + item[1][3])),
                    abs((bbox[0] + bbox[2]) - (item[1][0] + item[1][2])),
                    item[0],
                ),
            )
            if bbox[3] <= anchor_bbox[1]:
                slot = anchor_index
            elif bbox[1] >= anchor_bbox[3]:
                slot = anchor_index + 1
            else:
                slot = anchor_index + int(
                    (bbox[1], bbox[0], bbox[3], bbox[2])
                    >= (
                        anchor_bbox[1],
                        anchor_bbox[0],
                        anchor_bbox[3],
                        anchor_bbox[2],
                    )
                )
        else:
            slot = len(children)
        placements.setdefault(slot, []).append(
            ((bbox[1], bbox[0], bbox[3], bbox[2]), annotation_index, annotation)
        )

    merged: list[Dictionary] = []
    for slot in range(len(children) + 1):
        merged.extend(
            annotation for _key, _index, annotation in sorted(placements.get(slot, []))
        )
        if slot < len(children):
            merged.append(children[slot])
    return merged


def _append_semantic_annotations(
    pdf: pikepdf.Pdf,
    page_elements: dict[int, Dictionary],
    parent_tree: NumberTree,
    next_key: int,
    prelinked_annotations: dict[_ObjectKey, Dictionary],
    optional_content: _DefaultOCVisibility,
) -> tuple[int, int, int]:
    tagged = 0
    link_review_required = 0
    seen_annotations: set[_ObjectKey] = set()
    for page_number, page in enumerate(pdf.pages, start=1):
        parent = page_elements.get(page_number)
        if parent is None:
            raise ConversionError(
                "Cannot create semantic structure: page container is missing"
            )
        elements, next_key = _make_annotation_elements(
            pdf,
            page,
            parent,
            parent_tree,
            next_key,
            page_number,
            seen_annotations,
            prelinked_annotations,
            optional_content,
        )
        if not elements:
            continue
        annotation_count = len(elements)
        link_review_required += sum(
            resolve_indirect(element.get("/S")) == Name.Link for element in elements
        )
        existing = resolve_indirect(parent.get("/K"))
        if isinstance(existing, Array):
            children = list(existing)
        elif existing is None:
            children = []
        else:
            children = [existing]
        children = _merge_semantic_annotations(
            page,
            page_number,
            children,
            elements,
        )
        if len(children) > _MAX_ARRAY_ITEMS:
            children = _bounded_structure_children(
                pdf,
                parent,
                children,
                parent_capacity=_MAX_ARRAY_ITEMS,
            )
        parent["/K"] = children[0] if len(children) == 1 else Array(children)
        tagged += annotation_count
    return next_key, tagged + len(prelinked_annotations), link_review_required


def _rebuild_semantic_structure(
    pdf: pikepdf.Pdf,
    ocr_manifest: dict[str, object] | None,
    optional_content: _DefaultOCVisibility,
) -> dict[str, int | bool]:
    from collections import Counter

    from .semantics import ArtifactKind, SemanticPage, build_semantic_plan

    ocr_spans: dict[int, tuple[object, ...]] = {}
    ocr_bindings: dict[str, _SemanticBinding] = {}
    form_targets: dict[int, tuple[Name, Stream]] = {}
    preserved_forms: frozenset[_ObjectKey] = frozenset()
    forced_artifacts: dict[str, object] = {}
    ocr_actual_text_overrides: dict[str, str] = {}
    ocr_column_gutters: dict[int, tuple[float, ...]] = {}
    if ocr_manifest is not None:
        (
            ocr_spans,
            ocr_bindings,
            form_targets,
            preserved_forms,
            forced_artifacts,
            ocr_actual_text_overrides,
            ocr_column_gutters,
        ) = _ocr_semantic_inputs(pdf, ocr_manifest)

    digital_spans: dict[int, tuple[object, ...]] = {}
    digital_bindings: dict[str, _SemanticBinding] = {}
    digital_dimensions: dict[int, tuple[float, float]] = {}
    source_artifact_ids: frozenset[str] = frozenset()
    source_actual_texts: dict[str, str] = {}
    source_alt_texts: dict[str, str] = {}
    form_vector_review_pages: frozenset[int] = frozenset()
    native_reading_pages: frozenset[int] = frozenset()
    optional_artifacts: dict[str, object] = {}
    digital_form_invocations: dict[
        int,
        tuple[_SemanticFormInvocation, ...],
    ] = {}
    (
        digital_spans,
        digital_bindings,
        digital_dimensions,
        source_artifact_ids,
        digital_form_invocations,
        source_actual_texts,
        source_alt_texts,
        form_vector_review_pages,
        native_reading_pages,
        optional_artifacts,
    ) = _digital_semantic_inputs(pdf, form_targets, optional_content)
    if forced_artifacts.keys() & optional_artifacts.keys():
        raise ConversionError(
            "Cannot create semantic structure: conflicting optional-content "
            "artifact binding"
        )
    forced_artifacts.update(optional_artifacts)
    _deduplicate_invisible_digital_text(
        digital_spans,
        forced_artifacts,
        source_actual_texts,
        source_alt_texts,
    )
    if ocr_manifest is not None:
        _deduplicate_ocr_native_text(
            pdf,
            ocr_manifest,
            ocr_spans,
            ocr_bindings,
            forced_artifacts,
            ocr_actual_text_overrides,
            digital_spans,
            source_actual_texts,
        )
    vector_review_required = len(form_vector_review_pages)

    bindings = {**digital_bindings, **ocr_bindings}
    pages = []
    for page_index, page in enumerate(pdf.pages):
        page_spans = digital_spans.get(page_index, ()) + ocr_spans.get(page_index, ())
        if page_index in digital_dimensions:
            width, height = digital_dimensions[page_index]
        else:
            width, height = _page_geometry(page).visual_size
        pages.append(
            SemanticPage(
                page_index + 1,
                width,
                height,
                page_spans,
                reading_order_hint=(
                    tuple(span.id for span in page_spans)
                    if page_index in ocr_spans
                    and page_index not in native_reading_pages
                    else None
                ),
                column_gutters=ocr_column_gutters.get(page_index),
            )
        )

    plan = build_semantic_plan(pages)
    scanned_visual_review_pages = {
        artifact.page_number
        for artifact in plan.artifacts
        if artifact.kind is ArtifactKind.BACKGROUND
        and artifact.page_number - 1 in form_targets
    }
    alternatives_review_required = _missing_plan_alternatives(
        plan.root,
        source_actual_texts,
        source_alt_texts,
    )
    reference_counts = Counter(
        reference.span_id for node in plan.root.walk() for reference in node.content
    )
    artifacts = {artifact.span_id: artifact for artifact in plan.artifacts}
    if artifacts.keys() & forced_artifacts.keys():
        raise ConversionError(
            "Cannot create semantic structure: conflicting forced artifact binding"
        )
    artifacts.update(forced_artifacts)
    if any(count != 1 for count in reference_counts.values()):
        raise ConversionError(
            "Cannot create semantic structure: duplicate plan content reference"
        )
    referenced_ids = frozenset(reference_counts)
    if (
        referenced_ids & artifacts.keys()
        or referenced_ids & source_artifact_ids
        or artifacts.keys() & source_artifact_ids
        or referenced_ids | artifacts.keys() | source_artifact_ids != set(bindings)
    ):
        raise ConversionError(
            "Cannot create semantic structure: plan does not cover every content span"
        )

    stream_structure_keys_removed = _remove_stream_structure_keys(pdf)
    artifacts_tagged = 0
    page_mcids_removed = 0
    named_properties_to_clean: dict[_ObjectKey | int, Dictionary] = {}
    semantic_form_keys: set[_ObjectKey] = set()
    for page_index, (_name, form) in form_targets.items():
        artifacts_tagged += _rewrite_ocr_form_semantics(
            form,
            page_index + 1,
            artifacts,
            ocr_actual_text_overrides,
        )
    for page_index, page in enumerate(pdf.pages):
        (
            replacement_resources,
            form_replacements,
            page_form_keys,
            form_artifacts,
            form_removed,
            form_named_properties,
        ) = _prepare_semantic_form_invocations(
            pdf,
            page,
            page_index + 1,
            digital_form_invocations.get(page_index, ()),
            referenced_ids,
            artifacts,
            bindings,
            source_artifact_ids,
        )
        semantic_form_keys.update(page_form_keys)
        artifacts_tagged += form_artifacts
        page_mcids_removed += form_removed
        for properties in form_named_properties:
            key: _ObjectKey | int = _object_key(properties) or id(properties)
            named_properties_to_clean[key] = properties
        _, page_artifacts, page_removed, named_properties = _rewrite_semantic_page(
            pdf,
            page,
            page_index + 1,
            referenced_ids,
            artifacts,
            bindings,
            source_artifact_ids,
            form_targets.get(page_index),
            form_replacements,
            replacement_resources,
        )
        artifacts_tagged += page_artifacts
        page_mcids_removed += page_removed
        for properties in named_properties:
            key: _ObjectKey | int = _object_key(properties) or id(properties)
            named_properties_to_clean[key] = properties

    (
        marked_content_languages_normalized,
        mcids_removed,
        nested_structure_keys_removed,
    ) = _sanitize_marked_content(
        pdf,
        remove_mcids=True,
        include_page_streams=False,
        preserve_stream_keys=preserved_forms | frozenset(semantic_form_keys),
    )
    stream_structure_keys_removed += nested_structure_keys_removed
    for properties in named_properties_to_clean.values():
        if "/MCID" in properties:
            del properties["/MCID"]

    structure_root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    parent_tree = NumberTree.new(pdf)
    structure_root["/ParentTree"] = parent_tree.obj
    owners: dict[tuple[_ObjectKey, int], Dictionary] = {}
    page_elements: dict[int, Dictionary] = {}
    document = _make_semantic_element(
        pdf,
        plan.root,
        structure_root,
        list(pdf.pages),
        bindings,
        owners,
        page_elements,
        source_actual_texts,
        source_alt_texts,
    )
    structure_root["/K"] = document
    prelinked_annotations = _link_semantic_annotations(
        pdf,
        bindings,
        referenced_ids,
        owners,
        optional_content,
    )
    next_key = _assign_semantic_parent_tree(
        parent_tree,
        bindings,
        referenced_ids,
        owners,
    )
    next_key, annotations_tagged, link_review_required = _append_semantic_annotations(
        pdf,
        page_elements,
        parent_tree,
        next_key,
        prelinked_annotations,
        optional_content,
    )
    structure_root["/ParentTreeNextKey"] = next_key
    pdf.Root["/StructTreeRoot"] = structure_root

    if ocr_manifest is not None and "/Lang" not in pdf.Root:
        languages = ocr_manifest.get("languages")
        if isinstance(languages, list) and languages and isinstance(languages[0], str):
            pdf.Root["/Lang"] = String(languages[0])

    return {
        "structure_preserved": False,
        "structure_rebuilt": True,
        "semantic_structure_generated": True,
        "pages_tagged": len(pdf.pages),
        "annotations_tagged": annotations_tagged,
        "structure_languages_normalized": 0,
        "marked_content_languages_normalized": (marked_content_languages_normalized),
        "mcids_removed": mcids_removed + page_mcids_removed,
        "stream_structure_keys_removed": stream_structure_keys_removed,
        "path_artifacts_tagged": 0,
        "artifacts_tagged": artifacts_tagged,
        "semantic_content_items": len(referenced_ids),
        "semantic_repairs": 0,
        "semantic_alternatives_review_required": alternatives_review_required,
        "semantic_vector_review_required": vector_review_required,
        "semantic_scanned_visual_review_required": len(scanned_visual_review_pages),
        "semantic_link_review_required": link_review_required,
        "semantic_form_review_required": _widgets_requiring_name_review(
            pdf,
            optional_content,
        ),
    }


def _tag_page(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    document: Dictionary,
    parent_tree: NumberTree,
    page_key: int,
    next_key: int,
    seen_annotations: set[_ObjectKey],
) -> tuple[Dictionary, int, int, int]:
    page_number = page_key + 1
    streams = _content_streams(page, page_number)
    native_streams, generated_mcid = _remove_generated_wrapper(streams, page_number)
    if generated_mcid is not None:
        page.obj["/Contents"] = Array(native_streams)
    _, mcids_removed = _sanitize_content_stream(
        page,
        _page_resources(page),
        remove_mcids=True,
        description=f"page {page_number} content",
        rewrite_streams=native_streams,
        pdf=pdf,
    )
    native_streams = _content_streams(page, page_number)
    native_streams = _repair_marked_content(
        native_streams,
        f"page {page_number} content",
        pdf=pdf,
    )
    if len(native_streams) > _MAX_ARRAY_ITEMS - 2:
        page.obj["/Contents"] = Array(native_streams)
        page.contents_coalesce()
        native_streams = _content_streams(page, page_number)
    mcid = 0

    if "/StructParent" in page.obj:
        del page.obj["/StructParent"]
    page.obj["/StructParents"] = page_key
    page.obj["/Tabs"] = Name.S
    element = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Div,
            P=document,
            Pg=page.obj,
        )
    )
    annotation_elements, next_key = _make_annotation_elements(
        pdf,
        page,
        element,
        parent_tree,
        next_key,
        page_number,
        seen_annotations,
    )
    annotation_count = len(annotation_elements)
    annotation_elements = _bounded_structure_children(
        pdf,
        element,
        annotation_elements,
        parent_capacity=_MAX_ARRAY_ITEMS - 1,
    )
    element["/K"] = Array([mcid, *annotation_elements])

    parent_array = Array([None] * (mcid + 1))
    parent_array[mcid] = element
    parent_tree[page_key] = parent_array

    prefix = pdf.make_stream(f"/Div <</MCID {mcid}>> BDC\n".encode("ascii"))
    suffix = pdf.make_stream(b"\nEMC\n")
    page.obj["/Contents"] = Array([prefix, *native_streams, suffix])
    return element, next_key, annotation_count, mcids_removed


def _validate_tagging_content_preflight(
    pdf: pikepdf.Pdf,
    *,
    semantic: bool = False,
) -> None:
    from .digital_layout import (
        _MAX_FORM_NESTING_DEPTH,
        _DecodedContentBudget,
        _validate_content_work_budget,
    )

    budget = _DecodedContentBudget()
    for page_index, page in enumerate(pdf.pages):
        budget.charge_once(page, page_index)
        for owner, _resources in _iter_content_streams_with_resources(page):
            if isinstance(owner, Stream):
                budget.charge_once(owner, page_index)

        page_resources = _page_resources(page)
        annotations = resolve_indirect(page.obj.get("/Annots"))
        if isinstance(annotations, Array):
            for value in annotations:
                annotation = resolve_indirect(value)
                if not isinstance(annotation, Dictionary):
                    continue
                for appearance, _resources in _annotation_appearance_streams(
                    annotation,
                    page_resources,
                ):
                    budget.charge_once(appearance, page_index)

    for item in pdf.objects:
        if not isinstance(item, Stream):
            continue
        if (
            resolve_indirect(item.get("/Subtype")) == Name.Form
            or resolve_indirect(item.get("/PatternType")) == 1
        ):
            budget.charge_once(item, None)
    for charproc, _contexts in _type3_charproc_streams(pdf).values():
        budget.charge_once(charproc, None)

    _validate_content_work_budget(
        pdf,
        frozenset(range(len(pdf.pages))),
        budget.new_counter(),
        strict_provenance=False,
        max_form_nesting_depth=_MAX_FORM_NESTING_DEPTH if semantic else None,
    )


def _ensure_logical_structure_in_place(
    pdf: pikepdf.Pdf,
    *,
    rebuild: bool = False,
    semantic: bool = False,
    ocr_manifest: dict[str, object] | None = None,
    _content_preflight_complete: bool = False,
) -> dict[str, int | bool]:
    """Preserve, repair, or create a deterministic logical structure tree.

    Existing valid rich logical structure is preserved and safe missing role
    properties are repaired locally. With ``semantic=True``, untagged or
    unusable digital documents receive inferred headings, paragraphs, lists,
    tables, figures, artifacts, and annotation relationships based on their
    final direct-painting provenance and layout. ``ocr_manifest`` supplies the
    corresponding line-level provenance and layout for OCR Forms.

    Semantic generation fails closed when extraction, planning, or content
    binding cannot be completed safely. Calls without semantic input retain the
    legacy generic page/content/annotation-order fallback.

    Args:
        pdf: Opened pikepdf PDF object to modify.
        rebuild: Replace even a plausible existing logical structure.
        semantic: Infer a semantic structure for digital page content.
        ocr_manifest: Validated OCR document metadata for line-level binding.

    Returns:
        Statistics describing whether structure was preserved or rebuilt and
        how many pages and annotations were tagged.

    Raises:
        ConversionError: If page content or annotations are malformed.
    """
    semantic_requested = semantic or ocr_manifest is not None
    if not _content_preflight_complete:
        _validate_tagging_content_preflight(pdf, semantic=semantic_requested)
    optional_content: _DefaultOCVisibility | None = None
    if semantic_requested:
        try:
            optional_content = _default_optional_content_visibility(pdf)
        except ValueError as exc:
            raise ConversionError(
                "Cannot create semantic structure: malformed optional-content "
                "default configuration"
            ) from exc
    mark_info_updated = _ensure_mark_info(pdf)
    marked_content_languages_normalized, _, _ = _sanitize_marked_content(
        pdf,
        remove_mcids=False,
        include_page_streams=True,
    )
    existing_content_references: dict[
        tuple[_ObjectKey, int],
        tuple[
            Dictionary | Stream,
            Dictionary,
            Dictionary,
            Dictionary | Stream | None,
        ],
    ] = {}
    existing_object_owners: dict[
        tuple[_ObjectKey, _ObjectKey],
        tuple[Dictionary | Stream, _ObjectKey, Dictionary],
    ] = {}
    existing_elements = (
        None
        if rebuild
        else _existing_structure_elements(
            pdf,
            content_references_out=existing_content_references,
            object_owners_out=existing_object_owners,
        )
    )
    path_artifacts_tagged = 0
    if (
        existing_elements is None
        and not rebuild
        and ocr_manifest is None
        and _has_semantic_structure_roles(pdf)
    ):
        existing_elements, path_artifacts_tagged = _artifact_untagged_path_painting(pdf)
        if existing_elements is not None:
            existing_content_references.clear()
            existing_object_owners.clear()
            existing_elements = _existing_structure_elements(
                pdf,
                content_references_out=existing_content_references,
                object_owners_out=existing_object_owners,
            )
    rich_existing_structure = (
        existing_elements is not None and _has_semantic_structure_roles(pdf)
    )
    hidden_optional_content_references = False
    if (
        semantic_requested
        and optional_content is not None
        and (rich_existing_structure or "/OCProperties" in pdf.Root)
    ):
        hidden_optional_content_references = (
            _existing_structure_references_hidden_optional_content(
                pdf,
                existing_content_references,
                existing_object_owners,
                optional_content,
            )
        )
    if rich_existing_structure and hidden_optional_content_references:
        rich_existing_structure = False
    if (
        rich_existing_structure
        and semantic_requested
        and existing_elements is not None
        and existing_content_references
    ):
        try:
            image_rebuild, described_image_uncertainty = (
                _requires_existing_image_visibility_rebuild(
                    pdf,
                    existing_content_references,
                )
            )
        except ConversionError:
            pass
        else:
            if described_image_uncertainty:
                raise ConversionError(
                    "Cannot preserve logical structure: described image has "
                    "uncertain intrinsic visibility"
                )
            rich_existing_structure = not image_rebuild
    if (
        rich_existing_structure
        and semantic_requested
        and existing_elements is not None
        and existing_content_references
    ):
        structure_root = resolve_indirect(pdf.Root.get("/StructTreeRoot"))
        assert isinstance(structure_root, Dictionary)
        try:
            reading_order_inverted = _has_unambiguous_existing_reading_order_inversion(
                pdf,
                structure_root,
                existing_elements,
                existing_content_references,
            )
        except ConversionError:
            reading_order_inverted = False
        rich_existing_structure = not reading_order_inverted
    if existing_elements is not None and (
        rich_existing_structure
        or not semantic_requested
        or (not existing_content_references and not hidden_optional_content_references)
    ):
        languages_normalized = _normalize_structure_languages(existing_elements)
        semantic_repairs = _repair_existing_semantics(pdf, existing_elements)
        semantic_repairs += _propagate_existing_marked_text_evidence(
            pdf,
            existing_content_references,
        )
        if semantic_repairs and _existing_structure_elements(pdf) is None:
            raise ConversionError(
                "Cannot preserve logical structure after semantic repairs"
            )
        alternatives_review_required = _missing_structure_alternatives(
            pdf,
            existing_elements,
            existing_content_references,
        )
        return {
            "structure_preserved": True,
            "structure_rebuilt": False,
            "semantic_structure_generated": False,
            "pages_tagged": 0,
            "annotations_tagged": 0,
            "mark_info_updated": mark_info_updated,
            "structure_languages_normalized": languages_normalized,
            "marked_content_languages_normalized": (
                marked_content_languages_normalized
            ),
            "mcids_removed": 0,
            "stream_structure_keys_removed": 0,
            "path_artifacts_tagged": path_artifacts_tagged,
            "artifacts_tagged": path_artifacts_tagged,
            "semantic_content_items": 0,
            "semantic_repairs": semantic_repairs,
            "semantic_alternatives_review_required": (alternatives_review_required),
            "semantic_vector_review_required": int(path_artifacts_tagged > 0),
            "semantic_scanned_visual_review_required": 0,
            "semantic_link_review_required": 0,
            "semantic_form_review_required": _widgets_requiring_name_review(
                pdf,
                optional_content,
            ),
        }

    if semantic_requested:
        assert optional_content is not None
        semantic_error: Exception | None = None
        try:
            semantic_result = _rebuild_semantic_structure(
                pdf,
                ocr_manifest,
                optional_content,
            )
        except (ConversionError, TypeError, ValueError) as exc:
            semantic_error = exc
            semantic_result = None
        if semantic_result is not None:
            semantic_result["mark_info_updated"] = mark_info_updated
            return semantic_result
        source = "OCR" if ocr_manifest is not None else "digital PDF"
        raise ConversionError(
            f"Cannot create semantic {source} structure: {semantic_error}"
        ) from semantic_error

    stream_structure_keys_removed = _remove_stream_structure_keys(pdf)
    (
        nested_languages_normalized,
        nested_mcids_removed,
        nested_structure_keys_removed,
    ) = _sanitize_marked_content(
        pdf,
        remove_mcids=True,
        include_page_streams=False,
    )
    stream_structure_keys_removed += nested_structure_keys_removed
    marked_content_languages_normalized += nested_languages_normalized

    structure_root = pdf.make_indirect(Dictionary(Type=Name.StructTreeRoot))
    document = pdf.make_indirect(
        Dictionary(
            Type=Name.StructElem,
            S=Name.Document,
            P=structure_root,
        )
    )
    parent_tree = NumberTree.new(pdf)
    structure_root["/K"] = document
    structure_root["/ParentTree"] = parent_tree.obj

    page_elements: list[Dictionary] = []
    next_key = len(pdf.pages)
    annotations_tagged = 0
    page_mcids_removed = 0
    seen_annotations: set[_ObjectKey] = set()
    for page_key, page in enumerate(pdf.pages):
        element, next_key, annotation_count, removed_count = _tag_page(
            pdf,
            page,
            document,
            parent_tree,
            page_key,
            next_key,
            seen_annotations,
        )
        page_elements.append(element)
        annotations_tagged += annotation_count
        page_mcids_removed += removed_count

    if page_elements:
        document["/K"] = Array(
            _bounded_structure_children(pdf, document, page_elements)
        )
    structure_root["/ParentTreeNextKey"] = next_key
    pdf.Root["/StructTreeRoot"] = structure_root

    return {
        "structure_preserved": False,
        "structure_rebuilt": True,
        "semantic_structure_generated": False,
        "pages_tagged": len(pdf.pages),
        "annotations_tagged": annotations_tagged,
        "mark_info_updated": mark_info_updated,
        "structure_languages_normalized": 0,
        "marked_content_languages_normalized": (marked_content_languages_normalized),
        "mcids_removed": nested_mcids_removed + page_mcids_removed,
        "stream_structure_keys_removed": stream_structure_keys_removed,
        "path_artifacts_tagged": path_artifacts_tagged,
        "artifacts_tagged": path_artifacts_tagged,
        "semantic_content_items": 0,
        "semantic_repairs": 0,
        "semantic_alternatives_review_required": 0,
        "semantic_vector_review_required": 0,
        "semantic_scanned_visual_review_required": 0,
        "semantic_link_review_required": 0,
        "semantic_form_review_required": _widgets_requiring_name_review(pdf),
    }


def ensure_logical_structure(
    pdf: pikepdf.Pdf,
    *,
    rebuild: bool = False,
    semantic: bool = False,
    ocr_manifest: dict[str, object] | None = None,
    preflight: bool = True,
) -> dict[str, int | bool]:
    """Preserve, repair, or create a deterministic logical structure tree.

    Semantic and forced-rebuild calls are by default first executed on an
    isolated, spill-to-disk copy. Input-dependent failures are therefore
    detected before the caller's PDF is changed. The successful operation is
    then applied to the original object.

    That safety net costs a full serialization plus a second complete run of
    the most expensive stage in the pipeline. Pass ``preflight=False`` when the
    caller discards ``pdf`` unsaved on failure - a half-built structure tree is
    then unobservable and rehearsing it is pure overhead.

    Args:
        pdf: Opened pikepdf PDF object to modify.
        rebuild: Replace even a plausible existing logical structure.
        semantic: Infer a semantic structure for digital page content.
        ocr_manifest: Validated OCR document metadata for line-level binding.
        preflight: Rehearse the operation on an isolated copy first.

    Returns:
        Statistics describing whether structure was preserved or rebuilt and
        how many pages and annotations were tagged.

    Raises:
        ConversionError: If page content or annotations are malformed, or if
            the structure operation cannot be completed safely.
    """
    semantic_requested = semantic or ocr_manifest is not None
    _validate_tagging_content_preflight(pdf, semantic=semantic_requested)
    if preflight and (rebuild or semantic_requested):
        try:
            preflight_password = "pdftopdfa-structure-preflight"
            preflight_encryption = (
                pikepdf.Encryption(
                    owner=preflight_password,
                    user=preflight_password,
                    allow=pdf.allow,
                    R=6,
                )
                if pdf.is_encrypted
                else None
            )
            with SpooledTemporaryFile(
                max_size=_PREFLIGHT_MEMORY_LIMIT,
                mode="w+b",
            ) as serialized:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", pikepdf.PageCopyWarning)
                    pdf.save(serialized, encryption=preflight_encryption)
                serialized.seek(0)
                open_options = (
                    {"password": preflight_password}
                    if preflight_encryption is not None
                    else {}
                )
                with pikepdf.Pdf.open(serialized, **open_options) as preflight_pdf:
                    _ensure_logical_structure_in_place(
                        preflight_pdf,
                        rebuild=rebuild,
                        semantic=semantic,
                        ocr_manifest=ocr_manifest,
                        _content_preflight_complete=True,
                    )
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(
                "Cannot preflight logical structure generation"
            ) from exc
    return _ensure_logical_structure_in_place(
        pdf,
        rebuild=rebuild,
        semantic=semantic,
        ocr_manifest=ocr_manifest,
        _content_preflight_complete=True,
    )


__all__ = ["ensure_logical_structure", "get_structural_actualtext_references"]
