# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Preserve or create a generic logical structure for PDF/A level A."""

import re

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
from .utils import resolve_indirect

_BDC = Operator("BDC")
_BMC = Operator("BMC")
_BT = Operator("BT")
_DP = Operator("DP")
_EMC = Operator("EMC")
_ET = Operator("ET")
_MAX_ARRAY_ITEMS = 8_191
_MAX_STRING_BYTES = 32_767
_ObjectKey = tuple[int, int]
_PAINTING_OPERATORS = frozenset(
    {"Tj", "TJ", "'", '"', "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "sh"}
)
_WRAPPER_PREFIX = re.compile(rb"/Div\s*<<\s*/MCID\s+(\d+)\s*>>\s*BDC\s*")
_WRAPPER_SUFFIX = re.compile(rb"\s*EMC\s*")
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


def _valid_class_map(root: Dictionary, elements: list[Dictionary]) -> bool:
    raw_class_map = root.get("/ClassMap")
    class_map = resolve_indirect(raw_class_map)
    if raw_class_map is not None:
        if not isinstance(class_map, Dictionary):
            return False
        for value in class_map.values():
            value = resolve_indirect(value)
            if isinstance(value, (Dictionary, Stream)):
                continue
            if not isinstance(value, Array) or not value:
                return False
            if any(
                not isinstance(resolve_indirect(attribute), (Dictionary, Stream))
                for attribute in value
            ):
                return False

    for element in elements:
        raw_classes = element.get("/C")
        if raw_classes is None:
            continue
        classes = resolve_indirect(raw_classes)
        if isinstance(classes, Name):
            class_names = [classes]
        elif isinstance(classes, Array) and classes:
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
                        if revision < 0:
                            return False
                        index += 1
        else:
            return False
        if not isinstance(class_map, Dictionary) or any(
            str(class_name) not in class_map for class_name in class_names
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
    try:
        instructions = pikepdf.parse_content_stream(owner)
    except Exception as exc:
        raise ConversionError(
            f"Cannot create logical structure: {description} cannot be parsed"
        ) from exc

    mcids: set[int] = set()
    invalid = False
    nesting: list[str] = []
    for instruction in instructions:
        if isinstance(instruction, pikepdf.ContentStreamInlineImage):
            continue
        if instruction.operator == _BMC:
            nesting.append("marked")
        elif instruction.operator == _BDC:
            item = "marked"
            operands = instruction.operands
            tag = resolve_indirect(operands[0]) if operands else None
            is_artifact = isinstance(tag, Name) and str(tag) == "/Artifact"
            if len(operands) >= 2:
                properties = resolve_indirect(operands[1])
                if isinstance(properties, Name):
                    properties = _named_property(resources, properties)
                if isinstance(properties, Dictionary):
                    mcid = resolve_indirect(properties.get("/MCID"))
                    if (
                        isinstance(mcid, int)
                        and not isinstance(mcid, bool)
                        and mcid >= 0
                    ):
                        if is_artifact:
                            invalid = True
                        else:
                            invalid = invalid or "mcid" in nesting or mcid in mcids
                            mcids.add(mcid)
                            item = "mcid"
            nesting.append(item)
        elif instruction.operator == _BT:
            if "text" in nesting:
                invalid = True
            nesting.append("text")
        elif instruction.operator == _EMC:
            if not nesting or nesting[-1] not in {"marked", "mcid"}:
                invalid = True
            else:
                nesting.pop()
        elif instruction.operator == _ET:
            if not nesting or nesting[-1] != "text":
                invalid = True
            else:
                nesting.pop()
    return mcids, invalid or bool(nesting)


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
                    properties = _named_property(resources, properties)
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
            if isinstance(nested_resources, Dictionary)
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
                resources if isinstance(resources, Dictionary) else page_resources
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


def _make_annotation_elements(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    parent: Dictionary,
    parent_tree: NumberTree,
    next_key: int,
    page_number: int,
    seen_annotations: set[_ObjectKey],
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
        object_reference = pdf.make_indirect(
            Dictionary(Type=Name.OBJR, Obj=annotation, Pg=page.obj)
        )
        element = pdf.make_indirect(
            Dictionary(
                Type=Name.StructElem,
                S=Name.Annot,
                P=parent,
                Pg=page.obj,
                K=object_reference,
            )
        )
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


def ensure_logical_structure(
    pdf: pikepdf.Pdf,
    *,
    rebuild: bool = False,
) -> dict[str, int | bool]:
    """Preserve a plausible structure tree or create a deterministic fallback.

    Existing logical structure is preserved by default. Untagged documents, or
    documents passed with ``rebuild=True``, receive a generic hierarchy of one
    ``Document`` element containing one ``Div`` per page. Each page's native
    content streams remain in their original order inside a marked-content
    wrapper. Annotations are represented by ``Annot`` elements and ``OBJR``
    dictionaries.

    The generated fallback records technical reading order only: page order,
    content-stream order, then annotation array order. It does not infer
    semantic headings, paragraphs, tables, or alternative text.

    Args:
        pdf: Opened pikepdf PDF object to modify.
        rebuild: Replace even a plausible existing logical structure.

    Returns:
        Statistics describing whether structure was preserved or rebuilt and
        how many pages and annotations were tagged.

    Raises:
        ConversionError: If page content or annotations are malformed.
    """
    mark_info_updated = _ensure_mark_info(pdf)
    marked_content_languages_normalized, _, _ = _sanitize_marked_content(
        pdf,
        remove_mcids=False,
        include_page_streams=True,
    )
    existing_elements = None if rebuild else _existing_structure_elements(pdf)
    if existing_elements is not None:
        languages_normalized = _normalize_structure_languages(existing_elements)
        return {
            "structure_preserved": True,
            "structure_rebuilt": False,
            "pages_tagged": 0,
            "annotations_tagged": 0,
            "mark_info_updated": mark_info_updated,
            "structure_languages_normalized": languages_normalized,
            "marked_content_languages_normalized": (
                marked_content_languages_normalized
            ),
            "mcids_removed": 0,
            "stream_structure_keys_removed": 0,
        }

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
        "pages_tagged": len(pdf.pages),
        "annotations_tagged": annotations_tagged,
        "mark_info_updated": mark_info_updated,
        "structure_languages_normalized": 0,
        "marked_content_languages_normalized": (marked_content_languages_normalized),
        "mcids_removed": nested_mcids_removed + page_mcids_removed,
        "stream_structure_keys_removed": stream_structure_keys_removed,
    }


__all__ = ["ensure_logical_structure", "get_structural_actualtext_references"]
