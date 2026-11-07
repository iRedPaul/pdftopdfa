# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Repair ICCBased CMYK overprint at its point of use, not in shared resources.

ISO 19005-2/3, 6.2.4.2 restricts the combination of colour space, overprint
flag and OPM at painting time. An ExtGState can be shared by unrelated paints,
pages and forms, so its OPM cannot be changed globally. Repairs temporarily
select OPM 0 around the affected instruction and restore OPM 1 afterwards.
"""

from dataclasses import dataclass, replace

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..exceptions import ConversionError
from ..fonts.traversal import get_page_resources

_STROKE = frozenset({"S", "s", "B", "B*", "b", "b*"})
_FILL = frozenset({"f", "F", "f*", "B", "B*", "b", "b*"})
_TEXT = frozenset({"Tj", "TJ", "'", '"'})


@dataclass(frozen=True)
class _State:
    stroke: str = "other"
    fill: str = "other"
    stroke_overprint: bool = False
    fill_overprint: bool = False
    mode: int = 0
    text_mode: int = 0
    type3: bool = False
    font: object | None = None
    soft_mask: object | None = None
    stroke_pattern: "_Pattern | None" = None
    fill_pattern: "_Pattern | None" = None


@dataclass(frozen=True)
class _Pattern:
    definition: object | None
    base: str
    initial: _State
    resources: Dictionary


@dataclass
class _Plan:
    owner: object
    resources: Dictionary
    instructions: list
    repairs: frozenset[int]
    resource_signature: bytes


@dataclass
class _Budget:
    calls: int = 0


def _object_key(value):
    """Identify a PDF object without traversing its resource-reference graph."""
    if value is None:
        return None
    if value.objgen != (0, 0):
        return value.objgen
    return value.unparse()


def _state_key(state: _State):
    """Include every painting parameter and selected pattern's definition context."""

    def pattern_key(pattern):
        """Represent the resource and initial state captured by a selected pattern."""
        if pattern is None:
            return None
        return (
            _object_key(pattern.definition),
            pattern.base,
            _state_key(pattern.initial),
            pattern.resources.unparse(),
        )

    return (
        state.stroke,
        state.fill,
        state.stroke_overprint,
        state.fill_overprint,
        state.mode,
        state.text_mode,
        state.type3,
        _object_key(state.font),
        _object_key(state.soft_mask),
        pattern_key(state.stroke_pattern),
        pattern_key(state.fill_pattern),
    )


def _inline_space(cs):
    """Expand color-space abbreviations only at an inline-image boundary."""
    if isinstance(cs, Name):
        return {
            "/G": Name.DeviceGray,
            "/RGB": Name.DeviceRGB,
            "/CMYK": Name.DeviceCMYK,
        }.get(str(cs), cs)
    if isinstance(cs, Array) and len(cs) >= 2 and str(cs[0]) in {"/I", "/Indexed"}:
        return Array([Name.Indexed, _inline_space(cs[1]), *list(cs)[2:]])
    return cs


def _error(detail: str) -> ConversionError:
    """Build an error handled by the converter's original-input fallback."""
    return ConversionError(f"Cannot safely repair PDF/A overprint: {detail}")


def _resource(resources: Dictionary, category: str, name: object):
    """Resolve a named resource or fail before any painting stream is changed."""
    table = resources.get(category)
    if not isinstance(table, Dictionary) or not isinstance(name, Name):
        raise _error(f"missing {category} resource {name}")
    value = table.get(str(name))
    if value is None:
        raise _error(f"undefined {category} resource {name}")
    return value


def _colour_kind(cs, resources: Dictionary, seen: frozenset[str] = frozenset()) -> str:
    """Classify actual source spaces; spot-colour alternates are not the source."""
    if isinstance(cs, Name):
        name = str(cs)
        if name in seen:
            return "unknown"
        if name == "/Pattern":
            return "pattern"
        device = {
            "/DeviceGray": "/DefaultGray",
            "/DeviceRGB": "/DefaultRGB",
            "/DeviceCMYK": "/DefaultCMYK",
        }
        table = resources.get("/ColorSpace", Dictionary())
        if name in device:
            default = table.get(device[name]) if isinstance(table, Dictionary) else None
            if default is None:
                return "device_cmyk" if name == "/DeviceCMYK" else "other"
            return _colour_kind(default, resources, seen | {name})
        if isinstance(table, Dictionary) and name in table:
            return _colour_kind(table[name], resources, seen | {name})
        return "unknown"
    if not isinstance(cs, Array) or not cs:
        return "unknown"
    name = str(cs[0])
    if name == "/ICCBased" and len(cs) >= 2 and isinstance(cs[1], Stream):
        return "icc_cmyk" if cs[1].get("/N") == 4 else "other"
    if name == "/Indexed" and len(cs) >= 2:
        return _colour_kind(cs[1], resources, seen)
    if name == "/Pattern":
        return "pattern"
    if name in {"/CalGray", "/CalRGB", "/Lab", "/Separation", "/DeviceN"}:
        return "other"
    return "unknown"


def _pattern_base(cs, resources: Dictionary) -> str:
    """Resolve an uncolored pattern's source color space through resource aliases."""
    seen = set()
    while isinstance(cs, Name) and cs != Name.Pattern:
        if str(cs) in seen:
            return "unknown"
        seen.add(str(cs))
        table = resources.get("/ColorSpace", Dictionary())
        cs = table.get(str(cs)) if isinstance(table, Dictionary) else None
    if isinstance(cs, Array) and len(cs) >= 2 and cs[0] == Name.Pattern:
        return _colour_kind(cs[1], resources)
    return "unknown"


def _graphics_state(state: _State, gs: Dictionary) -> _State:
    """Apply the independently persistent ExtGState parameters used by analysis."""
    values = {}
    # OP also sets nonstroking overprint when this dictionary omits op.
    if "/OP" in gs:
        if not isinstance(gs.OP, bool):
            raise _error("non-boolean /OP")
        values["stroke_overprint"] = gs.OP
        values["fill_overprint"] = gs.OP
    if "/op" in gs:
        if not isinstance(gs.op, bool):
            raise _error("non-boolean /op")
        values["fill_overprint"] = gs.op
    if "/OPM" in gs:
        # ExtGState sanitization has already repaired every reachable dictionary.
        if (
            isinstance(gs.OPM, bool)
            or not isinstance(gs.OPM, int)
            or gs.OPM not in (0, 1)
        ):
            raise _error("invalid /OPM")
        values["mode"] = int(gs.OPM)
    if "/Font" in gs:
        font = gs.Font
        if isinstance(font, Array) and font and isinstance(font[0], Dictionary):
            values["type3"] = font[0].get("/Subtype") == Name.Type3
            values["font"] = font[0]
    if "/SMask" in gs:
        mask = gs.SMask
        values["soft_mask"] = mask.get("/G") if isinstance(mask, Dictionary) else None
    return replace(state, **values)


def _needs_repair(state: _State, *, stroke: bool, fill: bool) -> bool:
    """Identify prohibited paints while rejecting ambiguous combined operations."""
    if state.mode != 1:
        return False
    active = []
    if stroke and state.stroke_overprint:
        active.append(state.stroke)
    if fill and state.fill_overprint:
        active.append(state.fill)
    if any(kind in {"unknown", "pattern"} for kind in active):
        raise _error("overprinted paint uses an unresolved colour space or pattern")
    if "icc_cmyk" not in active:
        return False
    if "device_cmyk" in active:
        # One combined fill/stroke instruction cannot use two OPM values.
        # Resetting both would silently alter the valid DeviceCMYK paint.
        raise _error("combined fill/stroke needs different overprint modes")
    return True


class _Analyzer:
    def __init__(self, budget: _Budget | None = None):
        """Keep completed contexts separate from recursion checks and repair plans."""
        self.plans: dict[tuple[int, int], _Plan] = {}
        self.budget = budget if budget is not None else _Budget()
        self.contexts: set[tuple] = set()
        self.proofs: set[tuple] = set()
        self.instructions: dict[tuple[int, int], list] = {}

    @property
    def calls(self) -> int:
        """Return the number of unique contexts analyzed across nested proofs."""
        return self.budget.calls

    def record(self, plan: _Plan) -> None:
        """Reject shared streams whose contexts require incompatible rewrites."""
        key = plan.owner.objgen
        previous = self.plans.get(key)
        if previous is not None and (previous.repairs or plan.repairs):
            if (
                previous.repairs != plan.repairs
                or previous.resource_signature != plan.resource_signature
            ):
                raise _error("shared form has incompatible painting/resource contexts")
        self.plans[key] = plan

    def prove_programs(
        self, streams, resources, state, ancestors, description, resource_initial
    ) -> None:
        """Reuse successful glyph or pattern proofs under equivalent full contexts."""
        # Prove all glyph/cell programs harmless without guessing which glyph
        # or cell gets painted. A repair requires a specific invocation.
        streams = list(streams)
        context = (
            tuple(_object_key(stream) for stream in streams),
            resources.unparse(),
            _state_key(state),
            _state_key(resource_initial),
            ancestors,
        )
        if context in self.proofs:
            return
        proof = _Analyzer(self.budget)
        for stream in streams:
            proof.form(stream, resources, state, ancestors, resource_initial)
        if any(plan.repairs for plan in proof.plans.values()):
            raise _error(f"{description} needs invocation-specific overprint repair")
        for plan in proof.plans.values():
            self.record(plan)
        self.proofs.add(context)

    def pattern(self, selected: _Pattern | None, ancestors) -> None:
        """Evaluate a pattern using the initial state of its defining resource scope."""
        if selected is None or selected.definition is None:
            raise _error("pattern paint without a selected pattern")
        pattern = selected.definition
        resources = selected.resources
        # ISO 32000-1, 11.6.7: use the initial state of the content stream
        # defining the resource, not the state at its eventual painting site.
        # Pattern evaluation also resets the transparency parameters.
        pattern_state = replace(selected.initial, soft_mask=None)
        kind = pattern.get("/PatternType")
        if kind == 1 and isinstance(pattern, Stream):
            if pattern.get("/PaintType") == 2:
                pattern_state = replace(
                    pattern_state,
                    stroke=selected.base,
                    fill=selected.base,
                    stroke_pattern=None,
                    fill_pattern=None,
                )
            self.prove_programs(
                [pattern],
                resources,
                pattern_state,
                ancestors,
                "tiling pattern",
                selected.initial,
            )
        elif kind == 2:
            gs = pattern.get("/ExtGState")
            if isinstance(gs, Dictionary):
                pattern_state = _graphics_state(pattern_state, gs)
            shading = pattern.get("/Shading")
            if not isinstance(shading, (Dictionary, Stream)):
                raise _error("invalid shading pattern")
            kind = _colour_kind(shading.get("/ColorSpace"), resources)
            pattern_state = replace(pattern_state, stroke=kind, fill=kind)
            if _needs_repair(pattern_state, stroke=False, fill=True):
                raise _error(
                    "shading pattern needs invocation-specific overprint repair"
                )
        else:
            raise _error("unknown pattern type")

    def walk(
        self,
        owner,
        resources: Dictionary,
        initial: _State,
        ancestors: frozenset[tuple[int, int]] = frozenset(),
        resource_initial: _State | None = None,
    ) -> None:
        """Plan a stream once per full context, checking recursion before cache hits."""
        if resource_initial is None:
            resource_initial = initial
        key = owner.objgen
        if key in ancestors or len(ancestors) >= 64:
            raise _error("recursive or excessively nested painting stream")
        context = (
            key,
            resources.unparse(),
            _state_key(initial),
            _state_key(resource_initial),
            ancestors,
        )
        if context in self.contexts:
            return
        self.budget.calls += 1
        if self.calls > 100_000:
            raise _error("too many painting contexts")
        ancestors = ancestors | {key}
        try:
            instructions = self.instructions.get(key)
            if instructions is None:
                instructions = list(pikepdf.parse_content_stream(owner))
                self.instructions[key] = instructions
        except Exception as exc:
            raise _error("unreadable painting stream") from exc
        state = initial
        stack = []
        repairs = set()
        for index, instruction in enumerate(instructions):
            operands = instruction.operands
            op = str(instruction.operator)
            stroke = op in _STROKE
            fill = op in _FILL
            painted = state
            try:
                if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                    image = instruction.iimage.obj
                    kind = (
                        state.fill
                        if image.get("/ImageMask", False)
                        else _colour_kind(
                            _inline_space(image.get("/ColorSpace")), resources
                        )
                    )
                    painted = replace(state, fill=kind)
                    fill = True
                elif op == "q":
                    stack.append(state)
                elif op == "Q":
                    if not stack:
                        raise _error("unbalanced Q operator")
                    state = stack.pop()
                elif op == "gs":
                    gs = _resource(resources, "/ExtGState", operands[0])
                    if not isinstance(gs, Dictionary):
                        raise _error("invalid ExtGState dictionary")
                    state = _graphics_state(state, gs)
                elif op in {"CS", "cs"}:
                    kind = _colour_kind(operands[0], resources)
                    side = "stroke" if op == "CS" else "fill"
                    selected = (
                        _Pattern(
                            None,
                            _pattern_base(operands[0], resources),
                            resource_initial,
                            resources,
                        )
                        if kind == "pattern"
                        else None
                    )
                    state = replace(state, **{side: kind, f"{side}_pattern": selected})
                elif op in {"SCN", "scn"}:
                    side = "stroke" if op == "SCN" else "fill"
                    if getattr(state, side) == "pattern":
                        previous = getattr(state, f"{side}_pattern")
                        base = previous.base if previous is not None else "unknown"
                        pattern = _resource(resources, "/Pattern", operands[-1])
                        state = replace(
                            state,
                            **{
                                f"{side}_pattern": _Pattern(
                                    pattern, base, resource_initial, resources
                                )
                            },
                        )
                elif op in {"K", "k", "G", "g", "RG", "rg"}:
                    device = {
                        "k": Name.DeviceCMYK,
                        "g": Name.DeviceGray,
                        "rg": Name.DeviceRGB,
                    }
                    kind = _colour_kind(device[op.lower()], resources)
                    state = replace(
                        state,
                        **{
                            "stroke" if op.isupper() else "fill": kind,
                            "stroke_pattern" if op.isupper() else "fill_pattern": None,
                        },
                    )
                elif op == "Tr":
                    state = replace(state, text_mode=int(operands[0]))
                elif op == "Tf":
                    font = _resource(resources, "/Font", operands[0])
                    state = replace(
                        state, type3=font.get("/Subtype") == Name.Type3, font=font
                    )
                elif op in _TEXT:
                    if state.type3:
                        # ISO 32000-1, 9.3.6: only mode 3 suppresses Type3 paint.
                        if state.text_mode == 3:
                            continue
                        charprocs = state.font.get("/CharProcs")
                        if not isinstance(charprocs, Dictionary):
                            raise _error("invalid Type3 CharProcs")
                        self.prove_programs(
                            charprocs.values(),
                            state.font.get("/Resources", resources),
                            state,
                            ancestors,
                            "Type3 glyph",
                            state if "/Resources" in state.font else resource_initial,
                        )
                    else:
                        stroke = state.text_mode in {1, 2, 5, 6}
                        fill = state.text_mode in {0, 2, 4, 6}
                elif op == "Do":
                    xobject = _resource(resources, "/XObject", operands[0])
                    if xobject.get("/Subtype") == Name.Form:
                        self.form(
                            xobject, resources, state, ancestors, resource_initial
                        )
                    elif xobject.get("/Subtype") == Name.Image:
                        kind = (
                            state.fill
                            if xobject.get("/ImageMask", False)
                            else _colour_kind(xobject.get("/ColorSpace"), resources)
                        )
                        painted = replace(state, fill=kind)
                        fill = True
                elif op == "sh":
                    shading = _resource(resources, "/Shading", operands[0])
                    painted = replace(
                        state, fill=_colour_kind(shading.get("/ColorSpace"), resources)
                    )
                    fill = True
            except (IndexError, TypeError, ValueError, AttributeError) as exc:
                raise _error(f"malformed {op} instruction") from exc
            for side, used in (("stroke", stroke), ("fill", fill)):
                if used and getattr(painted, side) == "pattern":
                    self.pattern(getattr(state, f"{side}_pattern"), ancestors)
                    painted = replace(painted, **{side: "other"})
            repair = _needs_repair(painted, stroke=stroke, fill=fill)
            if (stroke or fill) and state.soft_mask is not None:
                # A selected mask is evaluated for the actual painting state,
                # not when an earlier gs instruction happened to select it.
                mask_state = replace(
                    state, soft_mask=None, mode=0 if repair else state.mode
                )
                self.form(
                    state.soft_mask, resources, mask_state, ancestors, resource_initial
                )
            if repair:
                repairs.add(index)
        if stack:
            raise _error("unbalanced q operator")
        plan = _Plan(
            owner, resources, instructions, frozenset(repairs), resources.unparse()
        )
        self.record(plan)
        self.contexts.add(context)

    def form(self, stream, inherited, state, ancestors, resource_initial=None) -> None:
        """Enter a Form while preserving the initial state of inherited resources."""
        if not isinstance(stream, Stream):
            raise _error("invalid Form XObject")
        resources = stream.get("/Resources", inherited)
        if not isinstance(resources, Dictionary):
            raise _error("invalid form resources")
        if "/Resources" in stream:
            resource_initial = state
        self.walk(stream, resources, state, ancestors, resource_initial)


def sanitize_overprint(pdf: Pdf) -> int:
    """Plan all context-sensitive repairs before changing any content streams.

    Ambiguous shared forms and patterns/Type3 glyphs needing invocation-specific
    repair fail explicitly instead of applying a global change. The converter's
    original-input fallback then preserves the input. Provably harmless programs,
    unmodified streams and shared ExtGStates stay intact.
    """
    analyzer = _Analyzer()
    for page in pdf.pages:
        resources = get_page_resources(page)
        if resources is None:
            resources = Dictionary()
        if "/Contents" in page.obj:
            analyzer.walk(page, resources, _State())
        for annotation in page.get("/Annots", []):
            appearances = annotation.get("/AP")
            if not isinstance(appearances, Dictionary):
                continue
            for appearance in appearances.values():
                streams = (
                    appearance.values()
                    if isinstance(appearance, Dictionary)
                    else [appearance]
                )
                for stream in streams:
                    if isinstance(stream, Stream):
                        analyzer.form(stream, resources, _State(), frozenset())

    fixed = 0
    for plan in analyzer.plans.values():
        if not plan.repairs:
            continue
        resources = Dictionary(plan.resources)
        graphics = Dictionary(resources.get("/ExtGState", Dictionary()))
        names = []
        for mode in (0, 1):
            name = f"/PdfaOverprint{mode}"
            while name in graphics:
                name += "_"
            graphics[name] = Dictionary(OPM=mode)
            names.append(Name(name))
        resources.ExtGState = graphics
        rewritten = []
        for index, instruction in enumerate(plan.instructions):
            if index in plan.repairs:
                rewritten.append(
                    pikepdf.ContentStreamInstruction([names[0]], pikepdf.Operator("gs"))
                )
            rewritten.append(instruction)
            if index in plan.repairs:
                rewritten.append(
                    pikepdf.ContentStreamInstruction([names[1]], pikepdf.Operator("gs"))
                )
        data = pikepdf.unparse_content_stream(rewritten)
        if isinstance(plan.owner, pikepdf.Page):
            plan.owner.obj.Resources = resources
            plan.owner.obj.Contents = pdf.make_stream(data)
        else:
            plan.owner.Resources = resources
            plan.owner.write(data)
        fixed += len(plan.repairs)
    return fixed
