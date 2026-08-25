# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Optional Content (Layers) sanitization for PDF/A-2/3 compliance."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pikepdf import Array, Dictionary, Name, Pdf, Stream, String

from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

_MAX_OPTIONAL_CONTENT_ITEMS = 8_191
_MAX_OPTIONAL_CONTENT_DEPTH = 64


def _resource_budget(pdf: Pdf) -> int:
    """Return the resource-traversal ceiling for one document.

    The traversal deduplicates by object identity, so it cannot visit more
    resource dictionaries than the document has indirect objects. Scaling the
    ceiling with the document keeps the malformed-input guard while leaving
    large but well-formed files - which carry one ``/Resources`` dictionary per
    page - convertible.
    """
    try:
        object_count = len(pdf.objects)
    except Exception:
        object_count = 0
    return max(_MAX_OPTIONAL_CONTENT_ITEMS, object_count)


class _ObjectIdentities:
    """Return collision-free identities while direct wrappers are in use."""

    def __init__(self) -> None:
        self._direct_objects: dict[int, object] = {}

    def key(self, value: object) -> tuple[object, ...]:
        try:
            objgen = value.objgen
        except Exception:
            objgen = (0, 0)
        if objgen != (0, 0):
            return ("indirect", *objgen)
        object_id = id(value)
        self._direct_objects.setdefault(object_id, value)
        return ("direct", object_id)

    @staticmethod
    def indirect_key(value: object) -> tuple[object, ...] | None:
        try:
            objgen = value.objgen
        except Exception:
            return None
        if objgen == (0, 0):
            return None
        return ("indirect", *objgen)


def _intent_names(
    value: object,
    *,
    default_view: bool,
    consume: Callable[[], None] | None = None,
) -> frozenset[str]:
    """Return validated optional-content intent names."""
    value = _resolve_indirect(value)
    if value is None:
        return frozenset({str(Name.View)}) if default_view else frozenset()
    if isinstance(value, Name):
        if consume is not None:
            consume()
        return frozenset({str(value)})
    if not isinstance(value, Array) or len(value) > _MAX_OPTIONAL_CONTENT_ITEMS:
        raise ValueError("Optional-content Intent is malformed")
    names: set[str] = set()
    for item in value:
        if consume is not None:
            consume()
        item = _resolve_indirect(item)
        if not isinstance(item, Name):
            raise ValueError("Optional-content Intent is malformed")
        names.add(str(item))
    return frozenset(names)


def _intent_matches(configuration: frozenset[str], group: frozenset[str]) -> bool:
    all_intent = str(Name.All)
    return (
        (all_intent in configuration and bool(group))
        or (all_intent in group and bool(configuration))
        or bool(configuration & group)
    )


class _DefaultOCVisibility:
    """Evaluate OCG/OCMD visibility in the document's default configuration.

    The evaluator is intentionally read-only. It expects the PDF/A sanitizer's
    default configuration invariant (missing ``/BaseState`` or ``/ON``) and
    raises ``ValueError`` for malformed, unregistered, cyclic, or excessively
    complex optional-content data. If the catalog has no ``/OCProperties``, PDF
    consumers ignore optional-content controls, so every membership is visible.

    ``is_visible()`` is the internal API used by semantic tagging. It accepts an
    OCG or OCMD dictionary (direct or indirect), implements all four ``/P``
    policies, and gives a valid ``/VE`` expression precedence over ``/P``.
    """

    def __init__(self, pdf: Pdf):
        self._enabled = False
        self._identities = _ObjectIdentities()
        self._states: dict[tuple[object, ...], bool] = {}
        self._active_groups: set[tuple[object, ...]] = set()
        self._memberships: dict[tuple[object, ...], bool] = {}
        self._expressions: dict[tuple[object, ...], bool | None] = {}
        self._work = 0

        raw_properties = pdf.Root.get("/OCProperties")
        if raw_properties is None:
            return
        properties = _resolve_indirect(raw_properties)
        if not isinstance(properties, Dictionary):
            raise ValueError("Optional-content properties are malformed")
        ocgs = _resolve_indirect(properties.get("/OCGs"))
        default = _resolve_indirect(properties.get("/D"))
        if not isinstance(ocgs, Array) or not isinstance(default, Dictionary):
            raise ValueError("Optional-content default configuration is malformed")
        if len(ocgs) > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError("Optional-content group limit exceeded")

        base_state = _resolve_indirect(default.get("/BaseState", Name.ON))
        if base_state != Name.ON:
            raise ValueError("Optional-content default BaseState is not /ON")
        configuration_intents = _intent_names(
            default.get("/Intent"),
            default_view=True,
            consume=self._consume,
        )

        for value in ocgs:
            ocg = _resolve_indirect(value)
            if not isinstance(ocg, Dictionary) or (
                _resolve_indirect(ocg.get("/Type")) != Name.OCG
            ):
                raise ValueError("Optional-content group is malformed")
            key = self._identities.key(ocg)
            if key[0] != "indirect":
                raise ValueError("Optional-content group is not indirect")
            self._states[key] = True
            group_intents = _intent_names(
                ocg.get("/Intent"),
                default_view=True,
                consume=self._consume,
            )
            if _intent_matches(configuration_intents, group_intents):
                self._active_groups.add(key)

        on = self._configuration_groups(default, "/ON")
        off = self._configuration_groups(default, "/OFF")
        # With /BaseState /ON, /ON is redundant and /OFF is authoritative.
        # Still validate /ON so malformed or unregistered references fail closed.
        del on
        for key in off:
            self._states[key] = False
        self._enabled = True

    def _configuration_groups(
        self,
        configuration: Dictionary,
        name: str,
    ) -> set[tuple[object, ...]]:
        raw_values = configuration.get(name)
        if raw_values is None:
            return set()
        values = _resolve_indirect(raw_values)
        if not isinstance(values, Array) or len(values) > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError(f"Optional-content configuration {name} is malformed")
        return {self._ocg_key(value) for value in values}

    def _ocg_key(self, value: object) -> tuple[object, ...]:
        ocg = _resolve_indirect(value)
        if not isinstance(ocg, Dictionary) or (
            _resolve_indirect(ocg.get("/Type")) != Name.OCG
        ):
            raise ValueError("Optional-content membership contains a malformed OCG")
        key = self._identities.key(ocg)
        if key not in self._states:
            raise ValueError("Optional-content membership contains an unregistered OCG")
        return key

    def _ocg_state(self, value: object) -> bool | None:
        key = self._ocg_key(value)
        if key not in self._active_groups:
            return None
        return self._states[key]

    def _consume(self) -> None:
        self._work += 1
        if self._work > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError("Optional-content visibility evaluation limit exceeded")

    def _evaluate_expression(
        self,
        value: object,
        *,
        depth: int,
        active: set[tuple[object, ...]],
    ) -> bool | None:
        value = _resolve_indirect(value)
        cache_key = self._identities.indirect_key(value)
        if cache_key is not None and cache_key in self._expressions:
            return self._expressions[cache_key]

        self._consume()
        if isinstance(value, Dictionary):
            return self._ocg_state(value)
        if not isinstance(value, Array):
            raise ValueError("Optional-content visibility expression is malformed")
        if len(value) > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError("Optional-content visibility expression limit exceeded")
        if depth >= _MAX_OPTIONAL_CONTENT_DEPTH:
            raise ValueError(
                "Optional-content visibility expression is too deeply nested"
            )

        key = self._identities.key(value)
        if key in active:
            raise ValueError("Optional-content visibility expression is cyclic")
        if not value:
            raise ValueError("Optional-content visibility expression is empty")
        operator = _resolve_indirect(value[0])
        operands = list(value)[1:]
        if operator == Name.Not:
            if len(operands) != 1:
                raise ValueError("Optional-content /Not expression is malformed")
        elif operator not in {Name.And, Name.Or} or not operands:
            raise ValueError("Optional-content visibility operator is malformed")

        active.add(key)
        try:
            results = [
                self._evaluate_expression(
                    operand,
                    depth=depth + 1,
                    active=active,
                )
                for operand in operands
            ]
        finally:
            active.remove(key)
        effective_results = [result for result in results if result is not None]
        if not effective_results:
            result = None
        elif operator == Name.Not:
            result = not effective_results[0]
        elif operator == Name.And:
            result = all(effective_results)
        else:
            result = any(effective_results)
        if cache_key is not None:
            self._expressions[cache_key] = result
        return result

    def _evaluate_membership(self, membership: Dictionary) -> bool:
        if "/VE" in membership:
            expression = _resolve_indirect(membership.get("/VE"))
            if not isinstance(expression, Array):
                raise ValueError("Optional-content /VE entry is malformed")
            result = self._evaluate_expression(
                expression,
                depth=0,
                active=set(),
            )
            return True if result is None else result

        policy = _resolve_indirect(membership.get("/P", Name.AnyOn))
        if policy not in {Name.AllOn, Name.AnyOn, Name.AllOff, Name.AnyOff}:
            raise ValueError("Optional-content visibility policy is malformed")
        if "/OCGs" not in membership:
            return True
        raw_ocgs = _resolve_indirect(membership.get("/OCGs"))
        if raw_ocgs is None:
            values: list[object] = []
        elif isinstance(raw_ocgs, Dictionary):
            self._consume()
            values = [raw_ocgs]
        elif isinstance(raw_ocgs, Array):
            if len(raw_ocgs) > _MAX_OPTIONAL_CONTENT_ITEMS:
                raise ValueError("Optional-content membership limit exceeded")
            values = []
            for value in raw_ocgs:
                self._consume()
                if _resolve_indirect(value) is not None:
                    values.append(value)
        else:
            raise ValueError("Optional-content membership OCGs are malformed")
        if not values:
            return True
        states = [
            state for value in values if (state := self._ocg_state(value)) is not None
        ]
        if not states:
            return True
        if policy == Name.AllOn:
            return all(states)
        if policy == Name.AnyOn:
            return any(states)
        if policy == Name.AllOff:
            return not any(states)
        return not all(states)

    def is_visible(self, membership: object) -> bool:
        """Return default visibility for one OCG or OCMD membership."""
        if not self._enabled:
            return True
        membership = _resolve_indirect(membership)
        if not isinstance(membership, Dictionary):
            raise ValueError("Optional-content membership is malformed")
        kind = _resolve_indirect(membership.get("/Type"))
        if kind == Name.OCG:
            state = self._ocg_state(membership)
            return True if state is None else state
        if kind != Name.OCMD:
            raise ValueError("Optional-content membership type is malformed")
        key = self._identities.indirect_key(membership)
        if key is not None and key in self._memberships:
            return self._memberships[key]
        self._consume()
        visible = self._evaluate_membership(membership)
        if key is not None:
            self._memberships[key] = visible
        return visible


def _default_optional_content_visibility(pdf: Pdf) -> _DefaultOCVisibility:
    """Return a read-only evaluator for the sanitized default OC configuration."""
    return _DefaultOCVisibility(pdf)


class _OptionalContentCollector:
    """Validate and collect optional-content references without mutation."""

    def __init__(self, *, resource_limit: int = _MAX_OPTIONAL_CONTENT_ITEMS) -> None:
        self._resource_limit = resource_limit
        self.identities = _ObjectIdentities()
        self.ocgs: dict[tuple[object, ...], Dictionary] = {}
        self.has_controls = False
        self._memberships: set[tuple[object, ...]] = set()
        self._expressions: set[tuple[object, ...]] = set()
        self._resources: set[tuple[object, ...]] = set()
        self._work = 0

    def _consume(self) -> None:
        self._work += 1
        if self._work > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError("Optional-content collection limit exceeded")

    def add_ocg(self, value: object) -> tuple[object, ...]:
        self._consume()
        ocg = _resolve_indirect(value)
        if not isinstance(ocg, Dictionary) or (
            _resolve_indirect(ocg.get("/Type")) != Name.OCG
        ):
            raise ValueError("Optional-content group is malformed")
        key = self.identities.key(ocg)
        if key[0] != "indirect":
            raise ValueError("Optional-content group is not indirect")
        self.ocgs.setdefault(key, ocg)
        return key

    def collect_membership(self, value: object, *, is_control: bool) -> None:
        self._consume()
        membership = _resolve_indirect(value)
        if is_control:
            self.has_controls = True
        if not isinstance(membership, Dictionary):
            raise ValueError("Optional-content membership is malformed")
        kind = _resolve_indirect(membership.get("/Type"))
        if kind == Name.OCG:
            self.add_ocg(membership)
            return
        if kind != Name.OCMD:
            raise ValueError("Optional-content membership type is malformed")

        key = self.identities.key(membership)
        if key in self._memberships:
            return
        self._memberships.add(key)

        policy = _resolve_indirect(membership.get("/P", Name.AnyOn))
        if policy not in {Name.AllOn, Name.AnyOn, Name.AllOff, Name.AnyOff}:
            raise ValueError("Optional-content visibility policy is malformed")

        if "/OCGs" in membership:
            raw_ocgs = _resolve_indirect(membership.get("/OCGs"))
            if raw_ocgs is None:
                values: list[object] = []
            elif isinstance(raw_ocgs, Dictionary):
                self._consume()
                values = [raw_ocgs]
            elif isinstance(raw_ocgs, Array):
                if len(raw_ocgs) > _MAX_OPTIONAL_CONTENT_ITEMS:
                    raise ValueError("Optional-content membership limit exceeded")
                values = []
                for item in raw_ocgs:
                    self._consume()
                    if _resolve_indirect(item) is not None:
                        values.append(item)
            else:
                raise ValueError("Optional-content membership OCGs are malformed")
            for ocg in values:
                self.add_ocg(ocg)

        if "/VE" in membership:
            self.collect_expression(
                membership.get("/VE"),
                depth=0,
                active=set(),
            )

    def collect_expression(
        self,
        value: object,
        *,
        depth: int,
        active: set[tuple[object, ...]],
    ) -> None:
        self._consume()
        value = _resolve_indirect(value)
        if isinstance(value, Dictionary):
            self.add_ocg(value)
            return
        if not isinstance(value, Array):
            raise ValueError("Optional-content visibility expression is malformed")
        if len(value) > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError("Optional-content visibility expression limit exceeded")
        if depth >= _MAX_OPTIONAL_CONTENT_DEPTH:
            raise ValueError(
                "Optional-content visibility expression is too deeply nested"
            )
        key = self.identities.key(value)
        if key in active:
            raise ValueError("Optional-content visibility expression is cyclic")
        if key in self._expressions:
            return
        if not value:
            raise ValueError("Optional-content visibility expression is empty")
        operator = _resolve_indirect(value[0])
        operands = list(value)[1:]
        if operator == Name.Not:
            if len(operands) != 1:
                raise ValueError("Optional-content /Not expression is malformed")
        elif operator not in {Name.And, Name.Or} or not operands:
            raise ValueError("Optional-content visibility operator is malformed")
        active.add(key)
        try:
            for operand in operands:
                self.collect_expression(
                    operand,
                    depth=depth + 1,
                    active=active,
                )
        finally:
            active.remove(key)
        self._expressions.add(key)

    def collect_ocg_array(self, value: object, *, name: str) -> set[tuple[object, ...]]:
        values = _resolve_indirect(value)
        if not isinstance(values, Array) or len(values) > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError(f"Optional-content configuration {name} is malformed")
        return {self.add_ocg(item) for item in values}

    def collect_order(self, value: object) -> set[tuple[object, ...]]:
        present: set[tuple[object, ...]] = set()
        completed: set[tuple[object, ...]] = set()
        active: set[tuple[object, ...]] = set()
        pending: list[tuple[object, bool]] = [(value, False)]
        while pending:
            current, exiting = pending.pop()
            current = _resolve_indirect(current)
            if not isinstance(current, Array):
                raise ValueError("Optional-content Order is malformed")
            key = self.identities.key(current)
            if exiting:
                active.remove(key)
                completed.add(key)
                continue
            self._consume()
            if len(current) > _MAX_OPTIONAL_CONTENT_ITEMS:
                raise ValueError("Optional-content Order limit exceeded")
            if key in active:
                raise ValueError("Optional-content Order is cyclic")
            if key in completed:
                continue
            active.add(key)
            pending.append((current, True))
            for item in reversed(list(current)):
                self._consume()
                resolved = _resolve_indirect(item)
                if isinstance(resolved, Array):
                    pending.append((resolved, False))
                elif isinstance(resolved, Dictionary):
                    present.add(self.add_ocg(resolved))
                elif not isinstance(resolved, String):
                    raise ValueError("Optional-content Order is malformed")
        return present

    def collect_rbgroups(self, value: object) -> None:
        groups = _resolve_indirect(value)
        if not isinstance(groups, Array) or len(groups) > _MAX_OPTIONAL_CONTENT_ITEMS:
            raise ValueError("Optional-content RBGroups is malformed")
        for group in groups:
            self.collect_ocg_array(group, name="/RBGroups")

    def enqueue_resources(
        self,
        value: object,
        pending: list[Dictionary],
    ) -> None:
        resources = _resolve_indirect(value)
        if not isinstance(resources, Dictionary):
            raise ValueError("Optional-content resources are malformed")
        key = self.identities.key(resources)
        if key in self._resources:
            return
        self._resources.add(key)
        if len(self._resources) > self._resource_limit:
            raise ValueError("Optional-content resource collection limit exceeded")
        pending.append(resources)


@dataclass
class _OrderUpdate:
    config: Dictionary
    create: bool
    ocgs: list[Dictionary]


@dataclass
class _OptionalContentPlan:
    properties: Dictionary
    default: Dictionary
    default_created: bool
    ocgs_missing: bool
    all_ocgs: list[Dictionary]
    missing_ocgs: list[Dictionary]
    as_configs: list[Dictionary]
    config_name_updates: list[tuple[Dictionary, str]]
    d_name_changed: bool
    ocg_name_updates: list[Dictionary]
    normalize_default_base: bool
    normalize_default_intent: bool
    default_hidden: list[Dictionary]
    order_updates: list[_OrderUpdate]


def sanitize_optional_content(pdf: Pdf) -> dict:
    """Sanitize optional content after a complete preflight.

    The preflight validates the whole optional-content graph before any repair
    is written, so either every planned repair is applied or the document keeps
    its original optional-content state. Its only write is the temporary alias
    probe of :func:`_validate_configuration_aliases`, which always removes the
    keys it sets. Design intents, ``/ListMode`` and ``/RBGroups`` are preserved
    as authored; they are only validated, never rewritten.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with keys:
        - as_entries_removed: Number of /AS entries removed from configs
        - ocgs_processed: Total number of OCGs processed
        - d_created: Whether a default configuration was created
        - d_name_added: Whether the default configuration gained a /Name
        - base_state_fixed: 1 if the default /BaseState was normalized to /ON
        - default_intent_fixed: 1 if a pure /View intent array was canonicalized
        - config_names_added: Number of alternate configs given a unique /Name
        - missing_ocgs_added: Number of OCGs added to /OCProperties/OCGs
        - ocg_names_added: Number of OCGs given a placeholder /Name
        - order_ocgs_added: Number of OCGs added to an /Order array

    Raises:
        ValueError: The optional-content graph is malformed or cannot be made
            PDF/A compliant without changing the document's appearance.
    """
    result = {
        "as_entries_removed": 0,
        "ocgs_processed": 0,
        "d_created": False,
        "d_name_added": False,
        "base_state_fixed": 0,
        "default_intent_fixed": 0,
        "config_names_added": 0,
        "missing_ocgs_added": 0,
        "ocg_names_added": 0,
        "order_ocgs_added": 0,
    }
    plan = _preflight_optional_content(pdf)
    if plan is None:
        return result

    if plan.ocgs_missing:
        plan.properties["/OCGs"] = Array(plan.all_ocgs)
    else:
        ocgs = _resolve_indirect(plan.properties.get("/OCGs"))
        for ocg in plan.missing_ocgs:
            ocgs.append(ocg)

    if plan.default_created:
        plan.properties["/D"] = plan.default

    if plan.normalize_default_base:
        plan.default["/BaseState"] = Name.ON
        if "/ON" in plan.default:
            del plan.default["/ON"]
        if plan.default_hidden:
            plan.default["/OFF"] = Array(plan.default_hidden)
        elif "/OFF" in plan.default:
            del plan.default["/OFF"]

    if plan.normalize_default_intent:
        plan.default["/Intent"] = Name.View

    for config in plan.as_configs:
        del config["/AS"]
    for config, name in plan.config_name_updates:
        config["/Name"] = name
    for ocg in plan.ocg_name_updates:
        ocg["/Name"] = "Unnamed OCG"
    for update in plan.order_updates:
        if update.create:
            update.config["/Order"] = Array(update.ocgs)
        else:
            order = _resolve_indirect(update.config.get("/Order"))
            for ocg in update.ocgs:
                order.append(ocg)

    result.update(
        {
            "as_entries_removed": len(plan.as_configs),
            "ocgs_processed": len(plan.all_ocgs),
            "d_created": plan.default_created,
            "d_name_added": plan.d_name_changed,
            "base_state_fixed": int(plan.normalize_default_base),
            "default_intent_fixed": int(plan.normalize_default_intent),
            "config_names_added": len(plan.config_name_updates)
            - int(plan.d_name_changed),
            "missing_ocgs_added": len(plan.missing_ocgs),
            "ocg_names_added": len(plan.ocg_name_updates),
            "order_ocgs_added": sum(len(update.ocgs) for update in plan.order_updates),
        }
    )

    changes = (
        result["as_entries_removed"]
        + result["base_state_fixed"]
        + result["default_intent_fixed"]
        + result["config_names_added"]
        + result["missing_ocgs_added"]
        + result["ocg_names_added"]
        + result["order_ocgs_added"]
        + int(result["d_name_added"])
        + int(result["d_created"])
    )
    if changes > 0:
        logger.info(
            "Optional content sanitized: %d AS removed, "
            "%d BaseState fixed, %d default Intent fixed, "
            "%d config names added, "
            "%d missing OCGs added, "
            "%d OCG names added, %d Order OCGs added, "
            "D created: %s, D /Name added: %s",
            result["as_entries_removed"],
            result["base_state_fixed"],
            result["default_intent_fixed"],
            result["config_names_added"],
            result["missing_ocgs_added"],
            result["ocg_names_added"],
            result["order_ocgs_added"],
            result["d_created"],
            result["d_name_added"],
        )

    return result


def _read_config_name(config) -> str:
    """Return a normalized OC config name (empty string when missing/invalid)."""
    if "/Name" not in config:
        return ""
    try:
        return str(config["/Name"]).strip()
    except Exception:
        return ""


def _make_unique_name(base: str, used: set[str]) -> str:
    """Generate a unique config name not present in *used*."""
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


@dataclass
class _ConfigurationInfo:
    config: Dictionary
    on: set[tuple[object, ...]]
    off: set[tuple[object, ...]]
    order: set[tuple[object, ...]] | None
    intents: frozenset[str]


def _plan_config_names(
    default: Dictionary,
    configs: list[Dictionary],
) -> tuple[list[tuple[Dictionary, str]], bool]:
    entries = [("D", -1, default), *[("Config", i, c) for i, c in enumerate(configs)]]
    used_names: set[str] = set()
    updates: list[tuple[Dictionary, str]] = []
    d_changed = False
    for kind, index, config in entries:
        current_name = _read_config_name(config)
        fallback_name = "Default" if kind == "D" else f"Config{index}"
        unique_name = _make_unique_name(current_name or fallback_name, used_names)
        used_names.add(unique_name)
        if unique_name != current_name:
            updates.append((config, unique_name))
            d_changed = d_changed or kind == "D"
    return updates, d_changed


def _probe_key(configs: list[Dictionary]) -> Name:
    """Return a dictionary key that is absent from every configuration."""
    for attempt in range(len(configs) + 1):
        candidate = Name(f"/PdftopdfaAliasProbe{attempt}")
        if all(candidate not in config for config in configs):
            return candidate
    raise ValueError("Optional-content configuration is malformed")


def _validate_configuration_aliases(
    default: Dictionary,
    configs: list[Dictionary],
) -> None:
    """Reject configurations that share one underlying dictionary.

    Aliased configurations cannot be renamed apart: writing ``/Name`` to one of
    them overwrites the other, so the required uniqueness could never hold.

    Indirect configurations are compared by object number. Direct dictionaries
    have no object number and pikepdf hands out a fresh wrapper on every access,
    so identity is established by writing a distinct probe value into each one
    and reading it back: aliased dictionaries report the same value. Comparing
    them by value instead would reject two independent but identical
    configurations, which are renameable and therefore repairable.
    """
    indirect: set[tuple[object, ...]] = set()
    direct: list[Dictionary] = []
    for config in [default, *configs]:
        key = _ObjectIdentities.indirect_key(config)
        if key is None:
            direct.append(config)
            continue
        if key in indirect:
            raise ValueError("Optional-content configuration is aliased")
        indirect.add(key)

    if len(direct) < 2:
        return
    probe = _probe_key(direct)
    try:
        for index, config in enumerate(direct):
            config[probe] = index
        for index, config in enumerate(direct):
            if config[probe] != index:
                raise ValueError("Optional-content configuration is aliased")
    finally:
        for config in direct:
            if probe in config:
                del config[probe]


def _validate_configuration(
    collector: _OptionalContentCollector,
    config: Dictionary,
    *,
    is_default: bool,
) -> _ConfigurationInfo:
    base_state = _resolve_indirect(config.get("/BaseState", Name.ON))
    if base_state not in {Name.ON, Name.OFF, Name.Unchanged}:
        raise ValueError("Optional-content BaseState is malformed")
    if is_default and base_state == Name.Unchanged:
        raise ValueError(
            "Optional-content default BaseState /Unchanged cannot be normalized safely"
        )

    # ISO 19005-2, 6.9 places no intent restriction on configuration
    # dictionaries, so /Design and mixed intents are validated and preserved.
    intents = _intent_names(
        config.get("/Intent"),
        default_view=True,
        consume=collector._consume,
    )

    if "/ListMode" in config:
        list_mode = _resolve_indirect(config.get("/ListMode"))
        if list_mode not in {Name.AllPages, Name.VisiblePages}:
            raise ValueError("Optional-content ListMode is malformed")

    on = (
        collector.collect_ocg_array(config.get("/ON"), name="/ON")
        if "/ON" in config
        else set()
    )
    off = (
        collector.collect_ocg_array(config.get("/OFF"), name="/OFF")
        if "/OFF" in config
        else set()
    )
    if "/Locked" in config:
        collector.collect_ocg_array(config.get("/Locked"), name="/Locked")
    if "/RBGroups" in config:
        collector.collect_rbgroups(config.get("/RBGroups"))
    order = (
        collector.collect_order(config.get("/Order")) if "/Order" in config else None
    )
    return _ConfigurationInfo(
        config=config,
        on=on,
        off=off,
        order=order,
        intents=intents,
    )


def _scan_optional_content(pdf: Pdf, collector: _OptionalContentCollector) -> None:
    pending_resources: list[Dictionary] = []
    scanned_owners: set[tuple[object, ...]] = set()

    def enqueue_resources(value: object) -> None:
        resources = _resolve_indirect(value)
        if isinstance(resources, Dictionary):
            collector.enqueue_resources(resources, pending_resources)

    def scan_owner(value: object) -> None:
        owner = _resolve_indirect(value)
        if not isinstance(owner, (Dictionary, Stream)):
            return
        key = collector.identities.key(owner)
        if key in scanned_owners:
            return
        scanned_owners.add(key)
        if "/OC" in owner:
            collector.collect_membership(owner.get("/OC"), is_control=True)
        if "/Resources" in owner:
            enqueue_resources(owner.get("/Resources"))

    def scan_appearance(value: object) -> None:
        appearance = _resolve_indirect(value)
        if isinstance(appearance, Stream):
            scan_owner(appearance)
        elif isinstance(appearance, Dictionary):
            for item in appearance.values():
                stream = _resolve_indirect(item)
                if isinstance(stream, Stream):
                    scan_owner(stream)

    def inherited_page_resources(page_dict: Dictionary) -> Dictionary | None:
        current: object = page_dict
        visited: set[tuple[object, ...]] = set()
        while isinstance(current := _resolve_indirect(current), Dictionary):
            key = collector.identities.key(current)
            if key in visited:
                raise ValueError("Optional-content page resource inheritance is cyclic")
            visited.add(key)
            resources = _resolve_indirect(current.get("/Resources"))
            if isinstance(resources, Dictionary):
                return resources
            current = current.get("/Parent")
        return None

    for page in pdf.pages:
        page_dict = page.obj
        resources = inherited_page_resources(page_dict)
        if resources is not None:
            enqueue_resources(resources)
        annotations = _resolve_indirect(page_dict.get("/Annots"))
        if not isinstance(annotations, Array):
            continue
        for value in annotations:
            annotation = _resolve_indirect(value)
            if not isinstance(annotation, Dictionary):
                continue
            scan_owner(annotation)
            appearances = _resolve_indirect(annotation.get("/AP"))
            if isinstance(appearances, Dictionary):
                for appearance in appearances.values():
                    scan_appearance(appearance)

    for item in pdf.objects:
        scan_owner(item)

    while pending_resources:
        resources = pending_resources.pop()
        properties = _resolve_indirect(resources.get("/Properties"))
        if isinstance(properties, Dictionary):
            for value in properties.values():
                membership = _resolve_indirect(value)
                if isinstance(membership, Dictionary) and _resolve_indirect(
                    membership.get("/Type")
                ) in {Name.OCG, Name.OCMD}:
                    collector.collect_membership(membership, is_control=True)
        for resource_name in ("/XObject", "/Pattern"):
            objects = _resolve_indirect(resources.get(resource_name))
            if not isinstance(objects, Dictionary):
                continue
            for value in objects.values():
                resource = _resolve_indirect(value)
                if isinstance(resource, (Dictionary, Stream)):
                    scan_owner(resource)


def _preflight_optional_content(pdf: Pdf) -> _OptionalContentPlan | None:
    collector = _OptionalContentCollector(resource_limit=_resource_budget(pdf))
    raw_properties = pdf.Root.get("/OCProperties")
    if raw_properties is None:
        _scan_optional_content(pdf, collector)
        if collector.has_controls:
            raise ValueError(
                "Optional-content controls require a catalog /OCProperties entry"
            )
        return None

    properties = _resolve_indirect(raw_properties)
    if not isinstance(properties, Dictionary):
        raise ValueError("Optional-content properties are malformed")

    ocgs_missing = "/OCGs" not in properties
    original_keys: set[tuple[object, ...]] = set()
    if not ocgs_missing:
        original_ocgs = _resolve_indirect(properties.get("/OCGs"))
        if (
            not isinstance(original_ocgs, Array)
            or len(original_ocgs) > _MAX_OPTIONAL_CONTENT_ITEMS
        ):
            raise ValueError("Optional-content group array is malformed")
        for ocg in original_ocgs:
            original_keys.add(collector.add_ocg(ocg))

    default_created = "/D" not in properties
    if default_created:
        default = Dictionary(Name="Default", BaseState=Name.ON)
    else:
        default = _resolve_indirect(properties.get("/D"))
        if not isinstance(default, Dictionary):
            raise ValueError("Optional-content default configuration is malformed")

    configs: list[Dictionary] = []
    if "/Configs" in properties:
        raw_configs = _resolve_indirect(properties.get("/Configs"))
        if (
            not isinstance(raw_configs, Array)
            or len(raw_configs) > _MAX_OPTIONAL_CONTENT_ITEMS
        ):
            raise ValueError("Optional-content alternate configurations are malformed")
        for value in raw_configs:
            config = _resolve_indirect(value)
            if not isinstance(config, Dictionary):
                raise ValueError(
                    "Optional-content alternate configuration is malformed"
                )
            configs.append(config)

    _validate_configuration_aliases(default, configs)

    _scan_optional_content(pdf, collector)
    default_info = _validate_configuration(collector, default, is_default=True)
    config_infos = [
        _validate_configuration(collector, config, is_default=False)
        for config in configs
    ]

    all_items = list(collector.ocgs.items())
    all_ocgs = [ocg for _key, ocg in all_items]
    missing_ocgs = [ocg for key, ocg in all_items if key not in original_keys]

    ocg_name_updates: list[Dictionary] = []
    for ocg in all_ocgs:
        _intent_names(
            ocg.get("/Intent"),
            default_view=True,
            consume=collector._consume,
        )
        try:
            name = str(ocg.get("/Name", "")).strip()
        except Exception:
            name = ""
        if not name:
            ocg_name_updates.append(ocg)

    base_state = _resolve_indirect(default.get("/BaseState", Name.ON))
    normalize_default_base = base_state == Name.OFF
    # Only a pure /View array is canonicalized to the /View name; any other
    # intent array is kept as authored.
    normalize_default_intent = isinstance(
        _resolve_indirect(default.get("/Intent")), Array
    ) and default_info.intents == frozenset({str(Name.View)})
    default_hidden = (
        [
            ocg
            for key, ocg in all_items
            if key not in default_info.on or key in default_info.off
        ]
        if normalize_default_base
        else []
    )

    name_updates, d_name_changed = _plan_config_names(default, configs)
    as_configs = [config for config in [default, *configs] if "/AS" in config]

    order_updates: list[_OrderUpdate] = []
    for info in [default_info, *config_infos]:
        if info.order is None:
            if info.config is default and all_ocgs:
                order_updates.append(
                    _OrderUpdate(config=info.config, create=True, ocgs=all_ocgs)
                )
            continue
        missing_order = [ocg for key, ocg in all_items if key not in info.order]
        if missing_order:
            order_updates.append(
                _OrderUpdate(config=info.config, create=False, ocgs=missing_order)
            )

    return _OptionalContentPlan(
        properties=properties,
        default=default,
        default_created=default_created,
        ocgs_missing=ocgs_missing,
        all_ocgs=all_ocgs,
        missing_ocgs=missing_ocgs,
        as_configs=as_configs,
        config_name_updates=name_updates,
        d_name_changed=d_name_changed,
        ocg_name_updates=ocg_name_updates,
        normalize_default_base=normalize_default_base,
        normalize_default_intent=normalize_default_intent,
        default_hidden=default_hidden,
        order_updates=order_updates,
    )
