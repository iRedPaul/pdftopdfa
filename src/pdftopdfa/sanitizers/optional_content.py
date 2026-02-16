# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Optional Content (Layers) sanitization for PDF/A-2/3 compliance."""

import logging

from pikepdf import Array, Dictionary, Name, Pdf

from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)


def sanitize_optional_content(pdf: Pdf) -> dict:
    """Sanitizes Optional Content for PDF/A compliance.

    PDF/A-2 and PDF/A-3 require:
    - No /AS entry in OCProperties (auto-state triggers)
    - Each OCG must have /Intent /View (or no Intent, which defaults to View)

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with keys:
        - as_entries_removed: Number of AS entries removed from configs
        - intents_fixed: Number of OCG intents corrected to /View
        - ocgs_processed: Total number of OCGs processed
    """
    result = {
        "as_entries_removed": 0,
        "intents_fixed": 0,
        "ocgs_processed": 0,
        "d_created": False,
        "d_name_added": False,
        "list_mode_fixed": 0,
        "base_state_fixed": 0,
        "config_names_added": 0,
        "missing_ocgs_added": 0,
        "rbgroups_fixed": 0,
        "ocg_names_added": 0,
        "order_ocgs_added": 0,
    }

    try:
        if "/OCProperties" not in pdf.Root:
            return result

        oc_props = pdf.Root.OCProperties

        # Create /D default configuration if missing (ISO 19005-2, 6.9)
        if "/D" not in oc_props:
            d_config = Dictionary()
            d_config["/Name"] = "Default"
            d_config["/BaseState"] = Name.ON
            if "/OCGs" in oc_props:
                d_config["/Order"] = Array(list(oc_props.OCGs))
            oc_props["/D"] = d_config
            result["d_created"] = True
            logger.debug("Created missing /D default configuration")

        # Fix default configuration (/D)
        if "/D" in oc_props:
            default_config = oc_props.D
            if "/AS" in default_config:
                del default_config["/AS"]
                result["as_entries_removed"] += 1
                logger.debug("Removed /AS entry from default OCProperties config")
            # /BaseState must be /ON (ISO 19005-2, 6.8)
            if "/BaseState" in default_config:
                if str(default_config.BaseState) != "/ON":
                    default_config["/BaseState"] = Name.ON
                    result["base_state_fixed"] += 1
                    logger.debug("Fixed /BaseState to /ON in default config")
            # /ListMode must be absent or /AllPages (ISO 19005-2, 6.8)
            if "/ListMode" in default_config:
                if str(default_config.ListMode) != "/AllPages":
                    del default_config["/ListMode"]
                    result["list_mode_fixed"] += 1
                    logger.debug("Removed non-compliant /ListMode from default config")

        # Fix alternate configurations (/Configs)
        if "/Configs" in oc_props:
            configs = oc_props.Configs
            for i, config in enumerate(configs):
                if "/AS" in config:
                    del config["/AS"]
                    result["as_entries_removed"] += 1
                    logger.debug("Removed /AS entry from alternate config %d", i)
                # /BaseState must be /ON (ISO 19005-2, 6.8)
                if "/BaseState" in config:
                    if str(config.BaseState) != "/ON":
                        config["/BaseState"] = Name.ON
                        result["base_state_fixed"] += 1
                        logger.debug(
                            "Fixed /BaseState to /ON in alternate config %d", i
                        )
                # /ListMode must be absent or /AllPages (ISO 19005-2, 6.8)
                if "/ListMode" in config:
                    if str(config.ListMode) != "/AllPages":
                        del config["/ListMode"]
                        result["list_mode_fixed"] += 1
                        logger.debug(
                            "Removed non-compliant /ListMode from alternate config %d",
                            i,
                        )

        # Rule 6.9: every OC config needs a non-empty /Name, and
        # names must be unique across /D and /Configs.
        d_name_changed, config_names_changed = _sanitize_config_names(oc_props)
        if d_name_changed:
            result["d_name_added"] = True
        result["config_names_added"] += config_names_changed

        # Validate /RBGroups in /D and /Configs (ISO 19005-2, 6.8)
        # Each inner array must only reference OCGs in the /OCGs array.
        if "/OCGs" in oc_props:
            registered = set()
            for ocg in oc_props.OCGs:
                try:
                    registered.add(ocg.objgen)
                except Exception:
                    pass

            if "/D" in oc_props:
                result["rbgroups_fixed"] += _sanitize_rbgroups(oc_props.D, registered)
            if "/Configs" in oc_props:
                for config in oc_props.Configs:
                    result["rbgroups_fixed"] += _sanitize_rbgroups(config, registered)

        # Fix /Intent for each OCG
        if "/OCGs" in oc_props:
            ocgs = oc_props.OCGs
            for ocg in ocgs:
                try:
                    ocg = _resolve_indirect(ocg)
                    if not isinstance(ocg, Dictionary):
                        continue
                    result["ocgs_processed"] += 1

                    # Each OCG must have /Name (ISO 19005-2, 6.8)
                    if "/Name" not in ocg:
                        ocg["/Name"] = "Unnamed OCG"
                        result["ocg_names_added"] += 1
                        logger.debug("Added /Name to OCG dictionary")

                    if "/Intent" in ocg:
                        intent = _resolve_indirect(ocg.Intent)

                        # If Intent is an array, filter to only /View
                        if isinstance(intent, Array):
                            has_non_view = any(str(item) != "/View" for item in intent)
                            if has_non_view:
                                ocg["/Intent"] = Name.View
                                result["intents_fixed"] += 1
                                logger.debug("Fixed OCG Intent array to /View")
                        # If Intent is a single name that's not /View
                        elif str(intent) != "/View":
                            ocg["/Intent"] = Name.View
                            result["intents_fixed"] += 1
                            logger.debug("Fixed OCG Intent from %s to /View", intent)
                    # If no /Intent present, the default is /View
                    # per PDF spec, so no action needed
                except Exception:
                    logger.debug("Skipping unreadable OCG entry")

        # Ensure all OCGs used in the document are in /OCGs array
        # (ISO 19005-2, 6.8)
        result["missing_ocgs_added"] = _collect_missing_ocgs(pdf, oc_props)

        # Ensure /Order array in /D lists all OCGs (ISO 19005-2, 6.8)
        if "/OCGs" in oc_props and "/D" in oc_props:
            result["order_ocgs_added"] = _sync_order_array(oc_props.D, oc_props.OCGs)
            if "/Configs" in oc_props:
                for config in oc_props.Configs:
                    if "/Order" in config:
                        result["order_ocgs_added"] += _sync_order_array(
                            config, oc_props.OCGs
                        )

    except Exception as e:
        logger.debug("Error sanitizing optional content: %s", e)

    changes = (
        result["as_entries_removed"]
        + result["intents_fixed"]
        + result["list_mode_fixed"]
        + result["base_state_fixed"]
        + result["config_names_added"]
        + result["missing_ocgs_added"]
        + result["rbgroups_fixed"]
        + result["ocg_names_added"]
        + result["order_ocgs_added"]
        + int(result["d_name_added"])
        + int(result["d_created"])
    )
    if changes > 0:
        logger.info(
            "Optional content sanitized: %d AS removed, %d intents fixed, "
            "%d ListMode fixed, %d BaseState fixed, %d config names added, "
            "%d missing OCGs added, %d RBGroups fixed, "
            "%d OCG names added, %d Order OCGs added, "
            "D created: %s, D /Name added: %s",
            result["as_entries_removed"],
            result["intents_fixed"],
            result["list_mode_fixed"],
            result["base_state_fixed"],
            result["config_names_added"],
            result["missing_ocgs_added"],
            result["rbgroups_fixed"],
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


def _sanitize_config_names(oc_props) -> tuple[bool, int]:
    """Fix missing/empty and duplicate OC config names.

    Returns:
        Tuple ``(d_changed, configs_changed_count)``.
    """
    entries = []
    if "/D" in oc_props and isinstance(_resolve_indirect(oc_props.D), Dictionary):
        entries.append(("D", -1, oc_props.D))
    if "/Configs" in oc_props:
        for i, config in enumerate(oc_props.Configs):
            if isinstance(_resolve_indirect(config), Dictionary):
                entries.append(("Config", i, config))

    used_names = set()
    d_changed = False
    configs_changed = 0

    for kind, index, config in entries:
        current_name = _read_config_name(config)
        fallback_name = "Default" if kind == "D" else f"Config{index}"
        base_name = current_name or fallback_name
        unique_name = _make_unique_name(base_name, used_names)
        used_names.add(unique_name)

        if unique_name != current_name:
            config["/Name"] = unique_name
            if kind == "D":
                d_changed = True
                logger.debug(
                    "Fixed /Name in default OCProperties config to %r",
                    unique_name,
                )
            else:
                configs_changed += 1
                logger.debug(
                    "Fixed /Name in alternate OCProperties config %d to %r",
                    index,
                    unique_name,
                )

    return d_changed, configs_changed


def _sanitize_rbgroups(config, registered_objgens: set) -> int:
    """Remove invalid OCG references from /RBGroups in a config dictionary.

    Args:
        config: An OC configuration dictionary (/D or alternate config).
        registered_objgens: Set of (objgen) tuples for OCGs in /OCGs array.

    Returns:
        Number of invalid OCG references removed.
    """
    if "/RBGroups" not in config:
        return 0

    removed = 0
    rbgroups = config.RBGroups

    empty_groups = []
    for gi, group in enumerate(rbgroups):
        if not isinstance(group, Array):
            continue
        bad_indices = []
        for i, ocg_ref in enumerate(group):
            try:
                objgen = ocg_ref.objgen
            except Exception:
                bad_indices.append(i)
                continue
            if objgen not in registered_objgens:
                bad_indices.append(i)
        for i in reversed(bad_indices):
            del group[i]
            removed += 1
            logger.debug("Removed invalid OCG reference from RBGroups[%d]", gi)
        if len(group) == 0:
            empty_groups.append(gi)

    for gi in reversed(empty_groups):
        del rbgroups[gi]
        logger.debug("Removed empty RBGroups entry at index %d", gi)

    if len(rbgroups) == 0:
        del config["/RBGroups"]
        logger.debug("Removed empty /RBGroups array")

    return removed


def _collect_order_objgens(order_array, objgens: set) -> None:
    """Recursively collect objgen tuples of OCG references in an /Order array.

    The /Order array may contain indirect OCG references and nested arrays
    (for layer groups).  Text strings used as labels are skipped.
    """
    for item in order_array:
        if isinstance(item, Array):
            _collect_order_objgens(item, objgens)
        else:
            try:
                objgens.add(item.objgen)
            except Exception:
                pass


def _sync_order_array(config, ocgs_array) -> int:
    """Ensure every OCG in *ocgs_array* appears in the config's /Order array.

    If /Order does not exist, it is created with all OCGs.  If it exists,
    missing OCGs are appended at the top level.

    Returns the number of OCGs added.
    """
    # Build set of registered OCG objgens
    registered = set()
    ocg_list = []
    for ocg in ocgs_array:
        try:
            registered.add(ocg.objgen)
            ocg_list.append(ocg)
        except Exception:
            pass

    if not registered:
        return 0

    if "/Order" not in config:
        config["/Order"] = Array(ocg_list)
        logger.debug("Created /Order array with %d OCGs", len(ocg_list))
        return len(ocg_list)

    # Collect objgens already present in /Order
    present = set()
    _collect_order_objgens(config.Order, present)

    added = 0
    for ocg in ocg_list:
        try:
            if ocg.objgen not in present:
                config.Order.append(ocg)
                added += 1
                logger.debug("Added missing OCG %s to /Order array", ocg.objgen)
        except Exception:
            pass

    return added


def _collect_missing_ocgs(pdf: Pdf, oc_props) -> int:
    """Find OCGs referenced in the document but missing from /OCGs array.

    Returns the number of OCGs added.
    """
    if "/OCGs" not in oc_props:
        return 0

    ocgs_array = oc_props.OCGs
    # Build set of object IDs already registered
    registered = set()
    for ocg in ocgs_array:
        try:
            registered.add(ocg.objgen)
        except Exception:
            pass

    added = 0
    found_ocgs = _find_all_ocgs_in_pages(pdf)

    for ocg_ref in found_ocgs:
        try:
            objgen = ocg_ref.objgen
        except Exception:
            continue
        if objgen not in registered:
            ocgs_array.append(ocg_ref)
            registered.add(objgen)
            added += 1
            logger.debug("Added missing OCG %s to /OCGs array", objgen)

    return added


def _find_all_ocgs_in_pages(pdf: Pdf) -> list:
    """Collect all OCG references from page resources and annotations."""
    found = []
    for page in pdf.pages:
        try:
            page_dict = page.obj
        except Exception:
            continue

        # Check /Resources/Properties for OCG references
        try:
            if "/Resources" in page_dict and "/Properties" in page_dict.Resources:
                props = page_dict.Resources.Properties
                for key in props.keys():
                    prop = props[key]
                    try:
                        if "/Type" in prop and str(prop.Type) == "/OCG":
                            found.append(prop)
                    except Exception:
                        pass
        except Exception:
            pass

        # Check annotations for /OC entries
        try:
            if "/Annots" in page_dict:
                for annot in page_dict.Annots:
                    try:
                        if "/OC" in annot:
                            oc = annot.OC
                            if "/Type" in oc and str(oc.Type) == "/OCG":
                                found.append(oc)
                    except Exception:
                        pass
        except Exception:
            pass

    return found
