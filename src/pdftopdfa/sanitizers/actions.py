# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Non-compliant action removal and destination validation for PDF/A compliance."""

import logging

import pikepdf
from pikepdf import Array, Name, Pdf, String

from ..utils import resolve_indirect as _resolve_indirect
from .base import _is_non_compliant_action, _sanitize_next_chain

logger = logging.getLogger(__name__)


def remove_actions(pdf: Pdf) -> int:
    """Removes non-PDF/A-compliant actions from the PDF.

    Removes Launch, Sound, Movie and other forbidden actions.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of actions removed.
    """
    removed_count = 0

    # Check OpenAction
    try:
        if "/OpenAction" in pdf.Root:
            open_action = _resolve_indirect(pdf.Root.OpenAction)
            if _is_non_compliant_action(open_action):
                del pdf.Root["/OpenAction"]
                removed_count += 1
                logger.debug("Non-compliant OpenAction removed")
            else:
                removed_count += _sanitize_next_chain(open_action)
    except Exception as e:
        logger.debug("Error checking OpenAction: %s", e)

    # Check Document AA — forbidden entirely (ISO 19005-2 Section 6.6.1)
    try:
        if "/AA" in pdf.Root:
            del pdf.Root["/AA"]
            removed_count += 1
            logger.debug("Catalog /AA removed (ISO 19005-2 Section 6.6.1)")
    except Exception as e:
        logger.debug("Error processing Document AA: %s", e)

    # Process pages and annotations
    for page_num, page in enumerate(pdf.pages, start=1):
        # Page AA — forbidden entirely (ISO 19005-2 Section 6.6.2)
        try:
            if "/AA" in page:
                del page["/AA"]
                removed_count += 1
                logger.debug(
                    "Page /AA removed on page %d (ISO 19005-2 Section 6.6.2)",
                    page_num,
                )
        except Exception as e:
            logger.debug("Error with Page AA on page %d: %s", page_num, e)

        # Annotations
        try:
            annots = page.get("/Annots")
            if annots is not None:
                try:
                    annots = annots.get_object()
                except (AttributeError, TypeError, ValueError):
                    pass
                for annot in annots:
                    try:
                        annot = annot.get_object()
                    except (AttributeError, TypeError, ValueError):
                        pass

                    # Widget annotations must not have /A or /AA
                    # in PDF/A (Rule 6.4.1)
                    subtype = annot.get("/Subtype")
                    is_widget = subtype is not None and str(subtype) == "/Widget"

                    if "/A" in annot:
                        if is_widget or _is_non_compliant_action(annot.A):
                            del annot["/A"]
                            removed_count += 1
                        else:
                            removed_count += _sanitize_next_chain(annot.A)

                    if "/AA" in annot:
                        if is_widget:
                            del annot["/AA"]
                            removed_count += 1
                        else:
                            aa = _resolve_indirect(annot.AA)
                            bad_keys = []
                            for key in aa.keys():
                                action = aa.get(key)
                                if action and _is_non_compliant_action(action):
                                    bad_keys.append(key)
                                elif action:
                                    removed_count += _sanitize_next_chain(action)
                            for key in bad_keys:
                                del aa[key]
                                removed_count += 1
                            if len(aa) == 0:
                                del annot["/AA"]
        except Exception as e:
            logger.debug("Error with Annotations on page %d: %s", page_num, e)

    # Check Outline (bookmark) actions
    try:
        if "/Outlines" in pdf.Root:
            outlines = _resolve_indirect(pdf.Root.Outlines)
            removed_count += _remove_actions_from_outlines(outlines)
    except Exception as e:
        logger.debug("Error processing Outline actions: %s", e)

    # Check AcroForm field actions
    try:
        if "/AcroForm" in pdf.Root:
            acroform = _resolve_indirect(pdf.Root.AcroForm)
            fields = acroform.get("/Fields")
            if fields is not None:
                removed_count += _remove_actions_from_fields(fields)
    except Exception as e:
        logger.debug("Error processing AcroForm actions: %s", e)

    if removed_count > 0:
        logger.info("%d non-compliant action(s) removed", removed_count)
    return removed_count


def _remove_actions_from_outlines(
    outline_root: pikepdf.Dictionary,
    _visited: set[tuple[int, int]] | None = None,
) -> int:
    """Removes non-compliant actions from outline (bookmark) items.

    Traverses the outline tree via /First//Next sibling links and recurses
    into children.

    Args:
        outline_root: Outline root or item dictionary with /First child.
        _visited: Set of visited objgen tuples for cycle detection.

    Returns:
        Number of actions removed.
    """
    if _visited is None:
        _visited = set()

    removed_count = 0
    item = outline_root.get("/First")
    while item is not None:
        try:
            item = _resolve_indirect(item)
        except Exception:
            break

        try:
            item_key = item.objgen
        except Exception:
            item_key = None
        if item_key is not None and item_key != (0, 0):
            if item_key in _visited:
                break
            _visited.add(item_key)

        try:
            if "/A" in item:
                if _is_non_compliant_action(item.A):
                    del item["/A"]
                    removed_count += 1
                else:
                    removed_count += _sanitize_next_chain(item.A)

            # Recurse into children
            if "/First" in item:
                removed_count += _remove_actions_from_outlines(item, _visited)
        except Exception as e:
            logger.debug("Error processing outline item action: %s", e)

        try:
            item = item.get("/Next")
        except Exception:
            break

    return removed_count


def _remove_actions_from_fields(
    fields: pikepdf.Array,
    _visited: set[tuple[int, int]] | None = None,
) -> int:
    """Removes non-compliant actions from form fields recursively.

    Args:
        fields: Array of form field objects.
        _visited: Set of visited objgen tuples for cycle detection.

    Returns:
        Number of actions removed.
    """
    if _visited is None:
        _visited = set()

    removed_count = 0

    for field in fields:
        try:
            field = _resolve_indirect(field)

            try:
                field_key = field.objgen
            except Exception:
                field_key = None
            if field_key is not None and field_key != (0, 0):
                if field_key in _visited:
                    continue
                _visited.add(field_key)

            # Fields must not have /A or /AA in PDF/A (Rule 6.4.1)
            if "/A" in field:
                del field["/A"]
                removed_count += 1

            if "/AA" in field:
                del field["/AA"]
                removed_count += 1

            # Recursively process children
            kids = field.get("/Kids")
            if kids is not None:
                removed_count += _remove_actions_from_fields(kids, _visited)
        except Exception as e:
            logger.debug("Error processing form field action: %s", e)

    return removed_count


# ---------------------------------------------------------------------------
# Destination validation
# ---------------------------------------------------------------------------


def _collect_valid_page_objgens(pdf: Pdf) -> set[tuple[int, int]]:
    """Collects objgen tuples for all pages in the PDF."""
    objgens: set[tuple[int, int]] = set()
    for page in pdf.pages:
        try:
            objgens.add(page.obj.objgen)
        except Exception:
            pass
    return objgens


def _collect_named_destinations(pdf: Pdf) -> set[str]:
    """Collects all defined named destination keys.

    Checks both the modern /Names/Dests name tree and the legacy
    /Root/Dests dictionary.
    """
    names: set[str] = set()

    # Modern format: /Root/Names/Dests (name tree with /Names array)
    try:
        if "/Names" in pdf.Root:
            names_dict = _resolve_indirect(pdf.Root.Names)
            if "/Dests" in names_dict:
                dests_tree = _resolve_indirect(names_dict.Dests)
                _collect_from_name_tree(dests_tree, names)
    except Exception as e:
        logger.debug("Error collecting named destinations from name tree: %s", e)

    # Legacy format: /Root/Dests (plain dictionary)
    try:
        if "/Dests" in pdf.Root:
            dests = _resolve_indirect(pdf.Root.Dests)
            for key in dests.keys():
                names.add(str(key))
    except Exception as e:
        logger.debug("Error collecting named destinations from /Root/Dests: %s", e)

    return names


def _collect_from_name_tree(node: pikepdf.Dictionary, names: set[str]) -> None:
    """Recursively collects keys from a PDF name tree node."""
    try:
        if "/Names" in node:
            arr = node.Names
            # Name trees have alternating key/value pairs
            for i in range(0, len(arr) - 1, 2):
                try:
                    names.add(str(arr[i]))
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if "/Kids" in node:
            for kid in node.Kids:
                try:
                    kid = _resolve_indirect(kid)
                    _collect_from_name_tree(kid, names)
                except Exception:
                    pass
    except Exception:
        pass


def _is_invalid_destination(
    dest: object,
    valid_objgens: set[tuple[int, int]],
    named_dests: set[str],
) -> bool:
    """Checks whether a destination reference is invalid.

    Args:
        dest: The destination value (Array, String, Name, or other).
        valid_objgens: Set of valid page objgen tuples.
        named_dests: Set of defined named destination keys.

    Returns:
        True if the destination is invalid and should be removed.
    """
    try:
        dest = _resolve_indirect(dest)
    except Exception:
        return True

    # Array form: [page_ref, /fit_type, ...]
    if isinstance(dest, Array):
        if len(dest) == 0:
            return True
        try:
            page_ref = dest[0]
            page_ref = _resolve_indirect(page_ref)
            return page_ref.objgen not in valid_objgens
        except Exception:
            return True

    # Named destination (string or name)
    if isinstance(dest, String):
        return str(dest) not in named_dests
    if isinstance(dest, Name):
        return str(dest) not in named_dests

    # Unknown type — treat as invalid
    return True


def validate_destinations(pdf: Pdf) -> int:
    """Removes destinations that reference non-existent pages.

    Checks OpenAction, GoTo actions, outline /Dest entries, link
    annotation /Dest entries, and the named destination tree itself.
    GoToR/GoToE actions (external files) are not validated.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of invalid destinations removed.
    """
    valid_objgens = _collect_valid_page_objgens(pdf)
    named_dests = _collect_named_destinations(pdf)
    removed = 0

    # 1. OpenAction (direct dest array on catalog)
    try:
        if "/OpenAction" in pdf.Root:
            oa = _resolve_indirect(pdf.Root.OpenAction)
            if isinstance(oa, Array):
                if _is_invalid_destination(oa, valid_objgens, named_dests):
                    del pdf.Root["/OpenAction"]
                    removed += 1
                    logger.debug("Invalid OpenAction destination removed")
    except Exception as e:
        logger.debug("Error validating OpenAction destination: %s", e)

    # 2. GoTo actions and /Dest entries on annotations
    for page in pdf.pages:
        try:
            annots = page.get("/Annots")
            if annots is None:
                continue
            try:
                annots = annots.get_object()
            except (AttributeError, TypeError, ValueError):
                pass
            for annot in annots:
                try:
                    annot = annot.get_object()
                except (AttributeError, TypeError, ValueError):
                    pass

                # GoTo action with /D
                try:
                    if "/A" in annot:
                        action = _resolve_indirect(annot.A)
                        if _is_goto_action(action):
                            dest = action.get("/D")
                            if dest is not None and _is_invalid_destination(
                                dest, valid_objgens, named_dests
                            ):
                                del annot["/A"]
                                removed += 1
                except Exception as e:
                    logger.debug("Error validating annotation action: %s", e)

                # Direct /Dest on annotation
                try:
                    if "/Dest" in annot:
                        dest = annot.Dest
                        if _is_invalid_destination(dest, valid_objgens, named_dests):
                            del annot["/Dest"]
                            removed += 1
                except Exception as e:
                    logger.debug("Error validating annotation /Dest: %s", e)
        except Exception as e:
            logger.debug("Error validating annotations on page: %s", e)

    # 3. Outline items
    try:
        if "/Outlines" in pdf.Root:
            outlines = _resolve_indirect(pdf.Root.Outlines)
            removed += _validate_outline_destinations(
                outlines, valid_objgens, named_dests
            )
    except Exception as e:
        logger.debug("Error validating outline destinations: %s", e)

    # 4. Named destinations tree — remove entries with invalid page refs
    removed += _validate_named_dest_tree(pdf, valid_objgens)

    if removed > 0:
        logger.info("%d invalid destination(s) removed", removed)
    return removed


def _is_goto_action(action: pikepdf.Dictionary) -> bool:
    """Returns True if action is a /GoTo action (not GoToR/GoToE)."""
    try:
        s = str(action.get("/S"))
        return s == "/GoTo"
    except Exception:
        return False


def _validate_outline_destinations(
    outline_root: pikepdf.Dictionary,
    valid_objgens: set[tuple[int, int]],
    named_dests: set[str],
    _visited: set[tuple[int, int]] | None = None,
) -> int:
    """Removes invalid destinations from outline (bookmark) items."""
    if _visited is None:
        _visited = set()

    removed = 0
    item = outline_root.get("/First")
    while item is not None:
        try:
            item = _resolve_indirect(item)
        except Exception:
            break

        try:
            item_key = item.objgen
        except Exception:
            item_key = None
        if item_key is not None and item_key != (0, 0):
            if item_key in _visited:
                break
            _visited.add(item_key)

        # GoTo action with /D
        try:
            if "/A" in item:
                action = _resolve_indirect(item.A)
                if _is_goto_action(action):
                    dest = action.get("/D")
                    if dest is not None and _is_invalid_destination(
                        dest, valid_objgens, named_dests
                    ):
                        del item["/A"]
                        removed += 1
        except Exception as e:
            logger.debug("Error validating outline action dest: %s", e)

        # Direct /Dest on outline item
        try:
            if "/Dest" in item:
                dest = item.Dest
                if _is_invalid_destination(dest, valid_objgens, named_dests):
                    del item["/Dest"]
                    removed += 1
        except Exception as e:
            logger.debug("Error validating outline /Dest: %s", e)

        # Recurse into children
        try:
            if "/First" in item:
                removed += _validate_outline_destinations(
                    item, valid_objgens, named_dests, _visited
                )
        except Exception:
            pass

        try:
            item = item.get("/Next")
        except Exception:
            break

    return removed


def _validate_named_dest_tree(pdf: Pdf, valid_objgens: set[tuple[int, int]]) -> int:
    """Removes named destination entries whose page ref is invalid."""
    removed = 0

    # Modern name tree: /Root/Names/Dests
    try:
        if "/Names" in pdf.Root:
            names_dict = _resolve_indirect(pdf.Root.Names)
            if "/Dests" in names_dict:
                dests_tree = _resolve_indirect(names_dict.Dests)
                removed += _prune_name_tree_node(dests_tree, valid_objgens)
    except Exception as e:
        logger.debug("Error pruning name tree: %s", e)

    # Legacy dict: /Root/Dests
    try:
        if "/Dests" in pdf.Root:
            dests = _resolve_indirect(pdf.Root.Dests)
            bad_keys = []
            for key in dests.keys():
                try:
                    val = _resolve_indirect(dests[key])
                    if _is_dest_entry_invalid(val, valid_objgens):
                        bad_keys.append(key)
                except Exception:
                    bad_keys.append(key)
            for key in bad_keys:
                del dests[key]
                removed += 1
    except Exception as e:
        logger.debug("Error pruning legacy /Dests dict: %s", e)

    return removed


def _is_dest_entry_invalid(val: object, valid_objgens: set[tuple[int, int]]) -> bool:
    """Checks if a named dest value has an invalid page reference.

    Named dest values can be an Array [page, /type, ...] or a
    Dictionary with a /D key containing such an array.
    """
    try:
        val = _resolve_indirect(val)
    except Exception:
        return True

    if isinstance(val, pikepdf.Dictionary):
        dest = val.get("/D")
        if dest is None:
            return True
        val = _resolve_indirect(dest)

    if isinstance(val, Array):
        if len(val) == 0:
            return True
        try:
            page_ref = _resolve_indirect(val[0])
            return page_ref.objgen not in valid_objgens
        except Exception:
            return True

    return True


def _prune_name_tree_node(
    node: pikepdf.Dictionary, valid_objgens: set[tuple[int, int]]
) -> int:
    """Removes invalid entries from a name tree node."""
    removed = 0

    try:
        if "/Names" in node:
            arr = node.Names
            indices_to_remove = []
            for i in range(0, len(arr) - 1, 2):
                try:
                    val = _resolve_indirect(arr[i + 1])
                    if _is_dest_entry_invalid(val, valid_objgens):
                        indices_to_remove.append(i)
                except Exception:
                    indices_to_remove.append(i)
            # Remove in reverse order to keep indices valid
            for i in reversed(indices_to_remove):
                del arr[i + 1]
                del arr[i]
                removed += 1
    except Exception:
        pass

    try:
        if "/Kids" in node:
            for kid in node.Kids:
                try:
                    kid = _resolve_indirect(kid)
                    removed += _prune_name_tree_node(kid, valid_objgens)
                except Exception:
                    pass
    except Exception:
        pass

    return removed
