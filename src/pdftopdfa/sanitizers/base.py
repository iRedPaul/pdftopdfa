# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Base constants and helper functions for PDF/A sanitizers."""

import logging

import pikepdf

logger = logging.getLogger(__name__)

# Actions allowed in PDF/A per ISO 19005-2 Section 6.6.1
COMPLIANT_ACTIONS = frozenset(
    {
        "/GoTo",
        "/GoToR",
        "/GoToE",
        "/Thread",
        "/URI",
        "/Named",
        "/SubmitForm",
    }
)

# Named actions allowed in PDF/A per ISO 19005-2 Clause 6.6.1:
# "Named actions other than NextPage, PrevPage, FirstPage, LastPage
#  shall not be permitted."
ALLOWED_NAMED_ACTIONS = frozenset(
    {
        "/NextPage",
        "/PrevPage",
        "/FirstPage",
        "/LastPage",
    }
)

# Annotation subtypes forbidden in PDF/A-1, PDF/A-2, PDF/A-3
# Note: FileAttachment is handled separately in remove_embedded_files()
FORBIDDEN_ANNOTATION_SUBTYPES = frozenset(
    {
        "/Sound",
        "/Movie",
        "/Screen",
        "/3D",
        "/RichMedia",
        "/TrapNet",
    }
)

# XObject subtypes forbidden in PDF/A
FORBIDDEN_XOBJECT_SUBTYPES = frozenset(
    {
        "/PS",
        "/Ref",
    }
)

# SubmitForm Action Flags (PDF Reference, Table 237)
# Bit 3 = ExportFormat (HTML when set; FDF when clear)
SUBMITFORM_FLAG_EXPORTFORMAT = 1 << 2
# Bit 6 = XFDF (submit as XFDF)
SUBMITFORM_FLAG_XFDF = 1 << 5
# Bit 9 = SubmitPDF (submit entire PDF)
SUBMITFORM_FLAG_SUBMITPDF = 1 << 8

# Annotation Flag Bits (PDF Reference)
# Bit 1 = Invisible flag (hide if subtype unknown)
ANNOT_FLAG_INVISIBLE = 1 << 0
# Bit 3 (0-indexed from bit 1) = Print flag
ANNOT_FLAG_PRINT = 1 << 2
# Bit 2 = Hidden flag (annotation is completely hidden)
ANNOT_FLAG_HIDDEN = 1 << 1
# Bit 4 = NoZoom flag (annotation size stays constant when zooming)
ANNOT_FLAG_NOZOOM = 1 << 3
# Bit 5 = NoRotate flag (annotation orientation stays constant when rotating)
ANNOT_FLAG_NOROTATE = 1 << 4
# Bit 6 = NoView flag (not visible on screen)
ANNOT_FLAG_NOVIEW = 1 << 5
# Bit 9 = ToggleNoView flag (toggles NoView when interacted with)
ANNOT_FLAG_TOGGLENOVIEW = 1 << 8


def _is_javascript_action(action: pikepdf.Object) -> bool:
    """Checks if an action is a JavaScript action.

    Args:
        action: pikepdf Action object.

    Returns:
        True if it is a JavaScript action.
    """
    try:
        try:
            action = action.get_object()
        except (AttributeError, ValueError, TypeError):
            pass

        action_type = action.get("/S")
        if action_type is not None:
            return str(action_type) == "/JavaScript"
    except Exception:
        logger.debug("Error inspecting action for JavaScript", exc_info=True)
    return False


def _is_non_compliant_action(action: pikepdf.Object) -> bool:
    """Checks if an action is non-PDF/A-compliant.

    Args:
        action: pikepdf Action object.

    Returns:
        True if the action is not PDF/A compliant.
    """
    try:
        try:
            action = action.get_object()
        except (AttributeError, ValueError, TypeError):
            pass

        action_type = action.get("/S")
        if action_type is not None:
            type_str = str(action_type)
            if type_str not in COMPLIANT_ACTIONS:
                return True
            # ISO 19005-2 Clause 6.6.1: SubmitForm is only allowed
            # when the submission format is PDF or XFDF.
            if type_str == "/SubmitForm":
                flags = 0
                raw_flags = action.get("/Flags")
                if raw_flags is not None:
                    try:
                        flags = int(raw_flags)
                    except (ValueError, TypeError):
                        return True
                is_pdf = bool(flags & SUBMITFORM_FLAG_SUBMITPDF)
                is_xfdf = bool(flags & SUBMITFORM_FLAG_XFDF)
                if not is_pdf and not is_xfdf:
                    return True
            # ISO 19005-2 Clause 6.6.1: only NextPage, PrevPage,
            # FirstPage, LastPage are permitted for Named actions.
            if type_str == "/Named":
                named_name = action.get("/N")
                if named_name is None:
                    return True
                return str(named_name) not in ALLOWED_NAMED_ACTIONS
            return False
        else:
            # Action without /S key is malformed and non-compliant
            return True
    except Exception:
        logger.debug("Error inspecting action for compliance", exc_info=True)
    return True


def _sanitize_next_chain(
    action: pikepdf.Object,
    _visited: set[tuple[int, int]] | None = None,
) -> int:
    """Remove non-compliant actions from a /Next chain.

    PDF actions can reference follow-up actions via /Next (single action or
    array).  This function recursively walks the chain and strips any
    non-compliant entries so that forbidden actions cannot hide behind a
    compliant head action.

    Args:
        action: pikepdf Action dictionary (already determined to be compliant).
        _visited: Objgen tuples already seen (cycle guard).

    Returns:
        Number of non-compliant actions removed from the chain.
    """
    if _visited is None:
        _visited = set()

    removed = 0
    try:
        try:
            action = action.get_object()
        except (AttributeError, ValueError, TypeError):
            pass

        try:
            obj_key = action.objgen
        except Exception:
            obj_key = None
        if obj_key is not None and obj_key != (0, 0):
            if obj_key in _visited:
                return 0
            _visited.add(obj_key)

        next_val = action.get("/Next")
        if next_val is None:
            return 0

        try:
            next_val = next_val.get_object()
        except (AttributeError, ValueError, TypeError):
            pass

        if isinstance(next_val, pikepdf.Array):
            bad_indices: list[int] = []
            for i, next_action in enumerate(next_val):
                try:
                    next_action = next_action.get_object()
                except (AttributeError, ValueError, TypeError):
                    pass
                if _is_non_compliant_action(next_action):
                    bad_indices.append(i)
                else:
                    removed += _sanitize_next_chain(next_action, _visited)

            for i in reversed(bad_indices):
                del next_val[i]
                removed += 1

            if len(next_val) == 0:
                del action["/Next"]
            elif len(next_val) == 1:
                action["/Next"] = next_val[0]

        elif isinstance(next_val, pikepdf.Dictionary):
            if _is_non_compliant_action(next_val):
                del action["/Next"]
                removed += 1
            else:
                removed += _sanitize_next_chain(next_val, _visited)
    except Exception:
        logger.debug("Error sanitizing /Next chain", exc_info=True)

    return removed
