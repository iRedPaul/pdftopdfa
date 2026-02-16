# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""XFA form removal for PDF/A compliance."""

import logging

from pikepdf import Pdf

from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)


def remove_xfa_forms(pdf: Pdf) -> int:
    """Removes XFA forms from the PDF.

    XFA (XML Forms Architecture) is forbidden in PDF/A.
    This function removes:
    - /XFA key from AcroForm (can be stream or array)
    - /NeedsRendering key from AcroForm (indicates XFA rendering required)

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of XFA elements removed.
    """
    removed_count = 0

    try:
        if "/AcroForm" not in pdf.Root:
            return 0

        acroform = _resolve_indirect(pdf.Root.AcroForm)

        # Remove /XFA (can be stream or array of alternating name/stream pairs)
        if "/XFA" in acroform:
            # Detect pure-XFA PDFs: XFA present but no /Fields (or empty).
            # Removing XFA from such PDFs destroys all form content.
            fields = acroform.get("/Fields")
            is_pure_xfa = fields is None or len(fields) == 0
            if is_pure_xfa:
                logger.warning(
                    "Pure-XFA PDF detected: /XFA is present but /Fields "
                    "is missing or empty. Removing /XFA will destroy all "
                    "form content. This is required for PDF/A compliance."
                )

            del acroform["/XFA"]
            removed_count += 1
            logger.debug("XFA form data removed from AcroForm")

        # Remove /NeedsRendering (boolean indicating XFA rendering required)
        if "/NeedsRendering" in acroform:
            del acroform["/NeedsRendering"]
            removed_count += 1
            logger.debug("NeedsRendering flag removed from AcroForm")

    except Exception as e:
        logger.debug("Error removing XFA forms: %s", e)

    if removed_count > 0:
        logger.info("%d XFA element(s) removed", removed_count)
    return removed_count
