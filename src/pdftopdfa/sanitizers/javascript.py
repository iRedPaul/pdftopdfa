# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""JavaScript removal for PDF/A compliance.

Handles removal of the Named JavaScript tree (/Root/Names/JavaScript).
JavaScript actions in OpenAction, AA dicts, annotations, and form fields
are handled by remove_actions() which covers all non-compliant action types.
"""

import logging

from pikepdf import Pdf

from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)


def remove_javascript(pdf: Pdf) -> int:
    """Removes the Named JavaScript tree from the PDF.

    Only handles /Root/Names/JavaScript. JavaScript actions in OpenAction,
    document/page AA, annotations, and form fields are removed by
    remove_actions() as part of general non-compliant action cleanup.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of JavaScript elements removed (0 or 1).
    """
    try:
        if "/Names" in pdf.Root:
            names = _resolve_indirect(pdf.Root.Names)
            if "/JavaScript" in names:
                del names["/JavaScript"]
                logger.info("Named JavaScript removed from Names dictionary")
                return 1
    except Exception as e:
        logger.debug("Error removing named JavaScript: %s", e)

    return 0
