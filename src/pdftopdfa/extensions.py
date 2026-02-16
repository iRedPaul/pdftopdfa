# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Extensions dictionary handling for PDF/A compliance.

PDF/A-3 documents require an Extensions dictionary in the PDF Catalog
to signal the use of ISO 32000-1:2008 Extension Level 3 features.
"""

import logging

import pikepdf

logger = logging.getLogger(__name__)


def needs_extension_level(level: str) -> tuple[bool, int]:
    """Check if a PDF/A level requires an Extensions dictionary.

    Args:
        level: PDF/A level string (e.g., '2b', '3b').

    Returns:
        Tuple of (needs_extension, extension_level).
        For PDF/A-3, returns (True, 3).
        For PDF/A-2 and others, returns (False, 0).
    """
    level_lower = level.lower()
    part = int(level_lower[0]) if level_lower else 0

    if part == 3:
        return (True, 3)
    return (False, 0)


def add_adbe_extension(pdf: pikepdf.Pdf, extension_level: int) -> bool:
    """Add or update the ADBE extension entry in the Extensions dictionary.

    Args:
        pdf: pikepdf Pdf object to modify.
        extension_level: Extension level to set (e.g., 3).

    Returns:
        True if extension was added/updated, False if already present
        with equal or higher level.
    """
    # Get or create Extensions dictionary
    if "/Extensions" not in pdf.Root:
        pdf.Root.Extensions = pikepdf.Dictionary()
        logger.debug("Created Extensions dictionary")

    extensions = pdf.Root.Extensions

    # Check for existing ADBE extension
    if "/ADBE" in extensions:
        existing_adbe = extensions.ADBE
        existing_level = int(existing_adbe.get("/ExtensionLevel", 0))

        if existing_level >= extension_level:
            logger.debug(
                "ADBE extension already at level %d (requested %d), not downgrading",
                existing_level,
                extension_level,
            )
            return False

        logger.debug(
            "Upgrading ADBE extension from level %d to %d",
            existing_level,
            extension_level,
        )

    # Create ADBE extension dictionary
    adbe_dict = pikepdf.Dictionary(
        BaseVersion=pikepdf.Name("/1.7"),
        ExtensionLevel=extension_level,
    )

    extensions.ADBE = adbe_dict
    logger.debug("Added ADBE extension with ExtensionLevel %d", extension_level)
    return True


def remove_pdf20_extensions(pdf: pikepdf.Pdf) -> int:
    """Remove PDF 2.0 extension entries incompatible with PDF/A-2/3.

    PDF/A-2 and PDF/A-3 are based on PDF 1.7 (ISO 32000-1). Non-ADBE
    extension entries (e.g., ISO for PDF 2.0) are not valid and should
    be removed.

    Args:
        pdf: pikepdf Pdf object to modify.

    Returns:
        Number of extension entries removed.
    """
    if "/Extensions" not in pdf.Root:
        return 0

    extensions = pdf.Root.Extensions
    removed = 0

    for key in list(extensions.keys()):
        if key == "/ADBE":
            continue
        del extensions[key]
        removed += 1
        logger.debug("Removed non-ADBE extension entry: %s", key)

    if removed > 0:
        logger.info("%d non-ADBE extension(s) removed", removed)

    return removed


def add_extensions_if_needed(pdf: pikepdf.Pdf, level: str) -> bool:
    """Add Extensions dictionary to PDF if required by the PDF/A level.

    This is the main entry point for converter.py.

    Args:
        pdf: pikepdf Pdf object to modify.
        level: PDF/A level string (e.g., '2b', '3b').

    Returns:
        True if extensions were added, False otherwise.
    """
    # Remove non-ADBE extensions (PDF 2.0 etc.) for all levels
    remove_pdf20_extensions(pdf)

    needs_ext, ext_level = needs_extension_level(level)

    if not needs_ext:
        logger.debug("PDF/A-%s does not require Extensions dictionary", level)
        return False

    logger.info("Adding Extensions dictionary for PDF/A-%s", level)
    return add_adbe_extension(pdf, ext_level)
