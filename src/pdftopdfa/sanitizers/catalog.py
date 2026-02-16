# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Document Catalog sanitization for PDF/A compliance.

ISO 19005-2, clause 6.1.2 requires that the document header version and
the catalog /Version entry (if present) do not exceed 1.7 for PDF/A-2/3.
Per ISO 32000-1 §7.2.2, the effective PDF version is the higher of the
header and /Root/Version, so the catalog /Version must be removed or
overwritten to prevent it from elevating the effective version beyond
what ``force_version`` sets in the header.

ISO 19005-2, clauses 6.1.10–6.1.13 forbid the /Perms, /Requirements,
and /Collection keys in the Document Catalog.  /NeedsRendering
(ISO 32000-1, Table 28) is also removed as it is incompatible with
PDF/A archival requirements.

ISO 19005-2, clause 6.1.2 forbids the /ViewArea, /ViewClip, /PrintArea,
and /PrintClip keys in the /ViewerPreferences dictionary.

ISO 19005-2, clause 6.1.11 forbids /AlternatePresentations in
the document /Names dictionary.

Rule 6.10 forbids /PresSteps entries in Page dictionaries.

ISO 19005-2, clause 6.7.3 recommends the /Lang key in the Document
Catalog for accessibility and structural compliance.
"""

import logging
import re

from lxml import etree
from pikepdf import Dictionary, Pdf, String

from ..utils import resolve_indirect as _resolve_indirect

logger = logging.getLogger(__name__)

_SECURE_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)

# BCP 47 language tag pattern (RFC 5646).
# Covers: language[-extlang][-script][-region][-variant][-extension][-privateuse],
# privateuse-only tags (x-...), and grandfathered tags.
_GRANDFATHERED = (
    "en-GB-oed|i-ami|i-bnn|i-default|i-enochian|i-hak|i-klingon|"
    "i-lux|i-mingo|i-navajo|i-no|i-pwn|i-tao|i-tay|i-tsu|"
    "sgn-BE-FR|sgn-BE-NL|sgn-CH-DE|"
    "art-lojban|cel-gaulish|no-bok|no-nyn|"
    "zh-guoyu|zh-hakka|zh-min|zh-min-nan|zh-xiang"
)
_BCP47_RE = re.compile(
    r"^(?:"
    r"[A-Za-z]{2,3}(?:-[A-Za-z]{3}){0,3}"  # language [-extlang]
    r"(?:-[A-Za-z]{4})?"  # [-script]
    r"(?:-(?:[A-Za-z]{2}|[0-9]{3}))?"  # [-region]
    r"(?:-(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3}))*"  # [-variant]
    r"(?:-[A-Za-z0-9](?:-[A-Za-z0-9]{2,8})+)*"  # [-extension]
    r"(?:-x(?:-[A-Za-z0-9]{1,8})+)?"  # [-privateuse]
    r"|x(?:-[A-Za-z0-9]{1,8})+"  # privateuse
    r"|" + _GRANDFATHERED + r")$"
)


def _is_valid_bcp47(tag: str) -> bool:
    """Check whether *tag* is a syntactically valid BCP 47 language tag."""
    return bool(_BCP47_RE.match(tag))


# Catalog keys forbidden by ISO 19005-2, clauses 6.1.10–6.1.13,
# plus /NeedsRendering (ISO 32000-1, Table 28),
# /Threads (article threads) and /SpiderInfo (web capture)
FORBIDDEN_CATALOG_KEYS = (
    "/Perms",
    "/Requirements",
    "/Collection",
    "/NeedsRendering",
    "/Threads",
    "/SpiderInfo",
)


# ViewerPreferences keys forbidden by ISO 19005-2, clause 6.1.2
# (deprecated since PDF 2.0)
FORBIDDEN_VIEWER_PREF_KEYS = ("/ViewArea", "/ViewClip", "/PrintArea", "/PrintClip")


# Name dictionary keys forbidden by ISO 19005-2, clause 6.1.11
FORBIDDEN_NAMES_DICT_KEYS = ("/AlternatePresentations",)


# Page dictionary keys forbidden by Rule 6.10,
# plus /Duration (presentation timing)
FORBIDDEN_PAGE_DICT_KEYS = ("/PresSteps", "/Duration")


def remove_catalog_version(pdf: Pdf, required_version: str) -> bool:
    """Remove or overwrite /Version in the Document Catalog.

    ISO 32000-1 §7.2.2 states that the effective PDF version is the
    higher of the file header version and the /Version entry in the
    Catalog dictionary.  ``pikepdf.Pdf.save(force_version=...)`` only
    sets the header, so a leftover /Version that is higher than the
    required version would cause the effective version to exceed the
    PDF/A-2/3 limit of 1.7 (ISO 19005-2, clause 6.1.2).

    If /Version is present and exceeds *required_version*, it is
    overwritten with *required_version*.  If it equals the header
    version it is simply removed (redundant).

    Args:
        pdf: Opened pikepdf PDF object (modified in place).
        required_version: The target PDF version string (e.g. ``"1.7"``).

    Returns:
        True if /Version was removed or overwritten, False if no change
        was needed.
    """
    try:
        cat_version = pdf.Root.get("/Version")
    except Exception:
        return False

    if cat_version is None:
        return False

    cat_version_str = str(cat_version).lstrip("/")

    if cat_version_str > required_version:
        del pdf.Root["/Version"]
        logger.info(
            "Catalog /Version /%s removed (exceeds required %s)",
            cat_version_str,
            required_version,
        )
        return True

    if cat_version_str == required_version:
        # Redundant — header already carries this version.
        del pdf.Root["/Version"]
        logger.debug(
            "Catalog /Version /%s removed (redundant with header)",
            cat_version_str,
        )
        return True

    # /Version is lower than required — harmless, leave it.
    return False


def remove_forbidden_viewer_preferences(pdf: Pdf) -> int:
    """Remove forbidden keys from the /ViewerPreferences dictionary.

    ISO 19005-2, clause 6.1.2 forbids the following keys in
    /ViewerPreferences:
    - /ViewArea
    - /ViewClip
    - /PrintArea
    - /PrintClip

    If the dictionary is empty after removal it is removed entirely.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of forbidden ViewerPreferences entries removed.
    """
    try:
        vp = pdf.Root.get("/ViewerPreferences")
    except Exception:
        return 0

    if vp is None:
        return 0

    removed_count = 0
    for key in FORBIDDEN_VIEWER_PREF_KEYS:
        if key in vp:
            del vp[key]
            removed_count += 1
            logger.debug("Forbidden ViewerPreferences entry %s removed", key)

    if removed_count > 0:
        logger.info(
            "%d forbidden ViewerPreferences entry/entries removed",
            removed_count,
        )

    # Remove empty ViewerPreferences dict
    if len(vp) == 0:
        del pdf.Root["/ViewerPreferences"]
        logger.debug("Empty /ViewerPreferences dictionary removed")

    return removed_count


def remove_forbidden_catalog_entries(pdf: Pdf) -> int:
    """Removes forbidden entries from the Document Catalog.

    PDF/A-2 and PDF/A-3 forbid the following keys in the Catalog:
    - /Perms (clause 6.1.12)
    - /Requirements (clause 6.1.10)
    - /Collection (clause 6.1.13)
    - /NeedsRendering (ISO 32000-1, Table 28)
    - /Threads (article threads, no archival purpose)
    - /SpiderInfo (web capture metadata, no archival purpose)

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of forbidden catalog entries removed.
    """
    removed_count = 0

    for key in FORBIDDEN_CATALOG_KEYS:
        if key in pdf.Root:
            del pdf.Root[key]
            removed_count += 1
            logger.debug("Forbidden catalog entry %s removed", key)

    if removed_count > 0:
        logger.info("%d forbidden catalog entry/entries removed", removed_count)
    return removed_count


def remove_forbidden_name_dictionary_entries(pdf: Pdf) -> int:
    """Remove forbidden entries from the document /Names dictionary.

    PDF/A forbids:
    - /AlternatePresentations (ISO 19005-2, clause 6.1.11)

    If /Names becomes empty after cleanup, the key is removed from
    the Catalog.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of forbidden name dictionary entries removed.
    """
    names_dict = pdf.Root.get("/Names")
    if names_dict is None:
        return 0
    names_dict = _resolve_indirect(names_dict)

    removed_count = 0
    for key in FORBIDDEN_NAMES_DICT_KEYS:
        if key in names_dict:
            del names_dict[key]
            removed_count += 1
            logger.debug("Forbidden /Names entry %s removed", key)

    if removed_count > 0 and len(names_dict) == 0:
        del pdf.Root["/Names"]
        logger.debug("Empty /Names dictionary removed from catalog")

    if removed_count > 0:
        logger.info(
            "%d forbidden /Names dictionary entry/entries removed",
            removed_count,
        )

    return removed_count


def remove_forbidden_page_entries(pdf: Pdf) -> int:
    """Remove forbidden entries from all Page dictionaries.

    PDF/A Rule 6.10 forbids:
    - /PresSteps
    - /Duration (presentation timing)

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Number of forbidden Page dictionary entries removed.
    """
    removed_count = 0
    for page in pdf.pages:
        page_dict = page.obj if hasattr(page, "obj") else page
        for key in FORBIDDEN_PAGE_DICT_KEYS:
            if key in page_dict:
                del page_dict[key]
                removed_count += 1
                logger.debug("Forbidden Page entry %s removed", key)

    if removed_count > 0:
        logger.info(
            "%d forbidden Page dictionary entry/entries removed",
            removed_count,
        )

    return removed_count


# XMP namespace for dc:language extraction
_DC_NS = "http://purl.org/dc/elements/1.1/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _extract_lang_from_xmp(pdf: Pdf) -> str | None:
    """Extract the first dc:language value from XMP metadata.

    Returns:
        BCP 47 language tag string, or None if not found.
    """
    try:
        metadata_obj = pdf.Root.get("/Metadata")
        if metadata_obj is None:
            return None
        xmp_bytes = bytes(metadata_obj.read_bytes())
    except Exception:
        return None

    try:
        root = etree.fromstring(xmp_bytes, _SECURE_XML_PARSER)
    except etree.XMLSyntaxError:
        return None

    # Look for dc:language/rdf:Bag/rdf:li or dc:language/rdf:Seq/rdf:li
    for container_tag in ("Bag", "Seq"):
        elems = root.findall(
            f".//{{{_DC_NS}}}language/{{{_RDF_NS}}}{container_tag}/{{{_RDF_NS}}}li"
        )
        if elems:
            lang = (elems[0].text or "").strip()
            if lang:
                return lang

    return None


def ensure_mark_info(pdf: Pdf) -> bool:
    """Ensure /MarkInfo is present in the Document Catalog.

    ISO 19005-2, §6.7.1 requires /MarkInfo with /Marked true for "a"
    levels.  For "b"/"u" levels, best practice is to include /MarkInfo
    with /Marked false.  Several PDF/A validators warn about its absence.

    If /MarkInfo does not exist, it is created with ``/Marked false``.
    If /MarkInfo exists but has no /Marked key, ``/Marked false`` is added.
    An existing ``/Marked true`` is never changed to false (the PDF may
    be tagged).

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        True if /MarkInfo was created or /Marked was added, False if
        no change was needed.
    """
    try:
        mark_info = pdf.Root.get("/MarkInfo")
    except Exception:
        mark_info = None

    if mark_info is None:
        pdf.Root["/MarkInfo"] = Dictionary(Marked=False)
        logger.info("Added /MarkInfo with /Marked false to catalog")
        return True

    mark_info = _resolve_indirect(mark_info)

    if "/Marked" not in mark_info:
        mark_info["/Marked"] = False
        logger.info("Added /Marked false to existing /MarkInfo")
        return True

    logger.debug("/MarkInfo already has /Marked key: %s", mark_info.get("/Marked"))
    return False


def ensure_catalog_lang(pdf: Pdf) -> bool:
    """Ensure the /Lang key is present in the Document Catalog.

    If /Lang already exists it is kept unchanged.  Otherwise the language
    is extracted from existing XMP metadata (``dc:language``).  If that is
    also absent the BCP 47 code ``"und"`` (undetermined) is used as
    fallback.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        True if /Lang was set by this function, False if it was already
        present.
    """
    if "/Lang" in pdf.Root:
        logger.debug("/Lang already present in catalog: %s", pdf.Root["/Lang"])
        return False

    lang = _extract_lang_from_xmp(pdf)
    if lang and not _is_valid_bcp47(lang):
        logger.warning(
            "Invalid BCP 47 language tag %r from XMP; falling back to 'und'",
            lang,
        )
        lang = None
    lang = lang or "und"
    pdf.Root["/Lang"] = String(lang)
    logger.info("Set /Lang in catalog to %r", lang)
    return True
