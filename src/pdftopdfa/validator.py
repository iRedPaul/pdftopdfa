# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PDF/A validation utilities for PDF documents.

This module provides helper functions for PDF/A conformance checks.
For complete ISO-compliant validation, use the verapdf module which
integrates with veraPDF: https://verapdf.org/
"""

import logging
from dataclasses import dataclass

import pikepdf
from lxml import etree

from .metadata import NAMESPACES

logger = logging.getLogger(__name__)

_SECURE_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)


def _extract_xmp_bytes(pdf: pikepdf.Pdf) -> bytes | None:
    """Extracts XMP metadata bytes from the PDF.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        XMP metadata as bytes or None if not present.
    """
    try:
        metadata = pdf.Root.get("/Metadata")
        if metadata is None:
            return None

        # Dereference if necessary
        try:
            metadata = metadata.get_object()
        except (AttributeError, ValueError, TypeError):
            pass

        # Read stream data
        return bytes(metadata.read_bytes())
    except Exception as e:
        logger.debug("Error extracting XMP metadata: %s", e)
        return None


def _parse_xmp_tree(xmp_bytes: bytes) -> etree._Element | None:
    """Parses XMP XML from the metadata bytes.

    Handles the XMP packet wrapper format (<?xpacket...?>).

    Args:
        xmp_bytes: Raw XMP metadata bytes.

    Returns:
        Parsed XML tree or None on errors.
    """
    try:
        # Remove <?xpacket...?> header and trailer if present
        content = xmp_bytes
        if b"<?xpacket" in content:
            # Find start after the header
            start_idx = content.find(b"?>")
            if start_idx != -1:
                content = content[start_idx + 2 :]

            # Find end before the trailer
            end_idx = content.rfind(b"<?xpacket")
            if end_idx != -1:
                content = content[:end_idx]

        # Parse XML
        content = content.strip()
        if not content:
            return None

        return etree.fromstring(content, _SECURE_XML_PARSER)
    except etree.XMLSyntaxError as e:
        logger.debug("XMP XML parsing error: %s", e)
        return None
    except Exception as e:
        logger.debug("Error parsing XMP metadata: %s", e)
        return None


def _get_pdfa_identification(
    tree: etree._Element,
) -> tuple[int | None, str | None]:
    """Extracts PDF/A identification from the XMP tree.

    Checks both element and attribute forms of pdfaid entries.

    Args:
        tree: Parsed XMP XML tree.

    Returns:
        Tuple of (part, conformance) or (None, None).
    """
    ns_pdfaid = NAMESPACES["pdfaid"]
    ns_rdf = NAMESPACES["rdf"]

    part: int | None = None
    conformance: str | None = None

    try:
        # Search in rdf:Description elements
        for desc in tree.iter(f"{{{ns_rdf}}}Description"):
            # Check element form: <pdfaid:part>2</pdfaid:part>
            part_elem = desc.find(f"{{{ns_pdfaid}}}part")
            if part_elem is not None and part_elem.text:
                try:
                    part = int(part_elem.text.strip())
                except ValueError:
                    pass

            conf_elem = desc.find(f"{{{ns_pdfaid}}}conformance")
            if conf_elem is not None and conf_elem.text:
                conformance = conf_elem.text.strip().upper()

            # Check attribute form: pdfaid:part="2" pdfaid:conformance="B"
            if part is None:
                part_attr = desc.get(f"{{{ns_pdfaid}}}part")
                if part_attr:
                    try:
                        part = int(part_attr.strip())
                    except ValueError:
                        pass

            if conformance is None:
                conf_attr = desc.get(f"{{{ns_pdfaid}}}conformance")
                if conf_attr:
                    conformance = conf_attr.strip().upper()

            # If both found, exit early
            if part is not None and conformance is not None:
                break

    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug("Error extracting PDF/A identification: %s", e)

    return part, conformance


def detect_pdfa_level(pdf: pikepdf.Pdf) -> str | None:
    """Detects the PDF/A conformance level from XMP metadata.

    Reads the pdfaid:part and pdfaid:conformance values from the
    XMP metadata and combines them into the level string.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        PDF/A level as string (e.g., "1b", "2b", "3a") or None
        if no PDF/A level was detected.

    Example:
        >>> with pikepdf.open("document.pdf") as pdf:
        ...     level = detect_pdfa_level(pdf)
        ...     if level:
        ...         print(f"PDF/A-{level}")
    """
    xmp_bytes = _extract_xmp_bytes(pdf)
    if xmp_bytes is None:
        logger.debug("No XMP metadata present")
        return None

    tree = _parse_xmp_tree(xmp_bytes)
    if tree is None:
        logger.debug("XMP metadata could not be parsed")
        return None

    part, conformance = _get_pdfa_identification(tree)

    if part is None:
        logger.debug(
            "PDF/A identification incomplete: part=%s, conformance=%s",
            part,
            conformance,
        )
        return None

    # PDF/A-4: conformance is E, F, or absent (base)
    if part == 4:
        if conformance is None or conformance == "":
            level = "4"
        elif conformance in ("E", "F"):
            level = f"4{conformance.lower()}"
        else:
            logger.warning("Invalid PDF/A-4 conformance: %s", conformance)
            return None
        logger.debug("PDF/A level detected: %s", level)
        return level

    # PDF/A-1/2/3: conformance is required
    if conformance is None:
        logger.debug(
            "PDF/A identification incomplete: part=%s, conformance=%s",
            part,
            conformance,
        )
        return None

    if part not in (1, 2, 3):
        logger.warning("Invalid PDF/A part: %s", part)
        return None

    if conformance not in ("A", "B", "U"):
        logger.warning("Invalid PDF/A conformance: %s", conformance)
        return None

    level = f"{part}{conformance.lower()}"
    logger.debug("PDF/A level detected: %s", level)
    return level


@dataclass
class ISOStandardInfo:
    """Information about a detected ISO PDF standard.

    Attributes:
        standard: Standard name (e.g. "PDF/X", "PDF/UA", "PDF/E", "PDF/VT").
        version: Version string as found in XMP (e.g. "PDF/X-4", "1", "2").
    """

    standard: str
    version: str


def _detect_iso_standard(
    tree: etree._Element,
    ns_key: str,
    element_name: str,
    standard: str,
) -> ISOStandardInfo | None:
    """Detects an ISO standard identification from XMP metadata."""
    ns = NAMESPACES[ns_key]
    ns_rdf = NAMESPACES["rdf"]

    for desc in tree.iter(f"{{{ns_rdf}}}Description"):
        # Element form
        elem = desc.find(f"{{{ns}}}{element_name}")
        if elem is not None and elem.text:
            return ISOStandardInfo(standard=standard, version=elem.text.strip())

        # Attribute form
        attr = desc.get(f"{{{ns}}}{element_name}")
        if attr:
            return ISOStandardInfo(standard=standard, version=attr.strip())

    return None


_ISO_STANDARD_DEFS = [
    ("pdfxid", "GTS_PDFXVersion", "PDF/X"),
    ("pdfuaid", "part", "PDF/UA"),
    ("pdfeid", "part", "PDF/E"),
    ("pdfvtid", "GTS_PDFVTVersion", "PDF/VT"),
]


def detect_iso_standards(pdf: pikepdf.Pdf) -> list[ISOStandardInfo]:
    """Detects ISO PDF standards (PDF/X, PDF/UA, PDF/E, PDF/VT) from XMP metadata.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        List of detected ISO standards (may be empty).
    """
    xmp_bytes = _extract_xmp_bytes(pdf)
    if xmp_bytes is None:
        return []

    tree = _parse_xmp_tree(xmp_bytes)
    if tree is None:
        return []

    standards: list[ISOStandardInfo] = []

    for ns_key, element_name, standard in _ISO_STANDARD_DEFS:
        try:
            result = _detect_iso_standard(tree, ns_key, element_name, standard)
            if result is not None:
                standards.append(result)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Error detecting ISO standard: %s", e)

    if standards:
        logger.debug(
            "ISO standards detected: %s",
            ", ".join(f"{s.standard} {s.version}" for s in standards),
        )

    return standards
