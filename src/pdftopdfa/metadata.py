# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""XMP metadata handling for PDF/A conversion."""

import copy
import logging
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pikepdf
from lxml import etree
from lxml.builder import ElementMaker

from .exceptions import ConversionError
from .utils import validate_pdfa_level

logger = logging.getLogger(__name__)

_SECURE_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)

# Regex matching control characters forbidden in XML 1.0
# (U+0000-U+0008, U+000B-U+000C, U+000E-U+001F)
_XML_ILLEGAL_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Regex for validating XMP Date (ISO 8601 subset)
_XMP_DATE_RE = re.compile(
    r"\d{4}"
    r"(-\d{2}"
    r"(-\d{2}"
    r"(T\d{2}:\d{2}"
    r"(:\d{2}(\.\d+)?)?"
    r"(Z|[+-]\d{2}:\d{2})?"
    r")?"
    r")?"
    r")?$"
)


def _sanitize_xml_text(text: str) -> str:
    """Remove control characters that are illegal in XML 1.0."""
    return _XML_ILLEGAL_CTRL_RE.sub("", text)


def _strip_xpacket_wrapper(content: bytes) -> bytes:
    """Strip XMP xpacket processing instructions and return inner content.

    Removes the ``<?xpacket begin=...?>`` header and ``<?xpacket end=...?>``
    trailer if present, returning the stripped and trimmed payload.
    """
    if b"<?xpacket" in content:
        start_idx = content.find(b"?>")
        if start_idx != -1:
            content = content[start_idx + 2 :]
        end_idx = content.rfind(b"<?xpacket")
        if end_idx != -1:
            content = content[:end_idx]
    return content.strip()


# XML namespaces for XMP metadata
NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "pdf": "http://ns.adobe.com/pdf/1.3/",
    "pdfaid": "http://www.aiim.org/pdfa/ns/id/",
    "pdfxid": "http://www.npes.org/pdfx/ns/id/",
    "pdfuaid": "http://www.aiim.org/pdfua/ns/id/",
    "pdfeid": "http://www.aiim.org/pdfe/ns/id/",
    "pdfvtid": "http://www.npes.org/pdfvt/ns/id/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
    "xmpTPg": "http://ns.adobe.com/xap/1.0/t/pg/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
    "stRef": "http://ns.adobe.com/xap/1.0/sType/ResourceRef#",
    "stFnt": "http://ns.adobe.com/xap/1.0/sType/Font#",
    "stDim": "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
    "stVer": "http://ns.adobe.com/xap/1.0/sType/Version#",
    "stJob": "http://ns.adobe.com/xap/1.0/sType/Job#",
}

# PDF/A Extension Schema namespaces
_NS_PDFA_EXTENSION = "http://www.aiim.org/pdfa/ns/extension/"
_NS_PDFA_SCHEMA = "http://www.aiim.org/pdfa/ns/schema#"
_NS_PDFA_PROPERTY = "http://www.aiim.org/pdfa/ns/property#"

# Register all namespaces globally so lxml serializes them with canonical prefixes
for _prefix, _uri in NAMESPACES.items():
    etree.register_namespace(_prefix, _uri)
etree.register_namespace("pdfaExtension", _NS_PDFA_EXTENSION)
etree.register_namespace("pdfaSchema", _NS_PDFA_SCHEMA)
etree.register_namespace("pdfaProperty", _NS_PDFA_PROPERTY)

# XMP packet header and trailer
XMP_HEADER = b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
XMP_TRAILER = b'\n<?xpacket end="w"?>'

# XMP property tags that create_xmp_metadata() writes fresh.
# Everything else found in existing XMP is preserved as-is.
_MANAGED_ELEMENTS = {
    f"{{{NAMESPACES['pdfaid']}}}part",
    f"{{{NAMESPACES['pdfaid']}}}conformance",
    f"{{{NAMESPACES['dc']}}}title",
    f"{{{NAMESPACES['dc']}}}creator",
    f"{{{NAMESPACES['dc']}}}description",
    f"{{{NAMESPACES['dc']}}}format",
    f"{{{NAMESPACES['xmp']}}}CreateDate",
    f"{{{NAMESPACES['xmp']}}}ModifyDate",
    f"{{{NAMESPACES['xmp']}}}MetadataDate",
    f"{{{NAMESPACES['xmp']}}}CreatorTool",
    f"{{{NAMESPACES['pdf']}}}Producer",
    f"{{{NAMESPACES['pdf']}}}Keywords",
    f"{{{NAMESPACES['pdf']}}}Trapped",
}

# Attribute-form equivalents of managed elements (Clark notation)
_MANAGED_ATTRS = _MANAGED_ELEMENTS


# Predefined XMP properties that do NOT need extension schema declarations.
# Map of namespace URI -> set of property local names.
_PREDEFINED_PROPERTIES: dict[str, set[str]] = {
    NAMESPACES["dc"]: {
        "title",
        "creator",
        "description",
        "subject",
        "publisher",
        "contributor",
        "date",
        "type",
        "format",
        "identifier",
        "source",
        "language",
        "relation",
        "coverage",
        "rights",
    },
    NAMESPACES["xmp"]: {
        "CreateDate",
        "ModifyDate",
        "MetadataDate",
        "CreatorTool",
        "Identifier",
        "Label",
        "Rating",
        "BaseURL",
        "Nickname",
        "Thumbnails",
    },
    NAMESPACES["pdf"]: {
        "Producer",
        "Keywords",
        "Trapped",
        "PDFVersion",
    },
    NAMESPACES["pdfaid"]: {
        "part",
        "conformance",
        "amd",
    },
    NAMESPACES["xmpRights"]: {
        "Certificate",
        "Marked",
        "Owner",
        "UsageTerms",
        "WebStatement",
    },
    NAMESPACES["xmpMM"]: {
        "DocumentID",
        "InstanceID",
        "OriginalDocumentID",
        "RenditionClass",
        "VersionID",
        "DerivedFrom",
        "History",
        "Ingredients",
        "ManagedFrom",
        "Manager",
        "ManageTo",
        "ManageURI",
        "Pantry",
    },
    NAMESPACES["xmpTPg"]: {
        "NPages",
        "MaxPageSize",
        "Fonts",
        "Colorants",
        "PlateNames",
    },
    NAMESPACES["photoshop"]: {
        "AuthorsPosition",
        "CaptionWriter",
        "Category",
        "City",
        "Country",
        "Credit",
        "DateCreated",
        "Headline",
        "History",
        "Instructions",
        "Source",
        "State",
        "SupplementalCategories",
        "TransmissionReference",
        "Urgency",
        "ICCProfile",
        "ColorMode",
        "DocumentAncestors",
    },
    NAMESPACES["tiff"]: {
        "ImageWidth",
        "ImageLength",
        "BitsPerSample",
        "Compression",
        "PhotometricInterpretation",
        "Orientation",
        "SamplesPerPixel",
        "PlanarConfiguration",
        "YCbCrSubSampling",
        "XResolution",
        "YResolution",
        "ResolutionUnit",
        "TransferFunction",
        "WhitePoint",
        "PrimaryChromaticities",
        "YCbCrCoefficients",
        "ReferenceBlackWhite",
        "DateTime",
        "ImageDescription",
        "Make",
        "Model",
        "Software",
        "Artist",
        "Copyright",
        "NativeDigest",
    },
    NAMESPACES["exif"]: {
        "ExifVersion",
        "FlashpixVersion",
        "ColorSpace",
        "ComponentsConfiguration",
        "CompressedBitsPerPixel",
        "PixelXDimension",
        "PixelYDimension",
        "UserComment",
        "RelatedSoundFile",
        "DateTimeOriginal",
        "DateTimeDigitized",
        "ExposureTime",
        "FNumber",
        "ExposureProgram",
        "SpectralSensitivity",
        "ISOSpeedRatings",
        "OECF",
        "ShutterSpeedValue",
        "ApertureValue",
        "BrightnessValue",
        "ExposureBiasValue",
        "MaxApertureValue",
        "SubjectDistance",
        "MeteringMode",
        "LightSource",
        "Flash",
        "FocalLength",
        "SubjectArea",
        "FlashEnergy",
        "SpatialFrequencyResponse",
        "FocalPlaneXResolution",
        "FocalPlaneYResolution",
        "FocalPlaneResolutionUnit",
        "SubjectLocation",
        "ExposureIndex",
        "SensingMethod",
        "FileSource",
        "SceneType",
        "CFAPattern",
        "CustomRendered",
        "ExposureMode",
        "WhiteBalance",
        "DigitalZoomRatio",
        "FocalLengthIn35mmFilm",
        "SceneCaptureType",
        "GainControl",
        "Contrast",
        "Saturation",
        "Sharpness",
        "DeviceSettingDescription",
        "SubjectDistanceRange",
        "ImageUniqueID",
        "GPSVersionID",
        "NativeDigest",
    },
}

# Namespace URI aliases for _PREDEFINED_PROPERTY_TYPES
_DC = NAMESPACES["dc"]
_XMP = NAMESPACES["xmp"]
_PDF = NAMESPACES["pdf"]
_PDFAID = NAMESPACES["pdfaid"]
_XMPRIGHTS = NAMESPACES["xmpRights"]
_XMPMM = NAMESPACES["xmpMM"]
_XMPTPG = NAMESPACES["xmpTPg"]
_PHOTOSHOP = NAMESPACES["photoshop"]
_TIFF = NAMESPACES["tiff"]
_EXIF = NAMESPACES["exif"]

# Expected XMP structure type for each predefined property.
# s=simple, i=integer, r=rational, d=Date, B=Boolean,
# b=Bag, q=Seq, qi=Seq Integer, qr=Seq Rational, qd=Seq Date,
# a=Alt, la=Lang Alt (requires xml:lang), x=struct
_PREDEFINED_PROPERTY_TYPES: dict[tuple[str, str], str] = {
    # dc (Dublin Core)
    (_DC, "title"): "la",
    (_DC, "creator"): "q",
    (_DC, "description"): "la",
    (_DC, "subject"): "b",
    (_DC, "publisher"): "b",
    (_DC, "contributor"): "b",
    (_DC, "date"): "qd",
    (_DC, "type"): "b",
    (_DC, "format"): "s",
    (_DC, "identifier"): "s",
    (_DC, "source"): "s",
    (_DC, "language"): "b",
    (_DC, "relation"): "b",
    (_DC, "coverage"): "s",
    (_DC, "rights"): "la",
    # xmp
    (_XMP, "CreateDate"): "d",
    (_XMP, "ModifyDate"): "d",
    (_XMP, "MetadataDate"): "d",
    (_XMP, "CreatorTool"): "s",
    (_XMP, "Identifier"): "b",
    (_XMP, "Label"): "s",
    (_XMP, "Rating"): "s",
    (_XMP, "BaseURL"): "s",
    (_XMP, "Nickname"): "s",
    (_XMP, "Thumbnails"): "a",
    # pdf
    (_PDF, "Producer"): "s",
    (_PDF, "Keywords"): "s",
    (_PDF, "Trapped"): "s",
    (_PDF, "PDFVersion"): "s",
    # pdfaid
    (_PDFAID, "part"): "i",
    (_PDFAID, "conformance"): "s",
    (_PDFAID, "amd"): "s",
    # xmpRights
    (_XMPRIGHTS, "Certificate"): "s",
    (_XMPRIGHTS, "Marked"): "B",
    (_XMPRIGHTS, "Owner"): "b",
    (_XMPRIGHTS, "UsageTerms"): "la",
    (_XMPRIGHTS, "WebStatement"): "s",
    # xmpMM
    (_XMPMM, "DocumentID"): "s",
    (_XMPMM, "InstanceID"): "s",
    (_XMPMM, "OriginalDocumentID"): "s",
    (_XMPMM, "RenditionClass"): "s",
    (_XMPMM, "VersionID"): "s",
    (_XMPMM, "DerivedFrom"): "x",
    (_XMPMM, "History"): "q",
    (_XMPMM, "Ingredients"): "b",
    (_XMPMM, "ManagedFrom"): "x",
    (_XMPMM, "Manager"): "s",
    (_XMPMM, "ManageTo"): "s",
    (_XMPMM, "ManageURI"): "s",
    (_XMPMM, "Pantry"): "b",
    # xmpTPg
    (_XMPTPG, "NPages"): "i",
    (_XMPTPG, "MaxPageSize"): "x",
    (_XMPTPG, "Fonts"): "b",
    (_XMPTPG, "Colorants"): "q",
    (_XMPTPG, "PlateNames"): "q",
    # photoshop
    (_PHOTOSHOP, "AuthorsPosition"): "s",
    (_PHOTOSHOP, "CaptionWriter"): "s",
    (_PHOTOSHOP, "Category"): "s",
    (_PHOTOSHOP, "City"): "s",
    (_PHOTOSHOP, "Country"): "s",
    (_PHOTOSHOP, "Credit"): "s",
    (_PHOTOSHOP, "DateCreated"): "d",
    (_PHOTOSHOP, "Headline"): "s",
    (_PHOTOSHOP, "History"): "s",
    (_PHOTOSHOP, "Instructions"): "s",
    (_PHOTOSHOP, "Source"): "s",
    (_PHOTOSHOP, "State"): "s",
    (_PHOTOSHOP, "SupplementalCategories"): "b",
    (_PHOTOSHOP, "TransmissionReference"): "s",
    (_PHOTOSHOP, "Urgency"): "i",
    (_PHOTOSHOP, "ICCProfile"): "s",
    (_PHOTOSHOP, "ColorMode"): "i",
    (_PHOTOSHOP, "DocumentAncestors"): "b",
    # tiff
    (_TIFF, "ImageWidth"): "i",
    (_TIFF, "ImageLength"): "i",
    (_TIFF, "BitsPerSample"): "qi",
    (_TIFF, "Compression"): "i",
    (_TIFF, "PhotometricInterpretation"): "i",
    (_TIFF, "Orientation"): "i",
    (_TIFF, "SamplesPerPixel"): "i",
    (_TIFF, "PlanarConfiguration"): "i",
    (_TIFF, "YCbCrSubSampling"): "qi",
    (_TIFF, "XResolution"): "r",
    (_TIFF, "YResolution"): "r",
    (_TIFF, "ResolutionUnit"): "i",
    (_TIFF, "TransferFunction"): "qi",
    (_TIFF, "WhitePoint"): "qr",
    (_TIFF, "PrimaryChromaticities"): "qr",
    (_TIFF, "YCbCrCoefficients"): "qr",
    (_TIFF, "ReferenceBlackWhite"): "qr",
    (_TIFF, "DateTime"): "d",
    (_TIFF, "ImageDescription"): "la",
    (_TIFF, "Make"): "s",
    (_TIFF, "Model"): "s",
    (_TIFF, "Software"): "s",
    (_TIFF, "Artist"): "s",
    (_TIFF, "Copyright"): "la",
    (_TIFF, "NativeDigest"): "s",
    # exif
    (_EXIF, "ExifVersion"): "s",
    (_EXIF, "FlashpixVersion"): "s",
    (_EXIF, "ColorSpace"): "i",
    (_EXIF, "ComponentsConfiguration"): "qi",
    (_EXIF, "CompressedBitsPerPixel"): "r",
    (_EXIF, "PixelXDimension"): "i",
    (_EXIF, "PixelYDimension"): "i",
    (_EXIF, "UserComment"): "la",
    (_EXIF, "RelatedSoundFile"): "s",
    (_EXIF, "DateTimeOriginal"): "d",
    (_EXIF, "DateTimeDigitized"): "d",
    (_EXIF, "ExposureTime"): "r",
    (_EXIF, "FNumber"): "r",
    (_EXIF, "ExposureProgram"): "i",
    (_EXIF, "SpectralSensitivity"): "s",
    (_EXIF, "ISOSpeedRatings"): "qi",
    (_EXIF, "OECF"): "x",
    (_EXIF, "ShutterSpeedValue"): "r",
    (_EXIF, "ApertureValue"): "r",
    (_EXIF, "BrightnessValue"): "r",
    (_EXIF, "ExposureBiasValue"): "r",
    (_EXIF, "MaxApertureValue"): "r",
    (_EXIF, "SubjectDistance"): "r",
    (_EXIF, "MeteringMode"): "i",
    (_EXIF, "LightSource"): "i",
    (_EXIF, "Flash"): "x",
    (_EXIF, "FocalLength"): "r",
    (_EXIF, "SubjectArea"): "qi",
    (_EXIF, "FlashEnergy"): "r",
    (_EXIF, "SpatialFrequencyResponse"): "x",
    (_EXIF, "FocalPlaneXResolution"): "r",
    (_EXIF, "FocalPlaneYResolution"): "r",
    (_EXIF, "FocalPlaneResolutionUnit"): "i",
    (_EXIF, "SubjectLocation"): "qi",
    (_EXIF, "ExposureIndex"): "r",
    (_EXIF, "SensingMethod"): "i",
    (_EXIF, "FileSource"): "i",
    (_EXIF, "SceneType"): "i",
    (_EXIF, "CFAPattern"): "x",
    (_EXIF, "CustomRendered"): "i",
    (_EXIF, "ExposureMode"): "i",
    (_EXIF, "WhiteBalance"): "i",
    (_EXIF, "DigitalZoomRatio"): "r",
    (_EXIF, "FocalLengthIn35mmFilm"): "i",
    (_EXIF, "SceneCaptureType"): "i",
    (_EXIF, "GainControl"): "i",
    (_EXIF, "Contrast"): "i",
    (_EXIF, "Saturation"): "i",
    (_EXIF, "Sharpness"): "i",
    (_EXIF, "DeviceSettingDescription"): "x",
    (_EXIF, "SubjectDistanceRange"): "i",
    (_EXIF, "ImageUniqueID"): "s",
    (_EXIF, "GPSVersionID"): "s",
    (_EXIF, "NativeDigest"): "s",
}

# Non-standard property names in structural namespaces -> corrected form
_STRUCTURAL_PROPERTY_CORRECTIONS: dict[str, str] = {
    f"{{{NAMESPACES['stEvt']}}}When": f"{{{NAMESPACES['stEvt']}}}when",
}


def _normalize_structural_properties(elem: etree._Element) -> None:
    """Recursively correct non-standard property names in structural types."""
    corrected = _STRUCTURAL_PROPERTY_CORRECTIONS.get(elem.tag)
    if corrected is not None:
        elem.tag = corrected
    for child in elem:
        _normalize_structural_properties(child)


# Structural namespaces that never contain user properties (skip during scanning)
_STRUCTURAL_NAMESPACES: frozenset[str] = frozenset(
    {
        NAMESPACES["rdf"],
        NAMESPACES["x"],
        "http://www.w3.org/XML/1998/namespace",
        "http://www.w3.org/2000/xmlns/",
        _NS_PDFA_EXTENSION,
        _NS_PDFA_SCHEMA,
        _NS_PDFA_PROPERTY,
        "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
        "http://ns.adobe.com/xap/1.0/sType/Font#",
        "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
        "http://ns.adobe.com/xap/1.0/sType/ResourceRef#",
        "http://ns.adobe.com/xap/1.0/sType/Version#",
        "http://ns.adobe.com/xap/1.0/sType/Job#",
    }
)

# Map from XMP extension schema valueType to internal type codes used by
# _is_valid_simple_value().  Only simple (non-container) types are listed.
_EXTENSION_VALUE_TYPE_MAP: dict[str, str] = {
    "Integer": "i",
    "Text": "s",
    "Boolean": "B",
    "Date": "d",
    "Rational": "r",
}

# Known extension schemas with full property definitions.
# namespace URI -> (schema_name, prefix, {prop: (valType, cat, desc)})
_KNOWN_EXTENSION_SCHEMAS: dict[
    str, tuple[str, str, dict[str, tuple[str, str, str]]]
] = {
    NAMESPACES["pdfuaid"]: (
        "PDF/UA Universal Accessibility",
        "pdfuaid",
        {
            "part": ("Integer", "internal", "PDF/UA version identifier"),
            "rev": ("Integer", "internal", "PDF/UA revision year"),
        },
    ),
    NAMESPACES["pdfxid"]: (
        "PDF/X ID",
        "pdfxid",
        {
            "GTS_PDFXVersion": ("Text", "internal", "PDF/X version identifier"),
            "GTS_PDFXConformance": ("Text", "internal", "PDF/X conformance level"),
        },
    ),
    NAMESPACES["pdfeid"]: (
        "PDF/E ID",
        "pdfeid",
        {
            "GTS_PDFEVersion": ("Text", "internal", "PDF/E version identifier"),
            "part": ("Integer", "internal", "PDF/E part number"),
        },
    ),
    NAMESPACES["pdfvtid"]: (
        "PDF/VT ID",
        "pdfvtid",
        {
            "GTS_PDFVTVersion": ("Text", "internal", "PDF/VT version identifier"),
        },
    ),
    NAMESPACES["pdfaid"]: (
        "PDF/A ID",
        "pdfaid",
        {
            "rev": ("Integer", "internal", "PDF/A revision year"),
            "corr": ("Text", "internal", "PDF/A corrigendum identifier"),
        },
    ),
}


def _collect_non_predefined_properties(
    description: etree._Element,
) -> dict[str, set[str]]:
    """Scan an rdf:Description for properties that need extension schemas.

    Checks both child elements and attributes.  Returns a dict of
    namespace_uri -> {property_local_names} for properties that are NOT
    predefined in the standard XMP / PDF / PDF/A schemas.
    """
    ns_rdf = NAMESPACES["rdf"]
    result: dict[str, set[str]] = {}

    def _needs_extension(uri: str, local_name: str) -> bool:
        if uri in _STRUCTURAL_NAMESPACES:
            return False
        predefined = _PREDEFINED_PROPERTIES.get(uri)
        if predefined is not None and local_name in predefined:
            return False
        if predefined is not None and local_name not in predefined:
            # Known namespace but non-predefined property (e.g. pdfaid:rev)
            return True
        # Completely unknown namespace
        return True

    # Scan child elements
    for child in description:
        tag = child.tag
        if not isinstance(tag, str) or not tag.startswith("{"):
            continue
        uri, local = tag[1:].split("}", 1)
        if _needs_extension(uri, local):
            result.setdefault(uri, set()).add(local)

    # Scan attributes
    for attr_name in description.attrib:
        if not attr_name.startswith("{"):
            continue
        uri, local = attr_name[1:].split("}", 1)
        if uri == ns_rdf:
            continue
        if _needs_extension(uri, local):
            result.setdefault(uri, set()).add(local)

    return result


def _get_declared_namespace_uris(
    description: etree._Element,
) -> set[str]:
    """Extract namespace URIs already declared in existing extension schemas."""
    ns_rdf = NAMESPACES["rdf"]
    ext_tag = f"{{{_NS_PDFA_EXTENSION}}}schemas"
    ns_uri_tag = f"{{{_NS_PDFA_SCHEMA}}}namespaceURI"
    declared: set[str] = set()

    for schemas_elem in description.findall(ext_tag):
        for bag in schemas_elem.findall(f"{{{ns_rdf}}}Bag"):
            for li in bag.findall(f"{{{ns_rdf}}}li"):
                ns_uri_elem = li.find(ns_uri_tag)
                if ns_uri_elem is not None and ns_uri_elem.text:
                    declared.add(ns_uri_elem.text)

    return declared


def _extract_extension_schema_blocks(
    old_tree: etree._Element,
) -> dict[str, etree._Element]:
    """Extract per-namespace extension schema rdf:li blocks from existing XMP.

    Returns a dict of namespace_uri -> deep-copied rdf:li element for each
    schema block found in the existing XMP's pdfaExtension:schemas.
    """
    ns_rdf = NAMESPACES["rdf"]
    ext_tag = f"{{{_NS_PDFA_EXTENSION}}}schemas"
    ns_uri_tag = f"{{{_NS_PDFA_SCHEMA}}}namespaceURI"
    result: dict[str, etree._Element] = {}

    for rdf_root in old_tree.iter(f"{{{ns_rdf}}}RDF"):
        for desc in rdf_root.findall(f"{{{ns_rdf}}}Description"):
            for schemas_elem in desc.findall(ext_tag):
                for bag in schemas_elem.findall(f"{{{ns_rdf}}}Bag"):
                    for li in bag.findall(f"{{{ns_rdf}}}li"):
                        ns_uri_elem = li.find(ns_uri_tag)
                        if ns_uri_elem is not None and ns_uri_elem.text:
                            result[ns_uri_elem.text] = copy.deepcopy(li)

    return result


def _detect_structure(elem: etree._Element, ns_rdf: str) -> str:
    """Detect the actual XMP structure type of a property element.

    Returns: "s" (simple), "q" (Seq), "b" (Bag), "a" (Alt), "x" (struct)
    """
    for child in elem:
        tag = child.tag
        if tag == f"{{{ns_rdf}}}Seq":
            return "q"
        if tag == f"{{{ns_rdf}}}Bag":
            return "b"
        if tag == f"{{{ns_rdf}}}Alt":
            return "a"
    if elem.get(f"{{{ns_rdf}}}parseType") == "Resource":
        return "x"
    # Child elements (non-rdf) indicate struct without parseType
    for child in elem:
        if isinstance(child.tag, str) and not child.tag.startswith(f"{{{ns_rdf}}}"):
            return "x"
    return "s"


def _has_undeclarable_structure(elem: etree._Element) -> bool:
    """Check if property uses structured types that can't be declared."""
    ns_rdf = NAMESPACES["rdf"]
    rdf_desc_tag = f"{{{ns_rdf}}}Description"
    # Direct Resource type
    if elem.get(f"{{{ns_rdf}}}parseType") == "Resource":
        return True
    # Explicit rdf:Description children (equivalent to parseType="Resource")
    for child in elem:
        if child.tag == rdf_desc_tag:
            return True
    # Container with Resource items (Bag/Seq/Alt of structs)
    for child in elem:
        if child.tag in (
            f"{{{ns_rdf}}}Seq",
            f"{{{ns_rdf}}}Bag",
            f"{{{ns_rdf}}}Alt",
        ):
            for li in child:
                if li.get(f"{{{ns_rdf}}}parseType") == "Resource":
                    return True
                # rdf:Description in container items
                for sub in li:
                    if sub.tag == rdf_desc_tag:
                        return True
    return False


def _is_valid_simple_value(text: str, type_code: str) -> bool:
    """Validate a simple text value against its expected type code."""
    if not text:
        return False
    if type_code == "i":
        return text.lstrip("-").isdigit()
    if type_code == "r":
        parts = text.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return False
        return parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit()
    if type_code == "d":
        return bool(_XMP_DATE_RE.match(text))
    if type_code == "B":
        return text in ("True", "False")
    return True  # "s" — any non-empty text


def _validate_seq_items(
    elem: etree._Element,
    ns_rdf: str,
    item_type: str,
) -> bool:
    """Check that Seq items match the expected simple type."""
    for child in elem:
        if child.tag == f"{{{ns_rdf}}}Seq":
            for li in child:
                if li.tag != f"{{{ns_rdf}}}li":
                    continue
                text = (li.text or "").strip()
                if not _is_valid_simple_value(text, item_type):
                    return False
            return True
    return False  # No Seq found


def _validate_alt_lang(elem: etree._Element, ns_rdf: str) -> bool:
    """Check that Lang Alt items have xml:lang attributes."""
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
    for child in elem:
        if child.tag == f"{{{ns_rdf}}}Alt":
            for li in child:
                if li.tag != f"{{{ns_rdf}}}li":
                    continue
                if li.get(xml_lang) is None:
                    return False
            return True
    return False  # No Alt found


def _get_extension_type_code(uri: str, local_name: str) -> str | None:
    """Look up the internal type code for a known extension schema property.

    Returns the type code (e.g. "i" for Integer) or None if the property
    is not in _KNOWN_EXTENSION_SCHEMAS or has no simple-type mapping.
    """
    ext_schema = _KNOWN_EXTENSION_SCHEMAS.get(uri)
    if ext_schema is not None:
        prop_def = ext_schema[2].get(local_name)
        if prop_def is not None:
            return _EXTENSION_VALUE_TYPE_MAP.get(prop_def[0])
    return None


def _is_valid_preserved_property(
    elem: etree._Element,
    uri: str,
    local_name: str,
) -> bool:
    """Check if a preserved property conforms to its predefined schema type.

    Returns True if valid or if no type info is available (unknown property).
    Returns False if structure/value violates the expected type.
    """
    expected = _PREDEFINED_PROPERTY_TYPES.get((uri, local_name))
    if expected is None:
        # Check extension schema types (e.g. pdfaid:rev -> Integer)
        ext_code = _get_extension_type_code(uri, local_name)
        if ext_code is not None:
            ns_rdf = NAMESPACES["rdf"]
            actual = _detect_structure(elem, ns_rdf)
            if actual != "s":
                return False
            text = (elem.text or "").strip()
            return _is_valid_simple_value(text, ext_code)
        # Not a known predefined property — check for undeclarable structures
        return not _has_undeclarable_structure(elem)

    ns_rdf = NAMESPACES["rdf"]
    actual = _detect_structure(elem, ns_rdf)

    # xmp:Identifier is a bag of identifiers, not a bag of structured resources.
    # Resource items here lead to malformed RDF/XMP in downstream validators.
    if (uri, local_name) == (_XMP, "Identifier"):
        if actual != "b":
            return False
        return not _has_undeclarable_structure(elem)

    # Lang Alt — Alt container with xml:lang on items
    if expected == "la":
        if actual != "a":
            return False
        return _validate_alt_lang(elem, ns_rdf)
    # General Alt or plain containers
    if expected in ("b", "q", "a"):
        return actual == expected
    # Seq with typed items (qi, qr, qd)
    if expected.startswith("q") and len(expected) == 2:
        if actual != "q":
            return False
        return _validate_seq_items(elem, ns_rdf, expected[1])
    if expected == "x":
        return actual == "x"
    # Expect simple value (s, i, r, d, B)
    if actual != "s":
        return False
    text = (elem.text or "").strip()
    return _is_valid_simple_value(text, expected)


def _infer_value_type(
    description: etree._Element,
    uri: str,
    prop_name: str,
) -> str:
    """Infer XMP valueType from the element structure in description.

    Checks whether the property element contains rdf:Seq, rdf:Bag, or
    rdf:Alt children, returning the appropriate XMP valueType string.
    Falls back to "Text" for simple/text values.
    """
    ns_rdf = NAMESPACES["rdf"]
    tag = f"{{{uri}}}{prop_name}"
    for child in description:
        if child.tag == tag:
            for sub in child:
                if sub.tag == f"{{{ns_rdf}}}Seq":
                    return "Seq Text"
                if sub.tag == f"{{{ns_rdf}}}Bag":
                    return "Bag Text"
                if sub.tag == f"{{{ns_rdf}}}Alt":
                    return "Alt Text"
            break
    return "Text"


def _build_extension_schemas(
    description: etree._Element,
    nsmap: dict[str, str] | None = None,
    extra_properties: dict[str, set[str]] | None = None,
    original_schema_blocks: dict[str, etree._Element] | None = None,
) -> etree._Element | None:
    """Build pdfaExtension:schemas element for non-predefined properties.

    Checks which namespaces already have extension schema declarations
    (from preserved source XMP) and only generates declarations for
    namespaces that are missing.

    Args:
        description: The rdf:Description element to scan for properties.
        nsmap: Optional namespace prefix mapping for URI->prefix lookup.
        extra_properties: Additional namespace_uri -> {property_names}
            from non-catalog XMP streams that need extension schemas.
        original_schema_blocks: Pre-extracted rdf:li elements from the
            original catalog XMP, keyed by namespace URI.  Used to
            preserve custom valueType declarations for non-catalog
            properties instead of generating potentially incorrect ones.

    Returns the element to append to the rdf:Description, or None if no
    additional extension schemas are needed.
    """
    needed = _collect_non_predefined_properties(description)
    if extra_properties:
        for uri, props in extra_properties.items():
            needed.setdefault(uri, set()).update(props)
    if not needed:
        return None

    # Check which namespaces already have extension schema declarations
    already_declared = _get_declared_namespace_uris(description)
    missing = {
        uri: props for uri, props in needed.items() if uri not in already_declared
    }
    if not missing:
        return None

    ns_rdf = NAMESPACES["rdf"]

    # Invert NAMESPACES for URI->prefix lookup
    uri_to_prefix = {uri: prefix for prefix, uri in NAMESPACES.items()}
    if nsmap:
        uri_to_prefix.update({uri: prefix for prefix, uri in nsmap.items()})

    schemas_elem = etree.Element(f"{{{_NS_PDFA_EXTENSION}}}schemas")
    bag = etree.SubElement(schemas_elem, f"{{{ns_rdf}}}Bag")

    for uri, props in sorted(missing.items()):
        # If the original catalog XMP had an extension schema block for
        # this namespace, reuse it (preserves custom valueTypes, etc.).
        if original_schema_blocks and uri in original_schema_blocks:
            bag.append(original_schema_blocks[uri])
            continue

        known = _KNOWN_EXTENSION_SCHEMAS.get(uri)
        if known is not None:
            schema_name, prefix, known_props = known
            # Only declare properties that are actually used
            prop_defs = {}
            for prop_name in sorted(props):
                if prop_name in known_props:
                    prop_defs[prop_name] = known_props[prop_name]
                else:
                    prop_defs[prop_name] = (
                        _infer_value_type(description, uri, prop_name),
                        "external",
                        f"{prop_name} property",
                    )
        else:
            # Unknown namespace — derive prefix and create generic schema
            prefix = uri_to_prefix.get(uri, "")
            if not prefix:
                # Try to derive from URI
                parts = uri.rstrip("/#").rsplit("/", 1)
                prefix = parts[-1] if len(parts) > 1 else "ns"
                prefix = re.sub(r"[^a-zA-Z0-9]", "", prefix).lower() or "ns"
            schema_name = f"{prefix} schema"
            prop_defs = {
                name: (
                    _infer_value_type(description, uri, name),
                    "external",
                    f"{name} property",
                )
                for name in sorted(props)
            }

        li = etree.SubElement(bag, f"{{{ns_rdf}}}li")
        li.set(f"{{{ns_rdf}}}parseType", "Resource")

        schema_elem = etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}schema")
        schema_elem.text = schema_name

        ns_uri_elem = etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}namespaceURI")
        ns_uri_elem.text = uri

        prefix_elem = etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}prefix")
        prefix_elem.text = prefix

        if prop_defs:
            property_elem = etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}property")
            seq = etree.SubElement(property_elem, f"{{{ns_rdf}}}Seq")

            for prop_name, (value_type, category, desc) in sorted(prop_defs.items()):
                prop_li = etree.SubElement(seq, f"{{{ns_rdf}}}li")
                prop_li.set(f"{{{ns_rdf}}}parseType", "Resource")

                name_elem = etree.SubElement(prop_li, f"{{{_NS_PDFA_PROPERTY}}}name")
                name_elem.text = prop_name

                vt_elem = etree.SubElement(prop_li, f"{{{_NS_PDFA_PROPERTY}}}valueType")
                vt_elem.text = value_type

                cat_elem = etree.SubElement(prop_li, f"{{{_NS_PDFA_PROPERTY}}}category")
                cat_elem.text = category

                tag = f"{{{_NS_PDFA_PROPERTY}}}description"
                desc_elem = etree.SubElement(prop_li, tag)
                desc_elem.text = desc

    return schemas_elem


def _extract_existing_xmp(pdf: pikepdf.Pdf) -> etree._Element | None:
    """Read and parse existing XMP metadata from the PDF.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        Parsed XMP XML tree or None if not present or unparseable.
    """
    try:
        metadata = pdf.Root.get("/Metadata")
        if metadata is None:
            return None

        try:
            metadata = metadata.get_object()
        except (AttributeError, ValueError, TypeError):
            pass

        xmp_bytes = bytes(metadata.read_bytes())
    except Exception as e:
        logger.warning("Error reading existing XMP metadata: %s", e)
        return None

    try:
        content = _strip_xpacket_wrapper(xmp_bytes)
        if not content:
            return None

        return etree.fromstring(content, _SECURE_XML_PARSER)
    except etree.XMLSyntaxError as e:
        logger.debug("Existing XMP XML parsing error: %s", e)
        return None
    except Exception as e:
        logger.debug("Error parsing existing XMP metadata: %s", e)
        return None


def _collect_preserved_elements(
    old_tree: etree._Element,
) -> tuple[list[etree._Element], dict[str, str], dict[str, str]]:
    """Collect non-managed elements and attributes from existing XMP.

    Walks all rdf:Description elements in the old XMP tree and collects
    child elements and attributes whose Clark-notation tags are NOT in
    _MANAGED_ELEMENTS/_MANAGED_ATTRS.

    Args:
        old_tree: Parsed XMP XML tree (x:xmpmeta root).

    Returns:
        Tuple of (preserved_elements, preserved_attrs, extra_namespaces)
        where preserved_elements is a list of deep-copied elements,
        preserved_attrs is a dict of {clark_attr: value},
        extra_namespaces is a dict of {prefix: uri} for serialization.
    """
    ns_rdf = NAMESPACES["rdf"]
    preserved_elements: list[etree._Element] = []
    preserved_attrs: dict[str, str] = {}
    extra_namespaces: dict[str, str] = {}

    # Invert NAMESPACES for URI->prefix lookup
    uri_to_prefix = {uri: prefix for prefix, uri in NAMESPACES.items()}

    # Only preserve properties from top-level rdf:Description nodes directly
    # under rdf:RDF. Nested rdf:Description nodes are property value structs.
    for rdf_root in old_tree.iter(f"{{{ns_rdf}}}RDF"):
        for desc in rdf_root.findall(f"{{{ns_rdf}}}Description"):
            # Collect non-managed child elements
            for child in desc:
                tag = child.tag
                if tag in _MANAGED_ELEMENTS:
                    continue
                # Skip rdf: structural elements that aren't properties
                if tag.startswith(f"{{{ns_rdf}}}"):
                    continue
                # Skip existing extension schemas (we regenerate them fresh
                # to ensure correct structure and completeness)
                if tag == f"{{{_NS_PDFA_EXTENSION}}}schemas":
                    continue
                # Validate property structure/value against predefined schema
                if isinstance(tag, str) and tag.startswith("{"):
                    uri, local = tag[1:].split("}", 1)
                    if not _is_valid_preserved_property(child, uri, local):
                        logger.debug(
                            "Stripping non-conforming property: %s",
                            tag,
                        )
                        continue
                preserved_elements.append(copy.deepcopy(child))
                _normalize_structural_properties(preserved_elements[-1])

                # Track namespace for serialization
                _register_element_namespaces(child, uri_to_prefix, extra_namespaces)

            # Collect non-managed attributes
            for attr_name, attr_value in desc.attrib.items():
                if attr_name in _MANAGED_ATTRS:
                    continue
                # Skip rdf:about (structural, already set)
                if attr_name == f"{{{ns_rdf}}}about":
                    continue
                # Skip bare RDF structural attributes (about/ID/nodeID are
                # mutually exclusive per RDF spec; some source PDFs use the
                # non-namespaced form which would conflict with rdf:about)
                if attr_name in ("about", "ID", "nodeID"):
                    continue
                # Validate attribute value against known type constraints
                if attr_name.startswith("{"):
                    a_uri, a_local = attr_name[1:].split("}", 1)
                    text = attr_value.strip()
                    # Check predefined property types first
                    type_code = _PREDEFINED_PROPERTY_TYPES.get(
                        (a_uri, a_local),
                    )
                    if type_code is None:
                        # Check extension schema types
                        type_code = _get_extension_type_code(a_uri, a_local)
                    if type_code is not None:
                        if not _is_valid_simple_value(text, type_code):
                            logger.debug(
                                "Stripping non-conforming attribute: %s=%r",
                                attr_name,
                                attr_value,
                            )
                            continue
                preserved_attrs[attr_name] = attr_value

                # Track namespace
                if attr_name.startswith("{"):
                    uri = attr_name.split("}")[0][1:]
                    if (
                        uri not in uri_to_prefix
                        and uri not in extra_namespaces.values()
                    ):
                        # Unknown namespace — prefer original prefix from source
                        original = _prefix_from_nsmap(desc, uri)
                        if (
                            original
                            and original not in extra_namespaces
                            and original not in NAMESPACES
                        ):
                            prefix = original
                        else:
                            prefix = _generate_prefix(uri, extra_namespaces)
                        extra_namespaces[prefix] = uri

    return preserved_elements, preserved_attrs, extra_namespaces


_RESERVED_NS_URIS = {
    "http://www.w3.org/XML/1998/namespace",
    "http://www.w3.org/2000/xmlns/",
}


def _prefix_from_nsmap(elem: etree._Element, uri: str) -> str | None:
    """Find the original prefix for a URI from the element's namespace map."""
    for prefix, ns_uri in elem.nsmap.items():
        if ns_uri == uri and prefix is not None:
            return prefix
    return None


def _register_element_namespaces(
    elem: etree._Element,
    uri_to_prefix: dict[str, str],
    extra_namespaces: dict[str, str],
) -> None:
    """Register namespaces used by an element and its descendants."""
    for node in elem.iter():
        tag = node.tag
        if isinstance(tag, str) and tag.startswith("{"):
            uri = tag.split("}")[0][1:]
            if (
                uri not in uri_to_prefix
                and uri not in extra_namespaces.values()
                and uri not in _RESERVED_NS_URIS
            ):
                original = _prefix_from_nsmap(node, uri)
                if (
                    original
                    and original not in extra_namespaces
                    and original not in NAMESPACES
                ):
                    prefix = original
                else:
                    prefix = _generate_prefix(uri, extra_namespaces)
                extra_namespaces[prefix] = uri
        for attr_name in node.attrib:
            if attr_name.startswith("{"):
                uri = attr_name.split("}")[0][1:]
                if (
                    uri not in uri_to_prefix
                    and uri not in extra_namespaces.values()
                    and uri not in _RESERVED_NS_URIS
                ):
                    original = _prefix_from_nsmap(node, uri)
                    if (
                        original
                        and original not in extra_namespaces
                        and original not in NAMESPACES
                    ):
                        prefix = original
                    else:
                        prefix = _generate_prefix(uri, extra_namespaces)
                    extra_namespaces[prefix] = uri


def _generate_prefix(uri: str, extra_namespaces: dict[str, str]) -> str:
    """Generate a unique namespace prefix for an unknown URI."""
    # Try to derive from the URI
    parts = uri.rstrip("/").rsplit("/", 1)
    base = parts[-1] if len(parts) > 1 else "ns"
    base = re.sub(r"[^a-zA-Z0-9]", "", base).lower()
    if not base:
        base = "ns"
    prefix = base
    counter = 0
    while prefix in extra_namespaces or prefix in NAMESPACES:
        counter += 1
        prefix = f"{base}{counter}"
    return prefix


def _normalize_trapped(value: Any) -> str:
    """
    Normalize /Trapped value to PDF/A compliant value.

    PDF/A requires /Trapped to be True, False, or Unknown (as Name objects).
    This function normalizes any input to one of these values.

    Args:
        value: The value from the Info Dictionary.

    Returns:
        "True", "False", or "Unknown" as string.
    """
    if value is None:
        return "Unknown"

    # Convert to string and normalize
    str_value = str(value).strip().lower()

    # Handle pikepdf Name objects (come as "/True", "/False", "/Unknown")
    if str_value.startswith("/"):
        str_value = str_value[1:].lower()

    if str_value == "true":
        return "True"
    elif str_value == "false":
        return "False"
    else:
        return "Unknown"


def _parse_pdf_date(date_str: str) -> datetime | None:
    """
    Parse PDF date format to Python datetime.

    PDF dates have format: D:YYYYMMDDHHmmSS+HH'mm' or variations.

    Args:
        date_str: PDF date string.

    Returns:
        Parsed datetime or None if parsing fails.
    """
    if not date_str:
        return None

    # Remove 'D:' prefix if present
    if date_str.startswith("D:"):
        date_str = date_str[2:]

    # Pattern for PDF date: YYYYMMDDHHMMSS with optional timezone
    pattern = r"(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?([+-Z])?([\d']+)?"

    match = re.match(pattern, date_str)
    if not match:
        logger.debug("Could not parse PDF date: %s", date_str)
        return None

    groups = match.groups()
    year = int(groups[0])
    month = int(groups[1]) if groups[1] else 1
    day = int(groups[2]) if groups[2] else 1
    hour = int(groups[3]) if groups[3] else 0
    minute = int(groups[4]) if groups[4] else 0
    second = int(groups[5]) if groups[5] else 0

    # Parse timezone offset
    tz_sign = groups[6]
    tz_offset_str = groups[7]
    tz = UTC
    if tz_sign in ("+", "-"):
        offset_hours = 0
        offset_minutes = 0
        if tz_offset_str:
            # Format: HH'mm' — strip quotes and split
            parts = tz_offset_str.replace("'", " ").split()
            if len(parts) >= 1:
                offset_hours = int(parts[0])
            if len(parts) >= 2:
                offset_minutes = int(parts[1])
        delta = timedelta(hours=offset_hours, minutes=offset_minutes)
        if tz_sign == "-":
            delta = -delta
        tz = timezone(delta)

    try:
        dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
        return dt.astimezone(UTC)
    except ValueError as e:
        logger.debug("Invalid date: %s (%s)", date_str, e)
        return None


def _format_pdf_date(dt: datetime) -> str:
    """
    Format datetime to PDF date string.

    Inverse of _parse_pdf_date() — produces format D:YYYYMMDDHHmmSS+00'00'.

    Args:
        dt: Datetime object (must be timezone-aware).

    Returns:
        PDF date string in UTC.
    """
    utc_dt = dt.astimezone(UTC)
    return utc_dt.strftime("D:%Y%m%d%H%M%S+00'00'")


def _format_iso_date(dt: datetime | None) -> str:
    """
    Format datetime to ISO 8601 for XMP.

    Args:
        dt: Datetime object or None.

    Returns:
        ISO 8601 formatted string or current time if dt is None.
    """
    if dt is None:
        dt = datetime.now(UTC)

    # Format: YYYY-MM-DDTHH:MM:SS+00:00
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def extract_pdf_info(pdf: pikepdf.Pdf) -> dict[str, Any]:
    """
    Extract metadata from PDF Info dictionary.

    Args:
        pdf: pikepdf Pdf object.

    Returns:
        Dictionary with normalized metadata keys:
        title, author, subject, creator, producer, creation_date, modification_date
    """
    info: dict[str, Any] = {
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None,
        "trapped": None,
    }

    try:
        pdf_info = pdf.docinfo
    except Exception:
        logger.debug("No Info dictionary present")
        return info

    if pdf_info is None:
        return info

    # Mapping from PDF keys to our normalized keys
    key_mapping = {
        "/Title": "title",
        "/Author": "author",
        "/Subject": "subject",
        "/Keywords": "keywords",
        "/Creator": "creator",
        "/Producer": "producer",
        "/CreationDate": "creation_date",
        "/ModDate": "modification_date",
        "/Trapped": "trapped",
    }

    for pdf_key, info_key in key_mapping.items():
        try:
            value = pdf_info.get(pdf_key)
            if value is not None:
                # Convert pikepdf string to Python string
                str_value = str(value)

                # Parse date fields
                if info_key in ("creation_date", "modification_date"):
                    info[info_key] = _parse_pdf_date(str_value)
                elif info_key == "trapped":
                    info[info_key] = _normalize_trapped(value)
                else:
                    info[info_key] = _sanitize_xml_text(str_value)
        except Exception as e:
            logger.debug("Error reading %s: %s", pdf_key, e)

    logger.debug("Extracted metadata: %s", info)
    return info


def create_xmp_metadata(
    info: dict[str, Any],
    pdfa_part: int,
    pdfa_conformance: str,
    now: datetime | None = None,
    existing_xmp_tree: etree._Element | None = None,
    non_catalog_extension_needs: dict[str, set[str]] | None = None,
) -> bytes:
    """
    Create XMP metadata XML for PDF/A.

    Non-managed properties from existing_xmp_tree are preserved
    (e.g. PDF/X, PDF/UA, PDF/E, PDF/VT identifications, custom
    namespaces). Managed properties are always written fresh.

    Args:
        info: Metadata dictionary from extract_pdf_info.
        pdfa_part: PDF/A part number (1, 2, or 3).
        pdfa_conformance: PDF/A conformance level ('A', 'B', or 'U').
        now: Current timestamp for modification/metadata dates.
             If None, datetime.now(timezone.utc) is used.
        existing_xmp_tree: Parsed XML tree of existing XMP metadata
             to preserve non-managed properties from, or None.
        non_catalog_extension_needs: Extra namespace_uri -> {prop_names}
             from non-catalog XMP streams that need extension schema
             declarations in the catalog XMP.

    Returns:
        UTF-8 encoded XMP metadata bytes with packet wrapper.
    """
    if now is None:
        now = datetime.now(UTC)
    # Create namespace-aware element makers
    ns_rdf = NAMESPACES["rdf"]
    ns_dc = NAMESPACES["dc"]
    ns_xmp = NAMESPACES["xmp"]
    ns_pdf = NAMESPACES["pdf"]
    ns_pdfaid = NAMESPACES["pdfaid"]

    # Build the RDF description content
    nsmap = {
        "rdf": ns_rdf,
        "dc": ns_dc,
        "xmp": ns_xmp,
        "pdf": ns_pdf,
        "pdfaid": ns_pdfaid,
        "pdfaExtension": _NS_PDFA_EXTENSION,
        "pdfaSchema": _NS_PDFA_SCHEMA,
        "pdfaProperty": _NS_PDFA_PROPERTY,
    }

    # Create namespace-aware element makers (only rdf and dc are used as
    # factory functions; other namespaces use etree.SubElement directly)
    rdf = ElementMaker(namespace=ns_rdf, nsmap=nsmap)
    dc = ElementMaker(namespace=ns_dc, nsmap=nsmap)

    # Get metadata values with defaults (strip XML-illegal control chars)
    title = _sanitize_xml_text(info.get("title") or "Untitled")
    author = _sanitize_xml_text(info.get("author") or "")
    subject = _sanitize_xml_text(info.get("subject") or "")
    creation_date = _format_iso_date(info.get("creation_date") or now)
    modification_date = _format_iso_date(now)

    # Build dc:title (rdf:Alt with language alternative)
    title_elem = dc("title")
    title_alt = etree.SubElement(title_elem, f"{{{ns_rdf}}}Alt")
    title_li = etree.SubElement(title_alt, f"{{{ns_rdf}}}li")
    title_li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    title_li.text = title

    # Build dc:creator (rdf:Seq)
    creator_elem = dc("creator")
    creator_seq = etree.SubElement(creator_elem, f"{{{ns_rdf}}}Seq")
    creator_li = etree.SubElement(creator_seq, f"{{{ns_rdf}}}li")
    creator_li.text = author if author else "Unknown"

    # Build dc:description (rdf:Alt)
    desc_elem = dc("description")
    desc_alt = etree.SubElement(desc_elem, f"{{{ns_rdf}}}Alt")
    desc_li = etree.SubElement(desc_alt, f"{{{ns_rdf}}}li")
    desc_li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    desc_li.text = subject if subject else ""

    # Build the RDF Description
    description = rdf(
        "Description",
        {f"{{{ns_rdf}}}about": ""},
    )

    # Add PDF/A identification
    part_elem = etree.SubElement(description, f"{{{ns_pdfaid}}}part")
    part_elem.text = str(pdfa_part)
    conformance_elem = etree.SubElement(description, f"{{{ns_pdfaid}}}conformance")
    conformance_elem.text = pdfa_conformance.upper()

    # Add Dublin Core elements
    format_elem = dc("format")
    format_elem.text = "application/pdf"
    description.append(format_elem)
    description.append(title_elem)
    description.append(creator_elem)
    description.append(desc_elem)

    # Add XMP elements
    create_date_elem = etree.SubElement(description, f"{{{ns_xmp}}}CreateDate")
    create_date_elem.text = creation_date
    modify_date_elem = etree.SubElement(description, f"{{{ns_xmp}}}ModifyDate")
    modify_date_elem.text = modification_date
    metadata_date_elem = etree.SubElement(description, f"{{{ns_xmp}}}MetadataDate")
    metadata_date_elem.text = _format_iso_date(now)

    # Add pdf:Producer (synchronized with DocInfo /Producer)
    producer = info.get("producer") or "pdftopdfa"
    producer_elem = etree.SubElement(description, f"{{{ns_pdf}}}Producer")
    producer_elem.text = producer

    # Add xmp:CreatorTool (synchronized with DocInfo /Creator)
    creator_tool = info.get("creator") or ""
    if creator_tool:
        creator_tool_elem = etree.SubElement(description, f"{{{ns_xmp}}}CreatorTool")
        creator_tool_elem.text = creator_tool

    # Add pdf:Keywords (synchronized with DocInfo /Keywords)
    keywords = info.get("keywords") or ""
    if keywords:
        keywords_elem = etree.SubElement(description, f"{{{ns_pdf}}}Keywords")
        keywords_elem.text = keywords

    # Add pdf:Trapped (synchronized with DocInfo /Trapped)
    trapped = info.get("trapped")
    if trapped:
        trapped_elem = etree.SubElement(description, f"{{{ns_pdf}}}Trapped")
        trapped_elem.text = trapped

    # Merge preserved elements from existing XMP
    if existing_xmp_tree is not None:
        try:
            preserved_elems, preserved_attrs, extra_ns = _collect_preserved_elements(
                existing_xmp_tree
            )
        except Exception as e:
            logger.warning("Failed to collect preserved XMP properties: %s", e)
            preserved_elems, preserved_attrs, extra_ns = [], {}, {}

        # Register extra namespaces for serialization
        for prefix, uri in extra_ns.items():
            if uri not in _RESERVED_NS_URIS:
                etree.register_namespace(prefix, uri)
                nsmap[prefix] = uri

        # Re-register canonical extension schema prefixes to prevent
        # pollution from preserved element namespace maps
        etree.register_namespace("pdfaExtension", _NS_PDFA_EXTENSION)
        etree.register_namespace("pdfaSchema", _NS_PDFA_SCHEMA)
        etree.register_namespace("pdfaProperty", _NS_PDFA_PROPERTY)

        # Append preserved child elements
        for elem in preserved_elems:
            try:
                description.append(elem)
            except Exception as e:
                logger.warning("Failed to preserve XMP element %s: %s", elem.tag, e)

        # Set preserved attributes
        for attr_name, attr_value in preserved_attrs.items():
            try:
                description.set(attr_name, attr_value)
            except Exception as e:
                logger.warning("Failed to preserve XMP attribute %s: %s", attr_name, e)

    # Extract original extension schema blocks from the source XMP so we
    # can reuse them for non-catalog properties (preserves custom
    # valueTypes that we cannot infer).
    original_blocks: dict[str, etree._Element] | None = None
    if existing_xmp_tree is not None and non_catalog_extension_needs:
        original_blocks = _extract_extension_schema_blocks(existing_xmp_tree)

    # Build extension schemas for non-predefined properties
    # (includes properties from non-catalog XMP that lack their own
    # extension schema declarations — veraPDF rule 6.6.2.3.1)
    extension_elem = _build_extension_schemas(
        description,
        nsmap=nsmap,
        extra_properties=non_catalog_extension_needs,
        original_schema_blocks=original_blocks,
    )
    if extension_elem is not None:
        for prefix, uri in [
            ("pdfaExtension", _NS_PDFA_EXTENSION),
            ("pdfaSchema", _NS_PDFA_SCHEMA),
            ("pdfaProperty", _NS_PDFA_PROPERTY),
        ]:
            nsmap[prefix] = uri

        description.append(extension_elem)

    # Build RDF root
    rdf_root = rdf("RDF")
    rdf_root.append(description)

    # Build xmpmeta wrapper
    xmpmeta = etree.Element(
        f"{{{NAMESPACES['x']}}}xmpmeta",
        nsmap={"x": NAMESPACES["x"]},
    )
    xmpmeta.append(rdf_root)

    # Serialize to bytes
    xml_bytes = etree.tostring(
        xmpmeta,
        encoding="utf-8",
        xml_declaration=False,
        pretty_print=True,
    )

    # Add XMP padding before trailer (standard practice for in-place editing)
    xmp_padding_size = 2048
    _padding_line = b" " * 100 + b"\n"
    _num_lines = xmp_padding_size // len(_padding_line)
    _remainder = xmp_padding_size % len(_padding_line)
    padding_block = _padding_line * _num_lines + b" " * _remainder

    # Wrap with XMP packet markers
    result = XMP_HEADER + xml_bytes + b"\n" + padding_block + XMP_TRAILER

    logger.debug("XMP metadata created: %d bytes", len(result))
    return result


def embed_xmp_metadata(pdf: pikepdf.Pdf, xmp: bytes) -> None:
    """
    Embed XMP metadata into PDF document.

    Args:
        pdf: pikepdf Pdf object to modify.
        xmp: XMP metadata bytes.

    Raises:
        ConversionError: If embedding fails.
    """
    try:
        # Create metadata stream
        metadata_stream = pikepdf.Stream(pdf, xmp)
        metadata_stream.Type = pikepdf.Name.Metadata
        metadata_stream.Subtype = pikepdf.Name.XML
        # PDF/A requires XMP metadata stream to be uncompressed
        if pikepdf.Name.Filter in metadata_stream:
            del metadata_stream[pikepdf.Name.Filter]

        # Assign to document catalog
        pdf.Root.Metadata = pdf.make_indirect(metadata_stream)

        logger.debug("XMP metadata embedded in PDF")
    except Exception as e:
        raise ConversionError(f"Error embedding XMP metadata: {e}") from e


def _parse_xmp_bytes(data: bytes) -> etree._Element | None:
    """Try to parse raw bytes as XMP, stripping packet wrappers.

    Returns the parsed XML root element or None if the data is not
    well-formed XMP.
    """
    try:
        content = _strip_xpacket_wrapper(data)
        if not content:
            return None
        return etree.fromstring(content, _SECURE_XML_PARSER)
    except (etree.XMLSyntaxError, ValueError):
        return None


def _reserialize_xmp(tree: etree._Element) -> bytes:
    """Re-serialize a parsed XMP tree to bytes with packet wrapper."""
    xml_bytes = etree.tostring(
        tree,
        encoding="utf-8",
        xml_declaration=False,
        pretty_print=True,
    )
    return XMP_HEADER + xml_bytes + XMP_TRAILER


def _collect_non_catalog_extension_needs(
    pdf: pikepdf.Pdf,
) -> dict[str, set[str]]:
    """Scan non-catalog XMP metadata streams for non-predefined properties.

    PDF/A validators (veraPDF) check rule 6.6.2.3.1 across ALL XMP packets.
    Properties in non-catalog XMP that lack extension schema declarations
    in the main catalog cause validation failures.  Even non-catalog XMP
    with its own extension schemas may depend on custom valueType
    definitions from the catalog (veraPDF rule 6.6.2.3.3).

    Returns a dict of namespace_uri -> {property_local_names}.
    """
    result: dict[str, set[str]] = {}
    root_objgen = pdf.Root.objgen
    ns_rdf = NAMESPACES["rdf"]

    for obj in pdf.objects:
        if not isinstance(obj, pikepdf.Dictionary):
            continue
        try:
            if "/Metadata" not in obj or obj.objgen == root_objgen:
                continue

            meta_ref = obj["/Metadata"]
            try:
                meta_stream = meta_ref.get_object()
            except (AttributeError, ValueError, TypeError):
                meta_stream = meta_ref

            try:
                raw = bytes(meta_stream.read_bytes())
            except Exception:
                continue

            tree = _parse_xmp_bytes(raw)
            if tree is None:
                continue

            # Collect non-predefined properties from all rdf:Description
            # elements in this packet.  Even if the packet has its own
            # extension schemas, those schemas may reference custom
            # valueTypes that are defined in the catalog's extension
            # schemas (veraPDF rule 6.6.2.3.3).
            for rdf_root in tree.iter(f"{{{ns_rdf}}}RDF"):
                for desc in rdf_root.findall(f"{{{ns_rdf}}}Description"):
                    needed = _collect_non_predefined_properties(desc)
                    for uri, props in needed.items():
                        result.setdefault(uri, set()).update(props)

        except Exception as e:
            logger.debug(
                "Error scanning non-catalog /Metadata for extensions: %s",
                e,
            )

    return result


def _sanitize_non_catalog_metadata(pdf: pikepdf.Pdf) -> tuple[int, int]:
    """Sanitize /Metadata entries outside the document catalog.

    PDF/A validators check all XMP metadata streams in a document.  Rather
    than removing every non-catalog /Metadata reference (which destroys EXIF
    and other object-level metadata), this function:

    1. Tries to parse each non-catalog /Metadata stream as XMP.
    2. If the XMP is well-formed, re-serializes it (fixing encoding) and
       ensures the stream is uncompressed (PDF/A requirement).
    3. If the XMP is malformed or unreadable, removes the reference.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        Tuple of (sanitized_count, removed_count).
    """
    sanitized = 0
    removed = 0
    root_objgen = pdf.Root.objgen

    for obj in pdf.objects:
        try:
            obj = obj.get_object()
        except Exception:
            pass
        if not isinstance(obj, pikepdf.Dictionary):
            continue
        try:
            if "/Metadata" not in obj or obj.objgen == root_objgen:
                continue

            meta_ref = obj["/Metadata"]
            try:
                meta_stream = meta_ref.get_object()
            except (AttributeError, ValueError, TypeError):
                meta_stream = meta_ref

            # Read the raw stream bytes
            try:
                raw = bytes(meta_stream.read_bytes())
            except Exception:
                del obj["/Metadata"]
                removed += 1
                continue

            # Try to parse as XMP
            tree = _parse_xmp_bytes(raw)
            if tree is None:
                del obj["/Metadata"]
                removed += 1
                continue

            # Valid XMP — re-serialize cleanly and ensure uncompressed
            clean_bytes = _reserialize_xmp(tree)
            meta_stream.write(clean_bytes)
            if pikepdf.Name.Filter in meta_stream:
                del meta_stream[pikepdf.Name.Filter]
            if pikepdf.Name.DecodeParms in meta_stream:
                del meta_stream[pikepdf.Name.DecodeParms]
            sanitized += 1

        except Exception as e:
            logger.debug(
                "Error processing non-catalog /Metadata reference: %s",
                e,
            )

    if sanitized > 0:
        logger.info("Re-serialized %d non-catalog /Metadata stream(s)", sanitized)
    if removed > 0:
        logger.info("Removed %d malformed non-catalog /Metadata stream(s)", removed)
    return sanitized, removed


def sync_metadata(pdf: pikepdf.Pdf, pdfa_level: str) -> None:
    """
    Synchronize PDF metadata and embed XMP for PDF/A compliance.

    Extracts existing metadata from PDF Info dictionary,
    creates XMP metadata, and embeds it in the document.

    Args:
        pdf: pikepdf Pdf object to modify.
        pdfa_level: PDF/A level string (e.g., '2b', '3b').

    Raises:
        ConversionError: If level is invalid or metadata sync fails.
    """
    level_lower = validate_pdfa_level(pdfa_level)

    # Extract part and conformance from level
    pdfa_part = int(level_lower[0])
    pdfa_conformance = level_lower[1].upper()

    logger.info("Synchronizing metadata for PDF/A-%d%s", pdfa_part, pdfa_conformance)

    # Capture current time once for consistency between XMP and DocInfo
    now = datetime.now(UTC)

    # Extract existing metadata
    info = extract_pdf_info(pdf)

    # Extract existing XMP tree for preservation of non-managed properties
    existing_xmp_tree = _extract_existing_xmp(pdf)

    # Scan non-catalog XMP streams for properties that need extension
    # schema declarations in the catalog XMP (veraPDF rule 6.6.2.3.1).
    non_catalog_needs = _collect_non_catalog_extension_needs(pdf)

    # Create XMP metadata (preserving non-managed properties from existing XMP)
    xmp = create_xmp_metadata(
        info,
        pdfa_part,
        pdfa_conformance,
        now=now,
        existing_xmp_tree=existing_xmp_tree,
        non_catalog_extension_needs=non_catalog_needs or None,
    )

    # Embed in PDF
    embed_xmp_metadata(pdf, xmp)
    _sanitize_non_catalog_metadata(pdf)

    # Synchronize DocInfo with XMP values (PDF/A requires consistency)
    try:
        docinfo = pdf.docinfo
    except Exception as e:
        logger.warning("Could not access docinfo: %s", e)
        docinfo = None

    if docinfo is not None:
        # Remove non-standard keys from DocInfo (PDF/A only allows standard keys)
        try:
            allowed_keys = {
                "/Title",
                "/Author",
                "/Subject",
                "/Keywords",
                "/Creator",
                "/Producer",
                "/CreationDate",
                "/ModDate",
                "/Trapped",
            }
            for key in list(docinfo.keys()):
                if key not in allowed_keys:
                    del docinfo[key]
                    logger.debug("Removed non-standard key %s from DocInfo", key)
        except Exception as e:
            logger.warning("Error removing non-standard DocInfo keys: %s", e)

        # Keep Producer and Creator in DocInfo (synchronized with XMP)
        # Only log if present for debugging
        if "/Producer" in docinfo:
            logger.debug("Keeping DocInfo Producer: %s", docinfo.get("/Producer"))
        if "/Creator" in docinfo:
            logger.debug("Keeping DocInfo Creator: %s", docinfo.get("/Creator"))

        # Synchronize /Trapped with XMP pdf:Trapped
        try:
            if "/Trapped" in docinfo:
                trapped_value = _normalize_trapped(docinfo["/Trapped"])
                docinfo["/Trapped"] = pikepdf.Name(f"/{trapped_value}")
                logger.debug("Normalized /Trapped in DocInfo to /%s", trapped_value)
        except Exception as e:
            logger.warning("Error normalizing /Trapped in DocInfo: %s", e)

        # Synchronize /Author with XMP dc:creator fallback
        try:
            if "/Author" not in docinfo or not str(docinfo["/Author"]).strip():
                docinfo["/Author"] = "Unknown"
        except Exception as e:
            logger.warning("Error synchronizing /Author in DocInfo: %s", e)

        # Synchronize date fields between DocInfo and XMP
        try:
            docinfo["/CreationDate"] = _format_pdf_date(
                info.get("creation_date") or now
            )
            docinfo["/ModDate"] = _format_pdf_date(now)
        except Exception as e:
            logger.warning("Error synchronizing dates in DocInfo: %s", e)

    logger.info("XMP metadata successfully embedded")
