# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""XMP metadata handling for PDF/A conversion."""

import copy
import logging
import re
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone
from itertools import chain
from typing import Any
from uuid import uuid4

import pikepdf
from lxml import etree
from lxml.builder import ElementMaker

from .exceptions import ConversionError
from .utils import log_suppressed_error, resolve_indirect, validate_pdfa_level

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


_EMPTY_METADATA_PLACEHOLDERS = frozenset(
    {
        "none",
        "null",
        "nil",
        "set()",
        "[]",
        "{}",
        "()",
    }
)


def _clean_metadata_text(value: Any) -> str | None:
    """Return a clean metadata string, or None for empty placeholder values."""
    if value is None:
        return None

    text = _sanitize_xml_text(str(value)).strip()
    if not text:
        return None
    if text.lower() in _EMPTY_METADATA_PLACEHOLDERS:
        return None
    return text


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
    "fx": "urn:factur-x:pdfa:CrossIndustryDocument:invoice:1p0#",
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
_NS_PDFA_TYPE = "http://www.aiim.org/pdfa/ns/type#"
_NS_PDFA_FIELD = "http://www.aiim.org/pdfa/ns/field#"

# Register all namespaces globally so lxml serializes them with canonical prefixes
for _prefix, _uri in NAMESPACES.items():
    etree.register_namespace(_prefix, _uri)
etree.register_namespace("pdfaExtension", _NS_PDFA_EXTENSION)
etree.register_namespace("pdfaSchema", _NS_PDFA_SCHEMA)
etree.register_namespace("pdfaProperty", _NS_PDFA_PROPERTY)
etree.register_namespace("pdfaType", _NS_PDFA_TYPE)
etree.register_namespace("pdfaField", _NS_PDFA_FIELD)

# XMP packet header and trailer
XMP_HEADER = b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
XMP_TRAILER = b'\n<?xpacket end="w"?>'

_XRECHNUNG_FILENAME = "xrechnung.xml"
_FACTUR_X_FILENAME = "factur-x.xml"
_CII_NAMESPACE = "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100"
_RAM_NAMESPACE = (
    "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100"
)
_XRECHNUNG_30_GUIDELINE_ID = (
    "urn:cen.eu:en16931:2017#compliant#urn:xeinkauf.de:kosit:xrechnung_3.0"
)
_XRECHNUNG_30_FACTUR_X_PROPERTIES = {
    "DocumentType": "INVOICE",
    "DocumentFileName": _XRECHNUNG_FILENAME,
    "Version": "3.0",
    "ConformanceLevel": "XRECHNUNG",
}
_FACTUR_X_PROPERTY_NAMES = tuple(_XRECHNUNG_30_FACTUR_X_PROPERTIES)

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
    f"{{{NAMESPACES['xmpMM']}}}DocumentID",
    f"{{{NAMESPACES['xmpMM']}}}InstanceID",
}

# Attribute-form equivalents of managed elements (Clark notation)
_MANAGED_ATTRS = _MANAGED_ELEMENTS

# These xmpMM properties consistently fail veraPDF PDF/A-2/3 validation
# (clause 6.6.2.3.1) even when copied verbatim from source files. Preserve
# stable identifiers such as DocumentID/InstanceID, but strip the known
# problematic fields from catalog and non-catalog XMP packets.
_PDF_A_UNSAFE_PRESERVED_PROPERTIES = frozenset(
    {
        (NAMESPACES["xmpMM"], "OriginalDocumentID"),
        (NAMESPACES["xmpMM"], "DerivedFrom"),
        (NAMESPACES["xmpMM"], "History"),
        (NAMESPACES["exif"], "NativeDigest"),
        (NAMESPACES["tiff"], "NativeDigest"),
    }
)

# OCR/image pipelines frequently attach object-level XMP packets to image or
# form XObject streams. veraPDF reports these particular properties as
# non-compliant in non-catalog metadata packets under PDF/A-2/3 clause 6.6.2.3.1,
# so drop them during non-catalog sanitization instead of re-emitting invalid XMP.
_PDF_A_UNSAFE_NON_CATALOG_PROPERTIES = frozenset(
    {
        (NAMESPACES["photoshop"], "ColorMode"),
        (NAMESPACES["photoshop"], "ICCProfile"),
        (NAMESPACES["photoshop"], "DocumentAncestors"),
        (NAMESPACES["photoshop"], "LegacyIPTCDigest"),
        (NAMESPACES["exif"], "NativeDigest"),
        (NAMESPACES["tiff"], "NativeDigest"),
    }
)


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

_PREDEFINED_STRUCTURED_CONTAINERS = frozenset(
    {
        (_XMP, "Thumbnails"),
        (_XMPMM, "Ingredients"),
        (_XMPMM, "Pantry"),
        (_XMPTPG, "Fonts"),
        (_XMPTPG, "Colorants"),
    }
)

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


def _clone_with_registered_namespaces(elem: etree._Element) -> etree._Element:
    """Clone an XML subtree so lxml re-serializes it with registered prefixes."""
    clone = etree.Element(elem.tag)

    for attr_name, attr_value in elem.attrib.items():
        clone.set(attr_name, attr_value)

    clone.text = elem.text
    clone.tail = elem.tail

    for child in elem:
        clone.append(_clone_with_registered_namespaces(child))

    return clone


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
    "Real": "f",
    "URI": "s",
    "URL": "s",
    "AgentName": "s",
    "GUID": "s",
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
        "pdfe",
        {
            "ISO_PDFEVersion": ("Text", "internal", "PDF/E version identifier"),
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
    NAMESPACES["fx"]: (
        "Factur-X PDFA Extension Schema",
        "fx",
        {
            "DocumentFileName": (
                "Text",
                "external",
                "Name of the embedded XML invoice file",
            ),
            "DocumentType": ("Text", "external", "INVOICE"),
            "Version": (
                "Text",
                "external",
                "The actual version of the Factur-X XML schema",
            ),
            "ConformanceLevel": (
                "Text",
                "external",
                "The conformance level of the Factur-X data",
            ),
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


def _sanitize_extension_schema_blocks(
    blocks: dict[str, etree._Element],
) -> dict[str, etree._Element]:
    """Validate and repair extension schema rdf:li blocks.

    Drops entire blocks missing required schema-level fields, removes individual
    property entries that are missing required property-level fields, and drops
    blocks whose property Seq becomes empty after sanitization.

    Required schema-level fields: pdfaSchema:schema, pdfaSchema:namespaceURI,
    pdfaSchema:prefix, pdfaSchema:property (with rdf:Seq child).

    Required property-level fields: pdfaProperty:name (non-empty),
    pdfaProperty:valueType (non-empty), pdfaProperty:category (internal/external),
    pdfaProperty:description (non-empty).

    Optional value-type declarations (pdfaSchema:valueType) are also sanitized:
    they must be an rdf:Seq of ValueType entries, each requiring non-empty
    pdfaType:type, pdfaType:namespaceURI, pdfaType:prefix, and
    pdfaType:description fields. Optional pdfaType:field, when present, must
    itself be an rdf:Seq and each field entry must have non-empty
    pdfaField:name, pdfaField:valueType, and pdfaField:description; field
    value types must resolve to known XMP types or declared custom types in
    the same schema block.
    """
    ns_rdf = NAMESPACES["rdf"]
    schema_tag = f"{{{_NS_PDFA_SCHEMA}}}schema"
    ns_uri_tag = f"{{{_NS_PDFA_SCHEMA}}}namespaceURI"
    prefix_tag = f"{{{_NS_PDFA_SCHEMA}}}prefix"
    property_tag = f"{{{_NS_PDFA_SCHEMA}}}property"
    schema_value_type_tag = f"{{{_NS_PDFA_SCHEMA}}}valueType"
    seq_tag = f"{{{ns_rdf}}}Seq"
    li_tag = f"{{{ns_rdf}}}li"
    name_tag = f"{{{_NS_PDFA_PROPERTY}}}name"
    value_type_tag = f"{{{_NS_PDFA_PROPERTY}}}valueType"
    category_tag = f"{{{_NS_PDFA_PROPERTY}}}category"
    description_tag = f"{{{_NS_PDFA_PROPERTY}}}description"
    type_name_tag = f"{{{_NS_PDFA_TYPE}}}type"
    type_ns_uri_tag = f"{{{_NS_PDFA_TYPE}}}namespaceURI"
    type_prefix_tag = f"{{{_NS_PDFA_TYPE}}}prefix"
    type_description_tag = f"{{{_NS_PDFA_TYPE}}}description"
    type_field_tag = f"{{{_NS_PDFA_TYPE}}}field"
    field_name_tag = f"{{{_NS_PDFA_FIELD}}}name"
    field_value_type_tag = f"{{{_NS_PDFA_FIELD}}}valueType"
    field_description_tag = f"{{{_NS_PDFA_FIELD}}}description"

    known_field_base_value_types = frozenset(
        {
            "Text",
            "Date",
            "URI",
            "Integer",
            "Real",
            "Boolean",
            "AgentName",
            "GUID",
            "URL",
            # Keep Rational accepted for compatibility with existing extension data.
            "Rational",
        }
    )
    container_prefixes = frozenset({"Bag", "Seq", "Alt"})

    def _strip_unexpected_attrs(
        elem: etree._Element,
        allowed_attrs: set[str],
    ) -> None:
        for attr_name in list(elem.attrib):
            if attr_name not in allowed_attrs:
                del elem.attrib[attr_name]

    def _strip_unexpected_children(
        elem: etree._Element,
        allowed_tags: set[str],
    ) -> None:
        for child in list(elem):
            if child.tag not in allowed_tags:
                elem.remove(child)

    def _remove_duplicate_children(
        elem: etree._Element,
        tag: str,
    ) -> None:
        seen_first = False
        for child in list(elem):
            if child.tag != tag:
                continue
            if not seen_first:
                seen_first = True
                continue
            elem.remove(child)

    def _is_recognized_field_value_type(
        raw_value_type: str,
        declared_value_types: set[str],
    ) -> bool:
        """Check whether a pdfaField:valueType references a recognized type."""
        normalized = " ".join(raw_value_type.split())
        if not normalized:
            return False

        if (
            normalized in known_field_base_value_types
            or normalized in declared_value_types
        ):
            return True

        parts = normalized.split(" ")
        if len(parts) != 2 or parts[0] not in container_prefixes:
            return False

        subtype = parts[1]
        return (
            subtype in known_field_base_value_types or subtype in declared_value_types
        )

    result: dict[str, etree._Element] = {}

    for uri, li_elem in blocks.items():
        _strip_unexpected_attrs(li_elem, {f"{{{ns_rdf}}}parseType"})
        _strip_unexpected_children(
            li_elem,
            {
                schema_tag,
                ns_uri_tag,
                prefix_tag,
                property_tag,
                schema_value_type_tag,
            },
        )
        _remove_duplicate_children(li_elem, schema_tag)
        _remove_duplicate_children(li_elem, ns_uri_tag)
        _remove_duplicate_children(li_elem, prefix_tag)
        _remove_duplicate_children(li_elem, property_tag)
        _remove_duplicate_children(li_elem, schema_value_type_tag)

        # Schema-level checks
        schema_elem = li_elem.find(schema_tag)
        if schema_elem is None or not (schema_elem.text or "").strip():
            logger.warning(
                "Extension schema block for %s dropped: missing pdfaSchema:schema", uri
            )
            continue

        ns_uri_elem = li_elem.find(ns_uri_tag)
        if ns_uri_elem is None or not (ns_uri_elem.text or "").strip():
            logger.warning(
                "Extension schema block for %s dropped:"
                " missing pdfaSchema:namespaceURI",
                uri,
            )
            continue

        prefix_elem = li_elem.find(prefix_tag)
        if prefix_elem is None or not (prefix_elem.text or "").strip():
            logger.warning(
                "Extension schema block for %s dropped: missing pdfaSchema:prefix", uri
            )
            continue

        property_elem = li_elem.find(property_tag)
        if property_elem is None:
            logger.warning(
                "Extension schema block for %s dropped: missing pdfaSchema:property",
                uri,
            )
            continue
        _strip_unexpected_attrs(property_elem, set())
        _strip_unexpected_children(property_elem, {seq_tag})
        _remove_duplicate_children(property_elem, seq_tag)

        seq = property_elem.find(seq_tag)
        if seq is None:
            logger.warning(
                "Extension schema block for %s dropped:"
                " pdfaSchema:property has no rdf:Seq",
                uri,
            )
            continue

        # Property-level checks
        to_remove = []
        for prop_li in seq.findall(li_tag):
            _strip_unexpected_attrs(prop_li, {f"{{{ns_rdf}}}parseType"})
            _strip_unexpected_children(
                prop_li,
                {
                    name_tag,
                    value_type_tag,
                    category_tag,
                    description_tag,
                },
            )
            _remove_duplicate_children(prop_li, name_tag)
            _remove_duplicate_children(prop_li, value_type_tag)
            _remove_duplicate_children(prop_li, category_tag)
            _remove_duplicate_children(prop_li, description_tag)

            name_elem = prop_li.find(name_tag)
            prop_name = (name_elem.text or "").strip() if name_elem is not None else ""

            vt_elem = prop_li.find(value_type_tag)
            cat_elem = prop_li.find(category_tag)
            desc_elem = prop_li.find(description_tag)

            if name_elem is None or not prop_name:
                logger.warning(
                    "Extension schema %s: removing property entry with missing name",
                    uri,
                )
                to_remove.append(prop_li)
                continue

            if vt_elem is None or not (vt_elem.text or "").strip():
                logger.warning(
                    "Extension schema %s: removing property '%s'"
                    " — missing pdfaProperty:valueType",
                    uri,
                    prop_name,
                )
                to_remove.append(prop_li)
                continue

            if cat_elem is None or (cat_elem.text or "").strip() not in {
                "internal",
                "external",
            }:
                logger.warning(
                    "Extension schema %s: removing property '%s'"
                    " — invalid pdfaProperty:category",
                    uri,
                    prop_name,
                )
                to_remove.append(prop_li)
                continue

            if desc_elem is None or not (desc_elem.text or "").strip():
                logger.warning(
                    "Extension schema %s: removing property '%s'"
                    " — missing pdfaProperty:description",
                    uri,
                    prop_name,
                )
                to_remove.append(prop_li)
                continue

        for prop_li in to_remove:
            seq.remove(prop_li)

        # Post-property check: drop block if Seq is now empty
        if not seq.findall(li_tag):
            logger.warning(
                "Extension schema block for %s dropped: no valid properties remain", uri
            )
            continue

        # Optional pdfaSchema:valueType checks
        schema_value_type_elem = li_elem.find(schema_value_type_tag)
        if schema_value_type_elem is not None:
            _strip_unexpected_attrs(schema_value_type_elem, set())
            _strip_unexpected_children(schema_value_type_elem, {seq_tag})
            _remove_duplicate_children(schema_value_type_elem, seq_tag)
            value_type_seq = schema_value_type_elem.find(seq_tag)
            if value_type_seq is None:
                logger.warning(
                    "Extension schema %s: removing pdfaSchema:valueType"
                    " — no rdf:Seq child",
                    uri,
                )
                li_elem.remove(schema_value_type_elem)
            else:
                value_types_to_remove = []
                valid_value_type_entries: list[tuple[etree._Element, str]] = []

                for value_type_li in value_type_seq.findall(li_tag):
                    _strip_unexpected_attrs(value_type_li, {f"{{{ns_rdf}}}parseType"})
                    _strip_unexpected_children(
                        value_type_li,
                        {
                            type_name_tag,
                            type_ns_uri_tag,
                            type_prefix_tag,
                            type_description_tag,
                            type_field_tag,
                        },
                    )
                    _remove_duplicate_children(value_type_li, type_name_tag)
                    _remove_duplicate_children(value_type_li, type_ns_uri_tag)
                    _remove_duplicate_children(value_type_li, type_prefix_tag)
                    _remove_duplicate_children(value_type_li, type_description_tag)
                    _remove_duplicate_children(value_type_li, type_field_tag)

                    type_name_elem = value_type_li.find(type_name_tag)
                    type_name = (
                        (type_name_elem.text or "").strip()
                        if type_name_elem is not None
                        else ""
                    )
                    type_ns_uri_elem = value_type_li.find(type_ns_uri_tag)
                    type_prefix_elem = value_type_li.find(type_prefix_tag)
                    type_description_elem = value_type_li.find(type_description_tag)

                    if type_name_elem is None or not type_name:
                        logger.warning(
                            "Extension schema %s: removing ValueType entry"
                            " — missing pdfaType:type",
                            uri,
                        )
                        value_types_to_remove.append(value_type_li)
                        continue

                    if (
                        type_ns_uri_elem is None
                        or not (type_ns_uri_elem.text or "").strip()
                    ):
                        logger.warning(
                            "Extension schema %s: removing ValueType '%s'"
                            " — missing pdfaType:namespaceURI",
                            uri,
                            type_name,
                        )
                        value_types_to_remove.append(value_type_li)
                        continue

                    if (
                        type_prefix_elem is None
                        or not (type_prefix_elem.text or "").strip()
                    ):
                        logger.warning(
                            "Extension schema %s: removing ValueType '%s'"
                            " — missing pdfaType:prefix",
                            uri,
                            type_name,
                        )
                        value_types_to_remove.append(value_type_li)
                        continue

                    if (
                        type_description_elem is None
                        or not (type_description_elem.text or "").strip()
                    ):
                        logger.warning(
                            "Extension schema %s: removing ValueType '%s'"
                            " — missing pdfaType:description",
                            uri,
                            type_name,
                        )
                        value_types_to_remove.append(value_type_li)
                        continue

                    valid_value_type_entries.append((value_type_li, type_name))

                declared_type_names = {
                    type_name for _, type_name in valid_value_type_entries
                }

                for value_type_li, type_name in valid_value_type_entries:
                    type_field_elem = value_type_li.find(type_field_tag)
                    if type_field_elem is None:
                        continue
                    _strip_unexpected_attrs(type_field_elem, set())
                    _strip_unexpected_children(type_field_elem, {seq_tag})
                    _remove_duplicate_children(type_field_elem, seq_tag)

                    field_seq = type_field_elem.find(seq_tag)
                    if field_seq is None:
                        logger.warning(
                            "Extension schema %s: removing pdfaType:field from"
                            " ValueType '%s' — expected rdf:Seq",
                            uri,
                            type_name,
                        )
                        value_type_li.remove(type_field_elem)
                        continue

                    field_entries_to_remove = []
                    for field_li in field_seq.findall(li_tag):
                        _strip_unexpected_attrs(field_li, {f"{{{ns_rdf}}}parseType"})
                        _strip_unexpected_children(
                            field_li,
                            {
                                field_name_tag,
                                field_value_type_tag,
                                field_description_tag,
                            },
                        )
                        _remove_duplicate_children(field_li, field_name_tag)
                        _remove_duplicate_children(field_li, field_value_type_tag)
                        _remove_duplicate_children(field_li, field_description_tag)

                        field_name_elem = field_li.find(field_name_tag)
                        field_name = (
                            (field_name_elem.text or "").strip()
                            if field_name_elem is not None
                            else ""
                        )
                        field_value_type_elem = field_li.find(field_value_type_tag)
                        field_value_type = (
                            (field_value_type_elem.text or "").strip()
                            if field_value_type_elem is not None
                            else ""
                        )
                        field_description_elem = field_li.find(field_description_tag)

                        if field_name_elem is None or not field_name:
                            logger.warning(
                                "Extension schema %s: removing field entry from"
                                " ValueType '%s' — missing pdfaField:name",
                                uri,
                                type_name,
                            )
                            field_entries_to_remove.append(field_li)
                            continue

                        if field_value_type_elem is None or not field_value_type:
                            logger.warning(
                                "Extension schema %s: removing field '%s' from"
                                " ValueType '%s' — missing pdfaField:valueType",
                                uri,
                                field_name,
                                type_name,
                            )
                            field_entries_to_remove.append(field_li)
                            continue

                        if not _is_recognized_field_value_type(
                            field_value_type,
                            declared_type_names,
                        ):
                            logger.warning(
                                "Extension schema %s: removing field '%s' from"
                                " ValueType '%s' — invalid pdfaField:valueType '%s'",
                                uri,
                                field_name,
                                type_name,
                                field_value_type,
                            )
                            field_entries_to_remove.append(field_li)
                            continue

                        if (
                            field_description_elem is None
                            or not (field_description_elem.text or "").strip()
                        ):
                            logger.warning(
                                "Extension schema %s: removing field '%s' from"
                                " ValueType '%s' — missing pdfaField:description",
                                uri,
                                field_name,
                                type_name,
                            )
                            field_entries_to_remove.append(field_li)
                            continue

                    for field_li in field_entries_to_remove:
                        field_seq.remove(field_li)

                    if not field_seq.findall(li_tag):
                        logger.warning(
                            "Extension schema %s: removing pdfaType:field from"
                            " ValueType '%s' — no valid field entries remain",
                            uri,
                            type_name,
                        )
                        value_type_li.remove(type_field_elem)

                for value_type_li in value_types_to_remove:
                    value_type_seq.remove(value_type_li)

                if not value_type_seq.findall(li_tag):
                    logger.warning(
                        "Extension schema %s: removing pdfaSchema:valueType"
                        " — no valid ValueType entries remain",
                        uri,
                    )
                    li_elem.remove(schema_value_type_elem)

                invalid_property_value_types: list[etree._Element] = []
                for prop_li in seq.findall(li_tag):
                    name_elem = prop_li.find(name_tag)
                    prop_name = (
                        (name_elem.text or "").strip() if name_elem is not None else ""
                    )
                    vt_elem = prop_li.find(value_type_tag)
                    prop_value_type = (
                        (vt_elem.text or "").strip() if vt_elem is not None else ""
                    )
                    if _is_recognized_field_value_type(
                        prop_value_type,
                        declared_type_names,
                    ):
                        continue
                    logger.warning(
                        "Extension schema %s: removing property '%s'"
                        " — invalid pdfaProperty:valueType '%s'",
                        uri,
                        prop_name,
                        prop_value_type,
                    )
                    invalid_property_value_types.append(prop_li)

                for prop_li in invalid_property_value_types:
                    seq.remove(prop_li)

                if not seq.findall(li_tag):
                    logger.warning(
                        "Extension schema block for %s dropped:"
                        " no valid properties remain after valueType checks",
                        uri,
                    )
                    continue

        result[uri] = _clone_with_registered_namespaces(li_elem)

    return result


def _collect_declared_extension_schema_details(
    blocks: dict[str, etree._Element],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    """Collect declared property and custom type definitions from schema blocks."""
    ns_rdf = NAMESPACES["rdf"]
    property_tag = f"{{{_NS_PDFA_SCHEMA}}}property"
    schema_value_type_tag = f"{{{_NS_PDFA_SCHEMA}}}valueType"
    seq_tag = f"{{{ns_rdf}}}Seq"
    li_tag = f"{{{ns_rdf}}}li"
    name_tag = f"{{{_NS_PDFA_PROPERTY}}}name"
    value_type_tag = f"{{{_NS_PDFA_PROPERTY}}}valueType"
    type_name_tag = f"{{{_NS_PDFA_TYPE}}}type"
    type_field_tag = f"{{{_NS_PDFA_TYPE}}}field"
    field_name_tag = f"{{{_NS_PDFA_FIELD}}}name"
    field_value_type_tag = f"{{{_NS_PDFA_FIELD}}}valueType"

    property_types: dict[str, dict[str, str]] = {}
    custom_types: dict[str, dict[str, dict[str, str]]] = {}

    for uri, block in blocks.items():
        prop_elem = block.find(property_tag)
        if prop_elem is not None:
            seq = prop_elem.find(seq_tag)
            if seq is not None:
                for prop_li in seq.findall(li_tag):
                    name_elem = prop_li.find(name_tag)
                    value_type_elem = prop_li.find(value_type_tag)
                    prop_name = (
                        (name_elem.text or "").strip() if name_elem is not None else ""
                    )
                    prop_value_type = (
                        (value_type_elem.text or "").strip()
                        if value_type_elem is not None
                        else ""
                    )
                    if prop_name and prop_value_type:
                        property_types.setdefault(uri, {})[prop_name] = prop_value_type

        schema_value_type_elem = block.find(schema_value_type_tag)
        if schema_value_type_elem is None:
            continue
        value_type_seq = schema_value_type_elem.find(seq_tag)
        if value_type_seq is None:
            continue
        for value_type_li in value_type_seq.findall(li_tag):
            type_name_elem = value_type_li.find(type_name_tag)
            type_name = (
                (type_name_elem.text or "").strip()
                if type_name_elem is not None
                else ""
            )
            if not type_name:
                continue

            fields: dict[str, str] = {}
            type_field_elem = value_type_li.find(type_field_tag)
            if type_field_elem is not None:
                field_seq = type_field_elem.find(seq_tag)
                if field_seq is not None:
                    for field_li in field_seq.findall(li_tag):
                        field_name_elem = field_li.find(field_name_tag)
                        field_value_type_elem = field_li.find(field_value_type_tag)
                        field_name = (
                            (field_name_elem.text or "").strip()
                            if field_name_elem is not None
                            else ""
                        )
                        field_value_type = (
                            (field_value_type_elem.text or "").strip()
                            if field_value_type_elem is not None
                            else ""
                        )
                        if field_name and field_value_type:
                            fields[field_name] = field_value_type

            custom_types.setdefault(uri, {})[type_name] = fields

    return property_types, custom_types


def _build_extension_schemas_from_blocks(
    blocks: dict[str, etree._Element],
) -> etree._Element | None:
    """Build a pdfaExtension:schemas element from sanitized schema blocks."""
    if not blocks:
        return None

    ns_rdf = NAMESPACES["rdf"]
    schemas_elem = etree.Element(f"{{{_NS_PDFA_EXTENSION}}}schemas")
    bag = etree.SubElement(schemas_elem, f"{{{ns_rdf}}}Bag")

    for uri in sorted(blocks):
        bag.append(copy.deepcopy(blocks[uri]))

    return schemas_elem


def _augment_extension_schema_block(
    block: etree._Element,
    *,
    uri: str,
    props: set[str],
    description: etree._Element,
    known_prop_defs: dict[str, tuple[str, str, str]] | None = None,
) -> etree._Element:
    """Clone a schema block and add declarations for any missing properties."""
    ns_rdf = NAMESPACES["rdf"]
    property_tag = f"{{{_NS_PDFA_SCHEMA}}}property"
    seq_tag = f"{{{ns_rdf}}}Seq"
    li_tag = f"{{{ns_rdf}}}li"
    name_tag = f"{{{_NS_PDFA_PROPERTY}}}name"
    value_type_tag = f"{{{_NS_PDFA_PROPERTY}}}valueType"
    category_tag = f"{{{_NS_PDFA_PROPERTY}}}category"
    description_tag = f"{{{_NS_PDFA_PROPERTY}}}description"

    block_copy = copy.deepcopy(block)
    property_elem = block_copy.find(property_tag)
    if property_elem is None:
        property_elem = etree.SubElement(block_copy, property_tag)

    seq = property_elem.find(seq_tag)
    if seq is None:
        seq = etree.SubElement(property_elem, seq_tag)

    declared_props: set[str] = set()
    for prop_li in seq.findall(li_tag):
        name_elem = prop_li.find(name_tag)
        prop_name = (name_elem.text or "").strip() if name_elem is not None else ""
        if prop_name:
            declared_props.add(prop_name)

    missing_props = sorted(props - declared_props)
    for prop_name in missing_props:
        if known_prop_defs is not None and prop_name in known_prop_defs:
            value_type, category, prop_description = known_prop_defs[prop_name]
        else:
            value_type = _infer_value_type(description, uri, prop_name)
            category = "external"
            prop_description = f"{prop_name} property"

        prop_li = etree.SubElement(seq, li_tag)
        prop_li.set(f"{{{ns_rdf}}}parseType", "Resource")
        etree.SubElement(prop_li, name_tag).text = prop_name
        etree.SubElement(prop_li, value_type_tag).text = value_type
        etree.SubElement(prop_li, category_tag).text = category
        etree.SubElement(prop_li, description_tag).text = prop_description

    return block_copy


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
    if type_code == "f":
        try:
            float(text)
        except ValueError:
            return False
        return True
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
            valid_items = 0
            for li in child:
                if li.tag != f"{{{ns_rdf}}}li":
                    continue
                text = _clean_metadata_text(li.text)
                if text is None:
                    return False
                if not _is_valid_simple_value(text, item_type):
                    return False
                valid_items += 1
            return valid_items > 0
    return False  # No Seq found


def _validate_alt_lang(elem: etree._Element, ns_rdf: str) -> bool:
    """Check that Lang Alt items have xml:lang attributes."""
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
    for child in elem:
        if child.tag == f"{{{ns_rdf}}}Alt":
            valid_items = 0
            for li in child:
                if li.tag != f"{{{ns_rdf}}}li":
                    continue
                if li.get(xml_lang) is None:
                    return False
                if _clean_metadata_text(li.text) is None:
                    return False
                valid_items += 1
            return valid_items > 0
    return False  # No Alt found


def _validate_text_container(elem: etree._Element, ns_rdf: str, container: str) -> bool:
    """Check that a text container has at least one useful list item."""
    container_tag = f"{{{ns_rdf}}}{container}"
    li_tag = f"{{{ns_rdf}}}li"

    for child in elem:
        if child.tag != container_tag:
            continue
        valid_items = 0
        for li in child:
            if li.tag != li_tag:
                continue
            if any(isinstance(grand.tag, str) for grand in li):
                return False
            if _clean_metadata_text(li.text) is None:
                return False
            valid_items += 1
        return valid_items > 0

    return False


def _validate_structured_container(
    elem: etree._Element,
    ns_rdf: str,
    container: str,
) -> bool:
    """Check that a structured container has at least one non-empty list item."""
    container_tag = f"{{{ns_rdf}}}{container}"
    li_tag = f"{{{ns_rdf}}}li"
    parse_type = f"{{{ns_rdf}}}parseType"

    for child in elem:
        if child.tag != container_tag:
            continue
        valid_items = 0
        for li in child:
            if li.tag != li_tag:
                continue
            has_child_content = any(isinstance(grand.tag, str) for grand in li)
            has_value_attr = any(attr_name != parse_type for attr_name in li.attrib)
            if (
                not has_child_content
                and not has_value_attr
                and _clean_metadata_text(li.text) is None
            ):
                return False
            valid_items += 1
        return valid_items > 0

    return False


def _normalize_extension_value_type(value_type: str) -> str:
    """Normalize XMP extension valueType whitespace."""
    return " ".join(value_type.split())


def _split_extension_container_type(value_type: str) -> tuple[str, str] | None:
    """Split a container valueType into (container, subtype)."""
    normalized = _normalize_extension_value_type(value_type)
    parts = normalized.split(" ", 1)
    if len(parts) != 2 or parts[0] not in {"Bag", "Seq", "Alt"}:
        return None
    return parts[0], parts[1]


def _validate_extension_attribute_value(
    value: str,
    value_type: str,
) -> bool:
    """Validate an attribute-form extension property against its valueType."""
    normalized = _normalize_extension_value_type(value_type)
    type_code = _EXTENSION_VALUE_TYPE_MAP.get(normalized)
    if type_code is None:
        return False
    text = _clean_metadata_text(value)
    return text is not None and _is_valid_simple_value(text, type_code)


def _validate_extension_struct_fields(
    elem: etree._Element,
    field_types: dict[str, str],
    custom_types: dict[str, dict[str, str]],
) -> bool:
    """Validate a structured extension value against its declared fields."""
    ns_rdf = NAMESPACES["rdf"]
    rdf_desc_tag = f"{{{ns_rdf}}}Description"
    source = elem
    rdf_descriptions = [child for child in elem if child.tag == rdf_desc_tag]
    if len(rdf_descriptions) > 1:
        return False
    if rdf_descriptions:
        if len(elem) != 1:
            return False
        source = rdf_descriptions[0]

    for attr_name, attr_value in source.attrib.items():
        if not attr_name.startswith("{"):
            return False
        attr_uri, attr_local = attr_name[1:].split("}", 1)
        if attr_uri == ns_rdf:
            continue
        declared_type = field_types.get(attr_local)
        if declared_type is None:
            return False
        if not _validate_extension_attribute_value(attr_value, declared_type):
            return False

    for child in source:
        if not isinstance(child.tag, str) or not child.tag.startswith("{"):
            return False
        child_uri, child_local = child.tag[1:].split("}", 1)
        if child_uri == ns_rdf:
            return False
        declared_type = field_types.get(child_local)
        if declared_type is None:
            return False
        if not _is_valid_extension_property_value(
            child,
            declared_type,
            custom_types,
        ):
            return False

    return True


def _validate_extension_container_items(
    elem: etree._Element,
    expected_container: str,
    item_value_type: str,
    custom_types: dict[str, dict[str, str]],
) -> bool:
    """Validate items inside a Bag/Seq/Alt extension property."""
    ns_rdf = NAMESPACES["rdf"]
    container_tag = f"{{{ns_rdf}}}{expected_container}"
    li_tag = f"{{{ns_rdf}}}li"
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"

    for child in elem:
        if child.tag != container_tag:
            continue
        valid_items = 0
        for li in child:
            if li.tag != li_tag:
                continue
            if expected_container == "Alt" and item_value_type == "Lang Alt":
                if li.get(xml_lang) is None:
                    return False
            type_code = _EXTENSION_VALUE_TYPE_MAP.get(item_value_type)
            if type_code is not None:
                if any(isinstance(grand.tag, str) for grand in li):
                    return False
                text = _clean_metadata_text(li.text)
                if text is None or not _is_valid_simple_value(text, type_code):
                    return False
                valid_items += 1
                continue
            custom_field_types = custom_types.get(item_value_type)
            if custom_field_types is None:
                return False
            if not _validate_extension_struct_fields(
                li, custom_field_types, custom_types
            ):
                return False
            valid_items += 1
        return valid_items > 0

    return False


def _is_valid_extension_property_value(
    elem: etree._Element,
    value_type: str,
    custom_types: dict[str, dict[str, str]],
) -> bool:
    """Validate an XMP property against its declared extension valueType."""
    normalized = _normalize_extension_value_type(value_type)
    ns_rdf = NAMESPACES["rdf"]
    actual = _detect_structure(elem, ns_rdf)

    if normalized == "Lang Alt":
        return actual == "a" and _validate_alt_lang(elem, ns_rdf)

    type_code = _EXTENSION_VALUE_TYPE_MAP.get(normalized)
    if type_code is not None:
        if actual != "s":
            return False
        text = _clean_metadata_text(elem.text)
        return text is not None and _is_valid_simple_value(text, type_code)

    container_info = _split_extension_container_type(normalized)
    if container_info is not None:
        container_name, item_value_type = container_info
        expected_structure = {"Bag": "b", "Seq": "q", "Alt": "a"}[container_name]
        if actual != expected_structure:
            return False
        return _validate_extension_container_items(
            elem,
            container_name,
            item_value_type,
            custom_types,
        )

    custom_field_types = custom_types.get(normalized)
    if custom_field_types is None:
        return False
    if actual != "x":
        return False
    return _validate_extension_struct_fields(elem, custom_field_types, custom_types)


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
    if (uri, local_name) in _PDF_A_UNSAFE_PRESERVED_PROPERTIES:
        return False

    expected = _PREDEFINED_PROPERTY_TYPES.get((uri, local_name))
    if expected is None:
        # Check extension schema types (e.g. pdfaid:rev -> Integer)
        ext_code = _get_extension_type_code(uri, local_name)
        if ext_code is not None:
            ns_rdf = NAMESPACES["rdf"]
            actual = _detect_structure(elem, ns_rdf)
            if actual != "s":
                return False
            text = _clean_metadata_text(elem.text)
            return text is not None and _is_valid_simple_value(text, ext_code)
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
        if actual != expected:
            return False
        container = {"b": "Bag", "q": "Seq", "a": "Alt"}[expected]
        if (uri, local_name) in _PREDEFINED_STRUCTURED_CONTAINERS:
            return _validate_structured_container(elem, ns_rdf, container)
        return _validate_text_container(elem, ns_rdf, container)
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
    text = _clean_metadata_text(elem.text)
    if text is None:
        return False
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
        known = _KNOWN_EXTENSION_SCHEMAS.get(uri)
        # If the original catalog XMP had an extension schema block for
        # this namespace, reuse it (preserves custom valueTypes, etc.).
        if original_schema_blocks and uri in original_schema_blocks:
            known_props = known[2] if known is not None else None
            bag.append(
                _augment_extension_schema_block(
                    original_schema_blocks[uri],
                    uri=uri,
                    props=props,
                    description=description,
                    known_prop_defs=known_props,
                )
            )
            continue

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
        log_suppressed_error(logger, e, "Error parsing existing XMP metadata: %s", e)
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
    preserved_element_tags: set[str] = set()
    extra_namespaces: dict[str, str] = {}

    # Invert NAMESPACES for URI->prefix lookup
    uri_to_prefix = {uri: prefix for prefix, uri in NAMESPACES.items()}
    schema_blocks = _sanitize_extension_schema_blocks(
        _extract_extension_schema_blocks(old_tree)
    )
    declared_property_types, declared_custom_types = (
        _collect_declared_extension_schema_details(schema_blocks)
    )

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
                    if (uri, local) in _PDF_A_UNSAFE_PRESERVED_PROPERTIES:
                        logger.debug(
                            "Stripping PDF/A-unsafe preserved property: %s",
                            tag,
                        )
                        continue
                    declared_value_type = declared_property_types.get(uri, {}).get(
                        local
                    )
                    if declared_value_type is not None:
                        if not _is_valid_extension_property_value(
                            child,
                            declared_value_type,
                            declared_custom_types.get(uri, {}),
                        ):
                            logger.debug(
                                "Stripping extension property with invalid"
                                " valueType: %s",
                                tag,
                            )
                            continue
                    elif not _is_valid_preserved_property(child, uri, local):
                        logger.debug(
                            "Stripping non-conforming property: %s",
                            tag,
                        )
                        continue
                if tag in preserved_element_tags:
                    logger.debug("Stripping duplicate preserved property: %s", tag)
                    continue
                preserved_attrs.pop(tag, None)
                preserved_elements.append(copy.deepcopy(child))
                preserved_element_tags.add(tag)
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
                if attr_name in preserved_element_tags:
                    logger.debug(
                        "Stripping attribute duplicate of preserved property: %s",
                        attr_name,
                    )
                    continue
                # Validate attribute value against known type constraints
                if attr_name.startswith("{"):
                    a_uri, a_local = attr_name[1:].split("}", 1)
                    if (
                        a_uri,
                        a_local,
                    ) in _PDF_A_UNSAFE_PRESERVED_PROPERTIES:
                        logger.debug(
                            "Stripping PDF/A-unsafe preserved attribute: %s=%r",
                            attr_name,
                            attr_value,
                        )
                        continue
                    text = _clean_metadata_text(attr_value)
                    if text is None:
                        logger.debug(
                            "Stripping empty preserved attribute: %s=%r",
                            attr_name,
                            attr_value,
                        )
                        continue
                    declared_value_type = declared_property_types.get(a_uri, {}).get(
                        a_local
                    )
                    if declared_value_type is not None:
                        if not _validate_extension_attribute_value(
                            text,
                            declared_value_type,
                        ):
                            logger.debug(
                                "Stripping extension attribute with invalid valueType:"
                                " %s=%r",
                                attr_name,
                                attr_value,
                            )
                            continue
                    else:
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
    date_str = date_str.removeprefix("D:")

    # Pattern for PDF date: YYYYMMDDHHMMSS with optional timezone
    pattern = r"(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?([-+Z])?([\d']+)?"

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
        dt: Datetime object or None. Naive datetimes are assumed to be UTC;
            timezone-aware datetimes are converted to UTC.

    Returns:
        ISO 8601 formatted string in UTC or current time if dt is None.
    """
    if dt is None:
        dt = datetime.now(UTC)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    # Format: YYYY-MM-DDTHH:MM:SS+00:00
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def extract_pdf_info(pdf: pikepdf.Pdf, *, log_result: bool = True) -> dict[str, Any]:
    """
    Extract metadata from PDF Info dictionary.

    Args:
        pdf: pikepdf Pdf object.
        log_result: Whether to emit a debug log with the extracted metadata.

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
                    info[info_key] = _clean_metadata_text(str_value)
        except Exception as e:
            log_suppressed_error(logger, e, "Error reading %s: %s", pdf_key, e)

    if log_result:
        logger.debug("Extracted metadata: %s", info)
    return info


def _new_xmp_identifier() -> str:
    """Create a GUID suitable for xmpMM identifiers."""
    return f"urn:uuid:{uuid4()}"


def _extract_simple_xmp_property(
    tree: etree._Element | None,
    namespace: str,
    local_name: str,
) -> str | None:
    """Extract a simple top-level XMP property from element or attribute form."""
    if tree is None:
        return None

    ns_rdf = NAMESPACES["rdf"]
    property_name = f"{{{namespace}}}{local_name}"

    for rdf_root in tree.iter(f"{{{ns_rdf}}}RDF"):
        for desc in rdf_root.findall(f"{{{ns_rdf}}}Description"):
            attr_value = desc.get(property_name)
            if attr_value is not None:
                cleaned = _clean_metadata_text(attr_value)
                if cleaned is not None:
                    return cleaned

            prop = desc.find(property_name)
            if prop is None:
                continue
            if len(prop):
                continue
            cleaned = _clean_metadata_text(prop.text)
            if cleaned is not None:
                return cleaned

    return None


def _extract_lang_alt_xmp_property(
    tree: etree._Element | None,
    namespace: str,
    local_name: str,
) -> str | None:
    """Extract a Lang-Alt XMP property, preferring the x-default entry."""
    if tree is None:
        return None

    ns_rdf = NAMESPACES["rdf"]
    property_name = f"{{{namespace}}}{local_name}"
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"

    for rdf_root in tree.iter(f"{{{ns_rdf}}}RDF"):
        for desc in rdf_root.findall(f"{{{ns_rdf}}}Description"):
            prop = desc.find(property_name)
            if prop is None:
                continue
            fallback = None
            for li in prop.findall(f"{{{ns_rdf}}}Alt/{{{ns_rdf}}}li"):
                cleaned = _clean_metadata_text(li.text)
                if cleaned is None:
                    continue
                if li.get(xml_lang, "x-default") == "x-default":
                    return cleaned
                if fallback is None:
                    fallback = cleaned
            if fallback is not None:
                return fallback

    # Some producers write Lang-Alt properties in simple text or
    # attribute form; accept those as a last resort.
    return _extract_simple_xmp_property(tree, namespace, local_name)


def _extract_array_xmp_property(
    tree: etree._Element | None,
    namespace: str,
    local_name: str,
) -> list[str]:
    """Extract item texts of an rdf:Seq/rdf:Bag/rdf:Alt XMP array property."""
    if tree is None:
        return []

    ns_rdf = NAMESPACES["rdf"]
    property_name = f"{{{namespace}}}{local_name}"

    for rdf_root in tree.iter(f"{{{ns_rdf}}}RDF"):
        for desc in rdf_root.findall(f"{{{ns_rdf}}}Description"):
            prop = desc.find(property_name)
            if prop is None:
                continue
            for container in ("Seq", "Bag", "Alt"):
                items = prop.findall(f"{{{ns_rdf}}}{container}/{{{ns_rdf}}}li")
                values = [
                    cleaned
                    for li in items
                    if (cleaned := _clean_metadata_text(li.text)) is not None
                ]
                if values:
                    return values

    simple = _extract_simple_xmp_property(tree, namespace, local_name)
    return [simple] if simple is not None else []


def create_xmp_metadata(
    info: dict[str, Any],
    pdfa_part: int,
    pdfa_conformance: str,
    now: datetime | None = None,
    existing_xmp_tree: etree._Element | None = None,
    non_catalog_extension_needs: dict[str, set[str]] | None = None,
    factur_x_properties: Mapping[str, str] | None = None,
    *,
    pdfua: bool = False,
    fallback_title: str | None = None,
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
        factur_x_properties: Canonical Factur-X properties inferred from an
             embedded XRechnung, or None.
        pdfua: If True, identify the document as PDF/UA-1 and include the
             required PDF/A extension schema declaration.
        fallback_title: Document title to use for PDF/UA when the source has
             no title metadata.

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
    ns_pdfuaid = NAMESPACES["pdfuaid"]
    ns_fx = NAMESPACES["fx"]
    ns_xmpmm = NAMESPACES["xmpMM"]

    # Build the RDF description content
    nsmap = {
        "rdf": ns_rdf,
        "dc": ns_dc,
        "xmp": ns_xmp,
        "pdf": ns_pdf,
        "pdfaid": ns_pdfaid,
        "xmpMM": ns_xmpmm,
        "pdfaExtension": _NS_PDFA_EXTENSION,
        "pdfaSchema": _NS_PDFA_SCHEMA,
        "pdfaProperty": _NS_PDFA_PROPERTY,
    }
    if factur_x_properties is not None:
        nsmap["fx"] = ns_fx
    if pdfua:
        nsmap["pdfuaid"] = ns_pdfuaid

    # Create namespace-aware element makers (only rdf and dc are used as
    # factory functions; other namespaces use etree.SubElement directly)
    rdf = ElementMaker(namespace=ns_rdf, nsmap=nsmap)
    dc = ElementMaker(namespace=ns_dc, nsmap=nsmap)

    # Get metadata values with defaults (strip XML-illegal control chars)
    title = _clean_metadata_text(info.get("title"))
    author = _clean_metadata_text(info.get("author"))
    subject = _clean_metadata_text(info.get("subject"))

    # dc:title/creator/description are managed properties, so they are not
    # copied over from the existing XMP. Modern producers often store these
    # values only in XMP (no DocInfo entries); fall back to the source XMP
    # so they are not lost.
    if title is None:
        title = _extract_lang_alt_xmp_property(existing_xmp_tree, ns_dc, "title")
    if title is None and pdfua:
        title = _clean_metadata_text(fallback_title) or "Untitled"
    if author is None:
        creators = _extract_array_xmp_property(existing_xmp_tree, ns_dc, "creator")
        if creators:
            author = ", ".join(creators)
    if subject is None:
        subject = _extract_lang_alt_xmp_property(
            existing_xmp_tree, ns_dc, "description"
        )
    creation_date = _format_iso_date(info.get("creation_date") or now)
    modification_date = _format_iso_date(now)
    document_id = _extract_simple_xmp_property(
        existing_xmp_tree,
        ns_xmpmm,
        "DocumentID",
    )
    existing_instance_id = _extract_simple_xmp_property(
        existing_xmp_tree,
        ns_xmpmm,
        "InstanceID",
    )

    def _build_lang_alt(tag_name: str, text: str) -> etree._Element:
        elem = dc(tag_name)
        alt = etree.SubElement(elem, f"{{{ns_rdf}}}Alt")
        li = etree.SubElement(alt, f"{{{ns_rdf}}}li")
        li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
        li.text = text
        return elem

    def _build_seq(tag_name: str, text: str) -> etree._Element:
        elem = dc(tag_name)
        seq = etree.SubElement(elem, f"{{{ns_rdf}}}Seq")
        li = etree.SubElement(seq, f"{{{ns_rdf}}}li")
        li.text = text
        return elem

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
    if pdfua:
        pdfua_part_elem = etree.SubElement(description, f"{{{ns_pdfuaid}}}part")
        pdfua_part_elem.text = "1"

    # Add Dublin Core elements
    format_elem = dc("format")
    format_elem.text = "application/pdf"
    description.append(format_elem)
    if title is not None:
        description.append(_build_lang_alt("title", title))
    if author is not None:
        description.append(_build_seq("creator", author))
    if subject is not None:
        description.append(_build_lang_alt("description", subject))

    # Add XMP elements
    create_date_elem = etree.SubElement(description, f"{{{ns_xmp}}}CreateDate")
    create_date_elem.text = creation_date
    modify_date_elem = etree.SubElement(description, f"{{{ns_xmp}}}ModifyDate")
    modify_date_elem.text = modification_date
    metadata_date_elem = etree.SubElement(description, f"{{{ns_xmp}}}MetadataDate")
    metadata_date_elem.text = _format_iso_date(now)

    if document_id is not None:
        document_id_elem = etree.SubElement(description, f"{{{ns_xmpmm}}}DocumentID")
        document_id_elem.text = document_id
    if document_id is not None or existing_instance_id is not None:
        instance_id_elem = etree.SubElement(description, f"{{{ns_xmpmm}}}InstanceID")
        instance_id_elem.text = _new_xmp_identifier()

    # Add pdf:Producer (synchronized with DocInfo /Producer)
    producer = _clean_metadata_text(info.get("producer")) or "pdftopdfa"
    producer_elem = etree.SubElement(description, f"{{{ns_pdf}}}Producer")
    producer_elem.text = producer

    # Add xmp:CreatorTool (synchronized with DocInfo /Creator)
    creator_tool = _clean_metadata_text(info.get("creator"))
    if creator_tool is not None:
        creator_tool_elem = etree.SubElement(description, f"{{{ns_xmp}}}CreatorTool")
        creator_tool_elem.text = creator_tool

    # Add pdf:Keywords (synchronized with DocInfo /Keywords)
    keywords = _clean_metadata_text(info.get("keywords"))
    if keywords is not None:
        keywords_elem = etree.SubElement(description, f"{{{ns_pdf}}}Keywords")
        keywords_elem.text = keywords

    if factur_x_properties is not None:
        for property_name in _FACTUR_X_PROPERTY_NAMES:
            if property_name not in factur_x_properties:
                continue
            elem = etree.SubElement(description, f"{{{ns_fx}}}{property_name}")
            elem.text = factur_x_properties[property_name]

    # Merge preserved elements from existing XMP
    if existing_xmp_tree is not None:
        try:
            preserved_elems, preserved_attrs, extra_ns = _collect_preserved_elements(
                existing_xmp_tree
            )
        except Exception as e:
            logger.warning("Failed to collect preserved XMP properties: %s", e)
            preserved_elems, preserved_attrs, extra_ns = [], {}, {}

        if factur_x_properties is not None:
            factur_x_tags = {
                f"{{{ns_fx}}}{property_name}"
                for property_name in _FACTUR_X_PROPERTY_NAMES
            }
            preserved_elems = [
                elem for elem in preserved_elems if elem.tag not in factur_x_tags
            ]
            preserved_attrs = {
                name: value
                for name, value in preserved_attrs.items()
                if name not in factur_x_tags
            }

        if pdfua:
            preserved_elems = [
                elem
                for elem in preserved_elems
                if etree.QName(elem).namespace != ns_pdfuaid
            ]
            preserved_attrs = {
                name: value
                for name, value in preserved_attrs.items()
                if etree.QName(name).namespace != ns_pdfuaid
            }

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
    if existing_xmp_tree is not None:
        original_blocks = _extract_extension_schema_blocks(existing_xmp_tree)
        if original_blocks:
            original_blocks = _sanitize_extension_schema_blocks(original_blocks)
            if factur_x_properties is not None:
                original_blocks.pop(ns_fx, None)

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
        nsmap.update(
            {
                "pdfaExtension": _NS_PDFA_EXTENSION,
                "pdfaSchema": _NS_PDFA_SCHEMA,
                "pdfaProperty": _NS_PDFA_PROPERTY,
            }
        )

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


def _has_unqualified_xml_names(tree: etree._Element) -> bool:
    """Return True if the XML tree contains unqualified element/attribute names.

    veraPDF rejects XMP packets that contain bare RDF property names without a
    namespace prefix/default namespace, even if the XML is otherwise well-formed.
    """
    for elem in tree.iter():
        if not isinstance(elem.tag, str):
            continue
        if not elem.tag.startswith("{"):
            return True
        for attr_name in elem.attrib:
            if not attr_name.startswith("{"):
                return True
    return False


def _reserialize_xmp(tree: etree._Element) -> bytes:
    """Re-serialize a parsed XMP tree to bytes with packet wrapper."""
    xml_bytes = etree.tostring(
        tree,
        encoding="utf-8",
        xml_declaration=False,
        pretty_print=True,
    )
    return XMP_HEADER + xml_bytes + XMP_TRAILER


def _remove_identification_properties(
    pdf: pikepdf.Pdf,
    *,
    namespace_key: str,
    standard: str,
) -> bool:
    """Remove one ISO standard's identification properties from catalog XMP."""
    metadata = pdf.Root.get("/Metadata")
    if metadata is None:
        return False

    try:
        data = bytes(metadata.read_bytes())
    except Exception:
        return False

    tree = _parse_xmp_bytes(data)
    if tree is None:
        return False

    namespace_prefix = f"{{{NAMESPACES[namespace_key]}}}"
    removed = False
    for elem in list(tree.iter()):
        tag = elem.tag
        if isinstance(tag, str) and tag.startswith(namespace_prefix):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
                removed = True
            continue
        for attr_name in list(elem.attrib):
            if attr_name.startswith(namespace_prefix):
                del elem.attrib[attr_name]
                removed = True

    if not removed:
        return False

    # Drop rdf:Description elements left empty by the removal
    ns_rdf = NAMESPACES["rdf"]
    about_attr = f"{{{ns_rdf}}}about"
    for rdf_root in tree.iter(f"{{{ns_rdf}}}RDF"):
        for desc in list(rdf_root.findall(f"{{{ns_rdf}}}Description")):
            if len(desc) == 0 and all(a == about_attr for a in desc.attrib):
                rdf_root.remove(desc)

    embed_xmp_metadata(pdf, _reserialize_xmp(tree))
    logger.info("Removed %s identification from XMP metadata", standard)
    return True


def remove_pdfx_identification(pdf: pikepdf.Pdf) -> bool:
    """Remove PDF/X identification after its required OutputIntent is dropped."""
    return _remove_identification_properties(
        pdf,
        namespace_key="pdfxid",
        standard="PDF/X",
    )


def remove_pdfua_identification(pdf: pikepdf.Pdf) -> bool:
    """Remove PDF/UA identification after the logical structure is rebuilt."""
    return _remove_identification_properties(
        pdf,
        namespace_key="pdfuaid",
        standard="PDF/UA",
    )


def remove_pdfvt_identification(pdf: pikepdf.Pdf) -> bool:
    """Remove PDF/VT identification after its required PDF/X basis is dropped."""
    return _remove_identification_properties(
        pdf,
        namespace_key="pdfvtid",
        standard="PDF/VT",
    )


def remove_pdfe_identification(pdf: pikepdf.Pdf) -> bool:
    """Remove PDF/E identification after its required DocInfo marker is dropped."""
    return _remove_identification_properties(
        pdf,
        namespace_key="pdfeid",
        standard="PDF/E",
    )


def _sanitize_non_catalog_xmp_tree(tree: etree._Element) -> None:
    """Strip undeclared custom properties from a non-catalog XMP packet."""
    ns_rdf = NAMESPACES["rdf"]
    ext_tag = f"{{{_NS_PDFA_EXTENSION}}}schemas"
    top_level_descriptions: list[etree._Element] = []

    for rdf_root in tree.iter(f"{{{ns_rdf}}}RDF"):
        top_level_descriptions.extend(rdf_root.findall(f"{{{ns_rdf}}}Description"))

    if not top_level_descriptions:
        return

    sanitized_blocks = _sanitize_extension_schema_blocks(
        _extract_extension_schema_blocks(tree)
    )
    declared_property_types, declared_custom_types = (
        _collect_declared_extension_schema_details(sanitized_blocks)
    )

    for desc in top_level_descriptions:
        for ext_elem in desc.findall(ext_tag):
            desc.remove(ext_elem)

    extension_elem = _build_extension_schemas_from_blocks(sanitized_blocks)
    if extension_elem is not None:
        top_level_descriptions[0].append(extension_elem)

    for desc in top_level_descriptions:
        for child in list(desc):
            tag = child.tag
            if not isinstance(tag, str):
                continue
            if tag == ext_tag:
                continue
            if tag.startswith(f"{{{ns_rdf}}}"):
                continue
            if not tag.startswith("{"):
                desc.remove(child)
                continue

            uri, local = tag[1:].split("}", 1)
            if (uri, local) in _PDF_A_UNSAFE_NON_CATALOG_PROPERTIES:
                desc.remove(child)
                continue
            predefined = _PREDEFINED_PROPERTIES.get(uri)
            declared_value_type = declared_property_types.get(uri, {}).get(local)

            if predefined is not None and local in predefined:
                if _is_valid_preserved_property(child, uri, local):
                    continue
                desc.remove(child)
                continue

            if declared_value_type is not None and _is_valid_extension_property_value(
                child,
                declared_value_type,
                declared_custom_types.get(uri, {}),
            ):
                continue

            desc.remove(child)

        for attr_name in list(desc.attrib):
            if attr_name == f"{{{ns_rdf}}}about":
                continue
            if attr_name in ("about", "ID", "nodeID"):
                del desc.attrib[attr_name]
                continue
            if not attr_name.startswith("{"):
                del desc.attrib[attr_name]
                continue

            uri, local = attr_name[1:].split("}", 1)
            if uri == ns_rdf:
                del desc.attrib[attr_name]
                continue
            if (uri, local) in _PDF_A_UNSAFE_NON_CATALOG_PROPERTIES:
                del desc.attrib[attr_name]
                continue
            if (uri, local) in _PDF_A_UNSAFE_PRESERVED_PROPERTIES:
                del desc.attrib[attr_name]
                continue

            predefined = _PREDEFINED_PROPERTIES.get(uri)
            declared_value_type = declared_property_types.get(uri, {}).get(local)
            if predefined is None and declared_value_type is None:
                del desc.attrib[attr_name]
                continue
            if (
                predefined is not None
                and local not in predefined
                and declared_value_type is None
            ):
                del desc.attrib[attr_name]
                continue

            text = (desc.attrib[attr_name] or "").strip()
            if declared_value_type is not None:
                if not _validate_extension_attribute_value(text, declared_value_type):
                    del desc.attrib[attr_name]
                continue

            type_code = _PREDEFINED_PROPERTY_TYPES.get((uri, local))
            if type_code is None:
                type_code = _get_extension_type_code(uri, local)
            if type_code is not None and not _is_valid_simple_value(text, type_code):
                del desc.attrib[attr_name]


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
    ns_rdf = NAMESPACES["rdf"]

    for obj in _iter_non_catalog_metadata_holders(pdf):
        try:
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
            if _has_unqualified_xml_names(tree):
                logger.debug(
                    "Ignoring non-catalog /Metadata with unqualified XML names: %s",
                    meta_stream.objgen,
                )
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
                        safe_props = {
                            prop
                            for prop in props
                            if (uri, prop) not in _PDF_A_UNSAFE_NON_CATALOG_PROPERTIES
                        }
                        if safe_props:
                            result.setdefault(uri, set()).update(safe_props)

        except Exception as e:
            log_suppressed_error(
                logger,
                e,
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

    for obj in _iter_non_catalog_metadata_holders(pdf):
        try:
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
            if _has_unqualified_xml_names(tree):
                del obj["/Metadata"]
                removed += 1
                continue

            _sanitize_non_catalog_xmp_tree(tree)

            # Valid XMP — re-serialize cleanly and ensure uncompressed
            clean_bytes = _reserialize_xmp(tree)
            meta_stream.write(clean_bytes)
            if pikepdf.Name.Filter in meta_stream:
                del meta_stream[pikepdf.Name.Filter]
            if pikepdf.Name.DecodeParms in meta_stream:
                del meta_stream[pikepdf.Name.DecodeParms]
            sanitized += 1

        except Exception as e:
            log_suppressed_error(
                logger,
                e,
                "Error processing non-catalog /Metadata reference: %s",
                e,
            )

    if sanitized > 0:
        logger.info("Re-serialized %d non-catalog /Metadata stream(s)", sanitized)
    if removed > 0:
        logger.info("Removed %d malformed non-catalog /Metadata stream(s)", removed)
    return sanitized, removed


def _iter_non_catalog_metadata_holders(
    pdf: pikepdf.Pdf,
) -> Iterator[pikepdf.Object]:
    """Yield non-catalog objects that carry a /Metadata reference.

    OCR pipelines may attach XMP to image/Form XObject streams directly, not
    only to dictionaries such as pages or auxiliary resource containers. Those
    stream objects must be sanitized as well because veraPDF validates every
    XMP packet in the document.
    """
    root_objgen = pdf.Root.objgen
    visited: set[tuple[int, int]] = set()

    for start in chain((pdf.Root,), pdf.objects):
        stack = [start]
        while stack:
            try:
                resolved = resolve_indirect(stack.pop())
            except Exception as exc:
                log_suppressed_error(
                    logger,
                    exc,
                    "Error resolving object while scanning /Metadata: %s",
                    exc,
                )
                continue

            if not isinstance(resolved, pikepdf.Object):
                continue

            objgen = resolved.objgen
            if objgen != (0, 0):
                if objgen in visited:
                    continue
                visited.add(objgen)

            if isinstance(resolved, (pikepdf.Dictionary, pikepdf.Stream)):
                if objgen != root_objgen and "/Metadata" in resolved:
                    yield resolved
                try:
                    stack.extend(resolved.values())
                except Exception as exc:
                    log_suppressed_error(
                        logger,
                        exc,
                        "Error traversing object while scanning /Metadata: %s",
                        exc,
                    )
            elif isinstance(resolved, pikepdf.Array):
                stack.extend(resolved)


def _iter_embedded_file_name_tree_pairs(
    node: object,
) -> Iterator[tuple[object, object]]:
    """Yield embedded-file Name Tree pairs without mutating the tree."""
    pending = [node]
    visited: set[tuple[int, int]] = set()
    while pending:
        resolved = resolve_indirect(pending.pop())
        objgen = resolved.objgen
        if objgen != (0, 0):
            if objgen in visited:
                raise ValueError("EmbeddedFiles Name Tree contains a cycle")
            visited.add(objgen)

        names = resolved.get("/Names")
        kids = resolved.get("/Kids")
        if names is not None and kids is not None:
            raise ValueError("EmbeddedFiles Name Tree node has both /Names and /Kids")

        if names is not None:
            if len(names) % 2:
                raise ValueError("EmbeddedFiles Name Tree has an odd /Names array")
            for index in range(0, len(names), 2):
                yield names[index], names[index + 1]

        if kids is not None:
            pending.extend(reversed(kids))


def _filespec_names(filespec: object) -> tuple[str | None, str | None]:
    """Return a FileSpec's /F and /UF names as strings."""
    resolved = resolve_indirect(filespec)
    file_name = resolved.get("/F")
    unicode_name = resolved.get("/UF")
    return (
        str(file_name) if file_name is not None else None,
        str(unicode_name) if unicode_name is not None else None,
    )


def _indirect_objgen(obj: object) -> tuple[int, int] | None:
    """Return a non-zero indirect object identifier."""
    try:
        objgen = resolve_indirect(obj).objgen
    except (AttributeError, TypeError, ValueError):
        return None
    return objgen if objgen != (0, 0) else None


def _extract_xrechnung_30_xml(pdf: pikepdf.Pdf) -> bytes | None:
    """Return an unambiguous embedded XRechnung 3.0 CII invoice."""
    try:
        names = pdf.Root.get("/Names")
        if names is None:
            return None
        names = resolve_indirect(names)
        embedded_files = names.get("/EmbeddedFiles")
        if embedded_files is None:
            return None
        embedded_files = resolve_indirect(embedded_files)
        pairs = list(_iter_embedded_file_name_tree_pairs(embedded_files))

        entries: list[tuple[str, str | None, str | None, object]] = []
        for tree_key, raw_filespec in pairs:
            tree_name = str(tree_key)
            file_name, unicode_name = _filespec_names(raw_filespec)
            names_for_entry = (tree_name, file_name, unicode_name)
            if _FACTUR_X_FILENAME in names_for_entry:
                return None
            if _XRECHNUNG_FILENAME in names_for_entry:
                entries.append((tree_name, file_name, unicode_name, raw_filespec))

        if len(entries) != 1:
            return None

        tree_name, file_name, unicode_name, raw_filespec = entries[0]
        if (tree_name, file_name, unicode_name) != (
            _XRECHNUNG_FILENAME,
            _XRECHNUNG_FILENAME,
            _XRECHNUNG_FILENAME,
        ):
            return None

        filespec = resolve_indirect(raw_filespec)
        if str(filespec.get("/Type")) != "/Filespec":
            return None
        if str(filespec.get("/AFRelationship")) != "/Alternative":
            return None

        filespec_objgen = _indirect_objgen(filespec)
        if filespec_objgen is None:
            return None

        root_af = pdf.Root.get("/AF")
        if root_af is None:
            return None
        af_entries = []
        for raw_af_filespec in root_af:
            af_file_name, af_unicode_name = _filespec_names(raw_af_filespec)
            if _FACTUR_X_FILENAME in (af_file_name, af_unicode_name):
                return None
            if _XRECHNUNG_FILENAME in (af_file_name, af_unicode_name):
                af_entries.append(raw_af_filespec)
        if len(af_entries) != 1:
            return None
        if _filespec_names(af_entries[0]) != (
            _XRECHNUNG_FILENAME,
            _XRECHNUNG_FILENAME,
        ):
            return None
        if _indirect_objgen(af_entries[0]) != filespec_objgen:
            return None

        embedded_streams = filespec.get("/EF")
        if embedded_streams is None:
            return None
        embedded_streams = resolve_indirect(embedded_streams)
        file_stream = resolve_indirect(embedded_streams.get("/F"))
        unicode_stream = resolve_indirect(embedded_streams.get("/UF"))
        if not isinstance(file_stream, pikepdf.Stream) or not isinstance(
            unicode_stream, pikepdf.Stream
        ):
            return None
        if str(file_stream.get("/Type")) != "/EmbeddedFile":
            return None
        if str(unicode_stream.get("/Type")) != "/EmbeddedFile":
            return None
        if str(file_stream.get("/Subtype")) != "/text/xml":
            return None
        if str(unicode_stream.get("/Subtype")) != "/text/xml":
            return None

        xml_bytes = bytes(file_stream.read_bytes())
        if not xml_bytes or xml_bytes != bytes(unicode_stream.read_bytes()):
            return None
    except Exception as e:
        log_suppressed_error(
            logger,
            e,
            "Could not inspect embedded XRechnung XML: %s",
            e,
        )
        return None

    try:
        root = etree.fromstring(xml_bytes, _SECURE_XML_PARSER)
    except (etree.XMLSyntaxError, ValueError):
        return None
    if root.getroottree().docinfo.doctype:
        return None
    if root.tag != f"{{{_CII_NAMESPACE}}}CrossIndustryInvoice":
        return None

    guideline_ids = root.findall(
        f"{{{_CII_NAMESPACE}}}ExchangedDocumentContext/"
        f"{{{_RAM_NAMESPACE}}}GuidelineSpecifiedDocumentContextParameter/"
        f"{{{_RAM_NAMESPACE}}}ID"
    )
    type_codes = root.findall(
        f"{{{_CII_NAMESPACE}}}ExchangedDocument/{{{_RAM_NAMESPACE}}}TypeCode"
    )
    if len(guideline_ids) != 1 or len(type_codes) != 1:
        return None
    if len(guideline_ids[0]) or guideline_ids[0].text != _XRECHNUNG_30_GUIDELINE_ID:
        return None
    if len(type_codes[0]) or type_codes[0].text != "380":
        return None

    return xml_bytes


def _iter_top_level_xmp_descriptions(
    tree: etree._Element | None,
) -> Iterator[etree._Element]:
    """Yield top-level rdf:Description elements from an XMP tree."""
    if tree is None:
        return
    ns_rdf = NAMESPACES["rdf"]
    for rdf_root in tree.iter(f"{{{ns_rdf}}}RDF"):
        yield from rdf_root.findall(f"{{{ns_rdf}}}Description")


def _collect_factur_x_property_occurrences(
    tree: etree._Element | None,
) -> dict[str, list[str | None]]:
    """Collect canonical Factur-X property occurrences, including invalid ones."""
    result: dict[str, list[str | None]] = {
        name: [] for name in _FACTUR_X_PROPERTY_NAMES
    }
    namespace = NAMESPACES["fx"]
    for description in _iter_top_level_xmp_descriptions(tree):
        for property_name in _FACTUR_X_PROPERTY_NAMES:
            tag = f"{{{namespace}}}{property_name}"
            if tag in description.attrib:
                result[property_name].append(description.attrib[tag])
            for elem in description.findall(tag):
                value = None if len(elem) else elem.text
                result[property_name].append(value)
    return result


def _has_noncanonical_hybrid_invoice_properties(
    tree: etree._Element | None,
) -> bool:
    """Return whether another hybrid-invoice XMP namespace is in use."""
    canonical_namespace = NAMESPACES["fx"]
    for description in _iter_top_level_xmp_descriptions(tree):
        names = list(description.attrib) + [
            child.tag for child in description if isinstance(child.tag, str)
        ]
        for name in names:
            if not name.startswith("{"):
                continue
            namespace, local_name = name[1:].split("}", 1)
            if (
                namespace != canonical_namespace
                and "CrossIndustryDocument" in namespace
                and local_name in _FACTUR_X_PROPERTY_NAMES
            ):
                return True
    return False


def _infer_factur_x_properties(
    pdf: pikepdf.Pdf,
    existing_xmp_tree: etree._Element | None,
) -> dict[str, str] | None:
    """Infer canonical Factur-X XMP only for an exact XRechnung 3.0 invoice."""
    if _extract_xrechnung_30_xml(pdf) is None:
        return None

    occurrences = _collect_factur_x_property_occurrences(existing_xmp_tree)
    conflicts = {
        name: values
        for name, values in occurrences.items()
        if values and values != [_XRECHNUNG_30_FACTUR_X_PROPERTIES[name]]
    }
    if conflicts:
        logger.warning(
            "Existing Factur-X XMP metadata conflicts with or duplicates the "
            "embedded XRechnung values; leaving it unchanged"
        )
        return None

    if _has_noncanonical_hybrid_invoice_properties(existing_xmp_tree):
        logger.warning(
            "Existing non-canonical hybrid-invoice XMP metadata conflicts with "
            "the embedded XRechnung; leaving it unchanged"
        )
        return None

    logger.info("Adding Factur-X XMP metadata for embedded XRechnung 3.0")
    return dict(_XRECHNUNG_30_FACTUR_X_PROPERTIES)


def sync_metadata(
    pdf: pikepdf.Pdf,
    pdfa_level: str,
    *,
    source_info: Mapping[str, Any] | None = None,
    source_xmp_tree: etree._Element | None = None,
    pdfua: bool = False,
    fallback_title: str | None = None,
) -> None:
    """
    Synchronize PDF metadata and embed XMP for PDF/A compliance.

    Extracts existing metadata from PDF Info dictionary,
    creates XMP metadata, and embeds it in the document.

    Args:
        pdf: pikepdf Pdf object to modify.
        pdfa_level: PDF/A level string (e.g., '2b', '3b').
        source_info: Optional metadata snapshot captured from the original
            input PDF before intermediate processing (for example OCR) may
            rewrite DocInfo values like /Creator or /Producer.
        source_xmp_tree: Optional parsed XMP tree captured from the original
            input PDF. When provided, preserved non-managed XMP properties are
            taken from this tree instead of the current in-memory PDF.
        pdfua: If True, add PDF/UA-1 identification metadata.
        fallback_title: Document title to use for PDF/UA when the source has
            no title metadata.

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

    # Extract existing metadata. Prefer a caller-provided snapshot from the
    # original input so OCR/temp saves do not overwrite user-authored values.
    info = extract_pdf_info(pdf, log_result=source_info is None)
    if source_info is not None:
        restored_fields: list[str] = []
        for key, value in source_info.items():
            if info.get(key) != value:
                restored_fields.append(key)
            info[key] = value
        if restored_fields:
            logger.debug(
                "Restoring original metadata snapshot for: %s",
                ", ".join(sorted(restored_fields)),
            )
        logger.debug("Effective metadata after applying source snapshot: %s", info)

    # Extract existing XMP tree for preservation of non-managed properties
    existing_xmp_tree = (
        copy.deepcopy(source_xmp_tree)
        if source_xmp_tree is not None
        else _extract_existing_xmp(pdf)
    )
    factur_x_properties = (
        _infer_factur_x_properties(pdf, existing_xmp_tree) if pdfa_part == 3 else None
    )

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
        factur_x_properties=factur_x_properties,
        pdfua=pdfua,
        fallback_title=fallback_title,
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

        # Synchronize standard DocInfo fields with the metadata we decided to keep.
        try:
            text_fields = {
                "/Title": _clean_metadata_text(info.get("title")),
                "/Author": _clean_metadata_text(info.get("author")),
                "/Subject": _clean_metadata_text(info.get("subject")),
                "/Keywords": _clean_metadata_text(info.get("keywords")),
                "/Creator": _clean_metadata_text(info.get("creator")),
                "/Producer": _clean_metadata_text(info.get("producer")) or "pdftopdfa",
            }
            for key, value in text_fields.items():
                if value is None:
                    if key in docinfo:
                        del docinfo[key]
                        logger.debug("Removed empty DocInfo %s", key)
                    continue
                docinfo[key] = value
                logger.debug("Synchronized DocInfo %s: %s", key, value)
        except Exception as e:
            logger.warning("Error synchronizing text fields in DocInfo: %s", e)

        # Synchronize /Trapped with XMP pdf:Trapped
        try:
            if "/Trapped" in docinfo:
                trapped_value = _normalize_trapped(docinfo["/Trapped"])
                docinfo["/Trapped"] = pikepdf.Name(f"/{trapped_value}")
                logger.debug("Normalized /Trapped in DocInfo to /%s", trapped_value)
        except Exception as e:
            logger.warning("Error normalizing /Trapped in DocInfo: %s", e)

        # Synchronize date fields between DocInfo and XMP
        try:
            docinfo["/CreationDate"] = _format_pdf_date(
                info.get("creation_date") or now
            )
            docinfo["/ModDate"] = _format_pdf_date(now)
        except Exception as e:
            logger.warning("Error synchronizing dates in DocInfo: %s", e)

    logger.info("XMP metadata successfully embedded")
