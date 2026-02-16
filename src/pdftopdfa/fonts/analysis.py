# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font analysis and validation for PDF/A compliance."""

import logging
from dataclasses import dataclass

import pikepdf

from ..exceptions import FontEmbeddingError
from ..utils import resolve_indirect as _resolve_indirect
from .constants import STANDARD_14_FONTS, UTF16_ENCODING_NAMES
from .tounicode import resolve_glyph_to_unicode
from .traversal import iter_all_page_fonts
from .utils import get_encoding_name as _get_encoding_name
from .utils import safe_str as _safe_str

logger = logging.getLogger(__name__)


@dataclass
class FontInfo:
    """Information about a font in a PDF.

    Attributes:
        name: Font name (e.g., "Arial", "Times-Roman").
        type: Font type (Type1, TrueType, CIDFont, Type3, MMType1).
        embedded: True if font data is embedded.
        subset: True if font is a subset (prefix like "ABCDEF+").
        has_tounicode: True if font has a ToUnicode CMap.
    """

    name: str
    type: str
    embedded: bool
    subset: bool
    has_tounicode: bool = False
    unicode_derivable: bool = False


def is_symbolic_font(font: pikepdf.Object) -> bool:
    """Checks if a font is symbolic via FontDescriptor Flags.

    The Symbolic flag is bit 3 (value 4) in the FontDescriptor Flags.
    ISO 19005-2, 6.2.11.6 requires non-symbolic simple fonts to have
    an explicit /Encoding entry.

    Args:
        font: pikepdf font object.

    Returns:
        True if the font has the Symbolic flag set.
    """
    try:
        font_descriptor = font.get("/FontDescriptor")
        if font_descriptor is None:
            return False
        font_descriptor = _resolve_indirect(font_descriptor)
        flags = font_descriptor.get("/Flags")
        if flags is None:
            return False
        return bool(int(flags) & 4)
    except Exception:
        return False


def get_font_name(font: pikepdf.Object) -> str:
    """Extracts the font name from a font object.

    Args:
        font: pikepdf font object.

    Returns:
        Font name as string.
    """
    base_font = font.get("/BaseFont")
    if base_font is not None:
        return _safe_str(base_font)[1:]  # Remove leading "/"
    return "Unknown"


def get_font_type(font: pikepdf.Object) -> str:
    """Determines the font type from a font object.

    Args:
        font: pikepdf font object.

    Returns:
        Font type as string.
    """
    subtype = font.get("/Subtype")
    if subtype is None:
        return "Unknown"

    subtype_str = _safe_str(subtype)
    type_mapping = {
        "/Type1": "Type1",
        "/TrueType": "TrueType",
        "/Type0": "CIDFont",
        "/Type3": "Type3",
        "/MMType1": "MMType1",
        "/CIDFontType0": "CIDFont",
        "/CIDFontType2": "CIDFont",
    }
    return type_mapping.get(subtype_str, subtype_str[1:])


def _is_subset_font(font_name: str) -> bool:
    """Checks if a font is a subset (6-letter prefix).

    Args:
        font_name: Name of the font.

    Returns:
        True if the font is a subset.
    """
    if "+" not in font_name:
        return False
    prefix = font_name.split("+")[0]
    return len(prefix) == 6 and prefix.isalpha() and prefix.isupper()


def get_base_font_name(font_name: str) -> str:
    """Removes the subset prefix from a font name.

    Args:
        font_name: Full font name (possibly with subset prefix).

    Returns:
        Font name without subset prefix.
    """
    if _is_subset_font(font_name):
        return font_name.split("+", 1)[1]
    return font_name


def is_font_embedded(font: pikepdf.Object) -> bool:
    """Checks if a font object has embedded font data.

    Args:
        font: pikepdf font object.

    Returns:
        True if the font is embedded.
    """
    subtype = font.get("/Subtype")
    subtype_str = str(subtype) if subtype else ""

    # Type3 fonts are always "embedded" (procedurally defined)
    if subtype_str == "/Type3":
        return True

    # Type0 (CIDFont) - check DescendantFonts
    if subtype_str == "/Type0":
        descendants = font.get("/DescendantFonts")
        if descendants is not None:
            for desc_font in descendants:
                if isinstance(desc_font, pikepdf.Object):
                    # Dereference if necessary
                    if isinstance(desc_font, pikepdf.Array):
                        continue
                    resolved = _resolve_indirect(desc_font)
                    if not _check_font_descriptor_embedded(resolved):
                        return False
            return True
        return False

    # Standard fonts without FontDescriptor check
    font_descriptor = font.get("/FontDescriptor")
    if font_descriptor is None:
        font_name = get_font_name(font)
        base_name = get_base_font_name(font_name)
        # Standard 14 fonts often have no FontDescriptor
        if base_name in STANDARD_14_FONTS:
            return False
        # No FontDescriptor means no embedded data, even with subset prefix
        return False

    return _check_font_descriptor_embedded(font)


def has_tounicode_cmap(font: pikepdf.Object) -> bool:
    """Checks if a font has a ToUnicode CMap.

    PDF/A-2/3 (all levels) require Unicode mappings per rule 6.2.11.7.2.
    This function checks for the presence of a /ToUnicode stream.

    Args:
        font: pikepdf font object.

    Returns:
        True if the font has a ToUnicode CMap.
    """
    subtype = font.get("/Subtype")
    subtype_str = str(subtype) if subtype else ""

    # Type3 fonts: check /ToUnicode directly
    if subtype_str == "/Type3":
        return font.get("/ToUnicode") is not None

    # Type0 (CIDFont): check /ToUnicode on the Type0 font
    if subtype_str == "/Type0":
        return font.get("/ToUnicode") is not None

    # Simple fonts (Type1, TrueType): check /ToUnicode directly
    return font.get("/ToUnicode") is not None


def can_derive_unicode(font: pikepdf.Object) -> bool:
    """Checks if Unicode mapping can be derived without a ToUnicode CMap.

    Per ISO 19005-2 Annex B, fonts can provide Unicode mappings through
    standard encodings (WinAnsiEncoding, MacRomanEncoding), encoding
    dictionaries with AGL-resolvable Differences, or CIDFonts with
    Identity CIDToGIDMap and embedded font data.

    Args:
        font: pikepdf font object.

    Returns:
        True if Unicode is derivable without an explicit ToUnicode CMap.
    """
    subtype = font.get("/Subtype")
    subtype_str = str(subtype) if subtype else ""

    # Simple Fonts: Type1, TrueType, MMType1, Type3
    if subtype_str in ("/Type1", "/TrueType", "/MMType1", "/Type3"):
        encoding = font.get("/Encoding")
        if encoding is None:
            return False
        if isinstance(encoding, pikepdf.Name):
            enc_name = _safe_str(encoding)
            return enc_name in (
                "/WinAnsiEncoding",
                "/MacRomanEncoding",
                "/StandardEncoding",
            )
        # Encoding dictionary
        try:
            return _can_derive_unicode_from_encoding_dict(encoding)
        except Exception:
            return False

    # CIDFont (Type0)
    if subtype_str == "/Type0":
        return _can_derive_unicode_from_cidfont(font)

    # Type3 and others: not derivable
    return False


def _can_derive_unicode_from_encoding_dict(encoding: pikepdf.Object) -> bool:
    """Checks if an Encoding dictionary allows Unicode derivation.

    Requires a standard BaseEncoding and all Differences glyph names
    must be resolvable via the Adobe Glyph List.

    Args:
        encoding: pikepdf Encoding dictionary object.

    Returns:
        True if Unicode is derivable from this encoding.
    """
    # Dereference if needed
    encoding = _resolve_indirect(encoding)

    base_encoding = encoding.get("/BaseEncoding")
    if base_encoding is not None:
        base_name = _safe_str(base_encoding)
        if base_name not in (
            "/WinAnsiEncoding",
            "/MacRomanEncoding",
            "/StandardEncoding",
        ):
            return False

    # Check Differences array: all glyph names must be AGL-resolvable
    differences = encoding.get("/Differences")
    if differences is None:
        return True

    for item in differences:
        try:
            int(item)
            continue
        except (TypeError, ValueError):
            pass
        if isinstance(item, pikepdf.Name):
            glyph_name = _safe_str(item)[1:]  # Remove leading "/"
            if glyph_name == ".notdef":
                continue
            if resolve_glyph_to_unicode(glyph_name) is None:
                return False

    return True


def _can_derive_unicode_from_cidfont(font: pikepdf.Object) -> bool:
    """Checks if a CIDFont (Type0) can derive Unicode without ToUnicode.

    Returns True if either:
    1. The encoding is a UTF-16/UCS-2 CMap (character codes are
       already Unicode values), or
    2. CIDToGIDMap is /Identity with embedded font data.

    Args:
        font: pikepdf Type0 font object.

    Returns:
        True if Unicode is derivable from embedded CIDFont data.
    """
    # UTF-16/UCS-2 encodings: character codes ARE Unicode values
    encoding = font.get("/Encoding")
    if encoding is not None:
        encoding_name = _get_encoding_name(encoding)
        if encoding_name in UTF16_ENCODING_NAMES:
            return True

    descendants = font.get("/DescendantFonts")
    if descendants is None or len(descendants) == 0:
        return False

    desc_font = _resolve_indirect(descendants[0])

    # Check CIDToGIDMap is /Identity (Name, not Stream)
    cid_to_gid = desc_font.get("/CIDToGIDMap")
    if not isinstance(cid_to_gid, pikepdf.Name):
        return False
    if str(cid_to_gid) != "/Identity":
        return False

    # Check embedded font data
    return _check_font_descriptor_embedded(desc_font)


def _has_valid_font_signature(data: bytes, key: str) -> bool:
    """Checks if font file data has a valid signature for its type.

    Args:
        data: Raw font file bytes.
        key: FontFile key ("/FontFile", "/FontFile2", or "/FontFile3").

    Returns:
        True if the data starts with a recognized font signature.
    """
    if len(data) < 4:
        return False
    magic = data[:4]
    if key == "/FontFile":
        # Type1: PFA starts with '%!' or PFB starts with 0x80
        return data[:2] == b"%!" or data[0:1] == b"\x80"
    if key == "/FontFile2":
        # TrueType/OpenType: 00010000, 'true', 'OTTO', or 'ttcf'
        return magic in (b"\x00\x01\x00\x00", b"true", b"OTTO", b"ttcf")
    if key == "/FontFile3":
        # OpenType or TrueType collection
        if magic in (b"OTTO", b"ttcf", b"\x00\x01\x00\x00"):
            return True
        # Raw CFF: major version 1, header size >= 4
        if data[0] == 1 and data[2] >= 4:
            return True
        return False
    return False


def _check_font_descriptor_embedded(font: pikepdf.Object) -> bool:
    """Checks if a font is embedded via its FontDescriptor.

    Validates that the FontFile stream exists, is non-empty, and
    contains a recognized font program signature.

    Args:
        font: pikepdf font object.

    Returns:
        True if embedded font data was found.
    """
    font_descriptor = font.get("/FontDescriptor")
    if font_descriptor is None:
        return False

    # Dereference FontDescriptor if necessary
    font_descriptor = _resolve_indirect(font_descriptor)

    # Check for embedded font data with valid content
    # /FontFile for Type1
    # /FontFile2 for TrueType
    # /FontFile3 for CIDFont/OpenType
    for key in ("/FontFile", "/FontFile2", "/FontFile3"):
        font_file = font_descriptor.get(key)
        if font_file is not None:
            try:
                resolved = _resolve_indirect(font_file)
                data = bytes(resolved.read_bytes())
                if _has_valid_font_signature(data, key):
                    return True
            except Exception:
                pass

    return False


def analyze_fonts(pdf: pikepdf.Pdf) -> list[FontInfo]:
    """Analyzes all fonts in the PDF document.

    Scans page-level Resources, Form XObjects, Tiling Patterns, and
    Annotation Appearance Streams recursively.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        List of FontInfo objects for all found fonts.
    """
    fonts_seen: dict[str, FontInfo] = {}
    seen_font_ids: set[tuple[int, int]] = set()

    for page_num, page in enumerate(pdf.pages, start=1):
        for font_key, font in iter_all_page_fonts(page):
            try:
                # Skip same indirect object already analyzed
                obj_key = font.objgen
                if obj_key != (0, 0):
                    if obj_key in seen_font_ids:
                        continue
                    seen_font_ids.add(obj_key)

                font_name = get_font_name(font)
                font_type = get_font_type(font)
                embedded = is_font_embedded(font)
                subset = _is_subset_font(font_name)
                tounicode = has_tounicode_cmap(font)
                derivable = can_derive_unicode(font)

                # Use combined key for deduplication
                key = f"{font_name}:{font_type}"
                if key not in fonts_seen:
                    fonts_seen[key] = FontInfo(
                        name=font_name,
                        type=font_type,
                        embedded=embedded,
                        subset=subset,
                        has_tounicode=tounicode,
                        unicode_derivable=derivable,
                    )
                    logger.debug(
                        "Font found on page %d: %s (%s, embedded=%s,"
                        " subset=%s, tounicode=%s, derivable=%s)",
                        page_num,
                        font_name,
                        font_type,
                        embedded,
                        subset,
                        tounicode,
                        derivable,
                    )
            except UnicodeDecodeError:
                logger.debug(
                    "Skipping font %s on page %d: non-UTF-8 bytes in font data",
                    font_key,
                    page_num,
                )
                continue
            except Exception as e:
                logger.warning(
                    "Error analyzing font %s on page %d: %s",
                    font_key,
                    page_num,
                    e,
                )
                continue

    # Scan AcroForm DR (Default Resources) fonts
    try:
        root = pdf.Root
        if root is not None and "/AcroForm" in root:
            acroform = _resolve_indirect(root.AcroForm)
            dr = acroform.get("/DR")
            if dr is not None:
                dr = _resolve_indirect(dr)
                font_dict = dr.get("/Font")
                if font_dict is not None:
                    font_dict = _resolve_indirect(font_dict)
                    for font_key in list(font_dict.keys()):
                        try:
                            font = _resolve_indirect(font_dict[font_key])
                            obj_key = font.objgen
                            if obj_key != (0, 0):
                                if obj_key in seen_font_ids:
                                    continue
                                seen_font_ids.add(obj_key)

                            font_name = get_font_name(font)
                            font_type = get_font_type(font)
                            embedded = is_font_embedded(font)
                            subset = _is_subset_font(font_name)
                            tounicode = has_tounicode_cmap(font)
                            derivable = can_derive_unicode(font)

                            key = f"{font_name}:{font_type}"
                            if key not in fonts_seen:
                                fonts_seen[key] = FontInfo(
                                    name=font_name,
                                    type=font_type,
                                    embedded=embedded,
                                    subset=subset,
                                    has_tounicode=tounicode,
                                    unicode_derivable=derivable,
                                )
                                logger.debug(
                                    "Font found in AcroForm DR: %s (%s, embedded=%s)",
                                    font_name,
                                    font_type,
                                    embedded,
                                )
                        except Exception:
                            continue
    except Exception:
        pass

    return list(fonts_seen.values())


def get_missing_fonts(pdf: pikepdf.Pdf) -> list[str]:
    """Determines all non-embedded fonts in the PDF.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        List of names of non-embedded fonts.
    """
    fonts = analyze_fonts(pdf)
    return [font.name for font in fonts if not font.embedded]


def check_font_compliance(
    pdf: pikepdf.Pdf,
    *,
    raise_on_error: bool = True,
) -> tuple[bool, list[str]]:
    """Checks PDF/A font compliance.

    PDF/A requires that all fonts are embedded.

    Args:
        pdf: Opened pikepdf PDF object.
        raise_on_error: If True, raises FontEmbeddingError when
            non-compliant fonts are found.

    Returns:
        Tuple of (compliant: bool, missing_fonts: list[str]).

    Raises:
        FontEmbeddingError: If raise_on_error=True and non-embedded
            fonts are found.
    """
    missing_fonts = get_missing_fonts(pdf)
    is_compliant = len(missing_fonts) == 0

    if not is_compliant:
        logger.warning(
            "PDF/A font compliance not met. Non-embedded fonts: %s",
            ", ".join(missing_fonts),
        )

        if raise_on_error:
            raise FontEmbeddingError(
                f"The following fonts are not embedded: {', '.join(missing_fonts)}. "
                "PDF/A requires all fonts to be embedded."
            )

    return is_compliant, missing_fonts


def get_fonts_missing_tounicode(pdf: pikepdf.Pdf) -> list[str]:
    """Determines all embedded fonts without ToUnicode CMap.

    For PDF/A-2/3 compliance (all levels), all embedded fonts must
    have Unicode mappings (ToUnicode CMap) per rule 6.2.11.7.2.

    Args:
        pdf: Opened pikepdf PDF object.

    Returns:
        List of names of embedded fonts missing ToUnicode.
    """
    fonts = analyze_fonts(pdf)
    return [
        font.name
        for font in fonts
        if font.embedded and not font.has_tounicode and not font.unicode_derivable
    ]


def check_unicode_compliance(
    pdf: pikepdf.Pdf,
    *,
    raise_on_error: bool = True,
) -> tuple[bool, list[str]]:
    """Checks PDF/A Unicode compliance (rule 6.2.11.7.2).

    PDF/A-2/3 (all levels) require that all fonts have Unicode
    mappings (ToUnicode CMaps) for text extraction.

    Args:
        pdf: Opened pikepdf PDF object.
        raise_on_error: If True, raises FontEmbeddingError when
            non-compliant fonts are found.

    Returns:
        Tuple of (compliant: bool, missing_tounicode: list[str]).

    Raises:
        FontEmbeddingError: If raise_on_error=True and fonts without
            ToUnicode are found.
    """
    missing_tounicode = get_fonts_missing_tounicode(pdf)
    is_compliant = len(missing_tounicode) == 0

    if not is_compliant:
        logger.info(
            "PDF/A Unicode compliance not met (rule 6.2.11.7.2). "
            "Fonts missing ToUnicode: %s",
            ", ".join(missing_tounicode),
        )

        if raise_on_error:
            raise FontEmbeddingError(
                "The following fonts lack ToUnicode mappings: "
                f"{', '.join(missing_tounicode)}. "
                "PDF/A-2/3 requires all fonts to have "
                "Unicode mappings (rule 6.2.11.7.2)."
            )

    return is_compliant, missing_tounicode
