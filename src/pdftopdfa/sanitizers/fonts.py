# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font structure sanitization for PDF/A-2/3 compliance.

PDF/A-2 (ISO 19005-2) has stricter CIDFont requirements than PDF/A-1:

- 6.2.11.3.1: CIDSystemInfo in CIDFont dict must be consistent with the CMap
- 6.2.11.3.2: CIDFontType2 fonts must have a valid CIDToGIDMap entry
- 6.2.11.3.3: CMap encoding must use standard names or valid embedded streams
- 6.2.11.4.2: CIDSet, if present, must be correct (safer to remove)
- 6.3.5: FontDescriptor /FontName must match /BaseFont (ignoring subset prefix)
- 6.3.7: Type1 /CharSet may be removed in PDF/A-2/3 (safer than rewriting)
"""

import logging
import re
from collections.abc import Iterator
from typing import Any

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from ..fonts.traversal import iter_all_page_fonts
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

# Named CMap → CIDSystemInfo mapping
# Registry, Ordering, Supplement
_NAMED_CMAP_CIDSYSTEMINFO: dict[str, tuple[str, str, int]] = {
    "Identity-H": ("Adobe", "Identity", 0),
    "Identity-V": ("Adobe", "Identity", 0),
    "UniGB-UTF16-H": ("Adobe", "GB1", 5),
    "UniGB-UTF16-V": ("Adobe", "GB1", 5),
    "UniJIS-UTF16-H": ("Adobe", "Japan1", 6),
    "UniJIS-UTF16-V": ("Adobe", "Japan1", 6),
    "UniCNS-UTF16-H": ("Adobe", "CNS1", 6),
    "UniCNS-UTF16-V": ("Adobe", "CNS1", 6),
    "UniKS-UTF16-H": ("Adobe", "Korea1", 2),
    "UniKS-UTF16-V": ("Adobe", "Korea1", 2),
}

# Predefined CMap names from ISO 32000-1 Table 118 + Identity.
# These are the only CMap names allowed as /Encoding values in PDF/A.
_STANDARD_CMAP_NAMES: set[str] = {
    # Identity
    "Identity-H",
    "Identity-V",
    # Japanese (Adobe-Japan1)
    "83pv-RKSJ-H",
    "90ms-RKSJ-H",
    "90ms-RKSJ-V",
    "90msp-RKSJ-H",
    "90msp-RKSJ-V",
    "90pv-RKSJ-H",
    "Add-RKSJ-H",
    "Add-RKSJ-V",
    "EUC-H",
    "EUC-V",
    "Ext-RKSJ-H",
    "Ext-RKSJ-V",
    "H",
    "V",
    "UniJIS-UCS2-H",
    "UniJIS-UCS2-V",
    "UniJIS-UCS2-HW-H",
    "UniJIS-UCS2-HW-V",
    "UniJIS-UTF16-H",
    "UniJIS-UTF16-V",
    # Chinese Simplified (Adobe-GB1)
    "GB-EUC-H",
    "GB-EUC-V",
    "GBpc-EUC-H",
    "GBpc-EUC-V",
    "GBK-EUC-H",
    "GBK-EUC-V",
    "GBKp-EUC-H",
    "GBKp-EUC-V",
    "GBK2K-H",
    "GBK2K-V",
    "UniGB-UCS2-H",
    "UniGB-UCS2-V",
    "UniGB-UTF16-H",
    "UniGB-UTF16-V",
    # Chinese Traditional (Adobe-CNS1)
    "B5pc-H",
    "B5pc-V",
    "HKscs-B5-H",
    "HKscs-B5-V",
    "ETen-B5-H",
    "ETen-B5-V",
    "ETenms-B5-H",
    "ETenms-B5-V",
    "CNS-EUC-H",
    "CNS-EUC-V",
    "UniCNS-UCS2-H",
    "UniCNS-UCS2-V",
    "UniCNS-UTF16-H",
    "UniCNS-UTF16-V",
    # Korean (Adobe-Korea1)
    "KSC-EUC-H",
    "KSC-EUC-V",
    "KSCms-UHC-H",
    "KSCms-UHC-V",
    "KSCms-UHC-HW-H",
    "KSCms-UHC-HW-V",
    "KSCpc-EUC-H",
    "UniKS-UCS2-H",
    "UniKS-UCS2-V",
    "UniKS-UTF16-H",
    "UniKS-UTF16-V",
}


def _get_cidsysteminfo_from_cmap(encoding: Any) -> tuple[str, str, int] | None:
    """Extract CIDSystemInfo from a CMap encoding entry.

    Args:
        encoding: The /Encoding value from a Type0 font (Name or Stream).

    Returns:
        Tuple of (Registry, Ordering, Supplement) or None if not determinable.
    """
    encoding = _resolve(encoding)

    # Named CMap
    if isinstance(encoding, Name):
        cmap_name = str(encoding).lstrip("/")
        return _NAMED_CMAP_CIDSYSTEMINFO.get(cmap_name)

    # Stream CMap: parse CIDSystemInfo from the stream data
    if isinstance(encoding, Stream):
        try:
            data = encoding.read_bytes().decode("latin-1")
        except Exception:
            return None

        # Match patterns like:
        #   /Registry (Adobe)
        #   /Ordering (Identity)
        #   /Supplement 0
        registry_match = re.search(r"/Registry\s*\(([^)]+)\)", data)
        ordering_match = re.search(r"/Ordering\s*\(([^)]+)\)", data)
        supplement_match = re.search(r"/Supplement\s+(\d+)", data)

        if registry_match and ordering_match and supplement_match:
            return (
                registry_match.group(1),
                ordering_match.group(1),
                int(supplement_match.group(1)),
            )

    return None


def _fix_cidsysteminfo(
    cidfont_dict: Dictionary,
    expected: tuple[str, str, int],
) -> bool:
    """Fix CIDSystemInfo in a CIDFont dictionary to match the expected values.

    Args:
        cidfont_dict: The CIDFont dictionary (modified in place).
        expected: Tuple of (Registry, Ordering, Supplement).

    Returns:
        True if CIDSystemInfo was modified.
    """
    registry, ordering, supplement = expected

    existing = cidfont_dict.get("/CIDSystemInfo")
    if existing is not None:
        existing = _resolve(existing)
        if isinstance(existing, Dictionary):
            existing_reg = str(existing.get("/Registry", ""))
            existing_ord = str(existing.get("/Ordering", ""))
            existing_sup = int(existing.get("/Supplement", -1))

            if (
                existing_reg == registry
                and existing_ord == ordering
                and existing_sup == supplement
            ):
                return False

    cidfont_dict[Name.CIDSystemInfo] = Dictionary(
        Registry=registry,
        Ordering=ordering,
        Supplement=supplement,
    )
    return True


def _iter_type0_fonts(pdf: Pdf) -> Iterator[tuple[str, Dictionary]]:
    """Yield (font_name, font_dict) for all Type0 fonts across all pages.

    Discovers fonts in all nested structures (Form XObjects, Tiling Patterns,
    Annotation Appearance Streams) via iter_all_page_fonts().

    Args:
        pdf: Opened pikepdf PDF object.

    Yields:
        Tuples of (font_name_str, resolved_font_dict).
    """
    seen_objgens: set[tuple[int, int]] = set()

    for page in pdf.pages:
        for font_name, font_obj in iter_all_page_fonts(pikepdf.Page(page)):
            font = _resolve(font_obj)
            if not isinstance(font, Dictionary):
                continue

            # Deduplicate by objgen
            objgen = font.objgen
            if objgen != (0, 0):
                if objgen in seen_objgens:
                    continue
                seen_objgens.add(objgen)

            subtype = font.get("/Subtype")
            if subtype is None or str(subtype) != "/Type0":
                continue

            yield font_name, font


def _is_embedded_font_descriptor(font_descriptor: Dictionary) -> bool:
    """Returns True when a FontDescriptor contains embedded font data."""
    return any(
        font_descriptor.get(key) is not None
        for key in (
            "/FontFile",
            "/FontFile2",
            "/FontFile3",
        )
    )


def _iter_embedded_type1_fonts(
    pdf: Pdf,
) -> Iterator[tuple[str, Dictionary, Dictionary]]:
    """Yield embedded Type1/MMType1 fonts with their FontDescriptor.

    Args:
        pdf: Opened pikepdf PDF object.

    Yields:
        Tuples of (font_name, font_dict, resolved_font_descriptor).
    """
    seen_objgens: set[tuple[int, int]] = set()

    for page in pdf.pages:
        for font_name, font_obj in iter_all_page_fonts(pikepdf.Page(page)):
            font = _resolve(font_obj)
            if not isinstance(font, Dictionary):
                continue

            objgen = font.objgen
            if objgen != (0, 0):
                if objgen in seen_objgens:
                    continue
                seen_objgens.add(objgen)

            subtype = font.get("/Subtype")
            subtype_str = str(subtype) if subtype is not None else ""
            if subtype_str not in ("/Type1", "/MMType1"):
                continue

            font_descriptor = font.get("/FontDescriptor")
            if font_descriptor is None:
                continue
            font_descriptor = _resolve(font_descriptor)
            if not isinstance(font_descriptor, Dictionary):
                continue
            if not _is_embedded_font_descriptor(font_descriptor):
                continue

            yield font_name, font, font_descriptor


def _sanitize_cmap_encoding(
    font_dict: Dictionary,
    font_name: str,
) -> dict[str, int]:
    """Fix CMap encoding issues for PDF/A-2 compliance (6.2.11.3.3).

    Handles three sub-issues:
    - Non-standard named CMaps replaced with /Identity-H or /Identity-V
    - WMode mismatch between CMap stream dict and stream content
    - /UseCMap references to non-predefined CMaps

    Args:
        font_dict: The Type0 font dictionary (modified in place).
        font_name: Font name for logging.

    Returns:
        Dict with counts: cmap_encoding_fixed, cmap_wmode_fixed.
    """
    counts: dict[str, int] = {"cmap_encoding_fixed": 0, "cmap_wmode_fixed": 0}

    encoding = font_dict.get("/Encoding")
    if encoding is None:
        return counts
    encoding = _resolve(encoding)

    # Case 1: Named CMap — must be in the standard set
    if isinstance(encoding, Name):
        cmap_name = str(encoding).lstrip("/")
        if cmap_name not in _STANDARD_CMAP_NAMES:
            replacement = "Identity-V" if cmap_name.endswith("-V") else "Identity-H"
            font_dict[Name.Encoding] = Name("/" + replacement)
            counts["cmap_encoding_fixed"] = 1
            logger.info(
                "Replaced non-standard CMap /%s with /%s for font %s",
                cmap_name,
                replacement,
                font_name,
            )
        return counts

    # Case 2: Stream CMap
    if not isinstance(encoding, Stream):
        return counts

    try:
        data = encoding.read_bytes().decode("latin-1")
    except Exception:
        return counts

    # 2a: WMode consistency — dict value must match stream content
    wmode_match = re.search(r"/WMode\s+(\d+)\s+def", data)
    if wmode_match:
        stream_wmode = int(wmode_match.group(1))
        dict_wmode_obj = encoding.get("/WMode")
        dict_wmode = int(dict_wmode_obj) if dict_wmode_obj is not None else None

        if dict_wmode != stream_wmode:
            encoding[Name("/WMode")] = stream_wmode
            counts["cmap_wmode_fixed"] = 1
            logger.info(
                "Fixed CMap WMode for font %s: dict had %s, stream defines %d",
                font_name,
                dict_wmode,
                stream_wmode,
            )

    # 2b: /UseCMap — must reference a predefined CMap name, not an
    # embedded stream or a non-standard name
    usecmap = encoding.get("/UseCMap")
    if usecmap is not None:
        usecmap_resolved = _resolve(usecmap)
        replace = False

        if isinstance(usecmap_resolved, Stream):
            replace = True
        elif isinstance(usecmap_resolved, Name):
            usecmap_name = str(usecmap_resolved).lstrip("/")
            if usecmap_name not in _STANDARD_CMAP_NAMES:
                replace = True

        if replace:
            # Try stripping only the /UseCMap entry to preserve real
            # character mappings (e.g. Shift-JIS → CIDs).
            try:
                del encoding["/UseCMap"]
                # Remove the `usecmap` operator from the CMap program text
                new_data = re.sub(r"(?m)^\s*\S+\s+usecmap\s*$", "", data)
                if new_data != data:
                    encoding.write(new_data.encode("latin-1"))
                counts["cmap_encoding_fixed"] = 1
                logger.warning(
                    "Stripped non-standard /UseCMap from CMap for font %s; "
                    "character mappings may be affected",
                    font_name,
                )
            except Exception as strip_err:
                # Fallback: wholesale replacement (destroys mappings)
                wmode = int(wmode_match.group(1)) if wmode_match else 0
                replacement = "Identity-V" if wmode == 1 else "Identity-H"
                font_dict[Name.Encoding] = Name("/" + replacement)
                counts["cmap_encoding_fixed"] = 1
                logger.warning(
                    "Replaced CMap with /%s for font %s "
                    "(stripping /UseCMap failed: %s); "
                    "character mappings were destroyed",
                    replacement,
                    font_name,
                    strip_err,
                )

    return counts


_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")


def _strip_subset_prefix(name: str) -> str:
    """Strip the 6-letter subset prefix (e.g. 'ABCDEF+Arial' → 'Arial')."""
    return _SUBSET_PREFIX_RE.sub("", name)


def sanitize_fontname_consistency(pdf: Pdf) -> dict[str, int]:
    """Ensure /FontName in FontDescriptor matches /BaseFont (ISO 19005-2 §6.3.5).

    For each font with a FontDescriptor, strips subset prefixes from both
    /BaseFont and /FontName and compares them. If they differ, /FontName is
    updated to match /BaseFont (preserving any subset prefix on /FontName).

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with ``{"fontname_fixed": N}``.
    """
    total_fixed = 0
    processed: set[tuple[int, int]] = set()

    for page in pdf.pages:
        for font_key, font_obj in iter_all_page_fonts(page):
            try:
                font = _resolve(font_obj)
                if not isinstance(font, Dictionary):
                    continue

                objgen = font.objgen
                if objgen != (0, 0):
                    if objgen in processed:
                        continue
                    processed.add(objgen)

                # For Type0 fonts, check DescendantFonts CIDFont too
                subtype = font.get("/Subtype")
                subtype_str = str(subtype) if subtype is not None else ""

                fonts_to_check: list[tuple[str, Dictionary]] = []

                if subtype_str == "/Type0":
                    desc = font.get("/DescendantFonts")
                    if desc is not None:
                        desc = _resolve(desc)
                        if isinstance(desc, Array) and len(desc) > 0:
                            cidfont = _resolve(desc[0])
                            if isinstance(cidfont, Dictionary):
                                fonts_to_check.append((font_key, cidfont))
                    # Also check the Type0 font itself
                    fonts_to_check.append((font_key, font))
                else:
                    fonts_to_check.append((font_key, font))

                for fk, fdict in fonts_to_check:
                    base_font = fdict.get("/BaseFont")
                    if base_font is None:
                        continue
                    base_font_str = str(base_font).lstrip("/")

                    fd = fdict.get("/FontDescriptor")
                    if fd is None:
                        continue
                    fd = _resolve(fd)
                    if not isinstance(fd, Dictionary):
                        continue

                    fd_objgen = fd.objgen
                    if fd_objgen != (0, 0):
                        if fd_objgen in processed:
                            continue
                        processed.add(fd_objgen)

                    font_name = fd.get("/FontName")
                    if font_name is None:
                        continue
                    font_name_str = str(font_name).lstrip("/")

                    base_core = _strip_subset_prefix(base_font_str)
                    name_core = _strip_subset_prefix(font_name_str)

                    if base_core != name_core:
                        # Set FontName to match BaseFont exactly
                        new_font_name = base_font_str
                        fd[Name.FontName] = Name("/" + new_font_name)
                        total_fixed += 1
                        logger.info(
                            "Fixed FontDescriptor /FontName: %s → %s "
                            "(to match /BaseFont %s)",
                            font_name_str,
                            new_font_name,
                            base_font_str,
                        )

            except Exception as e:
                logger.debug(
                    "Error checking FontName for font %s: %s",
                    font_key,
                    e,
                )
                continue

    if total_fixed > 0:
        logger.info(
            "FontName consistency: %d font descriptor(s) fixed",
            total_fixed,
        )

    return {"fontname_fixed": total_fixed}


def _check_cid_values_over_65535(cidfont: Dictionary, font_name: str) -> int:
    """Check /W, /W2, and /CIDToGIDMap for CID values exceeding 65535.

    Logs a warning for each violation found and returns the count.
    Cannot fix these automatically — out-of-range CID values indicate a
    severely malformed font (ISO 19005-2 rule 6.1.13-10).
    """
    warned = 0
    for key in ("/W", "/W2"):
        w = cidfont.get(key)
        if w is None:
            continue
        w = _resolve(w)
        if not isinstance(w, Array):
            continue
        items = list(w)
        i = 0
        while i < len(items):
            item = _resolve(items[i])
            if isinstance(item, Array):
                # Unexpected array at top level; skip
                i += 1
                continue
            try:
                cid = int(item)
            except Exception:
                i += 1
                continue
            if i + 1 < len(items):
                nxt = _resolve(items[i + 1])
                if isinstance(nxt, Array):
                    # Format 1: cid [w1 w2 ...]
                    if cid > 65535:
                        logger.warning(
                            "CIDFont %s: CID %d in /%s exceeds 65535"
                            " (ISO 19005-2 rule 6.1.13-10); cannot fix automatically",
                            font_name,
                            cid,
                            key[1:],
                        )
                        warned += 1
                    i += 2
                else:
                    # Format 2: cid_first cid_last width
                    try:
                        cid_last = int(nxt)
                    except Exception:
                        i += 1
                        continue
                    if cid > 65535 or cid_last > 65535:
                        logger.warning(
                            "CIDFont %s: CID range %d\u2013%d in /%s exceeds 65535"
                            " (ISO 19005-2 rule 6.1.13-10); cannot fix automatically",
                            font_name,
                            cid,
                            cid_last,
                            key[1:],
                        )
                        warned += 1
                    i += 3
            else:
                i += 1

    # /CIDToGIDMap stream: 2 bytes per CID entry; > 131072 bytes means CIDs > 65535
    cidtogidmap = cidfont.get("/CIDToGIDMap")
    if cidtogidmap is not None:
        resolved = _resolve(cidtogidmap)
        if isinstance(resolved, Stream):
            data = resolved.read_bytes()
            if len(data) > 131072:
                logger.warning(
                    "CIDFont %s: /CIDToGIDMap stream length %d implies CID values"
                    " > 65535 (ISO 19005-2 rule 6.1.13-10); cannot fix automatically",
                    font_name,
                    len(data),
                )
                warned += 1

    return warned


def sanitize_cidfont_structures(pdf: Pdf) -> dict[str, int]:
    """Sanitize CIDFont structures for PDF/A-2 compliance.

    Fixes:
    1. CMap encoding issues — non-standard names, WMode, UseCMap (6.2.11.3.3)
    2. CIDSystemInfo consistency between CMap and CIDFont dict (6.2.11.3.1)
    3. Missing/invalid CIDToGIDMap on CIDFontType2 fonts (6.2.11.3.2)
    4. Removes CIDSet entries that may be incorrect (6.2.11.4.2)
    5. Removes Type1/MMType1 FontDescriptor /CharSet (allowed in PDF/A-2/3)

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with counts of fixes applied.
    """
    result: dict[str, int] = {
        "cidsysteminfo_fixed": 0,
        "cidtogidmap_fixed": 0,
        "cidset_removed": 0,
        "type1_charset_removed": 0,
        "cmap_encoding_fixed": 0,
        "cmap_wmode_fixed": 0,
        "cid_values_over_65535_warned": 0,
    }

    for font_name, font_dict in _iter_type0_fonts(pdf):
        descendant_fonts = font_dict.get("/DescendantFonts")

        if descendant_fonts is None:
            continue
        descendant_fonts = _resolve(descendant_fonts)
        if not isinstance(descendant_fonts, Array):
            continue
        if len(descendant_fonts) == 0:
            continue

        cidfont = _resolve(descendant_fonts[0])
        if not isinstance(cidfont, Dictionary):
            continue

        cidfont_subtype = cidfont.get("/Subtype")
        cidfont_subtype_str = str(cidfont_subtype) if cidfont_subtype else ""

        # 1. Fix CMap encoding issues (6.2.11.3.3) — must run before
        #    CIDSystemInfo fix so the encoding is correct when we look it up.
        cmap_counts = _sanitize_cmap_encoding(font_dict, font_name)
        result["cmap_encoding_fixed"] += cmap_counts["cmap_encoding_fixed"]
        result["cmap_wmode_fixed"] += cmap_counts["cmap_wmode_fixed"]

        # Re-read encoding after possible CMap fix
        encoding = font_dict.get("/Encoding")

        # 2. Fix CIDSystemInfo consistency (6.2.11.3.1)
        if encoding is not None:
            expected_info = _get_cidsysteminfo_from_cmap(encoding)
            if expected_info is not None:
                # Don't force Identity-0 on CIDFontType0 — these fonts use
                # their CFF program's internal CID ordering, not GID mapping.
                _registry, ordering, _supplement = expected_info
                skip = ordering == "Identity" and cidfont_subtype_str == "/CIDFontType0"
                if not skip and _fix_cidsysteminfo(cidfont, expected_info):
                    result["cidsysteminfo_fixed"] += 1
                    logger.info(
                        "Fixed CIDSystemInfo for font %s: %s-%s-%d",
                        font_name,
                        *expected_info,
                    )

        # 3. Fix CIDToGIDMap for CIDFontType2 (6.2.11.3.2)
        if cidfont_subtype_str == "/CIDFontType2":
            cidtogidmap = cidfont.get("/CIDToGIDMap")
            needs_fix = cidtogidmap is None
            if not needs_fix and cidtogidmap is not None:
                resolved = _resolve(cidtogidmap)
                # Valid values: /Identity (Name) or a Stream
                if not isinstance(resolved, Stream):
                    if not isinstance(resolved, Name) or str(resolved) != "/Identity":
                        needs_fix = True
            if needs_fix:
                cidfont[Name.CIDToGIDMap] = Name.Identity
                result["cidtogidmap_fixed"] += 1
                logger.info(
                    "Fixed CIDToGIDMap for CIDFontType2 font %s",
                    font_name,
                )

        # 4. Check for CID values > 65535 (6.1.13-10) — warn only, cannot fix
        result["cid_values_over_65535_warned"] += _check_cid_values_over_65535(
            cidfont, font_name
        )

        # 5. Remove CIDSet from font descriptor
        font_descriptor = cidfont.get("/FontDescriptor")
        if font_descriptor is not None:
            font_descriptor = _resolve(font_descriptor)
            if isinstance(font_descriptor, Dictionary):
                if "/CIDSet" in font_descriptor:
                    del font_descriptor[Name.CIDSet]
                    result["cidset_removed"] += 1
                    logger.info(
                        "Removed CIDSet from font descriptor for %s",
                        font_name,
                    )

    # 5. For PDF/A-2/3, /CharSet in Type1 font descriptors is optional.
    # Removing it avoids stale/incomplete lists after font manipulations.
    for font_name, _font_dict, font_descriptor in _iter_embedded_type1_fonts(pdf):
        if "/CharSet" not in font_descriptor:
            continue
        del font_descriptor[Name("/CharSet")]
        result["type1_charset_removed"] += 1
        logger.info(
            "Removed Type1 /CharSet from font descriptor for %s",
            font_name,
        )

    if any(v > 0 for v in result.values()):
        logger.info(
            "CIDFont sanitization: %d CIDSystemInfo fixed, "
            "%d CIDToGIDMap fixed, %d CIDSet removed, "
            "%d Type1 /CharSet removed, %d CMap encoding fixed, "
            "%d CMap WMode fixed, %d CID-value-over-65535 warned",
            result["cidsysteminfo_fixed"],
            result["cidtogidmap_fixed"],
            result["cidset_removed"],
            result["type1_charset_removed"],
            result["cmap_encoding_fixed"],
            result["cmap_wmode_fixed"],
            result["cid_values_over_65535_warned"],
        )

    return result
