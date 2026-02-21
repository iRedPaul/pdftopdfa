# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font dictionary structure sanitizer for PDF/A compliance.

Covers ISO 19005-2 rules 6.2.11.2-1 through 6.2.11.2-7:
- 6.2.11.2-1: Every font dict must have /Type /Font
- 6.2.11.2-2: /Subtype must be present and valid
- 6.2.11.2-3: /BaseFont must be present (except Type3)
- 6.2.11.2-4: /FirstChar must be present for simple fonts
- 6.2.11.2-5: /LastChar must be present for simple fonts
- 6.2.11.2-6: /Widths array length must equal LastChar - FirstChar + 1
- 6.2.11.2-7: Font stream /Subtype must be valid (FontFile3) or absent
"""

import logging

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf

from ..fonts.constants import STANDARD_14_FONTS
from ..fonts.traversal import iter_all_page_fonts
from ..fonts.utils import safe_str as _safe_str
from ..utils import resolve_indirect as _resolve

logger = logging.getLogger(__name__)

_VALID_SUBTYPES = frozenset(
    {
        "/Type1",
        "/MMType1",
        "/TrueType",
        "/Type3",
        "/Type0",
        "/CIDFontType0",
        "/CIDFontType2",
    }
)

_VALID_FONTFILE3_SUBTYPES = frozenset(
    {
        "/Type1C",
        "/CIDFontType0C",
        "/OpenType",
    }
)

# Subtypes that require /FirstChar, /LastChar, /Widths
_SIMPLE_FONT_SUBTYPES = frozenset(
    {
        "/Type1",
        "/MMType1",
        "/TrueType",
    }
)

# Subtypes that do NOT require /FirstChar, /LastChar, /Widths
_SKIP_CHAR_RANGE_SUBTYPES = frozenset(
    {
        "/Type0",
        "/Type3",
        "/CIDFontType0",
        "/CIDFontType2",
    }
)


def sanitize_font_structure(pdf: Pdf) -> dict[str, int]:
    """Fixes broken font dictionary structure for PDF/A compliance.

    Iterates all fonts in the PDF and applies structural fixes per
    ISO 19005-2, clause 6.2.11.2.

    Args:
        pdf: Opened pikepdf PDF object (modified in place).

    Returns:
        Dictionary with counts of fixes applied.
    """
    result: dict[str, int] = {
        "font_type_added": 0,
        "font_subtype_fixed": 0,
        "font_basefont_added": 0,
        "font_firstchar_added": 0,
        "font_lastchar_added": 0,
        "font_widths_size_fixed": 0,
        "font_stream_subtype_removed": 0,
    }

    for font in _iter_all_fonts(pdf):
        try:
            _fix_type(font, result)
            _fix_subtype(font, result)
            _fix_basefont(font, result)
            _fix_char_range_and_widths(font, result)
            _fix_font_stream_subtype(font, result)
        except Exception as e:
            name = _safe_str(font.get("/BaseFont") or b"")
            logger.debug("Skipping broken font object %s: %s", name, e)
            continue

    total = sum(result.values())
    if total > 0:
        logger.info(
            "Font structure sanitization: %d /Type added, %d /Subtype fixed, "
            "%d /BaseFont added, %d /FirstChar added, %d /LastChar added, "
            "%d /Widths size fixed, %d stream subtypes removed",
            result["font_type_added"],
            result["font_subtype_fixed"],
            result["font_basefont_added"],
            result["font_firstchar_added"],
            result["font_lastchar_added"],
            result["font_widths_size_fixed"],
            result["font_stream_subtype_removed"],
        )

    return result


def _iter_all_fonts(pdf: Pdf):
    """Yields all font dictionaries from all pages, including CIDFont descendants.

    Unlike _iter_all_embedded_fonts in font_widths.py, this does NOT skip
    fonts that are missing /Subtype — structural fixes need to see them.

    Args:
        pdf: Opened pikepdf PDF object.

    Yields:
        Resolved font Dictionary objects.
    """
    seen_objgens: set[tuple[int, int]] = set()

    for page in pdf.pages:
        for _font_key, font_obj in iter_all_page_fonts(pikepdf.Page(page)):
            font = _resolve(font_obj)
            if not isinstance(font, Dictionary):
                continue

            # Deduplicate by objgen
            objgen = font.objgen
            if objgen != (0, 0):
                if objgen in seen_objgens:
                    continue
                seen_objgens.add(objgen)

            yield font

            # Also yield CIDFont descendants from /DescendantFonts
            descendants = font.get("/DescendantFonts")
            if descendants is None:
                continue
            try:
                descendants = _resolve(descendants)
                if not isinstance(descendants, Array):
                    continue
                for item in descendants:
                    try:
                        desc = _resolve(item)
                        if not isinstance(desc, Dictionary):
                            continue
                        desc_objgen = desc.objgen
                        if desc_objgen != (0, 0):
                            if desc_objgen in seen_objgens:
                                continue
                            seen_objgens.add(desc_objgen)
                        yield desc
                    except Exception:
                        continue
            except Exception:
                continue


def _fix_type(font: pikepdf.Object, result: dict[str, int]) -> None:
    """Rule 6.2.11.2-1: Add /Type /Font if missing."""
    if font.get("/Type") is None:
        font[Name.Type] = Name.Font
        result["font_type_added"] += 1


def _infer_subtype(font: pikepdf.Object) -> str | None:
    """Infers the /Subtype for a font dictionary from available evidence.

    Returns the inferred subtype string (e.g. "/TrueType") or None if
    the subtype cannot be reliably determined.
    """
    # 1. CharProcs → Type3
    if font.get("/CharProcs") is not None:
        return "/Type3"

    # 2. DescendantFonts → Type0
    if font.get("/DescendantFonts") is not None:
        return "/Type0"

    # 3. CIDSystemInfo → CIDFont family
    if font.get("/CIDSystemInfo") is not None:
        fd = font.get("/FontDescriptor")
        if fd is not None:
            try:
                fd = _resolve(fd)
                if fd.get("/FontFile2") is not None:
                    return "/CIDFontType2"
            except Exception:
                pass
        return "/CIDFontType0"

    # Inspect FontDescriptor for font file type
    fd = font.get("/FontDescriptor")
    if fd is None:
        return None
    try:
        fd = _resolve(fd)
    except Exception:
        return None

    # 4. FontFile2 → TrueType
    if fd.get("/FontFile2") is not None:
        return "/TrueType"

    # 5. FontFile3 — subtype depends on stream /Subtype
    ff3 = fd.get("/FontFile3")
    if ff3 is not None:
        try:
            ff3 = _resolve(ff3)
            subtype_obj = ff3.get("/Subtype")
            if subtype_obj is not None:
                st = _safe_str(subtype_obj)
                if st == "/Type1C":
                    return "/Type1"
                if st == "/OpenType":
                    return "/TrueType"
                # CIDFontType0C → handled via CIDSystemInfo path above
        except Exception:
            pass
        return None  # Can't determine reliably

    # 6. FontFile → Type1
    if fd.get("/FontFile") is not None:
        return "/Type1"

    return None


def _fix_subtype(font: pikepdf.Object, result: dict[str, int]) -> None:
    """Rule 6.2.11.2-2: Add or correct /Subtype if missing/invalid."""
    current = font.get("/Subtype")
    if current is not None:
        if _safe_str(current) in _VALID_SUBTYPES:
            return  # Already valid

    inferred = _infer_subtype(font)
    if inferred is None:
        logger.debug(
            "Cannot infer /Subtype for font %s — skipping",
            _safe_str(font.get("/BaseFont") or b""),
        )
        return

    font[Name.Subtype] = Name(inferred)
    result["font_subtype_fixed"] += 1


def _fix_basefont(font: pikepdf.Object, result: dict[str, int]) -> None:
    """Rule 6.2.11.2-3: Add /BaseFont from FontDescriptor if missing.

    Skipped for Type3 fonts (not required by spec).
    """
    subtype = font.get("/Subtype")
    if subtype is not None and _safe_str(subtype) == "/Type3":
        return

    if font.get("/BaseFont") is not None:
        return

    # Try to get name from FontDescriptor
    fd = font.get("/FontDescriptor")
    if fd is None:
        logger.debug("No /BaseFont and no /FontDescriptor — cannot add /BaseFont")
        return
    try:
        fd = _resolve(fd)
        font_name = fd.get("/FontName")
        if font_name is not None:
            font[Name.BaseFont] = font_name
            result["font_basefont_added"] += 1
        else:
            logger.debug("FontDescriptor has no /FontName — cannot add /BaseFont")
    except Exception as e:
        logger.debug("Error reading FontDescriptor for /BaseFont: %s", e)


def _fix_char_range_and_widths(font: pikepdf.Object, result: dict[str, int]) -> None:
    """Rules 6.2.11.2-4, -5, -6: Fix FirstChar/LastChar and Widths array size.

    Skipped for Type0, Type3, CIDFontType0, CIDFontType2, and Standard-14 fonts.
    """
    subtype = font.get("/Subtype")
    if subtype is None:
        return
    subtype_str = _safe_str(subtype)

    if subtype_str in _SKIP_CHAR_RANGE_SUBTYPES:
        return

    # Standard-14 fonts don't need Widths (strip subset prefix ABCDEF+)
    if subtype_str in ("/Type1", "/MMType1"):
        base_font = font.get("/BaseFont")
        if base_font is not None:
            base_name = _safe_str(base_font)
            # Strip leading "/" and optional "ABCDEF+" subset prefix
            if base_name.startswith("/"):
                base_name = base_name[1:]
            if "+" in base_name:
                base_name = base_name.split("+", 1)[1]
            if base_name in STANDARD_14_FONTS:
                return

    # Read existing FirstChar/LastChar/Widths
    first_obj = font.get("/FirstChar")
    last_obj = font.get("/LastChar")
    widths_obj = font.get("/Widths")

    # Rule 6.2.11.2-6 (missing Widths): create zero-filled array when the
    # char range is known.  font_widths.py will correct the values later.
    if widths_obj is None:
        if first_obj is not None and last_obj is not None:
            try:
                first = int(first_obj)
                last = int(last_obj)
                expected = last - first + 1
                if expected > 0:
                    font[Name.Widths] = Array([0] * expected)
                    result["font_widths_size_fixed"] += 1
            except (TypeError, ValueError):
                pass
        return  # can't derive char range from a non-existent Widths

    try:
        widths_obj = _resolve(widths_obj)
        if not isinstance(widths_obj, Array):
            return
        n = len(widths_obj)
    except Exception:
        return

    # Read existing FirstChar/LastChar (widths_obj is present)

    first: int | None = None
    last: int | None = None
    first_added = False
    last_added = False

    try:
        if first_obj is not None:
            first = int(first_obj)
    except (TypeError, ValueError):
        pass

    try:
        if last_obj is not None:
            last = int(last_obj)
    except (TypeError, ValueError):
        pass

    # Derive missing values
    if first is None and last is not None:
        first = last - n + 1
        font[Name.FirstChar] = first
        result["font_firstchar_added"] += 1
        first_added = True
    elif last is None and first is not None:
        last = first + n - 1
        font[Name.LastChar] = last
        result["font_lastchar_added"] += 1
        last_added = True
    elif first is None and last is None:
        first = 0
        last = n - 1
        font[Name.FirstChar] = first
        font[Name.LastChar] = last
        result["font_firstchar_added"] += 1
        result["font_lastchar_added"] += 1
        first_added = True
        last_added = True

    if first is None or last is None:
        return

    # Fix Widths array size (rule 6.2.11.2-6)
    expected = last - first + 1
    if expected < 0:
        return

    actual = n
    if actual == expected:
        return

    if actual < expected:
        # Pad with zeros
        new_widths = list(widths_obj) + [0] * (expected - actual)
    else:
        # Truncate
        new_widths = list(widths_obj)[:expected]

    font[Name.Widths] = Array(new_widths)
    result["font_widths_size_fixed"] += 1

    _ = first_added or last_added  # silence unused warning


def _fix_font_stream_subtype(font: pikepdf.Object, result: dict[str, int]) -> None:
    """Rule 6.2.11.2-7: Validate/remove /Subtype on font file streams.

    - /FontFile and /FontFile2: spec does not use /Subtype — remove if present.
    - /FontFile3: /Subtype must be Type1C, CIDFontType0C, or OpenType.
      If missing or invalid, remove the key (safest fix).
    """
    fd = font.get("/FontDescriptor")
    if fd is None:
        return
    try:
        fd = _resolve(fd)
    except Exception:
        return

    # FontFile and FontFile2: must not have /Subtype
    for key in ("/FontFile", "/FontFile2"):
        ff = fd.get(key)
        if ff is None:
            continue
        try:
            ff = _resolve(ff)
            if ff.get("/Subtype") is not None:
                del ff["/Subtype"]
                result["font_stream_subtype_removed"] += 1
        except Exception:
            continue

    # FontFile3: /Subtype must be in _VALID_FONTFILE3_SUBTYPES
    ff3 = fd.get("/FontFile3")
    if ff3 is None:
        return
    try:
        ff3 = _resolve(ff3)
        subtype = ff3.get("/Subtype")
        if subtype is None:
            return
        if _safe_str(subtype) not in _VALID_FONTFILE3_SUBTYPES:
            del ff3["/Subtype"]
            result["font_stream_subtype_removed"] += 1
    except Exception:
        pass
