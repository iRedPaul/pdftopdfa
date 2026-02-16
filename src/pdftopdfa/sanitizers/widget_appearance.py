# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Widget annotation appearance stream generation for PDF/A compliance.

Generates visible appearance streams for form field (Widget) annotations.
PDF/A requires /AP /N on all annotations; this module creates meaningful
visual representations instead of empty streams.
"""

import logging
import math
import re

from pikepdf import Array, Dictionary, Name, Pdf

from ..fonts.tounicode import parse_tounicode_cmap as _parse_tounicode_cmap
from ..utils import resolve_indirect as _resolve
from . import font_metrics as _fm

logger = logging.getLogger(__name__)


def create_widget_appearance(pdf: Pdf, annot, acroform=None):
    """Create a visible appearance stream for a Widget annotation.

    Dispatches by field type (/FT) to build an appropriate visual
    representation. Falls back through 3 levels:
    1. Full widget appearance (text + border + background)
    2. Border-only (visible frame, no text)
    3. Empty stream (current behavior, last resort)

    Args:
        pdf: Opened pikepdf PDF object.
        annot: Resolved annotation dictionary.
        acroform: The document's /AcroForm dictionary (optional).

    Returns:
        A pikepdf Stream (single appearance) or Dictionary (state dict
        for checkboxes/radios) suitable for /AP /N.
    """
    try:
        ft = _get_inheritable(annot, "/FT", acroform)
        ft_str = str(ft) if ft is not None else None

        if ft_str == "/Tx":
            _remove_rich_text(annot)
            return _build_text_field_appearance(pdf, annot, acroform)
        elif ft_str == "/Btn":
            return _build_button_appearance(pdf, annot, acroform)
        elif ft_str == "/Ch":
            return _build_choice_field_appearance(pdf, annot, acroform)
        elif ft_str == "/Sig":
            return _build_signature_appearance(pdf, annot)
        else:
            return _build_border_only_appearance(pdf, annot)
    except Exception:
        logger.debug(
            "Widget appearance generation failed, using border-only fallback",
            exc_info=True,
        )
        try:
            return _build_border_only_appearance(pdf, annot)
        except Exception:
            logger.debug(
                "Border-only fallback failed, using empty stream",
                exc_info=True,
            )
            return _make_empty_stream(pdf, annot)


# ---------------------------------------------------------------------------
# Attribute inheritance
# ---------------------------------------------------------------------------


def _get_inheritable(annot, key, acroform=None):
    """Retrieve an inheritable attribute from the annotation hierarchy.

    Walks the /Parent chain, then falls back to AcroForm defaults.

    Args:
        annot: Annotation/field dictionary.
        key: The key to look up (e.g. "/FT", "/DA", "/DR", "/Q").
        acroform: The document's /AcroForm dictionary (optional).

    Returns:
        The value if found, otherwise None.
    """
    visited: set[tuple[int, int]] = set()
    current = annot
    while current is not None:
        try:
            objgen = current.objgen
        except Exception:
            objgen = (0, 0)
        if objgen != (0, 0):
            if objgen in visited:
                break
            visited.add(objgen)

        val = current.get(key)
        if val is not None:
            return val

        parent = current.get("/Parent")
        if parent is not None:
            try:
                current = _resolve(parent)
            except Exception:
                break
        else:
            break

    # Fall back to AcroForm defaults
    if acroform is not None:
        val = acroform.get(key)
        if val is not None:
            return val

    return None


def _remove_rich_text(annot):
    """Remove /RV (Rich Text) entries from a text field and its parents.

    PDF/A prohibits Rich Text in form fields. This walks the annotation
    and its /Parent chain, deleting any /RV key found.

    Args:
        annot: Annotation/field dictionary.
    """
    visited: set[tuple[int, int]] = set()
    current = annot
    while current is not None:
        try:
            objgen = current.objgen
        except Exception:
            objgen = (0, 0)
        if objgen != (0, 0):
            if objgen in visited:
                break
            visited.add(objgen)

        try:
            if "/RV" in current:
                del current["/RV"]
                logger.debug("Removed /RV (Rich Text) from field")
        except Exception:
            pass

        parent = current.get("/Parent")
        if parent is not None:
            try:
                current = _resolve(parent)
            except Exception:
                break
        else:
            break


# ---------------------------------------------------------------------------
# DA string parsing
# ---------------------------------------------------------------------------

# Regex to extract font name and size from DA string like "/Helv 12 Tf"
_DA_FONT_RE = re.compile(r"/(\S+)\s+([\d.]+)\s+Tf")


def _parse_da_string(da):
    """Parse a /DA (Default Appearance) string.

    Extracts font name, font size, and color operators.

    Args:
        da: The DA string value.

    Returns:
        Tuple of (font_name, font_size, color_ops) where:
        - font_name: str or None (e.g. "Helv")
        - font_size: float (0.0 means auto-size, 12.0 when missing)
        - color_ops: str -- the color-setting portion of DA
    """
    if da is None:
        return None, 12.0, ""

    da_str = str(da)
    if not da_str.strip():
        return None, 12.0, ""

    font_name = None
    font_size = 12.0

    m = _DA_FONT_RE.search(da_str)
    if m:
        font_name = m.group(1)
        try:
            font_size = float(m.group(2))
        except ValueError:
            font_size = 12.0
        if font_size < 0:
            font_size = 12.0
        # font_size == 0 means auto-size: keep it as 0.0

    # Extract color operators: everything before the Tf operator
    color_ops = ""
    color_part = _DA_FONT_RE.sub("", da_str).strip()
    if color_part:
        color_ops = color_part

    return font_name, font_size, color_ops


# ---------------------------------------------------------------------------
# Font resource resolution
# ---------------------------------------------------------------------------


def _resolve_font_resource(font_name, annot, acroform):
    """Resolve a font resource dictionary from widget or AcroForm /DR.

    Args:
        font_name: The font name from DA (without leading /).
        annot: The annotation dictionary.
        acroform: The /AcroForm dictionary.

    Returns:
        The font dictionary if found, otherwise None.
    """
    if font_name is None:
        return None

    # Check annotation's own /DR
    dr = _get_inheritable(annot, "/DR", acroform)
    if dr is not None:
        try:
            dr = _resolve(dr)
            fonts = dr.get("/Font")
            if fonts is not None:
                fonts = _resolve(fonts)
                font_obj = fonts.get(Name("/" + font_name))
                if font_obj is not None:
                    return _resolve(font_obj)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _color_array_to_ops(arr, stroke=False):
    """Convert a /MK color array to PDF content stream operators.

    Args:
        arr: pikepdf Array of color components.
        stroke: If True, use stroke operators (G/RG/K); otherwise fill (g/rg/k).

    Returns:
        Color operator string, e.g. "0.5 g" or "1 0 0 RG".
    """
    if arr is None:
        return ""

    try:
        components = [float(c) for c in arr]
    except (TypeError, ValueError):
        return ""

    n = len(components)
    if n == 0:
        return ""
    elif n == 1:
        op = "G" if stroke else "g"
        return f"{components[0]:.4g} {op}"
    elif n == 3:
        op = "RG" if stroke else "rg"
        return f"{components[0]:.4g} {components[1]:.4g} {components[2]:.4g} {op}"
    elif n == 4:
        op = "K" if stroke else "k"
        return (
            f"{components[0]:.4g} {components[1]:.4g} "
            f"{components[2]:.4g} {components[3]:.4g} {op}"
        )
    return ""


# ---------------------------------------------------------------------------
# String escaping
# ---------------------------------------------------------------------------


def _escape_pdf_string(text):
    """Escape special characters for a PDF literal string in a content stream.

    Args:
        text: The text string to escape.

    Returns:
        Escaped string safe for use within parentheses in PDF operators.
    """
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\\", "\\\\")
    text = text.replace("(", "\\(")
    text = text.replace(")", "\\)")
    return text


# ---------------------------------------------------------------------------
# CID font helpers
# ---------------------------------------------------------------------------


def _is_cid_font(font_resource):
    """Check if font is a Type0/CID font.

    Args:
        font_resource: Font dictionary.

    Returns:
        True if the font is a Type0 (CID) font.
    """
    try:
        subtype = font_resource.get("/Subtype")
        return subtype is not None and str(subtype) == "/Type0"
    except Exception:
        return False


def _build_unicode_to_code_map(font_resource):
    """Build reverse Unicode-to-character-code mapping from /ToUnicode CMap.

    For CID fonts with a /ToUnicode CMap, parses the CMap to get
    code->unicode mappings and inverts them. For Identity-H/V without
    ToUnicode, returns an empty dict (caller falls back to ord(ch)).

    Args:
        font_resource: Font dictionary.

    Returns:
        Dict mapping Unicode codepoints to character codes.
    """
    try:
        to_unicode = font_resource.get("/ToUnicode")
        if to_unicode is not None:
            to_unicode = _resolve(to_unicode)
            data = bytes(to_unicode.read_bytes())
            code_to_unicode = _parse_tounicode_cmap(data)
            # Invert: unicode -> code
            return {uni: code for code, uni in code_to_unicode.items()}
    except Exception:
        pass
    return {}


def _encode_cid_hex(text, font_resource):
    """Encode text as hex string for CID font content stream.

    Each character is encoded as a 4-digit hex code. If a /ToUnicode
    CMap is present, uses the reverse mapping. Otherwise falls back to
    Identity-H (code = Unicode codepoint).

    Args:
        text: The text string to encode.
        font_resource: Font dictionary.

    Returns:
        Hex string (e.g. "4E2D65B0") without angle brackets.
    """
    unicode_to_code = _build_unicode_to_code_map(font_resource)
    parts = []
    for ch in text:
        cp = ord(ch)
        code = unicode_to_code.get(cp, cp)
        parts.append(f"{code:04X}")
    return "".join(parts)


def _text_operator(text, font_resource):
    """Build a text show operator string safe for .encode('latin-1').

    CID fonts produce '<hex> Tj', simple fonts produce '(escaped) Tj'
    using WinAnsi encoding (unmappable chars replaced with '?').

    Args:
        text: The text to render.
        font_resource: Font dictionary.

    Returns:
        Complete Tj operator string (e.g. "<4E2D> Tj" or "(hello) Tj").
    """
    if _is_cid_font(font_resource):
        return f"<{_encode_cid_hex(text, font_resource)}> Tj"
    encoded = _fm.encode_for_content_stream(text, font_resource)
    return f"({encoded.decode('latin-1')}) Tj"


# ---------------------------------------------------------------------------
# Rect / BBox helpers
# ---------------------------------------------------------------------------


def _get_rect_dimensions(annot):
    """Extract width and height from annotation /Rect.

    Returns:
        Tuple of (width, height) as floats. Defaults to (0, 0).
    """
    try:
        rect = annot.get("/Rect")
        if rect is not None and len(rect) == 4:
            x1 = float(rect[0])
            y1 = float(rect[1])
            x2 = float(rect[2])
            y2 = float(rect[3])
            return abs(x2 - x1), abs(y2 - y1)
    except Exception:
        pass
    return 0.0, 0.0


def _make_form_stream(pdf, w, h, content, resources=None, matrix=None):
    """Create a Form XObject stream with the given content.

    Args:
        pdf: pikepdf Pdf object.
        w: BBox width.
        h: BBox height.
        content: Content stream bytes.
        resources: Optional resources dictionary.
        matrix: Optional transformation matrix as a 6-element list.

    Returns:
        pikepdf Stream configured as a Form XObject.
    """
    stream = pdf.make_stream(content)
    stream[Name.Type] = Name.XObject
    stream[Name.Subtype] = Name.Form
    stream[Name.BBox] = Array([0, 0, w, h])
    if resources is not None:
        stream[Name.Resources] = resources
    if matrix is not None:
        stream[Name.Matrix] = Array(matrix)
    return stream


def _make_empty_stream(pdf, annot):
    """Create an empty Form XObject (ultimate fallback).

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.

    Returns:
        pikepdf Stream with empty content.
    """
    w, h = _get_rect_dimensions(annot)
    return _make_form_stream(pdf, w, h, b"", Dictionary())


# ---------------------------------------------------------------------------
# Border & background builder
# ---------------------------------------------------------------------------


def _get_border_width(annot):
    """Get border width from /BS or /Border.

    Returns:
        Border width as float.
    """
    try:
        bs = annot.get("/BS")
        if bs is not None:
            bs = _resolve(bs)
            bw = bs.get("/W")
            if bw is not None:
                return float(bw)
    except Exception:
        pass

    try:
        border = annot.get("/Border")
        if border is not None and len(border) >= 3:
            return float(border[2])
    except Exception:
        pass

    return 1.0


def _get_border_style(annot):
    """Get border style from /BS /S.

    Returns:
        One of "S" (solid), "D" (dashed), "B" (beveled),
        "I" (inset), "U" (underline).  Default is "S".
    """
    try:
        bs = annot.get("/BS")
        if bs is not None:
            bs = _resolve(bs)
            s = bs.get("/S")
            if s is not None:
                style = str(s).lstrip("/")
                if style in ("S", "D", "B", "I", "U"):
                    return style
    except Exception:
        pass
    return "S"


def _build_border_background(w, h, annot):
    """Build content stream bytes for border and background.

    Supports border styles: Solid, Dashed, Beveled, Inset, Underline.

    Args:
        w: Widget width.
        h: Widget height.
        annot: Annotation dictionary.

    Returns:
        Content stream bytes for border and background rendering.
    """
    parts = []

    mk = annot.get("/MK")
    if mk is not None:
        try:
            mk = _resolve(mk)
        except Exception:
            mk = None

    border_width = _get_border_width(annot)
    border_style = _get_border_style(annot)

    # Background fill
    bg_ops = ""
    if mk is not None:
        bg = mk.get("/BG")
        if bg is not None:
            bg_ops = _color_array_to_ops(bg, stroke=False)
            if bg_ops:
                parts.append(f"{bg_ops}")
                parts.append(f"0 0 {w:.4g} {h:.4g} re f")

    # Border color
    bc_ops = ""
    if mk is not None:
        bc = mk.get("/BC")
        if bc is not None and border_width > 0:
            bc_ops = _color_array_to_ops(bc, stroke=True)

    if not bc_ops and border_width > 0:
        bc_ops = "0 G"

    if border_width <= 0 or not bc_ops:
        return ("\n".join(parts)).encode("latin-1") if parts else b""

    bw = border_width
    hw = bw / 2.0

    if border_style == "D":
        # Dashed border
        parts.append(f"{bw:.4g} w")
        parts.append("[3] 0 d")
        parts.append(bc_ops)
        parts.append(f"{hw:.4g} {hw:.4g} {w - bw:.4g} {h - bw:.4g} re S")
    elif border_style == "B":
        # Beveled (3D raised effect)
        # Light edges (left + top)
        parts.append("1 g")
        parts.append(
            f"0 0 m {w:.4g} 0 l {w - bw:.4g} {bw:.4g} l "
            f"{bw:.4g} {bw:.4g} l {bw:.4g} {h - bw:.4g} l 0 {h:.4g} l f"
        )
        # Dark edges (right + bottom)
        parts.append("0.5 g")
        parts.append(
            f"{w:.4g} {h:.4g} m {w:.4g} 0 l {w - bw:.4g} {bw:.4g} l "
            f"{w - bw:.4g} {h - bw:.4g} l {bw:.4g} {h - bw:.4g} l "
            f"0 {h:.4g} l f"
        )
    elif border_style == "I":
        # Inset (3D sunken effect) — inverted beveled
        parts.append("0.5 g")
        parts.append(
            f"0 0 m {w:.4g} 0 l {w - bw:.4g} {bw:.4g} l "
            f"{bw:.4g} {bw:.4g} l {bw:.4g} {h - bw:.4g} l 0 {h:.4g} l f"
        )
        parts.append("1 g")
        parts.append(
            f"{w:.4g} {h:.4g} m {w:.4g} 0 l {w - bw:.4g} {bw:.4g} l "
            f"{w - bw:.4g} {h - bw:.4g} l {bw:.4g} {h - bw:.4g} l "
            f"0 {h:.4g} l f"
        )
    elif border_style == "U":
        # Underline — only bottom edge
        parts.append(f"{bw:.4g} w")
        parts.append(bc_ops)
        parts.append(f"0 {hw:.4g} m {w:.4g} {hw:.4g} l S")
    else:
        # Solid (default)
        parts.append(f"{bw:.4g} w")
        parts.append(bc_ops)
        parts.append(f"{hw:.4g} {hw:.4g} {w - bw:.4g} {h - bw:.4g} re S")

    return ("\n".join(parts)).encode("latin-1") if parts else b""


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _get_rotation(annot):
    """Get widget rotation angle from /MK /R.

    Returns:
        Rotation in degrees (0, 90, 180, 270). Default 0.
    """
    try:
        mk = annot.get("/MK")
        if mk is not None:
            mk = _resolve(mk)
            r = mk.get("/R")
            if r is not None:
                angle = int(r) % 360
                if angle in (0, 90, 180, 270):
                    return angle
    except Exception:
        pass
    return 0


def _rotation_matrix(angle, w, h):
    """Compute the /Matrix for a rotated Form XObject.

    Args:
        angle: Rotation angle (0, 90, 180, 270).
        w: Original width.
        h: Original height.

    Returns:
        6-element list for the /Matrix entry, or None for 0.
    """
    if angle == 90:
        return [0, 1, -1, 0, w, 0]
    elif angle == 180:
        return [-1, 0, 0, -1, w, h]
    elif angle == 270:
        return [0, -1, 1, 0, 0, h]
    return None


# ---------------------------------------------------------------------------
# Vertical positioning helper
# ---------------------------------------------------------------------------


def _compute_text_y(field_height, font_size, font_resource, font_name, margin):
    """Compute the baseline y-coordinate for vertically centered text.

    Uses ascent/descent from font metrics for accurate centering.

    Returns:
        Baseline y-position.
    """
    ascent, descent = _fm.get_ascent_descent(font_resource, font_name)
    # descent is negative
    asc_pt = ascent * font_size / 1000.0
    desc_pt = abs(descent) * font_size / 1000.0
    ty = (field_height - asc_pt - desc_pt) / 2.0 + desc_pt
    if ty < margin:
        ty = margin
    return ty


# ---------------------------------------------------------------------------
# Appearance builders by field type
# ---------------------------------------------------------------------------


def _build_border_only_appearance(pdf, annot):
    """Build an appearance stream with border only (no text).

    Used as a fallback when text rendering is not possible.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.

    Returns:
        pikepdf Stream with border-only content.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        return _make_empty_stream(pdf, annot)

    content = _build_border_background(w, h, annot)
    return _make_form_stream(pdf, w, h, content, Dictionary())


def _build_text_field_appearance(pdf, annot, acroform):
    """Build appearance stream for a text field (/FT /Tx).

    Dispatches to comb, multiline, or single-line builders based on
    field flags.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        pikepdf Stream with text field appearance.
    """
    ff = _get_inheritable(annot, "/Ff", acroform)
    flags = int(ff) if ff is not None else 0
    is_multiline = bool(flags & (1 << 12))
    is_comb = bool(flags & (1 << 24))
    max_len = _get_inheritable(annot, "/MaxLen", acroform)

    if is_comb and max_len is not None:
        return _build_comb_field_appearance(pdf, annot, acroform)
    elif is_multiline:
        return _build_multiline_text_appearance(pdf, annot, acroform)
    else:
        return _build_single_line_text_appearance(pdf, annot, acroform)


def _build_single_line_text_appearance(pdf, annot, acroform):
    """Build appearance stream for a single-line text field.

    Uses real font metrics for text width and vertical positioning.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        pikepdf Stream with text field appearance.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        return _make_empty_stream(pdf, annot)

    da = _get_inheritable(annot, "/DA", acroform)
    font_name, font_size, color_ops = _parse_da_string(da)

    if font_name is None:
        return _build_border_only_appearance(pdf, annot)

    font_resource = _resolve_font_resource(font_name, annot, acroform)
    if font_resource is None:
        return _build_border_only_appearance(pdf, annot)

    # Field value
    v = _get_inheritable(annot, "/V", acroform)
    text = _fm.decode_pdf_string(v, font_resource)

    # Alignment
    q = _get_inheritable(annot, "/Q", acroform)
    alignment = int(q) if q is not None else 0

    # Border
    border_width = _get_border_width(annot)
    margin = max(border_width + 1, 2)

    # Rotation
    rotation = _get_rotation(annot)
    if rotation in (90, 270):
        layout_w, layout_h = h, w
    else:
        layout_w, layout_h = w, h

    # Auto-size
    if font_size == 0:
        font_size = _fm.compute_auto_font_size(
            text,
            font_resource,
            field_width=layout_w - 2 * margin,
            field_height=layout_h - 2 * margin,
            font_name=font_name,
        )

    # Text metrics
    text_width = _fm.get_text_width(text, font_resource, font_size, font_name)
    available_width = layout_w - 2 * margin

    if alignment == 1:  # center
        tx = margin + max(0, (available_width - text_width) / 2)
    elif alignment == 2:  # right
        tx = margin + max(0, available_width - text_width)
    else:  # left
        tx = margin

    # Vertical centering using real ascent/descent
    ty = _compute_text_y(layout_h, font_size, font_resource, font_name, margin)

    # Build content
    border_bg = _build_border_background(layout_w, layout_h, annot)

    parts = []
    if border_bg:
        parts.append(border_bg.decode("latin-1"))

    # Clip to widget bounds
    parts.append(
        f"{margin:.4g} {margin:.4g} "
        f"{layout_w - 2 * margin:.4g} {layout_h - 2 * margin:.4g} re W n"
    )

    # Text
    parts.append("BT")
    if color_ops:
        parts.append(color_ops)
    parts.append(f"/{font_name} {font_size:.4g} Tf")
    parts.append(f"{tx:.4g} {ty:.4g} Td")
    parts.append(_text_operator(text, font_resource))
    parts.append("ET")

    content = "\n".join(parts).encode("latin-1")

    # Resources
    font_dict = Dictionary()
    font_dict[Name("/" + font_name)] = font_resource
    resources = Dictionary(Font=font_dict)

    # Rotation
    matrix = _rotation_matrix(rotation, w, h)
    if rotation in (90, 270):
        return _make_form_stream(pdf, h, w, content, resources, matrix)
    elif rotation == 180:
        return _make_form_stream(pdf, w, h, content, resources, matrix)
    return _make_form_stream(pdf, w, h, content, resources)


def _build_multiline_text_appearance(pdf, annot, acroform):
    """Build appearance stream for a multiline text field.

    Handles word-wrapping and auto-size for multiline text.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        pikepdf Stream with multiline text field appearance.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        return _make_empty_stream(pdf, annot)

    da = _get_inheritable(annot, "/DA", acroform)
    font_name, font_size, color_ops = _parse_da_string(da)

    if font_name is None:
        return _build_border_only_appearance(pdf, annot)

    font_resource = _resolve_font_resource(font_name, annot, acroform)
    if font_resource is None:
        return _build_border_only_appearance(pdf, annot)

    v = _get_inheritable(annot, "/V", acroform)
    text = _fm.decode_pdf_string(v, font_resource)

    q = _get_inheritable(annot, "/Q", acroform)
    alignment = int(q) if q is not None else 0

    border_width = _get_border_width(annot)
    margin = max(border_width + 1, 2)
    avail_w = w - 2 * margin
    avail_h = h - 2 * margin

    # Auto-size
    if font_size == 0:
        font_size = _fm.compute_auto_font_size(
            text,
            font_resource,
            field_width=avail_w,
            field_height=avail_h,
            font_name=font_name,
            multiline=True,
        )

    # Line height
    ascent, descent = _fm.get_ascent_descent(font_resource, font_name)
    leading = font_size * 1.2

    # Word-wrap
    lines = _fm._wrap_text(text, font_resource, font_size, avail_w, font_name)

    # Build content
    border_bg = _build_border_background(w, h, annot)

    parts = []
    if border_bg:
        parts.append(border_bg.decode("latin-1"))

    # Clip
    parts.append(f"{margin:.4g} {margin:.4g} {avail_w:.4g} {avail_h:.4g} re W n")

    # Start text at top of field
    top_y = h - margin - ascent * font_size / 1000.0

    parts.append("BT")
    if color_ops:
        parts.append(color_ops)
    parts.append(f"/{font_name} {font_size:.4g} Tf")

    for i, line in enumerate(lines):
        # Alignment
        line_w = _fm.get_text_width(line, font_resource, font_size, font_name)
        if alignment == 1:
            lx = margin + max(0, (avail_w - line_w) / 2)
        elif alignment == 2:
            lx = margin + max(0, avail_w - line_w)
        else:
            lx = margin

        ly = top_y - i * leading

        if i == 0:
            parts.append(f"{lx:.4g} {ly:.4g} Td")
        else:
            # Relative move from previous line position
            prev_lx_w = _fm.get_text_width(
                lines[i - 1], font_resource, font_size, font_name
            )
            if alignment == 1:
                prev_lx = margin + max(0, (avail_w - prev_lx_w) / 2)
            elif alignment == 2:
                prev_lx = margin + max(0, avail_w - prev_lx_w)
            else:
                prev_lx = margin
            parts.append(f"{lx - prev_lx:.4g} {-leading:.4g} Td")
        parts.append(_text_operator(line, font_resource))

    parts.append("ET")

    content = "\n".join(parts).encode("latin-1")

    font_dict = Dictionary()
    font_dict[Name("/" + font_name)] = font_resource
    resources = Dictionary(Font=font_dict)

    return _make_form_stream(pdf, w, h, content, resources)


def _build_comb_field_appearance(pdf, annot, acroform):
    """Build appearance stream for a comb text field.

    Each character is evenly spaced within cells defined by /MaxLen.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        pikepdf Stream with comb field appearance.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        return _make_empty_stream(pdf, annot)

    da = _get_inheritable(annot, "/DA", acroform)
    font_name, font_size, color_ops = _parse_da_string(da)

    if font_name is None:
        return _build_border_only_appearance(pdf, annot)

    font_resource = _resolve_font_resource(font_name, annot, acroform)
    if font_resource is None:
        return _build_border_only_appearance(pdf, annot)

    max_len_val = _get_inheritable(annot, "/MaxLen", acroform)
    max_len = int(max_len_val) if max_len_val is not None else 1
    if max_len < 1:
        max_len = 1

    v = _get_inheritable(annot, "/V", acroform)
    text = _fm.decode_pdf_string(v, font_resource)
    # Truncate to MaxLen
    text = text[:max_len]

    border_width = _get_border_width(annot)
    cell_width = w / max_len

    # Auto-size: fit within one cell height
    if font_size == 0:
        font_size = _fm.compute_auto_font_size(
            "M",
            font_resource,
            field_width=cell_width - 2,
            field_height=h - 2 * max(border_width + 1, 2),
            font_name=font_name,
        )

    # Build content
    border_bg = _build_border_background(w, h, annot)

    parts = []
    if border_bg:
        parts.append(border_bg.decode("latin-1"))

    # Draw vertical divider lines between cells
    if border_width > 0:
        parts.append("0.5 G")
        parts.append(f"{max(0.5, border_width * 0.5):.4g} w")
        for i in range(1, max_len):
            x = i * cell_width
            parts.append(f"{x:.4g} 0 m {x:.4g} {h:.4g} l S")

    # Vertical baseline
    margin = max(border_width + 1, 2)
    ty = _compute_text_y(h, font_size, font_resource, font_name, margin)

    # Characters
    parts.append("BT")
    if color_ops:
        parts.append(color_ops)
    parts.append(f"/{font_name} {font_size:.4g} Tf")

    prev_x = 0.0
    for i, ch in enumerate(text):
        char_w = _fm.get_text_width(ch, font_resource, font_size, font_name)
        x = i * cell_width + (cell_width - char_w) / 2.0
        if i == 0:
            parts.append(f"{x:.4g} {ty:.4g} Td")
        else:
            parts.append(f"{x - prev_x:.4g} 0 Td")
        parts.append(_text_operator(ch, font_resource))
        prev_x = x

    parts.append("ET")

    content = "\n".join(parts).encode("latin-1")

    font_dict = Dictionary()
    font_dict[Name("/" + font_name)] = font_resource
    resources = Dictionary(Font=font_dict)

    return _make_form_stream(pdf, w, h, content, resources)


def _build_button_appearance(pdf, annot, acroform):
    """Build appearance stream for a button field (/FT /Btn).

    Handles checkboxes, radio buttons, and pushbuttons.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        For checkbox/radio: Dictionary mapping state names to Streams.
        For pushbutton: single Stream.
    """
    ff = _get_inheritable(annot, "/Ff", acroform)
    flags = int(ff) if ff is not None else 0

    is_pushbutton = bool(flags & (1 << 16))
    is_radio = bool(flags & (1 << 15))

    if is_pushbutton:
        return _build_pushbutton_appearance(pdf, annot, acroform)
    elif is_radio:
        return _build_radio_appearance(pdf, annot)
    else:
        return _build_checkbox_appearance(pdf, annot)


def _build_checkbox_appearance(pdf, annot):
    """Build appearance state dictionary for a checkbox.

    Creates Off state (empty box) and On state (box with checkmark).

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.

    Returns:
        Dictionary with state name keys mapping to Stream values.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        w = max(w, 12)
        h = max(h, 12)

    on_state_name = _get_on_state_name(annot)

    mk = annot.get("/MK")
    if mk is not None:
        try:
            mk = _resolve(mk)
        except Exception:
            mk = None

    bc_ops = ""
    if mk is not None:
        bc = mk.get("/BC")
        if bc is not None:
            bc_ops = _color_array_to_ops(bc, stroke=True)
    if not bc_ops:
        bc_ops = "0 G"

    bg_ops = ""
    if mk is not None:
        bg = mk.get("/BG")
        if bg is not None:
            bg_ops = _color_array_to_ops(bg, stroke=False)

    border_width = _get_border_width(annot)

    # Off state: empty box
    off_parts = []
    if bg_ops:
        off_parts.append(bg_ops)
        off_parts.append(f"0 0 {w:.4g} {h:.4g} re f")
    off_parts.append(f"{border_width:.4g} w")
    off_parts.append(bc_ops)
    hw = border_width / 2.0
    off_parts.append(
        f"{hw:.4g} {hw:.4g} {w - border_width:.4g} {h - border_width:.4g} re S"
    )
    off_content = "\n".join(off_parts).encode("latin-1")
    off_stream = _make_form_stream(pdf, w, h, off_content, Dictionary())

    # On state: box with checkmark (lines)
    on_parts = []
    if bg_ops:
        on_parts.append(bg_ops)
        on_parts.append(f"0 0 {w:.4g} {h:.4g} re f")
    on_parts.append(f"{border_width:.4g} w")
    on_parts.append(bc_ops)
    on_parts.append(
        f"{hw:.4g} {hw:.4g} {w - border_width:.4g} {h - border_width:.4g} re S"
    )
    margin = max(border_width + 1, 3)
    on_parts.append("0 G")
    on_parts.append(f"{max(1, border_width):.4g} w")
    on_parts.append(
        f"{margin:.4g} {h * 0.5:.4g} m "
        f"{w * 0.4:.4g} {margin:.4g} l "
        f"{w - margin:.4g} {h - margin:.4g} l S"
    )
    on_content = "\n".join(on_parts).encode("latin-1")
    on_stream = _make_form_stream(pdf, w, h, on_content, Dictionary())

    result = Dictionary()
    result[Name("/Off")] = off_stream
    result[Name("/" + on_state_name)] = on_stream
    return result


def _build_radio_appearance(pdf, annot):
    """Build appearance state dictionary for a radio button.

    Creates Off state (empty circle) and On state (circle with dot).

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.

    Returns:
        Dictionary with state name keys mapping to Stream values.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        w = max(w, 12)
        h = max(h, 12)

    on_state_name = _get_on_state_name(annot)

    mk = annot.get("/MK")
    if mk is not None:
        try:
            mk = _resolve(mk)
        except Exception:
            mk = None

    bc_ops = ""
    if mk is not None:
        bc = mk.get("/BC")
        if bc is not None:
            bc_ops = _color_array_to_ops(bc, stroke=True)
    if not bc_ops:
        bc_ops = "0 G"

    bg_ops = ""
    if mk is not None:
        bg = mk.get("/BG")
        if bg is not None:
            bg_ops = _color_array_to_ops(bg, stroke=False)

    border_width = _get_border_width(annot)

    cx = w / 2.0
    cy = h / 2.0
    r = min(cx, cy) - border_width

    if r < 1:
        r = min(cx, cy)
    if r < 0.5:
        return _build_checkbox_appearance(pdf, annot)

    # Off state: empty circle
    off_circle = _circle_path(cx, cy, r)
    off_parts = []
    if bg_ops:
        off_parts.append(bg_ops)
        off_parts.append(off_circle + " f")
    off_parts.append(f"{border_width:.4g} w")
    off_parts.append(bc_ops)
    off_parts.append(off_circle + " S")
    off_content = "\n".join(off_parts).encode("latin-1")
    off_stream = _make_form_stream(pdf, w, h, off_content, Dictionary())

    # On state: circle with filled dot
    dot_r = r * 0.4
    dot_circle = _circle_path(cx, cy, dot_r)
    on_parts = []
    if bg_ops:
        on_parts.append(bg_ops)
        on_parts.append(off_circle + " f")
    on_parts.append(f"{border_width:.4g} w")
    on_parts.append(bc_ops)
    on_parts.append(off_circle + " S")
    on_parts.append("0 g")
    on_parts.append(dot_circle + " f")
    on_content = "\n".join(on_parts).encode("latin-1")
    on_stream = _make_form_stream(pdf, w, h, on_content, Dictionary())

    result = Dictionary()
    result[Name("/Off")] = off_stream
    result[Name("/" + on_state_name)] = on_stream
    return result


def _build_pushbutton_appearance(pdf, annot, acroform):
    """Build appearance stream for a pushbutton.

    Renders border/background with centered caption from /MK /CA.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        pikepdf Stream.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        return _make_empty_stream(pdf, annot)

    border_bg = _build_border_background(w, h, annot)

    # Get caption from /MK /CA
    caption = ""
    mk = annot.get("/MK")
    if mk is not None:
        try:
            mk = _resolve(mk)
            ca = mk.get("/CA")
            if ca is not None:
                caption = str(ca)
        except Exception:
            pass

    if not caption:
        return _make_form_stream(pdf, w, h, border_bg, Dictionary())

    da = _get_inheritable(annot, "/DA", acroform)
    font_name, font_size, color_ops = _parse_da_string(da)

    if font_name is None:
        return _make_form_stream(pdf, w, h, border_bg, Dictionary())

    font_resource = _resolve_font_resource(font_name, annot, acroform)
    if font_resource is None:
        return _make_form_stream(pdf, w, h, border_bg, Dictionary())

    # Auto-size for pushbutton captions
    border_width = _get_border_width(annot)
    margin = max(border_width + 1, 2)
    if font_size == 0:
        font_size = _fm.compute_auto_font_size(
            caption,
            font_resource,
            field_width=w - 2 * margin,
            field_height=h - 2 * margin,
            font_name=font_name,
        )

    text_width = _fm.get_text_width(caption, font_resource, font_size, font_name)
    tx = (w - text_width) / 2.0
    ty = _compute_text_y(h, font_size, font_resource, font_name, margin)

    parts = []
    if border_bg:
        parts.append(border_bg.decode("latin-1"))
    parts.append("BT")
    if color_ops:
        parts.append(color_ops)
    parts.append(f"/{font_name} {font_size:.4g} Tf")
    parts.append(f"{tx:.4g} {ty:.4g} Td")
    parts.append(_text_operator(caption, font_resource))
    parts.append("ET")

    content = "\n".join(parts).encode("latin-1")
    font_dict = Dictionary()
    font_dict[Name("/" + font_name)] = font_resource
    resources = Dictionary(Font=font_dict)
    return _make_form_stream(pdf, w, h, content, resources)


def _build_signature_appearance(pdf, annot):
    """Build appearance stream for a signature field (/FT /Sig).

    Signature fields are special: unsigned fields need a visible placeholder,
    and signed fields should already have an appearance set by the signing
    software. When called (i.e. no /AP /N exists), we produce a border-only
    appearance so the field area remains visually identifiable.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.

    Returns:
        pikepdf Stream with border-only content.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        return _make_empty_stream(pdf, annot)

    return _build_border_only_appearance(pdf, annot)


def _build_choice_field_appearance(pdf, annot, acroform):
    """Build appearance stream for a choice field (/FT /Ch).

    Dispatches to listbox or combo box (text field) based on flags.

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        pikepdf Stream.
    """
    ff = _get_inheritable(annot, "/Ff", acroform)
    flags = int(ff) if ff is not None else 0
    is_combo = bool(flags & (1 << 17))

    if is_combo:
        return _build_single_line_text_appearance(pdf, annot, acroform)
    else:
        return _build_listbox_appearance(pdf, annot, acroform)


def _build_listbox_appearance(pdf, annot, acroform):
    """Build appearance stream for a listbox (/Ch without Combo flag).

    Shows visible options with highlight on selected item(s).

    Args:
        pdf: pikepdf Pdf object.
        annot: Annotation dictionary.
        acroform: /AcroForm dictionary.

    Returns:
        pikepdf Stream.
    """
    w, h = _get_rect_dimensions(annot)
    if w <= 0 or h <= 0:
        return _make_empty_stream(pdf, annot)

    da = _get_inheritable(annot, "/DA", acroform)
    font_name, font_size, color_ops = _parse_da_string(da)

    if font_name is None:
        return _build_border_only_appearance(pdf, annot)

    font_resource = _resolve_font_resource(font_name, annot, acroform)
    if font_resource is None:
        return _build_border_only_appearance(pdf, annot)

    border_width = _get_border_width(annot)
    margin = max(border_width + 1, 2)

    # Default font size for listbox if auto-size
    if font_size == 0:
        font_size = 12.0

    # Line height
    leading = font_size * 1.2

    # Options
    opt = _get_inheritable(annot, "/Opt", acroform)
    options = []
    if opt is not None:
        try:
            for item in opt:
                item = _resolve(item)
                if isinstance(item, Array):
                    # [export_value, display_value] pair
                    try:
                        display = str(_resolve(item[1]))
                    except (IndexError, Exception):
                        display = str(item)
                else:
                    display = str(item)
                options.append(display)
        except Exception:
            pass

    # Selected value(s)
    v = _get_inheritable(annot, "/V", acroform)
    selected_values = set()
    if v is not None:
        try:
            if isinstance(v, Array):
                for sv in v:
                    selected_values.add(str(_resolve(sv)))
            else:
                selected_values.add(str(v))
        except Exception:
            selected_values.add(str(v))

    # Top index (scroll offset)
    ti = _get_inheritable(annot, "/TI", acroform)
    top_index = int(ti) if ti is not None else 0
    top_index = max(0, min(top_index, max(0, len(options) - 1)))

    # Visible lines
    avail_h = h - 2 * margin
    visible_count = max(1, int(avail_h / leading))

    # Build content
    border_bg = _build_border_background(w, h, annot)

    parts = []
    if border_bg:
        parts.append(border_bg.decode("latin-1"))

    # Clip
    parts.append(f"{margin:.4g} {margin:.4g} {w - 2 * margin:.4g} {avail_h:.4g} re W n")

    # Draw highlight backgrounds for selected items
    for i in range(visible_count):
        idx = top_index + i
        if idx >= len(options):
            break
        if options[idx] in selected_values:
            row_y = h - margin - (i + 1) * leading
            parts.append("0 0 0.6 rg")
            parts.append(
                f"{margin:.4g} {row_y:.4g} {w - 2 * margin:.4g} {leading:.4g} re f"
            )

    # Draw option text
    ascent, _ = _fm.get_ascent_descent(font_resource, font_name)
    parts.append("BT")
    if color_ops:
        parts.append(color_ops)
    parts.append(f"/{font_name} {font_size:.4g} Tf")

    for i in range(visible_count):
        idx = top_index + i
        if idx >= len(options):
            break
        opt_text = options[idx]
        is_selected = opt_text in selected_values
        lx = margin + 1
        ly = h - margin - i * leading - ascent * font_size / 1000.0

        # White text on blue highlight
        if is_selected:
            parts.append("1 g")

        if i == 0 and idx == top_index:
            parts.append(f"{lx:.4g} {ly:.4g} Td")
        else:
            parts.append(f"0 {-leading:.4g} Td")

        parts.append(_text_operator(opt_text, font_resource))

        # Restore color
        if is_selected:
            if color_ops:
                parts.append(color_ops)
            else:
                parts.append("0 g")

    parts.append("ET")

    content = "\n".join(parts).encode("latin-1")

    font_dict = Dictionary()
    font_dict[Name("/" + font_name)] = font_resource
    resources = Dictionary(Font=font_dict)

    return _make_form_stream(pdf, w, h, content, resources)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Bezier control point factor for circle approximation
_KAPPA = 4.0 * (math.sqrt(2) - 1) / 3.0


def _circle_path(cx, cy, r):
    """Generate PDF path operators for a circle using Bezier curves.

    Args:
        cx: Center x coordinate.
        cy: Center y coordinate.
        r: Radius.

    Returns:
        String of PDF path operators (m, c) without a painting operator.
    """
    k = _KAPPA * r
    parts = [
        f"{cx + r:.4g} {cy:.4g} m",
        f"{cx + r:.4g} {cy + k:.4g} {cx + k:.4g} {cy + r:.4g} {cx:.4g} {cy + r:.4g} c",
        f"{cx - k:.4g} {cy + r:.4g} {cx - r:.4g} {cy + k:.4g} {cx - r:.4g} {cy:.4g} c",
        f"{cx - r:.4g} {cy - k:.4g} {cx - k:.4g} {cy - r:.4g} {cx:.4g} {cy - r:.4g} c",
        f"{cx + k:.4g} {cy - r:.4g} {cx + r:.4g} {cy - k:.4g} {cx + r:.4g} {cy:.4g} c",
    ]
    return "\n".join(parts)


def _get_on_state_name(annot):
    """Determine the "on" state name for a checkbox or radio button.

    Looks at existing /AP /N keys (any key other than "Off"),
    then falls back to /AS, then defaults to "Yes".

    Args:
        annot: Annotation dictionary.

    Returns:
        The on-state name string (without leading /).
    """
    try:
        ap = annot.get("/AP")
        if ap is not None:
            ap = _resolve(ap)
            n = ap.get("/N")
            if n is not None:
                n = _resolve(n)
                if isinstance(n, Dictionary):
                    for key in n.keys():
                        key_str = str(key)
                        if key_str != "/Off":
                            return key_str.lstrip("/")
    except Exception:
        pass

    try:
        as_val = annot.get("/AS")
        if as_val is not None:
            as_str = str(as_val)
            if as_str != "/Off":
                return as_str.lstrip("/")
    except Exception:
        pass

    return "Yes"
