# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Page box sanitizer for PDF/A compliance.

Validates and fixes MediaBox, CropBox, TrimBox, BleedBox, and ArtBox
on every page to satisfy ISO 32000-1 and ISO 19005-2/3 requirements.
"""

import logging

from pikepdf import Array, Name, Pdf

logger = logging.getLogger(__name__)

_SUB_BOX_NAMES = (Name.CropBox, Name.BleedBox, Name.TrimBox, Name.ArtBox)

# Tolerance for floating-point coordinate comparison
_EPSILON = 1e-6
_MIN_PAGE_BOUNDARY_SIZE = 3.0
_MAX_PAGE_BOUNDARY_SIZE = 14_400.0


def _resolve_mediabox_from_parent(page_dict: dict) -> Array | None:
    """Walk the /Parent chain to find an inherited /MediaBox.

    Includes cycle detection to avoid infinite loops.

    Args:
        page_dict: The page dictionary (pikepdf Dictionary).

    Returns:
        The inherited MediaBox Array, or None if not found.
    """
    visited: set[tuple[int, int]] = set()
    node = page_dict
    while True:
        try:
            objgen = node.objgen
        except Exception:
            objgen = (0, 0)
        if objgen != (0, 0):
            if objgen in visited:
                return None
            visited.add(objgen)
        try:
            parent = node.get(Name.Parent)
        except Exception:
            return None
        if parent is None:
            return None
        try:
            parent = parent.get_object()
        except Exception:
            pass  # already resolved
        try:
            mb = parent.get(Name.MediaBox)
        except Exception:
            mb = None
        if mb is not None:
            return mb
        node = parent


def _is_valid_box(box: object) -> bool:
    """Check whether *box* is an Array of exactly 4 numeric values.

    Args:
        box: A candidate page box object.

    Returns:
        True if valid, False otherwise.
    """
    try:
        if not isinstance(box, Array):
            return False
        if len(box) != 4:
            return False
        for val in box:
            float(val)
        return True
    except Exception:
        return False


def _normalize_box(box: Array) -> tuple[float, float, float, float]:
    """Extract coordinates and ensure llx <= urx, lly <= ury.

    Args:
        box: A valid 4-element Array.

    Returns:
        (llx, lly, urx, ury) with swapped coordinates if needed.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    llx = min(x1, x2)
    lly = min(y1, y2)
    urx = max(x1, x2)
    ury = max(y1, y2)
    return (llx, lly, urx, ury)


def _clip_to_mediabox(
    box_coords: tuple[float, float, float, float],
    media_coords: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Clip *box_coords* so they do not exceed *media_coords*.

    Args:
        box_coords: (llx, lly, urx, ury) of the sub-box.
        media_coords: (llx, lly, urx, ury) of the MediaBox.

    Returns:
        Clipped (llx, lly, urx, ury).
    """
    llx = max(box_coords[0], media_coords[0])
    lly = max(box_coords[1], media_coords[1])
    urx = min(box_coords[2], media_coords[2])
    ury = min(box_coords[3], media_coords[3])
    return (llx, lly, urx, ury)


def _coords_equal(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Compare two coordinate tuples with floating-point tolerance.

    Args:
        a: First coordinate tuple.
        b: Second coordinate tuple.

    Returns:
        True if all components are within _EPSILON.
    """
    return all(abs(av - bv) < _EPSILON for av, bv in zip(a, b))


def _dimensions_in_range(coords: tuple[float, float, float, float]) -> bool:
    """Return True when width and height satisfy ISO 32000 boundary limits."""
    width = coords[2] - coords[0]
    height = coords[3] - coords[1]
    return (
        _MIN_PAGE_BOUNDARY_SIZE <= width <= _MAX_PAGE_BOUNDARY_SIZE
        and _MIN_PAGE_BOUNDARY_SIZE <= height <= _MAX_PAGE_BOUNDARY_SIZE
    )


def _enforce_dimension_limits(
    coords: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], bool]:
    """Clamp width/height to ISO 32000 page-boundary size limits."""
    llx, lly, urx, ury = coords
    width = urx - llx
    height = ury - lly
    clamped_width = min(
        max(width, _MIN_PAGE_BOUNDARY_SIZE),
        _MAX_PAGE_BOUNDARY_SIZE,
    )
    clamped_height = min(
        max(height, _MIN_PAGE_BOUNDARY_SIZE),
        _MAX_PAGE_BOUNDARY_SIZE,
    )
    if (
        abs(clamped_width - width) < _EPSILON
        and abs(clamped_height - height) < _EPSILON
    ):
        return coords, False
    return (llx, lly, llx + clamped_width, lly + clamped_height), True


def sanitize_page_boxes(pdf: Pdf) -> dict[str, int]:
    """Validate and fix page boxes for PDF/A compliance.

    For each page the following steps are performed:
    1. Ensure MediaBox is present (resolve from /Parent chain if needed).
    2. Validate box format (array of exactly 4 numeric values).
    3. Normalize coordinates (swap if inverted) and enforce 3..14400 size limits.
    4. Clip sub-boxes (CropBox/BleedBox/TrimBox/ArtBox) to MediaBox.
    5. Enforce sub-box 3..14400 size limits.
    6. Ensure TrimBox or ArtBox is present.

    Args:
        pdf: An open pikepdf Pdf object (modified in place).

    Returns:
        Dictionary with counts of modifications performed.
    """
    result = {
        "mediabox_inherited": 0,
        "boxes_normalized": 0,
        "boxes_clipped": 0,
        "trimbox_added": 0,
        "malformed_boxes_removed": 0,
    }

    for page_idx, page in enumerate(pdf.pages):
        page_dict = page.obj

        # Step 1: Ensure MediaBox
        try:
            mediabox = page_dict.get(Name.MediaBox)
        except Exception:
            mediabox = None

        if mediabox is None:
            inherited = _resolve_mediabox_from_parent(page_dict)
            if inherited is None:
                logger.warning("Page %d: No MediaBox found, skipping", page_idx + 1)
                continue
            page_dict[Name.MediaBox] = inherited
            mediabox = inherited
            result["mediabox_inherited"] += 1
            logger.debug("Page %d: MediaBox inherited from parent", page_idx + 1)

        # Step 2: Validate MediaBox format
        if not _is_valid_box(mediabox):
            logger.warning("Page %d: Malformed MediaBox, skipping page", page_idx + 1)
            continue

        # Step 2b: Validate sub-box formats, remove malformed ones
        for box_name in _SUB_BOX_NAMES:
            try:
                box = page_dict.get(box_name)
            except Exception:
                box = None
            if box is not None and not _is_valid_box(box):
                del page_dict[box_name]
                result["malformed_boxes_removed"] += 1
                logger.debug("Page %d: Removed malformed %s", page_idx + 1, box_name)

        # Step 3: Normalize MediaBox coordinates
        media_coords = _normalize_box(mediabox)
        orig_media = tuple(float(v) for v in mediabox)
        media_changed = (
            orig_media[0],
            orig_media[1],
            orig_media[2],
            orig_media[3],
        ) != media_coords
        if media_changed:
            page_dict[Name.MediaBox] = Array(
                [media_coords[0], media_coords[1], media_coords[2], media_coords[3]]
            )
            result["boxes_normalized"] += 1
            logger.debug("Page %d: MediaBox coordinates normalized", page_idx + 1)

        # Step 3c: Enforce ISO boundary size limits on MediaBox.
        media_limited, media_size_changed = _enforce_dimension_limits(media_coords)
        if media_size_changed:
            media_coords = media_limited
            page_dict[Name.MediaBox] = Array(
                [media_coords[0], media_coords[1], media_coords[2], media_coords[3]]
            )
            result["boxes_normalized"] += 1
            logger.debug("Page %d: MediaBox size clamped to ISO limits", page_idx + 1)

        # Step 3b: Normalize sub-box coordinates
        for box_name in _SUB_BOX_NAMES:
            try:
                box = page_dict.get(box_name)
            except Exception:
                box = None
            if box is None:
                continue
            coords = _normalize_box(box)
            orig = tuple(float(v) for v in box)
            if (orig[0], orig[1], orig[2], orig[3]) != coords:
                page_dict[box_name] = Array(
                    [coords[0], coords[1], coords[2], coords[3]]
                )
                result["boxes_normalized"] += 1
                logger.debug(
                    "Page %d: %s coordinates normalized", page_idx + 1, box_name
                )

        # Step 4 + Step 5: Clip sub-boxes to MediaBox and enforce size limits.
        for box_name in _SUB_BOX_NAMES:
            try:
                box = page_dict.get(box_name)
            except Exception:
                box = None
            if box is None:
                continue
            coords = _normalize_box(box)
            adjusted = _clip_to_mediabox(coords, media_coords)

            # If clipped area is zero or negative, remove the box
            if adjusted[2] <= adjusted[0] or adjusted[3] <= adjusted[1]:
                del page_dict[box_name]
                result["boxes_clipped"] += 1
                logger.debug(
                    "Page %d: %s removed (outside MediaBox)",
                    page_idx + 1,
                    box_name,
                )
                continue

            # If still outside page-boundary size limits, try to expand/shrink.
            if not _dimensions_in_range(adjusted):
                limited, _ = _enforce_dimension_limits(adjusted)
                limited = _clip_to_mediabox(limited, media_coords)
                if not _dimensions_in_range(limited):
                    # If we cannot create a valid size inside MediaBox, fall back
                    # to MediaBox (already guaranteed valid at this point).
                    adjusted = media_coords
                else:
                    adjusted = limited

            if not _coords_equal(coords, adjusted):
                page_dict[box_name] = Array(
                    [adjusted[0], adjusted[1], adjusted[2], adjusted[3]]
                )
                result["boxes_clipped"] += 1
                logger.debug(
                    "Page %d: %s adjusted to valid MediaBox/size limits",
                    page_idx + 1,
                    box_name,
                )

        # Step 6: Ensure TrimBox or ArtBox
        try:
            has_trimbox = page_dict.get(Name.TrimBox) is not None
        except Exception:
            has_trimbox = False
        try:
            has_artbox = page_dict.get(Name.ArtBox) is not None
        except Exception:
            has_artbox = False

        if not has_trimbox and not has_artbox:
            # Use CropBox if present, otherwise MediaBox
            try:
                cropbox = page_dict.get(Name.CropBox)
            except Exception:
                cropbox = None
            if cropbox is not None and _is_valid_box(cropbox):
                page_dict[Name.TrimBox] = cropbox
            else:
                page_dict[Name.TrimBox] = page_dict[Name.MediaBox]
            result["trimbox_added"] += 1
            logger.debug("Page %d: TrimBox added", page_idx + 1)

    total = sum(result.values())
    if total > 0:
        logger.info(
            "Page boxes sanitized: %d inherited, %d normalized, "
            "%d clipped, %d TrimBox added, %d malformed removed",
            result["mediabox_inherited"],
            result["boxes_normalized"],
            result["boxes_clipped"],
            result["trimbox_added"],
            result["malformed_boxes_removed"],
        )

    return result
