# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for page box sanitization (MediaBox/CropBox/TrimBox/BleedBox/ArtBox)."""

from collections.abc import Generator

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.sanitizers.page_boxes import (
    _clip_to_mediabox,
    _coords_equal,
    _is_valid_box,
    _normalize_box,
    _resolve_mediabox_from_parent,
    sanitize_page_boxes,
)


def _make_page(pdf: Pdf, **boxes: Array) -> None:
    """Add a page to *pdf* with the given box entries.

    Args:
        pdf: An open pikepdf Pdf.
        **boxes: Keyword arguments mapping box names to Array values.
            Accepted keys: MediaBox, CropBox, TrimBox, BleedBox, ArtBox.
    """
    page_dict = Dictionary(Type=Name.Page)
    for key, val in boxes.items():
        page_dict[Name(f"/{key}")] = val
    page = pikepdf.Page(page_dict)
    pdf.pages.append(page)


# ---------------------------------------------------------------------------
# TestMediaBoxInheritance
# ---------------------------------------------------------------------------


class TestMediaBoxInheritance:
    """Tests for MediaBox resolution from /Parent chain."""

    @pytest.fixture
    def pdf_with_direct_mediabox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(pdf, MediaBox=Array([0, 0, 612, 792]))
        yield pdf

    @pytest.fixture
    def pdf_with_inherited_mediabox(self) -> Generator[Pdf, None, None]:
        """PDF where the page has no direct MediaBox but /Parent does."""
        pdf = new_pdf()
        # new_pdf() gives us a page tree whose root has no MediaBox.
        # We add a page without MediaBox first, then set MediaBox on the parent.
        page_dict = Dictionary(Type=Name.Page)
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        # The page tree root is /Root/Pages — set MediaBox there.
        pdf.Root.Pages[Name.MediaBox] = Array([0, 0, 595, 842])
        yield pdf

    @pytest.fixture
    def pdf_with_no_mediabox(self) -> Generator[Pdf, None, None]:
        """PDF where no MediaBox exists anywhere in the page tree."""
        pdf = new_pdf()
        page_dict = Dictionary(Type=Name.Page)
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)
        yield pdf

    def test_direct_mediabox_kept(self, pdf_with_direct_mediabox: Pdf):
        """A direct MediaBox is left unchanged."""
        result = sanitize_page_boxes(pdf_with_direct_mediabox)
        assert result["mediabox_inherited"] == 0
        page = pdf_with_direct_mediabox.pages[0]
        mb = page.obj[Name.MediaBox]
        assert [float(v) for v in mb] == [0, 0, 612, 792]

    def test_inherited_mediabox_materialized(self, pdf_with_inherited_mediabox: Pdf):
        """MediaBox inherited from /Parent is copied to the page."""
        result = sanitize_page_boxes(pdf_with_inherited_mediabox)
        assert result["mediabox_inherited"] == 1
        page = pdf_with_inherited_mediabox.pages[0]
        mb = page.obj[Name.MediaBox]
        assert [float(v) for v in mb] == [0, 0, 595, 842]

    def test_missing_mediabox_skips_page(self, pdf_with_no_mediabox: Pdf):
        """Page without any MediaBox is skipped (no crash)."""
        result = sanitize_page_boxes(pdf_with_no_mediabox)
        assert result["mediabox_inherited"] == 0
        # TrimBox should NOT be added since the page was skipped
        assert result["trimbox_added"] == 0


# ---------------------------------------------------------------------------
# TestBoxFormatValidation
# ---------------------------------------------------------------------------


class TestBoxFormatValidation:
    """Tests for box format validation (4 numeric values)."""

    @pytest.fixture
    def pdf_with_valid_boxes(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([10, 10, 600, 780]),
        )
        yield pdf

    @pytest.fixture
    def pdf_with_3_element_cropbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([10, 10, 600]),
        )
        yield pdf

    @pytest.fixture
    def pdf_with_5_element_bleedbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            BleedBox=Array([0, 0, 612, 792, 99]),
        )
        yield pdf

    @pytest.fixture
    def pdf_with_nonnumeric_artbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            ArtBox=Array([0, 0, Name.Foo, 792]),
        )
        yield pdf

    @pytest.fixture
    def pdf_with_malformed_mediabox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612]),
        )
        yield pdf

    def test_valid_boxes_unchanged(self, pdf_with_valid_boxes: Pdf):
        result = sanitize_page_boxes(pdf_with_valid_boxes)
        assert result["malformed_boxes_removed"] == 0

    def test_3_element_cropbox_removed(self, pdf_with_3_element_cropbox: Pdf):
        result = sanitize_page_boxes(pdf_with_3_element_cropbox)
        assert result["malformed_boxes_removed"] == 1
        page = pdf_with_3_element_cropbox.pages[0]
        assert Name.CropBox not in page.obj

    def test_5_element_bleedbox_removed(self, pdf_with_5_element_bleedbox: Pdf):
        result = sanitize_page_boxes(pdf_with_5_element_bleedbox)
        assert result["malformed_boxes_removed"] == 1
        page = pdf_with_5_element_bleedbox.pages[0]
        assert Name.BleedBox not in page.obj

    def test_nonnumeric_artbox_removed(self, pdf_with_nonnumeric_artbox: Pdf):
        result = sanitize_page_boxes(pdf_with_nonnumeric_artbox)
        assert result["malformed_boxes_removed"] == 1
        page = pdf_with_nonnumeric_artbox.pages[0]
        assert Name.ArtBox not in page.obj

    def test_malformed_mediabox_skips_page(self, pdf_with_malformed_mediabox: Pdf):
        result = sanitize_page_boxes(pdf_with_malformed_mediabox)
        # Page is skipped entirely — no trimbox added
        assert result["trimbox_added"] == 0


# ---------------------------------------------------------------------------
# TestCoordinateNormalization
# ---------------------------------------------------------------------------


class TestCoordinateNormalization:
    """Tests for coordinate normalization (swapping inverted coords)."""

    @pytest.fixture
    def pdf_inverted_x(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(pdf, MediaBox=Array([612, 0, 0, 792]))
        yield pdf

    @pytest.fixture
    def pdf_inverted_y(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(pdf, MediaBox=Array([0, 792, 612, 0]))
        yield pdf

    @pytest.fixture
    def pdf_inverted_both(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(pdf, MediaBox=Array([612, 792, 0, 0]))
        yield pdf

    @pytest.fixture
    def pdf_already_normal(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(pdf, MediaBox=Array([0, 0, 612, 792]))
        yield pdf

    def test_inverted_x_normalized(self, pdf_inverted_x: Pdf):
        result = sanitize_page_boxes(pdf_inverted_x)
        assert result["boxes_normalized"] >= 1
        mb = pdf_inverted_x.pages[0].obj[Name.MediaBox]
        assert [float(v) for v in mb] == [0, 0, 612, 792]

    def test_inverted_y_normalized(self, pdf_inverted_y: Pdf):
        result = sanitize_page_boxes(pdf_inverted_y)
        assert result["boxes_normalized"] >= 1
        mb = pdf_inverted_y.pages[0].obj[Name.MediaBox]
        assert [float(v) for v in mb] == [0, 0, 612, 792]

    def test_inverted_both_normalized(self, pdf_inverted_both: Pdf):
        result = sanitize_page_boxes(pdf_inverted_both)
        assert result["boxes_normalized"] >= 1
        mb = pdf_inverted_both.pages[0].obj[Name.MediaBox]
        assert [float(v) for v in mb] == [0, 0, 612, 792]

    def test_already_normal_not_counted(self, pdf_already_normal: Pdf):
        result = sanitize_page_boxes(pdf_already_normal)
        assert result["boxes_normalized"] == 0


# ---------------------------------------------------------------------------
# TestClipToMediaBox
# ---------------------------------------------------------------------------


class TestClipToMediaBox:
    """Tests for clipping sub-boxes to MediaBox."""

    @pytest.fixture
    def pdf_cropbox_exceeds(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([-10, -10, 620, 800]),
        )
        yield pdf

    @pytest.fixture
    def pdf_all_subboxes_exceed(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([-5, 0, 620, 792]),
            BleedBox=Array([0, -5, 612, 800]),
            TrimBox=Array([-1, -1, 613, 793]),
        )
        yield pdf

    @pytest.fixture
    def pdf_subbox_within(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([10, 10, 600, 780]),
        )
        yield pdf

    @pytest.fixture
    def pdf_subbox_completely_outside(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            BleedBox=Array([700, 800, 900, 1000]),
        )
        yield pdf

    def test_cropbox_clipped(self, pdf_cropbox_exceeds: Pdf):
        result = sanitize_page_boxes(pdf_cropbox_exceeds)
        assert result["boxes_clipped"] == 1
        crop = pdf_cropbox_exceeds.pages[0].obj[Name.CropBox]
        assert [float(v) for v in crop] == [0, 0, 612, 792]

    def test_all_subboxes_clipped(self, pdf_all_subboxes_exceed: Pdf):
        result = sanitize_page_boxes(pdf_all_subboxes_exceed)
        assert result["boxes_clipped"] == 3

    def test_subbox_within_not_clipped(self, pdf_subbox_within: Pdf):
        result = sanitize_page_boxes(pdf_subbox_within)
        assert result["boxes_clipped"] == 0
        crop = pdf_subbox_within.pages[0].obj[Name.CropBox]
        assert [float(v) for v in crop] == [10, 10, 600, 780]

    def test_subbox_outside_removed(self, pdf_subbox_completely_outside: Pdf):
        result = sanitize_page_boxes(pdf_subbox_completely_outside)
        assert result["boxes_clipped"] == 1
        page = pdf_subbox_completely_outside.pages[0]
        assert Name.BleedBox not in page.obj


# ---------------------------------------------------------------------------
# TestTrimBoxEnsurance
# ---------------------------------------------------------------------------


class TestTrimBoxEnsurance:
    """Tests for ensuring TrimBox or ArtBox presence."""

    @pytest.fixture
    def pdf_no_trimbox_no_artbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(pdf, MediaBox=Array([0, 0, 612, 792]))
        yield pdf

    @pytest.fixture
    def pdf_no_trimbox_has_cropbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([10, 10, 600, 780]),
        )
        yield pdf

    @pytest.fixture
    def pdf_has_trimbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            TrimBox=Array([5, 5, 607, 787]),
        )
        yield pdf

    @pytest.fixture
    def pdf_has_artbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            ArtBox=Array([20, 20, 590, 770]),
        )
        yield pdf

    def test_trimbox_from_mediabox(self, pdf_no_trimbox_no_artbox: Pdf):
        result = sanitize_page_boxes(pdf_no_trimbox_no_artbox)
        assert result["trimbox_added"] == 1
        tb = pdf_no_trimbox_no_artbox.pages[0].obj[Name.TrimBox]
        assert [float(v) for v in tb] == [0, 0, 612, 792]

    def test_trimbox_from_cropbox(self, pdf_no_trimbox_has_cropbox: Pdf):
        result = sanitize_page_boxes(pdf_no_trimbox_has_cropbox)
        assert result["trimbox_added"] == 1
        tb = pdf_no_trimbox_has_cropbox.pages[0].obj[Name.TrimBox]
        assert [float(v) for v in tb] == [10, 10, 600, 780]

    def test_existing_trimbox_kept(self, pdf_has_trimbox: Pdf):
        result = sanitize_page_boxes(pdf_has_trimbox)
        assert result["trimbox_added"] == 0
        tb = pdf_has_trimbox.pages[0].obj[Name.TrimBox]
        assert [float(v) for v in tb] == [5, 5, 607, 787]

    def test_artbox_sufficient(self, pdf_has_artbox: Pdf):
        """ArtBox alone satisfies the TrimBox/ArtBox requirement."""
        result = sanitize_page_boxes(pdf_has_artbox)
        assert result["trimbox_added"] == 0


# ---------------------------------------------------------------------------
# TestMultiplePages
# ---------------------------------------------------------------------------


class TestMultiplePages:
    """Tests with multiple pages having different issues."""

    @pytest.fixture
    def pdf_mixed(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        # Page 1: inverted MediaBox, no TrimBox
        _make_page(pdf, MediaBox=Array([612, 792, 0, 0]))
        # Page 2: normal MediaBox with TrimBox
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 595, 842]),
            TrimBox=Array([0, 0, 595, 842]),
        )
        # Page 3: CropBox exceeds MediaBox, no TrimBox
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([-10, -10, 620, 800]),
        )
        yield pdf

    def test_mixed_pages(self, pdf_mixed: Pdf):
        result = sanitize_page_boxes(pdf_mixed)
        # Page 1: normalized + trimbox added
        assert result["boxes_normalized"] >= 1
        assert result["trimbox_added"] >= 1
        # Page 3: clipped + trimbox added (from clipped CropBox)
        assert result["boxes_clipped"] >= 1
        # Total trimbox added: pages 1 and 3
        assert result["trimbox_added"] == 2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty PDF, float coordinates, exact boundary."""

    @pytest.fixture
    def empty_pdf(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        yield pdf

    @pytest.fixture
    def pdf_float_coords(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0.5, 0.5, 611.5, 791.5]),
            CropBox=Array([0.5, 0.5, 611.5, 791.5]),
        )
        yield pdf

    @pytest.fixture
    def pdf_exact_boundary(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 612, 792]),
            CropBox=Array([0, 0, 612, 792]),
        )
        yield pdf

    @pytest.fixture
    def pdf_mediabox_too_small(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 2, 2]),
            CropBox=Array([0, 0, 2, 2]),
        )
        yield pdf

    @pytest.fixture
    def pdf_mediabox_too_large(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 14402, 14402]),
        )
        yield pdf

    @pytest.fixture
    def pdf_small_subbox(self) -> Generator[Pdf, None, None]:
        pdf = new_pdf()
        _make_page(
            pdf,
            MediaBox=Array([0, 0, 4, 4]),
            CropBox=Array([0, 0, 2, 2]),
        )
        yield pdf

    def test_empty_pdf(self, empty_pdf: Pdf):
        """Empty PDF (no pages) should not crash."""
        result = sanitize_page_boxes(empty_pdf)
        assert all(v == 0 for v in result.values())

    def test_float_coordinates(self, pdf_float_coords: Pdf):
        """Float coordinates are handled correctly."""
        result = sanitize_page_boxes(pdf_float_coords)
        assert result["boxes_clipped"] == 0
        mb = pdf_float_coords.pages[0].obj[Name.MediaBox]
        assert [float(v) for v in mb] == [0.5, 0.5, 611.5, 791.5]

    def test_exact_boundary_not_clipped(self, pdf_exact_boundary: Pdf):
        """CropBox exactly equal to MediaBox is not clipped."""
        result = sanitize_page_boxes(pdf_exact_boundary)
        assert result["boxes_clipped"] == 0

    def test_too_small_mediabox_is_resized(self, pdf_mediabox_too_small: Pdf):
        """MediaBox/CropBox below 3 units are expanded to valid size."""
        result = sanitize_page_boxes(pdf_mediabox_too_small)
        assert result["boxes_normalized"] >= 1
        assert result["boxes_clipped"] >= 1
        page = pdf_mediabox_too_small.pages[0].obj
        mb = [float(v) for v in page[Name.MediaBox]]
        cb = [float(v) for v in page[Name.CropBox]]
        assert mb == [0, 0, 3, 3]
        assert cb == [0, 0, 3, 3]

    def test_too_large_mediabox_is_resized(self, pdf_mediabox_too_large: Pdf):
        """MediaBox above 14400 units is clamped down to valid size."""
        result = sanitize_page_boxes(pdf_mediabox_too_large)
        assert result["boxes_normalized"] >= 1
        mb = pdf_mediabox_too_large.pages[0].obj[Name.MediaBox]
        assert [float(v) for v in mb] == [0, 0, 14400, 14400]

    def test_small_subbox_is_expanded(self, pdf_small_subbox: Pdf):
        """Sub-boxes below 3 units are adjusted to meet size limits."""
        result = sanitize_page_boxes(pdf_small_subbox)
        assert result["boxes_clipped"] >= 1
        crop = pdf_small_subbox.pages[0].obj[Name.CropBox]
        assert [float(v) for v in crop] == [0, 0, 3, 3]


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Unit tests for internal helper functions."""

    def test_is_valid_box_correct(self):
        assert _is_valid_box(Array([0, 0, 612, 792])) is True

    def test_is_valid_box_3_elements(self):
        assert _is_valid_box(Array([0, 0, 612])) is False

    def test_is_valid_box_5_elements(self):
        assert _is_valid_box(Array([0, 0, 612, 792, 1])) is False

    def test_is_valid_box_non_array(self):
        assert _is_valid_box(Name.Foo) is False

    def test_normalize_box_normal(self):
        assert _normalize_box(Array([0, 0, 612, 792])) == (0, 0, 612, 792)

    def test_normalize_box_swapped(self):
        assert _normalize_box(Array([612, 792, 0, 0])) == (0, 0, 612, 792)

    def test_clip_to_mediabox_inside(self):
        result = _clip_to_mediabox((10, 10, 600, 780), (0, 0, 612, 792))
        assert result == (10, 10, 600, 780)

    def test_clip_to_mediabox_exceeds(self):
        result = _clip_to_mediabox((-10, -10, 620, 800), (0, 0, 612, 792))
        assert result == (0, 0, 612, 792)

    def test_coords_equal_same(self):
        assert _coords_equal((0, 0, 612, 792), (0, 0, 612, 792)) is True

    def test_coords_equal_tiny_diff(self):
        assert _coords_equal((0, 0, 612, 792), (0, 0, 612.0000001, 792)) is True

    def test_coords_equal_different(self):
        assert _coords_equal((0, 0, 612, 792), (0, 0, 611, 792)) is False

    def test_resolve_mediabox_no_parent(self):
        d = Dictionary(Type=Name.Page)
        assert _resolve_mediabox_from_parent(d) is None
