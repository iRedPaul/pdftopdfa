# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/utils.py — fsType embedding restrictions."""

from io import BytesIO

from pdftopdfa.fonts.utils import (
    FSTYPE_BITMAP_ONLY,
    FSTYPE_EDITABLE,
    FSTYPE_NO_SUBSETTING,
    FSTYPE_PREVIEW_AND_PRINT,
    FSTYPE_RESTRICTED_LICENSE,
    check_fstype_restrictions,
    get_fstype,
)


def _make_font_with_fstype(fstype_value: int) -> bytes:
    """Creates a minimal TrueType font with a specific fsType value."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "space"])
    fb.setupCharacterMap({32: "space"})
    fb.setupGlyf({".notdef": Glyph(), "space": Glyph()})
    fb.setupHorizontalMetrics({".notdef": (500, 0), "space": (250, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Test", "styleName": "Regular"})
    fb.setupOS2(fsType=fstype_value)
    fb.setupPost()

    buf = BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


class TestGetFstype:
    """Tests for get_fstype."""

    def test_installable_embedding(self):
        """fsType 0 means installable embedding (no restrictions)."""
        font_data = _make_font_with_fstype(0x0000)
        assert get_fstype(font_data) == 0

    def test_restricted_license(self):
        """fsType 2 means restricted license."""
        font_data = _make_font_with_fstype(FSTYPE_RESTRICTED_LICENSE)
        assert get_fstype(font_data) == FSTYPE_RESTRICTED_LICENSE

    def test_no_subsetting(self):
        """fsType 0x0100 means no subsetting allowed."""
        font_data = _make_font_with_fstype(FSTYPE_NO_SUBSETTING)
        assert get_fstype(font_data) == FSTYPE_NO_SUBSETTING

    def test_combined_flags(self):
        """Multiple fsType bits can be set simultaneously."""
        combined = FSTYPE_EDITABLE | FSTYPE_NO_SUBSETTING
        font_data = _make_font_with_fstype(combined)
        assert get_fstype(font_data) == combined

    def test_invalid_data_returns_none(self):
        """Invalid font data returns None."""
        assert get_fstype(b"not a font") is None

    def test_empty_data_returns_none(self):
        """Empty font data returns None."""
        assert get_fstype(b"") is None

    def test_real_font_has_fstype(self):
        """A real font file (LiberationSans) has a valid fsType."""
        from importlib import resources

        font_path = (
            resources.files("pdftopdfa")
            / "resources"
            / "fonts"
            / "LiberationSans-Regular.ttf"
        )
        font_data = font_path.read_bytes()
        result = get_fstype(font_data)
        assert result is not None
        assert isinstance(result, int)


class TestCheckFstypeRestrictions:
    """Tests for check_fstype_restrictions."""

    def test_installable_no_restrictions(self):
        """fsType 0 has no restrictions."""
        allowed, can_subset, warnings = check_fstype_restrictions(0x0000)
        assert allowed is True
        assert can_subset is True
        assert warnings == []

    def test_restricted_license(self):
        """fsType 0x0002 disallows embedding."""
        allowed, can_subset, warnings = check_fstype_restrictions(
            FSTYPE_RESTRICTED_LICENSE
        )
        assert allowed is False
        assert can_subset is True
        assert len(warnings) == 1
        assert "Restricted License" in warnings[0]

    def test_preview_and_print(self):
        """fsType 0x0004 allows embedding but warns."""
        allowed, can_subset, warnings = check_fstype_restrictions(
            FSTYPE_PREVIEW_AND_PRINT
        )
        assert allowed is True
        assert can_subset is True
        assert len(warnings) == 1
        assert "Preview & Print" in warnings[0]

    def test_editable_no_restrictions(self):
        """fsType 0x0008 allows embedding and subsetting."""
        allowed, can_subset, warnings = check_fstype_restrictions(FSTYPE_EDITABLE)
        assert allowed is True
        assert can_subset is True
        assert warnings == []

    def test_no_subsetting(self):
        """fsType 0x0100 disallows subsetting."""
        allowed, can_subset, warnings = check_fstype_restrictions(FSTYPE_NO_SUBSETTING)
        assert allowed is True
        assert can_subset is False
        assert len(warnings) == 1
        assert "No subsetting" in warnings[0]

    def test_bitmap_only(self):
        """fsType 0x0200 warns about bitmap-only embedding."""
        allowed, can_subset, warnings = check_fstype_restrictions(FSTYPE_BITMAP_ONLY)
        assert allowed is True
        assert can_subset is True
        assert len(warnings) == 1
        assert "Bitmap" in warnings[0]

    def test_restricted_plus_no_subsetting(self):
        """Combined restricted + no subsetting flags."""
        combined = FSTYPE_RESTRICTED_LICENSE | FSTYPE_NO_SUBSETTING
        allowed, can_subset, warnings = check_fstype_restrictions(combined)
        assert allowed is False
        assert can_subset is False
        assert len(warnings) == 2
