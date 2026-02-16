# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for extensions.py."""

from io import BytesIO

import pytest
from conftest import open_pdf
from pikepdf import Dictionary, Name

from pdftopdfa.extensions import (
    add_adbe_extension,
    add_extensions_if_needed,
    needs_extension_level,
    remove_pdf20_extensions,
)


class TestNeedsExtensionLevel:
    """Tests for needs_extension_level."""

    def test_pdfa3_needs_extension(self) -> None:
        """PDF/A-3 requires extension level 3."""
        result = needs_extension_level("3b")
        assert result == (True, 3)

    def test_pdfa3a_needs_extension(self) -> None:
        """PDF/A-3a requires extension level 3."""
        result = needs_extension_level("3a")
        assert result == (True, 3)

    def test_pdfa3u_needs_extension(self) -> None:
        """PDF/A-3u requires extension level 3."""
        result = needs_extension_level("3u")
        assert result == (True, 3)

    def test_pdfa2_no_extension(self) -> None:
        """PDF/A-2 does not require extension."""
        result = needs_extension_level("2b")
        assert result == (False, 0)

    def test_pdfa2a_no_extension(self) -> None:
        """PDF/A-2a does not require extension."""
        result = needs_extension_level("2a")
        assert result == (False, 0)

    def test_pdfa2u_no_extension(self) -> None:
        """PDF/A-2u does not require extension."""
        result = needs_extension_level("2u")
        assert result == (False, 0)

    def test_uppercase_level(self) -> None:
        """Uppercase level string works."""
        result = needs_extension_level("3B")
        assert result == (True, 3)


class TestAddAdbeExtension:
    """Tests for add_adbe_extension."""

    def test_creates_extensions_dict(self, sample_pdf_bytes: bytes) -> None:
        """Creates Extensions dictionary when not present."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        assert "/Extensions" not in pdf.Root

        result = add_adbe_extension(pdf, 3)

        assert result is True
        assert "/Extensions" in pdf.Root
        assert "/ADBE" in pdf.Root.Extensions

    def test_adbe_structure(self, sample_pdf_bytes: bytes) -> None:
        """ADBE extension has correct structure."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        add_adbe_extension(pdf, 3)

        adbe = pdf.Root.Extensions.ADBE
        assert str(adbe.BaseVersion) == "/1.7"
        assert int(adbe.ExtensionLevel) == 3

    def test_preserves_existing_extensions(self, sample_pdf_bytes: bytes) -> None:
        """Preserves other entries in Extensions dictionary."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        # Add existing extension
        pdf.Root.Extensions = Dictionary(
            OtherVendor=Dictionary(
                BaseVersion=Name("/1.5"),
                ExtensionLevel=1,
            )
        )

        add_adbe_extension(pdf, 3)

        assert "/OtherVendor" in pdf.Root.Extensions
        assert "/ADBE" in pdf.Root.Extensions

    def test_no_downgrade(self, sample_pdf_bytes: bytes) -> None:
        """Does not downgrade existing higher extension level."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        # Add ADBE extension with level 5
        pdf.Root.Extensions = Dictionary(
            ADBE=Dictionary(
                BaseVersion=Name("/1.7"),
                ExtensionLevel=5,
            )
        )

        result = add_adbe_extension(pdf, 3)

        assert result is False
        assert int(pdf.Root.Extensions.ADBE.ExtensionLevel) == 5

    def test_upgrades_lower_level(self, sample_pdf_bytes: bytes) -> None:
        """Upgrades existing lower extension level."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        # Add ADBE extension with level 1
        pdf.Root.Extensions = Dictionary(
            ADBE=Dictionary(
                BaseVersion=Name("/1.7"),
                ExtensionLevel=1,
            )
        )

        result = add_adbe_extension(pdf, 3)

        assert result is True
        assert int(pdf.Root.Extensions.ADBE.ExtensionLevel) == 3

    def test_equal_level_no_change(self, sample_pdf_bytes: bytes) -> None:
        """Does not change when level is already equal."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        # Add ADBE extension with level 3
        pdf.Root.Extensions = Dictionary(
            ADBE=Dictionary(
                BaseVersion=Name("/1.7"),
                ExtensionLevel=3,
            )
        )

        result = add_adbe_extension(pdf, 3)

        assert result is False
        assert int(pdf.Root.Extensions.ADBE.ExtensionLevel) == 3


class TestAddExtensionsIfNeeded:
    """Tests for add_extensions_if_needed."""

    def test_adds_for_pdfa3(self, sample_pdf_bytes: bytes) -> None:
        """Adds extensions for PDF/A-3."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        result = add_extensions_if_needed(pdf, "3b")

        assert result is True
        assert "/Extensions" in pdf.Root
        assert "/ADBE" in pdf.Root.Extensions
        assert int(pdf.Root.Extensions.ADBE.ExtensionLevel) == 3

    def test_skips_for_pdfa2(self, sample_pdf_bytes: bytes) -> None:
        """Does not add extensions for PDF/A-2."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        result = add_extensions_if_needed(pdf, "2b")

        assert result is False
        assert "/Extensions" not in pdf.Root

    @pytest.mark.parametrize("level", ["3a", "3b", "3u"])
    def test_all_pdfa3_variants(self, sample_pdf_bytes: bytes, level: str) -> None:
        """All PDF/A-3 variants get extensions."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        result = add_extensions_if_needed(pdf, level)

        assert result is True
        assert "/Extensions" in pdf.Root

    @pytest.mark.parametrize("level", ["2a", "2b", "2u"])
    def test_all_pdfa2_variants(self, sample_pdf_bytes: bytes, level: str) -> None:
        """No PDF/A-2 variants get extensions."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        result = add_extensions_if_needed(pdf, level)

        assert result is False
        assert "/Extensions" not in pdf.Root


class TestRemovePdf20Extensions:
    """Tests for remove_pdf20_extensions."""

    def test_removes_non_adbe_extension(self, sample_pdf_bytes: bytes) -> None:
        """Non-ADBE extension entries are removed."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        pdf.Root.Extensions = Dictionary(
            ADBE=Dictionary(
                BaseVersion=Name("/1.7"),
                ExtensionLevel=3,
            ),
            ISO_=Dictionary(
                BaseVersion=Name("/2.0"),
                ExtensionLevel=32000,
            ),
        )

        result = remove_pdf20_extensions(pdf)

        assert result == 1
        assert "/ADBE" in pdf.Root.Extensions
        assert "/ISO_" not in pdf.Root.Extensions

    def test_preserves_adbe_extension(self, sample_pdf_bytes: bytes) -> None:
        """ADBE extension is preserved."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        pdf.Root.Extensions = Dictionary(
            ADBE=Dictionary(
                BaseVersion=Name("/1.7"),
                ExtensionLevel=3,
            ),
        )

        result = remove_pdf20_extensions(pdf)

        assert result == 0
        assert "/ADBE" in pdf.Root.Extensions

    def test_no_extensions_returns_zero(self, sample_pdf_bytes: bytes) -> None:
        """Returns 0 when no Extensions dictionary exists."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        result = remove_pdf20_extensions(pdf)
        assert result == 0

    def test_multiple_non_adbe_removed(self, sample_pdf_bytes: bytes) -> None:
        """Multiple non-ADBE entries are all removed."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        pdf.Root.Extensions = Dictionary(
            ADBE=Dictionary(
                BaseVersion=Name("/1.7"),
                ExtensionLevel=3,
            ),
            ISO_=Dictionary(
                BaseVersion=Name("/2.0"),
                ExtensionLevel=32000,
            ),
            OtherVendor=Dictionary(
                BaseVersion=Name("/1.5"),
                ExtensionLevel=1,
            ),
        )

        result = remove_pdf20_extensions(pdf)

        assert result == 2
        assert "/ADBE" in pdf.Root.Extensions
        assert "/ISO_" not in pdf.Root.Extensions
        assert "/OtherVendor" not in pdf.Root.Extensions

    def test_called_from_add_extensions_if_needed(
        self, sample_pdf_bytes: bytes
    ) -> None:
        """add_extensions_if_needed removes non-ADBE extensions."""
        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        pdf.Root.Extensions = Dictionary(
            ISO_=Dictionary(
                BaseVersion=Name("/2.0"),
                ExtensionLevel=32000,
            ),
        )

        add_extensions_if_needed(pdf, "3b")

        assert "/ADBE" in pdf.Root.Extensions
        assert "/ISO_" not in pdf.Root.Extensions
