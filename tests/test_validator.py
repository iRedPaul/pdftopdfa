# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for validator.py."""

from pathlib import Path

import pikepdf
import pytest
from conftest import open_pdf
from pikepdf import Pdf

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.validator import (
    _extract_xmp_bytes,
    _parse_xmp_tree,
    detect_iso_standards,
    detect_pdfa_level,
)


class TestExtractXmpBytes:
    """Tests for _extract_xmp_bytes."""

    def test_extract_from_pdf_without_xmp(self, sample_pdf_bytes: bytes) -> None:
        """PDF without XMP returns None."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        result = _extract_xmp_bytes(pdf)

        assert result is None

    def test_extract_from_pdf_with_xmp(self, sample_pdf: Path, tmp_dir: Path) -> None:
        """PDF with XMP returns bytes with 'xmpmeta'."""
        # Convert to PDF/A to get XMP
        output_path = tmp_dir / "converted.pdf"
        convert_to_pdfa(sample_pdf, output_path)

        with Pdf.open(output_path) as pdf:
            xmp_bytes = _extract_xmp_bytes(pdf)

            assert xmp_bytes is not None
            assert b"xmpmeta" in xmp_bytes


class TestParseXmpTree:
    """Tests for _parse_xmp_tree."""

    def test_parse_valid_xmp(self) -> None:
        """Valid XMP is parsed and returns Element."""
        xmp = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>"""

        result = _parse_xmp_tree(xmp)
        assert result is not None

    def test_parse_invalid_xml(self) -> None:
        """Invalid XML returns None."""
        invalid_xmp = b"<invalid>not closed xml"
        result = _parse_xmp_tree(invalid_xmp)

        assert result is None

    def test_parse_empty_bytes(self) -> None:
        """Empty bytes return None."""
        result = _parse_xmp_tree(b"")
        assert result is None


class TestDetectPdfaLevel:
    """Tests for detect_pdfa_level."""

    def test_detect_level_from_converted_pdf(
        self, sample_pdf: Path, tmp_dir: Path
    ) -> None:
        """Converted PDF has detectable level."""
        output_path = tmp_dir / "converted.pdf"
        convert_to_pdfa(sample_pdf, output_path, level="2b")

        with Pdf.open(output_path) as pdf:
            level = detect_pdfa_level(pdf)
            assert level == "2b"

    def test_detect_returns_none_for_regular_pdf(self, sample_pdf: Path) -> None:
        """Regular PDF returns None."""
        with Pdf.open(sample_pdf) as pdf:
            level = detect_pdfa_level(pdf)
            assert level is None

    @pytest.mark.parametrize("target_level", ["2b", "2u", "3b", "3u"])
    def test_detect_all_levels(
        self, sample_pdf: Path, tmp_dir: Path, target_level: str
    ) -> None:
        """Detects all PDF/A levels correctly."""
        output_path = tmp_dir / f"converted_{target_level}.pdf"
        convert_to_pdfa(sample_pdf, output_path, level=target_level)

        with Pdf.open(output_path) as pdf:
            detected = detect_pdfa_level(pdf)
            assert detected == target_level

    @pytest.mark.parametrize(
        ("conformance_xml", "expected_level"),
        [
            ("", "4"),
            ("<pdfaid:conformance>E</pdfaid:conformance>", "4e"),
            ("<pdfaid:conformance>F</pdfaid:conformance>", "4f"),
        ],
        ids=["base", "4e", "4f"],
    )
    def test_detect_pdfa4_variants(
        self, sample_pdf_bytes: bytes, conformance_xml: str, expected_level: str
    ) -> None:
        """Detects PDF/A-4 variants (base, 4e, 4f) correctly."""
        from io import BytesIO

        xmp = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description rdf:about=""
                    xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
                    <pdfaid:part>4</pdfaid:part>
                    {conformance_xml}
                </rdf:Description>
            </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>""".encode()

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        metadata_stream = pikepdf.Stream(pdf, xmp)
        pdf.Root.Metadata = pdf.make_indirect(metadata_stream)

        level = detect_pdfa_level(pdf)
        assert level == expected_level

    def test_detect_pdfa4_invalid_conformance(self, sample_pdf_bytes: bytes) -> None:
        """PDF/A-4 with invalid conformance (X) returns None."""
        from io import BytesIO

        xmp = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description rdf:about=""
                    xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
                    <pdfaid:part>4</pdfaid:part>
                    <pdfaid:conformance>X</pdfaid:conformance>
                </rdf:Description>
            </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>"""

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        metadata_stream = pikepdf.Stream(pdf, xmp)
        pdf.Root.Metadata = pdf.make_indirect(metadata_stream)

        level = detect_pdfa_level(pdf)
        assert level is None


class TestDetectIsoStandards:
    """Tests for detect_iso_standards."""

    @pytest.mark.parametrize(
        (
            "ns_prefix",
            "ns_uri",
            "element",
            "expected_standard",
            "expected_version",
        ),
        [
            (
                "pdfxid",
                "http://www.npes.org/pdfx/ns/id/",
                "GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion",
                "PDF/X",
                "PDF/X-4",
            ),
            (
                "pdfuaid",
                "http://www.aiim.org/pdfua/ns/id/",
                "part>2</pdfuaid:part",
                "PDF/UA",
                "2",
            ),
            (
                "pdfeid",
                "http://www.aiim.org/pdfe/ns/id/",
                "part>1</pdfeid:part",
                "PDF/E",
                "1",
            ),
            (
                "pdfvtid",
                "http://www.npes.org/pdfvt/ns/id/",
                "GTS_PDFVTVersion>PDF/VT-1</pdfvtid:GTS_PDFVTVersion",
                "PDF/VT",
                "PDF/VT-1",
            ),
        ],
        ids=["PDF/X", "PDF/UA", "PDF/E", "PDF/VT"],
    )
    def test_detect_iso_standard(
        self,
        sample_pdf_bytes,
        ns_prefix,
        ns_uri,
        element,
        expected_standard,
        expected_version,
    ):
        """Detects individual ISO standards from XMP metadata."""
        from io import BytesIO

        xmp = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description rdf:about=""
                    xmlns:{ns_prefix}="{ns_uri}">
                    <{ns_prefix}:{element}>
                </rdf:Description>
            </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>""".encode()

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        metadata_stream = pikepdf.Stream(pdf, xmp)
        pdf.Root.Metadata = pdf.make_indirect(metadata_stream)

        standards = detect_iso_standards(pdf)
        assert len(standards) == 1
        assert standards[0].standard == expected_standard
        assert standards[0].version == expected_version

    def test_detect_multiple_standards(self, sample_pdf_bytes: bytes) -> None:
        """Detects multiple ISO standards from a single PDF."""
        from io import BytesIO

        xmp = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description rdf:about=""
                    xmlns:pdfxid="http://www.npes.org/pdfx/ns/id/"
                    xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/">
                    <pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>
                    <pdfuaid:part>1</pdfuaid:part>
                </rdf:Description>
            </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>"""

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        metadata_stream = pikepdf.Stream(pdf, xmp)
        pdf.Root.Metadata = pdf.make_indirect(metadata_stream)

        standards = detect_iso_standards(pdf)
        assert len(standards) == 2
        standard_names = {s.standard for s in standards}
        assert standard_names == {"PDF/X", "PDF/UA"}

    def test_detect_no_standards(self, sample_pdf: Path) -> None:
        """Regular PDF without ISO standards returns empty list."""
        with Pdf.open(sample_pdf) as pdf:
            standards = detect_iso_standards(pdf)
            assert standards == []

    def test_detect_pdfua_attribute_form(self, sample_pdf_bytes: bytes) -> None:
        """Detects PDF/UA from XMP attribute form."""
        from io import BytesIO

        xmp = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description rdf:about=""
                    xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/"
                    pdfuaid:part="1"/>
            </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>"""

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        metadata_stream = pikepdf.Stream(pdf, xmp)
        pdf.Root.Metadata = pdf.make_indirect(metadata_stream)

        standards = detect_iso_standards(pdf)
        assert len(standards) == 1
        assert standards[0].standard == "PDF/UA"
        assert standards[0].version == "1"
