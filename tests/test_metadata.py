# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for metadata.py."""

from datetime import UTC, datetime
from pathlib import Path

import pikepdf
import pytest
from conftest import new_pdf, open_pdf
from lxml import etree
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.exceptions import ConversionError
from pdftopdfa.metadata import (
    _NS_PDFA_EXTENSION,
    _NS_PDFA_PROPERTY,
    _NS_PDFA_SCHEMA,
    _NS_PDFA_TYPE,
    NAMESPACES,
    XMP_HEADER,
    XMP_TRAILER,
    _collect_non_catalog_extension_needs,
    _collect_preserved_elements,
    _extract_existing_xmp,
    _extract_extension_schema_blocks,
    _format_pdf_date,
    _has_undeclarable_structure,
    _normalize_structural_properties,
    _normalize_trapped,
    _parse_pdf_date,
    _parse_xmp_bytes,
    _reserialize_xmp,
    _sanitize_extension_schema_blocks,
    _sanitize_non_catalog_metadata,
    create_xmp_metadata,
    embed_xmp_metadata,
    extract_pdf_info,
    sync_metadata,
)


class TestNormalizeTrapped:
    """Tests for _normalize_trapped."""

    def test_normalize_none_returns_unknown(self) -> None:
        """None value returns Unknown."""
        assert _normalize_trapped(None) == "Unknown"

    def test_normalize_true_name(self) -> None:
        """pikepdf Name /True returns True."""
        assert _normalize_trapped(Name("/True")) == "True"

    def test_normalize_false_name(self) -> None:
        """pikepdf Name /False returns False."""
        assert _normalize_trapped(Name("/False")) == "False"

    def test_normalize_unknown_name(self) -> None:
        """pikepdf Name /Unknown returns Unknown."""
        assert _normalize_trapped(Name("/Unknown")) == "Unknown"

    def test_normalize_string_true(self) -> None:
        """String 'true' returns True."""
        assert _normalize_trapped("true") == "True"
        assert _normalize_trapped("True") == "True"
        assert _normalize_trapped("TRUE") == "True"

    def test_normalize_string_false(self) -> None:
        """String 'false' returns False."""
        assert _normalize_trapped("false") == "False"
        assert _normalize_trapped("False") == "False"
        assert _normalize_trapped("FALSE") == "False"

    def test_normalize_string_unknown(self) -> None:
        """String 'unknown' returns Unknown."""
        assert _normalize_trapped("unknown") == "Unknown"
        assert _normalize_trapped("Unknown") == "Unknown"

    def test_normalize_invalid_value(self) -> None:
        """Invalid value returns Unknown."""
        assert _normalize_trapped("invalid") == "Unknown"
        assert _normalize_trapped("yes") == "Unknown"
        assert _normalize_trapped("no") == "Unknown"
        assert _normalize_trapped(123) == "Unknown"

    def test_normalize_with_slash_prefix(self) -> None:
        """Values with slash prefix are handled."""
        assert _normalize_trapped("/True") == "True"
        assert _normalize_trapped("/False") == "False"
        assert _normalize_trapped("/Unknown") == "Unknown"


class TestParsePdfDate:
    """Tests for _parse_pdf_date."""

    def test_parse_standard_date(self) -> None:
        """Standard PDF date is parsed correctly."""
        date_str = "D:20240115120000+00'00'"
        result = _parse_pdf_date(date_str)

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 0
        assert result.second == 0

    def test_parse_date_without_prefix(self) -> None:
        """Date without D: prefix is parsed."""
        date_str = "20240115120000"
        result = _parse_pdf_date(date_str)

        assert result is not None
        assert result.year == 2024

    def test_parse_empty_string(self) -> None:
        """Empty string returns None."""
        result = _parse_pdf_date("")
        assert result is None

    def test_parse_invalid_date(self) -> None:
        """Invalid date returns None."""
        result = _parse_pdf_date("invalid")
        assert result is None

    def test_parse_positive_offset(self) -> None:
        """Positive timezone offset is converted to UTC."""
        result = _parse_pdf_date("D:20240101120000+02'00'")
        assert result is not None
        assert result.hour == 10
        assert result.tzinfo == UTC

    def test_parse_negative_offset(self) -> None:
        """Negative timezone offset is converted to UTC."""
        result = _parse_pdf_date("D:20240101120000-05'00'")
        assert result is not None
        assert result.hour == 17
        assert result.tzinfo == UTC

    def test_parse_z_suffix(self) -> None:
        """Z suffix means UTC, hour unchanged."""
        result = _parse_pdf_date("D:20240101120000Z")
        assert result is not None
        assert result.hour == 12
        assert result.tzinfo == UTC

    def test_parse_no_timezone(self) -> None:
        """No timezone defaults to UTC."""
        result = _parse_pdf_date("D:20240101120000")
        assert result is not None
        assert result.hour == 12
        assert result.tzinfo == UTC

    def test_parse_partial_date(self) -> None:
        """Partial date (year only) is parsed."""
        result = _parse_pdf_date("D:2024")
        assert result is not None
        assert result.year == 2024


class TestFormatPdfDate:
    """Tests for _format_pdf_date."""

    def test_format_produces_correct_format(self) -> None:
        """Formatted string matches D:YYYYMMDDHHmmSS+00'00'."""
        dt = datetime(2024, 1, 15, 12, 30, 45, tzinfo=UTC)
        result = _format_pdf_date(dt)
        assert result == "D:20240115123045+00'00'"

    def test_roundtrip_with_parse(self) -> None:
        """format -> parse roundtrip preserves datetime."""
        dt = datetime(2025, 6, 1, 8, 0, 0, tzinfo=UTC)
        formatted = _format_pdf_date(dt)
        parsed = _parse_pdf_date(formatted)
        assert parsed == dt


class TestExtractPdfInfo:
    """Tests for extract_pdf_info."""

    def test_extract_from_pdf_with_metadata(self, pdf_with_metadata: Path) -> None:
        """Extracts metadata from PDF with Info-Dictionary."""
        with Pdf.open(pdf_with_metadata) as pdf:
            info = extract_pdf_info(pdf)

        assert info["title"] == "Test Title"
        assert info["author"] == "Test Author"
        assert info["subject"] == "Test Description"
        assert info["creator"] == "Test Creator"
        assert info["producer"] == "Test Producer"
        assert info["creation_date"] is not None
        assert info["modification_date"] is not None

    def test_extract_from_empty_pdf(self, sample_pdf: Path) -> None:
        """Empty PDF has all values None."""
        with Pdf.open(sample_pdf) as pdf:
            info = extract_pdf_info(pdf)

        # All values should be None (except possibly auto-set ones)
        for key in ["title", "author", "subject"]:
            assert info[key] is None

    def test_extract_returns_all_keys(self, sample_pdf: Path) -> None:
        """Return contains all expected keys."""
        with Pdf.open(sample_pdf) as pdf:
            info = extract_pdf_info(pdf)

        expected_keys = {
            "title",
            "author",
            "subject",
            "keywords",
            "creator",
            "producer",
            "creation_date",
            "modification_date",
            "trapped",
        }
        assert set(info.keys()) == expected_keys

    def test_extract_trapped_value(self, tmp_dir: Path) -> None:
        """Trapped value is extracted and normalized."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.docinfo["/Trapped"] = Name("/True")

        pdf_path = tmp_dir / "trapped_true.pdf"
        pdf.save(pdf_path)

        with Pdf.open(pdf_path) as pdf:
            info = extract_pdf_info(pdf)

        assert info["trapped"] == "True"

    def test_extract_trapped_false(self, tmp_dir: Path) -> None:
        """Trapped False value is extracted."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.docinfo["/Trapped"] = Name("/False")

        pdf_path = tmp_dir / "trapped_false.pdf"
        pdf.save(pdf_path)

        with Pdf.open(pdf_path) as pdf:
            info = extract_pdf_info(pdf)

        assert info["trapped"] == "False"

    def test_extract_trapped_missing(self, sample_pdf: Path) -> None:
        """Missing Trapped returns None (not normalized yet)."""
        with Pdf.open(sample_pdf) as pdf:
            info = extract_pdf_info(pdf)

        assert info["trapped"] is None

    def test_extract_keywords(self, tmp_dir: Path) -> None:
        """Keywords value is extracted from DocInfo."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.docinfo["/Keywords"] = "pdf, archival, test"

        pdf_path = tmp_dir / "keywords.pdf"
        pdf.save(pdf_path)

        with Pdf.open(pdf_path) as pdf:
            info = extract_pdf_info(pdf)

        assert info["keywords"] == "pdf, archival, test"


class TestCreateXmpMetadata:
    """Tests for create_xmp_metadata."""

    def test_create_valid_xmp(self) -> None:
        """Creates valid XMP with header and trailer."""
        info = {
            "title": "Test",
            "author": "Author",
            "subject": "Description",
            "creation_date": datetime.now(UTC),
            "modification_date": datetime.now(UTC),
        }
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")

        assert xmp.startswith(XMP_HEADER)
        assert xmp.endswith(XMP_TRAILER)
        assert b"xmpmeta" in xmp

    def test_xmp_contains_pdfa_identification(self) -> None:
        """XMP contains pdfaid:part and pdfaid:conformance."""
        info = {"title": "Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")

        # Check pdfaid elements
        assert b"pdfaid:part" in xmp or b":part>" in xmp
        assert b"pdfaid:conformance" in xmp or b":conformance>" in xmp
        assert b">2<" in xmp  # Part = 2
        assert b">B<" in xmp  # Conformance = B

    def test_xmp_contains_title(self) -> None:
        """XMP contains the title."""
        info = {"title": "Mein Test Title"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")

        assert b"Mein Test Title" in xmp

    def test_xmp_contains_keywords(self) -> None:
        """XMP contains pdf:Keywords when keywords is set in info."""
        info = {"title": "Test", "keywords": "pdf, archival, compliance"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        assert b"Keywords" in xmp
        assert b"pdf, archival, compliance" in xmp

    def test_xmp_keywords_not_present_when_unset(self) -> None:
        """XMP does not contain pdf:Keywords when not provided in info."""
        info = {"title": "Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        assert b"Keywords" not in xmp

    def test_xmp_contains_trapped(self) -> None:
        """XMP contains pdf:Trapped when trapped is set in info."""
        info = {"title": "Test", "trapped": "True"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        assert b"Trapped" in xmp
        assert b">True<" in xmp

    def test_xmp_trapped_not_present_when_unset(self) -> None:
        """XMP does not contain Trapped when not provided in info."""
        info = {"title": "Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        assert b"Trapped" not in xmp

    def test_xmp_contains_metadata_date(self) -> None:
        """XMP contains xmp:MetadataDate element."""
        info = {"title": "Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        assert b"MetadataDate" in xmp

    def test_xmp_contains_padding(self) -> None:
        """XMP output contains padding spaces before end packet marker."""
        info = {"title": "Padding Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")

        # Padding should appear between XML content and the trailer
        trailer_pos = xmp.rfind(b'<?xpacket end="w"?>')
        assert trailer_pos > 0

        # Check there's substantial whitespace before the trailer
        pre_trailer = xmp[:trailer_pos]
        # Count padding spaces (at least 2048 bytes of padding)
        trailing_space = 0
        for b in reversed(pre_trailer):
            if b in (0x20, 0x0A):  # space or newline
                trailing_space += 1
            else:
                break
        assert trailing_space >= 2048


class TestEmbedXmpMetadata:
    """Tests for embed_xmp_metadata."""

    def test_embed_xmp_no_filter(self, sample_pdf_bytes: bytes) -> None:
        """XMP metadata stream must not have a /Filter (PDF/A requirement)."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        info = {"title": "Filter Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        embed_xmp_metadata(pdf, xmp)
        metadata_obj = pdf.Root.Metadata
        assert pikepdf.Name.Filter not in metadata_obj

    def test_embed_xmp_into_pdf(self, sample_pdf_bytes: bytes) -> None:
        """XMP is correctly embedded in PDF."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        info = {"title": "Embedded Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")

        embed_xmp_metadata(pdf, xmp)

        # Check if /Metadata is present in pdf.Root
        assert "/Metadata" in pdf.Root
        metadata_obj = pdf.Root.Metadata
        assert metadata_obj is not None


class TestSyncMetadata:
    """Tests for sync_metadata."""

    def test_sync_creates_xmp(self, sample_pdf_bytes: bytes) -> None:
        """sync_metadata creates /Metadata in PDF."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        sync_metadata(pdf, "2b")

        assert "/Metadata" in pdf.Root

    def test_sync_removes_malformed_non_catalog_metadata(
        self, sample_pdf_bytes: bytes, tmp_dir: Path
    ) -> None:
        """Malformed page metadata is removed; catalog metadata is kept."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        bad_page_metadata = pikepdf.Stream(pdf, b"<not-well-formed")
        bad_page_metadata.Type = Name.Metadata
        bad_page_metadata.Subtype = Name.XML
        pdf.pages[0].obj["/Metadata"] = pdf.make_indirect(bad_page_metadata)

        sync_metadata(pdf, "2u")

        assert "/Metadata" in pdf.Root
        assert "/Metadata" not in pdf.pages[0].obj

        out_path = tmp_dir / "metadata_cleaned.pdf"
        pdf.save(out_path)

        with Pdf.open(out_path) as reopened:
            assert "/Metadata" in reopened.Root
            assert "/Metadata" not in reopened.pages[0].obj

    def test_sync_preserves_valid_non_catalog_metadata(
        self, sample_pdf_bytes: bytes, tmp_dir: Path
    ) -> None:
        """Valid XMP page metadata is preserved after sync."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        # Create valid XMP metadata (e.g. EXIF-like data on page)
        valid_xmp = (
            b'<?xpacket begin="\xef\xbb\xbf"'
            b' id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
            b"<rdf:RDF xmlns:rdf="
            b'"http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            b'<rdf:Description rdf:about=""'
            b' xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
            b"<dc:description>\n"
            b"<rdf:Alt>\n"
            b'<rdf:li xml:lang="x-default">Test image</rdf:li>\n'
            b"</rdf:Alt>\n"
            b"</dc:description>\n"
            b"</rdf:Description>\n"
            b"</rdf:RDF>\n"
            b"</x:xmpmeta>\n"
            b'<?xpacket end="w"?>'
        )
        page_metadata = pikepdf.Stream(pdf, valid_xmp)
        page_metadata.Type = Name.Metadata
        page_metadata.Subtype = Name.XML
        pdf.pages[0].obj["/Metadata"] = pdf.make_indirect(page_metadata)

        sync_metadata(pdf, "2u")

        assert "/Metadata" in pdf.Root
        assert "/Metadata" in pdf.pages[0].obj

        out_path = tmp_dir / "metadata_preserved.pdf"
        pdf.save(out_path)

        with Pdf.open(out_path) as reopened:
            assert "/Metadata" in reopened.Root
            assert "/Metadata" in reopened.pages[0].obj
            meta_ref = reopened.pages[0].obj["/Metadata"]
            page_meta = reopened.get_object(meta_ref.objgen)
            content = bytes(page_meta.read_bytes())
            assert b"Test image" in content

    def test_sync_invalid_level_raises_error(self, sample_pdf_bytes: bytes) -> None:
        """Invalid level raises ConversionError."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        with pytest.raises(ConversionError, match="Invalid PDF/A level"):
            sync_metadata(pdf, "4b")

    @pytest.mark.parametrize("level", ["2b", "2u", "3b", "3u"])
    def test_sync_all_valid_levels(self, sample_pdf_bytes: bytes, level: str) -> None:
        """All valid levels are accepted."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        # Should not raise exception
        sync_metadata(pdf, level)
        assert "/Metadata" in pdf.Root

    def test_sync_preserves_existing_metadata(self, pdf_with_metadata: Path) -> None:
        """Existing metadata is transferred to XMP."""
        with Pdf.open(pdf_with_metadata) as pdf:
            sync_metadata(pdf, "2b")

            # Get XMP data
            metadata_stream = pdf.Root.Metadata
            xmp_bytes = bytes(metadata_stream.read_bytes())

            # Check if title was transferred
            assert b"Test Title" in xmp_bytes

    def test_sync_normalizes_trapped_in_docinfo(self, sample_pdf_bytes: bytes) -> None:
        """Normalizes /Trapped in DocInfo, writes pdf:Trapped to XMP."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        pdf.docinfo["/Trapped"] = Name("/Unknown")
        sync_metadata(pdf, "2b")

        # /Trapped stays in DocInfo as normalized Name
        assert "/Trapped" in pdf.docinfo
        assert str(pdf.docinfo["/Trapped"]) == "/Unknown"

        # pdf:Trapped is present in XMP
        xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
        assert b"Trapped" in xmp_bytes

    def test_sync_keeps_trapped_true(self, tmp_dir: Path) -> None:
        """Keeps /Trapped True in DocInfo, writes pdf:Trapped to XMP."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.docinfo["/Trapped"] = Name("/True")

        pdf_path = tmp_dir / "trapped_true.pdf"
        pdf.save(pdf_path)

        with Pdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            sync_metadata(pdf, "2b")

            # /Trapped stays in DocInfo as /True
            assert "/Trapped" in pdf.docinfo
            assert str(pdf.docinfo["/Trapped"]) == "/True"

            # pdf:Trapped is present in XMP
            xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
            assert b"Trapped" in xmp_bytes

    def test_sync_trapped_not_added_when_absent(self, sample_pdf_bytes: bytes) -> None:
        """When /Trapped is absent, it is not added to DocInfo or XMP."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        # Ensure no /Trapped exists
        if "/Trapped" in pdf.docinfo:
            del pdf.docinfo["/Trapped"]

        sync_metadata(pdf, "2b")

        assert "/Trapped" not in pdf.docinfo

        xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
        assert b"Trapped" not in xmp_bytes

    def test_sync_removes_non_standard_docinfo_keys(
        self, sample_pdf_bytes: bytes
    ) -> None:
        """sync_metadata removes non-standard keys from DocInfo."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        pdf.docinfo["/Title"] = "My Title"
        pdf.docinfo["/Author"] = "My Author"
        pdf.docinfo["/Company"] = "Acme Corp"
        pdf.docinfo["/SourceModified"] = "D:20200101"

        sync_metadata(pdf, "2b")

        assert "/Title" in pdf.docinfo
        assert "/Author" in pdf.docinfo
        assert "/Company" not in pdf.docinfo
        assert "/SourceModified" not in pdf.docinfo

    def test_sync_preserves_keywords_in_docinfo(self, sample_pdf_bytes: bytes) -> None:
        """Preserves /Keywords in DocInfo, writes pdf:Keywords to XMP."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        pdf.docinfo["/Keywords"] = "pdf, test, keywords"

        sync_metadata(pdf, "2b")

        assert "/Keywords" in pdf.docinfo

        # Verify pdf:Keywords appears in XMP
        xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
        assert b"pdf, test, keywords" in xmp_bytes

    def test_sync_sets_moddate_in_docinfo(self, sample_pdf_bytes: bytes) -> None:
        """/ModDate is set in DocInfo even when not previously present."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        # Ensure no /ModDate exists initially
        if "/ModDate" in pdf.docinfo:
            del pdf.docinfo["/ModDate"]

        sync_metadata(pdf, "2b")

        assert "/ModDate" in pdf.docinfo
        mod_date_str = str(pdf.docinfo["/ModDate"])
        assert mod_date_str.startswith("D:")

    def test_sync_sets_creationdate_in_docinfo(self, sample_pdf_bytes: bytes) -> None:
        """/CreationDate is set in DocInfo."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        sync_metadata(pdf, "2b")

        assert "/CreationDate" in pdf.docinfo
        creation_date_str = str(pdf.docinfo["/CreationDate"])
        assert creation_date_str.startswith("D:")

    def test_sync_docinfo_moddate_matches_xmp_modifydate(
        self, sample_pdf_bytes: bytes
    ) -> None:
        """DocInfo /ModDate matches XMP xmp:ModifyDate."""
        from io import BytesIO

        from lxml import etree

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        sync_metadata(pdf, "2b")

        # Parse DocInfo /ModDate
        mod_date_str = str(pdf.docinfo["/ModDate"])
        docinfo_date = _parse_pdf_date(mod_date_str)

        # Parse XMP xmp:ModifyDate
        xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
        # Strip XMP packet wrapper to get valid XML
        xml_start = xmp_bytes.index(b"<x:xmpmeta")
        xml_end = xmp_bytes.index(b"</x:xmpmeta>") + len(b"</x:xmpmeta>")
        xml_bytes = xmp_bytes[xml_start:xml_end]
        tree = etree.fromstring(xml_bytes)

        ns = {"xmp": "http://ns.adobe.com/xap/1.0/"}
        modify_date_text = tree.findall(".//xmp:ModifyDate", ns)[0].text

        # Both should represent the same timestamp
        assert docinfo_date is not None
        assert docinfo_date.strftime("%Y-%m-%dT%H:%M:%S") in modify_date_text

    def test_sync_preserves_original_creation_date(
        self, pdf_with_metadata: Path
    ) -> None:
        """Original CreationDate is preserved, not overwritten with now."""
        with Pdf.open(pdf_with_metadata) as pdf:
            original_creation = str(pdf.docinfo.get("/CreationDate", ""))

            sync_metadata(pdf, "2b")

            new_creation = str(pdf.docinfo["/CreationDate"])
            # The original date should be preserved (parsed and re-formatted)
            original_dt = _parse_pdf_date(original_creation)
            new_dt = _parse_pdf_date(new_creation)
            assert original_dt is not None
            assert new_dt is not None
            assert original_dt == new_dt

    def test_sync_sets_author_unknown_when_missing(
        self, pdf_with_metadata: Path
    ) -> None:
        """When /Author is missing, sync_metadata sets it to 'Unknown'."""
        with Pdf.open(pdf_with_metadata) as pdf:
            if "/Author" in pdf.docinfo:
                del pdf.docinfo["/Author"]

            sync_metadata(pdf, "2b")

            assert str(pdf.docinfo["/Author"]) == "Unknown"

            xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
            xmp_str = xmp_bytes.decode("utf-8")
            assert "Unknown" in xmp_str

    def test_sync_moddate_is_current_not_old(self, pdf_with_metadata: Path) -> None:
        """ModDate is set to current time, not the old modification date."""
        from datetime import timedelta

        with Pdf.open(pdf_with_metadata) as pdf:
            before = datetime.now(UTC)

            sync_metadata(pdf, "2b")

            after = datetime.now(UTC)

            mod_date = _parse_pdf_date(str(pdf.docinfo["/ModDate"]))
            assert mod_date is not None
            # Allow 1-second tolerance for truncation to seconds
            assert before - timedelta(seconds=1) <= mod_date <= after


def _build_xmp_with_extras(extra_xml: str) -> etree._Element:
    """Build an XMP tree with standard pdfaid + extra elements in Description."""
    ns_rdf = NAMESPACES["rdf"]
    ns_pdfaid = NAMESPACES["pdfaid"]
    xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{ns_pdfaid}"
           xmlns:dc="{NAMESPACES["dc"]}"
           xmlns:xmp="{NAMESPACES["xmp"]}"
           xmlns:pdf="{NAMESPACES["pdf"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfxid="{NAMESPACES["pdfxid"]}"
           xmlns:pdfeid="{NAMESPACES["pdfeid"]}"
           xmlns:pdfvtid="{NAMESPACES["pdfvtid"]}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      {extra_xml}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
    return etree.fromstring(xmp_xml.encode("utf-8"))


class TestXmpPreservation:
    """Tests for preservation of existing XMP metadata."""

    def test_pdfua_identification_preserved(self) -> None:
        """PDF/UA identification is preserved in output XMP."""
        tree = _build_xmp_with_extras("<pdfuaid:part>1</pdfuaid:part>")
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"pdfuaid:part" in xmp or (
            NAMESPACES["pdfuaid"].encode() in xmp and b">1<" in xmp
        )

    def test_pdfx_identification_preserved(self) -> None:
        """PDF/X identification is preserved in output XMP."""
        tree = _build_xmp_with_extras(
            "<pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"PDF/X-4" in xmp

    def test_pdfe_identification_preserved(self) -> None:
        """PDF/E identification is preserved in output XMP."""
        tree = _build_xmp_with_extras("<pdfeid:part>1</pdfeid:part>")
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"pdfeid" in xmp or NAMESPACES["pdfeid"].encode() in xmp

    def test_pdfvt_identification_preserved(self) -> None:
        """PDF/VT identification is preserved in output XMP."""
        tree = _build_xmp_with_extras(
            "<pdfvtid:GTS_PDFVTVersion>PDF/VT-1</pdfvtid:GTS_PDFVTVersion>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"PDF/VT-1" in xmp

    def test_dc_rights_preserved(self) -> None:
        """dc:rights (rdf:Alt) is preserved in output XMP."""
        tree = _build_xmp_with_extras(
            f'<dc:rights xmlns:dc="{NAMESPACES["dc"]}">'
            f'<rdf:Alt xmlns:rdf="{NAMESPACES["rdf"]}">'
            '<rdf:li xml:lang="x-default">Copyright 2024 Acme</rdf:li>'
            "</rdf:Alt></dc:rights>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"Copyright 2024 Acme" in xmp

    def test_dc_language_preserved(self) -> None:
        """dc:language (rdf:Bag) is preserved in output XMP."""
        tree = _build_xmp_with_extras(
            f'<dc:language xmlns:dc="{NAMESPACES["dc"]}">'
            f'<rdf:Bag xmlns:rdf="{NAMESPACES["rdf"]}">'
            "<rdf:li>en</rdf:li><rdf:li>de</rdf:li>"
            "</rdf:Bag></dc:language>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        assert "en" in xmp_str
        assert "de" in xmp_str

    def test_custom_namespace_preserved(self) -> None:
        """Elements from an unknown custom namespace are preserved."""
        custom_ns = "http://example.com/custom/ns/"
        tree = _build_xmp_with_extras(
            f'<custom:MyProp xmlns:custom="{custom_ns}">hello</custom:MyProp>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"hello" in xmp
        assert custom_ns.encode() in xmp

    def test_attribute_form_preserved(self) -> None:
        """Properties written as rdf:Description attributes are preserved."""
        ns_rdf = NAMESPACES["rdf"]
        ns_pdfuaid = NAMESPACES["pdfuaid"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfuaid="{ns_pdfuaid}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}">
    <rdf:Description rdf:about=""
                     pdfuaid:part="2">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        # The attribute should appear somewhere in the output
        xmp_str = xmp.decode("utf-8")
        assert "pdfuaid" in xmp_str or NAMESPACES["pdfuaid"] in xmp_str

    def test_managed_properties_not_duplicated(self) -> None:
        """Managed properties (dc:title etc.) appear only once (new value wins)."""
        tree = _build_xmp_with_extras(
            f'<dc:title xmlns:dc="{NAMESPACES["dc"]}">'
            f'<rdf:Alt xmlns:rdf="{NAMESPACES["rdf"]}">'
            '<rdf:li xml:lang="x-default">Old Title</rdf:li>'
            "</rdf:Alt></dc:title>"
        )
        info: dict = {"title": "New Title"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"New Title" in xmp
        assert b"Old Title" not in xmp

    def test_none_existing_xmp_produces_same_output(self) -> None:
        """existing_xmp_tree=None produces same output as before."""
        info: dict = {
            "title": "Test",
            "author": "Author",
            "subject": "Desc",
            "creator": "Tool",
            "producer": "Prod",
        }
        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        xmp_without = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            now=now,
        )
        xmp_with_none = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            now=now,
            existing_xmp_tree=None,
        )
        assert xmp_without == xmp_with_none

    def test_malformed_existing_xmp_does_not_break(self) -> None:
        """Malformed existing XMP tree doesn't break conversion."""
        # Build a tree that will cause _collect_preserved_elements to fail
        # by making something that looks like xmpmeta but has no rdf:Description
        malformed_xml = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><broken/></x:xmpmeta>'
        tree = etree.fromstring(malformed_xml)
        info: dict = {"title": "Test"}
        # Should not raise — falls through gracefully
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"Test" in xmp
        assert xmp.startswith(XMP_HEADER)

    def test_multiple_standards_preserved_together(self) -> None:
        """Multiple ISO standard identifications are all preserved."""
        tree = _build_xmp_with_extras(
            "<pdfuaid:part>1</pdfuaid:part>"
            "<pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>"
            "<pdfeid:part>1</pdfeid:part>"
        )
        info: dict = {"title": "Multi-standard"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        # All three should be present
        xmp_str = xmp.decode("utf-8")
        assert "PDF/X-4" in xmp_str
        # pdfuaid and pdfeid parts preserved
        parsed = etree.fromstring(
            xmp[xmp.index(b"<x:xmpmeta") : xmp.index(b"</x:xmpmeta>") + 13]
        )
        ns_rdf = NAMESPACES["rdf"]
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            pdfuaid_part = desc.find(f"{{{NAMESPACES['pdfuaid']}}}part")
            if pdfuaid_part is not None:
                assert pdfuaid_part.text == "1"
            pdfeid_part = desc.find(f"{{{NAMESPACES['pdfeid']}}}part")
            if pdfeid_part is not None:
                assert pdfeid_part.text == "1"

    def test_pdfa_identification_always_fresh(self) -> None:
        """PDF/A part and conformance are always the new values, not old."""
        tree = _build_xmp_with_extras("")  # old tree has part=2, conformance=B
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="U",
            existing_xmp_tree=tree,
        )
        parsed = etree.fromstring(
            xmp[xmp.index(b"<x:xmpmeta") : xmp.index(b"</x:xmpmeta>") + 13]
        )
        ns_rdf = NAMESPACES["rdf"]
        ns_pdfaid = NAMESPACES["pdfaid"]
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            part_elem = desc.find(f"{{{ns_pdfaid}}}part")
            if part_elem is not None:
                assert part_elem.text == "3"
            conf_elem = desc.find(f"{{{ns_pdfaid}}}conformance")
            if conf_elem is not None:
                assert conf_elem.text == "U"


class TestExtractExistingXmp:
    """Tests for _extract_existing_xmp helper."""

    def test_returns_none_for_pdf_without_metadata(
        self, sample_pdf_bytes: bytes
    ) -> None:
        """Returns None when PDF has no /Metadata stream."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        result = _extract_existing_xmp(pdf)
        assert result is None

    def test_returns_tree_for_pdf_with_xmp(self, sample_pdf_bytes: bytes) -> None:
        """Returns parsed tree when PDF has XMP metadata."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        # Embed XMP first
        info: dict = {"title": "Extraction Test"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        embed_xmp_metadata(pdf, xmp)

        result = _extract_existing_xmp(pdf)
        assert result is not None
        assert result.tag == f"{{{NAMESPACES['x']}}}xmpmeta"


class TestCollectPreservedElements:
    """Tests for _collect_preserved_elements helper."""

    def test_managed_elements_excluded(self) -> None:
        """Managed elements are not in the preserved list."""
        tree = _build_xmp_with_extras(
            f'<dc:title xmlns:dc="{NAMESPACES["dc"]}">'
            f'<rdf:Alt xmlns:rdf="{NAMESPACES["rdf"]}">'
            '<rdf:li xml:lang="x-default">Title</rdf:li>'
            "</rdf:Alt></dc:title>"
        )
        elems, attrs, ns = _collect_preserved_elements(tree)
        managed_tags = {e.tag for e in elems}
        assert f"{{{NAMESPACES['dc']}}}title" not in managed_tags

    def test_non_managed_elements_collected(self) -> None:
        """Non-managed elements are collected."""
        tree = _build_xmp_with_extras("<pdfuaid:part>1</pdfuaid:part>")
        elems, attrs, ns = _collect_preserved_elements(tree)
        tags = {e.tag for e in elems}
        assert f"{{{NAMESPACES['pdfuaid']}}}part" in tags

    def test_extra_namespaces_detected(self) -> None:
        """Unknown namespaces are detected and returned."""
        custom_ns = "http://example.com/test/ns/"
        tree = _build_xmp_with_extras(
            f'<myns:Prop xmlns:myns="{custom_ns}">value</myns:Prop>'
        )
        elems, attrs, ns = _collect_preserved_elements(tree)
        assert custom_ns in ns.values()


class TestSyncMetadataPreservation:
    """Integration tests for XMP preservation through sync_metadata."""

    def _make_pdf_with_xmp(
        self,
        tmp_dir: Path,
        extra_xmp_xml: str,
    ) -> Path:
        """Create a PDF with custom XMP metadata embedded."""
        pdf = new_pdf()
        page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, 612, 792]),
            )
        )
        pdf.pages.append(page)
        pdf.docinfo["/Title"] = "Test Title"

        # Build and embed custom XMP
        ns_rdf = NAMESPACES["rdf"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:dc="{NAMESPACES["dc"]}"
           xmlns:xmp="{NAMESPACES["xmp"]}"
           xmlns:pdf="{NAMESPACES["pdf"]}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfxid="{NAMESPACES["pdfxid"]}"
           xmlns:pdfeid="{NAMESPACES["pdfeid"]}"
           xmlns:pdfvtid="{NAMESPACES["pdfvtid"]}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <dc:title><rdf:Alt>
        <rdf:li xml:lang="x-default">Test Title</rdf:li>
      </rdf:Alt></dc:title>
      {extra_xmp_xml}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        xmp_bytes = XMP_HEADER + xmp_xml.encode("utf-8") + XMP_TRAILER

        metadata_stream = pikepdf.Stream(pdf, xmp_bytes)
        metadata_stream.Type = Name.Metadata
        metadata_stream.Subtype = Name.XML
        pdf.Root.Metadata = pdf.make_indirect(metadata_stream)

        pdf_path = tmp_dir / "xmp_test.pdf"
        pdf.save(pdf_path)

        return pdf_path

    def test_sync_preserves_pdfua(self, tmp_dir: Path) -> None:
        """sync_metadata preserves PDF/UA identification end-to-end."""
        pdf_path = self._make_pdf_with_xmp(
            tmp_dir,
            "<pdfuaid:part>1</pdfuaid:part>",
        )
        with Pdf.open(pdf_path) as pdf:
            sync_metadata(pdf, "2b")
            xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())

        # PDF/UA identification should still be present
        xml_start = xmp_bytes.index(b"<x:xmpmeta")
        xml_end = xmp_bytes.index(b"</x:xmpmeta>") + len(b"</x:xmpmeta>")
        tree = etree.fromstring(xmp_bytes[xml_start:xml_end])

        ns_rdf = NAMESPACES["rdf"]
        ns_pdfuaid = NAMESPACES["pdfuaid"]
        found = False
        for desc in tree.iter(f"{{{ns_rdf}}}Description"):
            elem = desc.find(f"{{{ns_pdfuaid}}}part")
            if elem is not None:
                assert elem.text == "1"
                found = True
        assert found, "pdfuaid:part not found in output XMP"

    def test_sync_preserves_pdfx(self, tmp_dir: Path) -> None:
        """sync_metadata preserves PDF/X identification end-to-end."""
        pdf_path = self._make_pdf_with_xmp(
            tmp_dir,
            "<pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>",
        )
        with Pdf.open(pdf_path) as pdf:
            sync_metadata(pdf, "2b")
            xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
        assert b"PDF/X-4" in xmp_bytes

    def test_sync_preserves_dc_rights(self, tmp_dir: Path) -> None:
        """sync_metadata preserves dc:rights end-to-end."""
        dc_rights_xml = (
            f'<dc:rights xmlns:dc="{NAMESPACES["dc"]}">'
            f'<rdf:Alt xmlns:rdf="{NAMESPACES["rdf"]}">'
            '<rdf:li xml:lang="x-default">Copyright 2024</rdf:li>'
            "</rdf:Alt></dc:rights>"
        )
        pdf_path = self._make_pdf_with_xmp(tmp_dir, dc_rights_xml)
        with Pdf.open(pdf_path) as pdf:
            sync_metadata(pdf, "2b")
            xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
        assert b"Copyright 2024" in xmp_bytes

    def test_sync_without_existing_xmp_works(self, sample_pdf_bytes: bytes) -> None:
        """sync_metadata still works fine when no existing XMP is present."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        sync_metadata(pdf, "2b")
        assert "/Metadata" in pdf.Root
        xmp_bytes = bytes(pdf.Root.Metadata.read_bytes())
        assert b"pdfaid:part" in xmp_bytes or b":part>" in xmp_bytes


def _parse_xmp_xml(xmp: bytes) -> etree._Element:
    """Extract and parse the x:xmpmeta element from XMP packet bytes."""
    xml_start = xmp.index(b"<x:xmpmeta")
    xml_end = xmp.index(b"</x:xmpmeta>") + len(b"</x:xmpmeta>")
    return etree.fromstring(xmp[xml_start:xml_end])


class TestExtensionSchemas:
    """Tests for PDF/A extension schema generation."""

    def test_no_extension_when_only_predefined(self) -> None:
        """Only predefined properties (dc/xmp/pdf/pdfaid) -> no pdfaExtension."""
        info: dict = {"title": "Test", "author": "Author", "producer": "Prod"}
        xmp = create_xmp_metadata(info, pdfa_part=2, pdfa_conformance="B")
        # No extension schema element should be present (xmlns declarations
        # are acceptable since extension NS are declared in the root nsmap)
        assert b"pdfaExtension:schemas" not in xmp

    def test_extension_for_pdfuaid(self) -> None:
        """pdfuaid:part in XMP triggers extension schema declaration."""
        tree = _build_xmp_with_extras("<pdfuaid:part>1</pdfuaid:part>")
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]

        # Find extension schemas
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                # Should contain pdfuaid namespace URI
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert NAMESPACES["pdfuaid"] in ext_xml
                assert "part" in ext_xml
                return
        pytest.fail("pdfaExtension:schemas not found in output XMP")

    def test_extension_for_pdfxid(self) -> None:
        """pdfxid:GTS_PDFXVersion triggers extension schema declaration."""
        tree = _build_xmp_with_extras(
            "<pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]

        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert NAMESPACES["pdfxid"] in ext_xml
                assert "GTS_PDFXVersion" in ext_xml
                return
        pytest.fail("pdfaExtension:schemas not found for pdfxid")

    def test_extension_for_pdfaid_rev(self) -> None:
        """pdfaid:rev (non-predefined in pdfaid) triggers extension."""
        ns_pdfaid = NAMESPACES["pdfaid"]
        tree = _build_xmp_with_extras(
            f'<pdfaid:rev xmlns:pdfaid="{ns_pdfaid}">2024</pdfaid:rev>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]

        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert ns_pdfaid in ext_xml
                assert "rev" in ext_xml
                # Should NOT declare 'part' or 'conformance' (those are predefined)
                # Only 'rev' should be in the property list
                props = ext.findall(f".//{{{_NS_PDFA_PROPERTY}}}name")
                prop_names = [p.text for p in props]
                assert "rev" in prop_names
                assert "part" not in prop_names
                assert "conformance" not in prop_names
                return
        pytest.fail("pdfaExtension:schemas not found for pdfaid:rev")

    def test_extension_for_unknown_namespace(self) -> None:
        """Unknown custom namespace gets a generic extension schema."""
        custom_ns = "http://example.com/custom/ns/"
        tree = _build_xmp_with_extras(
            f'<custom:Foo xmlns:custom="{custom_ns}">bar</custom:Foo>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]

        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert custom_ns in ext_xml
                assert "Foo" in ext_xml
                return
        pytest.fail("pdfaExtension:schemas not found for custom namespace")

    def test_extension_value_type_seq(self) -> None:
        """Property with rdf:Seq container gets valueType 'Seq Text'."""
        custom_ns = "http://example.com/custom/ns/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<custom:Items xmlns:custom="{custom_ns}">'
            f"<rdf:Seq><rdf:li>a</rdf:li></rdf:Seq>"
            f"</custom:Items>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert "Seq Text" in ext_xml
                return
        pytest.fail("Extension schema not found")

    def test_extension_value_type_bag(self) -> None:
        """Property with rdf:Bag container gets valueType 'Bag Text'."""
        custom_ns = "http://example.com/custom/ns/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<custom:Tags xmlns:custom="{custom_ns}">'
            f"<rdf:Bag><rdf:li>x</rdf:li></rdf:Bag>"
            f"</custom:Tags>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert "Bag Text" in ext_xml
                return
        pytest.fail("Extension schema not found")

    def test_extension_value_type_simple_text(self) -> None:
        """Simple text property gets valueType 'Text' (default)."""
        custom_ns = "http://example.com/custom/ns/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<custom:Foo xmlns:custom="{custom_ns}">bar</custom:Foo>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                # Find the valueType element
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert ">Text<" in ext_xml
                # Should NOT contain Seq/Bag/Alt
                assert "Seq Text" not in ext_xml
                assert "Bag Text" not in ext_xml
                assert "Alt Text" not in ext_xml
                return
        pytest.fail("Extension schema not found")

    def test_existing_extension_preserved_and_augmented(self) -> None:
        """Source extension schemas are stripped; fresh ones regenerated."""
        ns_rdf = NAMESPACES["rdf"]
        ns_ext = _NS_PDFA_EXTENSION
        ns_schema = _NS_PDFA_SCHEMA
        ns_prop = _NS_PDFA_PROPERTY
        # Build tree with existing extension for pdfuaid (complete)
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfxid="{NAMESPACES["pdfxid"]}"
           xmlns:pdfaExtension="{ns_ext}"
           xmlns:pdfaSchema="{ns_schema}"
           xmlns:pdfaProperty="{ns_prop}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>1</pdfuaid:part>
      <pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <pdfaSchema:schema>PDF/UA ID</pdfaSchema:schema>
            <pdfaSchema:namespaceURI>{NAMESPACES["pdfuaid"]}</pdfaSchema:namespaceURI>
            <pdfaSchema:prefix>pdfuaid</pdfaSchema:prefix>
            <pdfaSchema:property>
              <rdf:Seq>
                <rdf:li rdf:parseType="Resource">
                  <pdfaProperty:name>part</pdfaProperty:name>
                  <pdfaProperty:valueType>Integer</pdfaProperty:valueType>
                  <pdfaProperty:category>internal</pdfaProperty:category>
                  <pdfaProperty:description>PDF/UA version</pdfaProperty:description>
                </rdf:li>
              </rdf:Seq>
            </pdfaSchema:property>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{ns_ext}}}schemas")
            if ext is not None:
                ext_xml = etree.tostring(ext, encoding="unicode")
                # Freshly regenerated pdfuaid extension present
                assert NAMESPACES["pdfuaid"] in ext_xml
                assert "PDF/UA Universal Accessibility" in ext_xml
                # Missing pdfxid extension added
                assert NAMESPACES["pdfxid"] in ext_xml
                # Only one pdfaExtension:schemas element
                all_ext = desc.findall(f"{{{ns_ext}}}schemas")
                assert len(all_ext) == 1
                return
        pytest.fail("Extension schemas not found")

    def test_multiple_extensions(self) -> None:
        """Multiple non-predefined namespaces produce multiple schema entries."""
        tree = _build_xmp_with_extras(
            "<pdfuaid:part>1</pdfuaid:part>"
            "<pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]

        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is not None:
                bag = ext.find(f"{{{ns_rdf}}}Bag")
                assert bag is not None
                items = bag.findall(f"{{{ns_rdf}}}li")
                assert len(items) >= 2
                ext_xml = etree.tostring(ext, encoding="unicode")
                assert NAMESPACES["pdfuaid"] in ext_xml
                assert NAMESPACES["pdfxid"] in ext_xml
                return
        pytest.fail("Extension schemas not found for multiple namespaces")

    def test_extension_xml_structure(self) -> None:
        """Extension schema has correct XML hierarchy."""
        tree = _build_xmp_with_extras("<pdfuaid:part>1</pdfuaid:part>")
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]

        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{_NS_PDFA_EXTENSION}}}schemas")
            if ext is None:
                continue

            # schemas -> rdf:Bag -> rdf:li
            bag = ext.find(f"{{{ns_rdf}}}Bag")
            assert bag is not None, "Missing rdf:Bag under pdfaExtension:schemas"

            li = bag.find(f"{{{ns_rdf}}}li")
            assert li is not None, "Missing rdf:li under rdf:Bag"
            assert li.get(f"{{{ns_rdf}}}parseType") == "Resource"

            # Schema metadata elements
            schema = li.find(f"{{{_NS_PDFA_SCHEMA}}}schema")
            assert schema is not None and schema.text
            ns_uri = li.find(f"{{{_NS_PDFA_SCHEMA}}}namespaceURI")
            assert ns_uri is not None and ns_uri.text == NAMESPACES["pdfuaid"]
            prefix = li.find(f"{{{_NS_PDFA_SCHEMA}}}prefix")
            assert prefix is not None and prefix.text == "pdfuaid"

            # Property definitions
            prop_elem = li.find(f"{{{_NS_PDFA_SCHEMA}}}property")
            assert prop_elem is not None, "Missing pdfaSchema:property"
            seq = prop_elem.find(f"{{{ns_rdf}}}Seq")
            assert seq is not None, "Missing rdf:Seq under pdfaSchema:property"

            prop_li = seq.find(f"{{{ns_rdf}}}li")
            assert prop_li is not None
            assert prop_li.get(f"{{{ns_rdf}}}parseType") == "Resource"

            name = prop_li.find(f"{{{_NS_PDFA_PROPERTY}}}name")
            assert name is not None and name.text == "part"
            vt = prop_li.find(f"{{{_NS_PDFA_PROPERTY}}}valueType")
            assert vt is not None and vt.text == "Integer"
            cat = prop_li.find(f"{{{_NS_PDFA_PROPERTY}}}category")
            assert cat is not None and cat.text == "internal"
            desc_el = prop_li.find(f"{{{_NS_PDFA_PROPERTY}}}description")
            assert desc_el is not None and desc_el.text
            return

        pytest.fail("Extension schema structure not found")

    def test_bare_about_attribute_stripped(self) -> None:
        """Bare 'about' attribute is not preserved from source XMP."""
        ns_rdf = NAMESPACES["rdf"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}">
    <rdf:Description rdf:about="" about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        _, preserved_attrs, _ = _collect_preserved_elements(tree)
        assert "about" not in preserved_attrs

    def test_bare_rdf_structural_attrs_stripped(self) -> None:
        """Bare ID and nodeID attributes are not preserved from source XMP."""
        ns_rdf = NAMESPACES["rdf"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}">
    <rdf:Description rdf:about="" ID="x1" nodeID="n1">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        _, preserved_attrs, _ = _collect_preserved_elements(tree)
        assert "ID" not in preserved_attrs
        assert "nodeID" not in preserved_attrs

    def test_bare_about_not_in_output_xmp(self) -> None:
        """Output XMP has only rdf:about, no bare about attribute."""
        ns_rdf = NAMESPACES["rdf"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}">
    <rdf:Description rdf:about="" about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            assert "about" not in desc.attrib
            assert f"{{{ns_rdf}}}about" in desc.attrib

    def test_malformed_extension_schema_stripped(self) -> None:
        """Malformed source extension schemas are replaced with correct ones."""
        ns_rdf = NAMESPACES["rdf"]
        ns_ext = _NS_PDFA_EXTENSION
        ns_schema = _NS_PDFA_SCHEMA
        # Build XMP with a broken extension schema (wrong element names)
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfaExtension="{ns_ext}"
           xmlns:pdfaSchema="{ns_schema}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>1</pdfuaid:part>
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <pdfaSchema:schema>Broken Schema</pdfaSchema:schema>
            <pdfaSchema:namespaceURI>http://bogus.example.com/</pdfaSchema:namespaceURI>
            <pdfaSchema:prefix>bogus</pdfaSchema:prefix>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            ext = desc.find(f"{{{ns_ext}}}schemas")
            if ext is not None:
                ext_xml = etree.tostring(ext, encoding="unicode")
                # Broken schema should NOT be present
                assert "Broken Schema" not in ext_xml
                assert "bogus.example.com" not in ext_xml
                # Correct pdfuaid schema should be regenerated
                assert NAMESPACES["pdfuaid"] in ext_xml
                assert "PDF/UA Universal Accessibility" in ext_xml
                return
        pytest.fail("Extension schemas not found")

    def test_preserved_namespace_prefix_matches_extension_schema(self) -> None:
        """Extension schema prefix matches the actual XML namespace prefix."""
        ns_rdf = NAMESPACES["rdf"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <xmpDM:videoFrameRate>24</xmpDM:videoFrameRate>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        xmp = create_xmp_metadata(
            {"title": "Test"},
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ext_xml = etree.tostring(parsed, encoding="unicode")
        # Extension schema must use the actual prefix xmpDM, not a derived one
        assert ">xmpDM<" in ext_xml
        assert ">dynamicmedia<" not in ext_xml

    def test_source_extension_not_preserved_verbatim(self) -> None:
        """_collect_preserved_elements does not include pdfaExtension:schemas."""
        ns_rdf = NAMESPACES["rdf"]
        ns_ext = _NS_PDFA_EXTENSION
        ns_schema = _NS_PDFA_SCHEMA
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfaExtension="{ns_ext}"
           xmlns:pdfaSchema="{ns_schema}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>1</pdfuaid:part>
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <pdfaSchema:schema>PDF/UA ID</pdfaSchema:schema>
            <pdfaSchema:namespaceURI>{NAMESPACES["pdfuaid"]}</pdfaSchema:namespaceURI>
            <pdfaSchema:prefix>pdfuaid</pdfaSchema:prefix>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        preserved_elems, _, _ = _collect_preserved_elements(tree)
        ext_tag = f"{{{ns_ext}}}schemas"
        for elem in preserved_elems:
            assert elem.tag != ext_tag, (
                "pdfaExtension:schemas should not be in preserved elements"
            )


class TestNestedDescriptionExtension:
    """Tests for nested rdf:Description form in extension schemas."""

    def test_nested_description_extension_not_preserved(self) -> None:
        """pdfaSchema elements from nested Descriptions are not preserved."""
        ns_rdf = NAMESPACES["rdf"]
        ns_ext = _NS_PDFA_EXTENSION
        ns_schema = _NS_PDFA_SCHEMA
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfaExtension="{ns_ext}"
           xmlns:pdfaSchema="{ns_schema}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>1</pdfuaid:part>
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li>
            <rdf:Description
                pdfaSchema:schema="PDF/UA ID"
                pdfaSchema:namespaceURI="{NAMESPACES["pdfuaid"]}"
                pdfaSchema:prefix="pdfuaid">
            </rdf:Description>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        preserved_elems, preserved_attrs, _ = _collect_preserved_elements(tree)
        # No pdfaSchema attributes should leak through
        for attr_name in preserved_attrs:
            assert ns_schema not in attr_name, (
                f"pdfaSchema attribute leaked: {attr_name}"
            )
        # No pdfaSchema child elements should leak through
        for elem in preserved_elems:
            assert ns_schema not in elem.tag, f"pdfaSchema element leaked: {elem.tag}"

    def test_nested_description_namespace_no_pollution(self) -> None:
        """extra_namespaces should not contain duplicated schema prefixes."""
        ns_rdf = NAMESPACES["rdf"]
        ns_ext = _NS_PDFA_EXTENSION
        ns_schema = _NS_PDFA_SCHEMA
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfaExtension="{ns_ext}"
           xmlns:pdfaSchema="{ns_schema}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>1</pdfuaid:part>
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li>
            <rdf:Description
                pdfaSchema:schema="PDF/UA ID"
                pdfaSchema:namespaceURI="{NAMESPACES["pdfuaid"]}"
                pdfaSchema:prefix="pdfuaid">
            </rdf:Description>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        _, _, extra_ns = _collect_preserved_elements(tree)
        # No schema-related namespace should appear in extra_ns
        schema_prefixes = [
            p
            for p, u in extra_ns.items()
            if u in (ns_ext, ns_schema, _NS_PDFA_PROPERTY)
        ]
        assert schema_prefixes == [], (
            f"Extension schema NS leaked into extra_ns: {schema_prefixes}"
        )

    def test_nested_description_extension_correct_prefix(self) -> None:
        """E2E: output XMP uses pdfaSchema: not schema2: or similar."""
        ns_rdf = NAMESPACES["rdf"]
        ns_ext = _NS_PDFA_EXTENSION
        ns_schema = _NS_PDFA_SCHEMA
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfaExtension="{ns_ext}"
           xmlns:pdfaSchema="{ns_schema}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>1</pdfuaid:part>
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li>
            <rdf:Description
                pdfaSchema:schema="PDF/UA ID"
                pdfaSchema:namespaceURI="{NAMESPACES["pdfuaid"]}"
                pdfaSchema:prefix="pdfuaid">
            </rdf:Description>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        xmp = create_xmp_metadata(
            {"title": "Test"},
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        # Must use canonical prefix, not polluted variants
        assert "pdfaSchema:" in xmp_str
        assert "schema2:" not in xmp_str
        assert "schema1:" not in xmp_str

    def test_mixed_extension_forms_handled(self) -> None:
        """Mix of nested Description and parseType=Resource is handled."""
        ns_rdf = NAMESPACES["rdf"]
        ns_ext = _NS_PDFA_EXTENSION
        ns_schema = _NS_PDFA_SCHEMA
        ns_prop = _NS_PDFA_PROPERTY
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:pdfuaid="{NAMESPACES["pdfuaid"]}"
           xmlns:pdfaExtension="{ns_ext}"
           xmlns:pdfaSchema="{ns_schema}"
           xmlns:pdfaProperty="{ns_prop}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>1</pdfuaid:part>
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li>
            <rdf:Description
                pdfaSchema:schema="PDF/UA ID"
                pdfaSchema:namespaceURI="{NAMESPACES["pdfuaid"]}"
                pdfaSchema:prefix="pdfuaid">
            </rdf:Description>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <pdfaSchema:schema>PDF/A ID</pdfaSchema:schema>
            <pdfaSchema:namespaceURI>{NAMESPACES["pdfaid"]}</pdfaSchema:namespaceURI>
            <pdfaSchema:prefix>pdfaid</pdfaSchema:prefix>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        preserved_elems, preserved_attrs, extra_ns = _collect_preserved_elements(tree)
        # No extension schema data should leak
        for elem in preserved_elems:
            assert _NS_PDFA_SCHEMA not in elem.tag
        for attr_name in preserved_attrs:
            assert _NS_PDFA_SCHEMA not in attr_name
        # E2E: output uses canonical prefix
        xmp = create_xmp_metadata(
            {"title": "Test"},
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        assert "pdfaSchema:" in xmp_str
        assert "schema2:" not in xmp_str


class TestPreservationValidation:
    """Tests for validation of preserved XMP property structure/values."""

    def test_wrong_container_stripped(self) -> None:
        """dc:contributor as plain text (should be Bag) is stripped."""
        tree = _build_xmp_with_extras(
            f'<dc:contributor xmlns:dc="{NAMESPACES["dc"]}">John</dc:contributor>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"John" not in xmp

    def test_correct_container_preserved(self) -> None:
        """dc:contributor as rdf:Bag is preserved."""
        tree = _build_xmp_with_extras(
            f'<dc:contributor xmlns:dc="{NAMESPACES["dc"]}">'
            f'<rdf:Bag xmlns:rdf="{NAMESPACES["rdf"]}">'
            "<rdf:li>John</rdf:li>"
            "</rdf:Bag></dc:contributor>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"John" in xmp

    def test_simple_where_struct_expected_stripped(self) -> None:
        """xmpMM:DerivedFrom as text (should be struct) is stripped."""
        xmpmm = "http://ns.adobe.com/xap/1.0/mm/"
        tree = _build_xmp_with_extras(
            f'<xmpMM:DerivedFrom xmlns:xmpMM="{xmpmm}">doc123</xmpMM:DerivedFrom>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"doc123" not in xmp

    def test_invalid_integer_stripped(self) -> None:
        """tiff:ImageWidth with non-integer value is stripped."""
        tiff = "http://ns.adobe.com/tiff/1.0/"
        tree = _build_xmp_with_extras(
            f'<tiff:ImageWidth xmlns:tiff="{tiff}">256 mm</tiff:ImageWidth>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"256 mm" not in xmp

    def test_valid_integer_preserved(self) -> None:
        """tiff:ImageWidth with valid integer value is preserved."""
        tiff = "http://ns.adobe.com/tiff/1.0/"
        tree = _build_xmp_with_extras(
            f'<tiff:ImageWidth xmlns:tiff="{tiff}">4096</tiff:ImageWidth>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"4096" in xmp

    def test_empty_value_stripped(self) -> None:
        """exif:ExifVersion with only whitespace is stripped."""
        exif_ns = "http://ns.adobe.com/exif/1.0/"
        tree = _build_xmp_with_extras(
            f'<exif:ExifVersion xmlns:exif="{exif_ns}">   </exif:ExifVersion>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            assert desc.find(f"{{{exif_ns}}}ExifVersion") is None

    def test_non_predefined_resource_stripped(self) -> None:
        """Non-predefined property with Resource items is stripped."""
        xmpdm = "http://ns.adobe.com/xmp/1.0/DynamicMedia/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<xmpDM:markers xmlns:xmpDM="{xmpdm}">'
            f'<rdf:Bag xmlns:rdf="{ns_rdf}">'
            f'<rdf:li rdf:parseType="Resource">'
            f'<xmpDM:startTime xmlns:xmpDM="{xmpdm}">'
            "0</xmpDM:startTime>"
            "</rdf:li>"
            "</rdf:Bag></xmpDM:markers>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"markers" not in xmp

    def test_non_predefined_simple_preserved(self) -> None:
        """Non-predefined simple text property is preserved."""
        custom_ns = "http://example.com/custom/ns/"
        tree = _build_xmp_with_extras(
            f'<custom:Foo xmlns:custom="{custom_ns}">bar</custom:Foo>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"bar" in xmp

    def test_invalid_date_stripped(self) -> None:
        """photoshop:DateCreated with non-date prefix is stripped."""
        photoshop = "http://ns.adobe.com/photoshop/1.0/"
        tree = _build_xmp_with_extras(
            f'<photoshop:DateCreated xmlns:photoshop="{photoshop}">'
            "Date: 2016-02-01T13:19:21+01:00</photoshop:DateCreated>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"Date: 2016" not in xmp

    def test_valid_date_preserved(self) -> None:
        """photoshop:DateCreated with valid date is preserved."""
        photoshop = "http://ns.adobe.com/photoshop/1.0/"
        tree = _build_xmp_with_extras(
            f'<photoshop:DateCreated xmlns:photoshop="{photoshop}">'
            "2016-02-01T13:19:21+01:00</photoshop:DateCreated>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"2016-02-01T13:19:21+01:00" in xmp

    def test_invalid_boolean_stripped(self) -> None:
        """xmpRights:Marked with invalid Boolean is stripped."""
        xmprights = "http://ns.adobe.com/xap/1.0/rights/"
        tree = _build_xmp_with_extras(
            f'<xmpRights:Marked xmlns:xmpRights="{xmprights}">FALSE</xmpRights:Marked>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"FALSE" not in xmp

    def test_valid_boolean_preserved(self) -> None:
        """xmpRights:Marked with valid Boolean is preserved."""
        xmprights = "http://ns.adobe.com/xap/1.0/rights/"
        tree = _build_xmp_with_extras(
            f'<xmpRights:Marked xmlns:xmpRights="{xmprights}">True</xmpRights:Marked>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b">True<" in xmp

    def test_alt_without_lang_stripped(self) -> None:
        """xmpRights:UsageTerms Alt without xml:lang is stripped."""
        xmprights = "http://ns.adobe.com/xap/1.0/rights/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<xmpRights:UsageTerms xmlns:xmpRights="{xmprights}">'
            f'<rdf:Alt xmlns:rdf="{ns_rdf}">'
            "<rdf:li>SomeLangText</rdf:li>"
            "</rdf:Alt></xmpRights:UsageTerms>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"SomeLangText" not in xmp

    def test_alt_with_lang_preserved(self) -> None:
        """xmpRights:UsageTerms Alt with xml:lang is preserved."""
        xmprights = "http://ns.adobe.com/xap/1.0/rights/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<xmpRights:UsageTerms xmlns:xmpRights="{xmprights}">'
            f'<rdf:Alt xmlns:rdf="{ns_rdf}">'
            '<rdf:li xml:lang="x-default">Use freely</rdf:li>'
            "</rdf:Alt></xmpRights:UsageTerms>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"Use freely" in xmp

    def test_seq_integer_invalid_item_stripped(self) -> None:
        """tiff:BitsPerSample Seq with non-integer item is stripped."""
        tiff = "http://ns.adobe.com/tiff/1.0/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<tiff:BitsPerSample xmlns:tiff="{tiff}">'
            f'<rdf:Seq xmlns:rdf="{ns_rdf}">'
            "<rdf:li>8</rdf:li><rdf:li>8.0</rdf:li>"
            "</rdf:Seq></tiff:BitsPerSample>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"BitsPerSample" not in xmp

    def test_seq_integer_valid_items_preserved(self) -> None:
        """tiff:BitsPerSample Seq with valid integer items is preserved."""
        tiff = "http://ns.adobe.com/tiff/1.0/"
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<tiff:BitsPerSample xmlns:tiff="{tiff}">'
            f'<rdf:Seq xmlns:rdf="{ns_rdf}">'
            "<rdf:li>8</rdf:li><rdf:li>8</rdf:li><rdf:li>8</rdf:li>"
            "</rdf:Seq></tiff:BitsPerSample>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"BitsPerSample" in xmp

    def test_seq_date_invalid_item_stripped(self) -> None:
        """dc:date Seq with invalid date item is stripped."""
        ns_rdf = NAMESPACES["rdf"]
        tree = _build_xmp_with_extras(
            f'<dc:date xmlns:dc="{NAMESPACES["dc"]}">'
            f'<rdf:Seq xmlns:rdf="{ns_rdf}">'
            "<rdf:li>1997-07-16T19:20:15.45Z+01:00</rdf:li>"
            "</rdf:Seq></dc:date>"
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"1997-07-16" not in xmp

    def test_nested_rdf_description_not_promoted_and_identifier_stripped(self) -> None:
        """Nested rdf:Description values are not promoted to top-level properties."""
        ns_rdf = NAMESPACES["rdf"]
        ns_xmp = NAMESPACES["xmp"]
        ns_xmpidq = "http://ns.adobe.com/xmp/Identifier/qual/1.0/"
        tree = _build_xmp_with_extras(
            f'<xmp:Identifier xmlns:xmp="{ns_xmp}" xmlns:xmpidq="{ns_xmpidq}">'
            f'  <rdf:Bag xmlns:rdf="{ns_rdf}">'
            "    <rdf:li>"
            "      <rdf:Description>"
            "        <rdf:value>First value</rdf:value>"
            "        <xmpidq:Scheme>First name</xmpidq:Scheme>"
            "      </rdf:Description>"
            "    </rdf:li>"
            "  </rdf:Bag>"
            "</xmp:Identifier>"
        )
        xmp = create_xmp_metadata(
            {"title": "Test"},
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        root_desc = parsed.find(f".//{{{ns_rdf}}}RDF/{{{ns_rdf}}}Description")
        assert root_desc is not None
        assert root_desc.find(f"{{{ns_xmp}}}Identifier") is None
        assert root_desc.find(f"{{{ns_xmpidq}}}Scheme") is None


class TestExtensionSchemaValueValidation:
    """Tests for stripping invalid extension schema property values (6.6.2.3.1)."""

    def test_pdfaid_rev_empty_stripped(self) -> None:
        """pdfaid:rev with empty value is stripped (not a valid Integer)."""
        ns_rdf = NAMESPACES["rdf"]
        ns_pdfaid = NAMESPACES["pdfaid"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{ns_pdfaid}">
    <rdf:Description rdf:about="" pdfaid:rev="">
      <pdfaid:part>4</pdfaid:part>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        parsed = _parse_xmp_xml(xmp)
        ns_rdf = NAMESPACES["rdf"]
        for desc in parsed.iter(f"{{{ns_rdf}}}Description"):
            assert desc.get(f"{{{ns_pdfaid}}}rev") is None

    def test_pdfaid_rev_garbage_stripped(self) -> None:
        """pdfaid:rev with non-integer value '20_y' is stripped."""
        ns_rdf = NAMESPACES["rdf"]
        ns_pdfaid = NAMESPACES["pdfaid"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{ns_pdfaid}">
    <rdf:Description rdf:about="" pdfaid:rev="20_y">
      <pdfaid:part>4</pdfaid:part>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"20_y" not in xmp

    def test_pdfuaid_rev_non_integer_stripped(self) -> None:
        """pdfuaid:rev with '2024a' is stripped (not a valid Integer)."""
        ns_rdf = NAMESPACES["rdf"]
        ns_pdfuaid = NAMESPACES["pdfuaid"]
        ns_pdfaid = NAMESPACES["pdfaid"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfuaid="{ns_pdfuaid}"
           xmlns:pdfaid="{ns_pdfaid}">
    <rdf:Description rdf:about="" pdfuaid:rev="2024a">
      <pdfaid:part>3</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>2</pdfuaid:part>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"2024a" not in xmp

    def test_pdfuaid_rev_valid_integer_preserved(self) -> None:
        """pdfuaid:rev with valid integer '2024' is preserved."""
        ns_rdf = NAMESPACES["rdf"]
        ns_pdfuaid = NAMESPACES["pdfuaid"]
        ns_pdfaid = NAMESPACES["pdfaid"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfuaid="{ns_pdfuaid}"
           xmlns:pdfaid="{ns_pdfaid}">
    <rdf:Description rdf:about="" pdfuaid:rev="2024">
      <pdfaid:part>3</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <pdfuaid:part>2</pdfuaid:part>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"2024" in xmp

    def test_pdfaid_rev_element_invalid_stripped(self) -> None:
        """pdfaid:rev as element with non-integer value is stripped."""
        ns_pdfaid = NAMESPACES["pdfaid"]
        tree = _build_xmp_with_extras(
            f'<pdfaid:rev xmlns:pdfaid="{ns_pdfaid}">abc</pdfaid:rev>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"abc" not in xmp

    def test_pdfaid_rev_element_valid_preserved(self) -> None:
        """pdfaid:rev as element with valid integer value is preserved."""
        ns_pdfaid = NAMESPACES["pdfaid"]
        tree = _build_xmp_with_extras(
            f'<pdfaid:rev xmlns:pdfaid="{ns_pdfaid}">2024</pdfaid:rev>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        assert "2024" in xmp_str

    def test_pdfaid_corr_text_preserved(self) -> None:
        """pdfaid:corr (Text type) with any string value is preserved."""
        ns_pdfaid = NAMESPACES["pdfaid"]
        tree = _build_xmp_with_extras(
            f'<pdfaid:corr xmlns:pdfaid="{ns_pdfaid}">2</pdfaid:corr>'
        )
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        # "corr" with value "2" should appear
        assert "corr" in xmp_str

    def test_pdfaid_rev_attr_valid_integer_preserved(self) -> None:
        """pdfaid:rev attribute with valid integer is kept."""
        ns_rdf = NAMESPACES["rdf"]
        ns_pdfaid = NAMESPACES["pdfaid"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{ns_pdfaid}">
    <rdf:Description rdf:about="" pdfaid:rev="2020">
      <pdfaid:part>4</pdfaid:part>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=3,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        assert b"2020" in xmp


class TestNamespacePrefixSerialization:
    """Tests for correct namespace prefix serialization (6.6.2.3.1 fix)."""

    def test_pdfuaid_uses_canonical_prefix(self) -> None:
        """pdfuaid properties serialize with pdfuaid: prefix, not ns0:."""
        tree = _build_xmp_with_extras("<pdfuaid:part>2</pdfuaid:part>")
        info: dict = {"title": "Test"}
        xmp = create_xmp_metadata(
            info,
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        assert "pdfuaid:part" in xmp_str
        assert "ns0:" not in xmp_str

    def test_xmpmm_uses_canonical_prefix(self) -> None:
        """xmpMM properties serialize with xmpMM: prefix, not mm:."""
        ns_rdf = NAMESPACES["rdf"]
        ns_xmpmm = NAMESPACES["xmpMM"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:xmpMM="{ns_xmpmm}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <xmpMM:DocumentID>uuid:test-doc-id</xmpMM:DocumentID>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        xmp = create_xmp_metadata(
            {"title": "Test"},
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        assert "xmpMM:DocumentID" in xmp_str
        assert "mm:" not in xmp_str.lower().replace("xmpmm:", "")

    def test_duplicate_uri_single_prefix_entry(self) -> None:
        """Same namespace as element and attribute produces one prefix entry."""
        ns_rdf = NAMESPACES["rdf"]
        ns_xmpmm = NAMESPACES["xmpMM"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:xmpMM="{ns_xmpmm}">
    <rdf:Description rdf:about=""
                     xmpMM:InstanceID="uuid:inst-1">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <xmpMM:DocumentID>uuid:doc-1</xmpMM:DocumentID>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        _, _, extra_ns = _collect_preserved_elements(tree)
        # The xmpMM URI should appear at most once in extra_ns values
        xmpmm_count = list(extra_ns.values()).count(ns_xmpmm)
        assert xmpmm_count <= 1


class TestStructuralPropertyNormalization:
    """Tests for stEvt:When -> stEvt:when normalization."""

    def test_stevt_when_normalized_to_lowercase(self) -> None:
        """stEvt:When is corrected to stEvt:when in preserved elements."""
        ns_rdf = NAMESPACES["rdf"]
        ns_xmpmm = NAMESPACES["xmpMM"]
        ns_stevt = NAMESPACES["stEvt"]
        xmp_xml = f"""\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="{ns_rdf}"
           xmlns:pdfaid="{NAMESPACES["pdfaid"]}"
           xmlns:xmpMM="{ns_xmpmm}"
           xmlns:stEvt="{ns_stevt}">
    <rdf:Description rdf:about="">
      <pdfaid:part>2</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
      <xmpMM:History>
        <rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <stEvt:action>created</stEvt:action>
            <stEvt:When>2024-01-01T00:00:00Z</stEvt:When>
          </rdf:li>
        </rdf:Seq>
      </xmpMM:History>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        tree = etree.fromstring(xmp_xml.encode("utf-8"))
        xmp = create_xmp_metadata(
            {"title": "Test"},
            pdfa_part=2,
            pdfa_conformance="B",
            existing_xmp_tree=tree,
        )
        xmp_str = xmp.decode("utf-8")
        assert "stEvt:when" in xmp_str
        assert "stEvt:When" not in xmp_str

    def test_normalize_structural_properties_recursive(self) -> None:
        """_normalize_structural_properties fixes tags recursively."""
        ns_stevt = NAMESPACES["stEvt"]
        root = etree.Element("root")
        child = etree.SubElement(root, f"{{{ns_stevt}}}When")
        child.text = "2024-01-01"
        _normalize_structural_properties(root)
        assert child.tag == f"{{{ns_stevt}}}when"


class TestHasUndeclarableStructureExtended:
    """Tests for rdf:Description detection in _has_undeclarable_structure."""

    def test_explicit_rdf_description_child_is_undeclarable(self) -> None:
        """Property with rdf:Description child is detected as undeclarable."""
        ns_rdf = NAMESPACES["rdf"]
        elem = etree.fromstring(
            f'<foo:Prop xmlns:foo="http://example.com/foo/">'
            f'  <rdf:Description xmlns:rdf="{ns_rdf}">'
            f"    <foo:SubProp>value</foo:SubProp>"
            f"  </rdf:Description>"
            f"</foo:Prop>"
        )
        assert _has_undeclarable_structure(elem) is True

    def test_rdf_description_in_seq_item_is_undeclarable(self) -> None:
        """Property with rdf:Description inside Seq/li is undeclarable."""
        ns_rdf = NAMESPACES["rdf"]
        elem = etree.fromstring(
            f'<foo:Prop xmlns:foo="http://example.com/foo/"'
            f'          xmlns:rdf="{ns_rdf}">'
            f"  <rdf:Seq>"
            f"    <rdf:li>"
            f"      <rdf:Description>"
            f"        <foo:SubProp>val</foo:SubProp>"
            f"      </rdf:Description>"
            f"    </rdf:li>"
            f"  </rdf:Seq>"
            f"</foo:Prop>"
        )
        assert _has_undeclarable_structure(elem) is True

    def test_simple_property_not_undeclarable(self) -> None:
        """Simple text property is not undeclarable."""
        elem = etree.fromstring(
            '<foo:Prop xmlns:foo="http://example.com/foo/">simple</foo:Prop>'
        )
        assert _has_undeclarable_structure(elem) is False


class TestParseXmpBytes:
    """Tests for _parse_xmp_bytes."""

    def test_valid_xmp_with_packet_wrapper(self) -> None:
        """XMP with packet wrapper is parsed correctly."""
        xmp = (
            b'<?xpacket begin="\xef\xbb\xbf"'
            b' id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b"<root>hello</root>"
            b"</x:xmpmeta>\n"
            b'<?xpacket end="w"?>'
        )
        tree = _parse_xmp_bytes(xmp)
        assert tree is not None
        assert tree.tag == "{adobe:ns:meta/}xmpmeta"

    def test_valid_xmp_without_packet_wrapper(self) -> None:
        """Bare XML is parsed correctly."""
        xmp = b"<root><child>text</child></root>"
        tree = _parse_xmp_bytes(xmp)
        assert tree is not None
        assert tree.tag == "root"

    def test_malformed_xml_returns_none(self) -> None:
        """Malformed XML returns None."""
        assert _parse_xmp_bytes(b"<not-well-formed") is None

    def test_empty_bytes_returns_none(self) -> None:
        """Empty bytes return None."""
        assert _parse_xmp_bytes(b"") is None

    def test_whitespace_only_returns_none(self) -> None:
        """Whitespace-only bytes return None."""
        assert _parse_xmp_bytes(b"   \n  ") is None

    def test_empty_packet_returns_none(self) -> None:
        """Packet wrapper with empty content returns None."""
        xmp = (
            b'<?xpacket begin="\xef\xbb\xbf"'
            b' id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            b"   \n"
            b'<?xpacket end="w"?>'
        )
        assert _parse_xmp_bytes(xmp) is None


class TestReserializeXmp:
    """Tests for _reserialize_xmp."""

    def test_roundtrip_preserves_content(self) -> None:
        """Re-serialized XMP contains the original content."""
        tree = etree.fromstring(
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><child>text</child></x:xmpmeta>'
        )
        result = _reserialize_xmp(tree)
        assert result.startswith(XMP_HEADER)
        assert result.endswith(XMP_TRAILER)
        assert b"<child>text</child>" in result

    def test_output_is_utf8(self) -> None:
        """Re-serialized output is UTF-8 encoded (no XML declaration)."""
        tree = etree.fromstring(b"<root>hello</root>")
        result = _reserialize_xmp(tree)
        # No XML declaration (would confuse parsers after xpacket PI)
        assert b"<?xml" not in result
        # Content is valid UTF-8
        inner = result.replace(XMP_HEADER, b"").replace(XMP_TRAILER, b"").strip()
        inner.decode("utf-8")


class TestSanitizeNonCatalogMetadata:
    """Tests for _sanitize_non_catalog_metadata."""

    def test_removes_malformed_metadata(self, sample_pdf_bytes: bytes) -> None:
        """Malformed metadata stream is removed."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        bad = pikepdf.Stream(pdf, b"<broken xml!!!")
        bad.Type = Name.Metadata
        bad.Subtype = Name.XML
        pdf.pages[0].obj["/Metadata"] = pdf.make_indirect(bad)

        sanitized, removed = _sanitize_non_catalog_metadata(pdf)
        assert removed == 1
        assert sanitized == 0
        assert "/Metadata" not in pdf.pages[0].obj

    def test_preserves_valid_metadata(self, sample_pdf_bytes: bytes) -> None:
        """Valid XMP metadata stream is preserved."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        valid_xmp = (
            b'<?xpacket begin="\xef\xbb\xbf"'
            b' id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b"<rdf:RDF xmlns:rdf="
            b'"http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""/>'
            b"</rdf:RDF>"
            b"</x:xmpmeta>\n"
            b'<?xpacket end="w"?>'
        )
        meta = pikepdf.Stream(pdf, valid_xmp)
        meta.Type = Name.Metadata
        meta.Subtype = Name.XML
        pdf.pages[0].obj["/Metadata"] = pdf.make_indirect(meta)

        sanitized, removed = _sanitize_non_catalog_metadata(pdf)
        assert sanitized == 1
        assert removed == 0
        assert "/Metadata" in pdf.pages[0].obj

    def test_ensures_uncompressed_after_sanitize(
        self,
        sample_pdf_bytes: bytes,
    ) -> None:
        """After sanitization, metadata has no Filter (PDF/A requirement)."""
        import zlib
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        valid_xmp = b"<root>valid</root>"
        # Create a genuinely compressed stream
        compressed = zlib.compress(valid_xmp)
        meta = pikepdf.Stream(pdf, compressed)
        meta.Type = Name.Metadata
        meta.Subtype = Name.XML
        meta[Name.Filter] = Name.FlateDecode
        ref = pdf.make_indirect(meta)
        pdf.pages[0].obj["/Metadata"] = ref

        sanitized, removed = _sanitize_non_catalog_metadata(pdf)
        assert sanitized == 1
        assert removed == 0

        page_meta = pdf.get_object(ref.objgen)
        assert Name.Filter not in page_meta
        # Re-serialized content should be readable
        content = bytes(page_meta.read_bytes())
        assert b"root" in content

    def test_does_not_touch_catalog_metadata(
        self,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Catalog /Metadata is left untouched."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))
        # Set up catalog metadata
        cat_xmp = b"<catalog>original</catalog>"
        cat_meta = pikepdf.Stream(pdf, cat_xmp)
        cat_meta.Type = Name.Metadata
        cat_meta.Subtype = Name.XML
        pdf.Root.Metadata = pdf.make_indirect(cat_meta)

        sanitized, removed = _sanitize_non_catalog_metadata(pdf)
        assert sanitized == 0
        assert removed == 0
        # Catalog metadata still present
        content = bytes(pdf.Root.Metadata.read_bytes())
        assert b"<catalog>original</catalog>" in content

    def test_mixed_valid_and_malformed(self, sample_pdf_bytes: bytes) -> None:
        """Mix of valid and malformed streams handled correctly."""
        from io import BytesIO

        pdf = open_pdf(BytesIO(sample_pdf_bytes))

        # Add valid metadata to page
        valid_xmp = b"<root>good</root>"
        meta_good = pikepdf.Stream(pdf, valid_xmp)
        meta_good.Type = Name.Metadata
        meta_good.Subtype = Name.XML
        pdf.pages[0].obj["/Metadata"] = pdf.make_indirect(meta_good)

        # Add a dictionary with malformed metadata
        bad_dict = Dictionary()
        bad_meta = pikepdf.Stream(pdf, b"not xml at all <<<")
        bad_meta.Type = Name.Metadata
        bad_meta.Subtype = Name.XML
        bad_dict["/Metadata"] = pdf.make_indirect(bad_meta)
        bad_dict["/Type"] = Name("/XObject")
        pdf.make_indirect(bad_dict)

        sanitized, removed = _sanitize_non_catalog_metadata(pdf)
        assert sanitized == 1
        assert removed == 1
        assert "/Metadata" in pdf.pages[0].obj


class TestExtractExtensionSchemaBlocks:
    """Tests for _extract_extension_schema_blocks."""

    def test_empty_xmp_returns_empty(self) -> None:
        """XMP without extension schemas returns empty dict."""
        tree = etree.fromstring(
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""/>'
            b"</rdf:RDF></x:xmpmeta>"
        )
        assert _extract_extension_schema_blocks(tree) == {}

    def test_extracts_schema_block(self) -> None:
        """Schema blocks are extracted keyed by namespace URI."""
        xmp = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"'
            b' xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"'
            b' xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">'
            b"<pdfaExtension:schemas><rdf:Bag>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaSchema:schema>Test</pdfaSchema:schema>"
            b"<pdfaSchema:namespaceURI>http://example.com/ns/</pdfaSchema:namespaceURI>"
            b"<pdfaSchema:prefix>ex</pdfaSchema:prefix>"
            b"</rdf:li>"
            b"</rdf:Bag></pdfaExtension:schemas>"
            b"</rdf:Description>"
            b"</rdf:RDF></x:xmpmeta>"
        )
        tree = etree.fromstring(xmp)
        blocks = _extract_extension_schema_blocks(tree)
        assert "http://example.com/ns/" in blocks
        # Block should be an rdf:li element
        li = blocks["http://example.com/ns/"]
        ns_uri_tag = f"{{{_NS_PDFA_SCHEMA}}}namespaceURI"
        assert li.find(ns_uri_tag).text == "http://example.com/ns/"


class TestCollectNonCatalogExtensionNeeds:
    """Tests for _collect_non_catalog_extension_needs."""

    def test_no_non_catalog_metadata(self, sample_pdf_bytes: bytes) -> None:
        """PDF without non-catalog metadata returns empty dict."""
        from io import BytesIO

        pdf = pikepdf.open(BytesIO(sample_pdf_bytes))
        result = _collect_non_catalog_extension_needs(pdf)
        assert result == {}

    def test_collects_custom_namespace(self, sample_pdf_bytes: bytes) -> None:
        """Non-catalog XMP with custom namespace properties are collected."""
        from io import BytesIO

        pdf = pikepdf.open(BytesIO(sample_pdf_bytes))
        xmp_data = (
            b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:custom="http://example.com/custom/">'
            b"<custom:Foo>bar</custom:Foo>"
            b"</rdf:Description>"
            b"</rdf:RDF></x:xmpmeta>"
            b'<?xpacket end="w"?>'
        )
        meta_stream = pikepdf.Stream(pdf, xmp_data)
        meta_stream.Type = Name.Metadata
        meta_stream.Subtype = Name.XML
        pdf.pages[0].obj["/Metadata"] = pdf.make_indirect(meta_stream)

        result = _collect_non_catalog_extension_needs(pdf)
        assert "http://example.com/custom/" in result
        assert "Foo" in result["http://example.com/custom/"]

    def test_collects_even_with_own_extensions(
        self,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Non-catalog XMP with own extensions is still collected.

        Custom valueTypes in non-catalog extensions may depend on
        catalog-level definitions.
        """
        from io import BytesIO

        pdf = pikepdf.open(BytesIO(sample_pdf_bytes))
        xmp_data = (
            b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"'
            b' xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"'
            b' xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">'
            b"<pdfaExtension:schemas><rdf:Bag>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaSchema:schema>My</pdfaSchema:schema>"
            b"<pdfaSchema:namespaceURI>http://example.com/ns/</pdfaSchema:namespaceURI>"
            b"<pdfaSchema:prefix>ex</pdfaSchema:prefix>"
            b"</rdf:li>"
            b"</rdf:Bag></pdfaExtension:schemas>"
            b"</rdf:Description>"
            b'<rdf:Description rdf:about=""'
            b' xmlns:ex="http://example.com/ns/">'
            b"<ex:Prop>value</ex:Prop>"
            b"</rdf:Description>"
            b"</rdf:RDF></x:xmpmeta>"
            b'<?xpacket end="w"?>'
        )
        meta_stream = pikepdf.Stream(pdf, xmp_data)
        meta_stream.Type = Name.Metadata
        meta_stream.Subtype = Name.XML
        pdf.pages[0].obj["/Metadata"] = pdf.make_indirect(meta_stream)

        result = _collect_non_catalog_extension_needs(pdf)
        assert "http://example.com/ns/" in result
        assert "Prop" in result["http://example.com/ns/"]


class TestNonCatalogExtensionInCatalogXMP:
    """Tests for non-catalog extension schemas in catalog XMP generation."""

    def test_extension_schemas_for_non_catalog_properties(self) -> None:
        """create_xmp_metadata includes extension schemas for non-catalog needs."""
        info = {"title": "Test", "author": "Author"}
        extra = {"http://example.com/ns/": {"MyProp"}}
        xmp_bytes = create_xmp_metadata(
            info,
            2,
            "U",
            non_catalog_extension_needs=extra,
        )
        xmp_str = xmp_bytes.decode("utf-8")
        assert "http://example.com/ns/" in xmp_str
        assert "MyProp" in xmp_str
        assert "pdfaExtension:schemas" in xmp_str

    def test_original_schema_blocks_preserved(self) -> None:
        """Original extension schema blocks with custom valueTypes are reused."""
        # Build an existing XMP tree with custom valueType
        existing_xmp = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"'
            b' xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"'
            b' xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#"'
            b' xmlns:pdfaType="http://www.aiim.org/pdfa/ns/type#"'
            b' xmlns:pdfaField="http://www.aiim.org/pdfa/ns/field#">'
            b"<pdfaExtension:schemas><rdf:Bag>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaSchema:schema>Custom</pdfaSchema:schema>"
            b"<pdfaSchema:namespaceURI>http://example.com/ns/</pdfaSchema:namespaceURI>"
            b"<pdfaSchema:prefix>ex</pdfaSchema:prefix>"
            b"<pdfaSchema:property><rdf:Seq>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaProperty:name>MyProp</pdfaProperty:name>"
            b"<pdfaProperty:valueType>MyCustomType</pdfaProperty:valueType>"
            b"<pdfaProperty:category>external</pdfaProperty:category>"
            b"<pdfaProperty:description>test</pdfaProperty:description>"
            b"</rdf:li>"
            b"</rdf:Seq></pdfaSchema:property>"
            b"<pdfaSchema:valueType><rdf:Seq>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaType:type>MyCustomType</pdfaType:type>"
            b"<pdfaType:namespaceURI>http://example.com/types/</pdfaType:namespaceURI>"
            b"<pdfaType:prefix>extype</pdfaType:prefix>"
            b"<pdfaType:description>custom type</pdfaType:description>"
            b"</rdf:li>"
            b"</rdf:Seq></pdfaSchema:valueType>"
            b"</rdf:li>"
            b"</rdf:Bag></pdfaExtension:schemas>"
            b"</rdf:Description>"
            b"</rdf:RDF></x:xmpmeta>"
        )
        existing_tree = etree.fromstring(existing_xmp)

        info = {"title": "Test", "author": "Author"}
        extra = {"http://example.com/ns/": {"MyProp"}}
        xmp_bytes = create_xmp_metadata(
            info,
            2,
            "U",
            existing_xmp_tree=existing_tree,
            non_catalog_extension_needs=extra,
        )
        xmp_str = xmp_bytes.decode("utf-8")
        # Original custom valueType should be preserved
        assert "MyCustomType" in xmp_str
        assert "http://example.com/types/" in xmp_str

    def test_no_extension_when_not_needed(self) -> None:
        """No extension schemas when non_catalog_extension_needs is None."""
        info = {"title": "Test", "author": "Author"}
        xmp_bytes = create_xmp_metadata(info, 2, "B")
        xmp_str = xmp_bytes.decode("utf-8")
        assert "pdfaExtension:schemas" not in xmp_str

    def test_malformed_original_block_regenerated(self) -> None:
        """Malformed original extension schema block is dropped and regenerated."""
        # Build source XMP with a pdfaExtension:schemas entry that is malformed
        # (property missing pdfaProperty:category)
        existing_xmp = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"'
            b' xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"'
            b' xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">'
            b"<pdfaExtension:schemas><rdf:Bag>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaSchema:schema>Custom</pdfaSchema:schema>"
            b"<pdfaSchema:namespaceURI>http://example.com/custom/</pdfaSchema:namespaceURI>"
            b"<pdfaSchema:prefix>custom</pdfaSchema:prefix>"
            b"<pdfaSchema:property><rdf:Seq>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaProperty:name>Foo</pdfaProperty:name>"
            b"<pdfaProperty:valueType>Text</pdfaProperty:valueType>"
            # pdfaProperty:category intentionally missing
            b"<pdfaProperty:description>Foo property</pdfaProperty:description>"
            b"</rdf:li>"
            b"</rdf:Seq></pdfaSchema:property>"
            b"</rdf:li>"
            b"</rdf:Bag></pdfaExtension:schemas>"
            b"</rdf:Description>"
            b"</rdf:RDF></x:xmpmeta>"
        )
        existing_tree = etree.fromstring(existing_xmp)

        info = {"title": "Test"}
        extra = {"http://example.com/custom/": {"Foo"}}
        xmp_bytes = create_xmp_metadata(
            info,
            2,
            "B",
            existing_xmp_tree=existing_tree,
            non_catalog_extension_needs=extra,
        )
        parsed = _parse_xmp_xml(xmp_bytes)
        ns_rdf = NAMESPACES["rdf"]

        # Find the Foo property entry in the output extension schema
        category_tag = f"{{{_NS_PDFA_PROPERTY}}}category"
        name_tag = f"{{{_NS_PDFA_PROPERTY}}}name"
        for li in parsed.iter(f"{{{ns_rdf}}}li"):
            name_elem = li.find(name_tag)
            if name_elem is not None and name_elem.text == "Foo":
                cat_elem = li.find(category_tag)
                # Freshly generated block uses "external" as default category
                assert cat_elem is not None
                assert cat_elem.text == "external"
                return
        pytest.fail("Foo property not found in output extension schema")

    def test_valid_original_block_preserved(self) -> None:
        """Valid original extension schema block is reused (preserves valueType)."""
        existing_xmp = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about=""'
            b' xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"'
            b' xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"'
            b' xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">'
            b"<pdfaExtension:schemas><rdf:Bag>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaSchema:schema>Custom</pdfaSchema:schema>"
            b"<pdfaSchema:namespaceURI>http://example.com/custom/</pdfaSchema:namespaceURI>"
            b"<pdfaSchema:prefix>custom</pdfaSchema:prefix>"
            b"<pdfaSchema:property><rdf:Seq>"
            b'<rdf:li rdf:parseType="Resource">'
            b"<pdfaProperty:name>Foo</pdfaProperty:name>"
            b"<pdfaProperty:valueType>Date</pdfaProperty:valueType>"
            b"<pdfaProperty:category>internal</pdfaProperty:category>"
            b"<pdfaProperty:description>Foo property</pdfaProperty:description>"
            b"</rdf:li>"
            b"</rdf:Seq></pdfaSchema:property>"
            b"</rdf:li>"
            b"</rdf:Bag></pdfaExtension:schemas>"
            b"</rdf:Description>"
            b"</rdf:RDF></x:xmpmeta>"
        )
        existing_tree = etree.fromstring(existing_xmp)

        info = {"title": "Test"}
        extra = {"http://example.com/custom/": {"Foo"}}
        xmp_bytes = create_xmp_metadata(
            info,
            2,
            "B",
            existing_xmp_tree=existing_tree,
            non_catalog_extension_needs=extra,
        )
        parsed = _parse_xmp_xml(xmp_bytes)
        ns_rdf = NAMESPACES["rdf"]

        value_type_tag = f"{{{_NS_PDFA_PROPERTY}}}valueType"
        name_tag = f"{{{_NS_PDFA_PROPERTY}}}name"
        for li in parsed.iter(f"{{{ns_rdf}}}li"):
            name_elem = li.find(name_tag)
            if name_elem is not None and name_elem.text == "Foo":
                vt_elem = li.find(value_type_tag)
                # Original block declared Date; fresh generation would use Text
                assert vt_elem is not None
                assert vt_elem.text == "Date"
                return
        pytest.fail("Foo property not found in output extension schema")


def _make_valid_schema_li(
    uri: str = "http://example.com/ns/",
    schema_name: str = "Test",
    prefix: str = "ex",
    prop_name: str = "MyProp",
    value_type: str = "Text",
    category: str = "external",
    description: str = "My property",
) -> etree._Element:
    """Build a fully valid extension schema rdf:li element."""
    ns_rdf = NAMESPACES["rdf"]
    li = etree.Element(f"{{{ns_rdf}}}li")
    li.set(f"{{{ns_rdf}}}parseType", "Resource")

    etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}schema").text = schema_name
    etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}namespaceURI").text = uri
    etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}prefix").text = prefix

    prop_elem = etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}property")
    seq = etree.SubElement(prop_elem, f"{{{ns_rdf}}}Seq")
    prop_li = etree.SubElement(seq, f"{{{ns_rdf}}}li")
    prop_li.set(f"{{{ns_rdf}}}parseType", "Resource")
    etree.SubElement(prop_li, f"{{{_NS_PDFA_PROPERTY}}}name").text = prop_name
    etree.SubElement(prop_li, f"{{{_NS_PDFA_PROPERTY}}}valueType").text = value_type
    etree.SubElement(prop_li, f"{{{_NS_PDFA_PROPERTY}}}category").text = category
    etree.SubElement(prop_li, f"{{{_NS_PDFA_PROPERTY}}}description").text = description

    return li


def _add_value_type_entry(
    schema_li: etree._Element,
    *,
    type_name: str = "MyCustomType",
    namespace_uri: str = "http://example.com/types/",
    prefix: str = "extype",
    description: str = "custom type",
) -> etree._Element:
    """Append one valid pdfaSchema:valueType rdf:li entry and return it."""
    ns_rdf = NAMESPACES["rdf"]
    value_type_tag = f"{{{_NS_PDFA_SCHEMA}}}valueType"
    seq_tag = f"{{{ns_rdf}}}Seq"
    li_tag = f"{{{ns_rdf}}}li"

    value_type_elem = schema_li.find(value_type_tag)
    if value_type_elem is None:
        value_type_elem = etree.SubElement(schema_li, value_type_tag)

    seq = value_type_elem.find(seq_tag)
    if seq is None:
        seq = etree.SubElement(value_type_elem, seq_tag)

    value_type_li = etree.SubElement(seq, li_tag)
    value_type_li.set(f"{{{ns_rdf}}}parseType", "Resource")
    etree.SubElement(value_type_li, f"{{{_NS_PDFA_TYPE}}}type").text = type_name
    etree.SubElement(
        value_type_li, f"{{{_NS_PDFA_TYPE}}}namespaceURI"
    ).text = namespace_uri
    etree.SubElement(value_type_li, f"{{{_NS_PDFA_TYPE}}}prefix").text = prefix
    etree.SubElement(
        value_type_li, f"{{{_NS_PDFA_TYPE}}}description"
    ).text = description

    return value_type_li


class TestSanitizeExtensionSchemaBlocks:
    """Tests for _sanitize_extension_schema_blocks."""

    def test_well_formed_block_preserved(self) -> None:
        """A fully valid block passes through unchanged."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri)
        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri in result

    def test_missing_schema_name_drops_block(self) -> None:
        """Block without pdfaSchema:schema is dropped."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri)
        schema_elem = li.find(f"{{{_NS_PDFA_SCHEMA}}}schema")
        li.remove(schema_elem)
        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri not in result

    def test_missing_prefix_drops_block(self) -> None:
        """Block without pdfaSchema:prefix is dropped."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri)
        prefix_elem = li.find(f"{{{_NS_PDFA_SCHEMA}}}prefix")
        li.remove(prefix_elem)
        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri not in result

    def test_missing_property_element_drops_block(self) -> None:
        """Block without any pdfaSchema:property child is dropped."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri)
        prop_elem = li.find(f"{{{_NS_PDFA_SCHEMA}}}property")
        li.remove(prop_elem)
        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri not in result

    def test_invalid_category_removes_property(self) -> None:
        """Property with invalid category is removed.

        Block is kept when at least one valid property remains.
        """
        uri = "http://example.com/ns/"
        ns_rdf = NAMESPACES["rdf"]
        li = _make_valid_schema_li(uri=uri, prop_name="GoodProp")

        # Add a second property with bad category
        seq = li.find(f"{{{_NS_PDFA_SCHEMA}}}property").find(f"{{{ns_rdf}}}Seq")
        bad_li = etree.SubElement(seq, f"{{{ns_rdf}}}li")
        bad_li.set(f"{{{ns_rdf}}}parseType", "Resource")
        etree.SubElement(bad_li, f"{{{_NS_PDFA_PROPERTY}}}name").text = "BadProp"
        etree.SubElement(bad_li, f"{{{_NS_PDFA_PROPERTY}}}valueType").text = "Text"
        etree.SubElement(bad_li, f"{{{_NS_PDFA_PROPERTY}}}category").text = "wrong"
        etree.SubElement(bad_li, f"{{{_NS_PDFA_PROPERTY}}}description").text = "desc"

        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri in result
        # Only GoodProp should remain
        seq_out = (
            result[uri].find(f"{{{_NS_PDFA_SCHEMA}}}property").find(f"{{{ns_rdf}}}Seq")
        )
        names = [
            e.find(f"{{{_NS_PDFA_PROPERTY}}}name").text
            for e in seq_out.findall(f"{{{ns_rdf}}}li")
        ]
        assert "GoodProp" in names
        assert "BadProp" not in names

    def test_missing_value_type_removes_property(self) -> None:
        """Property missing pdfaProperty:valueType is removed."""
        uri = "http://example.com/ns/"
        ns_rdf = NAMESPACES["rdf"]
        li = _make_valid_schema_li(uri=uri, prop_name="GoodProp")

        seq = li.find(f"{{{_NS_PDFA_SCHEMA}}}property").find(f"{{{ns_rdf}}}Seq")
        bad_li = etree.SubElement(seq, f"{{{ns_rdf}}}li")
        bad_li.set(f"{{{ns_rdf}}}parseType", "Resource")
        etree.SubElement(bad_li, f"{{{_NS_PDFA_PROPERTY}}}name").text = "BadProp"
        # valueType intentionally omitted
        etree.SubElement(bad_li, f"{{{_NS_PDFA_PROPERTY}}}category").text = "external"
        etree.SubElement(bad_li, f"{{{_NS_PDFA_PROPERTY}}}description").text = "desc"

        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri in result
        seq_out = (
            result[uri].find(f"{{{_NS_PDFA_SCHEMA}}}property").find(f"{{{ns_rdf}}}Seq")
        )
        names = [
            e.find(f"{{{_NS_PDFA_PROPERTY}}}name").text
            for e in seq_out.findall(f"{{{ns_rdf}}}li")
        ]
        assert "GoodProp" in names
        assert "BadProp" not in names

    def test_all_properties_malformed_drops_block(self) -> None:
        """Block whose every property is invalid is dropped after Seq becomes empty."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri, prop_name="BadProp")

        # Make the single property malformed by removing its valueType
        ns_rdf = NAMESPACES["rdf"]
        seq = li.find(f"{{{_NS_PDFA_SCHEMA}}}property").find(f"{{{ns_rdf}}}Seq")
        prop_li = seq.find(f"{{{ns_rdf}}}li")
        vt = prop_li.find(f"{{{_NS_PDFA_PROPERTY}}}valueType")
        prop_li.remove(vt)

        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri not in result

    def test_value_type_without_seq_removed(self) -> None:
        """pdfaSchema:valueType without rdf:Seq is removed."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri)
        etree.SubElement(li, f"{{{_NS_PDFA_SCHEMA}}}valueType")

        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri in result
        assert result[uri].find(f"{{{_NS_PDFA_SCHEMA}}}valueType") is None

    @pytest.mark.parametrize(
        ("field_tag", "field_label"),
        [
            (f"{{{_NS_PDFA_TYPE}}}type", "type"),
            (f"{{{_NS_PDFA_TYPE}}}namespaceURI", "namespaceURI"),
            (f"{{{_NS_PDFA_TYPE}}}prefix", "prefix"),
            (f"{{{_NS_PDFA_TYPE}}}description", "description"),
        ],
    )
    def test_missing_required_value_type_field_removes_entry(
        self,
        field_tag: str,
        field_label: str,
    ) -> None:
        """Missing required pdfaType fields remove only malformed ValueType entries."""
        uri = "http://example.com/ns/"
        ns_rdf = NAMESPACES["rdf"]
        li = _make_valid_schema_li(uri=uri)

        _add_value_type_entry(li, type_name="GoodType")
        bad_entry = _add_value_type_entry(li, type_name="BadType")
        missing_field = bad_entry.find(field_tag)
        assert missing_field is not None
        bad_entry.remove(missing_field)

        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri in result, f"schema block unexpectedly dropped for {field_label}"

        value_type_elem = result[uri].find(f"{{{_NS_PDFA_SCHEMA}}}valueType")
        assert value_type_elem is not None
        seq = value_type_elem.find(f"{{{ns_rdf}}}Seq")
        assert seq is not None

        entries = seq.findall(f"{{{ns_rdf}}}li")
        assert len(entries) == 1
        type_elem = entries[0].find(f"{{{_NS_PDFA_TYPE}}}type")
        assert type_elem is not None
        assert type_elem.text == "GoodType"

    def test_invalid_value_type_field_structure_removed(self) -> None:
        """pdfaType:field without rdf:Seq is removed from an otherwise valid entry."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri)
        value_type_entry = _add_value_type_entry(li, type_name="TypeWithBadField")

        field_elem = etree.SubElement(value_type_entry, f"{{{_NS_PDFA_TYPE}}}field")
        field_elem.text = "not-a-seq"

        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri in result

        ns_rdf = NAMESPACES["rdf"]
        value_type_elem = result[uri].find(f"{{{_NS_PDFA_SCHEMA}}}valueType")
        assert value_type_elem is not None
        seq = value_type_elem.find(f"{{{ns_rdf}}}Seq")
        assert seq is not None
        entry = seq.find(f"{{{ns_rdf}}}li")
        assert entry is not None
        assert entry.find(f"{{{_NS_PDFA_TYPE}}}field") is None

    def test_value_type_removed_when_all_entries_invalid(self) -> None:
        """pdfaSchema:valueType is removed when all ValueType entries are invalid."""
        uri = "http://example.com/ns/"
        li = _make_valid_schema_li(uri=uri)

        bad_entry = _add_value_type_entry(li, type_name="BadType")
        prefix_elem = bad_entry.find(f"{{{_NS_PDFA_TYPE}}}prefix")
        assert prefix_elem is not None
        bad_entry.remove(prefix_elem)

        result = _sanitize_extension_schema_blocks({uri: li})
        assert uri in result
        assert result[uri].find(f"{{{_NS_PDFA_SCHEMA}}}valueType") is None

    def test_no_schemas_returns_empty(self) -> None:
        """Empty input returns empty result."""
        assert _sanitize_extension_schema_blocks({}) == {}
