# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for selective embedded file removal (PDF/A-2 compliance)."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.converter import ConversionResult, convert_to_pdfa
from pdftopdfa.sanitizers import sanitize_for_pdfa
from pdftopdfa.sanitizers.files import (
    _is_pdfa_compliant_embedded,
    _iter_all_filespecs,
    _iter_all_filespecs_by_scan,
    _iter_name_tree_pairs,
    _iter_name_tree_values,
    ensure_af_relationships,
    ensure_embedded_file_params,
    ensure_embedded_file_subtypes,
    ensure_filespec_desc,
    ensure_filespec_uf_entries,
    remove_embedded_files,
    remove_non_compliant_embedded_files,
    sanitize_embedded_file_filters,
)
from pdftopdfa.utils import resolve_indirect as _resolve_indirect
from pdftopdfa.verapdf import VeraPDFResult

# --- Test helpers ---


def _create_pdfa_xmp(part: int, conformance: str) -> bytes:
    """Create minimal XMP metadata with PDF/A identification."""
    return (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        b'<rdf:Description rdf:about=""\n'
        b'  xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">\n'
        b"  <pdfaid:part>" + str(part).encode() + b"</pdfaid:part>\n"
        b"  <pdfaid:conformance>" + conformance.encode() + b"</pdfaid:conformance>\n"
        b"</rdf:Description>\n"
        b"</rdf:RDF>\n"
        b"</x:xmpmeta>\n"
        b'<?xpacket end="w"?>'
    )


def _create_pdfa_pdf_bytes(level: str) -> bytes:
    """Create a minimal PDF with PDF/A XMP metadata and return as bytes."""
    part = int(level[0])
    conformance = level[1].upper()

    pdf = new_pdf()
    pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

    xmp_data = _create_pdfa_xmp(part, conformance)
    xmp_stream = pdf.make_stream(xmp_data)
    xmp_stream["/Type"] = Name.Metadata
    xmp_stream["/Subtype"] = Name("/XML")
    pdf.Root["/Metadata"] = xmp_stream

    buf = BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def _create_encrypted_pdf_bytes() -> bytes:
    """Create an encrypted PDF that can be opened with an empty user password."""
    pdf = new_pdf()
    pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
    buf = BytesIO()
    pdf.save(buf, encryption=pikepdf.Encryption(owner="owner-password"))
    return buf.getvalue()


def _create_signed_pdf_bytes() -> bytes:
    """Create a non-conformant PDF containing a live signature field."""
    pdf = new_pdf()
    pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
    signature = pdf.make_indirect(
        Dictionary(
            Type=Name.Sig,
            Filter=Name("/Adobe.PPKLite"),
            SubFilter=Name("/adbe.pkcs7.detached"),
            ByteRange=Array([0, 100, 200, 300]),
            Contents=pdf.make_stream(b"\x00" * 64),
        )
    )
    field = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            FT=Name.Sig,
            Rect=Array([0, 0, 200, 50]),
            V=signature,
        )
    )
    pdf.pages[0]["/Annots"] = Array([field])
    pdf.Root["/AcroForm"] = pdf.make_indirect(
        Dictionary(Fields=Array([field]), SigFlags=1)
    )
    buf = BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def _make_pdf_with_embedded(data: bytes, filename: str = "test.pdf") -> Pdf:
    """Create a PDF with a single embedded file."""
    pdf = new_pdf()
    pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

    ef_stream = pdf.make_stream(data)
    ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
    file_spec = Dictionary(
        Type=Name.Filespec,
        F=filename,
        UF=filename,
        EF=ef_dict,
    )
    embedded = Dictionary(Names=Array([filename, file_spec]))
    names = Dictionary(EmbeddedFiles=embedded)
    pdf.Root.Names = names

    return pdf


def _make_pdf_with_file_attachment(data: bytes, filename: str = "test.pdf") -> Pdf:
    """Create a PDF with a FileAttachment annotation."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page))
    pdf.pages.append(page)

    ef_stream = pdf.make_stream(data)
    ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
    file_spec = Dictionary(
        Type=Name.Filespec,
        F=filename,
        UF=filename,
        EF=ef_dict,
    )
    annot = Dictionary(
        Type=Name.Annot,
        Subtype=Name.FileAttachment,
        Rect=Array([0, 0, 100, 100]),
        FS=file_spec,
    )
    pdf.pages[0]["/Annots"] = Array([annot])

    return pdf


# --- Tests for _is_pdfa_compliant_embedded ---


class TestIsPdfaCompliantEmbedded:
    """Tests for _is_pdfa_compliant_embedded."""

    def test_pdfa_1b_is_compliant(self) -> None:
        """PDF/A-1b embedded file is considered compliant."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        host = _make_pdf_with_embedded(pdf_data)
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]
        assert _is_pdfa_compliant_embedded(filespec) is True

    def test_pdfa_2b_is_compliant(self) -> None:
        """PDF/A-2b embedded file is considered compliant."""
        pdf_data = _create_pdfa_pdf_bytes("2b")
        host = _make_pdf_with_embedded(pdf_data)
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]
        assert _is_pdfa_compliant_embedded(filespec) is True

    def test_pdfa_3b_is_not_compliant(self) -> None:
        """PDF/A-3b embedded file is NOT compliant for PDF/A-2."""
        pdf_data = _create_pdfa_pdf_bytes("3b")
        host = _make_pdf_with_embedded(pdf_data)
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]
        assert _is_pdfa_compliant_embedded(filespec) is False

    def test_non_pdf_data_is_not_compliant(self) -> None:
        """Non-PDF data (plain text) is not compliant."""
        host = _make_pdf_with_embedded(b"Hello, this is a text file.")
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]
        assert _is_pdfa_compliant_embedded(filespec) is False

    def test_normal_pdf_without_pdfa_is_not_compliant(self) -> None:
        """A normal PDF without PDF/A metadata is not compliant."""
        # Create a plain PDF without PDF/A XMP metadata
        plain_pdf = new_pdf()
        plain_pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        buf = BytesIO()
        plain_pdf.save(buf)

        host = _make_pdf_with_embedded(buf.getvalue())
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]
        assert _is_pdfa_compliant_embedded(filespec) is False

    def test_corrupt_data_is_not_compliant(self) -> None:
        """Corrupt/random data is not compliant."""
        host = _make_pdf_with_embedded(b"%PDF-1.4\x00\x01\x02GARBAGE")
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]
        assert _is_pdfa_compliant_embedded(filespec) is False

    def test_filespec_without_ef_is_not_compliant(self) -> None:
        """FileSpec without /EF dictionary is not compliant."""
        filespec = Dictionary(Type=Name.Filespec, F="test.txt")
        assert _is_pdfa_compliant_embedded(filespec) is False

    def test_empty_stream_is_not_compliant(self) -> None:
        """FileSpec with empty embedded stream is not compliant."""
        pdf = new_pdf()
        ef_stream = pdf.make_stream(b"")
        ef_dict = Dictionary(F=ef_stream)
        filespec = Dictionary(Type=Name.Filespec, F="test.pdf", EF=ef_dict)
        assert _is_pdfa_compliant_embedded(filespec) is False


# --- Tests for _is_pdfa_compliant_embedded with veraPDF ---

_VERAPDF_AVAILABLE = "pdftopdfa.sanitizers.files.is_verapdf_available"
_VERAPDF_VALIDATE = "pdftopdfa.sanitizers.files.validate_with_verapdf"


@pytest.fixture(autouse=True)
def mock_embedded_verapdf():
    """Keep embedded-file tests independent of a local veraPDF installation."""
    with (
        patch(_VERAPDF_AVAILABLE, return_value=True),
        patch(
            _VERAPDF_VALIDATE,
            side_effect=lambda _path, flavour: VeraPDFResult(
                compliant=True, flavour=flavour
            ),
        ),
    ):
        yield


class TestIsPdfaCompliantEmbeddedVeraPDF:
    """Tests for _is_pdfa_compliant_embedded with veraPDF integration."""

    def test_verapdf_available_and_compliant(self) -> None:
        """veraPDF confirms compliance -> True."""
        pdf_data = _create_pdfa_pdf_bytes("2b")
        host = _make_pdf_with_embedded(pdf_data)
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]

        with (
            patch(_VERAPDF_AVAILABLE, return_value=True),
            patch(
                _VERAPDF_VALIDATE,
                return_value=VeraPDFResult(compliant=True, flavour="2b"),
            ) as mock_validate,
        ):
            assert _is_pdfa_compliant_embedded(filespec) is True
            mock_validate.assert_called_once()

    def test_verapdf_available_and_non_compliant(self) -> None:
        """veraPDF says non-compliant despite XMP claiming PDF/A -> False."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        host = _make_pdf_with_embedded(pdf_data)
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]

        with (
            patch(_VERAPDF_AVAILABLE, return_value=True),
            patch(
                _VERAPDF_VALIDATE,
                return_value=VeraPDFResult(compliant=False, flavour="1b"),
            ),
        ):
            assert _is_pdfa_compliant_embedded(filespec) is False

    def test_verapdf_not_available_does_not_trust_xmp(self) -> None:
        """An XMP claim is unverified when veraPDF is unavailable."""
        pdf_data = _create_pdfa_pdf_bytes("2b")
        host = _make_pdf_with_embedded(pdf_data)
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]

        with (
            patch(_VERAPDF_AVAILABLE, return_value=False),
            patch(_VERAPDF_VALIDATE) as mock_validate,
        ):
            assert _is_pdfa_compliant_embedded(filespec) is False
            mock_validate.assert_not_called()

    def test_verapdf_error_does_not_trust_xmp(self) -> None:
        """A veraPDF execution failure leaves the XMP claim unverified."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        host = _make_pdf_with_embedded(pdf_data)
        names_array = host.Root.Names.EmbeddedFiles.Names
        filespec = names_array[1]

        with (
            patch(_VERAPDF_AVAILABLE, return_value=True),
            patch(
                _VERAPDF_VALIDATE,
                side_effect=RuntimeError("veraPDF crashed"),
            ),
        ):
            assert _is_pdfa_compliant_embedded(filespec) is False


# --- Tests for remove_non_compliant_embedded_files ---


class TestRemoveNonCompliantEmbeddedFiles:
    """Tests for remove_non_compliant_embedded_files."""

    def test_keeps_compliant_file(self) -> None:
        """Compliant PDF/A-1b embedded file is kept."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_embedded(pdf_data, "compliant.pdf")

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 0,
            "kept": 1,
            "converted": 0,
            "conversion_failed": 0,
        }
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_removes_non_compliant_file(self) -> None:
        """Non-compliant embedded file is removed."""
        pdf = _make_pdf_with_embedded(b"Not a PDF at all", "bad.txt")

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 1,
            "kept": 0,
            "converted": 0,
            "conversion_failed": 0,
        }
        # EmbeddedFiles should be deleted since all files were removed
        names = pdf.Root.Names
        names = _resolve_indirect(names)
        assert "/EmbeddedFiles" not in names

    def test_mixed_compliant_and_non_compliant(self) -> None:
        """Keeps compliant files and removes non-compliant ones."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Create one compliant and one non-compliant embedded file
        compliant_data = _create_pdfa_pdf_bytes("2b")
        compliant_stream = pdf.make_stream(compliant_data)
        compliant_ef = Dictionary(F=compliant_stream, UF=compliant_stream)
        compliant_fs = Dictionary(
            Type=Name.Filespec, F="good.pdf", UF="good.pdf", EF=compliant_ef
        )

        bad_stream = pdf.make_stream(b"Not a PDF")
        bad_ef = Dictionary(F=bad_stream, UF=bad_stream)
        bad_fs = Dictionary(Type=Name.Filespec, F="bad.txt", UF="bad.txt", EF=bad_ef)

        embedded = Dictionary(
            Names=Array(["good.pdf", compliant_fs, "bad.txt", bad_fs])
        )
        names = Dictionary(EmbeddedFiles=embedded)
        pdf.Root.Names = names

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 1,
            "kept": 1,
            "converted": 0,
            "conversion_failed": 0,
        }
        # EmbeddedFiles should still exist with one entry
        remaining = pdf.Root.Names.EmbeddedFiles.Names
        assert len(remaining) == 2  # [name, filespec]
        assert str(remaining[0]) == "good.pdf"

    def test_empty_pdf_returns_zero(self) -> None:
        """PDF without embedded files returns zeroes."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 0,
            "kept": 0,
            "converted": 0,
            "conversion_failed": 0,
        }

    def test_removes_non_compliant_file_attachment(self) -> None:
        """Non-compliant FileAttachment annotation is removed."""
        pdf = _make_pdf_with_file_attachment(b"Not a PDF", "bad.txt")

        result = remove_non_compliant_embedded_files(pdf)

        assert result["removed"] == 1
        assert result["kept"] == 0
        # Annotation should be removed
        annots = pdf.pages[0].get("/Annots")
        assert annots is None

    def test_keeps_compliant_file_attachment(self) -> None:
        """Compliant FileAttachment annotation is kept."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_file_attachment(pdf_data, "good.pdf")

        result = remove_non_compliant_embedded_files(pdf)

        assert result["removed"] == 0
        assert result["kept"] == 1
        # Annotation should be kept
        annots = pdf.pages[0].get("/Annots")
        assert annots is not None
        assert len(annots) == 1


# --- Tests for conversion of non-compliant embedded PDFs ---

_TRY_CONVERT = "pdftopdfa.sanitizers.files._try_convert_embedded_pdf_to_pdfa2"


class TestConvertNonCompliantEmbeddedFiles:
    """Tests for the convert-before-remove path (rule 6.8-5)."""

    def test_unverified_pdfa_claim_uses_conversion_path(self) -> None:
        """An XMP-only PDF/A claim cannot bypass attachment conversion."""
        pdf_data = _create_pdfa_pdf_bytes("2b")
        pdf = _make_pdf_with_embedded(pdf_data, "claimed-pdfa.pdf")
        converted_bytes = b"%PDF-1.4 verified conversion"

        with (
            patch(_VERAPDF_AVAILABLE, return_value=False),
            patch(_TRY_CONVERT, return_value=converted_bytes) as mock_convert,
        ):
            result = remove_non_compliant_embedded_files(pdf)

        mock_convert.assert_called_once_with(pdf_data)
        assert result == {
            "removed": 0,
            "kept": 0,
            "converted": 1,
            "conversion_failed": 0,
        }

    def test_converts_non_compliant_pdf_to_pdfa2(self) -> None:
        """Non-compliant embedded PDF is converted to PDF/A-2b instead of removed."""
        pdf = _make_pdf_with_embedded(b"%PDF-1.4 non-compliant data", "doc.pdf")

        converted_bytes = b"%PDF-1.4 converted content"
        with patch(_TRY_CONVERT, return_value=converted_bytes):
            result = remove_non_compliant_embedded_files(pdf)

        assert result["converted"] == 1
        assert result["removed"] == 0
        assert result["kept"] == 0
        # EmbeddedFiles must still exist (file was converted, not removed)
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_converts_reopened_indirect_name_tree_filespec_once(self) -> None:
        """An indirect Name Tree FileSpec is not converted again by the scan."""
        pdf = _make_pdf_with_embedded(b"%PDF-1.4 non-compliant", "doc.pdf")
        names_array = pdf.Root.Names.EmbeddedFiles.Names
        names_array[1] = pdf.make_indirect(names_array[1])
        buffer = BytesIO()
        pdf.save(buffer)
        pdf.close()
        buffer.seek(0)

        with pikepdf.open(buffer) as reopened:
            with patch(
                _TRY_CONVERT, return_value=b"%PDF-1.4 converted"
            ) as mock_convert:
                result = remove_non_compliant_embedded_files(reopened)

        mock_convert.assert_called_once()
        assert result == {
            "removed": 0,
            "kept": 0,
            "converted": 1,
            "conversion_failed": 0,
        }

    def test_shared_indirect_name_tree_and_annotation_checked_once(self) -> None:
        """One indirect FileSpec in two locations is validated only once."""
        pdf = _make_pdf_with_embedded(b"%PDF-1.4 attachment", "doc.pdf")
        names_array = pdf.Root.Names.EmbeddedFiles.Names
        filespec = pdf.make_indirect(names_array[1])
        names_array[1] = filespec
        pdf.pages[0]["/Annots"] = Array(
            [
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.FileAttachment,
                    Rect=Array([0, 0, 100, 100]),
                    FS=filespec,
                )
            ]
        )

        with patch(
            "pdftopdfa.sanitizers.files._is_pdfa_compliant_embedded",
            return_value=True,
        ) as mock_check:
            result = remove_non_compliant_embedded_files(pdf)

        mock_check.assert_called_once()
        assert result["kept"] == 1

    def test_removes_non_pdf_embedded_file(self) -> None:
        """Non-PDF embedded file (no %PDF- prefix) is removed without conversion."""
        pdf = _make_pdf_with_embedded(b"Not a PDF at all", "bad.txt")

        with patch(_TRY_CONVERT) as mock_convert:
            result = remove_non_compliant_embedded_files(pdf)
            mock_convert.assert_not_called()

        assert result["removed"] == 1
        assert result["converted"] == 0

    def test_preserves_pdf_when_conversion_fails(self) -> None:
        """When conversion returns None, the original PDF attachment is kept."""
        pdf = _make_pdf_with_embedded(b"%PDF-1.4 unconvertible", "bad.pdf")

        with patch(_TRY_CONVERT, return_value=None):
            result = remove_non_compliant_embedded_files(pdf)

        assert result["removed"] == 0
        assert result["converted"] == 0
        assert result["conversion_failed"] == 1
        names_array = pdf.Root.Names.EmbeddedFiles.Names
        filespec = _resolve_indirect(names_array[1])
        stream = _resolve_indirect(_resolve_indirect(filespec.EF).F)
        assert bytes(stream.read_bytes()) == b"%PDF-1.4 unconvertible"

    def test_return_dict_has_converted_key(self) -> None:
        """Return dict always contains 'converted' key."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        result = remove_non_compliant_embedded_files(pdf)

        assert "converted" in result
        assert "removed" in result
        assert "kept" in result
        assert isinstance(result["converted"], int)

    def test_converted_file_stream_is_updated(self) -> None:
        """After conversion the embedded stream contains the new PDF/A-2b bytes."""
        pdf = _make_pdf_with_embedded(b"%PDF-1.4 original content", "doc.pdf")

        new_data = b"%PDF-1.4 converted content"
        with patch(_TRY_CONVERT, return_value=new_data):
            result = remove_non_compliant_embedded_files(pdf)

        assert result["converted"] == 1
        names_array = pdf.Root.Names.EmbeddedFiles.Names
        filespec = _resolve_indirect(names_array[1])
        ef = _resolve_indirect(filespec.get("/EF"))
        stream_obj = ef.get("/UF") or ef.get("/F")
        stream = _resolve_indirect(stream_obj)
        assert bytes(stream.read_bytes()) == new_data

    def test_sanitize_for_pdfa_exposes_converted_count(self) -> None:
        """sanitize_for_pdfa result dict has 'embedded_files_converted' key."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "bad.txt")

        result = sanitize_for_pdfa(pdf, level="2b")

        assert "embedded_files_converted" in result
        assert isinstance(result["embedded_files_converted"], int)

    def test_openable_encrypted_embedded_pdf_is_converted(self) -> None:
        """An encrypted input with an empty user password is converted."""
        from pdftopdfa.sanitizers.files import (
            _try_convert_embedded_pdf_to_pdfa2,
        )

        with patch(
            "pdftopdfa.sanitizers.files.is_verapdf_available",
            return_value=False,
        ):
            converted = _try_convert_embedded_pdf_to_pdfa2(
                _create_encrypted_pdf_bytes()
            )

        assert converted is not None
        with Pdf.open(BytesIO(converted)) as converted_pdf:
            assert converted_pdf.is_encrypted is False
            assert converted_pdf.Root["/Metadata"]["/Type"] == Name.Metadata

    def test_signed_non_compliant_embedded_pdf_is_preserved(self) -> None:
        """A signed input skipped by conversion is retained and reported."""
        pdf = _make_pdf_with_embedded(_create_signed_pdf_bytes(), "signed.pdf")

        with patch(
            "pdftopdfa.sanitizers.files.is_verapdf_available",
            return_value=False,
        ):
            result = remove_non_compliant_embedded_files(pdf)

        assert result["converted"] == 0
        assert result["removed"] == 0
        assert result["conversion_failed"] == 1
        assert "/EmbeddedFiles" in _resolve_indirect(pdf.Root.Names)

    def test_failed_attachment_conversion_marks_outer_result_invalid(
        self, tmp_path: Path
    ) -> None:
        """An invalid outer candidate is published while the source stays intact."""
        input_path = tmp_path / "input.pdf"
        output_path = tmp_path / "output.pdf"
        pdf = _make_pdf_with_embedded(b"%PDF-1.4 unconvertible", "bad.pdf")
        pdf.save(input_path)

        with patch(_TRY_CONVERT, return_value=None):
            result = convert_to_pdfa(input_path, output_path, level="2b")

        assert result.success is False
        assert result.validation_failed is True
        assert any("attachment" in warning for warning in result.warnings)
        assert any("published despite" in warning for warning in result.warnings)
        assert output_path.is_file()
        with pikepdf.open(output_path) as output_pdf:
            assert "/EmbeddedFiles" in output_pdf.Root.Names
        with pikepdf.open(input_path) as source_pdf:
            names_array = source_pdf.Root.Names.EmbeddedFiles.Names
            filespec = _resolve_indirect(names_array[1])
            stream = _resolve_indirect(_resolve_indirect(filespec.EF).F)
            assert bytes(stream.read_bytes()) == b"%PDF-1.4 unconvertible"

    def test_successful_skipped_result_is_rejected(self) -> None:
        """success=True does not accept an unchanged skipped output."""
        from pdftopdfa.sanitizers.files import (
            _try_convert_embedded_pdf_to_pdfa2,
        )

        def fake_convert(input_path: Path, output_path: Path, **kwargs: object):
            output_path.write_bytes(input_path.read_bytes())
            return ConversionResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                level="2b",
                skipped=True,
            )

        with patch("pdftopdfa.converter.convert_to_pdfa", side_effect=fake_convert):
            converted = _try_convert_embedded_pdf_to_pdfa2(b"%PDF-1.4 unchanged")

        assert converted is None

    def test_validation_failed_result_is_rejected(self) -> None:
        """A conversion result reporting failed validation is rejected."""
        from pdftopdfa.sanitizers.files import (
            _try_convert_embedded_pdf_to_pdfa2,
        )

        def fake_convert(input_path: Path, output_path: Path, **kwargs: object):
            output_path.write_bytes(b"%PDF-1.7 invalid PDF/A")
            return ConversionResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                level="2b",
                validation_failed=True,
            )

        with patch("pdftopdfa.converter.convert_to_pdfa", side_effect=fake_convert):
            converted = _try_convert_embedded_pdf_to_pdfa2(b"%PDF-1.4 input")

        assert converted is None

    def test_successful_validated_embedded_pdf_is_retained(self) -> None:
        """A converted attachment is returned after PDF/A-2b validation."""
        from pdftopdfa.sanitizers.files import (
            _try_convert_embedded_pdf_to_pdfa2,
        )

        converted_data = b"%PDF-1.7 converted PDF/A-2b"

        def fake_convert(input_path: Path, output_path: Path, **kwargs: object):
            output_path.write_bytes(converted_data)
            return ConversionResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                level="2b",
            )

        validation = VeraPDFResult(compliant=True, flavour="2b")
        with (
            patch("pdftopdfa.converter.convert_to_pdfa", side_effect=fake_convert),
            patch(
                "pdftopdfa.sanitizers.files.is_verapdf_available",
                return_value=True,
            ),
            patch(
                "pdftopdfa.sanitizers.files.validate_with_verapdf",
                return_value=validation,
            ) as mock_validate,
        ):
            converted = _try_convert_embedded_pdf_to_pdfa2(b"%PDF-1.4 input")

        assert converted == converted_data
        assert mock_validate.call_args.kwargs["flavour"] == "2b"


class TestEmbeddedPdfConversionDepthLimit:
    """Tests for the recursion depth limit of embedded PDF conversion."""

    def test_conversion_skipped_at_max_depth(self) -> None:
        """At the maximum nesting depth, conversion is refused without work."""
        from pdftopdfa.sanitizers import files as files_mod

        token = files_mod._embedded_pdf_conversion_depth.set(
            files_mod._MAX_EMBEDDED_PDF_CONVERSION_DEPTH
        )
        try:
            with patch("pdftopdfa.converter.convert_to_pdfa") as mock_convert:
                result = files_mod._try_convert_embedded_pdf_to_pdfa2(
                    b"%PDF-1.4 nested data"
                )
                mock_convert.assert_not_called()
            assert result is None
        finally:
            files_mod._embedded_pdf_conversion_depth.reset(token)

    def test_conversion_increments_and_resets_depth(self) -> None:
        """The depth is incremented during conversion and reset afterwards."""
        from pdftopdfa.sanitizers import files as files_mod

        seen_depths: list[int] = []

        def fake_convert(*args: object, **kwargs: object) -> None:
            seen_depths.append(files_mod._embedded_pdf_conversion_depth.get())
            raise RuntimeError("stop conversion")

        with patch("pdftopdfa.converter.convert_to_pdfa", side_effect=fake_convert):
            result = files_mod._try_convert_embedded_pdf_to_pdfa2(b"%PDF-1.4 data")

        assert result is None
        assert seen_depths == [1]
        assert files_mod._embedded_pdf_conversion_depth.get() == 0


# --- Integration tests for sanitize_for_pdfa ---


class TestSanitizeForPdfaEmbeddedFiles:
    """Integration tests for sanitize_for_pdfa with embedded files."""

    def test_level_2b_keeps_compliant_embedded(self) -> None:
        """Level 2b keeps PDF/A-compliant embedded files."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_embedded(pdf_data, "compliant.pdf")

        result = sanitize_for_pdfa(pdf, level="2b")

        assert result["files_removed"] == 0
        assert result["embedded_files_kept"] == 1
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_level_2u_keeps_compliant_embedded(self) -> None:
        """Level 2u keeps PDF/A-compliant embedded files."""
        pdf_data = _create_pdfa_pdf_bytes("2b")
        pdf = _make_pdf_with_embedded(pdf_data, "compliant.pdf")

        result = sanitize_for_pdfa(pdf, level="2u")

        assert result["files_removed"] == 0
        assert result["embedded_files_kept"] == 1
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_level_2b_removes_non_compliant(self) -> None:
        """Level 2b removes non-compliant embedded files."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "bad.txt")

        result = sanitize_for_pdfa(pdf, level="2b")

        assert result["files_removed"] == 1
        assert result["embedded_files_kept"] == 0

    def test_level_2a_removes_non_compliant(self) -> None:
        """Level 2a applies the PDF/A-2 embedded-file restrictions."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "bad.txt")

        result = sanitize_for_pdfa(pdf, level="2a")

        assert result["files_removed"] == 1
        assert result["embedded_files_kept"] == 0

    def test_level_3b_behavior_unchanged(self) -> None:
        """Level 3b keeps all embedded files (no selective removal)."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "any.txt")

        result = sanitize_for_pdfa(pdf, level="3b")

        assert result["files_removed"] == 0
        assert result["embedded_files_kept"] == 0
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_level_3u_behavior_unchanged(self) -> None:
        """Level 3u keeps all embedded files (no selective removal)."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "any.txt")

        result = sanitize_for_pdfa(pdf, level="3u")

        assert result["files_removed"] == 0
        assert result["embedded_files_kept"] == 0
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_level_3a_behavior_unchanged(self) -> None:
        """Level 3a keeps arbitrary embedded files."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "any.txt")

        result = sanitize_for_pdfa(pdf, level="3a")

        assert result["files_removed"] == 0
        assert result["embedded_files_kept"] == 0
        assert "/EmbeddedFiles" in pdf.Root.Names

    def test_level_2a_does_not_reassociate_removed_indirect_filespec(self) -> None:
        """Level 2a does not rebuild /Root/AF from a removed FileSpec object."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "bad.txt")
        buffer = BytesIO()
        pdf.save(buffer)
        buffer.seek(0)

        with pikepdf.open(buffer) as reopened:
            result = sanitize_for_pdfa(reopened, level="2a")

            assert result["files_removed"] == 1
            assert "/EmbeddedFiles" not in reopened.Root.Names
            assert "/AF" not in reopened.Root

    def test_level_a_attachment_policy_end_to_end(self, tmp_path: Path) -> None:
        """2a removes arbitrary attachments while 3a preserves them."""
        input_path = tmp_path / "attached.pdf"
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        pdf.attachments["payload.txt"] = b"enterprise payload"
        pdf.save(input_path)

        output_2a = tmp_path / "attached-2a.pdf"
        convert_to_pdfa(input_path, output_2a, level="2a")
        with pikepdf.open(output_2a) as converted:
            assert "/EmbeddedFiles" not in converted.Root.Names
            assert "/AF" not in converted.Root

        output_3a = tmp_path / "attached-3a.pdf"
        convert_to_pdfa(input_path, output_3a, level="3a")
        with pikepdf.open(output_3a) as converted:
            assert list(converted.attachments) == ["payload.txt"]
            assert len(converted.Root.AF) == 1
            assert "/EF" in _resolve_indirect(converted.Root.AF[0])

    def test_embedded_files_kept_key_in_result(self) -> None:
        """Result dict always contains embedded_files_kept key."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        for lvl in ("2a", "2b", "2u", "3a", "3b", "3u"):
            result = sanitize_for_pdfa(pdf, level=lvl)
            assert "embedded_files_kept" in result
            assert isinstance(result["embedded_files_kept"], int)

    def test_level_2b_ensures_af_relationships_for_kept_files(self) -> None:
        """Level 2b calls ensure_af_relationships when files are kept."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_embedded(pdf_data, "compliant.pdf")

        result = sanitize_for_pdfa(pdf, level="2b")

        assert result["embedded_files_kept"] == 1
        assert result["af_relationships_fixed"] >= 0

    def test_level_2b_sets_af_array_for_kept_files(self) -> None:
        """Level 2b sets /AF on pdf.Root when compliant files are kept."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_embedded(pdf_data, "compliant.pdf")

        sanitize_for_pdfa(pdf, level="2b")

        assert "/AF" in pdf.Root
        af = pdf.Root["/AF"]
        assert len(af) == 1

    def test_direct_filespec_is_shared_with_root_af_after_save(self) -> None:
        """A direct embedded FileSpec is promoted before adding it to /Root/AF."""
        pdf = _make_pdf_with_embedded(b"data", "factur-x.xml")

        ensure_af_relationships(pdf)

        buf = BytesIO()
        pdf.save(buf)
        with pikepdf.open(buf) as saved_pdf:
            embedded = saved_pdf.Root.Names.EmbeddedFiles.Names[1]
            associated = saved_pdf.Root.AF[0]
            assert embedded.objgen != (0, 0)
            assert associated.objgen == embedded.objgen


# --- Tests for ensure_embedded_file_subtypes ---


class TestEnsureEmbeddedFileSubtypes:
    """Tests for ensure_embedded_file_subtypes."""

    def test_adds_missing_subtype_from_filename(self) -> None:
        """FileSpec with /F='report.pdf' -> stream gets /Subtype = application/pdf."""
        pdf = _make_pdf_with_embedded(b"fake data", "report.pdf")

        count = ensure_embedded_file_subtypes(pdf)

        # /F and /UF point to same stream object, so only 1 fix
        assert count == 1
        ef = pdf.Root.Names.EmbeddedFiles.Names[1].EF
        stream = ef.get("/F")
        stream = _resolve_indirect(stream)
        assert str(stream.get("/Subtype")) == "/application/pdf"

    def test_preserves_existing_subtype(self) -> None:
        """Stream already has /Subtype -> not modified, returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_stream["/Subtype"] = Name("/text/plain")
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="readme.txt", UF="readme.txt", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["readme.txt", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_subtypes(pdf)

        assert count == 0

    def test_replaces_invalid_mime_subtype(self) -> None:
        """Stream with invalid /Subtype (e.g. just 'application') -> replaced."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_stream["/Subtype"] = Name("/application")
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="File.pdf", UF="File.pdf", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["File.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_subtypes(pdf)

        assert count == 1
        stream = _resolve_indirect(ef_stream)
        assert str(stream.get("/Subtype")) == "/application/pdf"

    def test_replaces_empty_mime_subtype(self) -> None:
        """Stream with /Subtype that has no subtype part -> replaced."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_stream["/Subtype"] = Name("/text")
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="readme.txt", UF="readme.txt", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["readme.txt", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_subtypes(pdf)

        assert count == 1
        stream = _resolve_indirect(ef_stream)
        assert str(stream.get("/Subtype")) == "/text/plain"

    def test_preserves_valid_mime_with_slash(self) -> None:
        """Stream with valid /Subtype like 'text/xml' -> preserved."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_stream["/Subtype"] = Name("/text/xml")
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="data.xml", UF="data.xml", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["data.xml", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_subtypes(pdf)

        assert count == 0
        stream = _resolve_indirect(ef_stream)
        assert str(stream.get("/Subtype")) == "/text/xml"

    def test_fallback_to_octet_stream(self) -> None:
        """FileSpec with unknown/no filename -> application/octet-stream."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"binary data")
        ef_dict = Dictionary(F=ef_stream)
        # FileSpec with no /F or /UF filename
        file_spec = Dictionary(Type=Name.Filespec, EF=ef_dict)
        embedded = Dictionary(Names=Array(["noname", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_subtypes(pdf)

        assert count == 1
        ef_stream = _resolve_indirect(ef_stream)
        assert str(ef_stream.get("/Subtype")) == "/application/octet-stream"

    def test_handles_multiple_ef_entries(self) -> None:
        """Both /F and /UF streams get fixed when they are different objects."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream_f = pdf.make_stream(b"data F")
        ef_stream_uf = pdf.make_stream(b"data UF")
        ef_dict = Dictionary(F=ef_stream_f, UF=ef_stream_uf)
        file_spec = Dictionary(
            Type=Name.Filespec, F="image.png", UF="image.png", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["image.png", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_subtypes(pdf)

        assert count == 2
        for key in ("/F", "/UF"):
            stream = ef_dict.get(key)
            stream = _resolve_indirect(stream)
            assert str(stream.get("/Subtype")) == "/image/png"

    def test_file_attachment_annotation_streams(self) -> None:
        """FileAttachment annotation streams get fixed too."""
        pdf = _make_pdf_with_file_attachment(b"data", "spreadsheet.xlsx")

        count = ensure_embedded_file_subtypes(pdf)

        assert count >= 1
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        ef = annot.FS.EF
        stream = ef.get("/F")
        stream = _resolve_indirect(stream)
        subtype = str(stream.get("/Subtype"))
        assert "spreadsheet" in subtype or "octet-stream" in subtype

    def test_empty_pdf_returns_zero(self) -> None:
        """No embedded files -> returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        count = ensure_embedded_file_subtypes(pdf)

        assert count == 0


class TestSanitizeSubtypesIntegration:
    """Integration tests for embedded file subtypes via sanitize_for_pdfa."""

    def test_sanitize_3b_fixes_subtypes(self) -> None:
        """sanitize_for_pdfa(level='3b') populates embedded_file_subtypes_fixed."""
        pdf = _make_pdf_with_embedded(b"data", "report.pdf")

        result = sanitize_for_pdfa(pdf, level="3b")

        assert "embedded_file_subtypes_fixed" in result
        assert result["embedded_file_subtypes_fixed"] >= 1

    def test_sanitize_2b_fixes_subtypes_for_kept_files(self) -> None:
        """Level 2b with kept files also fixes subtypes."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_embedded(pdf_data, "compliant.pdf")

        result = sanitize_for_pdfa(pdf, level="2b")

        assert result["embedded_files_kept"] == 1
        assert "embedded_file_subtypes_fixed" in result
        assert result["embedded_file_subtypes_fixed"] >= 1


# --- Tests for ensure_filespec_uf_entries ---


class TestEnsureFilespecUfEntries:
    """Tests for ensure_filespec_uf_entries."""

    def test_adds_uf_from_f(self) -> None:
        """FileSpec with /F but no /UF -> /UF added with same value."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        file_spec = Dictionary(Type=Name.Filespec, F="report.pdf", EF=ef_dict)
        embedded = Dictionary(Names=Array(["report.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_filespec_uf_entries(pdf)

        assert count == 1
        assert str(file_spec.get("/UF")) == "report.pdf"
        # /EF should also have /UF mirrored from /F
        assert "/UF" in ef_dict
        assert ef_dict["/UF"].objgen == ef_stream.objgen

    def test_preserves_existing_uf(self) -> None:
        """FileSpec already has /UF -> not modified, returns 0."""
        pdf = _make_pdf_with_embedded(b"data", "test.pdf")

        count = ensure_filespec_uf_entries(pdf)

        assert count == 0

    def test_no_f_no_uf_adds_fallback(self) -> None:
        """FileSpec with neither /F nor /UF -> both added with fallback."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        file_spec = Dictionary(Type=Name.Filespec, EF=ef_dict)
        embedded = Dictionary(Names=Array(["noname", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_filespec_uf_entries(pdf)

        assert count == 1
        assert str(file_spec.get("/F")) == "embedded_file"
        assert str(file_spec.get("/UF")) == "embedded_file"

    def test_blank_f_and_uf_are_replaced_with_nonempty_fallbacks(self) -> None:
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        file_spec = Dictionary(Type=Name.Filespec, F="", UF="   ")
        annotation = Dictionary(
            Type=Name.Annot,
            Subtype=Name.FileAttachment,
            Rect=Array([0, 0, 100, 100]),
            FS=file_spec,
        )
        pdf.pages[0]["/Annots"] = Array([annotation])

        count = ensure_filespec_uf_entries(pdf)

        assert count == 1
        assert str(file_spec["/F"]) == "embedded_file"
        assert str(file_spec["/UF"]) == "embedded_file"

    def test_file_attachment_annotation(self) -> None:
        """FileAttachment annotation FileSpec gets /UF added."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        # FileSpec with /F only, no /UF
        file_spec = Dictionary(Type=Name.Filespec, F="attach.txt", EF=ef_dict)
        annot = Dictionary(
            Type=Name.Annot,
            Subtype=Name.FileAttachment,
            Rect=Array([0, 0, 100, 100]),
            FS=file_spec,
        )
        pdf.pages[0]["/Annots"] = Array([annot])

        count = ensure_filespec_uf_entries(pdf)

        assert count == 1
        assert str(file_spec.get("/UF")) == "attach.txt"

    def test_adds_uf_to_ef_dict(self) -> None:
        """FileSpec has /UF already, but /EF only has /F -> /EF/UF added."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="report.pdf", UF="report.pdf", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["report.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_filespec_uf_entries(pdf)

        assert count == 0  # /UF string already present, no string fix
        assert "/UF" in ef_dict
        assert ef_dict["/UF"].objgen == ef_stream.objgen

    def test_ef_already_has_uf_not_modified(self) -> None:
        """/EF already has /F and /UF -> no change."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream_f = pdf.make_stream(b"data-f")
        ef_stream_uf = pdf.make_stream(b"data-uf")
        ef_dict = Dictionary(F=ef_stream_f, UF=ef_stream_uf)
        file_spec = Dictionary(
            Type=Name.Filespec, F="report.pdf", UF="report.pdf", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["report.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_filespec_uf_entries(pdf)

        assert count == 0
        # /UF should still point to the original stream, not overwritten
        assert ef_dict["/UF"].objgen == ef_stream_uf.objgen

    def test_empty_pdf_returns_zero(self) -> None:
        """No embedded files -> returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        count = ensure_filespec_uf_entries(pdf)

        assert count == 0

    def test_sanitize_3b_fixes_uf(self) -> None:
        """sanitize_for_pdfa(level='3b') populates filespec_uf_fixed."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        file_spec = Dictionary(Type=Name.Filespec, F="report.pdf", EF=ef_dict)
        embedded = Dictionary(Names=Array(["report.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        result = sanitize_for_pdfa(pdf, level="3b")

        assert "filespec_uf_fixed" in result
        assert result["filespec_uf_fixed"] >= 1


# --- Tests for ensure_embedded_file_params ---


class TestEnsureEmbeddedFileParams:
    """Tests for ensure_embedded_file_params."""

    def test_adds_params_with_mod_date(self) -> None:
        """Stream without /Params gets /Params with /ModDate added."""
        pdf = _make_pdf_with_embedded(b"fake data", "report.pdf")

        count = ensure_embedded_file_params(pdf)

        # /F and /UF point to same stream object, so only 1 fix
        assert count == 1
        ef = pdf.Root.Names.EmbeddedFiles.Names[1].EF
        stream = ef.get("/F")
        stream = _resolve_indirect(stream)
        params = stream.get("/Params")
        assert params is not None
        mod_date = str(params.get("/ModDate"))
        assert mod_date.startswith("D:")

    def test_preserves_existing_mod_date(self) -> None:
        """Existing /ModDate is not overwritten, returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_stream["/Params"] = Dictionary(
            ModDate=pikepdf.String("D:20200101120000+00'00'")
        )
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="readme.txt", UF="readme.txt", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["readme.txt", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_params(pdf)

        assert count == 0
        stream = ef_stream
        stream = _resolve_indirect(stream)
        assert str(stream.Params.ModDate) == "D:20200101120000+00'00'"

    def test_adds_mod_date_to_existing_params(self) -> None:
        """/Params with /Size but no /ModDate -> /ModDate added, /Size preserved."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_stream["/Params"] = Dictionary(Size=42)
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="doc.pdf", UF="doc.pdf", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["doc.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_params(pdf)

        assert count == 1
        stream = ef_stream
        stream = _resolve_indirect(stream)
        params = stream.get("/Params")
        # /Size is preserved
        assert int(params.get("/Size")) == 42
        # /ModDate was added
        mod_date = str(params.get("/ModDate"))
        assert mod_date.startswith("D:")

    def test_handles_multiple_ef_entries(self) -> None:
        """Distinct /F and /UF streams both get fixed (count=2)."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream_f = pdf.make_stream(b"data F")
        ef_stream_uf = pdf.make_stream(b"data UF")
        ef_dict = Dictionary(F=ef_stream_f, UF=ef_stream_uf)
        file_spec = Dictionary(
            Type=Name.Filespec, F="image.png", UF="image.png", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["image.png", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_embedded_file_params(pdf)

        assert count == 2
        for key in ("/F", "/UF"):
            stream = ef_dict.get(key)
            stream = _resolve_indirect(stream)
            assert stream.get("/Params") is not None
            assert str(stream.Params.ModDate).startswith("D:")

    def test_file_attachment_annotation_streams(self) -> None:
        """FileAttachment annotation streams get /Params fixed."""
        pdf = _make_pdf_with_file_attachment(b"data", "spreadsheet.xlsx")

        count = ensure_embedded_file_params(pdf)

        assert count >= 1
        annot = pdf.pages[0].Annots[0]
        annot = _resolve_indirect(annot)
        ef = annot.FS.EF
        stream = ef.get("/F")
        stream = _resolve_indirect(stream)
        assert stream.get("/Params") is not None

    def test_empty_pdf_returns_zero(self) -> None:
        """No embedded files -> returns 0."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        count = ensure_embedded_file_params(pdf)

        assert count == 0

    def test_deduplicates_shared_stream(self) -> None:
        """Same stream for /F and /UF -> count=1."""
        pdf = _make_pdf_with_embedded(b"shared data", "shared.pdf")

        count = ensure_embedded_file_params(pdf)

        assert count == 1


class TestSanitizeParamsIntegration:
    """Integration tests for embedded file params via sanitize_for_pdfa."""

    def test_sanitize_3b_fixes_params(self) -> None:
        """sanitize_for_pdfa(level='3b') populates embedded_file_params_fixed."""
        pdf = _make_pdf_with_embedded(b"data", "report.pdf")

        result = sanitize_for_pdfa(pdf, level="3b")

        assert "embedded_file_params_fixed" in result
        assert result["embedded_file_params_fixed"] >= 1

    def test_sanitize_2b_fixes_params_for_kept_files(self) -> None:
        """Level 2b with kept files also fixes params."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_embedded(pdf_data, "compliant.pdf")

        result = sanitize_for_pdfa(pdf, level="2b")

        assert result["embedded_files_kept"] == 1
        assert "embedded_file_params_fixed" in result
        assert result["embedded_file_params_fixed"] >= 1

    def test_sanitize_3u_fixes_params(self) -> None:
        """Level 3u also fixes params."""
        pdf = _make_pdf_with_embedded(b"data", "report.pdf")

        result = sanitize_for_pdfa(pdf, level="3u")

        assert "embedded_file_params_fixed" in result
        assert result["embedded_file_params_fixed"] >= 1


# --- /Kids Name Tree helpers and tests ---


def _make_pdf_with_kids_name_tree(
    file_data_list: list[tuple[str, bytes]],
) -> Pdf:
    """Create a PDF with EmbeddedFiles using a /Kids tree structure.

    Splits entries across two child leaf nodes to simulate a balanced
    Name Tree as described in PDF spec §7.9.6.

    Args:
        file_data_list: List of (filename, data) tuples for embedded files.
    """
    pdf = new_pdf()
    pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

    mid = max(1, len(file_data_list) // 2)
    left_entries = file_data_list[:mid]
    right_entries = file_data_list[mid:]

    def _build_leaf(entries: list[tuple[str, bytes]]) -> Dictionary:
        names_array: list[object] = []
        for fname, data in entries:
            ef_stream = pdf.make_stream(data)
            ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
            file_spec = Dictionary(
                Type=Name.Filespec,
                F=fname,
                UF=fname,
                EF=ef_dict,
            )
            names_array.append(fname)
            names_array.append(file_spec)
        return Dictionary(Names=Array(names_array))

    kids: list[Dictionary] = []
    if left_entries:
        kids.append(pdf.make_indirect(_build_leaf(left_entries)))
    if right_entries:
        kids.append(pdf.make_indirect(_build_leaf(right_entries)))

    # Root node has only /Kids (no /Names)
    embedded = Dictionary(Kids=Array(kids))
    names = Dictionary(EmbeddedFiles=embedded)
    pdf.Root.Names = names

    return pdf


class TestNameTreeTraversal:
    """Unit tests for _iter_name_tree_values and _iter_name_tree_pairs."""

    def test_flat_names_yields_values(self) -> None:
        """Flat /Names array yields all values at odd positions."""
        pdf = _make_pdf_with_embedded(b"data", "test.pdf")
        embedded = _resolve_indirect(pdf.Root.Names.EmbeddedFiles)
        values = list(_iter_name_tree_values(embedded))
        assert len(values) == 1
        resolved = _resolve_indirect(values[0])
        assert str(resolved.get("/F")) == "test.pdf"

    def test_flat_names_yields_pairs(self) -> None:
        """Flat /Names array yields correct (name, value) pairs."""
        pdf = _make_pdf_with_embedded(b"data", "test.pdf")
        embedded = _resolve_indirect(pdf.Root.Names.EmbeddedFiles)
        pairs = list(_iter_name_tree_pairs(embedded))
        assert len(pairs) == 1
        name, value = pairs[0]
        assert str(name) == "test.pdf"
        resolved = _resolve_indirect(value)
        assert str(resolved.get("/F")) == "test.pdf"

    def test_two_level_kids_tree(self) -> None:
        """Two-level /Kids tree yields all values from both children."""
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("a.pdf", b"data-a"),
                ("b.pdf", b"data-b"),
                ("c.pdf", b"data-c"),
            ]
        )
        embedded = _resolve_indirect(pdf.Root.Names.EmbeddedFiles)
        values = list(_iter_name_tree_values(embedded))
        assert len(values) == 3
        filenames = []
        for v in values:
            resolved = _resolve_indirect(v)
            filenames.append(str(resolved.get("/F")))
        assert "a.pdf" in filenames
        assert "b.pdf" in filenames
        assert "c.pdf" in filenames

    def test_three_level_nested_tree(self) -> None:
        """Three-level nested tree yields all values."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Build leaf nodes
        ef1 = pdf.make_stream(b"d1")
        fs1 = Dictionary(
            Type=Name.Filespec,
            F="f1.txt",
            UF="f1.txt",
            EF=Dictionary(F=ef1, UF=ef1),
        )
        leaf1 = pdf.make_indirect(Dictionary(Names=Array(["f1.txt", fs1])))

        ef2 = pdf.make_stream(b"d2")
        fs2 = Dictionary(
            Type=Name.Filespec,
            F="f2.txt",
            UF="f2.txt",
            EF=Dictionary(F=ef2, UF=ef2),
        )
        leaf2 = pdf.make_indirect(Dictionary(Names=Array(["f2.txt", fs2])))

        # Intermediate node
        mid_node = pdf.make_indirect(Dictionary(Kids=Array([leaf1, leaf2])))

        # Root node
        root = Dictionary(Kids=Array([mid_node]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=root)

        values = list(_iter_name_tree_values(root))
        assert len(values) == 2

    def test_empty_node_yields_nothing(self) -> None:
        """Empty node (no /Names, no /Kids) yields nothing."""
        node = Dictionary()
        values = list(_iter_name_tree_values(node))
        assert values == []

    def test_tree_beyond_python_recursion_limit(self) -> None:
        """An exceptionally deep tree is traversed completely."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        leaf = pdf.make_indirect(
            Dictionary(Names=Array(["deep.txt", Dictionary(F="deep.txt")]))
        )
        current = leaf
        for _ in range(1200):
            current = pdf.make_indirect(Dictionary(Kids=Array([current])))

        values = list(_iter_name_tree_values(current))
        assert len(values) == 1
        assert str(values[0].get("/F")) == "deep.txt"

    def test_indirect_cycle_is_traversed_once(self) -> None:
        """A cyclic Kids array terminates without yielding duplicates."""
        pdf = new_pdf()
        node = pdf.make_indirect(Dictionary())
        node["/Kids"] = Array([node])

        assert list(_iter_name_tree_values(node)) == []
        assert list(_iter_name_tree_pairs(node)) == []

    def test_cyclic_tree_is_flattened_and_serializable(self) -> None:
        """A cyclic name tree is replaced by one valid flat leaf."""
        pdf = _make_pdf_with_embedded(b"payload", "payload.txt")
        embedded = pdf.make_indirect(pdf.Root.Names.EmbeddedFiles)
        embedded["/Kids"] = Array([embedded])

        assert ensure_af_relationships(pdf) == 1

        normalized = _resolve_indirect(pdf.Root.Names.EmbeddedFiles)
        assert "/Kids" not in normalized
        assert [str(item) for item in normalized.Names[::2]] == ["payload.txt"]
        output = BytesIO()
        pdf.save(output)
        with pikepdf.open(BytesIO(output.getvalue())) as reopened:
            assert "/Kids" not in reopened.Root.Names.EmbeddedFiles
            assert len(reopened.Root.Names.EmbeddedFiles.Names) == 2

    def test_direct_cycle_is_removed_and_terminates(self) -> None:
        """A direct self-cycle is removed instead of hanging traversal."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        embedded = Dictionary()
        embedded["/Kids"] = Array([embedded])
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        assert ensure_af_relationships(pdf) == 0
        assert "/EmbeddedFiles" not in pdf.Root.Names

    def test_malformed_tree_is_filtered_sorted_and_deduplicated(self) -> None:
        """Invalid pairs and duplicate keys are removed while flattening."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        first = Dictionary(Type=Name.Filespec, F="first.txt", UF="first.txt")
        duplicate = Dictionary(Type=Name.Filespec, F="duplicate.txt")
        second = Dictionary(Type=Name.Filespec, F="second.txt", UF="second.txt")
        leaf = pdf.make_indirect(
            Dictionary(
                Names=Array(
                    [
                        "z.txt",
                        second,
                        Name.Invalid,
                        first,
                        "a.txt",
                        first,
                        "a.txt",
                        duplicate,
                        "invalid-value",
                        42,
                        "orphan",
                    ]
                )
            )
        )
        root = pdf.make_indirect(Dictionary(Kids=leaf, Limits=Array(["x", "z"])))
        pdf.Root.Names = Dictionary(EmbeddedFiles=root)

        assert ensure_af_relationships(pdf) == 2

        normalized = _resolve_indirect(pdf.Root.Names.EmbeddedFiles)
        assert "/Kids" not in normalized
        assert "/Limits" not in normalized
        assert [str(item) for item in normalized.Names[::2]] == ["a.txt", "z.txt"]
        assert len(pdf.Root.AF) == 2

    def test_odd_length_names_no_crash(self) -> None:
        """Malformed /Names array with odd length doesn't crash."""
        node = Dictionary(Names=Array(["orphan_key"]))
        values = list(_iter_name_tree_values(node))
        assert values == []

        pairs = list(_iter_name_tree_pairs(node))
        assert pairs == []


class TestKidsNameTreeIntegration:
    """Integration tests for functions with /Kids-based Name Trees."""

    def test_remove_non_compliant_filters_across_kids(self) -> None:
        """remove_non_compliant_embedded_files filters across /Kids tree."""
        compliant_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("good.pdf", compliant_data),
                ("bad.txt", b"Not a PDF"),
                ("also_bad.bin", b"\x00\x01\x02"),
            ]
        )

        result = remove_non_compliant_embedded_files(pdf)

        assert result["kept"] == 1
        assert result["removed"] == 2
        # Tree should be flattened to root /Names
        embedded = _resolve_indirect(pdf.Root.Names.EmbeddedFiles)
        assert "/Names" in embedded
        assert "/Kids" not in embedded
        names_arr = embedded["/Names"]
        assert len(names_arr) == 2  # [name, filespec]
        assert str(names_arr[0]) == "good.pdf"

    def test_remove_non_compliant_all_removed_deletes_tree(self) -> None:
        """All non-compliant files across /Kids removes EmbeddedFiles."""
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("bad1.txt", b"Not a PDF"),
                ("bad2.txt", b"Also not"),
            ]
        )

        result = remove_non_compliant_embedded_files(pdf)

        assert result["removed"] == 2
        assert result["kept"] == 0
        names = _resolve_indirect(pdf.Root.Names)
        assert "/EmbeddedFiles" not in names

    def test_ensure_af_relationships_across_kids(self) -> None:
        """ensure_af_relationships finds filespecs in all /Kids."""
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("a.pdf", b"data-a"),
                ("b.pdf", b"data-b"),
            ]
        )

        count = ensure_af_relationships(pdf)

        # Both should get /AFRelationship added
        assert count == 2
        assert "/AF" in pdf.Root
        assert len(pdf.Root["/AF"]) == 2

    def test_ensure_embedded_file_subtypes_across_kids(self) -> None:
        """ensure_embedded_file_subtypes fixes streams across /Kids."""
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("report.pdf", b"data"),
                ("image.png", b"imgdata"),
            ]
        )

        count = ensure_embedded_file_subtypes(pdf)

        # Each filespec has /F and /UF pointing to same stream = 1 fix each
        assert count == 2

    def test_ensure_embedded_file_params_across_kids(self) -> None:
        """ensure_embedded_file_params adds params across /Kids."""
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("a.pdf", b"data-a"),
                ("b.pdf", b"data-b"),
            ]
        )

        count = ensure_embedded_file_params(pdf)

        assert count == 2

    def test_ensure_filespec_uf_entries_across_kids(self) -> None:
        """ensure_filespec_uf_entries adds /UF across /Kids."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Build leaf nodes with filespecs that lack /UF
        ef1 = pdf.make_stream(b"data1")
        fs1 = Dictionary(
            Type=Name.Filespec,
            F="a.txt",
            EF=Dictionary(F=ef1),
        )
        leaf1 = pdf.make_indirect(Dictionary(Names=Array(["a.txt", fs1])))

        ef2 = pdf.make_stream(b"data2")
        fs2 = Dictionary(
            Type=Name.Filespec,
            F="b.txt",
            EF=Dictionary(F=ef2),
        )
        leaf2 = pdf.make_indirect(Dictionary(Names=Array(["b.txt", fs2])))

        embedded = Dictionary(Kids=Array([leaf1, leaf2]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_filespec_uf_entries(pdf)

        assert count == 2
        assert str(fs1.get("/UF")) == "a.txt"
        assert str(fs2.get("/UF")) == "b.txt"

    def test_remove_embedded_files_counts_across_kids(self) -> None:
        """remove_embedded_files counts files across /Kids correctly."""
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("a.pdf", b"data-a"),
                ("b.pdf", b"data-b"),
                ("c.pdf", b"data-c"),
            ]
        )

        count = remove_embedded_files(pdf)

        assert count == 3
        names = _resolve_indirect(pdf.Root.Names)
        assert "/EmbeddedFiles" not in names

    def test_sanitize_3b_with_kids_tree(self) -> None:
        """sanitize_for_pdfa level 3b integration with /Kids tree."""
        pdf = _make_pdf_with_kids_name_tree(
            [
                ("a.pdf", b"data-a"),
                ("b.pdf", b"data-b"),
            ]
        )

        result = sanitize_for_pdfa(pdf, level="3b")

        assert result["files_removed"] == 0
        assert result["af_relationships_fixed"] == 2
        assert result["embedded_file_subtypes_fixed"] == 2
        assert result["embedded_file_params_fixed"] == 2
        assert result["filespec_uf_fixed"] >= 0


class TestEnsureFilespecDesc:
    """Tests for ensure_filespec_desc()."""

    def test_adds_desc_when_missing(self):
        """Adds /Desc to FileSpec missing it."""
        pdf = _make_pdf_with_embedded(b"dummy data", "report.pdf")
        result = ensure_filespec_desc(pdf)
        assert result == 1

        # Verify /Desc was added
        names = _resolve_indirect(pdf.Root.Names)
        embedded = _resolve_indirect(names.EmbeddedFiles)
        filespec = _resolve_indirect(embedded.Names[1])
        assert "/Desc" in filespec
        assert "report.pdf" in str(filespec.Desc)

    def test_preserves_existing_desc(self):
        """Does not overwrite existing /Desc."""
        pdf = _make_pdf_with_embedded(b"dummy data", "report.pdf")

        # Add /Desc manually
        names = _resolve_indirect(pdf.Root.Names)
        embedded = _resolve_indirect(names.EmbeddedFiles)
        filespec = _resolve_indirect(embedded.Names[1])
        filespec["/Desc"] = pikepdf.String("Custom description")

        result = ensure_filespec_desc(pdf)
        assert result == 0

        # Verify original /Desc is preserved
        filespec = _resolve_indirect(embedded.Names[1])
        assert str(filespec.Desc) == "Custom description"

    def test_fallback_desc_when_no_filename(self):
        """Uses generic description when no /UF or /F is present."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        # Create FileSpec without /UF or /F
        ef_stream = pdf.make_stream(b"dummy data")
        ef_dict = Dictionary(F=ef_stream)
        file_spec = Dictionary(Type=Name.Filespec, EF=ef_dict)
        embedded = Dictionary(Names=Array(["test", file_spec]))
        names = Dictionary(EmbeddedFiles=embedded)
        pdf.Root.Names = names

        result = ensure_filespec_desc(pdf)
        assert result == 1

        filespec = _resolve_indirect(embedded.Names[1])
        assert str(filespec.Desc) == "Embedded file"


# --- Tests for /Root/AF cleanup ---


class TestEnsureAfRelationshipsCleanup:
    """Tests for stale /Root/AF cleanup in ensure_af_relationships."""

    def test_removes_stale_root_af_when_no_filespecs(self) -> None:
        """PDF with /Root/AF but no EmbeddedFiles -> /AF is removed."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        # Set a stale /AF array on Root (no embedded files exist)
        pdf.Root["/AF"] = Array([Dictionary(Type=Name.Filespec)])

        ensure_af_relationships(pdf)

        assert "/AF" not in pdf.Root


class TestRemoveNonCompliantCleanup:
    """Tests for /Root/AF cleanup in remove_non_compliant_embedded_files."""

    def test_removes_root_af_when_all_removed(self) -> None:
        """Non-compliant file + /Root/AF -> both removed."""
        pdf = _make_pdf_with_embedded(b"Not a PDF at all", "bad.txt")
        # Simulate a pre-existing /Root/AF
        pdf.Root["/AF"] = Array([Dictionary(Type=Name.Filespec)])

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 1,
            "kept": 0,
            "converted": 0,
            "conversion_failed": 0,
        }
        assert "/AF" not in pdf.Root

    def test_preserves_root_af_when_some_kept(self) -> None:
        """Mixed compliant/non-compliant + /Root/AF -> /AF preserved."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        compliant_data = _create_pdfa_pdf_bytes("2b")
        compliant_stream = pdf.make_stream(compliant_data)
        compliant_ef = Dictionary(F=compliant_stream, UF=compliant_stream)
        compliant_fs = Dictionary(
            Type=Name.Filespec, F="good.pdf", UF="good.pdf", EF=compliant_ef
        )

        bad_stream = pdf.make_stream(b"Not a PDF")
        bad_ef = Dictionary(F=bad_stream, UF=bad_stream)
        bad_fs = Dictionary(Type=Name.Filespec, F="bad.txt", UF="bad.txt", EF=bad_ef)

        embedded = Dictionary(
            Names=Array(["good.pdf", compliant_fs, "bad.txt", bad_fs])
        )
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)
        # Set a /Root/AF that should survive
        pdf.Root["/AF"] = Array([compliant_fs])

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 1,
            "kept": 1,
            "converted": 0,
            "conversion_failed": 0,
        }
        assert "/AF" in pdf.Root


class TestRemoveEmbeddedFilesCleanup:
    """Tests for /Root/AF cleanup in remove_embedded_files."""

    def test_cleans_root_af(self) -> None:
        """remove_embedded_files() also removes /Root/AF."""
        pdf = _make_pdf_with_embedded(b"data", "test.pdf")
        pdf.Root["/AF"] = Array([Dictionary(Type=Name.Filespec)])

        count = remove_embedded_files(pdf)

        assert count == 1
        assert "/AF" not in pdf.Root


# --- Helper: PDF with FileSpec only in page-level /AF ---


def _make_pdf_with_page_af(data: bytes, filename: str = "orphan.pdf") -> Pdf:
    """Create a PDF with a FileSpec ONLY in page-level /AF (not in Name Tree)."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page))
    pdf.pages.append(page)

    ef_stream = pdf.make_stream(data)
    ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
    file_spec = pdf.make_indirect(
        Dictionary(
            Type=Name.Filespec,
            F=filename,
            UF=filename,
            EF=ef_dict,
        )
    )
    pdf.pages[0]["/AF"] = Array([file_spec])
    return pdf


# --- Tests for full object scan ---


class TestFullScanFindsFilespecs:
    """Tests for _iter_all_filespecs_by_scan and _iter_all_filespecs."""

    def test_finds_filespecs_on_page_af(self) -> None:
        """FileSpec only in page-level /AF is found by full scan."""
        pdf = _make_pdf_with_page_af(b"data", "orphan.pdf")

        results = list(_iter_all_filespecs(pdf))
        assert len(results) == 1
        resolved = _resolve_indirect(results[0])
        assert str(resolved.get("/F")) == "orphan.pdf"

    def test_finds_filespecs_without_type(self) -> None:
        """FileSpec with /EF but no /Type /Filespec is found by scan."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        # No /Type key — only has /EF
        file_spec = pdf.make_indirect(
            Dictionary(
                F="notype.bin",
                EF=ef_dict,
            )
        )
        pdf.pages[0]["/AF"] = Array([file_spec])

        results = list(_iter_all_filespecs_by_scan(pdf))
        assert len(results) >= 1
        found = False
        for r in results:
            resolved = _resolve_indirect(r)
            if str(resolved.get("/F")) == "notype.bin":
                found = True
                break
        assert found

    def test_deduplicates_across_name_tree_and_page_af(self) -> None:
        """Same FileSpec in Name Tree and page /AF is yielded only once."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                F="shared.pdf",
                UF="shared.pdf",
                EF=ef_dict,
            )
        )
        # Add to Name Tree
        embedded = Dictionary(Names=Array(["shared.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)
        # Also add to page /AF
        pdf.pages[0]["/AF"] = Array([file_spec])

        results = list(_iter_all_filespecs(pdf))
        assert len(results) == 1

    def test_finds_filespecs_on_root_af_only(self) -> None:
        """FileSpec only in /Root/AF (not Name Tree or annots) is found."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                F="root_af.pdf",
                UF="root_af.pdf",
                EF=ef_dict,
            )
        )
        pdf.Root["/AF"] = Array([file_spec])

        results = list(_iter_all_filespecs(pdf))
        assert len(results) == 1
        resolved = _resolve_indirect(results[0])
        assert str(resolved.get("/F")) == "root_af.pdf"

    @pytest.mark.parametrize("owner_kind", ["root", "page", "annotation"])
    def test_finds_type_only_filespec_in_reachable_af(self, owner_kind: str) -> None:
        """Reachable associated FileSpecs do not need an embedded-file stream."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)
        file_spec = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                F="external.dat",
                UF="external.dat",
            )
        )

        if owner_kind == "root":
            pdf.Root["/AF"] = Array([file_spec])
        elif owner_kind == "page":
            page["/AF"] = Array([file_spec])
        else:
            annotation = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([0, 0, 10, 10]),
                    AF=Array([file_spec]),
                )
            )
            page["/Annots"] = Array([annotation])

        results = list(_iter_all_filespecs(pdf))

        assert len(results) == 1
        assert _resolve_indirect(results[0]).objgen == file_spec.objgen
        assert ensure_af_relationships(pdf) == 1
        assert str(file_spec["/AFRelationship"]) == "/Unspecified"

    def test_ignores_unreachable_type_only_filespec(self) -> None:
        """A stale type-only object is not revived as an associated file."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                F="stale.dat",
                UF="stale.dat",
            )
        )

        assert list(_iter_all_filespecs(pdf)) == []

    def test_normalizes_dictionary_af_and_filters_invalid_entries(self) -> None:
        """A dictionary /AF becomes an array and invalid entries are discarded."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        filespec = Dictionary(
            Type=Name.Filespec,
            F="external.txt",
            UF="external.txt",
        )
        pdf.pages[0]["/AF"] = filespec

        assert ensure_af_relationships(pdf) == 1

        page_af = pdf.pages[0].AF
        assert isinstance(page_af, Array)
        assert len(page_af) == 1
        assert page_af[0].objgen != (0, 0)

        page_af.append(42)
        page_af.append(Dictionary(Type=Name.Filespec))
        page_af.append(Dictionary(Type=Name.Action, F="not-a-filespec"))
        assert ensure_af_relationships(pdf) == 0
        assert len(pdf.pages[0].AF) == 1

    def test_removes_invalid_dictionary_af(self) -> None:
        """A dictionary that is not a FileSpec is removed from /AF."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))
        pdf.pages[0]["/AF"] = Dictionary(Foo="bar")

        assert ensure_af_relationships(pdf) == 0
        assert "/AF" not in pdf.pages[0]
        assert "/AF" not in pdf.Root


# --- Tests for ensure_filespec_uf_entries with missing /F ---


class TestEnsureFilespecUfEntriesMissingF:
    """Tests for ensure_filespec_uf_entries handling missing /F."""

    def test_adds_f_from_uf(self) -> None:
        """FileSpec with /UF but no /F -> /F added from /UF."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(UF=ef_stream)
        file_spec = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                UF="report.pdf",
                EF=ef_dict,
            )
        )
        embedded = Dictionary(Names=Array(["report.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_filespec_uf_entries(pdf)

        assert count == 1
        resolved = _resolve_indirect(file_spec)
        assert str(resolved.get("/F")) == "report.pdf"

    def test_adds_both_when_missing(self) -> None:
        """FileSpec with neither /F nor /UF but has /EF -> both added."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        file_spec = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                EF=ef_dict,
            )
        )
        embedded = Dictionary(Names=Array(["noname", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        count = ensure_filespec_uf_entries(pdf)

        assert count == 1
        resolved = _resolve_indirect(file_spec)
        assert str(resolved.get("/F")) == "embedded_file"
        assert str(resolved.get("/UF")) == "embedded_file"

    def test_mirrors_uf_to_f_in_ef_dict(self) -> None:
        """/EF has /UF but not /F -> /EF/F added from /EF/UF."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(UF=ef_stream)
        file_spec = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                F="test.pdf",
                UF="test.pdf",
                EF=ef_dict,
            )
        )
        embedded = Dictionary(Names=Array(["test.pdf", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        ensure_filespec_uf_entries(pdf)

        assert "/F" in ef_dict
        assert ef_dict["/F"].objgen == ef_stream.objgen


# --- Tests for orphan FileSpec removal ---


class TestRemoveNonCompliantOrphanFilespecs:
    """Tests for remove_non_compliant_embedded_files with orphan FileSpecs."""

    def test_strips_ef_from_orphan_on_page_af(self) -> None:
        """FileSpec in page /AF but NOT in Name Tree -> /EF stripped."""
        pdf = _make_pdf_with_page_af(b"Not a PDF", "orphan.txt")

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 1,
            "kept": 0,
            "converted": 0,
            "conversion_failed": 0,
        }
        # The FileSpec should no longer have /EF
        page_af = pdf.pages[0].get("/AF")
        if page_af is not None:
            for entry in page_af:
                resolved = _resolve_indirect(entry)
                assert resolved.get("/EF") is None

    def test_cleans_page_af_after_removal(self) -> None:
        """Page-level /AF is cleaned after orphan non-compliant removal."""
        pdf = _make_pdf_with_page_af(b"Not a PDF", "orphan.txt")

        remove_non_compliant_embedded_files(pdf)

        # /AF should be cleaned up (entries removed or array deleted)
        page_af = pdf.pages[0].get("/AF")
        if page_af is not None:
            for entry in page_af:
                resolved = _resolve_indirect(entry)
                # Remaining entries should have no /EF
                assert resolved.get("/EF") is None

    def test_keeps_compliant_orphan_on_page_af(self) -> None:
        """Compliant FileSpec in page /AF is kept."""
        pdf_data = _create_pdfa_pdf_bytes("1b")
        pdf = _make_pdf_with_page_af(pdf_data, "compliant.pdf")

        result = remove_non_compliant_embedded_files(pdf)

        assert result["kept"] >= 1
        # FileSpec should still have /EF
        page_af = pdf.pages[0].get("/AF")
        assert page_af is not None
        resolved = _resolve_indirect(page_af[0])
        assert resolved.get("/EF") is not None

    def test_no_double_counting_with_name_tree(self) -> None:
        """FileSpec in Name Tree is not double-counted by orphan scan."""
        pdf = _make_pdf_with_embedded(b"Not a PDF", "bad.txt")

        result = remove_non_compliant_embedded_files(pdf)

        # Should be exactly 1 removal, not 2
        assert result["removed"] == 1

    @pytest.mark.parametrize("owner_kind", ["root", "page", "annotation"])
    def test_removes_direct_filespec_from_associated_file_arrays(
        self, owner_kind: str
    ) -> None:
        """Direct FileSpecs reachable only through /AF are removed for PDF/A-2."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)
        stream = pdf.make_stream(b"not a PDF")
        filespec = Dictionary(
            Type=Name.Filespec,
            F="payload.bin",
            UF="payload.bin",
            EF=Dictionary(F=stream, UF=stream),
        )
        if owner_kind == "root":
            owner = pdf.Root
        elif owner_kind == "page":
            owner = page
        else:
            owner = pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Text,
                    Rect=Array([0, 0, 10, 10]),
                )
            )
            page["/Annots"] = Array([owner])
        owner["/AF"] = filespec

        result = remove_non_compliant_embedded_files(pdf)

        assert result == {
            "removed": 1,
            "kept": 0,
            "converted": 0,
            "conversion_failed": 0,
        }
        assert "/AF" not in owner
        assert "/AF" not in pdf.Root
        assert "/EF" not in filespec

    def test_removes_direct_fileattachment_filespec(self) -> None:
        """A direct non-PDF FileAttachment /FS is removed without double count."""
        pdf = new_pdf()
        page = pikepdf.Page(Dictionary(Type=Name.Page))
        pdf.pages.append(page)
        stream = pdf.make_stream(b"not a PDF")
        filespec = Dictionary(
            Type=Name.Filespec,
            F="payload.bin",
            UF="payload.bin",
            EF=Dictionary(F=stream, UF=stream),
        )
        annotation = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.FileAttachment,
                Rect=Array([0, 0, 10, 10]),
                FS=filespec,
            )
        )
        page["/Annots"] = Array([annotation])

        result = remove_non_compliant_embedded_files(pdf)

        assert result["removed"] == 1
        assert "/Annots" not in page
        assert "/EF" not in filespec

    def test_deduplicates_direct_filespec_across_name_tree_and_af(self) -> None:
        """One shared direct FileSpec is processed once across all references."""
        pdf = _make_pdf_with_embedded(b"not a PDF", "payload.bin")
        filespec = pdf.Root.Names.EmbeddedFiles.Names[1]
        pdf.pages[0]["/AF"] = filespec

        result = remove_non_compliant_embedded_files(pdf)

        assert result["removed"] == 1
        assert "/EmbeddedFiles" not in pdf.Root.Names
        assert "/AF" not in pdf.pages[0]
        assert "/AF" not in pdf.Root


# --- Tests for fix functions with scan-found FileSpecs ---


class TestFixFunctionsWithScanFoundFilespecs:
    """Verify fix functions process FileSpecs found only by full scan."""

    def test_af_relationships_for_scan_found_filespecs(self) -> None:
        """ensure_af_relationships() processes FileSpecs on page /AF."""
        pdf = _make_pdf_with_page_af(b"data", "orphan.pdf")

        count = ensure_af_relationships(pdf)

        assert count >= 1
        # /Root/AF should be built
        assert "/AF" in pdf.Root

    def test_subtypes_for_scan_found_filespecs(self) -> None:
        """ensure_embedded_file_subtypes() processes FileSpecs on page /AF."""
        pdf = _make_pdf_with_page_af(b"data", "report.pdf")

        count = ensure_embedded_file_subtypes(pdf)

        assert count >= 1

    def test_params_for_scan_found_filespecs(self) -> None:
        """ensure_embedded_file_params() processes FileSpecs on page /AF."""
        pdf = _make_pdf_with_page_af(b"data", "report.pdf")

        count = ensure_embedded_file_params(pdf)

        assert count >= 1

    def test_uf_entries_for_scan_found_filespecs(self) -> None:
        """ensure_filespec_uf_entries() processes FileSpecs on page /AF."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream = pdf.make_stream(b"data")
        ef_dict = Dictionary(F=ef_stream)
        # FileSpec with /F only, no /UF — on page /AF only
        file_spec = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                F="attach.txt",
                EF=ef_dict,
            )
        )
        pdf.pages[0]["/AF"] = Array([file_spec])

        count = ensure_filespec_uf_entries(pdf)

        assert count == 1
        resolved = _resolve_indirect(file_spec)
        assert str(resolved.get("/UF")) == "attach.txt"

    def test_desc_for_scan_found_filespecs(self) -> None:
        """ensure_filespec_desc() processes FileSpecs on page /AF."""
        pdf = _make_pdf_with_page_af(b"data", "report.pdf")

        count = ensure_filespec_desc(pdf)

        assert count >= 1


# --- Tests for remove_embedded_files with full scan ---


class TestRemoveEmbeddedFilesFullScan:
    """Tests for remove_embedded_files() handling orphan FileSpecs."""

    def test_removes_orphan_on_page_af(self) -> None:
        """FileSpec only in page /AF is also removed."""
        pdf = _make_pdf_with_page_af(b"data", "orphan.pdf")

        count = remove_embedded_files(pdf)

        assert count >= 1
        # /EF should be stripped
        page_af = pdf.pages[0].get("/AF")
        if page_af is not None:
            for entry in page_af:
                resolved = _resolve_indirect(entry)
                assert resolved.get("/EF") is None

    def test_removes_mixed_name_tree_and_page_af(self) -> None:
        """Both Name Tree entries and page /AF orphans are removed."""
        pdf = _make_pdf_with_embedded(b"data1", "tree.pdf")
        # Add an orphan on page /AF
        ef_stream = pdf.make_stream(b"data2")
        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        orphan_fs = pdf.make_indirect(
            Dictionary(
                Type=Name.Filespec,
                F="orphan.pdf",
                UF="orphan.pdf",
                EF=ef_dict,
            )
        )
        pdf.pages[0]["/AF"] = Array([orphan_fs])

        count = remove_embedded_files(pdf)

        assert count == 2

    def test_no_double_counting(self) -> None:
        """FileSpec in Name Tree is not double-counted by orphan scan."""
        pdf = _make_pdf_with_embedded(b"data", "test.pdf")

        count = remove_embedded_files(pdf)

        # Should be exactly 1, not 2
        assert count == 1


# --- Tests for sanitize_embedded_file_filters ---


class TestSanitizeEmbeddedFileFilters:
    """Tests for sanitize_embedded_file_filters."""

    def test_no_embedded_files_returns_zero(self) -> None:
        """PDF without embedded files returns zeroes."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        result = sanitize_embedded_file_filters(pdf)

        assert result == {"lzw_converted": 0, "crypt_removed": 0}

    def test_normal_filter_not_touched(self) -> None:
        """Embedded file with FlateDecode is not modified."""
        pdf = _make_pdf_with_embedded(b"some data", "normal.pdf")

        result = sanitize_embedded_file_filters(pdf)

        assert result == {"lzw_converted": 0, "crypt_removed": 0}

    def test_lzw_filter_is_converted(self, monkeypatch) -> None:
        """Embedded file with LZWDecode filter is re-encoded."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        raw_data = b"Test data for LZW conversion"
        ef_stream = pdf.make_stream(raw_data)
        ef_stream[Name.Filter] = Name.LZWDecode

        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="test.bin", UF="test.bin", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["test.bin", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        # Patch read_bytes to return decoded data (avoids real LZW decode)
        monkeypatch.setattr(type(ef_stream), "read_bytes", lambda self: raw_data)

        result = sanitize_embedded_file_filters(pdf)

        # /F and /UF point to the same stream, so 1 conversion
        assert result["lzw_converted"] == 1

    def test_crypt_filter_is_removed(self, monkeypatch) -> None:
        """Embedded file with Crypt filter is cleaned."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        raw_data = b"Test data for Crypt removal"
        ef_stream = pdf.make_stream(raw_data)
        ef_stream[Name.Filter] = Name("/Crypt")

        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="test.bin", UF="test.bin", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["test.bin", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        monkeypatch.setattr(type(ef_stream), "read_bytes", lambda self: raw_data)

        result = sanitize_embedded_file_filters(pdf)

        assert result["crypt_removed"] == 1

    def test_deduplicates_shared_stream(self, monkeypatch) -> None:
        """Same stream for /F and /UF is only processed once."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        raw_data = b"Shared stream data"
        ef_stream = pdf.make_stream(raw_data)
        ef_stream[Name.Filter] = Name.LZWDecode

        ef_dict = Dictionary(F=ef_stream, UF=ef_stream)
        file_spec = Dictionary(
            Type=Name.Filespec, F="test.bin", UF="test.bin", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["test.bin", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        monkeypatch.setattr(type(ef_stream), "read_bytes", lambda self: raw_data)

        result = sanitize_embedded_file_filters(pdf)

        assert result["lzw_converted"] == 1

    def test_distinct_f_uf_streams_both_fixed(self, monkeypatch) -> None:
        """Distinct /F and /UF streams are both fixed."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        ef_stream_f = pdf.make_stream(b"data F")
        ef_stream_f[Name.Filter] = Name.LZWDecode
        ef_stream_uf = pdf.make_stream(b"data UF")
        ef_stream_uf[Name.Filter] = Name.LZWDecode

        ef_dict = Dictionary(F=ef_stream_f, UF=ef_stream_uf)
        file_spec = Dictionary(
            Type=Name.Filespec, F="test.bin", UF="test.bin", EF=ef_dict
        )
        embedded = Dictionary(Names=Array(["test.bin", file_spec]))
        pdf.Root.Names = Dictionary(EmbeddedFiles=embedded)

        monkeypatch.setattr(type(ef_stream_f), "read_bytes", lambda self: b"data")

        result = sanitize_embedded_file_filters(pdf)

        assert result["lzw_converted"] == 2

    def test_integration_via_sanitize_for_pdfa(self) -> None:
        """sanitize_for_pdfa includes embedded_file_lzw_converted in result."""
        pdf = new_pdf()
        pdf.pages.append(pikepdf.Page(Dictionary(Type=Name.Page)))

        result = sanitize_for_pdfa(pdf, level="3b")

        assert "embedded_file_lzw_converted" in result
        assert "embedded_file_crypt_removed" in result
