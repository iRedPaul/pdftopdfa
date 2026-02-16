# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for filter conversion and external stream key removal in PDF/A sanitization."""

import zlib
from collections.abc import Generator

import pikepdf
import pytest
from conftest import new_pdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

from pdftopdfa.sanitizers.filters import (
    _convert_lzw_stream,
    _has_crypt_filter,
    _has_external_stream_keys,
    _has_image_filter,
    _has_lzw_filter,
    _remove_crypt_stream,
    convert_lzw_streams,
    fix_stream_lengths,
    remove_crypt_streams,
    remove_external_stream_keys,
)


@pytest.fixture
def pdf() -> Generator[Pdf]:
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)
    yield pdf


_INLINE_FILTER_ABBREVIATIONS = {
    "/AHx": "/ASCIIHexDecode",
    "/A85": "/ASCII85Decode",
    "/LZW": "/LZWDecode",
    "/Fl": "/FlateDecode",
    "/RL": "/RunLengthDecode",
    "/CCF": "/CCITTFaxDecode",
    "/DCT": "/DCTDecode",
}


def _first_inline_image(page) -> pikepdf.PdfInlineImage:
    for operands, operator in pikepdf.parse_content_stream(page):
        if str(operator) == "INLINE IMAGE" and operands:
            return operands[0]
    raise AssertionError("Expected inline image in content stream")


def _normalized_inline_filters(inline_image: pikepdf.PdfInlineImage) -> list[str]:
    filter_obj = inline_image.obj.get("/Filter")
    if filter_obj is None:
        return []
    if isinstance(filter_obj, Name):
        return [_INLINE_FILTER_ABBREVIATIONS.get(str(filter_obj), str(filter_obj))]
    if isinstance(filter_obj, Array):
        return [
            _INLINE_FILTER_ABBREVIATIONS.get(str(entry), str(entry))
            for entry in filter_obj
            if isinstance(entry, Name)
        ]
    return []


# --- _has_crypt_filter tests ---


class TestHasCryptFilter:
    """Tests for _has_crypt_filter."""

    def test_single_crypt_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x00\x01", Dictionary(Filter=Name("/Crypt")))
        )
        assert _has_crypt_filter(stream) is True

    def test_crypt_in_filter_array(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(
                pdf,
                b"\x00\x01",
                Dictionary(Filter=Array([Name("/FlateDecode"), Name("/Crypt")])),
            )
        )
        assert _has_crypt_filter(stream) is True

    def test_no_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"\x00\x01"))
        assert _has_crypt_filter(stream) is False

    def test_other_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x00\x01", Dictionary(Filter=Name("/FlateDecode")))
        )
        assert _has_crypt_filter(stream) is False

    def test_other_filter_array(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(
                pdf,
                b"\x00\x01",
                Dictionary(
                    Filter=Array(
                        [
                            Name("/FlateDecode"),
                            Name("/ASCII85Decode"),
                        ]
                    )
                ),
            )
        )
        assert _has_crypt_filter(stream) is False


# --- _has_lzw_filter tests ---


class TestHasLzwFilter:
    """Tests for _has_lzw_filter."""

    def test_single_lzw_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x00\x01", Dictionary(Filter=Name("/LZWDecode")))
        )
        assert _has_lzw_filter(stream) is True

    def test_lzw_in_filter_array(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(
                pdf,
                b"\x00\x01",
                Dictionary(Filter=Array([Name("/LZWDecode"), Name("/FlateDecode")])),
            )
        )
        assert _has_lzw_filter(stream) is True

    def test_no_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"\x00\x01"))
        assert _has_lzw_filter(stream) is False

    def test_other_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x00\x01", Dictionary(Filter=Name("/FlateDecode")))
        )
        assert _has_lzw_filter(stream) is False


# --- _remove_crypt_stream tests ---


class TestRemoveCryptStream:
    """Tests for _remove_crypt_stream."""

    def test_removes_crypt_filter(self, pdf: Pdf) -> None:
        data = b"decrypted content here"
        stream = pdf.make_indirect(Stream(pdf, data, Dictionary(Filter=Name("/Crypt"))))
        result = _remove_crypt_stream(stream, pdf)
        assert result is True
        # After write(), the /Crypt filter should be gone
        assert stream.get("/Filter") is None or str(stream.get("/Filter")) != "/Crypt"

    def test_preserves_data(self, pdf: Pdf) -> None:
        data = b"important stream data"
        stream = pdf.make_indirect(Stream(pdf, data, Dictionary(Filter=Name("/Crypt"))))
        _remove_crypt_stream(stream, pdf)
        # read_bytes() should return the original data
        assert stream.read_bytes() == data

    def test_removes_decode_parms_single_filter(self, pdf: Pdf) -> None:
        data = b"decrypted content"
        stream = pdf.make_indirect(
            Stream(
                pdf,
                data,
                Dictionary(
                    Filter=Name("/Crypt"),
                    DecodeParms=Dictionary(Type=Name("/CryptFilterDecodeParms")),
                ),
            )
        )
        _remove_crypt_stream(stream, pdf)
        assert stream.get("/DecodeParms") is None

    def test_removes_decode_parms_array(self, pdf: Pdf) -> None:
        """Orphaned DecodeParms array is removed after Crypt filter removal."""
        data = b"decrypted content"
        stream = pdf.make_indirect(Stream(pdf, data))
        # Simulate orphaned DecodeParms (no matching filter chain)
        stream["/DecodeParms"] = Array(
            [
                Dictionary(),
                Dictionary(Type=Name("/CryptFilterDecodeParms")),
            ]
        )
        _remove_crypt_stream(stream, pdf)
        assert stream.get("/DecodeParms") is None


# --- _convert_lzw_stream tests ---


class TestConvertLzwStream:
    """Tests for _convert_lzw_stream."""

    def test_removes_decode_parms(self, pdf: Pdf) -> None:
        """Orphaned DecodeParms dict is removed after LZW conversion."""
        data = b"some data"
        stream = pdf.make_indirect(Stream(pdf, data))
        # Simulate orphaned DecodeParms (no actual LZW filter to trip up read_bytes)
        stream["/DecodeParms"] = Dictionary(EarlyChange=1)
        result = _convert_lzw_stream(stream, pdf)
        assert result is True
        assert stream.get("/DecodeParms") is None

    def test_removes_decode_parms_array(self, pdf: Pdf) -> None:
        """Orphaned DecodeParms array is removed after LZW conversion."""
        data = b"some data"
        stream = pdf.make_indirect(Stream(pdf, data))
        # Simulate orphaned DecodeParms array
        stream["/DecodeParms"] = Array(
            [
                Dictionary(EarlyChange=1),
                Dictionary(),
            ]
        )
        result = _convert_lzw_stream(stream, pdf)
        assert result is True
        assert stream.get("/DecodeParms") is None


class TestInlineImageFilters:
    """Tests for inline-image LZW/Crypt sanitization."""

    def test_convert_lzw_streams_rewrites_inline_image_filters(
        self,
        pdf: Pdf,
        monkeypatch,
    ) -> None:
        page = pdf.pages[0]
        page[Name("/Contents")] = pdf.make_stream(
            b"q\nBI\n/W 3 /H 3 /BPC 8 /CS /G /F /LZW\nID\nrawdata\nEI\nQ\n"
        )

        monkeypatch.setattr(
            "pdftopdfa.sanitizers.filters._decode_inline_image_payload",
            lambda *_args, **_kwargs: b"\x00\x01\x02\x03",
        )

        assert convert_lzw_streams(pdf) >= 1

        inline_image = _first_inline_image(page)
        assert _normalized_inline_filters(inline_image) == ["/FlateDecode"]
        assert inline_image.obj.get("/DecodeParms") is None

        payload = inline_image._data._inline_image_raw_bytes().rstrip(b"\t\n\f\r ")
        assert zlib.decompress(payload) == b"\x00\x01\x02\x03"

    def test_convert_lzw_streams_rewrites_mixed_case_inline_lzw_filter(
        self,
        pdf: Pdf,
        monkeypatch,
    ) -> None:
        page = pdf.pages[0]
        page[Name("/Contents")] = pdf.make_stream(
            b"q\nBI\n/W 3 /H 3 /BPC 8 /CS /G /F /Lzw\nID\nrawdata\nEI\nQ\n"
        )

        monkeypatch.setattr(
            "pdftopdfa.sanitizers.filters._decode_inline_image_payload",
            lambda *_args, **_kwargs: b"\xaa\xbb\xcc",
        )

        assert convert_lzw_streams(pdf) >= 1

        inline_image = _first_inline_image(page)
        assert _normalized_inline_filters(inline_image) == ["/FlateDecode"]
        assert inline_image.obj.get("/DecodeParms") is None

        payload = inline_image._data._inline_image_raw_bytes().rstrip(b"\t\n\f\r ")
        assert zlib.decompress(payload) == b"\xaa\xbb\xcc"

    def test_remove_crypt_streams_rewrites_inline_image_filters(
        self,
        pdf: Pdf,
        monkeypatch,
    ) -> None:
        page = pdf.pages[0]
        page[Name("/Contents")] = pdf.make_stream(
            b"q\nBI\n/W 3 /H 3 /BPC 8 /CS /G /F /Crypt\nID\nrawdata\nEI\nQ\n"
        )

        monkeypatch.setattr(
            "pdftopdfa.sanitizers.filters._decode_inline_image_payload",
            lambda *_args, **_kwargs: b"\x10\x20\x30",
        )

        assert remove_crypt_streams(pdf) >= 1

        inline_image = _first_inline_image(page)
        assert _normalized_inline_filters(inline_image) == ["/FlateDecode"]
        assert inline_image.obj.get("/DecodeParms") is None

        payload = inline_image._data._inline_image_raw_bytes().rstrip(b"\t\n\f\r ")
        assert zlib.decompress(payload) == b"\x10\x20\x30"

    def test_remove_crypt_streams_fallback_strips_inline_crypt_filter(
        self,
        pdf: Pdf,
        monkeypatch,
    ) -> None:
        page = pdf.pages[0]
        page[Name("/Contents")] = pdf.make_stream(
            b"q\nBI\n/W 3 /H 3 /BPC 8 /CS /G /F /Crypt\nID\nrawdata\nEI\nQ\n"
        )

        def _raise_decode_error(*_args, **_kwargs):
            raise ValueError("decode failed")

        monkeypatch.setattr(
            "pdftopdfa.sanitizers.filters._decode_inline_image_payload",
            _raise_decode_error,
        )

        assert remove_crypt_streams(pdf) >= 1

        inline_image = _first_inline_image(page)
        assert _normalized_inline_filters(inline_image) == []
        assert inline_image.obj.get("/DecodeParms") is None

    def test_convert_lzw_streams_normalizes_stream_filter_name_case(
        self,
        pdf: Pdf,
    ) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"plain"))
        stream["/Filter"] = Name("/Flatedecode")

        convert_lzw_streams(pdf)

        assert str(stream.get("/Filter")) == "/FlateDecode"


# --- remove_crypt_streams tests ---


class TestRemoveCryptStreams:
    """Tests for remove_crypt_streams."""

    def test_no_crypt_streams(self, pdf: Pdf) -> None:
        # PDF with no Crypt filters should return 0
        pdf.make_indirect(
            Stream(pdf, b"\x00\x01", Dictionary(Filter=Name("/FlateDecode")))
        )
        assert remove_crypt_streams(pdf) == 0

    def test_single_crypt_stream(self, pdf: Pdf) -> None:
        pdf.make_indirect(Stream(pdf, b"data", Dictionary(Filter=Name("/Crypt"))))
        assert remove_crypt_streams(pdf) == 1

    def test_multiple_crypt_streams(self, pdf: Pdf) -> None:
        for i in range(3):
            pdf.make_indirect(
                Stream(pdf, f"data{i}".encode(), Dictionary(Filter=Name("/Crypt")))
            )
        assert remove_crypt_streams(pdf) == 3

    def test_mixed_streams(self, pdf: Pdf) -> None:
        # One Crypt, one FlateDecode, one no filter
        pdf.make_indirect(Stream(pdf, b"crypt", Dictionary(Filter=Name("/Crypt"))))
        pdf.make_indirect(
            Stream(pdf, b"flate", Dictionary(Filter=Name("/FlateDecode")))
        )
        pdf.make_indirect(Stream(pdf, b"plain"))
        assert remove_crypt_streams(pdf) == 1

    def test_empty_pdf(self, pdf: Pdf) -> None:
        assert remove_crypt_streams(pdf) == 0


# --- _has_external_stream_keys tests ---


class TestHasExternalStreamKeys:
    """Tests for _has_external_stream_keys."""

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("/F", Name("/external.dat")),
            ("/FFilter", Name("/FlateDecode")),
            ("/FDecodeParms", Dictionary()),
        ],
    )
    def test_single_external_key(self, pdf: Pdf, key: str, value) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"data"))
        stream[key] = value
        assert _has_external_stream_keys(stream) == [key]

    def test_all_three_keys(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"data"))
        stream["/F"] = Name("/external.dat")
        stream["/FFilter"] = Name("/FlateDecode")
        stream["/FDecodeParms"] = Dictionary()
        assert _has_external_stream_keys(stream) == [
            "/F",
            "/FFilter",
            "/FDecodeParms",
        ]

    def test_clean_stream(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"data"))
        assert _has_external_stream_keys(stream) == []

    def test_normal_filter_not_detected(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"data", Dictionary(Filter=Name("/FlateDecode")))
        )
        assert _has_external_stream_keys(stream) == []


# --- remove_external_stream_keys tests ---


class TestRemoveExternalStreamKeys:
    """Tests for remove_external_stream_keys."""

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("/F", Name("/external.dat")),
            ("/FFilter", Name("/FlateDecode")),
            ("/FDecodeParms", Dictionary()),
        ],
    )
    def test_removes_single_key(self, pdf: Pdf, key: str, value) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"data"))
        stream[key] = value
        assert remove_external_stream_keys(pdf) == 1
        assert stream.get(key) is None

    def test_removes_all_three_keys(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"data"))
        stream["/F"] = Name("/external.dat")
        stream["/FFilter"] = Name("/FlateDecode")
        stream["/FDecodeParms"] = Dictionary()
        assert remove_external_stream_keys(pdf) == 1
        assert stream.get("/F") is None
        assert stream.get("/FFilter") is None
        assert stream.get("/FDecodeParms") is None

    def test_preserves_inline_data(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"important data"))
        stream["/F"] = Name("/external.dat")
        remove_external_stream_keys(pdf)
        assert stream.read_bytes() == b"important data"

    def test_preserves_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"data", Dictionary(Filter=Name("/FlateDecode")))
        )
        stream["/FFilter"] = Name("/ASCIIHexDecode")
        remove_external_stream_keys(pdf)
        assert stream.get("/FFilter") is None
        assert str(stream.get("/Filter")) == "/FlateDecode"

    def test_multiple_streams(self, pdf: Pdf) -> None:
        for i in range(3):
            s = pdf.make_indirect(Stream(pdf, f"data{i}".encode()))
            s["/F"] = Name("/ext.dat")
        assert remove_external_stream_keys(pdf) == 3

    def test_mixed_streams(self, pdf: Pdf) -> None:
        s1 = pdf.make_indirect(Stream(pdf, b"bad"))
        s1["/F"] = Name("/ext.dat")
        pdf.make_indirect(Stream(pdf, b"clean"))
        s3 = pdf.make_indirect(Stream(pdf, b"also bad"))
        s3["/FFilter"] = Name("/FlateDecode")
        assert remove_external_stream_keys(pdf) == 2

    def test_empty_pdf(self, pdf: Pdf) -> None:
        assert remove_external_stream_keys(pdf) == 0

    def test_counts_streams_not_keys(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"data"))
        stream["/F"] = Name("/external.dat")
        stream["/FFilter"] = Name("/FlateDecode")
        stream["/FDecodeParms"] = Dictionary()
        assert remove_external_stream_keys(pdf) == 1


# --- _has_image_filter tests ---


class TestHasImageFilter:
    """Tests for _has_image_filter."""

    @pytest.mark.parametrize(
        "filter_name",
        ["/DCTDecode", "/JPXDecode", "/JBIG2Decode", "/CCITTFaxDecode"],
    )
    def test_image_filter_detected(self, pdf: Pdf, filter_name: str) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x00", Dictionary(Filter=Name(filter_name)))
        )
        assert _has_image_filter(stream) is True

    def test_flatedecode_not_image(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(pdf, b"\x00", Dictionary(Filter=Name("/FlateDecode")))
        )
        assert _has_image_filter(stream) is False

    def test_no_filter(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(Stream(pdf, b"\x00"))
        assert _has_image_filter(stream) is False

    def test_image_filter_in_array(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(
                pdf,
                b"\x00",
                Dictionary(Filter=Array([Name("/FlateDecode"), Name("/DCTDecode")])),
            )
        )
        assert _has_image_filter(stream) is True

    def test_non_image_filters_in_array(self, pdf: Pdf) -> None:
        stream = pdf.make_indirect(
            Stream(
                pdf,
                b"\x00",
                Dictionary(
                    Filter=Array(
                        [
                            Name("/FlateDecode"),
                            Name("/ASCII85Decode"),
                        ]
                    )
                ),
            )
        )
        assert _has_image_filter(stream) is False


# --- fix_stream_lengths tests ---


class TestFixStreamLengths:
    """Tests for fix_stream_lengths."""

    def test_unfiltered_stream_reencoded(self, pdf: Pdf) -> None:
        pdf.make_indirect(Stream(pdf, b"hello world"))
        count = fix_stream_lengths(pdf)
        assert count > 0

    def test_multiple_unfiltered_streams_reencoded(self, pdf: Pdf) -> None:
        pdf.make_indirect(Stream(pdf, b"stream one"))
        pdf.make_indirect(Stream(pdf, b"stream two"))
        count = fix_stream_lengths(pdf)
        # At least the two we added (page content stream may also count)
        assert count >= 2

    def test_dctdecode_skipped(self, pdf: Pdf) -> None:
        pdf.make_indirect(
            Stream(pdf, b"\xff\xd8", Dictionary(Filter=Name("/DCTDecode")))
        )
        count = fix_stream_lengths(pdf)
        # Only the page content stream is counted, not the DCT stream
        page_streams = sum(
            1
            for obj in pdf.objects
            if isinstance(obj, Stream) and not _has_image_filter(obj)
        )
        assert count == page_streams

    def test_jpxdecode_skipped(self, pdf: Pdf) -> None:
        pdf.make_indirect(Stream(pdf, b"\x00", Dictionary(Filter=Name("/JPXDecode"))))
        count = fix_stream_lengths(pdf)
        page_streams = sum(
            1
            for obj in pdf.objects
            if isinstance(obj, Stream) and not _has_image_filter(obj)
        )
        assert count == page_streams

    def test_jbig2decode_skipped(self, pdf: Pdf) -> None:
        pdf.make_indirect(Stream(pdf, b"\x00", Dictionary(Filter=Name("/JBIG2Decode"))))
        count = fix_stream_lengths(pdf)
        page_streams = sum(
            1
            for obj in pdf.objects
            if isinstance(obj, Stream) and not _has_image_filter(obj)
        )
        assert count == page_streams

    def test_ccittfaxdecode_skipped(self, pdf: Pdf) -> None:
        pdf.make_indirect(
            Stream(pdf, b"\x00", Dictionary(Filter=Name("/CCITTFaxDecode")))
        )
        count = fix_stream_lengths(pdf)
        page_streams = sum(
            1
            for obj in pdf.objects
            if isinstance(obj, Stream) and not _has_image_filter(obj)
        )
        assert count == page_streams

    def test_data_preserved_after_reencode(self, pdf: Pdf) -> None:
        data = b"important stream content"
        stream = pdf.make_indirect(Stream(pdf, data))
        fix_stream_lengths(pdf)
        assert stream.read_bytes() == data

    def test_mixed_streams_count(self, pdf: Pdf) -> None:
        """Only non-image streams are counted."""
        pdf.make_indirect(Stream(pdf, b"plain text"))
        pdf.make_indirect(
            Stream(pdf, b"\xff\xd8", Dictionary(Filter=Name("/DCTDecode")))
        )
        pdf.make_indirect(Stream(pdf, b"more text"))
        count = fix_stream_lengths(pdf)
        # 2 added streams + page content stream(s) — all non-image
        assert count >= 2
        # The DCTDecode stream must not be counted
        total_streams = sum(1 for obj in pdf.objects if isinstance(obj, Stream))
        assert count < total_streams

    def test_roundtrip_preserves_data(self, pdf: Pdf, tmp_path) -> None:
        """Save + reopen preserves stream data after re-encoding."""
        data = b"BT /F1 12 Tf (Hello) Tj ET"
        content_stream = pdf.make_indirect(Stream(pdf, data))
        pdf.pages[0][Name("/Contents")] = content_stream
        fix_stream_lengths(pdf)
        out = tmp_path / "out.pdf"
        pdf.save(str(out))
        with pikepdf.open(out) as reopened:
            page_contents = reopened.pages[0]["/Contents"]
            assert page_contents.read_bytes() == data
