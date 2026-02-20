# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for annotation preservation through OCR."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pikepdf
from conftest import make_pdf_with_page, new_pdf, resolve
from pikepdf import Array, Dictionary, Name, Pdf

from pdftopdfa.converter import (
    _has_annotations,
    _restore_annotations_after_ocr,
    _strip_annotations_for_ocr,
)

# -- Helpers --


def _make_stamp_annotation(pdf: Pdf, page: pikepdf.Page) -> Dictionary:
    """Create a Stamp annotation with an AP stream on a page."""
    # Build a minimal appearance stream (Form XObject)
    ap_stream = pdf.make_stream(b"q 1 0 0 rg 0 0 50 20 re f Q")
    ap_stream[Name.Type] = Name.XObject
    ap_stream[Name.Subtype] = Name.Form
    ap_stream[Name.BBox] = Array([0, 0, 50, 20])

    annot = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Stamp,
            Rect=Array([100, 700, 150, 720]),
            F=4,
            AP=Dictionary(N=ap_stream),
            P=page.obj,
        )
    )
    return annot


def _save_pdf_to_path(pdf: Pdf, path: Path) -> None:
    """Save a PDF to a file path."""
    pdf.save(str(path))


def _make_pdf_with_stamp(tmp_dir: Path, name: str = "stamped.pdf") -> Path:
    """Create a single-page PDF with a Stamp annotation on disk."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)
    annot = _make_stamp_annotation(pdf, pdf.pages[0])
    pdf.pages[0].Annots = Array([annot])

    path = tmp_dir / name
    pdf.save(str(path))
    return path


# -- TestHasAnnotations --


class TestHasAnnotations:
    """Tests for _has_annotations."""

    def test_no_annotations(self, tmp_dir: Path) -> None:
        """PDF without annotations returns False."""
        pdf = make_pdf_with_page()
        path = tmp_dir / "no_annots.pdf"
        pdf.save(str(path))

        assert _has_annotations(path) is False

    def test_with_stamp_annotation(self, tmp_dir: Path) -> None:
        """PDF with Stamp annotation returns True."""
        path = _make_pdf_with_stamp(tmp_dir)

        assert _has_annotations(path) is True

    def test_empty_annots_array(self, tmp_dir: Path) -> None:
        """PDF with empty /Annots array returns False."""
        pdf = make_pdf_with_page()
        pdf.pages[0].Annots = Array([])
        path = tmp_dir / "empty_annots.pdf"
        pdf.save(str(path))

        assert _has_annotations(path) is False

    def test_annotation_on_second_page(self, tmp_dir: Path) -> None:
        """Annotation only on page 2 of a multi-page PDF returns True."""
        pdf = new_pdf()
        # Page 1: no annotations
        page1 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page1)
        # Page 2: with annotation
        page2 = pikepdf.Page(
            Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
        )
        pdf.pages.append(page2)
        annot = _make_stamp_annotation(pdf, pdf.pages[1])
        pdf.pages[1].Annots = Array([annot])

        path = tmp_dir / "annot_page2.pdf"
        pdf.save(str(path))

        assert _has_annotations(path) is True


# -- TestStripAnnotationsForOcr --


class TestStripAnnotationsForOcr:
    """Tests for _strip_annotations_for_ocr."""

    def test_removes_annots_from_all_pages(self, tmp_dir: Path) -> None:
        """Annotations are removed from every page."""
        pdf = new_pdf()
        for _ in range(2):
            page = pikepdf.Page(
                Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
            )
            pdf.pages.append(page)
            annot = _make_stamp_annotation(pdf, pdf.pages[-1])
            pdf.pages[-1].Annots = Array([annot])
        src = tmp_dir / "src.pdf"
        pdf.save(str(src))

        clean = tmp_dir / "clean.pdf"
        result = _strip_annotations_for_ocr(src, clean)

        assert result is True
        with pikepdf.open(clean) as cleaned:
            for page in cleaned.pages:
                assert page.get("/Annots") is None

    def test_removes_acroform(self, tmp_dir: Path) -> None:
        """AcroForm is removed from Root."""
        pdf = make_pdf_with_page()
        widget = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 20]),
                T="field1",
            )
        )
        pdf.pages[0].Annots = Array([widget])
        pdf.Root["/AcroForm"] = Dictionary(Fields=Array([widget]))

        src = tmp_dir / "form.pdf"
        pdf.save(str(src))

        clean = tmp_dir / "clean_form.pdf"
        _strip_annotations_for_ocr(src, clean)

        with pikepdf.open(clean) as cleaned:
            assert "/AcroForm" not in cleaned.Root

    def test_returns_true_when_stripped(self, tmp_dir: Path) -> None:
        """Returns True when annotations were present and removed."""
        path = _make_pdf_with_stamp(tmp_dir)
        clean = tmp_dir / "clean.pdf"

        assert _strip_annotations_for_ocr(path, clean) is True

    def test_returns_false_for_no_annotations(self, tmp_dir: Path) -> None:
        """Returns False when there are no annotations to strip."""
        pdf = make_pdf_with_page()
        src = tmp_dir / "no_annots.pdf"
        pdf.save(str(src))

        clean = tmp_dir / "clean.pdf"
        assert _strip_annotations_for_ocr(src, clean) is False

    def test_page_content_preserved(self, tmp_dir: Path) -> None:
        """Page content streams remain unchanged after stripping."""
        pdf = new_pdf()
        content_data = b"q 1 0 0 1 0 0 cm /Im0 Do Q"
        page_dict = Dictionary(
            Type=Name.Page,
            MediaBox=Array([0, 0, 612, 792]),
        )
        content_stream = pdf.make_stream(content_data)
        page_dict[Name.Contents] = content_stream
        page = pikepdf.Page(page_dict)
        pdf.pages.append(page)

        annot = _make_stamp_annotation(pdf, pdf.pages[0])
        pdf.pages[0].Annots = Array([annot])

        src = tmp_dir / "content.pdf"
        pdf.save(str(src))

        clean = tmp_dir / "clean.pdf"
        _strip_annotations_for_ocr(src, clean)

        with pikepdf.open(clean) as cleaned:
            stream = cleaned.pages[0].get("/Contents")
            assert stream is not None
            assert b"Do" in bytes(stream.read_bytes())


# -- TestRestoreAnnotationsAfterOcr --


class TestRestoreAnnotationsAfterOcr:
    """Tests for _restore_annotations_after_ocr."""

    def test_stamp_roundtrip(self, tmp_dir: Path) -> None:
        """Stamp annotation Subtype, Rect, AP/N survive roundtrip."""
        original = _make_pdf_with_stamp(tmp_dir, "original.pdf")

        # Simulate OCR output (same structure, no annotations)
        ocr_pdf = make_pdf_with_page()
        ocr_path = tmp_dir / "ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "merged.pdf"
        count = _restore_annotations_after_ocr(original, ocr_path, output)

        assert count == 1
        with pikepdf.open(output) as merged:
            annots = merged.pages[0].get("/Annots")
            assert annots is not None
            annot = resolve(annots[0])
            assert str(annot["/Subtype"]) == "/Stamp"
            assert len(annot["/Rect"]) == 4
            assert "/N" in annot["/AP"]

    def test_popup_reference_preserved(self, tmp_dir: Path) -> None:
        """Markup annotation's /Popup resolves to valid Popup in output."""
        pdf = make_pdf_with_page()
        popup = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Popup,
                Rect=Array([200, 700, 400, 800]),
            )
        )
        markup = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                Popup=popup,
                F=4,
            )
        )
        popup["/Parent"] = markup
        pdf.pages[0].Annots = Array([markup, popup])

        original = tmp_dir / "popup_orig.pdf"
        pdf.save(str(original))

        ocr_pdf = make_pdf_with_page()
        ocr_path = tmp_dir / "popup_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "popup_merged.pdf"
        count = _restore_annotations_after_ocr(original, ocr_path, output)

        assert count == 2
        with pikepdf.open(output) as merged:
            annots = merged.pages[0]["/Annots"]
            markup_out = resolve(annots[0])
            popup_ref = resolve(markup_out["/Popup"])
            assert str(popup_ref["/Subtype"]) == "/Popup"

    def test_irt_chain_preserved(self, tmp_dir: Path) -> None:
        """/IRT reference between annotations remains valid."""
        pdf = make_pdf_with_page()
        original_annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 700, 120, 720]),
                F=4,
            )
        )
        reply_annot = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Text,
                Rect=Array([100, 680, 120, 700]),
                IRT=original_annot,
                F=4,
            )
        )
        pdf.pages[0].Annots = Array([original_annot, reply_annot])

        original = tmp_dir / "irt_orig.pdf"
        pdf.save(str(original))

        ocr_pdf = make_pdf_with_page()
        ocr_path = tmp_dir / "irt_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "irt_merged.pdf"
        count = _restore_annotations_after_ocr(original, ocr_path, output)

        assert count == 2
        with pikepdf.open(output) as merged:
            annots = merged.pages[0]["/Annots"]
            reply = resolve(annots[1])
            irt_target = resolve(reply["/IRT"])
            assert str(irt_target["/Subtype"]) == "/Text"

    def test_p_reference_points_to_target_page(self, tmp_dir: Path) -> None:
        """/P reference points to the target page, not the original."""
        original = _make_pdf_with_stamp(tmp_dir, "p_orig.pdf")

        ocr_pdf = make_pdf_with_page()
        ocr_path = tmp_dir / "p_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "p_merged.pdf"
        _restore_annotations_after_ocr(original, ocr_path, output)

        with pikepdf.open(output) as merged:
            annot = resolve(merged.pages[0]["/Annots"][0])
            # /P is optional; if QPDF preserves it, verify it references
            # the target page rather than a stale foreign page.
            if "/P" in annot:
                page_ref = resolve(annot["/P"])
                actual_page = resolve(merged.pages[0].obj)
                assert page_ref.objgen == actual_page.objgen

    def test_acroform_copied(self, tmp_dir: Path) -> None:
        """AcroForm (/Fields, /DR, /DA) is copied to output."""
        pdf = make_pdf_with_page()
        widget = pdf.make_indirect(
            Dictionary(
                Type=Name.Annot,
                Subtype=Name.Widget,
                Rect=Array([0, 0, 100, 20]),
                T="field1",
            )
        )
        pdf.pages[0].Annots = Array([widget])
        da_string = pikepdf.String("0 0 0 rg /Helv 12 Tf")
        pdf.Root["/AcroForm"] = pdf.make_indirect(
            Dictionary(
                Fields=Array([widget]),
                DA=da_string,
                DR=Dictionary(Font=Dictionary()),
            )
        )

        original = tmp_dir / "acro_orig.pdf"
        pdf.save(str(original))

        ocr_pdf = make_pdf_with_page()
        ocr_path = tmp_dir / "acro_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "acro_merged.pdf"
        _restore_annotations_after_ocr(original, ocr_path, output)

        with pikepdf.open(output) as merged:
            acroform = merged.Root.get("/AcroForm")
            assert acroform is not None
            af = resolve(acroform)
            assert "/Fields" in af
            assert "/DA" in af
            assert "/DR" in af

    def test_page_count_mismatch_returns_zero(self, tmp_dir: Path) -> None:
        """Page count mismatch returns 0 and OCR file is unchanged."""
        original = _make_pdf_with_stamp(tmp_dir, "mismatch_orig.pdf")

        # OCR output with 2 pages
        ocr_pdf = new_pdf()
        for _ in range(2):
            page = pikepdf.Page(
                Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
            )
            ocr_pdf.pages.append(page)
        ocr_path = tmp_dir / "mismatch_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "mismatch_merged.pdf"
        count = _restore_annotations_after_ocr(original, ocr_path, output)

        assert count == 0

    def test_no_annotations_returns_zero(self, tmp_dir: Path) -> None:
        """PDF without annotations returns 0."""
        pdf = make_pdf_with_page()
        original = tmp_dir / "no_annots_orig.pdf"
        pdf.save(str(original))

        ocr_pdf = make_pdf_with_page()
        ocr_path = tmp_dir / "no_annots_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "no_annots_merged.pdf"
        count = _restore_annotations_after_ocr(original, ocr_path, output)

        assert count == 0

    def test_multipage_selective_annotations(self, tmp_dir: Path) -> None:
        """Annotations on pages 1 and 3 only; page 2 has no /Annots."""
        pdf = new_pdf()
        for _ in range(3):
            page = pikepdf.Page(
                Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
            )
            pdf.pages.append(page)

        # Pages 0 and 2 get annotations, page 1 does not
        for idx in (0, 2):
            annot = _make_stamp_annotation(pdf, pdf.pages[idx])
            pdf.pages[idx].Annots = Array([annot])

        original = tmp_dir / "multi_orig.pdf"
        pdf.save(str(original))

        ocr_pdf = new_pdf()
        for _ in range(3):
            page = pikepdf.Page(
                Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]))
            )
            ocr_pdf.pages.append(page)
        ocr_path = tmp_dir / "multi_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "multi_merged.pdf"
        count = _restore_annotations_after_ocr(original, ocr_path, output)

        assert count == 2
        with pikepdf.open(output) as merged:
            assert merged.pages[0].get("/Annots") is not None
            assert merged.pages[1].get("/Annots") is None
            assert merged.pages[2].get("/Annots") is not None

    def test_ap_stream_survives_roundtrip(self, tmp_dir: Path) -> None:
        """AP stream (Form XObject with BBox, Resources) survives."""
        original = _make_pdf_with_stamp(tmp_dir, "ap_orig.pdf")

        ocr_pdf = make_pdf_with_page()
        ocr_path = tmp_dir / "ap_ocr.pdf"
        ocr_pdf.save(str(ocr_path))

        output = tmp_dir / "ap_merged.pdf"
        _restore_annotations_after_ocr(original, ocr_path, output)

        with pikepdf.open(output) as merged:
            annot = resolve(merged.pages[0]["/Annots"][0])
            ap_n = resolve(annot["/AP"]["/N"])
            assert str(ap_n["/Subtype"]) == "/Form"
            assert "/BBox" in ap_n
            # Verify stream data is non-empty
            assert len(ap_n.read_bytes()) > 0


# -- TestOcrAnnotationIntegration --


class TestOcrAnnotationIntegration:
    """Integration tests for annotation preservation in convert_to_pdfa."""

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    @patch("pdftopdfa.ocr.needs_ocr", return_value=True)
    def test_ocr_preserves_annotations(
        self,
        mock_needs_ocr: MagicMock,
        mock_is_available: MagicMock,
        mock_apply_ocr: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """convert_to_pdfa with OCR preserves annotations."""
        from pdftopdfa.converter import convert_to_pdfa

        original = _make_pdf_with_stamp(tmp_dir, "integration.pdf")

        def fake_apply_ocr(src: Path, dst: Path, *args, **kwargs) -> None:
            """Simulate OCR by copying the source file."""
            import shutil

            shutil.copy2(str(src), str(dst))

        mock_apply_ocr.side_effect = fake_apply_ocr

        output = tmp_dir / "output_pdfa.pdf"
        result = convert_to_pdfa(original, output, level="2b", ocr_languages=["eng"])

        assert result.success
        # Check annotations survived
        with pikepdf.open(output) as merged:
            annots = merged.pages[0].get("/Annots")
            assert annots is not None
            assert len(annots) >= 1

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    @patch("pdftopdfa.ocr.needs_ocr", return_value=True)
    def test_no_annotations_no_overhead(
        self,
        mock_needs_ocr: MagicMock,
        mock_is_available: MagicMock,
        mock_apply_ocr: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """PDF without annotations: strip is not called."""
        from pdftopdfa.converter import convert_to_pdfa

        pdf = make_pdf_with_page()
        original = tmp_dir / "plain.pdf"
        pdf.save(str(original))

        def fake_apply_ocr(src: Path, dst: Path, *args, **kwargs) -> None:
            import shutil

            shutil.copy2(str(src), str(dst))

        mock_apply_ocr.side_effect = fake_apply_ocr

        output = tmp_dir / "plain_pdfa.pdf"
        with patch("pdftopdfa.converter._strip_annotations_for_ocr") as mock_strip:
            convert_to_pdfa(original, output, level="2b", ocr_languages=["eng"])
            mock_strip.assert_not_called()

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    @patch("pdftopdfa.ocr.needs_ocr", return_value=True)
    def test_warning_message_includes_count(
        self,
        mock_needs_ocr: MagicMock,
        mock_is_available: MagicMock,
        mock_apply_ocr: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Warning includes 'annotation(s) preserved through OCR'."""
        from pdftopdfa.converter import convert_to_pdfa

        original = _make_pdf_with_stamp(tmp_dir, "warn.pdf")

        def fake_apply_ocr(src: Path, dst: Path, *args, **kwargs) -> None:
            import shutil

            shutil.copy2(str(src), str(dst))

        mock_apply_ocr.side_effect = fake_apply_ocr

        output = tmp_dir / "warn_pdfa.pdf"
        result = convert_to_pdfa(original, output, level="2b", ocr_languages=["eng"])

        matching = [
            w for w in result.warnings if "annotation(s) preserved through OCR" in w
        ]
        assert len(matching) == 1

    @patch("pdftopdfa.ocr.apply_ocr")
    @patch("pdftopdfa.ocr.is_ocr_available", return_value=True)
    @patch("pdftopdfa.ocr.needs_ocr", return_value=True)
    def test_temp_files_cleaned_up(
        self,
        mock_needs_ocr: MagicMock,
        mock_is_available: MagicMock,
        mock_apply_ocr: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Temp files (clean, merged, ocr) are cleaned up after conversion."""
        from pdftopdfa.converter import convert_to_pdfa

        original = _make_pdf_with_stamp(tmp_dir, "cleanup.pdf")

        def fake_apply_ocr(src: Path, dst: Path, *args, **kwargs) -> None:
            import shutil

            shutil.copy2(str(src), str(dst))

        mock_apply_ocr.side_effect = fake_apply_ocr

        output = tmp_dir / "cleanup_pdfa.pdf"
        convert_to_pdfa(original, output, level="2b", ocr_languages=["eng"])

        # Check no temp files remain (hidden files starting with .)
        temp_files = [
            f
            for f in Path(tempfile.gettempdir()).glob(f".{original.stem}_*")
            if f != output
        ]
        assert len(temp_files) == 0
