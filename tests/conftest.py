# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pytest fixtures for the pdftopdfa test suite."""

from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name, Pdf

# -- Global PDF tracker --

_tracked_pdfs: list[Pdf] = []


@pytest.fixture(autouse=True)
def _auto_close_pdfs():
    """Close all tracked PDF objects after each test."""
    yield
    for pdf in reversed(_tracked_pdfs):
        try:
            pdf.close()
        except Exception:
            pass
    _tracked_pdfs.clear()


def new_pdf(**kwargs) -> Pdf:
    """Create a tracked Pdf (auto-closed after test)."""
    pdf = Pdf.new(**kwargs)
    _tracked_pdfs.append(pdf)
    return pdf


def open_pdf(source, **kwargs) -> Pdf:
    """Open a tracked Pdf (auto-closed after test)."""
    pdf = Pdf.open(source, **kwargs)
    _tracked_pdfs.append(pdf)
    return pdf


# -- Fixtures --


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for tests.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path to the temporary directory.
    """
    return tmp_path


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Minimal valid PDF as bytes.

    Returns:
        PDF data as bytes.
    """
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    buffer = BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def sample_pdf(tmp_dir: Path, sample_pdf_bytes: bytes) -> Path:
    """Minimal valid PDF on disk.

    Args:
        tmp_dir: Temporary directory.
        sample_pdf_bytes: PDF data as bytes.

    Returns:
        Path to the PDF file.
    """
    pdf_path = tmp_dir / "sample.pdf"
    pdf_path.write_bytes(sample_pdf_bytes)
    return pdf_path


@pytest.fixture
def sample_pdf_obj(sample_pdf_bytes: bytes) -> Generator[Pdf, None, None]:
    """Open pikepdf.Pdf object.

    Args:
        sample_pdf_bytes: PDF data as bytes.

    Yields:
        Open Pdf object.
    """
    buffer = BytesIO(sample_pdf_bytes)
    pdf = Pdf.open(buffer)
    yield pdf
    pdf.close()


@pytest.fixture
def encrypted_pdf(tmp_dir: Path) -> Path:
    """Encrypted PDF for error tests.

    Args:
        tmp_dir: Temporary directory.

    Returns:
        Path to the encrypted PDF file.
    """
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    encrypted_path = tmp_dir / "encrypted.pdf"
    pdf.save(encrypted_path, encryption=pikepdf.Encryption(owner="testpassword"))
    return encrypted_path


@pytest.fixture
def pdf_with_metadata(tmp_dir: Path) -> Path:
    """PDF with Info-Dictionary metadata.

    Args:
        tmp_dir: Temporary directory.

    Returns:
        Path to the PDF file with metadata.
    """
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    # Set metadata in Info-Dictionary
    with pdf.open_metadata() as meta:
        meta["dc:title"] = "Test Title"
        meta["dc:creator"] = ["Test Author"]
        meta["dc:description"] = "Test Description"

    # Also set classic Info-Dictionary values
    pdf.docinfo["/Title"] = "Test Title"
    pdf.docinfo["/Author"] = "Test Author"
    pdf.docinfo["/Subject"] = "Test Description"
    pdf.docinfo["/Creator"] = "Test Creator"
    pdf.docinfo["/Producer"] = "Test Producer"
    pdf.docinfo["/CreationDate"] = "D:20240115120000+00'00'"
    pdf.docinfo["/ModDate"] = "D:20240115130000+00'00'"

    pdf_path = tmp_dir / "with_metadata.pdf"
    pdf.save(pdf_path)
    return pdf_path


@pytest.fixture
def pdf_with_javascript(tmp_dir: Path) -> Path:
    """PDF with JavaScript OpenAction.

    Args:
        tmp_dir: Temporary directory.

    Returns:
        Path to the PDF file with JavaScript.
    """
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    # Add JavaScript OpenAction
    pdf.Root.OpenAction = Dictionary(
        S=Name.JavaScript,
        JS="app.alert('Hello World');",
    )

    pdf_path = tmp_dir / "with_javascript.pdf"
    pdf.save(pdf_path)
    return pdf_path


@pytest.fixture
def pdf_with_image(tmp_dir: Path) -> Path:
    """PDF with embedded image (simulates scanned document).

    Creates a PDF with an image XObject but without text in the content stream.
    This simulates a scanned document that needs OCR.

    Args:
        tmp_dir: Temporary directory.

    Returns:
        Path to the PDF file with image.
    """
    pdf = new_pdf()

    # Create a minimal 1x1 pixel grayscale image
    image_data = b"\x80"  # Gray pixel
    image_stream = pdf.make_stream(image_data)
    image_stream[Name.Type] = Name.XObject
    image_stream[Name.Subtype] = Name.Image
    image_stream[Name.Width] = 1
    image_stream[Name.Height] = 1
    image_stream[Name.ColorSpace] = Name.DeviceGray
    image_stream[Name.BitsPerComponent] = 8

    # Create page with the image
    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(
            XObject=Dictionary(Im0=image_stream),
        ),
    )

    # Content stream that draws the image (but no text)
    content_data = b"q 100 0 0 100 100 600 cm /Im0 Do Q"
    content_stream = pdf.make_stream(content_data)
    page_dict[Name.Contents] = content_stream

    page = pikepdf.Page(page_dict)
    pdf.pages.append(page)

    pdf_path = tmp_dir / "with_image.pdf"
    pdf.save(pdf_path)
    return pdf_path


@pytest.fixture
def pdf_with_image_obj(pdf_with_image: Path) -> Generator[Pdf, None, None]:
    """Open pikepdf.Pdf object with image.

    Args:
        pdf_with_image: Path to the PDF with image.

    Yields:
        Open Pdf object.
    """
    pdf = Pdf.open(pdf_with_image)
    yield pdf
    pdf.close()


@pytest.fixture
def pdf_with_text(tmp_dir: Path) -> Path:
    """PDF with text content.

    Creates a PDF with text operators (Tj/TJ) in the content stream.

    Args:
        tmp_dir: Temporary directory.

    Returns:
        Path to the PDF file with text.
    """
    pdf = new_pdf()

    # Create minimal font reference
    font_dict = Dictionary(
        Type=Name.Font,
        Subtype=Name.Type1,
        BaseFont=Name("/Helvetica"),
    )

    page_dict = Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, 612, 792]),
        Resources=Dictionary(
            Font=Dictionary(F1=font_dict),
        ),
    )

    # Content stream with text operators (Tj)
    content_data = b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET"
    content_stream = pdf.make_stream(content_data)
    page_dict[Name.Contents] = content_stream

    page = pikepdf.Page(page_dict)
    pdf.pages.append(page)

    pdf_path = tmp_dir / "with_text.pdf"
    pdf.save(pdf_path)
    return pdf_path


@pytest.fixture
def pdf_with_text_obj(pdf_with_text: Path) -> Generator[Pdf, None, None]:
    """Open pikepdf.Pdf object with text.

    Args:
        pdf_with_text: Path to the PDF with text.

    Yields:
        Open Pdf object.
    """
    pdf = Pdf.open(pdf_with_text)
    yield pdf
    pdf.close()


@pytest.fixture
def empty_pdf_obj() -> Generator[Pdf, None, None]:
    """Empty pikepdf.Pdf object without pages.

    Yields:
        Open empty Pdf object.
    """
    pdf = Pdf.new()
    yield pdf
    pdf.close()


@pytest.fixture
def pdf_with_annotations(tmp_dir: Path) -> Path:
    """Creates a PDF with various annotation types for testing.

    Creates a PDF with:
    - Link annotation (without Print flag)
    - Widget annotation (form field, Print flag should be added)
    - Text annotation (with Print flag already set)

    Args:
        tmp_dir: Temporary directory.

    Returns:
        Path to the PDF file with annotations.
    """
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)

    # Link annotation without Print flag (F=0)
    link_annot = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Link,
            Rect=Array([100, 700, 200, 720]),
            F=0,
        )
    )

    # Widget annotation (form field) - Print flag should be added
    widget_annot = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Widget,
            Rect=Array([100, 650, 200, 670]),
            F=0,
        )
    )

    # Text annotation with Print flag already set (F=4)
    text_annot = pdf.make_indirect(
        Dictionary(
            Type=Name.Annot,
            Subtype=Name.Text,
            Rect=Array([100, 600, 120, 620]),
            F=4,  # Print flag is bit 3 (value 4)
        )
    )

    pdf.pages[0].Annots = Array([link_annot, widget_annot, text_annot])

    pdf_path = tmp_dir / "with_annotations.pdf"
    pdf.save(pdf_path)
    return pdf_path


# -- Shared test helpers (not fixtures) --


def make_pdf_with_page() -> Pdf:
    """Create a minimal PDF with one page (auto-tracked)."""
    pdf = new_pdf()
    page = pikepdf.Page(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792])))
    pdf.pages.append(page)
    return pdf


@pytest.fixture(name="make_pdf_with_page")
def _make_pdf_with_page_fixture():
    return make_pdf_with_page


def resolve(obj: object) -> object:
    """Safely resolve an indirect reference."""
    try:
        return obj.get_object()
    except (AttributeError, TypeError, ValueError):
        return obj


def save_and_reopen(pdf: Pdf) -> Pdf:
    """Save a PDF to bytes and reopen it (auto-tracked)."""
    buf = BytesIO()
    pdf.save(buf)
    pdf.close()
    buf.seek(0)
    return open_pdf(buf)


@pytest.fixture(name="save_and_reopen")
def _save_and_reopen_fixture():
    return save_and_reopen
