# OCR Guide

This guide covers OCR behavior in `pdftopdfa` for scanned/image-based PDFs.

General CLI and API usage is documented in [Usage Guide](usage.md).

## Prerequisites

Install OCR extras:

```bash
pip install "pdftopdfa[ocr]"
```

System dependency:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed and available in `PATH`.

Optional environment variable:

- `TESSERACT_PATH`: path to the `tesseract` executable or its parent directory.

## Usage

### CLI

```bash
# English (default)
pdftopdfa --ocr scan.pdf

# German
pdftopdfa --ocr --ocr-lang deu scan.pdf

# Multilingual
pdftopdfa --ocr --ocr-lang deu+eng scan.pdf

# Highest OCR quality
pdftopdfa --ocr --ocr-quality best scan.pdf
```

### Python API

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa
from pdftopdfa.ocr import OcrQuality

result = convert_to_pdfa(
    input_path=Path("scan.pdf"),
    output_path=Path("scan_pdfa.pdf"),
    ocr_languages=["eng"],
    ocr_quality=OcrQuality.DEFAULT,
)
```

## When OCR Runs

If OCR is enabled (`--ocr` or `ocr_languages` is set), `pdftopdfa` checks whether OCR is needed:

- A page is considered OCR-relevant if it has images and no text operators.
- OCR runs if at least 50% of pages are OCR-relevant.

If OCR is not needed, conversion continues without OCR.

## Force OCR

Use force mode when a document already has a poor OCR layer and you want to regenerate it.

### CLI

```bash
pdftopdfa --ocr-force --ocr-lang deu scan.pdf
```

### Python API

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa
from pdftopdfa.ocr import OcrQuality

result = convert_to_pdfa(
    input_path=Path("scan.pdf"),
    output_path=Path("scan_pdfa.pdf"),
    ocr_languages=["deu"],
    ocr_force=True,
    ocr_quality=OcrQuality.BEST,
)
```

Behavior notes:

- `--ocr-force` implies `--ocr`.
- Existing OCR text layers are replaced.
- Original annotations are preserved when possible.

## Quality Presets

| Preset | Goal | Visual changes |
|---|---|---|
| `fast` | Fastest processing | No |
| `default` | Better recognition without changing page appearance | No |
| `best` | Highest recognition quality | Possible (deskew/rotation) |

Internal OCR settings:

| Parameter | `fast` | `default` | `best` |
|---|---|---|---|
| `deskew` | False | False | True |
| `rotate_pages` | False | False | True |
| `rotate_pages_threshold` | - | - | 5.0 |
| `oversample` | - | 300 | 200 |
| OpenCV preprocessing | No | Yes | Yes |

`default` and `best` use OpenCV preprocessing when available.
If OpenCV is unavailable, OCR still runs and preprocessing is skipped.

## Troubleshooting

### `OCR not available - pip install pdftopdfa[ocr]`

Install OCR extras:

```bash
pip install "pdftopdfa[ocr]"
```

### Tesseract not found

- Install Tesseract on your system.
- Ensure `tesseract --version` works.
- Or set `TESSERACT_PATH` to the executable or parent directory.

### OCR did not run

Possible reasons:

- OCR was not enabled (`--ocr` missing).
- The document already had sufficient text coverage.
- The file had fewer than 50% OCR-relevant pages.

Use `--ocr-force` to enforce OCR.
