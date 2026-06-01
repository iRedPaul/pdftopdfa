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

# Highest OCR quality, but retry with fast OCR if it takes over 60 seconds
pdftopdfa --ocr --ocr-quality best --ocr-fallback-quality fast scan.pdf
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
    ocr_fallback_quality=OcrQuality.FAST,
    ocr_fallback_after_seconds=60.0,
)
```

## When OCR Runs

If OCR is enabled (`--ocr` or `ocr_languages` is set), `pdftopdfa` always invokes OCR processing for the document:

- Pages that already contain text are skipped automatically.
- Pages without a text layer are OCRed.
- This makes mixed documents work page by page instead of using a document-wide threshold.

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
- Forced OCR bypasses the "already compliant PDF/A" skip path so OCR still runs.
- Forced OCR does not bypass the signed-PDF protection. Use
  `--allow-signature-invalidation` only when an unsigned OCR/PDF/A copy is
  intentional.
- With forced OCR, options incompatible with ocrmypdf's `redo_ocr`
  mode are disabled automatically.
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
| `oversample` | - | 600 | 600 |
| `tesseract_pagesegmode` | - | 11 (`sparse_text`) | 11 (`sparse_text`) |
| `tesseract_thresholding` | - | `adaptive-otsu` | `adaptive-otsu` |

`default` and `best` rely on Tesseract's built-in adaptive thresholding.
This is generally more robust for mixed scans with small text regions on large pages.
`best` now includes the same OCR detection baseline as `default` and adds
deskew/rotation on top.
When OCR is forced, redo-ocr-incompatible options such as `deskew`
are disabled automatically.

## Automatic Fallback

By default, OCR retries with the `fast` preset when the selected preset takes
longer than 60 seconds. The initial Tesseract OCR timeout is capped to this
threshold so fallback can happen promptly instead of waiting for the longer
quality-preset timeout.

```bash
# Default behavior: best first, fast fallback after 60 seconds
pdftopdfa --ocr --ocr-quality best scan.pdf

# Use default as fallback from best
pdftopdfa --ocr --ocr-quality best --ocr-fallback-quality default scan.pdf

# Change the fallback threshold
pdftopdfa --ocr --ocr-quality best --ocr-fallback-after 120 scan.pdf

# Disable automatic retry
pdftopdfa --ocr --ocr-quality best --ocr-fallback-quality none scan.pdf
```

Fallback presets only run when they are faster than the selected preset. For
example, `--ocr-quality fast --ocr-fallback-quality default` is ignored because
`default` is not a faster fallback.

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
- The input PDF was digitally signed and was skipped to avoid invalidating the
  signature.

Use `--ocr-force` to enforce OCR.
