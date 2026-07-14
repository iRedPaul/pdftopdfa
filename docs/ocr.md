# OCR Guide

This guide covers OCR behavior in `pdftopdfa` for scanned/image-based PDFs.

General CLI and API usage is documented in [Usage Guide](usage.md).

## Prerequisites

Install OCR extras:

```bash
pip install "pdftopdfa[ocr]"
```

System dependency:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed and available in `PATH` for text recognition and deskew.

The PaddleOCR document-orientation classifier and its ONNX model are installed
with `pdftopdfa`. The model is read exclusively from the package and does not
require an internet connection or a pre-populated user cache.

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

# Automatically orient pages without deskewing
pdftopdfa --rotate-pages scan.pdf

# Deskew pages without automatic orientation
pdftopdfa --deskew scan.pdf

# Enable both independent page-processing steps
pdftopdfa --deskew --rotate-pages scan.pdf

# Apply both steps without the PDF/A conversion pipeline
pdftopdfa --no-pdfa --deskew --rotate-pages scan.pdf
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
    ocr_deskew=False,
    ocr_rotate_pages=True,
)
```

Use `pdfa=False` and a neutral output name such as `scan_processed.pdf` to
write the OCR result directly without font embedding, PDF/A sanitization,
metadata synchronization, color-profile embedding, or PDF/A validation. The
OCR behavior itself is unchanged, so image-only pages can still receive a text
layer. The output is not guaranteed to be PDF/A compliant.

## When OCR Runs

If OCR is enabled (`--ocr`, `--deskew`, or `--rotate-pages`, or a corresponding
Python option is set), `pdftopdfa` always invokes OCR processing for the
document. The two page-processing options use English when no OCR language is
provided.

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
    ocr_quality=OcrQuality.DEFAULT,
)
```

Behavior notes:

- `--ocr-force` implies `--ocr`.
- Existing OCR text layers are replaced.
- Forced OCR bypasses the "already compliant PDF/A" skip path so OCR still runs.
- Forced OCR does not bypass the signed-PDF protection. Use
  `--allow-signature-invalidation` only when an unsigned OCR/PDF/A copy is
  intentional.
- `--deskew` cannot be combined with `--ocr-force`, because OCRmyPDF's
  `redo_ocr` mode does not support deskewing.
- Original annotations are preserved when possible.

## Quality Presets

| Preset | Goal | Visual changes |
|---|---|---|
| `fast` | Fastest processing | No |
| `default` | Better recognition | No |

Internal OCR settings:

| Parameter | `fast` | `default` |
|---|---|---|
| `oversample` | - | 600 |
| `tesseract_pagesegmode` | - | 11 (`sparse_text`) |
| `tesseract_thresholding` | - | `adaptive-otsu` |

`default` relies on Tesseract's built-in adaptive thresholding. This is
generally more robust for mixed scans with small text regions on large pages.
Quality presets never enable deskewing or page rotation.

## Page Processing

`--deskew` and `--rotate-pages` are independent, opt-in operations. Both imply
`--ocr`, remain active if OCR retries with the faster fallback preset, and
bypass the already-compliant PDF/A skip check so the requested processing runs.
With `--no-pdfa`, the processed OCR result is written directly instead of
continuing through the PDF/A conversion pipeline.

`--deskew` enables OCRmyPDF's skew correction and also straightens eligible
text-only pages that OCRmyPDF skips. Deskewing may alter page appearance and
increase file size.

PDFs containing annotations are OCRed without deskewing, and a warning is
emitted. Annotation rectangles, markup quadrilaterals, form widgets, and
appearance streams cannot be mapped reliably through OCRmyPDF's raster deskew
transformation, so skipping deskew prevents links and interactive content from
becoming misaligned.

Before OCR starts, `--rotate-pages` renders and classifies every page with the
bundled `PP-LCNet_x1_0_doc_ori` model, including pages that OCRmyPDF later skips
because they already contain text. Predictions below 0.80 leave the page
unchanged and produce a warning. Missing, corrupt, or unusable model files abort
the conversion. Tesseract OSD and OCRmyPDF's own `rotate_pages` option are not
used.

## Offline Deployment

The ONNX model, its inference configuration, integrity manifest, source notice,
and Apache-2.0 license are package data. Wheel builds contain these files under
`pdftopdfa/resources/models/PP-LCNet_x1_0_doc_ori/`.

Applications frozen with tools such as PyInstaller must collect the `pdftopdfa`
package data as well as PaddleOCR, PaddleX, ONNX Runtime, OpenCV, NumPy, and the
ONNX Runtime native libraries. No writable model cache is required at runtime.

## Automatic Fallback

By default, OCR retries with the `fast` preset when the selected preset takes
longer than 60 seconds. The initial Tesseract OCR timeout is capped to this
threshold so fallback can happen promptly instead of waiting for the longer
quality-preset timeout.

```bash
# Default behavior: default first, fast fallback after 60 seconds
pdftopdfa --ocr scan.pdf

# Change the fallback threshold
pdftopdfa --rotate-pages --ocr-fallback-after 120 scan.pdf

# Disable automatic retry
pdftopdfa --deskew --ocr-fallback-quality none scan.pdf
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

### Bundled Paddle model is missing or corrupt

Reinstall `pdftopdfa` from an intact wheel. The orientation preflight validates
the packaged files against SHA-256 hashes and intentionally does not attempt a
network download or external-cache fallback.

### OCR did not run

Possible reasons:

- OCR was not enabled (`--ocr`, `--deskew`, or `--rotate-pages` missing).
- The document already had sufficient text coverage.
- The file had fewer than 50% OCR-relevant pages.
- The input PDF was digitally signed and was skipped to avoid invalidating the
  signature.

Use `--ocr-force` to enforce OCR.
