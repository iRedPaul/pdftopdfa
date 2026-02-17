# OCR for Scanned PDFs

pdftopdfa can add a text layer to scanned PDFs using [Tesseract](https://github.com/tesseract-ocr/tesseract) via [ocrmypdf](https://ocrmypdf.readthedocs.io/). Pages that already contain text are skipped automatically.

## Installation

```bash
pip install "pdftopdfa[ocr]"
```

Tesseract must be installed on the system. You can specify a custom path to the Tesseract executable or its parent directory via the `TESSERACT_PATH` environment variable.

## Usage

### Command Line

```bash
# English (default)
pdftopdfa --ocr document.pdf

# German
pdftopdfa --ocr --ocr-lang deu document.pdf

# Multilingual
pdftopdfa --ocr --ocr-lang deu+eng document.pdf
```

### Python API

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa
from pdftopdfa.ocr import OcrQuality

result = convert_to_pdfa(
    input_path=Path("scan.pdf"),
    output_path=Path("scan_pdfa.pdf"),
    level="2b",
    ocr_language="eng",
    ocr_quality=OcrQuality.BEST,
)
```

## Quality Presets

| Preset | Description | Alters document visually? |
|--------|-------------|---------------------------|
| `fast` | Minimal processing, fastest | No |
| `default` | Best quality without visual changes | No |
| `best` | Best quality, may deskew/rotate pages | Yes |

### Detailed Parameter Mapping

The presets map to the following ocrmypdf parameters:

| ocrmypdf parameter | `fast` | `default` | `best` |
|--------------------|--------|-----------|--------|
| `skip_text` | True | True | True |
| `deskew` | False | False | True |
| `rotate_pages` | False | False | True |
| `oversample` | - | 300 | 300 |
| `optimize` | 0 | 1 | 1 |
| OpenCV preprocessing | No | Yes | Yes |

## Image Preprocessing

The `default` and `best` presets automatically preprocess page images before OCR using OpenCV (installed as part of `pdftopdfa[ocr]`).

The preprocessing pipeline applies:

1. **Grayscale conversion** -- color images are converted to grayscale
2. **Denoising** -- `cv2.fastNlMeansDenoising` removes scanner noise
3. **Adaptive thresholding** -- `cv2.adaptiveThreshold` with Gaussian method produces a clean binary image

The preprocessing only affects the image that Tesseract sees for recognition. The original page images in the PDF remain unchanged.

If OpenCV is not installed, preprocessing is skipped with a warning and OCR still runs normally.
