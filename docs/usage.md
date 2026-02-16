# Usage Guide

## Command Line

### Basic Usage

```bash
# Convert a single file (creates document_pdfa.pdf)
pdftopdfa document.pdf

# With explicit output path
pdftopdfa input.pdf output.pdf

# Specific PDF/A level
pdftopdfa -l 2b document.pdf

# With validation via veraPDF
pdftopdfa -v document.pdf

# Force overwrite existing output
pdftopdfa -f document.pdf
```

### Batch Processing

```bash
# Convert all PDFs in a directory
pdftopdfa ./input-folder/ ./output-folder/

# Recursive directory processing
pdftopdfa -r ./documents/

# Force overwrite + verbose output
pdftopdfa -r -f --verbose ./documents/ ./output/
```

### OCR for Scanned PDFs

```bash
# English (default language)
pdftopdfa --ocr document.pdf

# German
pdftopdfa --ocr --ocr-lang deu document.pdf

# Multilingual
pdftopdfa --ocr --ocr-lang deu+eng document.pdf

# Best quality (may deskew/rotate pages)
pdftopdfa --ocr --ocr-quality best document.pdf
```

See [docs/ocr.md](ocr.md) for details on OCR quality presets.

### CLI Options

| Option | Description |
|--------|-------------|
| `-l, --level [2b\|2u\|3b\|3u]` | PDF/A conformance level (default: 3b) |
| `-v, --validate` | Validate output with veraPDF after conversion |
| `-r, --recursive` | Process directories recursively |
| `-f, --force` | Overwrite existing output files |
| `-q, --quiet` | Only output errors |
| `--verbose` | Detailed logging output |
| `--ocr` | Enable OCR for scanned PDFs (requires ocrmypdf) |
| `--ocr-lang LANG` | OCR language code (default: eng) |
| `--ocr-quality [fast\|default\|best]` | OCR quality preset (default: default) |
| `--no-convert-calibrated` | Disable CalGray/CalRGB to ICCBased conversion |
| `--version` | Show version and exit |
| `--help` | Show help message and exit |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (invalid arguments, missing dependencies) |
| 2 | Input file not found |
| 3 | Conversion failed |
| 4 | Validation failed (veraPDF returned non-compliant) |
| 5 | Insufficient permissions |

---

## Python API

### Single File Conversion

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa

result = convert_to_pdfa(
    input_path=Path("input.pdf"),
    output_path=Path("output.pdf"),
    level="2b",
)

if result.success:
    print(f"Converted in {result.processing_time:.2f}s")
    for warning in result.warnings:
        print(f"Warning: {warning}")
else:
    print(f"Error: {result.error}")
```

### `convert_to_pdfa()`

```python
def convert_to_pdfa(
    input_path: Path,
    output_path: Path,
    level: str = "3b",
    *,
    validate: bool = False,
    ocr_language: str | None = None,
    ocr_quality: OcrQuality | None = None,
    convert_calibrated: bool = True,
) -> ConversionResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `Path` | *required* | Path to the input PDF |
| `output_path` | `Path` | *required* | Path for the output PDF/A |
| `level` | `str` | `"3b"` | PDF/A level: `"2b"`, `"2u"`, `"3b"`, or `"3u"` |
| `validate` | `bool` | `False` | Validate output with veraPDF |
| `ocr_language` | `str \| None` | `None` | OCR language (e.g. `"eng"`, `"deu+eng"`) |
| `ocr_quality` | `OcrQuality \| None` | `None` | OCR quality preset |
| `convert_calibrated` | `bool` | `True` | Convert CalGray/CalRGB to ICCBased |

### `convert_directory()`

```python
def convert_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    level: str = "3b",
    *,
    recursive: bool = False,
    validate: bool = False,
    show_progress: bool = True,
    ocr_language: str | None = None,
    ocr_quality: OcrQuality | None = None,
    force_overwrite: bool = False,
    convert_calibrated: bool = True,
) -> list[ConversionResult]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | `Path` | *required* | Directory containing PDF files |
| `output_dir` | `Path \| None` | `None` | Output directory (defaults to input directory) |
| `level` | `str` | `"3b"` | PDF/A level: `"2b"`, `"2u"`, `"3b"`, or `"3u"` |
| `recursive` | `bool` | `False` | Process subdirectories |
| `validate` | `bool` | `False` | Validate output with veraPDF |
| `show_progress` | `bool` | `True` | Show tqdm progress bar |
| `ocr_language` | `str \| None` | `None` | OCR language (e.g. `"eng"`, `"deu+eng"`) |
| `ocr_quality` | `OcrQuality \| None` | `None` | OCR quality preset |
| `force_overwrite` | `bool` | `False` | Overwrite existing output files |
| `convert_calibrated` | `bool` | `True` | Convert CalGray/CalRGB to ICCBased |

**Example:**

```python
from pathlib import Path
from pdftopdfa import convert_directory

results = convert_directory(
    input_dir=Path("./input-pdfs/"),
    output_dir=Path("./output-pdfs/"),
    level="3b",
    recursive=True,
)

for result in results:
    status = "OK" if result.success else f"FAILED: {result.error}"
    print(f"{result.input_path.name}: {status}")
```

### `generate_output_path()`

```python
def generate_output_path(
    input_path: Path,
    output_dir: Path | None = None,
) -> Path
```

Generates an output path by appending `_pdfa` to the input filename. If `output_dir` is provided, the output is placed in that directory.

```python
from pdftopdfa.converter import generate_output_path

generate_output_path(Path("report.pdf"))
# => Path("report_pdfa.pdf")

generate_output_path(Path("report.pdf"), Path("./output/"))
# => Path("output/report_pdfa.pdf")
```

### `ConversionResult`

Dataclass returned by `convert_to_pdfa()` and `convert_directory()`.

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | `True` if conversion succeeded |
| `input_path` | `Path` | Path to the input PDF |
| `output_path` | `Path` | Path to the output PDF/A |
| `level` | `str` | PDF/A level used (e.g. `"3b"`) |
| `warnings` | `list[str]` | Warning messages from conversion |
| `processing_time` | `float` | Processing time in seconds |
| `error` | `str \| None` | Error message if `success=False` |
| `validation_failed` | `bool` | `True` if veraPDF validation failed |

### OCR Quality Presets

```python
from pdftopdfa.ocr import OcrQuality

OcrQuality.FAST     # Minimal processing, fastest
OcrQuality.DEFAULT  # Best quality without visual changes
OcrQuality.BEST     # Best quality, may deskew/rotate pages
```

---

## Exceptions

All exceptions inherit from `PDFToPDFAError`:

```python
from pdftopdfa.exceptions import (
    PDFToPDFAError,       # Base exception
    ConversionError,      # General conversion failure
    ValidationError,      # Validation failed
    FontEmbeddingError,   # Font embedding failed
    UnsupportedPDFError,  # Encrypted or unsupported PDF
    OCRError,             # OCR processing failed
    VeraPDFError,         # veraPDF integration error
)
```

**Example:**

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa
from pdftopdfa.exceptions import UnsupportedPDFError, ConversionError

try:
    result = convert_to_pdfa(Path("input.pdf"), Path("output.pdf"))
except UnsupportedPDFError:
    print("PDF is encrypted or unsupported")
except ConversionError as e:
    print(f"Conversion failed: {e}")
```

---

## PDF/A Levels

| Level | ISO Standard | Attachments | Unicode Required | Recommended For |
|-------|--------------|-------------|------------------|-----------------|
| **2b** | ISO 19005-2 | PDF/A-1 only | No | Basic archiving |
| **2u** | ISO 19005-2 | PDF/A-1 only | Yes | Searchable archives |
| **3b** | ISO 19005-3 | Any format | No | Embedding original data (e.g. XML invoices) |
| **3u** | ISO 19005-3 | Any format | Yes | Searchable archives with attachments |

The default level is **3b**.

- **"b" levels** (basic) ensure visual reproduction.
- **"u" levels** (unicode) additionally require every text glyph to have a Unicode mapping, enabling reliable text extraction and search.

---

## Validation

pdftopdfa uses [veraPDF](https://verapdf.org/) for ISO-compliant PDF/A validation. veraPDF must be installed separately and available in `PATH`, or you can set the `VERAPDF_PATH` environment variable.

```bash
# Validate after conversion
pdftopdfa -v document.pdf
```

```python
result = convert_to_pdfa(
    input_path=Path("input.pdf"),
    output_path=Path("output.pdf"),
    validate=True,
)

if result.validation_failed:
    print("Output did not pass veraPDF validation")
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VERAPDF_PATH` | Custom path to the veraPDF executable or its parent directory |
| `TESSERACT_PATH` | Custom path to the Tesseract executable or its parent directory (for OCR) |
