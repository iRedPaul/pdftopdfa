# pdftopdfa

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MPL--2.0+-blue.svg)

I built pdftopdfa as a free and open-source alternative to [Ghostscript](https://www.ghostscript.com/)-based PDF/A converters.
Ghostscript uses a dual license (AGPL/commercial) that makes it difficult to use in commercial products without purchasing a license.
pdftopdfa is licensed under the permissive [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/) and can be freely used in commercial projects.
Instead of re-rendering via Ghostscript, it modifies the PDF structure directly using [pikepdf](https://pikepdf.readthedocs.io/) (based on [QPDF](https://qpdf.sourceforge.io/)), preserving the original content, fonts, and layout.                                                                                                                                  

## Highlights

- **No Ghostscript required** -- direct PDF manipulation via pikepdf/QPDF
- **PDF/A-2a, 2b, 2u, 3a, 3b, 3u** -- supports modern PDF/A levels
  (ISO 19005-2 and ISO 19005-3), including Tagged PDF output for scanned
  documents
- **Automatic font embedding** -- uses policy-approved Windows system fonts or bundled replacements
- **Font subsetting** -- reduces file size by removing unused glyphs
- **CJK support** -- embeds Noto Sans CJK for Chinese, Japanese, and Korean text
- **ICC color profiles** -- automatically embeds sRGB, CMYK, and grayscale profiles
- **Batch processing** -- converts entire directories, optionally recursive
- **Integrated validation** -- checks conformance via [veraPDF](https://verapdf.org/)
- **OCR support** -- optional PP-OCRv6 Medium text recognition on the CPU or
  through DirectML on Windows 11, with explicit offline model directories and
  no runtime model downloads
- **Layout-aware OCR** -- optional column-based reading order for multi-column
  documents without an additional model or OCR pass
- **Table recognition** -- recognizes already-cropped wired and wireless
  tables as typed cells and HTML using only explicitly supplied local ONNX
  models
- **Simple API** -- usable as CLI tool or Python library

## How It Works

pdftopdfa applies a multi-step conversion pipeline to make a PDF compliant with the PDF/A standard:

1. **Pre-check** -- skips encrypted and digitally signed PDFs, then detects if the PDF is already a valid PDF/A file (skips conversion if the existing level meets or exceeds the target; optionally skips any veraPDF-compliant PDF/A via `--skip-any-pdfa`; see the [Usage Guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/usage.md#already-compliant-pdfs) for details)
2. **OCR** (optional) -- optionally orients pages with the bundled PaddleOCR
   orientation model, straightens only scan-like raster-dominant pages, and
   recognizes text with externally supplied PP-OCRv6 Medium models; OCRmyPDF
   rasterizes OCR target pages and creates the searchable text layer
3. **Font compliance** -- analyzes all fonts, embeds missing ones, adds ToUnicode mappings, subsets embedded fonts, and fixes encoding issues
4. **Sanitization** -- removes or fixes non-compliant elements (JavaScript, non-standard actions, transparency groups, annotations, optional content, etc.)
5. **Metadata** -- synchronizes XMP metadata with the document info dictionary and sets the PDF/A conformance level
6. **Color profiles** -- detects color spaces and embeds the required ICC profiles (sRGB, CMYK/FOGRA39, sGray)
7. **Logical structure** -- for level A, preserves an existing Tagged PDF
   structure or creates a structure tree from the final page and annotation
   order, including OCR-processed scans
8. **Save** -- writes the output with the correct PDF version header

## Installation

### Prerequisites

- Python 3.12, 3.13, or 3.14
- macOS 14 or later on Apple Silicon, Linux, or Windows

Intel-based Macs are not supported. CPU OCR on macOS requires the ARM64 wheels
provided by ONNX Runtime for Apple Silicon.

```bash
pip install pdftopdfa
```

### Optional: OCR support

Install exactly one OCR runtime. CPU inference is the default:

```bash
pip install "pdftopdfa[ocr]"
```

For DirectML inference on Windows 11:

```bash
pip install "pdftopdfa[directml]"
```

Do not install both extras in the same Python installation: `onnxruntime` and
`onnxruntime-directml` provide overlapping runtime files. DirectML requires
Windows 11, a DirectX 12-capable integrated or dedicated Intel, AMD, or NVIDIA
GPU, and a current graphics driver.

OCR uses PaddleOCR 3.7 with the selected ONNX Runtime provider. Installing the
DirectML extra does not select it automatically; use
`--ocr-execution-provider directml` or
`ocr_execution_provider="directml"`. CPU remains the default. If DirectML is
requested but unavailable, processing stops with an error instead of falling
back to the CPU.

On a machine with several GPUs, `directml:<index>` picks a specific adapter,
for example `--ocr-execution-provider directml:1`. Plain `directml` leaves the
choice to DirectML. `pdftopdfa._ocr_runtime.list_directml_devices()` lists the
available adapters with the raw DXGI index each one uses. These indices may
have gaps because software adapters are omitted. Repeated DXGI entries with
the same PCI identity are listed once using their lowest index.

OCR and table recognition do not download models at runtime. Every model must
be obtained separately and passed through an explicit local directory on each
public API invocation. CPU and DirectML use the same FP32 ONNX model files.
See the [OCR guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/ocr.md) for the `recognize_table()` model contract and
typed result. Cell text and grid structure come from the table-structure model,
while bounding boxes come from the separate cell-detection model; if the two
models report different cell counts, cells are returned without
`bounding_box` and `confidence` instead of failing.

#### PP-OCRv6 model setup

Download the files from these exact model revisions:

- Detection:
  [`PP-OCRv6_medium_det_onnx` at `6132380`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/tree/61323801669c338b7891481ec7bac61ce31b576a)
- Recognition:
  [`PP-OCRv6_medium_rec_onnx` at `50c7eac`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/tree/50c7eacafc52fa7bcf4194e8cd08e46f8558504b)

Each model directory must contain exactly `inference.onnx` and
`inference.yml`. Before initialization, `pdftopdfa` performs a quick
structural check that rejects missing or extra entries, non-regular files, and
symbolic links. PaddleOCR then loads the model files and checks that they are
compatible detection and recognition models.

The models are not included in the source distribution or wheel. Keep them in
deployment-managed, read-only directories. Both
`--ocr-detection-model-dir` and `--ocr-recognition-model-dir` are required
together; supplying the pair enables OCR without an additional `--ocr` flag.
Conversely, `--ocr`, `--ocr-force`, `--deskew`, and `--rotate-pages` are
rejected unless both model options are present.

`--ocr-lang` defaults to `en`. Use `de` for German and `de+en` for mixed
German/English recognition. Latin-script languages restrict decoding to Latin
letters while retaining numbers, punctuation, and symbols, which prevents
Chinese-character output on German scans. The accepted PaddleOCR 3.7 codes are:

`af`, `az`, `bs`, `ca`, `ch`, `chinese_cht`, `cs`, `cy`, `da`, `de`, `en`,
`es`, `et`, `eu`, `fi`, `fr`, `french`, `ga`, `german`, `gl`, `hr`, `hu`,
`id`, `is`, `it`, `japan`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`,
`nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`,
`sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`.

Legacy codes such as `eng` and `deu` are not accepted. See the
[PaddleOCR language documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md)
for the language families represented by these codes.

## Quick Start

```bash
# Simple conversion (creates document_pdfa.pdf)
pdftopdfa document.pdf

# Specific PDF/A level
pdftopdfa -l 2b document.pdf

# Accessible PDF/A output
pdftopdfa -l 2a document.pdf

# With validation (note: -v = --validate, not verbose; use --verbose for logs)
pdftopdfa -v document.pdf

# Skip any existing veraPDF-compliant PDF/A
pdftopdfa --skip-any-pdfa document.pdf

# Explicitly convert a signed PDF, invalidating its digital signatures
pdftopdfa --allow-signature-invalidation document.pdf

# Convert an entire directory
pdftopdfa -r ./documents/ ./output/

# The OCR examples below use the externally managed model directories
DET_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_det_onnx
REC_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_rec_onnx

# OCR a German/English scan to tagged PDF/A-2a
pdftopdfa -l 2a --ocr-lang de+en \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  document.pdf

# Order OCR lines by detected columns for a cleaner reading order
pdftopdfa --ocr-layout \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  document.pdf

# Use the same models through DirectML on Windows 11
pdftopdfa --ocr-execution-provider directml \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  document.pdf

# Automatically orient pages without deskewing them
pdftopdfa --rotate-pages \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  document.pdf

# Deskew pages without changing their 90-degree orientation
pdftopdfa --deskew \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  document.pdf

# Deskew and orient pages without converting the result to PDF/A
# (creates document_processed.pdf)
pdftopdfa --no-pdfa --deskew --rotate-pages \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  document.pdf

# Preserve known proprietary stamps as PDF Stamp annotations
pdftopdfa --preserve-stamps document.pdf
```

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa

result = convert_to_pdfa(
    input_path=Path("input.pdf"),
    output_path=Path("output.pdf"),
    level="2b",
)

ocr_result = convert_to_pdfa(
    input_path=Path("scan.pdf"),
    output_path=Path("scan_pdfa.pdf"),
    level="2b",
    ocr_languages=["de", "en"],
    ocr_detection_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_det_onnx"
    ),
    ocr_recognition_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_rec_onnx"
    ),
    ocr_execution_provider="cpu",
)
```

Supplying both model directories enables OCR in `convert_to_pdfa()`,
`convert_files()`, and `convert_directory()`. Supplying only one directory, or
requesting an OCR option without both directories, raises `ValueError` before
processing starts. Set `ocr_execution_provider="directml"` to use an
installation made with the `directml` extra on Windows 11, or
`ocr_execution_provider="directml:1"` to pin a specific adapter.

Set `pdfa=False` to apply only the requested OCR processing. This skips font
embedding, PDF/A sanitization, metadata synchronization, color-profile
embedding, and PDF/A validation. The result is not validated or guaranteed to
remain PDF/A compliant.

See the [Usage Guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/usage.md) for the full CLI reference, Python API documentation, and examples.

## Limitations

- **No PDF/A-1 support** -- only PDF/A-2 and PDF/A-3 levels are supported
- **Automatic level A semantics** -- generated tags follow page and PDF
  content-stream order and include annotations. They provide the structural
  basis required by PDF/A-2a and PDF/A-3a, but automatic conversion cannot
  infer authorial semantics such as heading levels, table relationships, or
  alternative descriptions. PDF/A level A does not imply PDF/UA conformance.
- **Encrypted PDFs** -- password-protected PDFs cannot be converted
- **Digitally signed PDFs** -- signed PDFs are copied unchanged by default because conversion would invalidate their signatures; use `--allow-signature-invalidation` only when an unsigned archival copy is intentional
- **Font replacement** -- fonts without a suitable metrically compatible replacement produce a warning; the resulting file may not be fully compliant
- **Non-embedded CIDFonts (Identity encoding)** -- content streams reference glyph IDs of the original font; after replacement with a substitute font the same glyph IDs point to different or missing glyphs, so the affected text may render incorrectly or invisibly. Text extraction and copy/paste stay correct because the original ToUnicode mapping is preserved. A warning is emitted for each replaced CIDFont

## Font Sourcing

- On Windows, `pdftopdfa` may automatically embed a conservative fixed allowlist of local fonts from `%WINDIR%\Fonts`.
- A Windows system font is only used when the installed file lives under `%WINDIR%\Fonts`, its actual PostScript name is allowlisted, and its OpenType `fsType` permits outline embedding.
- On macOS and Linux, system fonts are never auto-embedded; bundled replacement fonts are used instead.
- `fsType` checks are a technical safeguard only and do not replace the font vendor's EULA or other license terms.
- For auditable deployments, keep the allowlist tied to reviewed target systems or golden images.

## Development

```bash
pip install -e ".[dev,ocr]"
```

### Running Tests

```bash
pytest
```

The test suite contains 2600+ tests covering fonts, color profiles, metadata, sanitization, and end-to-end conversion.

### Code Quality

```bash
ruff check src/ tests/   # Linting
ruff format src/ tests/  # Formatting
```

## Documentation

Additional documentation is available in the [docs/](https://github.com/iRedPaul/pdftopdfa/tree/main/docs) folder:

- [Usage Guide (CLI & Python API)](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/usage.md)
- [OCR Guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/ocr.md)
- [PDF/A-2/3 rules reference (veraPDF)](https://github.com/veraPDF/veraPDF-validation-profiles/wiki/PDFA-Parts-2-and-3-rules)

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/iredpaul/pdftopdfa/issues) to report bugs or suggest features, or submit a pull request.

## Dependencies

**Core:**

- [pikepdf](https://pikepdf.readthedocs.io/) -- PDF manipulation (based on QPDF)
- [lxml](https://lxml.de/) -- XMP metadata processing
- [fonttools](https://github.com/fonttools/fonttools) -- Font analysis, subsetting, and embedding
- [click](https://click.palletsprojects.com/) -- CLI framework
- [colorama](https://pypi.org/project/colorama/) -- Colored terminal output
- [tqdm](https://tqdm.github.io/) -- Progress bars
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) -- document
  orientation and PP-OCRv6 text recognition

**Optional:**

- [OCRmyPDF](https://ocrmypdf.readthedocs.io/) -- PDF rasterization, text-layer
  generation, and page merging for optional OCR
- [ONNX Runtime](https://onnxruntime.ai/) -- CPU or DirectML inference for
  Paddle models
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) -- PDF page rasterizer for OCR
- [veraPDF](https://verapdf.org/) -- ISO-compliant PDF/A validation

## Acknowledgments

This project bundles the following resources:

- **[Liberation Fonts](https://github.com/liberationfonts/liberation-fonts)** -- metrically compatible replacements for the PDF Standard 14 fonts (SIL OFL 1.1)
- **[PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori)** -- bundled document-orientation model (Apache-2.0)
- **[Noto Sans CJK](https://github.com/notofonts/noto-cjk)** -- CJK font coverage (SIL OFL 1.1)
- **[Noto Sans Symbols 2](https://github.com/notofonts/symbols)** -- symbol font replacement (SIL OFL 1.1)
- **[STIX Two Math](https://github.com/stipub/stixfonts)** -- math font replacement (SIL OFL 1.1)
- **[sRGB2014.icc](https://registry.color.org/rgb-registry/srgbprofiles)** -- ICC sRGB profile (ICC)
- **[ISOcoated_v2_300_bas.icc](https://www.eci.org/en/downloads)** -- ICC CMYK profile, FOGRA39 (zlib/libpng license)
- **[sGray](https://github.com/saucecontrol/Compact-ICC-Profiles)** -- compact grayscale ICC profile (CC0-1.0)
- **[Adobe cmap-resources](https://github.com/adobe-type-tools/cmap-resources)** -- CID-to-Unicode mapping data (BSD 3-Clause)

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) or later (MPL-2.0+) -- see [LICENSE](LICENSE) for details.
