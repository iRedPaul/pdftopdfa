# pdftopdfa

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MPL--2.0+-blue.svg)

pdftopdfa is a free and open-source alternative to [Ghostscript](https://www.ghostscript.com/)-based PDF/A converters.
Ghostscript uses a dual license (AGPL/commercial) that makes it difficult to use in commercial products without purchasing a license.
pdftopdfa uses [MPL-2.0-or-later](https://www.mozilla.org/en-US/MPL/2.0/),
a file-level copyleft license. It permits commercial use and combination with
proprietary code, provided its terms are followed.
For non-OCR conversions, pdftopdfa modifies the PDF structure directly using
[pikepdf](https://pikepdf.readthedocs.io/) (based on
[QPDF](https://qpdf.sourceforge.io/)), avoiding full-document re-rendering and
preserving the original content, fonts, and layout where possible.

## Highlights

- **No Ghostscript required** -- direct PDF manipulation via pikepdf/QPDF
- **PDF/A-2a, 2b, 2u, 3a, 3b, 3u** -- supports modern PDF/A levels
  (ISO 19005-2 and ISO 19005-3), including Tagged PDF output for scanned
  documents
- **PDF/UA-1** -- optional dual-conformance output with PDF/A-2a or PDF/A-3a
- **Auditable PDF/UA workflow** -- fail-closed publication, explicit machine and
  author-review states, and atomic JSON evidence reports
- **Automatic font embedding** -- uses policy-approved Windows system fonts or bundled replacements
- **Font subsetting** -- reduces file size by removing unused glyphs
- **CJK support** -- embeds Noto Sans CJK for Chinese, Japanese, and Korean text
- **ICC color profiles** -- automatically embeds sRGB, CMYK, and grayscale profiles
- **XRechnung metadata** -- adds canonical Factur-X XMP metadata for recognized,
  unambiguous embedded XRechnung 3.0 invoices in PDF/A-3 output
- **Batch processing** -- converts entire directories, optionally recursive
- **Integrated validation** -- checks conformance via [veraPDF](https://verapdf.org/)
- **OCR support** -- optional PP-OCRv6 Medium text recognition on the CPU or
  through DirectML in the supported Windows 11 configuration, with external
  offline text-model directories, a bundled page-orientation model, and no
  runtime model downloads
- **Layout-aware OCR** -- optional column-based reading order for multi-column
  documents without an additional model or OCR pass
- **Table recognition** -- recognizes already-cropped bordered ("wired") and
  borderless ("wireless") tables as typed cells and HTML using only explicitly
  supplied local ONNX models
- **Simple API** -- usable as CLI tool or Python library

## How It Works

pdftopdfa applies a multi-step conversion pipeline to make a PDF compliant with the PDF/A standard:

1. **Pre-check** -- converts encrypted PDFs that open with an empty user
   password, copies password-protected and, by default, digitally signed PDFs
   unchanged, and otherwise detects if the PDF is already a valid PDF/A file
   (skips conversion if the existing conformance level meets or exceeds the
   target within the same PDF/A part; optionally skips any veraPDF-compliant
   PDF/A via `--skip-any-pdfa`; see the [Usage Guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/usage.md#already-compliant-pdfs) for details)
2. **OCR** (optional) -- optionally orients pages with the bundled PP-LCNet
   document-orientation model, straightens only scan-like raster-dominant
   pages, and recognizes text with externally supplied PP-OCRv6 Medium models;
   OCRmyPDF rasterizes OCR target pages and creates the searchable text layer
3. **Font compliance** -- analyzes all fonts, embeds missing ones, adds ToUnicode mappings, subsets embedded fonts, and fixes encoding issues
4. **Sanitization** -- removes or fixes non-compliant elements (JavaScript, non-standard actions, transparency groups, annotations, optional content, etc.)
5. **Metadata** -- synchronizes XMP metadata with the document info dictionary,
   sets the PDF/A conformance level, and optionally adds PDF/UA-1 identification
   with the required PDF/A extension schema
6. **Color profiles** -- detects color spaces and embeds the required ICC profiles (sRGB, CMYK/FOGRA39, sGray)
7. **Logical structure** -- for level A, preserves and safely repairs valid
   rich tags or builds headings, paragraphs, lists, tables, figures, artifacts,
   links, forms, and reading order from final digital-PDF provenance or the
   internal OCR engine's line and layout data. A provably inverted
   single-column block order is rebuilt; ambiguous or multi-column existing
   orders are preserved
8. **Save and validate** -- stages the output with the correct PDF version
   header, validates requested profiles, and by default atomically publishes
   only an accepted candidate

## Installation

### Prerequisites

- Python 3.12, 3.13, or 3.14
- macOS 14 or later on Apple Silicon, Linux, or Windows

Intel-based Macs are not supported. CPU OCR on macOS requires the ARM64 wheels
provided by ONNX Runtime for Apple Silicon.

```bash
python -m pip install pdftopdfa
```

If the `pdftopdfa` console script is not on `PATH`, use
`python -m pdftopdfa` in the examples below.

### PDF/A and PDF/UA validation

Validation uses the external [veraPDF](https://verapdf.org/) application, which
is not bundled. Install version 1.30.2 or newer and make its launcher available on `PATH`, or set
`VERAPDF_PATH` to the executable or its parent directory. `--validate` and
`validate=True` opt ordinary PDF/A output into validation. PDF/UA output always
attempts validation against both the selected PDF/A profile and veraPDF's `ua1`
profile. Validation is fail-closed: if a requested check cannot run or fails,
the staged candidate is not published and an existing destination is preserved.
Use `--publish-noncompliant` or `publication_policy="always"` only when an
explicitly non-conforming review candidate is required.

Passing `--audit-report report.json` writes an atomic, machine-readable record
containing the exact staged-candidate hash when available, sanitizer and tagger
counters, structured veraPDF rule findings, and raw XML for completed validator
runs. A fatal invocation replaces stale evidence with a structured
`fatal_error` record. The report distinguishes machine validation from author
review; it is evidence, not an accessibility certification.

### Optional: OCR support

Install exactly one OCR runtime. CPU inference is the default:

```bash
python -m pip install "pdftopdfa[ocr]"
```

For the supported DirectML configuration on Windows 11:

```bash
python -m pip install "pdftopdfa[directml]"
```

Do not install both extras in the same Python installation: `onnxruntime` and
`onnxruntime-directml` provide overlapping runtime files. pdftopdfa supports
DirectML on Windows 11 with a DirectX 12-capable integrated or dedicated Intel,
AMD, or NVIDIA GPU and a current graphics driver.

OCR uses PaddleOCR 3.7 with the selected ONNX Runtime provider. Installing the
DirectML extra does not select it automatically; use
`--ocr-execution-provider directml` or
`ocr_execution_provider="directml"`. CPU remains the default. If DirectML is
requested but unavailable, processing stops with an error instead of falling
back to the CPU.

On a machine with several GPUs, `directml:<index>` passes a raw DXGI adapter
index, for example `--ocr-execution-provider directml:1`. Plain `directml` uses
DirectML's default adapter. The internal diagnostic helper
`pdftopdfa._ocr_runtime.list_directml_devices()` lists the available adapters
and their raw indices; as part of a private module, it has no public API
stability guarantee. The indices may have gaps because software adapters are
omitted, and repeated DXGI entries with the same PCI identity are listed once
using their lowest index. Use the reported index, not its position in the
filtered list.

The page-orientation model is bundled. PP-OCRv6 text-recognition and table
models are external and are never downloaded at runtime. Pass their local
directories to each top-level conversion or recognition call; an `OCRSession`
instead receives the PP-OCRv6 text-model pair once when it is created and
reuses it across its image-recognition calls. CPU and DirectML use the same
FP32 ONNX model files.
See the [OCR guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/ocr.md) for the `recognize_table()` model contract and
typed result. Cell text and grid structure come from the table-structure model,
while bounding boxes come from the separate cell-detection model; if the two
models report different cell counts, cells are returned without
`bounding_box` and `confidence` instead of failing.

#### PP-OCRv6 model setup

The following model revisions are tested and recommended:

- Detection:
  [`PP-OCRv6_medium_det_onnx` at `6132380`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/tree/61323801669c338b7891481ec7bac61ce31b576a)
- Recognition:
  [`PP-OCRv6_medium_rec_onnx` at `50c7eac`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/tree/50c7eacafc52fa7bcf4194e8cd08e46f8558504b)

Each model directory must contain exactly `inference.onnx` and
`inference.yml`. Before initialization, `pdftopdfa` performs a quick
structural check that rejects missing or extra entries, non-regular files, and
symbolic links. PaddleOCR then loads the model files and checks that they are
compatible detection and recognition models. pdftopdfa does not verify the
repository revision or model-file hashes.

The models are not included in the source distribution or wheel. Keep them in
deployment-managed, read-only directories. Both
`--ocr-detection-model-dir` and `--ocr-recognition-model-dir` are required
together; supplying the pair enables OCR without an additional `--ocr` flag.
Conversely, `--ocr`, `--ocr-force`, `--deskew`, `--rotate-pages`,
`--ocr-layout`, and a non-CPU `--ocr-execution-provider` value (`directml` or
`directml:INDEX`) are rejected unless both model options are present.

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

# Tagged PDF/A-2a output with veraPDF validation
pdftopdfa -l 2a --validate document.pdf

# PDF/A-2a and PDF/UA-1 output (both profiles are always validated)
pdftopdfa -l 2a --pdfua --document-title "Annual report" \
  --document-language en-GB --audit-report document-audit.json document.pdf

# Explicitly retain a failed validation candidate (not conforming output)
pdftopdfa -l 2a --pdfua --publish-noncompliant document.pdf

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

# OCR a German/English scan to tagged PDF/A-2a and validate it
pdftopdfa -l 2a --validate --ocr-lang de+en \
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

The OCR examples above use POSIX shell syntax. For DirectML in PowerShell on
Windows 11, for example:

```powershell
$DET_MODEL = "C:\models\PP-OCRv6_medium_det_onnx"
$REC_MODEL = "C:\models\PP-OCRv6_medium_rec_onnx"
pdftopdfa --ocr-execution-provider directml `
  --ocr-detection-model-dir "$DET_MODEL" `
  --ocr-recognition-model-dir "$REC_MODEL" document.pdf
```

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa

result = convert_to_pdfa(
    input_path=Path("input.pdf"),
    output_path=Path("output.pdf"),
    level="2b",
)

accessible_result = convert_to_pdfa(
    input_path=Path("input.pdf"),
    output_path=Path("accessible.pdf"),
    level="2a",
    pdfua=True,
    document_title="Annual report",
    document_language="en-GB",
)

ocr_result = convert_to_pdfa(
    input_path=Path("scan.pdf"),
    output_path=Path("scan_pdfa.pdf"),
    level="2b",
    ocr_languages=["de", "en"],
    ocr_detection_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_det_onnx"),
    ocr_recognition_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_rec_onnx"),
    ocr_execution_provider="cpu",
)
```

Supplying both model directories enables OCR in `convert_to_pdfa()`,
`convert_files()`, and `convert_directory()`. Supplying only one directory, or
requesting OCR through `ocr_languages`, `ocr_force`, `ocr_deskew`,
`ocr_rotate_pages`, `ocr_layout=True`, or a non-CPU execution provider without
both directories, raises `ValueError` before processing starts. Set
`ocr_execution_provider="directml"` to use the supported DirectML configuration
on Windows 11, or `ocr_execution_provider="directml:1"` to pass a specific raw
DXGI adapter index.

Set `pdfa=False` to apply only the requested OCR processing. This skips font
embedding, PDF/A sanitization, metadata synchronization, color-profile
embedding, and PDF/A validation. The result is not validated or guaranteed to
be PDF/A compliant.

See the [Usage Guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/usage.md)
for the full CLI reference, conversion API documentation, and examples. The
[OCR guide](https://github.com/iRedPaul/pdftopdfa/blob/main/docs/ocr.md) covers
image, table, and reusable `OCRSession` APIs.

## Limitations

- **No PDF/A-1 support** -- only PDF/A-2 and PDF/A-3 levels are supported
- **Automatic level A semantics** -- generated tags infer headings, paragraphs,
  lists, conservative tables, figures, artifacts, links, forms, and reading
  order from text styles, geometry, direct painting provenance, and, for scans,
  the internal OCR engine's line and layout data. Ambiguous content falls back
  to conservative paragraph or division structure. Trustworthy existing Alt,
  marked-content ActualText, and textual Captions are retained or propagated;
  software cannot invent an authoritative description for an otherwise
  undescribed image. Such Figure/Formula elements are reported for manual
  review instead of receiving invented descriptions. Pages with unclassified
  vector painting are likewise reported because decorative rules and
  meaningful vector diagrams cannot always be distinguished automatically.
  Full-page OCR scans that may contain unrecognized non-text visuals, Link
  annotations that cannot be associated with one content owner, and form fields
  without a trustworthy tooltip or field name are retained and reported for the
  same reason.
  Review automatically inferred semantics for accessibility-critical
  publications. PDF/A level A does not by itself imply PDF/UA conformance;
  request the additional PDF/UA-1 requirements with `--pdfua` or
  `pdfua=True`. A `review_required` result means both machine profiles passed
  but one or more semantics still require an author decision; it is not a
  certification. veraPDF cannot judge whether content order, descriptions,
  labels, language, contrast, color use, or media alternatives are meaningful.
  PDF/UA-2 generation is not implemented: it requires a separate PDF 2.0 and
  PDF/A-4 output track. The low-level validator API can inspect an existing
  file with the `ua2` profile.
- **Encrypted PDFs** -- encryption is removed from PDFs that open with an empty
  user password. PDFs that require a password cannot be converted and are
  ordinarily copied unchanged. With an automatically generated output name, the unchanged copy
  still receives the `_pdfa.pdf` suffix; it is not a converted PDF/A file.
  A requested PDF/UA target returns `success=False` and
  `pdfua_status="not_produced"` and preserves the destination without
  publishing the protected input. `--publish-noncompliant` explicitly enables
  unchanged copy-through.
- **Digitally signed PDFs** -- signed PDFs are ordinarily copied unchanged because conversion would invalidate their signatures. A requested PDF/UA target returns `success=False` and `pdfua_status="not_produced"` without publishing by default; use `--allow-signature-invalidation` only when an unsigned archival copy is intentional
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
python -m pip install -e ".[dev,ocr]"
```

### Running Tests

```bash
python -m pytest
```

The test suite covers fonts, color profiles, metadata, sanitization, OCR, and
end-to-end conversion.

CI additionally installs the checksum-pinned veraPDF 1.30.2 distribution and
runs a real dual PDF/A/PDF/UA-1 conversion gate rather than mocking the
validator boundary.

The real-model semantic OCR tests are opt-in and never modify the configured
model directories. Set `PDFTOPDFA_TEST_OCR_DETECTION_MODEL_DIR` and
`PDFTOPDFA_TEST_OCR_RECOGNITION_MODEL_DIR`, install veraPDF, then run:

```bash
python -m pytest tests/test_semantic_ocr_e2e.py
```

With the upstream `veraPDF-corpus-staging` checkout present at the repository
root, the corpus runner converts and validates every fixture against every
supported PDF/A level and writes a detailed report. It also records the exact
corpus hashes and execution environment in `run_metadata.json`; set
`PDFTOPDFA_CORPUS_WORKERS` to override the default worker count capped at eight:

```bash
python run_corpus_test.py
```

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
- [pdfminer.six](https://github.com/pdfminer/pdfminer.six) -- read-only digital
  text, layout, and painting-provenance extraction
- [NumPy](https://numpy.org/) -- Array processing for OCR and table recognition
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
- [PaddleX](https://github.com/PaddlePaddle/PaddleX) -- local-model OCR and
  table-recognition pipelines
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) -- PDF page rasterizer for OCR
- [veraPDF](https://verapdf.org/) -- external application for PDF/A and PDF/UA
  machine validation

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
