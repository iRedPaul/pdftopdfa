# Usage Guide

This guide covers everyday usage of `pdftopdfa` from the command line and Python.

For OCR model setup and processing behavior, see [OCR Guide](ocr.md).

OCR users must install exactly one runtime extra: `pdftopdfa[ocr]` for CPU or
`pdftopdfa[directml]` for the project's supported DirectML configuration on
Windows 11. CPU is the default execution provider; DirectML must be selected
explicitly.

For the full rule lists, see the veraPDF references for
[PDF/A-2 and PDF/A-3](https://github.com/veraPDF/veraPDF-validation-profiles/wiki/PDFA-Parts-2-and-3-rules)
and [PDF/UA-1](https://github.com/veraPDF/veraPDF-validation-profiles/wiki/PDFUA-Part-1-rules).

## Basic Usage

### Convert One File

```bash
# Creates input_pdfa.pdf next to input.pdf
pdftopdfa input.pdf

# Explicit output file
pdftopdfa input.pdf output.pdf

# Target a specific level
pdftopdfa -l 2b input.pdf

# Validate output with veraPDF
pdftopdfa -v input.pdf

# Create dual-conformance PDF/A-2a and PDF/UA-1 output
pdftopdfa --level 2a --pdfua --validate input.pdf

# Overwrite an existing output
pdftopdfa -f input.pdf output.pdf

# Apply OCR page processing without PDF/A conversion
pdftopdfa --no-pdfa --deskew --rotate-pages \
  --ocr-detection-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_det \
  --ocr-recognition-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_rec \
  input.pdf
```

If the console script is not on `PATH`, the module entry point
`python -m pdftopdfa` accepts the same arguments.

### Batch Processing

```bash
# Convert all PDFs in a directory
pdftopdfa ./input-dir/ ./output-dir/

# Convert recursively
pdftopdfa -r ./documents/

# Recursive, verbose, and overwrite existing outputs
pdftopdfa -r -f --verbose ./documents/ ./output/
```

## Output Paths and Overwrite Rules

- Default output filename is `<input_stem>_pdfa.pdf`, or
  `<input_stem>_processed.pdf` with `--no-pdfa`.
- Single-file conversion without explicit output writes next to the input file.
- Directory conversion without explicit output writes into the same directory.
- Recursive directory conversion with an explicit output directory preserves subdirectory structure.
- With same-directory output (`output_dir=None`), generated `_pdfa.pdf` or
  `_processed.pdf` files are skipped when the corresponding source file is
  present, avoiding reconversion loops without excluding standalone inputs
  whose names happen to use those suffixes.
- The CLI and batch APIs do not overwrite existing outputs unless
  `-f/--force` or `force_overwrite=True`, respectively, is used.
- Direct `convert_to_pdfa()` calls replace an existing, distinct output file;
  that API has no overwrite-protection option.
- Do not pass the source path or a hard link to it as the output path; actual
  processing requires distinct input and output file identities.

## Font Policy

- On Windows, `pdftopdfa` may automatically embed a conservative fixed allowlist of local fonts from `%WINDIR%\Fonts`.
- A Windows system font is only used when the file path stays under `%WINDIR%\Fonts`, its actual PostScript name is allowlisted, and its `fsType` permits outline embedding.
- If `fsType` disallows outline embedding, `pdftopdfa` falls back to the bundled replacement font path.
- On macOS and Linux, system fonts are never auto-embedded.
- `fsType` checks are a technical safeguard only and do not replace the rights holder's EULA or other license terms.
- For auditable deployments, tie the allowlist to validated target systems or golden images.

## CLI Reference

### Arguments

| Argument | Description |
|---|---|
| `input_path` | Input PDF file or input directory |
| `output` | Optional output PDF file or output directory |

### Options

| Option | Description |
|---|---|
| `-l, --level [2a\|2b\|2u\|3a\|3b\|3u]` | Target PDF/A level (default: `3b`) |
| `-v, --validate` | Validate output with veraPDF. Note: unlike the common CLI convention, `-v` does **not** mean verbose — use `--verbose` for detailed logs |
| `--pdfua` | Also produce PDF/UA-1; requires PDF/A level `2a` or `3a`. With `--validate`, both conformance profiles are checked |
| `--no-pdfa` | Apply requested OCR processing without running the PDF/A conversion pipeline |
| `-r, --recursive` | Process directories recursively |
| `-f, --force` | Overwrite existing output files |
| `-q, --quiet` | Show only errors |
| `--verbose` | Enable detailed logs |
| `--ocr` | Enable OCR for scanned/image-based PDFs; requires both model-directory options |
| `--ocr-force` | Replace an existing OCR layer; requires both model-directory options, implies `--ocr`, and cannot be combined with `--deskew` |
| `--ocr-lang LANG` | PaddleOCR language code (default: `en`), for example `de` or `de+en`; does not enable OCR by itself |
| `--ocr-detection-model-dir DIR` | Directory containing compatible PP-OCRv6 Medium detection `inference.onnx` and `inference.yml` |
| `--ocr-recognition-model-dir DIR` | Directory containing compatible PP-OCRv6 Medium recognition `inference.onnx` and `inference.yml` |
| `--ocr-execution-provider [cpu\|directml\|directml:INDEX]` | ONNX Runtime execution provider (default: `cpu`); any non-CPU provider requires both model-directory options; DirectML uses the `directml` extra and is project-supported on Windows 11, while `directml:INDEX` selects a specific raw DXGI adapter index |
| `--ocr-layout` | Order OCR lines by detected page columns without another OCR pass; requires both model-directory options and implies `--ocr`; see [OCR Page Layout](ocr.md#page-layout) |
| `--deskew` | Straighten scan-like, raster-dominant pages; requires both model-directory options and implies `--ocr` |
| `--rotate-pages` | Automatically orient pages with the bundled Paddle model; requires both model-directory options and implies `--ocr` |
| `--convert-calibrated/--no-convert-calibrated` | Convert CalGray/CalRGB to ICCBased (default: enabled) |
| `--preserve-stamps` | Convert known proprietary stamp annotations to standard PDF Stamp annotations instead of flattening them |
| `--skip-any-pdfa` | Skip inputs that veraPDF validates as any compliant PDF/A, regardless of target level |
| `--allow-signature-invalidation` | Convert digitally signed PDFs even though conversion removes or invalidates their signatures |
| `--version` | Show version and exit |
| `--help` | Show help and exit |

Providing both model-directory options enables OCR even when `--ocr` is
omitted. Providing only one model directory is an error.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | General error |
| `2` | CLI usage or argument error, including a nonexistent input path |
| `3` | Conversion failed |
| `4` | Validation failed |
| `5` | Permission error |

## Python API

### `convert_to_pdfa()`

```python
from pathlib import Path
from pdftopdfa import PDFToPDFAError, convert_to_pdfa

try:
    result = convert_to_pdfa(
        input_path=Path("input.pdf"),
        output_path=Path("output.pdf"),
        level="2a",
        pdfua=True,
        validate=True,
    )
except PDFToPDFAError as exc:
    print(f"Conversion failed: {exc}")
else:
    if result.validation_failed:
        print("Validation failed; review the published candidate")
    else:
        print("Done")
```

`convert_to_pdfa()` raises on conversion failure. If explicit validation fails,
the candidate remains published at the requested destination and the result has
`success=False`, `validation_failed=True`, and a validation error message.
Validation failure therefore differs from conversion failure because it still
produces a reviewable candidate.
The batch APIs represent handled per-file failures with `success=False` and an
`error` message so that later files can still be processed.

Signature:

```python
def convert_to_pdfa(
    input_path: Path,
    output_path: Path,
    level: str = "3b",
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    validate: bool = False,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> ConversionResult
```

To run OCR, provide both external model directories. The pair enables OCR;
`ocr_languages` defaults to `["en"]`:

```python
from pathlib import Path

from pdftopdfa import convert_to_pdfa

result = convert_to_pdfa(
    input_path=Path("scan.pdf"),
    output_path=Path("scan_pdfa.pdf"),
    ocr_languages=["de", "en"],
    ocr_detection_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_det"),
    ocr_recognition_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_rec"),
    ocr_execution_provider="cpu",
)
```

Set `ocr_execution_provider="directml"` to use the project's supported
DirectML configuration on Windows 11 after installing
`pdftopdfa[directml]`. The same FP32 ONNX model directories are used for both
providers. If DirectML is requested but unavailable, the call raises
`OCRError` instead of falling back to CPU. Use `"directml:<index>"`, for
example `"directml:1"`, to pass a specific raw DXGI adapter index; see
[docs/ocr.md](ocr.md#selecting-a-gpu).

Pass `pdfa=False` to keep the existing OCR behavior while skipping all
PDF/A-specific processing. If no OCR option is selected, the input is copied
unchanged. In this mode `ConversionResult.level` is `None`, `validate=True` is
rejected, and PDF/A-specific settings such as `level`, `skip_any_pdfa`,
`convert_calibrated`, and `preserve_stamps` are not applied. Existing PDF/A
identification metadata is not deliberately removed, but the output is not
validated or guaranteed to conform to PDF/A.

Requesting OCR processing disables the document-level optimization that copies
an already-compliant PDF/A unchanged. Page selection still happens normally:
without `ocr_force=True`, pages with meaningful non-whitespace text are
skipped; image pages whose extracted text is only whitespace still receive
OCR.

By default, non-standard visible annotations are flattened into page content for
maximum archival robustness. Set `preserve_stamps=True` or pass
`--preserve-stamps` to convert known proprietary stamp annotations to standard
`/Stamp` annotations instead.

For PDF/A-2a and PDF/A-3a, `pdftopdfa` preserves a valid rich Tagged PDF
structure when the page content remains unchanged and locally repairs safe,
missing properties such as table-header scope. Trustworthy existing Alt,
marked-content ActualText, and textual Captions are preserved or propagated.
An otherwise valid single-column structure is rebuilt only when final-content
provenance and geometry prove that its block order is inverted. Ambiguous and
multi-column existing orders are preserved rather than guessed.
For untagged digital PDFs it binds tags to final direct text, image, Form, and
annotation operations, then infers headings, paragraphs, lists, conservative
tables, figures, artifacts, links, forms, and reading order from styles and
geometry. OCR-processed pages use the internal OCR engine's line MCIDs, text,
confidence, geometry, language, and layout ordering. Mixed documents combine
both evidence sources page by page.

Pass `pdfua=True` or `--pdfua` with either Level A target to add PDF/UA-1
identification, the required PDF/A extension schema, and a document-title
fallback when the source has none. PDF/UA mode always runs veraPDF once for the
selected PDF/A profile and once for `ua1`, even without `validate=True` or
`--validate`. If veraPDF is unavailable or either profile fails, the candidate
is still published with `success=False` and the failure is reported. The program implements
PDF/UA-1; PDF/UA-2 requires
PDF 2.0 and is outside the PDF/A-2/3 output contract.

Strongly evidenced paragraph, list, and table continuations are joined across
page breaks. A structure element that genuinely spans pages has no `/Pg` entry;
its marked-content references and page-local descendants retain their exact
page references.

Automatic semantics remain an inference. Ambiguous layouts deliberately use
conservative paragraph or division structure. An image or formula without a
trustworthy existing Alt, ActualText, or Caption is left without an invented
description and reported for manual review. Accessibility-critical output
should therefore receive author review. Pages containing unclassified
vector painting are also reported because a decorative rule cannot always be
distinguished reliably from a meaningful vector diagram. PDF/A level A does
not by itself imply PDF/UA conformance. OCR pages with a full-page scan are
reported when the available OCR layout evidence cannot rule out an unrecognized
photo or diagram. Link annotations that cannot be associated safely with
content owned by one logical structure element are likewise retained and
reported instead of reassigning heading or paragraph content speculatively.
Form fields without a trustworthy tooltip or field name receive a generic
structural tooltip and are reported for author review rather than receiving an
invented accessible name.

### `convert_directory()`

```python
from pathlib import Path
from pdftopdfa import convert_directory

results = convert_directory(
    input_dir=Path("./input"),
    output_dir=Path("./output"),
    level="3b",
    recursive=True,
)

for r in results:
    print(r.input_path.name, "OK" if r.success else r.error)
```

Signature:

```python
def convert_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    level: str = "3b",
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    recursive: bool = False,
    validate: bool = False,
    skip_any_pdfa: bool = False,
    show_progress: bool = True,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    force_overwrite: bool = False,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> list[ConversionResult]
```

### `convert_files()`

```python
from pathlib import Path
from pdftopdfa import convert_files

pairs = [
    (Path("a.pdf"), Path("a_pdfa.pdf")),
    (Path("b.pdf"), Path("b_pdfa.pdf")),
]

results = convert_files(pairs, level="3b", force_overwrite=True)
```

Signature:

```python
def convert_files(
    file_pairs: list[tuple[Path, Path]],
    level: str = "3b",
    *,
    pdfa: bool = True,
    pdfua: bool = False,
    validate: bool = False,
    skip_any_pdfa: bool = False,
    ocr_languages: list[str] | None = None,
    ocr_detection_model_dir: Path | None = None,
    ocr_recognition_model_dir: Path | None = None,
    ocr_force: bool = False,
    ocr_deskew: bool = False,
    ocr_rotate_pages: bool = False,
    ocr_execution_provider: str = "cpu",
    ocr_layout: bool = False,
    force_overwrite: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    convert_calibrated: bool = True,
    preserve_stamps: bool = False,
    allow_signature_invalidation: bool = False,
) -> list[ConversionResult]
```

Before processing, `convert_files()` requires unique output paths and rejects
any output path that overlaps any batch input, including through a hard link.
These conflicts raise `ConversionError` even with `force_overwrite=True`. An
existing output without overwrite permission instead produces a per-file
result with `success=False`, and processing continues. `on_progress` is called
before each file with `(zero_based_index, total, filename)`. If `cancel_event`
is set, processing stops before the next file and the results accumulated so
far are returned.

### `needs_ocr()`

Library helper to analyze whether a PDF would benefit from OCR. It is not
called by the conversion pipeline itself. OCR is opt-in through a complete
detection/recognition model-directory pair, with optional language and
page-processing settings; use this helper to decide programmatically whether
to provide that pair.

```python
import pikepdf
from pdftopdfa.ocr import needs_ocr

with pikepdf.open("scan.pdf") as pdf:
    if needs_ocr(pdf, threshold=0.5):
        ...  # convert with both model directories and ocr_languages=["en"]
```

Signature:

```python
def needs_ocr(pdf: pikepdf.Pdf, *, threshold: float = 0.5) -> bool
```

A page counts as needing OCR when it contains images but no non-whitespace text
operands. `needs_ocr()` returns `True` when at least `threshold` (0.0-1.0) of
the pages need OCR.

### `ConversionResult`

`ConversionResult` is returned by all conversion APIs.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | `True` only if processing and every requested validation succeeded; a failed validation still publishes its candidate with `success=False` |
| `input_path` | `Path` | Input file path |
| `output_path` | `Path` | Output file path |
| `level` | `str \| None` | Requested level for converted output, detected level for a compliant skip, or `None` when no PDF/A level was produced or detected, such as a protected input copied unchanged or `pdfa=False` output |
| `warnings` | `list[str]` | Non-fatal conversion warnings |
| `processing_time` | `float` | Runtime in seconds |
| `error` | `str \| None` | Error message for a handled per-file or validation failure |
| `validation_failed` | `bool` | `True` if validation reported non-compliance or could not complete, or if a preserved embedded PDF could not be converted |
| `skipped` | `bool` | `True` if the original PDF was copied unchanged |

## Exceptions

All custom exceptions inherit from `PDFToPDFAError`:

- `ConversionError`
- `ValidationError`
- `FontEmbeddingError`
- `UnsupportedPDFError`
- `OCRError`
- `VeraPDFError`

`ValidationError` remains a public exception type, but when validation is
requested the high-level conversion APIs currently report veraPDF
non-compliance or an unavailable validator through
`ConversionResult.validation_failed` instead of raising it.

OCR configuration is validated before input processing:

- CLI use of `--ocr`, `--ocr-force`, `--deskew`, `--rotate-pages`,
  `--ocr-layout`, or a non-CPU execution provider such as `directml` or
  `directml:1` without both model-directory options raises a Click
  `UsageError`.
- Providing only one model-directory option also raises `UsageError`.
- The high-level Python APIs raise `ValueError` when only one model directory
  is supplied, or when languages, OCR processing options, or
  a non-CPU `ocr_execution_provider` are supplied without the complete pair.
- Forced OCR (`--ocr-force` or `ocr_force=True`) cannot be combined with
  deskew (`--deskew` or `ocr_deskew=True`).
- Invalid language codes such as `eng` and `deu` are rejected. Use `en`, `de`,
  or `["de", "en"]` for mixed German/English metadata.
- Requesting DirectML when `DmlExecutionProvider` is unavailable raises
  `OCRError`; CPU fallback is intentionally disabled.
- Missing, structurally invalid, or incompatible model artifacts raise
  `OCRError`.

Example:

```python
from pathlib import Path
from pdftopdfa import convert_to_pdfa
from pdftopdfa.exceptions import ConversionError, UnsupportedPDFError

try:
    convert_to_pdfa(Path("input.pdf"), Path("output.pdf"))
except UnsupportedPDFError:
    print("Unsupported PDF")
except ConversionError as exc:
    print(f"Conversion failed: {exc}")
```

Encrypted PDFs are copied to the output path unchanged and returned with
`success=True`, `skipped=True`, and a warning that conversion was skipped. The
copy has not been converted and is not guaranteed to conform to PDF/A, even if
its default output name ends in `_pdfa.pdf`. When validation is requested, the
copy is published with `success=False` if validation fails. PDF/UA mode still attempts both the
PDF/A and PDF/UA-1 profiles and reports either failure.

Digitally signed PDFs are also copied unchanged by default, because OCR,
metadata repair, font embedding, and PDF/A rewriting would invalidate the
cryptographic signature. The result has `success=True`, `skipped=True`, and a
warning, but the unchanged copy is not guaranteed to conform to the requested
PDF/A level. Use `--allow-signature-invalidation` or
`allow_signature_invalidation=True` only when you intentionally want an
unsigned converted copy. For signed archives, the recommended workflow is to
convert to PDF/A first and sign the PDF/A output afterwards.

## PDF/A Levels

| Level | ISO Standard | Attachments | Unicode Required | Tagged Structure | Recommended For |
|---|---|---|---|---|---|
| `2a` | ISO 19005-2 | PDF/A attachments only | Yes | Required | Structured archives |
| `2b` | ISO 19005-2 | PDF/A attachments only | No | No | Basic archiving |
| `2u` | ISO 19005-2 | PDF/A attachments only | Yes | No | Searchable archives |
| `3a` | ISO 19005-3 | Any format | Yes | Required | Structured hybrid documents |
| `3b` | ISO 19005-3 | Any format | No | No | Hybrid documents (for example PDF + XML) |
| `3u` | ISO 19005-3 | Any format | Yes | No | Searchable hybrid archives |

Default level: `3b`.

## Already Compliant PDFs

Before conversion, `pdftopdfa` checks whether a file already claims a PDF/A level.
If veraPDF is available, it validates that claim before deciding to skip conversion.

Default behavior:

| Detected | Behavior |
|---|---|
| Same level (`2b` -> `2b`) | Skip conversion |
| Higher conformance in same part (`2a` -> `2u` or `2b`) | Skip conversion |
| Lower conformance in same part (`2b` -> `2u`) | Convert |
| Different part (`1b` -> `3b`, `3u` -> `2b`) | Convert |

With `--skip-any-pdfa` or `skip_any_pdfa=True`, any input that veraPDF validates as
compliant PDF/A is skipped, regardless of part or conformance level.

- This broader skip only happens when veraPDF actually confirms compliance.
- If veraPDF is not available, the file is converted normally.
- If the document claims PDF/A but veraPDF reports non-compliance, the file is converted normally.

Notes:

- If the metadata claim fails veraPDF validation, conversion is not skipped.
- If veraPDF is unavailable, conversion is not skipped based only on metadata.
- Skipped files return warning: `Conversion skipped: PDF already valid PDF/A (veraPDF compliant)`.
- Level A targets (`1a`, `2a`, `3a`) never skip. Even an already compliant input
  is reconverted so that its logical structure is produced by the semantic
  tagger rather than trusted as authored.
- Any requested OCR processing bypasses this document-level skip logic.
- OCR does not bypass the signed-PDF protection; use
  `--allow-signature-invalidation` explicitly to convert signed inputs.

## Structural Defects Fail the Conversion

Repairs that cannot be made without changing how the document renders are
reported as a conversion failure instead of being applied approximately. A
`ConversionError` is raised when:

- Optional content (layers) is malformed, cyclic, references unregistered
  groups, or has a default configuration whose visibility cannot be normalized
  to `/BaseState /ON` without changing which layers are visible.
- The saved candidate fails post-save structure verification -- missing binary
  comment, wrong `%PDF-` header version, no `%%EOF` marker, or a trailer `/ID`
  that is not two non-empty byte strings.
- Level A semantic tagging cannot bind tags to page content safely.

Earlier releases logged a warning in these cases and continued with a
best-effort result. Nothing is published on failure: an existing destination
keeps its previous contents and a new destination stays absent.

## Validation

`pdftopdfa` integrates with [veraPDF](https://verapdf.org/) for PDF/A validation.
Install veraPDF 1.30.2 or newer separately and make its executable available on
`PATH`, or set `VERAPDF_PATH` to the executable or its parent directory. Older
versions reported by veraPDF's XML output are rejected as unsupported.

- CLI: `pdftopdfa -v input.pdf`
- API: pass `validate=True`

If veraPDF reports non-conformance or cannot complete, the Python APIs return a
result with `success=False` and `validation_failed=True` while keeping the candidate published. The
CLI exits with the validation-failure code; this status does not delete or
suppress the output file. A true conversion failure still publishes nothing.

## Environment Variables

| Variable | Description |
|---|---|
| `VERAPDF_PATH` | Path to `verapdf` executable or its parent directory |

## Related Docs

- OCR details: [ocr.md](ocr.md)
- PDF/A-2/3 rules reference: [veraPDF validation profiles wiki](https://github.com/veraPDF/veraPDF-validation-profiles/wiki/PDFA-Parts-2-and-3-rules)
