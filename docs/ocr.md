# OCR Guide

This guide covers OCR behavior in `pdftopdfa` for scanned and image-based
PDFs. General CLI and API usage is documented in the
[Usage Guide](usage.md).

## Architecture

Text detection, recognition, word positions, and deskew use PP-OCRv6 Medium
from PaddleOCR 3.7 through ONNX Runtime. CPU inference is the default;
DirectML inference is an explicit opt-in. OCRmyPDF is used for page
rasterization, searchable text-layer generation, page merging, and PDF geometry.

PDF/A-2a and PDF/A-3a are supported for scans. During level-A conversion,
semantic OCR lines receive MCIDs in the final OCR Form. Repeated headers,
footers, page numbers, and native-text duplicates can instead become typed
layout artifacts without an MCID. Lines wholly outside the CropBox likewise
remain typed layout artifacts; partially clipped lines use their visible words
as ActualText. Lines the text-layer renderer discards as implausible (for
example a line whose aspect ratio cannot be matched by any font size) never
reach the page, are reported as a warning, and are removed from the OCR
manifest, so a single discarded line does not fail the conversion. The
remaining MCIDs keep their original numbers and may therefore contain gaps.
The internal OCR text, confidence, language, word and line
geometry, and layout reading order are transported to the tagger, which creates
line-level marked-content references, paragraph and heading structure, typed
artifacts, and the required ParentTree links. No secondary recognition engine
is used.

The external text-detection and recognition models are supplied explicitly by
the application. There is no automatic model resolution, runtime download, or
secondary recognition engine. Document orientation instead uses a separate
`PP-LCNet_x1_0_doc_ori` model bundled with `pdftopdfa`.

Within one PDF conversion, the initialized Paddle session is reused across
pages and released when conversion ends, including after an error. OCRmyPDF runs
one page job at a time and Paddle predictions are serialized, avoiding
concurrent access to the PaddleX session. The detector limits its inference
image to 1,600 pixels on the longest side. OCR normally uses a rasterization
resolution of 600 DPI. Before rasterization, every selected page is checked at
the effective OCR resolution and rejected if it would exceed 250 million
pixels. Pages exceeding the limit at 600 DPI are retried at 300 DPI. This
admits A2 at 600 DPI and A0 at 300 DPI while bounding a grayscale page raster
to about 250 MB before decoder and renderer overhead. Page boxes that would
later be normalized or clipped for PDF/A are rejected before OCR so text-layer
and semantic coordinates cannot diverge. When page orientation is enabled,
every page is also checked at the orientation model's actual 108 DPI render size,
including text pages that OCRmyPDF itself skips. Model and temporary-resource
cleanup is always attempted independently; a cleanup failure aborts an
otherwise successful call, but is only logged when another OCR error is already
propagating so that the primary failure remains visible.

OCR input is limited to 100,000 pages. PDF and semantic-manifest candidates are
created in private same-filesystem staging directories, then checked for
pathname identity and byte-for-byte stability immediately before atomic
publication.

## Installation

Install exactly one OCR runtime. For CPU inference:

```bash
pip install "pdftopdfa[ocr]"
```

For the project's supported DirectML configuration on Windows 11:

```bash
pip install "pdftopdfa[directml]"
```

Do not install both extras in the same Python installation. The
`onnxruntime` and `onnxruntime-directml` distributions provide overlapping
runtime files.

No additional system OCR executable is required.

## Execution Providers

CPU is selected by default. DirectML must be selected explicitly:

```bash
pdftopdfa --ocr-execution-provider directml \
  --ocr-detection-model-dir C:/models/PP-OCRv6_medium_det \
  --ocr-recognition-model-dir C:/models/PP-OCRv6_medium_rec \
  scan.pdf
```

The Python equivalent is `ocr_execution_provider="directml"`. `pdftopdfa`
supports DirectML on Windows 11; this is a project support policy, not a
technical Windows-version requirement imposed by DirectML itself. The supported
configuration also requires DirectX 12, a current graphics driver, and an
integrated or dedicated Intel, AMD, or NVIDIA GPU. It uses the same FP32
detection, recognition, and bundled orientation ONNX models as CPU execution.

The selected provider applies to text detection, recognition, deskew, and page
orientation. If DirectML is requested but `DmlExecutionProvider` is not
available, `pdftopdfa` raises a clear error and does not fall back to CPU.

### Selecting a GPU

Plain `directml` uses the provider default, which
[ONNX Runtime documents](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
as device ID 0. It does not select the first item from the filtered diagnostic
listing below. On a machine with more than one GPU, append the raw DXGI adapter
index to choose explicitly:

```bash
pdftopdfa --ocr-execution-provider directml:1 \
  --ocr-detection-model-dir C:/models/PP-OCRv6_medium_det \
  --ocr-recognition-model-dir C:/models/PP-OCRv6_medium_rec \
  scan.pdf
```

The Python equivalent is `ocr_execution_provider="directml:1"`. The current
release exposes this internal diagnostic helper for inspecting the usable raw
DXGI indices:

```python
from pdftopdfa._ocr_runtime import list_directml_devices

for device in list_directml_devices():
    print(device.execution_provider, device.description)
```

`pdftopdfa._ocr_runtime` is a private module, so this helper has no public API
stability guarantee. The listing skips software adapters such as WARP and the
Microsoft Basic Render Driver and collapses duplicate entries for the same PCI
device to the lowest raw index. The remaining raw indices can therefore contain
gaps and need not match list positions. An index that no adapter uses fails at
session creation rather than silently falling back.

## Offline Model Contract

The following revisions are tested and recommended:

- Detection:
  [`PP-OCRv6_medium_det_onnx` at `6132380`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/tree/61323801669c338b7891481ec7bac61ce31b576a)
- Recognition:
  [`PP-OCRv6_medium_rec_onnx` at `50c7eac`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/tree/50c7eacafc52fa7bcf4194e8cd08e46f8558504b)

`pdftopdfa` does not hash or otherwise enforce those revisions. A later model
revision can work if it remains compatible with the expected PaddleOCR model
type and local directory contract.

Create one directory for each model. Each directory must contain exactly
`inference.onnx` and `inference.yml`:

```text
PP-OCRv6_medium_det/
├── inference.onnx
└── inference.yml

PP-OCRv6_medium_rec/
├── inference.onnx
└── inference.yml
```

At the start of each conversion, `pdftopdfa` quickly rejects missing or extra
directory entries, non-regular files, and symbolic links. PaddleOCR then loads
the ONNX and YAML files lazily and checks that they are compatible detection and
recognition models. The initialized inference session is reused across selected
pages in that conversion and released at its end.

The PP-OCRv6 models are intentionally not included in the repository, wheel,
or source distribution. Store them in deployment-managed, preferably read-only
directories. Pass both paths to each PDF conversion or standalone
`recognize_image()` call; an `OCRSession` receives them once and reuses its
models across image calls.

## Page Layout

`--ocr-layout` detects clear vertical columns from OCR word geometry and orders
all lines in the left column before the next column. It does not run another
OCR pass and does not require an additional model. Level-A conversion applies
the same safe layout ordering automatically because the tagger requires a
logical reading order. For other conformance levels, omitting the flag preserves
PaddleOCR's original line order.

The option cannot separate text that the full-page recognition already merged
into one line. If no safe column boundary is found, it only normalizes the
top-to-bottom line order.

```bash
DET_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_det
REC_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_rec

# Order existing OCR lines by detected columns
pdftopdfa --ocr-layout \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf
```

The Python equivalent is `ocr_layout=True`. The keyword is available on
`convert_to_pdfa()`, `convert_files()`, and `convert_directory()`.

## Figure Text

`--ocr-figure-text` recognizes text inside otherwise undescribed direct image
Figures during PDF/A-2a or PDF/A-3a tagging. It is intended for text-based
logos and similar images. Existing `Alt`, `ActualText`, or a textual Caption is
never replaced.

```bash
pdftopdfa -l 3a --pdfua --ocr-figure-text \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  document.pdf
```

The converter extracts eligible direct Image XObjects and reuses one
`OCRSession` across all candidates in the document. Non-empty OCR lines are
whitespace-normalized and joined in recognition order. The result is accepted
only when every non-empty line has confidence of at least `0.90`; an empty or
less-confident result marks the Figure as a `Layout` artifact and reports that
decision for manual review.

Accepted text is written as `ActualText`, because it replaces text visibly
contained in the image rather than describing all visual meaning. Every
generated value is still reported as requiring author review. Inline images,
Form XObjects, masks, diagrams without recognizable text, and authoritative
visual descriptions remain outside this automatic step.

The Python equivalent is `ocr_figure_text=True`. The flag requires both OCR
model directories and level `"2a"` or `"3a"` on `convert_to_pdfa()`,
`convert_files()`, and `convert_directory()`.

## Languages

The default language code is `en`. Use `de` for German and `de+en` on the CLI
for mixed German/English recognition:

```bash
pdftopdfa --ocr-lang de+en \
  --ocr-detection-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_det \
  --ocr-recognition-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_rec \
  scan.pdf
```

The Python equivalent is `ocr_languages=["de", "en"]`. Legacy codes such as
`eng` and `deu` are rejected without compatibility aliases.

When every selected language uses the Latin script, the recognition decoder
rejects non-Latin letters. Accented Latin letters, numbers, punctuation, and
symbols remain available. Selecting `ch`, `chinese_cht`, or `japan`, including
in a mixed language list, keeps the full model character set available.

Accepted PaddleOCR 3.7 codes:

`af`, `az`, `bs`, `ca`, `ch`, `chinese_cht`, `cs`, `cy`, `da`, `de`, `en`,
`es`, `et`, `eu`, `fi`, `fr`, `french`, `ga`, `german`, `gl`, `hr`, `hu`,
`id`, `is`, `it`, `japan`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`,
`nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`,
`sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`.

The unified Medium model performs recognition for these languages. The
primary selection is also written to the PDF catalog `/Lang` entry when the
input does not already declare a document language. Paddle-specific aliases
are written as their corresponding BCP 47 tag, for example `ch` as `zh-Hans`
and `chinese_cht` as `zh-Hant`.

## CLI Usage

Both model options must always be supplied together. Supplying the pair enables
OCR, so `--ocr` is optional in the first example:

```bash
DET_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_det
REC_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_rec

# English (default)
pdftopdfa \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# German
pdftopdfa --ocr --ocr-lang de \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# German and English
pdftopdfa --ocr-lang de+en \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# Apply OCR and page processing without PDF/A conversion
pdftopdfa --no-pdfa --deskew --rotate-pages \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf
```

`--ocr`, `--ocr-force`, `--deskew`, `--rotate-pages`, `--ocr-layout`,
`--ocr-figure-text`, and an explicit `--ocr-execution-provider directml` or
`directml:INDEX` are rejected unless both text-model options are present.
Providing only one text-model option is also rejected. `--ocr-lang` selects the
recognition script and metadata but does not activate OCR by itself.
`--ocr-execution-provider` defaults to `cpu`.

## Python API

The high-level conversion functions use the optional
`ocr_detection_model_dir` and `ocr_recognition_model_dir` keyword arguments.
Providing both enables OCR:

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
    ocr_rotate_pages=True,
)
```

The same keywords are available on `convert_files()` and
`convert_directory()`. Supplying only one directory, or supplying
`ocr_languages`, `ocr_force`, `ocr_deskew`, `ocr_rotate_pages`, or
`ocr_layout=True`, `ocr_figure_text=True`, or
`ocr_execution_provider="directml"` or `"directml:INDEX"` without the complete
text-model pair, raises `ValueError` before input processing starts. Set a
DirectML provider only in an installation made with the `directml` extra.

The lower-level function requires both model directories as keyword-only
arguments:

```python
from pathlib import Path

from pdftopdfa.ocr import apply_ocr

result_path = apply_ocr(
    Path("scan.pdf"),
    Path("scan_processed.pdf"),
    ["de", "en"],
    detection_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_det"),
    recognition_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_rec"),
    ocr_execution_provider="cpu",
    deskew=True,
)
```

`apply_ocr()` raises `OCRError` for missing, structurally invalid, unreadable,
or incompatible model artifacts, an unavailable requested provider, and OCR
execution failures.

Individual images can be recognized without creating a PDF:

```python
from pathlib import Path

from pdftopdfa import recognize_image

results = recognize_image(
    Path("serial-number.png"),
    detection_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_det"),
    recognition_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_rec"),
    layout="single_line",
    allowed_characters="0123456789",
)
```

The result is a list of `(text, confidence)` pairs. The default
`layout="auto"` uses text detection before recognizing each detected line.
Use `layout="single_line"` when the complete image is one text line; this
bypasses detection. `allowed_characters` restricts CTC decoding itself, so a
disallowed character cannot win a decoding step. Unlike PDF conversion, this
direct image API has no language option and uses the model's full character set
unless `allowed_characters` is supplied. The standalone convenience function
loads its models lazily for the call and closes its internal session before
returning.

For multiple images from one document, use `OCRSession`. It loads the models
lazily for the first image, reuses them for later images, and releases them
once when the context exits, including after an error:

```python
from pathlib import Path

from pdftopdfa import OCRSession

with OCRSession(
    detection_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_det"),
    recognition_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_rec"),
) as session:
    first = session.recognize_image(Path("page-1.png"))
    second = session.recognize_image(Path("page-2.png"))
```

Calls on one session are processed sequentially. The standalone
`recognize_image()` function remains the one-image convenience API.

An already-cropped table can be recognized from a local image path or a
`PIL.Image.Image`:

```python
from pathlib import Path

from pdftopdfa import recognize_table

result = recognize_table(
    Path("cropped-table.png"),
    table_classification_model_dir=Path(
        "/opt/pdftopdfa/models/PP-LCNet_x1_0_table_cls"
    ),
    wired_table_structure_recognition_model_dir=Path(
        "/opt/pdftopdfa/models/SLANeXt_wired"
    ),
    wireless_table_structure_recognition_model_dir=Path(
        "/opt/pdftopdfa/models/SLANeXt_wireless"
    ),
    wired_table_cells_detection_model_dir=Path(
        "/opt/pdftopdfa/models/RT-DETR-L_wired_table_cell_det"
    ),
    wireless_table_cells_detection_model_dir=Path(
        "/opt/pdftopdfa/models/RT-DETR-L_wireless_table_cell_det"
    ),
    detection_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_det"),
    recognition_model_dir=Path("/opt/pdftopdfa/models/PP-OCRv6_medium_rec"),
    ocr_execution_provider="cpu",
)
```

The five table-specific models are available from the official PaddlePaddle
ONNX repositories:

- [PP-LCNet_x1_0_table_cls_onnx](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_table_cls_onnx)
- [SLANeXt_wired_onnx](https://huggingface.co/PaddlePaddle/SLANeXt_wired_onnx)
- [SLANeXt_wireless_onnx](https://huggingface.co/PaddlePaddle/SLANeXt_wireless_onnx)
- [RT-DETR-L_wired_table_cell_det_onnx](https://huggingface.co/PaddlePaddle/RT-DETR-L_wired_table_cell_det_onnx)
- [RT-DETR-L_wireless_table_cell_det_onnx](https://huggingface.co/PaddlePaddle/RT-DETR-L_wireless_table_cell_det_onnx)

The project does not record or enforce a tested revision for these five models.
Pin the chosen upstream revisions in reproducible deployments.

`recognize_table()` returns an immutable `TableRecognitionResult` with its
`TableType`, HTML, and a tuple of `TableCell` values. Cell rows and columns are
zero-based. Spans, text, and pixel bounding boxes are plain Python values.
Confidence is the mean confidence of OCR text boxes assigned to the cell, or
`None` for an empty cell. PaddleX and NumPy result objects never escape the
function.

Grid structure and text come from the table-structure model, while
`bounding_box` comes from the independent cell-detection model. Both models can
report a different number of cells for the same table. In that case a warning
is logged and every returned cell has `bounding_box` and `confidence` set to
`None`, so `TableCell.bounding_box` is `TableBoundingBox | None`. Rows,
columns, spans, text, and HTML remain usable.

All seven model directories are required and must each contain exactly
`inference.onnx` and `inference.yml`. Directories and artifacts must be local,
regular, and not symbolic links. URL input is rejected. Classification runs
first; only the selected wired or wireless structure and cell-model paths are
then supplied to the cached `TableRecognitionPipelineV2`. Document
orientation, unwarping, layout detection, and table-orientation models are
disabled, so PaddleX never receives a missing model path that it could resolve
by downloading. Classification and table predictions are serialized for safe
reuse of the cached ONNX sessions.

`ocr_execution_provider` accepts `"cpu"`, `"directml"`, or
`"directml:INDEX"` and applies to classification and the selected table
pipeline. DirectML requires the `directml` extra and uses the same strict
no-CPU-fallback behavior as image and PDF OCR. In a frozen Windows application,
import `pdftopdfa` or `pdftopdfa.table` on the main thread before starting
workers so the table runtime can be preloaded; first loading it from a worker
thread is rejected.

Use `pdfa=False` with a neutral output name such as `scan_processed.pdf` to
write the result of a high-level PDF conversion call directly without font
embedding, PDF/A sanitization, metadata synchronization, color-profile
embedding, or PDF/A validation. This option is not part of
`recognize_table()`. The output PDF is not validated or guaranteed to be PDF/A
compliant.

## Page Selection and Force Mode

When OCR is enabled, processing runs for the document:

- Pages without meaningful text receive OCR; whitespace-only extracted text is
  treated as no text.
- Digital pages with non-whitespace text are skipped. A conservatively
  identified full-page scan with a visible native text overlay is instead
  re-OCRed in a targeted run that retains the native text.
- Mixed documents are handled page by page.

With deskew enabled, scan-like pages are selected separately as described
below. An existing invisible OCR text layer does not exclude such a page.

When annotations are detected, the high-level conversion APIs try to remove
them from a temporary OCR input so they are not rasterized. When that succeeds,
they restore the original annotation arrays and AcroForm after OCR and verify
the page and annotation counts. A stripping failure is logged and processing
continues with the original input; a restoration failure aborts rather than
silently dropping annotations. This preservation step is not part of the
lower-level `apply_ocr()` API. With `pdfa=False`, successfully restored
annotations remain in the processed PDF. The subsequent PDF/A sanitization
used by normal conversion can still flatten or remove annotations that are not
permitted in the requested archival output.

Use force mode to replace an existing OCR layer:

```bash
pdftopdfa --ocr-force --ocr-lang de \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf
```

Python uses `ocr_force=True`. Requesting any OCR processing bypasses the
document-level already-compliant PDF/A skip path. Force mode additionally:

- replaces existing OCR text layers;
- does not bypass signed-PDF protection;
- cannot be combined with `--deskew` or `ocr_deskew=True`.

Use `--allow-signature-invalidation` only when an unsigned OCR/PDF/A copy is
intentional.

## Deskew and Page Orientation

`--deskew` and `--rotate-pages` are independent opt-in operations. Both enable
OCR and require the external detection and recognition model directories.
Page orientation itself uses the bundled model described below; the external
text models are still required because `--rotate-pages` activates OCR.

Deskew first selects pages where one opaque, unclipped raster image visibly
covers at least 80% of the page and where there is no painted vector content or
shading. A visible native text overlay painted after an otherwise qualifying
full-page image selects a separate non-deskew `redo_ocr` run; OCRmyPDF retains
that native text while the internal engine recognizes the underlying scan.
Text painted before a later full-page image is treated as occluded. Image
coverage is intersected with the page, so off-page or tightly clipped images do
not qualify. Unknown clipping, transparency, blend, optional-content, or
overprint states fail closed. If such a page combines an apparent full-page scan
with an existing text layer, OCR aborts instead of publishing text of uncertain
provenance. This leaves ordinary digitally generated pages and pages with only
decorative small images unchanged and handles mixed documents page by page.
Text that is provably non-painting from its render mode or alpha state is removed
from a temporary page-local copy before a selected OCR run; clipping-only text
is not assumed invisible because arbitrary clip geometry is ambiguous.

Regular OCR pages, selected deskew pages, and mixed-content redo pages are
OCRed in disjoint targeted page runs. No page receives OCR twice, and a pure
scan needs only one run. Per-run Level-A sidecars are isolated and merged in
physical page order.

For each selected page, PaddleOCR derives a correction angle from at least two
sufficiently long text polygons within ±10 degrees. With insufficient evidence
the page is left geometrically unchanged. A page that is actually rotated is
recognized again after correction so its text and coordinates describe the
corrected raster. Deskewing may alter page appearance and increase file size.

An annotated scan-like page that otherwise needs OCR is OCRed without deskewing,
and a warning is emitted. Annotation geometry cannot be transformed safely
through a raster deskew operation, so this prevents links and interactive
content from becoming misaligned. Annotations on other pages do not disable
deskewing for the rest of the document.

Before OCR starts, `--rotate-pages` renders and classifies every page with the
bundled `PP-LCNet_x1_0_doc_ori` model, including pages later skipped because
they already contain text. Predictions below 0.80 leave the page unchanged and
produce a warning. Missing, corrupt, or unusable bundled model files abort the
conversion. The orientation model uses the same selected execution provider as
text detection and recognition.

## Offline Deployment

The external PP-OCRv6 directories are the only source for text detection and
recognition. Document-orientation files instead come from the bundled
`pdftopdfa/resources/models/PP-LCNet_x1_0_doc_ori/` directory. All document
orientation, text detection, recognition, and deskew operations use local ONNX
files; the text-model paths are configured explicitly, while the orientation
path is resolved from the installed package.

Applications frozen with tools such as PyInstaller must collect the
`pdftopdfa` package data as well as PaddleOCR, PaddleX, ONNX Runtime, OpenCV,
NumPy, OCRmyPDF, pypdfium2, and the ONNX Runtime native libraries. Network
access is not required at runtime. Bundle exactly one of the CPU or DirectML
ONNX Runtime distributions. PaddleX may create the empty housekeeping
directories `func_ret`, `locks`, and `temp` below its configured cache path,
but it does not store or download model artifacts there when the explicit
model directories are used. Point `PADDLE_PDX_CACHE_HOME` at a writable
scratch directory when the application filesystem is read-only.

## Troubleshooting

### OCR is not available

Install exactly one OCR extra:

```bash
# CPU
pip install "pdftopdfa[ocr]"

# DirectML (project-supported on Windows 11)
pip install "pdftopdfa[directml]"
```

### DirectML is unavailable

For the project's supported configuration, confirm that only the `directml`
extra is installed, Windows 11 is current, the GPU supports DirectX 12, and the
latest Intel, AMD, or NVIDIA driver is installed. `pdftopdfa` intentionally
stops instead of retrying the request on CPU.

### Model directories are rejected

Check that:

- both directory options were supplied;
- each directory contains only `inference.onnx` and `inference.yml`;
- neither directories nor files are symbolic links;
- the process can read all four files.

The structural check and PaddleOCR initialization are intentionally fail-closed
and never download a replacement.

### Unsupported language code

Use `en`, `de`, `de+en`, or another code from the accepted list above. Do not
use legacy three-letter aliases.

### Bundled orientation model is missing or corrupt

Reinstall `pdftopdfa` from an intact wheel. The orientation preflight validates
the packaged files against SHA-256 hashes and does not attempt a network
download.

### OCR did not alter a page

Possible reasons:

- the page already has a text layer and force mode was not enabled;
- deskew found insufficient geometric evidence;
- page-orientation confidence was below 0.80;
- the digitally signed input was copied unchanged to preserve its signature.

Use `--ocr-force` with both model directories only when replacing an existing
OCR layer is intentional.
