# OCR Guide

This guide covers OCR behavior in `pdftopdfa` for scanned and image-based
PDFs. General CLI and API usage is documented in the
[Usage Guide](usage.md).

## Architecture

Text detection, recognition, word positions, and deskew use PP-OCRv6 Medium
from PaddleOCR 3.7 through ONNX Runtime. CPU inference is the default;
DirectML inference is available explicitly on Windows 11. OCRmyPDF is used for
page rasterization, searchable text-layer generation, page merging, and PDF
geometry.

The recognition models are supplied explicitly by the application. There is no
automatic model resolution, runtime download, or secondary recognition engine.
Document orientation uses a separate `PP-LCNet_x1_0_doc_ori` model bundled
with `pdftopdfa`.

The initialized Paddle session is reused across pages. OCRmyPDF runs one page job
at a time and Paddle predictions are serialized, avoiding concurrent access to
the PaddleX session. The detector limits its inference image to 1,600 pixels
on the longest side. OCR uses a minimum rasterization resolution of 600 DPI;
higher-resolution source images can therefore still require more memory while
the source raster is prepared.

## Installation

Install exactly one OCR runtime. For CPU inference:

```bash
pip install "pdftopdfa[ocr]"
```

For DirectML inference on Windows 11:

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

The Python equivalent is `ocr_execution_provider="directml"`. DirectML
requires Windows 11, DirectX 12, a current graphics driver, and an integrated
or dedicated Intel, AMD, or NVIDIA GPU. It uses the same FP32 detection,
recognition, and bundled orientation ONNX models as CPU execution.

The selected provider applies to text detection, recognition, deskew, page
orientation, and optional model-based layout detection. If DirectML is
requested but `DmlExecutionProvider` is not available, `pdftopdfa` raises a
clear error and does not fall back to CPU.

### Selecting a GPU

Plain `directml` lets DirectML pick the adapter. On a machine with more than
one GPU, append the device index to choose explicitly:

```bash
pdftopdfa --ocr-execution-provider directml:1 \
  --ocr-detection-model-dir C:/models/PP-OCRv6_medium_det \
  --ocr-recognition-model-dir C:/models/PP-OCRv6_medium_rec \
  scan.pdf
```

The Python equivalent is `ocr_execution_provider="directml:1"`. Indices start
at 0 and follow the adapter order reported by
`pdftopdfa._ocr_runtime.list_directml_devices()`:

```python
from pdftopdfa._ocr_runtime import list_directml_devices

for device in list_directml_devices():
    print(device.execution_provider, device.description)
```

The listing skips software adapters such as WARP and the Microsoft Basic
Render Driver, because DirectML does not expose them as devices either. An
index that no adapter uses fails at session creation rather than silently
falling back.

## Offline Model Contract

Download the files from these exact revisions:

- Detection:
  [`PP-OCRv6_medium_det_onnx` at `6132380`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/tree/61323801669c338b7891481ec7bac61ce31b576a)
- Recognition:
  [`PP-OCRv6_medium_rec_onnx` at `50c7eac`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/tree/50c7eacafc52fa7bcf4194e8cd08e46f8558504b)

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

Before the first model initialization, `pdftopdfa` quickly rejects missing or
extra directory entries, non-regular files, and symbolic links. PaddleOCR then
loads the ONNX and YAML files and checks that they are compatible detection and
recognition models. Each conversion repeats only the quick structure and file
metadata check; an unchanged initialized inference session is reused.

The PP-OCRv6 models are intentionally not included in the repository, wheel,
or source distribution. Store them in deployment-managed, preferably read-only
directories and pass both paths explicitly on every OCR invocation.

## Page Layout

`--ocr-layout` controls how recognized text is ordered on pages containing
separate columns or document blocks:

| Mode | Behavior | Additional work |
|---|---|---|
| `none` | Keeps PaddleOCR's original line order. This is the default and preserves previous behavior. | None |
| `simple` | Detects clear vertical columns from OCR word geometry and orders all lines in the left column before the next column. | No additional OCR pass |
| `regions` | Detects clear vertical columns from an initial full-page OCR pass, then recognizes each detected column separately. | One OCR pass per detected column |
| `model` | Uses PP-DocLayout_plus-L to detect document blocks, then recognizes each block separately in reading order. | One layout inference plus one OCR pass per detected block |

`simple` is the fastest layout-aware option, but it cannot separate text that
the first full-page recognition already merged into one line. `regions`
addresses that case without another model. If it cannot identify a safe column
boundary, it keeps the full-page recognition and applies the same ordering as
`simple`. `model` is intended for more irregular layouts where columns alone
are insufficient. If it detects no usable block, the full page is recognized
once.

```bash
DET_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_det
REC_MODEL=/opt/pdftopdfa/models/PP-OCRv6_medium_rec
LAYOUT_MODEL=/opt/pdftopdfa/models/PP-DocLayout_plus-L

# Reorder existing OCR lines by detected columns
pdftopdfa --ocr-layout simple \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# Recognize each detected column independently
pdftopdfa --ocr-layout regions \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf

# Detect document blocks with the layout model before OCR
pdftopdfa --ocr-layout model \
  --ocr-layout-model-dir "$LAYOUT_MODEL" \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf
```

For `model`, download
[`PP-DocLayout_plus-L_onnx` at `feb7461`](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L_onnx/tree/feb74619326f634e0e883218598096a3733ad9f7).
Its local directory follows the same offline contract and must contain exactly
`inference.onnx` and `inference.yml`. The model is not downloaded at runtime.
`--ocr-layout-model-dir` is required only for `model` and is rejected for the
other modes.

The Python equivalents are `ocr_layout="simple"`, `"regions"`, or `"model"`.
For the model mode, also pass
`ocr_layout_model_dir=Path("/opt/pdftopdfa/models/PP-DocLayout_plus-L")`.
These keywords are available on `convert_to_pdfa()`, `convert_files()`, and
`convert_directory()`.

## Languages

The default language code is `en`. Use `de` for German and `de+en` on the CLI
for mixed German/English metadata:

```bash
pdftopdfa --ocr-lang de+en \
  --ocr-detection-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_det \
  --ocr-recognition-model-dir /opt/pdftopdfa/models/PP-OCRv6_medium_rec \
  scan.pdf
```

The Python equivalent is `ocr_languages=["de", "en"]`. Legacy codes such as
`eng` and `deu` are rejected without compatibility aliases.

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

`--ocr`, `--ocr-force`, `--deskew`, `--rotate-pages`, any layout mode other
than `none`, and an explicit `--ocr-execution-provider directml` are rejected
unless both text-model options are present. Providing only one text-model
option is also rejected. `--ocr-lang` selects metadata but does not activate
OCR by itself.
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
    ocr_detection_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_det"
    ),
    ocr_recognition_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_rec"
    ),
    ocr_execution_provider="cpu",
    ocr_rotate_pages=True,
)
```

The same keywords are available on `convert_files()` and
`convert_directory()`. Supplying only one directory, or supplying
`ocr_languages`, `ocr_force`, `ocr_deskew`, `ocr_rotate_pages`, or
an `ocr_layout` other than `"none"`, or
`ocr_execution_provider="directml"` without the complete text-model pair, raises
`ValueError` before input processing starts.
Set `ocr_execution_provider="directml"` only in an installation made with the
`directml` extra.

The lower-level function requires both model directories as keyword-only
arguments:

```python
from pathlib import Path

from pdftopdfa.ocr import apply_ocr

result_path = apply_ocr(
    Path("scan.pdf"),
    Path("scan_processed.pdf"),
    ["de", "en"],
    detection_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_det"
    ),
    recognition_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_rec"
    ),
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
    detection_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_det"
    ),
    recognition_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_rec"
    ),
    layout="single_line",
    allowed_characters="0123456789",
)
```

The result is a list of `(text, confidence)` pairs. The default
`layout="auto"` uses text detection before recognizing each detected line.
Use `layout="single_line"` when the complete image is one text line; this
bypasses detection. `allowed_characters` restricts CTC decoding itself, so a
disallowed character cannot win a decoding step. Omitting both options uses
the standard OCR pipeline unchanged.

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
    detection_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_det"
    ),
    recognition_model_dir=Path(
        "/opt/pdftopdfa/models/PP-OCRv6_medium_rec"
    ),
)
```

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

Use `pdfa=False` with a neutral output name such as `scan_processed.pdf` to
write the OCR result directly without font embedding, PDF/A sanitization,
metadata synchronization, color-profile embedding, or PDF/A validation. The
output is not validated or guaranteed to be PDF/A compliant.

## Page Selection and Force Mode

When OCR is enabled, processing runs for the document:

- Pages without a text layer receive OCR.
- Pages that already contain text are skipped.
- Mixed documents are handled page by page.

With deskew enabled, scan-like pages are selected separately as described
below. An existing invisible OCR text layer does not exclude such a page.

Use force mode to replace an existing OCR layer:

```bash
pdftopdfa --ocr-force --ocr-lang de \
  --ocr-detection-model-dir "$DET_MODEL" \
  --ocr-recognition-model-dir "$REC_MODEL" \
  scan.pdf
```

Python uses `ocr_force=True`. Force mode:

- replaces existing OCR text layers;
- bypasses the already-compliant PDF/A skip path;
- does not bypass signed-PDF protection;
- cannot be combined with `--deskew` or `ocr_deskew=True`.

Use `--allow-signature-invalidation` only when an unsigned OCR/PDF/A copy is
intentional.

## Deskew and Page Orientation

`--deskew` and `--rotate-pages` are independent opt-in operations. Both enable
OCR and require the external detection and recognition model directories.

Deskew first selects pages where one opaque, unclipped raster image visibly
covers at least 80% of the page and where there is no uncovered native text,
painted vector content, or shading. Image coverage is intersected with the page,
so off-page or tightly clipped images do not qualify. This leaves digitally
generated pages unchanged and handles mixed documents page by page. Invisible
OCR text, including clipping-only or fully transparent text, is allowed and is
removed from a temporary page-local copy before the selected scan is processed.

Non-deskew raster pages and selected scan pages are OCRed in disjoint targeted
page runs. No page receives OCR twice, and a pure scan needs only one run.

For each selected page, PaddleOCR derives a correction angle from at least two
sufficiently long text polygons within ±10 degrees. With insufficient evidence
the page is left geometrically unchanged, and that page- and run-scoped
prediction is reused for the OCR text layer instead of running the full model
again. A page that is actually rotated is recognized again after correction so
its text and coordinates describe the corrected raster. Deskewing may alter
page appearance and increase file size.

An annotated scan-like page is OCRed without deskewing, and a warning is
emitted. Annotation geometry cannot be transformed safely through a raster
deskew operation, so this prevents links and interactive content from becoming
misaligned. Annotations on other pages do not disable deskewing for the rest of
the document.

Before OCR starts, `--rotate-pages` renders and classifies every page with the
bundled `PP-LCNet_x1_0_doc_ori` model, including pages later skipped because
they already contain text. Predictions below 0.80 leave the page unchanged and
produce a warning. Missing, corrupt, or unusable bundled model files abort the
conversion. The orientation model uses the same selected execution provider as
text detection and recognition.

## Offline Deployment

The external PP-OCRv6 directories are the only recognition-model source.
Document-orientation files are bundled under
`pdftopdfa/resources/models/PP-LCNet_x1_0_doc_ori/`. All document orientation,
text detection, recognition, and deskew operations run through explicitly
configured local ONNX files.

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

# DirectML on Windows 11
pip install "pdftopdfa[directml]"
```

### DirectML is unavailable

Confirm that only the `directml` extra is installed, Windows 11 is current, the
GPU supports DirectX 12, and the latest Intel, AMD, or NVIDIA driver is
installed. `pdftopdfa` intentionally stops instead of retrying the request on
CPU.

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
