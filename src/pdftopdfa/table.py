# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Offline ONNX table recognition."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree
from lxml import html as lxml_html
from PIL import Image

from ._ocr_runtime import (
    _ModelSpec,
    _validate_model_directory,
    execution_provider_base,
    onnxruntime_engine_config,
    require_execution_provider,
    validate_ocr_execution_provider,
)
from .exceptions import OCRError

logger = logging.getLogger(__name__)

_TABLE_CLASSIFICATION_MODEL = _ModelSpec(name="PP-LCNet_x1_0_table_cls")
_WIRED_STRUCTURE_MODEL = _ModelSpec(name="SLANeXt_wired")
_WIRELESS_STRUCTURE_MODEL = _ModelSpec(name="SLANeXt_wireless")
_WIRED_CELLS_MODEL = _ModelSpec(name="RT-DETR-L_wired_table_cell_det")
_WIRELESS_CELLS_MODEL = _ModelSpec(name="RT-DETR-L_wireless_table_cell_det")
_TEXT_DETECTION_MODEL = _ModelSpec(name="PP-OCRv6_medium_det")
_TEXT_RECOGNITION_MODEL = _ModelSpec(name="PP-OCRv6_medium_rec")
_TEXT_DETECTION_LIMIT_SIDE_LEN = 1600
_OCR_CELL_OVERLAP = 0.7
# Maximum top-coordinate distance in pixels between a cell box and the first
# box of its row, matching sort_table_cells_boxes() in PaddleX.
_CELL_ROW_TOLERANCE = 10.0


class TableType(StrEnum):
    """Recognized table type."""

    WIRED = "wired"
    WIRELESS = "wireless"


@dataclass(frozen=True, slots=True)
class TableBoundingBox:
    """Axis-aligned table-cell coordinates in image pixels."""

    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class TableCell:
    """One recognized table cell using zero-based grid coordinates."""

    row: int
    column: int
    row_span: int
    column_span: int
    text: str
    bounding_box: TableBoundingBox | None
    confidence: float | None


@dataclass(frozen=True, slots=True)
class TableRecognitionResult:
    """Serialized result for one already-cropped table image."""

    table_type: TableType
    html: str
    cells: tuple[TableCell, ...]


_Fingerprint = tuple[tuple[int, int, int, int, int, int], ...]


@dataclass(frozen=True)
class _ResolvedModel:
    path: Path
    fingerprint: _Fingerprint


@dataclass(frozen=True)
class _ResolvedModels:
    classification: _ResolvedModel
    wired_structure: _ResolvedModel
    wireless_structure: _ResolvedModel
    wired_cells: _ResolvedModel
    wireless_cells: _ResolvedModel
    text_detection: _ResolvedModel
    text_recognition: _ResolvedModel


_prediction_lock = threading.RLock()
_cached_classifier_key: tuple[_ResolvedModel, str] | None = None
_cached_classifier: Any | None = None
_PipelineKey = tuple[
    _ResolvedModel,
    _ResolvedModel,
    _ResolvedModel,
    _ResolvedModel,
    _ResolvedModel,
    str,
]
_cached_pipelines: dict[TableType, tuple[_PipelineKey, Any]] = {}


def _resolve_model_directory(
    value: str | Path,
    spec: _ModelSpec,
) -> _ResolvedModel:
    try:
        unresolved = Path(value).expanduser()
    except (OSError, TypeError, ValueError) as exc:
        raise OCRError(f"Invalid {spec.name} model directory: {exc}") from exc
    if unresolved.is_symlink():
        raise OCRError(f"{spec.name} model directory must not be a symbolic link")
    try:
        model_dir = unresolved.resolve()
    except (OSError, RuntimeError) as exc:
        raise OCRError(f"Invalid {spec.name} model directory: {exc}") from exc
    return _ResolvedModel(
        path=model_dir,
        fingerprint=_validate_model_directory(model_dir, spec),
    )


def _resolve_models(
    *,
    table_classification_model_dir: str | Path,
    wired_table_structure_recognition_model_dir: str | Path,
    wireless_table_structure_recognition_model_dir: str | Path,
    wired_table_cells_detection_model_dir: str | Path,
    wireless_table_cells_detection_model_dir: str | Path,
    detection_model_dir: str | Path,
    recognition_model_dir: str | Path,
) -> _ResolvedModels:
    return _ResolvedModels(
        classification=_resolve_model_directory(
            table_classification_model_dir,
            _TABLE_CLASSIFICATION_MODEL,
        ),
        wired_structure=_resolve_model_directory(
            wired_table_structure_recognition_model_dir,
            _WIRED_STRUCTURE_MODEL,
        ),
        wireless_structure=_resolve_model_directory(
            wireless_table_structure_recognition_model_dir,
            _WIRELESS_STRUCTURE_MODEL,
        ),
        wired_cells=_resolve_model_directory(
            wired_table_cells_detection_model_dir,
            _WIRED_CELLS_MODEL,
        ),
        wireless_cells=_resolve_model_directory(
            wireless_table_cells_detection_model_dir,
            _WIRELESS_CELLS_MODEL,
        ),
        text_detection=_resolve_model_directory(
            detection_model_dir,
            _TEXT_DETECTION_MODEL,
        ),
        text_recognition=_resolve_model_directory(
            recognition_model_dir,
            _TEXT_RECOGNITION_MODEL,
        ),
    )


def _close(model: Any | None) -> None:
    if model is None:
        return
    close = getattr(model, "close", None)
    if callable(close):
        close()


class _ClassifiedTable:
    def __init__(self, table_type: TableType) -> None:
        self._label = f"{table_type.value}_table"

    def __call__(self, *_args: Any, **_kwargs: Any) -> tuple[dict[str, list[Any]], ...]:
        return (
            {
                "label_names": [self._label],
                "scores": [1.0],
            },
        )


def _create_table_classifier(
    model: _ResolvedModel,
    execution_provider: str,
) -> Any:
    try:
        from paddleocr import TableClassification

        classifier = TableClassification(
            model_name=_TABLE_CLASSIFICATION_MODEL.name,
            model_dir=str(model.path),
            topk=1,
            engine="onnxruntime",
            device="cpu",
            enable_hpi=False,
            engine_config=onnxruntime_engine_config(execution_provider),
        )
        if execution_provider_base(execution_provider) == "directml":
            try:
                require_execution_provider(
                    classifier.paddlex_predictor.runner.session,
                    execution_provider,
                )
            except Exception:
                _close(classifier)
                raise
        return classifier
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(
            f"Could not initialize PaddleOCR table classification: {exc}"
        ) from exc


def _selected_models(
    table_type: TableType,
    models: _ResolvedModels,
) -> tuple[_ModelSpec, _ResolvedModel, _ModelSpec, _ResolvedModel]:
    if table_type is TableType.WIRED:
        return (
            _WIRED_STRUCTURE_MODEL,
            models.wired_structure,
            _WIRED_CELLS_MODEL,
            models.wired_cells,
        )
    return (
        _WIRELESS_STRUCTURE_MODEL,
        models.wireless_structure,
        _WIRELESS_CELLS_MODEL,
        models.wireless_cells,
    )


def _create_table_pipeline(
    table_type: TableType,
    models: _ResolvedModels,
    execution_provider: str,
) -> Any:
    structure_spec, structure, cells_spec, cells = _selected_models(
        table_type,
        models,
    )
    try:
        from paddleocr import TableRecognitionPipelineV2
        from paddleocr._common_args import prepare_common_init_args
        from paddlex.inference.pipelines.table_recognition.pipeline_v2 import (
            _TableRecognitionPipelineV2 as PaddleXTableRecognitionPipelineV2,
        )

        selected_model_names = frozenset({structure_spec.name, cells_spec.name})

        class SelectedPaddleXTableRecognitionPipelineV2(
            PaddleXTableRecognitionPipelineV2
        ):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._selected_models: dict[str, Any] = {}
                super().__init__(*args, **kwargs)

            def create_model(
                self,
                config: dict[str, Any],
                **kwargs: Any,
            ) -> Any:
                model_name = config.get("model_name")
                if model_name == _TABLE_CLASSIFICATION_MODEL.name:
                    return _ClassifiedTable(table_type)
                if model_name in selected_model_names:
                    if model_name not in self._selected_models:
                        self._selected_models[model_name] = super().create_model(
                            config,
                            **kwargs,
                        )
                    return self._selected_models[model_name]
                return super().create_model(config, **kwargs)

            def cells_det_results_nms(
                self,
                cells_det_results: Any,
                cells_det_scores: Any,
                cells_det_threshold: float = 0.3,
            ) -> tuple[Any, Any]:
                if len(cells_det_results) == 0:
                    return cells_det_results, cells_det_scores
                return super().cells_det_results_nms(
                    cells_det_results,
                    cells_det_scores,
                    cells_det_threshold,
                )

        class SelectedTableRecognitionPipelineV2(TableRecognitionPipelineV2):
            def _create_paddlex_pipeline(self) -> Any:
                kwargs = prepare_common_init_args(None, self._common_args)
                return SelectedPaddleXTableRecognitionPipelineV2(
                    config=self._merged_paddlex_config,
                    **kwargs,
                )

        engine_config = onnxruntime_engine_config(execution_provider)
        pipeline = SelectedTableRecognitionPipelineV2(
            table_classification_model_name=_TABLE_CLASSIFICATION_MODEL.name,
            table_classification_model_dir=str(models.classification.path),
            wired_table_structure_recognition_model_name=structure_spec.name,
            wired_table_structure_recognition_model_dir=str(structure.path),
            wireless_table_structure_recognition_model_name=structure_spec.name,
            wireless_table_structure_recognition_model_dir=str(structure.path),
            wired_table_cells_detection_model_name=cells_spec.name,
            wired_table_cells_detection_model_dir=str(cells.path),
            wireless_table_cells_detection_model_name=cells_spec.name,
            wireless_table_cells_detection_model_dir=str(cells.path),
            text_detection_model_name=_TEXT_DETECTION_MODEL.name,
            text_detection_model_dir=str(models.text_detection.path),
            text_recognition_model_name=_TEXT_RECOGNITION_MODEL.name,
            text_recognition_model_dir=str(models.text_recognition.path),
            text_det_limit_side_len=_TEXT_DETECTION_LIMIT_SIDE_LEN,
            text_det_limit_type="max",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            use_ocr_model=True,
            engine="onnxruntime",
            device="cpu",
            enable_hpi=False,
            engine_config=engine_config,
        )
        paddlex_pipeline = pipeline.paddlex_pipeline
        if (
            paddlex_pipeline.wired_table_rec_model
            is not paddlex_pipeline.wireless_table_rec_model
            or paddlex_pipeline.wired_table_cells_detection_model
            is not paddlex_pipeline.wireless_table_cells_detection_model
        ):
            _close(pipeline)
            raise OCRError("PaddleOCR did not preserve the selected table-model cache")
        if execution_provider_base(execution_provider) == "directml":
            try:
                selected_structure_model = (
                    paddlex_pipeline.wired_table_rec_model
                    if table_type is TableType.WIRED
                    else paddlex_pipeline.wireless_table_rec_model
                )
                selected_cells_model = (
                    paddlex_pipeline.wired_table_cells_detection_model
                    if table_type is TableType.WIRED
                    else paddlex_pipeline.wireless_table_cells_detection_model
                )
                models_to_check = (
                    selected_structure_model,
                    selected_cells_model,
                    paddlex_pipeline.general_ocr_pipeline.text_det_model,
                    paddlex_pipeline.general_ocr_pipeline.text_rec_model,
                )
                for child_model in models_to_check:
                    require_execution_provider(
                        child_model.runner.session,
                        execution_provider,
                    )
            except Exception:
                _close(pipeline)
                raise
        return pipeline
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(
            f"Could not initialize PaddleOCR table recognition: {exc}"
        ) from exc


def _get_classifier(models: _ResolvedModels, execution_provider: str) -> Any:
    global _cached_classifier, _cached_classifier_key

    key = models.classification, execution_provider
    if key != _cached_classifier_key:
        _close(_cached_classifier)
        _cached_classifier = None
        _cached_classifier_key = key
    if _cached_classifier is None:
        _cached_classifier = _create_table_classifier(
            models.classification,
            execution_provider,
        )
    return _cached_classifier


def _get_pipeline(
    table_type: TableType,
    models: _ResolvedModels,
    execution_provider: str,
) -> Any:
    _structure_spec, structure, _cells_spec, cells = _selected_models(
        table_type,
        models,
    )
    key = (
        models.classification,
        structure,
        cells,
        models.text_detection,
        models.text_recognition,
        execution_provider,
    )
    cached = _cached_pipelines.get(table_type)
    if cached is not None and key != cached[0]:
        _close(cached[1])
        cached = None
        del _cached_pipelines[table_type]
    if cached is None:
        pipeline = _create_table_pipeline(
            table_type,
            models,
            execution_provider,
        )
        cached = key, pipeline
        _cached_pipelines[table_type] = cached
    return cached[1]


def _prepare_input(input_image: Image.Image | str | Path) -> str | np.ndarray:
    if isinstance(input_image, Image.Image):
        rgb = np.asarray(input_image.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])
    if not isinstance(input_image, str | Path):
        raise TypeError("input_image must be a PIL.Image.Image or a local image path")
    try:
        input_path = Path(input_image).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OCRError(f"Could not open local table image: {exc}") from exc
    if not input_path.is_file():
        raise OCRError(f"Table image is not a regular local file: {input_path}")
    return str(input_path)


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, str | bytes):
        raise OCRError(f"PaddleOCR table result field {name} must be a sequence")
    try:
        return tuple(value)
    except TypeError as exc:
        raise OCRError(
            f"PaddleOCR table result field {name} must be a sequence"
        ) from exc


def _field(result: Any, name: str) -> Any:
    try:
        return result[name]
    except (KeyError, TypeError) as exc:
        raise OCRError(f"PaddleOCR table result is missing {name}") from exc


def _optional_field(result: Any, name: str) -> Any | None:
    try:
        return result[name]
    except (KeyError, TypeError):
        return None


def _classify(classifier: Any, input_data: str | np.ndarray) -> TableType:
    results = _sequence(classifier.predict(input_data), "classification results")
    if len(results) != 1:
        raise OCRError(
            "PaddleOCR returned "
            f"{len(results)} classification results for one table image"
        )
    labels = _sequence(
        _field(results[0], "label_names"),
        "classification label_names",
    )
    if not labels:
        raise OCRError("PaddleOCR table classification returned no label")
    label = labels[0]
    if label == "wired_table":
        return TableType.WIRED
    if label == "wireless_table":
        return TableType.WIRELESS
    raise OCRError(f"PaddleOCR returned unsupported table type {label!r}")


def _positive_span(value: str | None, name: str) -> int:
    if value is None:
        return 1
    try:
        span = int(value)
    except ValueError as exc:
        raise OCRError(f"PaddleOCR table HTML has invalid {name}") from exc
    if span < 1:
        raise OCRError(f"PaddleOCR table HTML has invalid {name}")
    return span


def _html_cell_data(
    html: str,
) -> tuple[tuple[int, int, int, int, str], ...]:
    if not html.strip():
        return ()
    try:
        root = lxml_html.fromstring(html)
    except (etree.ParserError, ValueError) as exc:
        raise OCRError(f"PaddleOCR returned invalid table HTML: {exc}") from exc
    tables = root.xpath("self::table | .//table")
    if not tables:
        raise OCRError("PaddleOCR table HTML does not contain a table")
    rows = tables[0].xpath("./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr")
    occupied: set[tuple[int, int]] = set()
    cells = []
    for row, row_element in enumerate(rows):
        column = 0
        for cell_element in row_element.xpath("./td | ./th"):
            while (row, column) in occupied:
                column += 1
            row_span = _positive_span(cell_element.get("rowspan"), "rowspan")
            column_span = _positive_span(cell_element.get("colspan"), "colspan")
            text = "".join(cell_element.itertext())
            cells.append((row, column, row_span, column_span, text))
            for occupied_row in range(row, row + row_span):
                for occupied_column in range(column, column + column_span):
                    occupied.add((occupied_row, occupied_column))
            column += column_span
    return tuple(cells)


def _bounding_box(value: Any, name: str) -> TableBoundingBox:
    coordinates = _sequence(value, name)
    try:
        numbers = tuple(float(coordinate) for coordinate in coordinates)
    except (TypeError, ValueError) as exc:
        raise OCRError(f"PaddleOCR table result field {name} is invalid") from exc
    if len(numbers) == 4:
        left, top, right, bottom = numbers
    elif len(numbers) == 8:
        left = min(numbers[0::2])
        top = min(numbers[1::2])
        right = max(numbers[0::2])
        bottom = max(numbers[1::2])
    else:
        raise OCRError(
            f"PaddleOCR table result field {name} must contain 4 or 8 coordinates"
        )
    if (
        not all(math.isfinite(number) for number in (left, top, right, bottom))
        or right <= left
        or bottom <= top
    ):
        raise OCRError(f"PaddleOCR table result field {name} is invalid")
    return TableBoundingBox(left=left, top=top, right=right, bottom=bottom)


def _confidence(value: Any, name: str) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise OCRError(f"PaddleOCR table result field {name} is invalid") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise OCRError(f"PaddleOCR table result field {name} is invalid")
    return confidence


def _ocr_data(ocr_result: Any) -> tuple[tuple[TableBoundingBox, float], ...]:
    if not ocr_result:
        return ()
    boxes = _sequence(_field(ocr_result, "rec_boxes"), "rec_boxes")
    scores = _sequence(_field(ocr_result, "rec_scores"), "rec_scores")
    if len(boxes) != len(scores):
        raise OCRError(
            "PaddleOCR returned inconsistent table OCR arrays: "
            f"rec_boxes={len(boxes)}, rec_scores={len(scores)}"
        )
    data = []
    for index, (box, score) in enumerate(zip(boxes, scores, strict=True)):
        data.append(
            (
                _bounding_box(box, f"rec_boxes[{index}]"),
                _confidence(score, f"rec_scores[{index}]"),
            )
        )
    return tuple(data)


def _overlap(cell: TableBoundingBox, ocr: TableBoundingBox) -> float:
    left = max(cell.left, ocr.left)
    top = max(cell.top, ocr.top)
    right = min(cell.right, ocr.right)
    bottom = min(cell.bottom, ocr.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    ocr_area = (ocr.right - ocr.left) * (ocr.bottom - ocr.top)
    return intersection / ocr_area


def _cell_confidence(
    cell_box: TableBoundingBox,
    ocr_data: tuple[tuple[TableBoundingBox, float], ...],
) -> float | None:
    scores = [
        score
        for ocr_box, score in ocr_data
        if _overlap(cell_box, ocr_box) > _OCR_CELL_OVERLAP
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _order_cell_boxes(box_values: tuple[Any, ...]) -> tuple[TableBoundingBox, ...]:
    """Sort detected cell boxes into reading order.

    Rows are grouped by top-coordinate proximity instead of by the row lengths
    of the recognized HTML, because the two come from independent models. A box
    joins the current row while its top stays within ``_CELL_ROW_TOLERANCE`` of
    the top of that row's first box, so a skewed scan does not shift boxes into
    a neighboring row.
    """
    boxes = [
        _bounding_box(box, f"cell_box_list[{index}]")
        for index, box in enumerate(box_values)
    ]
    boxes.sort(key=lambda box: box.top)

    ordered: list[TableBoundingBox] = []
    row: list[TableBoundingBox] = []
    row_top: float | None = None
    for box in boxes:
        if row_top is not None and box.top - row_top <= _CELL_ROW_TOLERANCE:
            row.append(box)
            continue
        row.sort(key=lambda row_box: row_box.left)
        ordered.extend(row)
        row = [box]
        row_top = box.top
    row.sort(key=lambda row_box: row_box.left)
    ordered.extend(row)
    return tuple(ordered)


def _serialize_result(
    table_type: TableType,
    predictions: Any,
) -> TableRecognitionResult:
    results = _sequence(predictions, "prediction results")
    if len(results) != 1:
        raise OCRError(f"PaddleOCR returned {len(results)} results for one table image")
    result = results[0]
    table_results = _sequence(
        _field(result, "table_res_list"),
        "table_res_list",
    )
    if not table_results:
        return TableRecognitionResult(table_type=table_type, html="", cells=())
    if len(table_results) != 1:
        raise OCRError(
            "PaddleOCR returned "
            f"{len(table_results)} tables for one already-cropped table image"
        )

    table_result = table_results[0]
    html = _field(table_result, "pred_html")
    if not isinstance(html, str):
        raise OCRError("PaddleOCR table result field pred_html must be a string")
    box_values = _sequence(_field(table_result, "cell_box_list"), "cell_box_list")
    if not box_values:
        return TableRecognitionResult(table_type=table_type, html=html, cells=())
    cell_data = _html_cell_data(html)
    boxes: tuple[TableBoundingBox | None, ...]
    confidences: tuple[float | None, ...]
    if len(cell_data) == len(box_values):
        detected = _order_cell_boxes(box_values)
        ocr_result = _optional_field(result, "overall_ocr_res")
        if not ocr_result:
            ocr_result = _field(table_result, "table_ocr_pred")
        ocr_data = _ocr_data(ocr_result)
        boxes = detected
        confidences = tuple(_cell_confidence(box, ocr_data) for box in detected)
    else:
        # The structure model builds the HTML and the cell-detection model
        # builds the boxes, so differing counts are expected model behaviour.
        # Text and grid come from the HTML alone and stay usable without boxes.
        logger.warning(
            "PaddleOCR table cell counts differ (html=%d, cell_box_list=%d); "
            "returning cells without bounding boxes and confidence",
            len(cell_data),
            len(box_values),
        )
        boxes = (None,) * len(cell_data)
        confidences = (None,) * len(cell_data)
    cells = tuple(
        TableCell(
            row=row,
            column=column,
            row_span=row_span,
            column_span=column_span,
            text=text,
            bounding_box=box,
            confidence=confidence,
        )
        for (row, column, row_span, column_span, text), box, confidence in zip(
            cell_data,
            boxes,
            confidences,
            strict=True,
        )
    )
    return TableRecognitionResult(table_type=table_type, html=html, cells=cells)


def recognize_table(
    input_image: Image.Image | str | Path,
    *,
    table_classification_model_dir: str | Path,
    wired_table_structure_recognition_model_dir: str | Path,
    wireless_table_structure_recognition_model_dir: str | Path,
    wired_table_cells_detection_model_dir: str | Path,
    wireless_table_cells_detection_model_dir: str | Path,
    detection_model_dir: str | Path,
    recognition_model_dir: str | Path,
    ocr_execution_provider: str = "cpu",
) -> TableRecognitionResult:
    """Recognize one already-cropped table using only local ONNX models.

    Cell ``row`` and ``column`` values are zero-based. Cell confidence is the
    mean OCR confidence of text boxes contained by that cell, or ``None`` for
    an empty cell.

    Grid structure and text come from the structure model, while bounding boxes
    come from the independent cell-detection model. If the two models disagree
    on the number of cells, a warning is logged and every cell is returned with
    ``bounding_box`` and ``confidence`` set to ``None``; rows, columns, spans,
    and text stay usable.
    """
    execution_provider = validate_ocr_execution_provider(ocr_execution_provider)
    with _prediction_lock:
        input_data = _prepare_input(input_image)
        models = _resolve_models(
            table_classification_model_dir=table_classification_model_dir,
            wired_table_structure_recognition_model_dir=(
                wired_table_structure_recognition_model_dir
            ),
            wireless_table_structure_recognition_model_dir=(
                wireless_table_structure_recognition_model_dir
            ),
            wired_table_cells_detection_model_dir=(
                wired_table_cells_detection_model_dir
            ),
            wireless_table_cells_detection_model_dir=(
                wireless_table_cells_detection_model_dir
            ),
            detection_model_dir=detection_model_dir,
            recognition_model_dir=recognition_model_dir,
        )
        try:
            classifier = _get_classifier(models, execution_provider)
            table_type = _classify(classifier, input_data)
            pipeline = _get_pipeline(table_type, models, execution_provider)
            predictions = pipeline.predict(
                input_data,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                use_ocr_model=True,
                use_table_orientation_classify=False,
                use_ocr_results_with_table_cells=True,
                text_det_limit_side_len=_TEXT_DETECTION_LIMIT_SIDE_LEN,
                text_det_limit_type="max",
            )
            return _serialize_result(table_type, predictions)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"PaddleOCR table inference failed: {exc}") from exc


def _reset_model_cache_for_tests() -> None:
    global _cached_classifier, _cached_classifier_key

    with _prediction_lock:
        _close(_cached_classifier)
        for _key, pipeline in _cached_pipelines.values():
            _close(pipeline)
        _cached_classifier = None
        _cached_classifier_key = None
        _cached_pipelines.clear()
