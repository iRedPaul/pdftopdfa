# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for offline table recognition."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

import pdftopdfa.table as table
from pdftopdfa import (
    TableBoundingBox,
    TableCell,
    TableRecognitionResult,
    TableType,
    recognize_table,
)
from pdftopdfa.exceptions import OCRError


@pytest.fixture(autouse=True)
def _isolated_model_cache():
    table._reset_model_cache_for_tests()
    yield
    table._reset_model_cache_for_tests()


@pytest.fixture
def model_dirs(tmp_path: Path) -> dict[str, Path]:
    directories = {}
    for name in (
        "table_classification_model_dir",
        "wired_table_structure_recognition_model_dir",
        "wireless_table_structure_recognition_model_dir",
        "wired_table_cells_detection_model_dir",
        "wireless_table_cells_detection_model_dir",
        "detection_model_dir",
        "recognition_model_dir",
    ):
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / "inference.onnx").write_bytes(f"{name} onnx".encode())
        (model_dir / "inference.yml").write_bytes(f"{name} yaml".encode())
        directories[name] = model_dir
    return directories


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    path = tmp_path / "table.png"
    Image.new("RGB", (120, 80), "white").save(path)
    return path


def _prediction(
    *,
    html: str = "<table><tbody><tr><td>Value</td></tr></tbody></table>",
    boxes: object = ((0, 0, 120, 80),),
    ocr_boxes: object = ((10, 10, 100, 60),),
    texts: object = ("Value",),
    scores: object = (0.9,),
) -> list[dict[str, object]]:
    return [
        {
            "table_res_list": [
                {
                    "pred_html": html,
                    "cell_box_list": boxes,
                    "table_ocr_pred": {
                        "rec_boxes": ocr_boxes,
                        "rec_texts": texts,
                        "rec_scores": scores,
                    },
                }
            ]
        }
    ]


def _fake_models(
    label: str,
    prediction: object,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    classifier = SimpleNamespace(
        predict=MagicMock(return_value=[{"label_names": [label], "scores": [0.99]}]),
        close=MagicMock(),
    )
    pipeline = SimpleNamespace(
        predict=MagicMock(return_value=prediction),
        close=MagicMock(),
    )
    return classifier, pipeline


def test_table_api_is_exported_on_package_level() -> None:
    assert recognize_table is table.recognize_table
    assert TableType is table.TableType
    assert TableBoundingBox is table.TableBoundingBox
    assert TableCell is table.TableCell
    assert TableRecognitionResult is table.TableRecognitionResult


@pytest.mark.parametrize(
    ("label", "expected_type", "draw_lines"),
    [
        ("wired_table", TableType.WIRED, True),
        ("wireless_table", TableType.WIRELESS, False),
    ],
)
def test_recognizes_tables_with_and_without_lines(
    model_dirs: dict[str, Path],
    label: str,
    expected_type: TableType,
    draw_lines: bool,
) -> None:
    image = Image.new("RGB", (120, 80), "white")
    if draw_lines:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 119, 79), outline="black")
        draw.line((60, 0, 60, 79), fill="black")
    classifier, pipeline = _fake_models(label, _prediction())

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        result = recognize_table(image, **model_dirs)

    assert result == TableRecognitionResult(
        table_type=expected_type,
        html="<table><tbody><tr><td>Value</td></tr></tbody></table>",
        cells=(
            TableCell(
                row=0,
                column=0,
                row_span=1,
                column_span=1,
                text="Value",
                bounding_box=TableBoundingBox(0.0, 0.0, 120.0, 80.0),
                confidence=0.9,
            ),
        ),
    )


def test_pil_input_is_converted_to_bgr_without_closing_source(
    model_dirs: dict[str, Path],
) -> None:
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    seen_inputs = []

    class Classifier:
        @staticmethod
        def predict(input_data: object) -> list[dict[str, object]]:
            seen_inputs.append(input_data)
            return [{"label_names": ["wired_table"], "scores": [1.0]}]

        @staticmethod
        def close() -> None:
            pass

    class Pipeline:
        @staticmethod
        def predict(input_data: object, **_kwargs: object) -> object:
            seen_inputs.append(input_data)
            return _prediction(boxes=(), ocr_boxes=(), texts=(), scores=())

        @staticmethod
        def close() -> None:
            pass

    with (
        patch.object(table, "_create_table_classifier", return_value=Classifier()),
        patch.object(table, "_create_table_pipeline", return_value=Pipeline()),
    ):
        recognize_table(image, **model_dirs)

    assert image.getpixel((0, 0)) == (10, 20, 30)
    assert len(seen_inputs) == 2
    for input_data in seen_inputs:
        assert isinstance(input_data, np.ndarray)
        assert input_data.flags.c_contiguous
        assert input_data.tolist() == [[[30, 20, 10], [30, 20, 10]]]


@pytest.mark.parametrize("as_string", [False, True])
def test_local_path_is_resolved_before_prediction(
    model_dirs: dict[str, Path],
    image_path: Path,
    as_string: bool,
) -> None:
    classifier, pipeline = _fake_models(
        "wired_table",
        _prediction(boxes=(), ocr_boxes=(), texts=(), scores=()),
    )
    input_path: str | Path = str(image_path) if as_string else image_path

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        recognize_table(input_path, **model_dirs)

    classifier.predict.assert_called_once_with(str(image_path.resolve()))
    assert pipeline.predict.call_args.args == (str(image_path.resolve()),)


def test_url_input_is_rejected_before_paddle_is_loaded(
    model_dirs: dict[str, Path],
) -> None:
    with (
        patch.object(
            table,
            "_create_table_classifier",
            side_effect=AssertionError("PaddleOCR must not be loaded"),
        ),
        pytest.raises(OCRError, match="local table image"),
    ):
        recognize_table("https://example.invalid/table.png", **model_dirs)


def test_missing_local_model_fails_before_paddle_is_loaded(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    (model_dirs["wireless_table_cells_detection_model_dir"] / "inference.onnx").unlink()

    with (
        patch.object(
            table,
            "_create_table_classifier",
            side_effect=AssertionError("PaddleOCR must not be loaded"),
        ),
        pytest.raises(OCRError, match="missing"),
    ):
        recognize_table(image_path, **model_dirs)


@pytest.mark.parametrize(
    ("table_type", "selected_structure", "selected_cells", "unselected"),
    [
        (
            TableType.WIRED,
            "wired_table_structure_recognition_model_dir",
            "wired_table_cells_detection_model_dir",
            (
                "wireless_table_structure_recognition_model_dir",
                "wireless_table_cells_detection_model_dir",
            ),
        ),
        (
            TableType.WIRELESS,
            "wireless_table_structure_recognition_model_dir",
            "wireless_table_cells_detection_model_dir",
            (
                "wired_table_structure_recognition_model_dir",
                "wired_table_cells_detection_model_dir",
            ),
        ),
    ],
)
def test_pipeline_constructor_receives_only_selected_local_model_paths(
    model_dirs: dict[str, Path],
    table_type: TableType,
    selected_structure: str,
    selected_cells: str,
    unselected: tuple[str, str],
) -> None:
    import paddleocr

    constructor_calls = []
    shared_structure = object()
    shared_cells = object()

    class FakeTableRecognitionPipelineV2:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)
            self.paddlex_pipeline = SimpleNamespace(
                wired_table_rec_model=shared_structure,
                wireless_table_rec_model=shared_structure,
                wired_table_cells_detection_model=shared_cells,
                wireless_table_cells_detection_model=shared_cells,
            )

    models = table._resolve_models(**model_dirs)

    with patch.object(
        paddleocr,
        "TableRecognitionPipelineV2",
        FakeTableRecognitionPipelineV2,
    ):
        created = table._create_table_pipeline(table_type, models, "cpu")

    assert isinstance(created, FakeTableRecognitionPipelineV2)
    assert len(constructor_calls) == 1
    kwargs = constructor_calls[0]
    selected_structure_path = str(model_dirs[selected_structure].resolve())
    selected_cells_path = str(model_dirs[selected_cells].resolve())
    assert kwargs["wired_table_structure_recognition_model_dir"] == (
        selected_structure_path
    )
    assert kwargs["wireless_table_structure_recognition_model_dir"] == (
        selected_structure_path
    )
    assert kwargs["wired_table_cells_detection_model_dir"] == selected_cells_path
    assert kwargs["wireless_table_cells_detection_model_dir"] == selected_cells_path
    assert not any(
        str(model_dirs[name].resolve()) in kwargs.values() for name in unselected
    )
    assert kwargs["table_classification_model_dir"] == str(
        model_dirs["table_classification_model_dir"].resolve()
    )
    assert kwargs["text_detection_model_dir"] == str(
        model_dirs["detection_model_dir"].resolve()
    )
    assert kwargs["text_recognition_model_dir"] == str(
        model_dirs["recognition_model_dir"].resolve()
    )
    assert kwargs["engine"] == "onnxruntime"
    assert kwargs["device"] == "cpu"
    assert kwargs["enable_hpi"] is False
    assert kwargs["engine_config"] == {"providers": ["CPUExecutionProvider"]}
    assert kwargs["use_doc_orientation_classify"] is False
    assert kwargs["use_doc_unwarping"] is False
    assert kwargs["use_layout_detection"] is False
    assert kwargs["use_ocr_model"] is True


def test_classifier_constructor_is_local_onnx_only(
    model_dirs: dict[str, Path],
) -> None:
    import paddleocr

    constructor = MagicMock(return_value=SimpleNamespace())
    models = table._resolve_models(**model_dirs)

    with patch.object(paddleocr, "TableClassification", constructor):
        created = table._create_table_classifier(models.classification, "cpu")

    assert created is constructor.return_value
    constructor.assert_called_once_with(
        model_name="PP-LCNet_x1_0_table_cls",
        model_dir=str(model_dirs["table_classification_model_dir"].resolve()),
        topk=1,
        engine="onnxruntime",
        device="cpu",
        enable_hpi=False,
        engine_config={"providers": ["CPUExecutionProvider"]},
    )


def test_frozen_windows_runtime_rejects_first_initialization_in_worker() -> None:
    with (
        patch.object(table, "_cached_runtime", None),
        patch.object(table, "_is_frozen_windows", return_value=True),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        future = executor.submit(table._get_table_runtime)
        with pytest.raises(OCRError, match="imported on the main thread"):
            future.result()


def test_pipeline_builds_selected_structure_and_cells_once(
    model_dirs: dict[str, Path],
) -> None:
    import paddleocr
    import paddleocr._common_args as common_args
    import paddlex.inference.pipelines.table_recognition.pipeline_v2 as paddlex_table

    created_model_names = []

    class FakePaddleXTableRecognitionPipelineV2:
        def __init__(self, config: dict[str, object], **_kwargs: object) -> None:
            submodules = config["SubModules"]
            self.table_cls_model = self.create_model(submodules["TableClassification"])
            self.wired_table_rec_model = self.create_model(
                submodules["WiredTableStructureRecognition"]
            )
            self.wireless_table_rec_model = self.create_model(
                submodules["WirelessTableStructureRecognition"]
            )
            self.wired_table_cells_detection_model = self.create_model(
                submodules["WiredTableCellsDetection"]
            )
            self.wireless_table_cells_detection_model = self.create_model(
                submodules["WirelessTableCellsDetection"]
            )
            self.general_ocr_pipeline = SimpleNamespace(
                text_det_model=object(),
                text_rec_model=object(),
            )

        @staticmethod
        def create_model(config: dict[str, object], **_kwargs: object) -> object:
            created_model_names.append(config["model_name"])
            return object()

        @staticmethod
        def cells_det_results_nms(
            _cells: object,
            _scores: object,
            _threshold: float = 0.3,
        ) -> object:
            raise AssertionError("empty detections must not reach PaddleX NMS")

    class FakeTableRecognitionPipelineV2:
        def __init__(self, **kwargs: object) -> None:
            self._common_args = {}
            self._merged_paddlex_config = {
                "SubModules": {
                    "TableClassification": {
                        "model_name": kwargs["table_classification_model_name"]
                    },
                    "WiredTableStructureRecognition": {
                        "model_name": kwargs[
                            "wired_table_structure_recognition_model_name"
                        ]
                    },
                    "WirelessTableStructureRecognition": {
                        "model_name": kwargs[
                            "wireless_table_structure_recognition_model_name"
                        ]
                    },
                    "WiredTableCellsDetection": {
                        "model_name": kwargs["wired_table_cells_detection_model_name"]
                    },
                    "WirelessTableCellsDetection": {
                        "model_name": kwargs[
                            "wireless_table_cells_detection_model_name"
                        ]
                    },
                }
            }
            self.paddlex_pipeline = self._create_paddlex_pipeline()

    models = table._resolve_models(**model_dirs)
    with (
        patch.object(
            paddleocr,
            "TableRecognitionPipelineV2",
            FakeTableRecognitionPipelineV2,
        ),
        patch.object(
            common_args,
            "prepare_common_init_args",
            return_value={},
        ),
        patch.object(
            paddlex_table,
            "_TableRecognitionPipelineV2",
            FakePaddleXTableRecognitionPipelineV2,
        ),
    ):
        pipeline = table._create_table_pipeline(TableType.WIRED, models, "cpu")

    paddlex_pipeline = pipeline.paddlex_pipeline
    assert created_model_names == [
        "SLANeXt_wired",
        "RT-DETR-L_wired_table_cell_det",
    ]
    assert (
        paddlex_pipeline.wired_table_rec_model
        is paddlex_pipeline.wireless_table_rec_model
    )
    assert (
        paddlex_pipeline.wired_table_cells_detection_model
        is paddlex_pipeline.wireless_table_cells_detection_model
    )
    assert tuple(paddlex_pipeline.table_cls_model(None))[0]["label_names"] == [
        "wired_table"
    ]
    assert paddlex_pipeline.cells_det_results_nms([], []) == ([], [])


def test_classifies_before_creating_selected_pipeline(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    events = []

    class Classifier:
        @staticmethod
        def predict(_input_data: object) -> list[dict[str, object]]:
            events.append("classify")
            return [{"label_names": ["wireless_table"], "scores": [0.99]}]

        @staticmethod
        def close() -> None:
            pass

    pipeline = SimpleNamespace(
        predict=MagicMock(
            side_effect=lambda *_args, **_kwargs: (
                events.append("predict")
                or _prediction(boxes=(), ocr_boxes=(), texts=(), scores=())
            )
        ),
        close=MagicMock(),
    )

    def create_classifier(*_args: object) -> Classifier:
        events.append("create classifier")
        return Classifier()

    def create_pipeline(
        table_type: TableType,
        *_args: object,
    ) -> SimpleNamespace:
        assert table_type is TableType.WIRELESS
        events.append("create wireless pipeline")
        return pipeline

    with (
        patch.object(table, "_create_table_classifier", side_effect=create_classifier),
        patch.object(table, "_create_table_pipeline", side_effect=create_pipeline),
    ):
        recognize_table(image_path, **model_dirs)

    assert events == [
        "create classifier",
        "classify",
        "create wireless pipeline",
        "predict",
    ]


def test_predict_disables_every_implicit_model_source(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    classifier, pipeline = _fake_models(
        "wired_table",
        _prediction(boxes=(), ocr_boxes=(), texts=(), scores=()),
    )

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        recognize_table(image_path, **model_dirs)

    assert pipeline.predict.call_args.args == (str(image_path.resolve()),)
    assert pipeline.predict.call_args.kwargs == {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": False,
        "use_ocr_model": True,
        "use_table_orientation_classify": False,
        "use_ocr_results_with_table_cells": True,
        "text_det_limit_side_len": 1600,
        "text_det_limit_type": "max",
    }


def test_merged_cells_use_zero_based_grid_and_serialize_numpy(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    html = (
        "<table><tbody>"
        '<tr><td rowspan="2">A</td><td colspan="2"><b>B</b></td></tr>'
        "<tr><td>C</td><td>D</td></tr>"
        "</tbody></table>"
    )
    boxes = np.array(
        [
            [0, 0, 30, 80],
            [30, 0, 120, 40],
            [30, 40, 75, 80],
            [75, 40, 120, 80],
        ],
        dtype=np.float32,
    )
    ocr_boxes = np.array(
        [
            [5, 10, 25, 70],
            [40, 10, 110, 30],
            [40, 50, 65, 70],
            [85, 50, 110, 70],
        ],
        dtype=np.float32,
    )
    classifier, pipeline = _fake_models(
        "wired_table",
        _prediction(
            html=html,
            boxes=boxes,
            ocr_boxes=ocr_boxes,
            texts=np.array(["A", "B", "C", "D"], dtype=object),
            scores=np.array([0.95, 0.85, 0.75, 0.65], dtype=np.float32),
        ),
    )

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        result = recognize_table(image_path, **model_dirs)

    assert [
        (cell.row, cell.column, cell.row_span, cell.column_span, cell.text)
        for cell in result.cells
    ] == [
        (0, 0, 2, 1, "A"),
        (0, 1, 1, 2, "B"),
        (1, 1, 1, 1, "C"),
        (1, 2, 1, 1, "D"),
    ]
    assert [cell.confidence for cell in result.cells] == pytest.approx(
        [0.95, 0.85, 0.75, 0.65]
    )
    assert isinstance(result.cells, tuple)
    for cell in result.cells:
        assert type(cell.row) is int
        assert type(cell.column) is int
        assert type(cell.bounding_box.left) is float
        assert type(cell.confidence) is float


def test_cell_boxes_are_mapped_geometrically_and_use_unsplit_ocr_scores(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    html = (
        "<table><tbody>"
        '<tr><td rowspan="2">A</td><td colspan="2">B</td></tr>'
        "<tr><td>C</td><td>D</td></tr>"
        "</tbody></table>"
    )
    prediction = _prediction(
        html=html,
        boxes=(
            (75, 40, 120, 80),
            (30, 0, 120, 40),
            (0, 0, 30, 80),
            (30, 40, 75, 80),
        ),
        ocr_boxes=((5, 10, 25, 70), (40, 10, 110, 30)),
        texts=("A", "split", "B"),
        scores=(0.1,),
    )
    prediction[0]["overall_ocr_res"] = {
        "rec_boxes": (
            (5, 10, 25, 70),
            (40, 10, 110, 30),
            (40, 50, 65, 70),
            (85, 50, 110, 70),
        ),
        "rec_scores": (0.9, 0.8, 0.7, 0.6),
    }
    classifier, pipeline = _fake_models("wired_table", prediction)

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        result = recognize_table(image_path, **model_dirs)

    assert [
        (cell.row, cell.column, cell.bounding_box, cell.confidence)
        for cell in result.cells
    ] == [
        (0, 0, TableBoundingBox(0.0, 0.0, 30.0, 80.0), 0.9),
        (0, 1, TableBoundingBox(30.0, 0.0, 120.0, 40.0), 0.8),
        (1, 1, TableBoundingBox(30.0, 40.0, 75.0, 80.0), 0.7),
        (1, 2, TableBoundingBox(75.0, 40.0, 120.0, 80.0), 0.6),
    ]


def test_cell_confidence_averages_contained_ocr_and_empty_cell_is_none(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    classifier, pipeline = _fake_models(
        "wired_table",
        _prediction(
            html="<table><tr><td>First Second</td><td></td></tr></table>",
            boxes=((0, 0, 60, 80), (60, 0, 120, 80)),
            ocr_boxes=((5, 5, 55, 30), (5, 40, 55, 70)),
            texts=("First", "Second"),
            scores=(0.8, 0.6),
        ),
    )

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        result = recognize_table(image_path, **model_dirs)

    assert result.cells[0].confidence == pytest.approx(0.7)
    assert result.cells[1].confidence is None


@pytest.mark.parametrize(
    "prediction",
    [
        [{"table_res_list": []}],
        [
            {
                "table_res_list": [
                    {
                        "pred_html": "<table></table>",
                        "cell_box_list": [],
                        "table_ocr_pred": {},
                    }
                ]
            }
        ],
    ],
)
def test_empty_results_are_plain_typed_results(
    model_dirs: dict[str, Path],
    image_path: Path,
    prediction: object,
) -> None:
    classifier, pipeline = _fake_models("wireless_table", prediction)

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        result = recognize_table(image_path, **model_dirs)

    assert result.table_type is TableType.WIRELESS
    assert isinstance(result.html, str)
    assert result.cells == ()


@pytest.mark.parametrize("boxes", [(), ((0, 0, 120, 80),)])
def test_inconsistent_html_and_box_cells_keep_structure_without_boxes(
    model_dirs: dict[str, Path],
    image_path: Path,
    caplog: pytest.LogCaptureFixture,
    boxes: tuple,
) -> None:
    html = (
        "<table><tbody>"
        '<tr><td rowspan="2">A</td><td>B</td></tr>'
        "<tr><td>C</td></tr>"
        "</tbody></table>"
    )
    classifier, pipeline = _fake_models(
        "wired_table",
        _prediction(html=html, boxes=boxes),
    )

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
        caplog.at_level(logging.WARNING, logger="pdftopdfa.table"),
    ):
        result = recognize_table(image_path, **model_dirs)

    assert result.html == html
    assert [
        (
            cell.row,
            cell.column,
            cell.row_span,
            cell.column_span,
            cell.text,
            cell.bounding_box,
            cell.confidence,
        )
        for cell in result.cells
    ] == [
        (0, 0, 2, 1, "A", None, None),
        (0, 1, 1, 1, "B", None, None),
        (1, 1, 1, 1, "C", None, None),
    ]
    assert "html=3" in caplog.text
    assert f"cell_box_list={len(boxes)}" in caplog.text


def test_skewed_cell_boxes_are_grouped_into_rows_by_top_tolerance(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    # A skewed scan whose second row starts above the bottom of the first row,
    # and whose HTML row lengths (2, 3) do not match the detected row sizes
    # (3, 2). Rows must be grouped by top proximity, not by the HTML lengths.
    html = (
        "<table><tbody>"
        '<tr><td colspan="2">A</td><td>B</td></tr>'
        "<tr><td>C</td><td>D</td><td>E</td></tr>"
        "</tbody></table>"
    )
    classifier, pipeline = _fake_models(
        "wired_table",
        _prediction(
            html=html,
            boxes=(
                (40, 33, 80, 63),
                (80, 9, 120, 39),
                (0, 0, 40, 30),
                (0, 28, 40, 58),
                (40, 5, 80, 35),
            ),
            ocr_boxes=(),
            texts=(),
            scores=(),
        ),
    )

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(table, "_create_table_pipeline", return_value=pipeline),
    ):
        result = recognize_table(image_path, **model_dirs)

    assert [(cell.text, cell.bounding_box) for cell in result.cells] == [
        ("A", TableBoundingBox(0.0, 0.0, 40.0, 30.0)),
        ("B", TableBoundingBox(40.0, 5.0, 80.0, 35.0)),
        ("C", TableBoundingBox(80.0, 9.0, 120.0, 39.0)),
        ("D", TableBoundingBox(0.0, 28.0, 40.0, 58.0)),
        ("E", TableBoundingBox(40.0, 33.0, 80.0, 63.0)),
    ]


def test_repeated_calls_reuse_models_but_repeat_predictions(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    classifier, pipeline = _fake_models(
        "wired_table",
        _prediction(boxes=(), ocr_boxes=(), texts=(), scores=()),
    )

    with (
        patch.object(
            table,
            "_create_table_classifier",
            return_value=classifier,
        ) as create_classifier,
        patch.object(
            table,
            "_create_table_pipeline",
            return_value=pipeline,
        ) as create_pipeline,
    ):
        first = recognize_table(image_path, **model_dirs)
        second = recognize_table(image_path, **model_dirs)

    assert first == second
    create_classifier.assert_called_once()
    create_pipeline.assert_called_once()
    assert classifier.predict.call_count == 2
    assert pipeline.predict.call_count == 2


def test_type_change_lazily_caches_both_selected_pipelines(
    model_dirs: dict[str, Path],
    image_path: Path,
) -> None:
    classifier = SimpleNamespace(
        predict=MagicMock(
            side_effect=[
                [{"label_names": ["wired_table"], "scores": [1.0]}],
                [{"label_names": ["wireless_table"], "scores": [1.0]}],
                [{"label_names": ["wired_table"], "scores": [1.0]}],
            ]
        ),
        close=MagicMock(),
    )
    wired_pipeline = SimpleNamespace(
        predict=MagicMock(
            return_value=_prediction(boxes=(), ocr_boxes=(), texts=(), scores=())
        ),
        close=MagicMock(),
    )
    wireless_pipeline = SimpleNamespace(
        predict=MagicMock(
            return_value=_prediction(boxes=(), ocr_boxes=(), texts=(), scores=())
        ),
        close=MagicMock(),
    )

    with (
        patch.object(table, "_create_table_classifier", return_value=classifier),
        patch.object(
            table,
            "_create_table_pipeline",
            side_effect=[wired_pipeline, wireless_pipeline],
        ) as create_pipeline,
    ):
        first = recognize_table(image_path, **model_dirs)
        second = recognize_table(image_path, **model_dirs)
        third = recognize_table(image_path, **model_dirs)

    assert first.table_type is TableType.WIRED
    assert second.table_type is TableType.WIRELESS
    assert third.table_type is TableType.WIRED
    assert [call.args[0] for call in create_pipeline.call_args_list] == [
        TableType.WIRED,
        TableType.WIRELESS,
    ]
    wired_pipeline.close.assert_not_called()
    wireless_pipeline.close.assert_not_called()
    assert wired_pipeline.predict.call_count == 2
    classifier.close.assert_not_called()


def test_parallel_predictions_share_cached_models_and_are_serialized(
    model_dirs: dict[str, Path],
) -> None:
    image = Image.new("RGB", (120, 80), "white")
    state_lock = threading.Lock()
    active_predictions = 0
    maximum_active = 0

    def run_serialized(result: object) -> object:
        nonlocal active_predictions, maximum_active
        with state_lock:
            active_predictions += 1
            maximum_active = max(maximum_active, active_predictions)
        time.sleep(0.005)
        with state_lock:
            active_predictions -= 1
        return result

    class Classifier:
        @staticmethod
        def predict(_input_data: object) -> object:
            return run_serialized([{"label_names": ["wired_table"], "scores": [1.0]}])

        @staticmethod
        def close() -> None:
            pass

    class Pipeline:
        @staticmethod
        def predict(_input_data: object, **_kwargs: object) -> object:
            return run_serialized(
                _prediction(boxes=(), ocr_boxes=(), texts=(), scores=())
            )

        @staticmethod
        def close() -> None:
            pass

    with (
        patch.object(
            table,
            "_create_table_classifier",
            return_value=Classifier(),
        ) as create_classifier,
        patch.object(
            table,
            "_create_table_pipeline",
            return_value=Pipeline(),
        ) as create_pipeline,
        ThreadPoolExecutor(max_workers=6) as executor,
    ):
        results = list(
            executor.map(
                lambda _index: recognize_table(image, **model_dirs),
                range(12),
            )
        )

    assert all(result.table_type is TableType.WIRED for result in results)
    assert create_classifier.call_count == 1
    assert create_pipeline.call_count == 1
    assert maximum_active == 1


def test_parallel_mixed_types_each_build_one_lazy_pipeline(
    model_dirs: dict[str, Path],
) -> None:
    wired_image = Image.new("RGB", (1, 1), (255, 0, 0))
    wireless_image = Image.new("RGB", (1, 1), (0, 0, 255))

    class Classifier:
        @staticmethod
        def predict(input_data: np.ndarray) -> object:
            label = "wired_table" if input_data[0, 0, 2] == 255 else "wireless_table"
            return [{"label_names": [label], "scores": [1.0]}]

        @staticmethod
        def close() -> None:
            pass

    pipelines = {
        table_type: SimpleNamespace(
            predict=MagicMock(
                return_value=_prediction(
                    boxes=(),
                    ocr_boxes=(),
                    texts=(),
                    scores=(),
                )
            ),
            close=MagicMock(),
        )
        for table_type in TableType
    }

    with (
        patch.object(table, "_create_table_classifier", return_value=Classifier()),
        patch.object(
            table,
            "_create_table_pipeline",
            side_effect=lambda table_type, *_args: pipelines[table_type],
        ) as create_pipeline,
        ThreadPoolExecutor(max_workers=6) as executor,
    ):
        inputs = [wired_image, wireless_image] * 6
        results = list(
            executor.map(
                lambda image: recognize_table(image, **model_dirs),
                inputs,
            )
        )

    assert [result.table_type for result in results] == [
        TableType.WIRED,
        TableType.WIRELESS,
    ] * 6
    assert sorted(call.args[0] for call in create_pipeline.call_args_list) == [
        TableType.WIRED,
        TableType.WIRELESS,
    ]
    assert pipelines[TableType.WIRED].predict.call_count == 6
    assert pipelines[TableType.WIRELESS].predict.call_count == 6


def test_real_table_models_run_fully_offline_with_onnx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_names = {
        "table_classification_model_dir": (
            "PDFTOPDFA_TEST_TABLE_CLASSIFICATION_MODEL_DIR"
        ),
        "wired_table_structure_recognition_model_dir": (
            "PDFTOPDFA_TEST_WIRED_TABLE_STRUCTURE_MODEL_DIR"
        ),
        "wireless_table_structure_recognition_model_dir": (
            "PDFTOPDFA_TEST_WIRELESS_TABLE_STRUCTURE_MODEL_DIR"
        ),
        "wired_table_cells_detection_model_dir": (
            "PDFTOPDFA_TEST_WIRED_TABLE_CELLS_MODEL_DIR"
        ),
        "wireless_table_cells_detection_model_dir": (
            "PDFTOPDFA_TEST_WIRELESS_TABLE_CELLS_MODEL_DIR"
        ),
        "detection_model_dir": "PDFTOPDFA_TEST_OCR_DETECTION_MODEL_DIR",
        "recognition_model_dir": "PDFTOPDFA_TEST_OCR_RECOGNITION_MODEL_DIR",
    }
    configured_dirs = {
        argument: os.environ.get(environment_name)
        for argument, environment_name in environment_names.items()
    }
    wired_image = os.environ.get("PDFTOPDFA_TEST_WIRED_TABLE_IMAGE")
    wireless_image = os.environ.get("PDFTOPDFA_TEST_WIRELESS_TABLE_IMAGE")
    if (
        any(value is None for value in configured_dirs.values())
        or not wired_image
        or not wireless_image
    ):
        pytest.skip("Offline table model directories and images are not configured")

    for variable in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "MODELSCOPE_CACHE",
        "PADDLE_HOME",
        "PADDLE_PDX_CACHE_HOME",
    ):
        cache_dir = tmp_path / variable.lower()
        cache_dir.mkdir()
        monkeypatch.setenv(variable, str(cache_dir))

    script = """
import builtins
import json
import os
import socket
import sys
from pathlib import Path

def block_network(*_args, **_kwargs):
    raise AssertionError("network access is forbidden")

original_import = builtins.__import__

def block_other_engines(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.partition(".")[0] in {
        "paddle",
        "torch",
        "transformers",
    }:
        raise AssertionError(f"{name} must not be imported")
    return original_import(name, globals, locals, fromlist, level)

socket.socket.connect = block_network
socket.socket.connect_ex = block_network
socket.create_connection = block_network
builtins.__import__ = block_other_engines

from onnxruntime import InferenceSession
from pdftopdfa import TableType, recognize_table
import pdftopdfa.table as table

environment_names = {
    "table_classification_model_dir":
        "PDFTOPDFA_TEST_TABLE_CLASSIFICATION_MODEL_DIR",
    "wired_table_structure_recognition_model_dir":
        "PDFTOPDFA_TEST_WIRED_TABLE_STRUCTURE_MODEL_DIR",
    "wireless_table_structure_recognition_model_dir":
        "PDFTOPDFA_TEST_WIRELESS_TABLE_STRUCTURE_MODEL_DIR",
    "wired_table_cells_detection_model_dir":
        "PDFTOPDFA_TEST_WIRED_TABLE_CELLS_MODEL_DIR",
    "wireless_table_cells_detection_model_dir":
        "PDFTOPDFA_TEST_WIRELESS_TABLE_CELLS_MODEL_DIR",
    "detection_model_dir": "PDFTOPDFA_TEST_OCR_DETECTION_MODEL_DIR",
    "recognition_model_dir": "PDFTOPDFA_TEST_OCR_RECOGNITION_MODEL_DIR",
}
kwargs = {
    argument: Path(os.environ[environment_name])
    for argument, environment_name in environment_names.items()
}
wired = recognize_table(
    Path(os.environ["PDFTOPDFA_TEST_WIRED_TABLE_IMAGE"]),
    **kwargs,
)
wireless = recognize_table(
    Path(os.environ["PDFTOPDFA_TEST_WIRELESS_TABLE_IMAGE"]),
    **kwargs,
)

assert wired.table_type is TableType.WIRED
assert wireless.table_type is TableType.WIRELESS
assert wired.html
assert wireless.html

def assert_cpu_onnx_session(model):
    session = model.runner.session
    assert isinstance(session, InferenceSession)
    assert session.get_providers() == ["CPUExecutionProvider"]

assert_cpu_onnx_session(table._cached_classifier.paddlex_predictor)
for table_type in TableType:
    _key, wrapper = table._cached_pipelines[table_type]
    pipeline = wrapper.paddlex_pipeline
    assert pipeline.wired_table_rec_model is pipeline.wireless_table_rec_model
    assert (
        pipeline.wired_table_cells_detection_model
        is pipeline.wireless_table_cells_detection_model
    )
    assert_cpu_onnx_session(pipeline.wired_table_rec_model)
    assert_cpu_onnx_session(pipeline.wired_table_cells_detection_model)
    assert_cpu_onnx_session(pipeline.general_ocr_pipeline.text_det_model)
    assert_cpu_onnx_session(pipeline.general_ocr_pipeline.text_rec_model)

for module_name in sys.modules:
    assert module_name.partition(".")[0] not in {
        "paddle",
        "torch",
        "transformers",
    }

print(json.dumps({"wired": len(wired.cells), "wireless": len(wireless.cells)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert set(json.loads(completed.stdout.strip().splitlines()[-1])) == {
        "wired",
        "wireless",
    }


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows")
def test_windows_pyinstaller_first_table_call_from_thread_for_cpu_and_directml(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("PyInstaller") is None:
        pytest.skip("PyInstaller is not installed")

    project_root = Path(__file__).resolve().parents[1]
    fake_sources = {
        "pdftopdfa/__init__.py": """
from .table import TableType, recognize_table

__all__ = ["TableType", "recognize_table"]
""",
        "paddleocr/__init__.py": """
import threading
from sklearn.preprocessing import StandardScaler

IMPORT_THREAD_IDENT = threading.current_thread().ident
TableClassification = object
TableRecognitionPipelineV2 = object
""",
        "paddleocr/_common_args.py": """
def prepare_common_init_args(_unused, common_args):
    return common_args
""",
        "paddlex/__init__.py": "",
        "paddlex/inference/__init__.py": "",
        "paddlex/inference/pipelines/__init__.py": "",
        "paddlex/inference/pipelines/table_recognition/__init__.py": "",
        "paddlex/inference/pipelines/table_recognition/pipeline_v2.py": """
class _TableRecognitionPipelineV2:
    pass
""",
        "sklearn/__init__.py": """
import threading

IMPORT_THREAD_IDENT = threading.current_thread().ident
""",
        "sklearn/preprocessing.py": """
import scipy

class StandardScaler:
    pass
""",
        "scipy/__init__.py": """
import threading

IMPORT_THREAD_IDENT = threading.current_thread().ident
""",
    }
    for relative_path, source in fake_sources.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.lstrip(), encoding="utf-8")
    for filename in ("_ocr_runtime.py", "exceptions.py", "table.py"):
        source = project_root / "src" / "pdftopdfa" / filename
        (tmp_path / "pdftopdfa" / filename).write_bytes(source.read_bytes())

    launcher = tmp_path / "table_thread_app.py"
    launcher.write_text(
        """
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from pdftopdfa import TableType, recognize_table
import paddleocr
import pdftopdfa.table as table
import scipy
import sklearn

assert getattr(sys, "frozen", False)
assert paddleocr.IMPORT_THREAD_IDENT == threading.main_thread().ident
assert sklearn.IMPORT_THREAD_IDENT == threading.main_thread().ident
assert scipy.IMPORT_THREAD_IDENT == threading.main_thread().ident
assert table._cached_runtime is not None
assert "sklearn" in sys.modules
assert "scipy" in sys.modules

provider = sys.argv[1]
prediction = [{"table_res_list": []}]
classifier = SimpleNamespace(
    predict=lambda _input: [{"label_names": ["wired_table"], "scores": [1.0]}],
    close=lambda: None,
)
pipeline = SimpleNamespace(
    predict=lambda _input, **_kwargs: prediction,
    close=lambda: None,
)
table._create_table_classifier = lambda _model, _provider: classifier
table._create_table_pipeline = lambda _type, _models, _provider: pipeline

with tempfile.TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory)
    model_dirs = {}
    for name in (
        "table_classification_model_dir",
        "wired_table_structure_recognition_model_dir",
        "wireless_table_structure_recognition_model_dir",
        "wired_table_cells_detection_model_dir",
        "wireless_table_cells_detection_model_dir",
        "detection_model_dir",
        "recognition_model_dir",
    ):
        model_dir = root / name
        model_dir.mkdir()
        (model_dir / "inference.onnx").write_bytes(b"onnx")
        (model_dir / "inference.yml").write_bytes(b"yaml")
        model_dirs[name] = model_dir

    results = []
    errors = []

    def worker():
        try:
            results.append(
                recognize_table(
                    Image.new("RGB", (2, 2), "white"),
                    ocr_execution_provider=provider,
                    **model_dirs,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker, name="table-worker")
    thread.start()
    thread.join(60)
    assert not thread.is_alive()
    if errors:
        raise errors[0]

assert len(results) == 1
assert results[0].table_type is TableType.WIRED
assert table._cached_classifier_key[1] == provider
assert table._cached_pipelines[TableType.WIRED][0][-1] == provider
print(provider)
""".lstrip(),
        encoding="utf-8",
    )

    dist_path = tmp_path / "dist"
    hooks_path = tmp_path / "hooks"
    hooks_path.mkdir()
    for package in ("scipy", "sklearn"):
        (hooks_path / f"hook-{package}.py").write_text("", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--noupx",
            "--additional-hooks-dir",
            str(hooks_path),
            "--name",
            "table-thread",
            "--distpath",
            str(dist_path),
            "--workpath",
            str(tmp_path / "build"),
            "--specpath",
            str(tmp_path),
            str(launcher),
        ],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    executable = dist_path / "table-thread" / "table-thread.exe"
    for provider in ("cpu", "directml"):
        completed = subprocess.run(
            [executable, provider],
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert completed.stdout.strip().splitlines()[-1] == provider
