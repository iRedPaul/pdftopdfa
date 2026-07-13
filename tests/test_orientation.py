# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the bundled PaddleOCR orientation preflight."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pikepdf
import pytest
from PIL import Image, ImageDraw, ImageFont

import pdftopdfa.orientation as orientation
from pdftopdfa.exceptions import OCRError
from pdftopdfa.orientation import (
    MODEL_NAME,
    ORIENTATION_BATCH_SIZE,
    ORIENTATION_CONFIDENCE_THRESHOLD,
    _classify_pdf_pages,
    _corrected_page_rotate,
    _get_model,
    _parse_prediction,
    _predict_batch,
    _Prediction,
    _refine_low_confidence_predictions,
    _reset_model_cache_for_tests,
    _validate_model_directory,
    normalize_pdf_orientation,
)


@pytest.fixture(autouse=True)
def _reset_orientation_model():
    """Isolate the process-wide model singleton between tests."""
    _reset_model_cache_for_tests()
    yield
    _reset_model_cache_for_tests()


def _bundled_model_dir() -> Path:
    return (
        Path(__file__).parents[1]
        / "src"
        / "pdftopdfa"
        / "resources"
        / "models"
        / MODEL_NAME
    )


def _make_pdf(path: Path, page_count: int = 1) -> None:
    with pikepdf.Pdf.new() as pdf:
        for _ in range(page_count):
            pdf.add_blank_page(page_size=(595.0, 842.0))
        pdf.save(path)


class TestBundledModel:
    """Tests for local model integrity and lazy loading."""

    def test_bundled_model_manifest_and_hashes_are_valid(self) -> None:
        """The committed model matches its package manifest."""
        model_dir = _bundled_model_dir()

        _validate_model_directory(model_dir)

        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["model_name"] == MODEL_NAME
        assert manifest["engine"] == "onnxruntime"
        assert set(manifest["files"]) == {"inference.onnx", "inference.yml"}
        assert (model_dir / "LICENSE").is_file()
        assert (model_dir / "SOURCE.md").is_file()

    def test_model_validation_rejects_corrupt_artifact(self, tmp_dir: Path) -> None:
        """A model hash mismatch fails before inference."""
        model_dir = tmp_dir / MODEL_NAME
        model_dir.mkdir()
        (model_dir / "inference.onnx").write_bytes(b"corrupt")
        (model_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "engine": "onnxruntime",
                    "files": {"inference.onnx": "0" * 64},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(OCRError, match="corrupt"):
            _validate_model_directory(model_dir)

    @patch("pdftopdfa.orientation._resolve_model_directory")
    @patch("paddleocr.DocImgOrientationClassification")
    def test_model_uses_only_local_cpu_onnx_runtime(
        self,
        mock_model_class: MagicMock,
        mock_resolve: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Model creation fixes the engine, provider, device, and local path."""
        mock_resolve.return_value = tmp_dir

        orientation._create_model()

        mock_model_class.assert_called_once_with(
            model_name=MODEL_NAME,
            model_dir=str(tmp_dir),
            engine="onnxruntime",
            device="cpu",
            engine_config={"providers": ["CPUExecutionProvider"]},
        )

    @patch("pdftopdfa.orientation._create_model")
    def test_model_is_created_once_per_process(self, mock_create: MagicMock) -> None:
        """Repeated calls reuse the same lazy model instance."""
        instance = MagicMock()
        mock_create.return_value = instance

        assert _get_model() is instance
        assert _get_model() is instance
        mock_create.assert_called_once_with()

    @patch("pdftopdfa.orientation._create_model")
    def test_parallel_model_access_creates_one_instance(
        self,
        mock_create: MagicMock,
    ) -> None:
        """Concurrent callers share the same process-wide model instance."""
        instance = MagicMock()
        mock_create.return_value = instance

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: _get_model(), range(32)))

        assert all(result is instance for result in results)
        mock_create.assert_called_once_with()


class TestPredictionValidation:
    """Tests for PaddleOCR output validation and batching."""

    @pytest.mark.parametrize("angle", [0, 90, 180, 270])
    def test_parse_prediction_accepts_supported_angles(self, angle: int) -> None:
        """Every supported correction angle is accepted."""
        result = _parse_prediction(
            {"label_names": [str(angle)], "scores": [0.8]},
            page_number=3,
        )

        assert result == _Prediction(3, angle, 0.8)

    @pytest.mark.parametrize(
        "result",
        [
            {"label_names": ["45"], "scores": [0.9]},
            {"label_names": ["90"], "scores": [float("nan")]},
            {"label_names": ["90"], "scores": [1.1]},
            {"label_names": [], "scores": []},
            {},
        ],
    )
    def test_parse_prediction_rejects_invalid_output(self, result: dict) -> None:
        """Malformed angles and scores abort orientation analysis."""
        with pytest.raises(OCRError, match="Paddle orientation returned"):
            _parse_prediction(result, page_number=1)

    @patch("pdftopdfa.orientation._get_model")
    def test_prediction_errors_are_wrapped(self, mock_get_model: MagicMock) -> None:
        """Inference failures include the affected page range."""
        mock_get_model.return_value.predict.side_effect = RuntimeError("broken")

        with pytest.raises(OCRError, match="pages 5-6"):
            _predict_batch([object(), object()], [5, 6])

    @patch("pdftopdfa.orientation._get_model")
    def test_parallel_predictions_are_serialized(
        self,
        mock_get_model: MagicMock,
    ) -> None:
        """ONNX inference never runs concurrently on the shared model."""
        state_lock = threading.Lock()
        active = 0
        maximum_active = 0

        def predict(_images, batch_size):
            nonlocal active, maximum_active
            assert batch_size == 1
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return [{"label_names": ["0"], "scores": [0.9]}]

        mock_get_model.return_value.predict.side_effect = predict
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda page: _predict_batch([object()], [page]),
                    (1, 2),
                )
            )

        assert maximum_active == 1
        assert results == [[_Prediction(1, 0, 0.9)], [_Prediction(2, 0, 0.9)]]

    @patch("pdftopdfa.orientation._predict_batch")
    def test_all_pdf_pages_are_processed_in_bounded_batches(
        self,
        mock_predict: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Every page is rendered, including pages beyond the first batch."""
        pdf_path = tmp_dir / "nine_pages.pdf"
        _make_pdf(pdf_path, ORIENTATION_BATCH_SIZE + 1)

        def make_results(images, page_numbers):
            assert all(image.flags.c_contiguous for image in images)
            assert all(image.shape[2] == 3 for image in images)
            return [_Prediction(page, 0, 0.9) for page in page_numbers]

        mock_predict.side_effect = make_results

        results = _classify_pdf_pages(pdf_path)

        assert [result.page_number for result in results] == list(range(1, 10))
        assert [len(call.args[0]) for call in mock_predict.call_args_list] == [8, 1]

    @patch("pdftopdfa.orientation._predict_batch")
    def test_uncertain_prediction_uses_rotational_consensus(
        self,
        mock_predict: MagicMock,
    ) -> None:
        """Three agreeing retries can raise an uncertain page above threshold."""
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        mock_predict.side_effect = [
            [
                _Prediction(7, 180, 0.82),
                _Prediction(7, 90, 0.91),
                _Prediction(7, 0, 0.87),
            ]
        ]

        results = _refine_low_confidence_predictions(
            [image],
            [_Prediction(7, 90, 0.70)],
        )

        assert results == [_Prediction(7, 270, 0.91)]
        assert all(item.flags.c_contiguous for item in mock_predict.call_args.args[0])
        assert mock_predict.call_args.args[1] == [7, 7, 7]

    @patch("pdftopdfa.orientation._predict_batch")
    def test_very_weak_prediction_is_not_retried(
        self,
        mock_predict: MagicMock,
    ) -> None:
        """Blank or nearly blank pages avoid unnecessary retry inference."""
        prediction = _Prediction(4, 0, 0.25)

        results = _refine_low_confidence_predictions(
            [np.zeros((10, 10, 3), dtype=np.uint8)],
            [prediction],
        )

        assert results == [prediction]
        mock_predict.assert_not_called()

    @patch("pdftopdfa.orientation._predict_batch")
    def test_spatial_consensus_recovers_sparse_page_orientation(
        self,
        mock_predict: MagicMock,
    ) -> None:
        """Repeated high-confidence page regions can orient a sparse drawing."""
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        mock_predict.side_effect = [
            [
                _Prediction(9, 0, 0.60),
                _Prediction(9, 0, 0.60),
                _Prediction(9, 0, 0.60),
            ],
            [
                _Prediction(9, 180, 0.85),
                _Prediction(9, 180, 0.90),
                _Prediction(9, 0, 0.70),
                _Prediction(9, 0, 0.70),
                _Prediction(9, 180, 0.88),
                _Prediction(9, 90, 0.82),
                _Prediction(9, 0, 0.70),
                _Prediction(9, 0, 0.70),
            ],
        ]

        results = _refine_low_confidence_predictions(
            [image],
            [_Prediction(9, 90, 0.60)],
        )

        assert results == [_Prediction(9, 180, 0.90)]
        assert [len(call.args[0]) for call in mock_predict.call_args_list] == [3, 8]


class TestPdfRotation:
    """Tests for confidence handling and PDF rotation composition."""

    @pytest.mark.parametrize("existing", [0, 90, 180, 270])
    @pytest.mark.parametrize("correction", [0, 90, 180, 270])
    def test_rotation_formula(self, existing: int, correction: int) -> None:
        """Paddle counter-clockwise corrections map to PDF clockwise rotation."""
        assert (
            _corrected_page_rotate(existing, correction)
            == (existing - correction) % 360
        )

    @patch("pdftopdfa.orientation._classify_pdf_pages")
    def test_threshold_and_inherited_rotate_are_applied(
        self,
        mock_classify: MagicMock,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A score of 0.80 rotates, while a lower score leaves the page intact."""
        input_path = tmp_dir / "input.pdf"
        output_path = tmp_dir / "output.pdf"
        _make_pdf(input_path, 2)
        with pikepdf.open(input_path, allow_overwriting_input=True) as pdf:
            parent = pdf.pages[0].obj["/Parent"]
            parent["/Rotate"] = 90
            for page in pdf.pages:
                if "/Rotate" in page.obj:
                    del page.obj["/Rotate"]
            pdf.save(input_path)

        mock_classify.return_value = [
            _Prediction(1, 90, ORIENTATION_CONFIDENCE_THRESHOLD),
            _Prediction(2, 270, ORIENTATION_CONFIDENCE_THRESHOLD - 0.001),
        ]

        with caplog.at_level(logging.WARNING, logger="pdftopdfa.orientation"):
            results = normalize_pdf_orientation(input_path, output_path)

        assert results[0].previous_rotate == 90
        assert results[0].final_rotate == 0
        assert results[0].applied is True
        assert results[1].previous_rotate == 90
        assert results[1].final_rotate == 90
        assert results[1].applied is False
        assert "page 2" in caplog.text
        assert "score=0.7990" in caplog.text
        with pikepdf.open(output_path) as pdf:
            assert int(pdf.pages[0].obj["/Rotate"]) == 0
            assert int(pdf.pages[1].obj["/Rotate"]) == 90
            assert [float(value) for value in pdf.pages[0].MediaBox] == [
                0.0,
                0.0,
                595.0,
                842.0,
            ]


class TestRealOfflineModel:
    """Integration tests for the committed ONNX model."""

    def test_real_model_classifies_four_orientations_without_network(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_dir: Path,
    ) -> None:
        """The packaged model works from an empty cache with networking blocked."""
        for variable in (
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "MODELSCOPE_CACHE",
            "PADDLE_HOME",
            "PADDLE_PDX_CACHE_HOME",
        ):
            cache_dir = tmp_dir / variable.lower()
            cache_dir.mkdir()
            monkeypatch.setenv(variable, str(cache_dir))

        def block_network(*args, **kwargs):
            raise AssertionError("network access is forbidden")

        monkeypatch.setattr(socket.socket, "connect", block_network)

        font_path = (
            Path(__file__).parents[1]
            / "src"
            / "pdftopdfa"
            / "resources"
            / "fonts"
            / "LiberationSans-Regular.ttf"
        )
        font = ImageFont.truetype(font_path, 32)
        upright = Image.new("RGB", (900, 1200), "white")
        draw = ImageDraw.Draw(upright)
        lines = [
            "INVOICE AND DELIVERY DOCUMENT",
            "Customer: Example Industries GmbH",
            "Invoice number: 2026-0713-0042",
            "Description     Quantity       Price",
            "Professional service  12      120.00",
            "Software maintenance   1      450.00",
            "Total                         2249.10",
            "Thank you for your business.",
        ]
        for repeat in range(4):
            for line_number, line in enumerate(lines):
                y = 45 + (repeat * len(lines) + line_number) * 34
                draw.text((50, y), line, fill="black", font=font)

        source_angles = (0, 90, 180, 270)
        images = []
        for angle in source_angles:
            rotated = upright.rotate(angle, expand=True, fillcolor="white")
            rgb = np.asarray(rotated)
            images.append(np.ascontiguousarray(rgb[..., ::-1]))

        results = _predict_batch(images, [1, 2, 3, 4])

        assert [result.correction_angle for result in results] == [0, 270, 180, 90]
        assert all(
            result.score >= ORIENTATION_CONFIDENCE_THRESHOLD for result in results
        )
