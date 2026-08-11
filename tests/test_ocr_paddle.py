# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the offline PaddleOCR engine."""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ocrmypdf.helpers import Resolution
from ocrmypdf.hocrtransform import Baseline, BoundingBox, OcrClass
from pdfminer.high_level import extract_text
from pikepdf import Dictionary, Name, Pdf, parse_content_stream
from PIL import Image, ImageDraw, ImageFont

import pdftopdfa.ocr_paddle as ocr_paddle
from pdftopdfa import OCRSession
from pdftopdfa.exceptions import OCRError
from pdftopdfa.ocr import apply_ocr, validate_ocr_languages


@pytest.fixture(autouse=True)
def _isolated_model_cache():
    """Keep the process-wide Paddle session isolated between tests."""
    ocr_paddle._release_model_cache()
    yield
    ocr_paddle._release_model_cache()


@pytest.fixture
def model_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Create small artifacts governed by the same exact-file contract."""
    contents = {
        "detection": {
            "inference.onnx": b"test detection onnx",
            "inference.yml": b"test detection yaml",
        },
        "recognition": {
            "inference.onnx": b"test recognition onnx",
            "inference.yml": b"test recognition yaml",
        },
    }
    specs = {}
    directories = {}
    for kind, artifacts in contents.items():
        model_dir = tmp_path / kind
        model_dir.mkdir()
        for filename, data in artifacts.items():
            (model_dir / filename).write_bytes(data)
        specs[kind] = ocr_paddle._ModelSpec(name=f"test_{kind}")
        directories[kind] = model_dir

    monkeypatch.setattr(ocr_paddle, "_DETECTION_MODEL", specs["detection"])
    monkeypatch.setattr(ocr_paddle, "_RECOGNITION_MODEL", specs["recognition"])
    return directories["detection"], directories["recognition"]


@pytest.fixture
def page_image(tmp_path: Path) -> Path:
    """Create a deterministic page image with useful OCR dimensions."""
    image_path = tmp_path / "page.png"
    Image.new("RGB", (400, 200), "white").save(image_path, dpi=(300, 300))
    return image_path


def _options(
    model_dirs: tuple[Path, Path],
    languages: list[str] | None = None,
    *,
    deskew: bool = False,
    force_ocr: bool = False,
    clean: bool = False,
    execution_provider: str = "cpu",
    layout: bool = False,
) -> SimpleNamespace:
    detection_dir, recognition_dir = model_dirs
    return SimpleNamespace(
        paddle=SimpleNamespace(
            detection_model_dir=detection_dir,
            recognition_model_dir=recognition_dir,
            execution_provider=execution_provider,
            layout=layout,
        ),
        languages=languages or ["en"],
        ocr_engine="paddle",
        deskew=deskew,
        force_ocr=force_ocr,
        clean=clean,
    )


def _result(
    *,
    texts: list[str] | None = None,
    scores: list[float] | None = None,
    polygons: list[list[list[float]]] | None = None,
    boxes: list[list[float]] | None = None,
    **word_data: object,
) -> dict[str, object]:
    texts = ["Hello world"] if texts is None else texts
    scores = [0.95] if scores is None else scores
    polygons = (
        [[[10.0, 20.0], [130.0, 20.0], [130.0, 40.0], [10.0, 40.0]]]
        if polygons is None
        else polygons
    )
    boxes = [[10.0, 20.0, 130.0, 40.0]] if boxes is None else boxes
    return {
        "rec_texts": texts,
        "rec_scores": scores,
        "rec_polys": polygons,
        "rec_boxes": boxes,
        **word_data,
    }


def _two_column_result() -> dict[str, object]:
    texts = ["Right top", "Left top", "Right bottom", "Left bottom"]
    polygons = [
        [[270, 20], [370, 20], [370, 40], [270, 40]],
        [[20, 20], [100, 20], [100, 40], [20, 40]],
        [[270, 80], [370, 80], [370, 100], [270, 100]],
        [[20, 80], [100, 80], [100, 100], [20, 100]],
    ]
    return _result(
        texts=texts,
        scores=[0.9] * 4,
        polygons=polygons,
        boxes=[
            [
                min(point[0] for point in polygon),
                min(point[1] for point in polygon),
                max(point[0] for point in polygon),
                max(point[1] for point in polygon),
            ]
            for polygon in polygons
        ],
        text_word=[[text] for text in texts],
        text_word_region=[[polygon] for polygon in polygons],
    )


def _assert_polygon(
    actual: list[tuple[float, float]] | None,
    expected: list[tuple[float, float]],
) -> None:
    assert actual is not None
    assert len(actual) == len(expected)
    for actual_point, expected_point in zip(actual, expected, strict=True):
        assert actual_point == pytest.approx(expected_point)


def test_model_contract_uses_expected_paddle_names() -> None:
    assert ocr_paddle._DETECTION_MODEL.name == "PP-OCRv6_medium_det"
    assert ocr_paddle._RECOGNITION_MODEL.name == "PP-OCRv6_medium_rec"


@pytest.mark.parametrize(
    ("detection", "recognition"),
    [(None, Path("recognition")), (Path("detection"), None), (None, None)],
)
def test_model_directories_require_complete_pair(
    detection: Path | None,
    recognition: Path | None,
) -> None:
    with pytest.raises(OCRError, match="Both PaddleOCR detection and recognition"):
        ocr_paddle.validate_model_directories(detection, recognition)


def test_model_structure_is_checked_once_before_lazy_session_initialization(
    model_dirs: tuple[Path, Path],
) -> None:
    expected = tuple(path.resolve() for path in model_dirs)

    with (
        patch.object(
            ocr_paddle,
            "_validate_model_directory",
            wraps=ocr_paddle._validate_model_directory,
        ) as validate_directory,
        patch.object(ocr_paddle, "_create_model", return_value=object()),
    ):
        assert ocr_paddle.validate_model_directories(*model_dirs) == expected
        assert ocr_paddle._get_model(_options(model_dirs)) is not None

    assert validate_directory.call_count == 2


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_model_directory_requires_exact_files(
    model_dirs: tuple[Path, Path],
    fault: str,
) -> None:
    detection_dir, recognition_dir = model_dirs
    if fault == "missing":
        (detection_dir / "inference.yml").unlink()
    else:
        (recognition_dir / "README.txt").write_text("unexpected", encoding="utf-8")

    expected_message = "missing" if fault == "missing" else "unexpected"
    with pytest.raises(OCRError, match=expected_message):
        ocr_paddle.validate_model_directories(detection_dir, recognition_dir)


def test_model_validation_accepts_unpinned_regular_artifacts(
    model_dirs: tuple[Path, Path],
) -> None:
    detection_dir, recognition_dir = model_dirs
    (detection_dir / "inference.onnx").write_bytes(b"different model revision")
    (recognition_dir / "inference.yml").write_bytes(b"different model config")

    assert ocr_paddle.validate_model_directories(
        detection_dir,
        recognition_dir,
    ) == (detection_dir.resolve(), recognition_dir.resolve())


@pytest.mark.parametrize("symlink_kind", ["directory", "artifact"])
def test_model_validation_rejects_symlinks(
    model_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    symlink_kind: str,
) -> None:
    detection_dir, recognition_dir = model_dirs
    target = (
        detection_dir
        if symlink_kind == "directory"
        else detection_dir / "inference.onnx"
    )
    original_is_symlink = Path.is_symlink

    def simulated_symlink(path: Path) -> bool:
        return path == target or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", simulated_symlink)

    with pytest.raises(
        OCRError,
        match="symbolic links|does not exist|not a regular file",
    ):
        ocr_paddle.validate_model_directories(detection_dir, recognition_dir)


def test_paddle_constructor_is_offline_and_cpu_only(
    model_dirs: tuple[Path, Path],
) -> None:
    paddle_constructor = MagicMock(return_value=object())
    fake_module = SimpleNamespace(PaddleOCR=paddle_constructor)

    with patch.dict(sys.modules, {"paddleocr": fake_module}):
        model = ocr_paddle._create_model(*model_dirs)

    assert model is paddle_constructor.return_value
    paddle_constructor.assert_called_once_with(
        text_detection_model_name="test_detection",
        text_detection_model_dir=str(model_dirs[0]),
        text_recognition_model_name="test_recognition",
        text_recognition_model_dir=str(model_dirs[1]),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
        return_word_box=True,
        engine="onnxruntime",
        device="cpu",
        engine_config={"providers": ["CPUExecutionProvider"]},
    )


def test_paddle_constructor_uses_directml_without_cpu_fallback(
    model_dirs: tuple[Path, Path],
) -> None:
    engine_config = {
        "providers": ["DmlExecutionProvider"],
        "execution_mode": "sequential",
        "enable_mem_pattern": False,
    }
    detection_session = MagicMock()
    recognition_session = MagicMock()
    detection_session.get_providers.return_value = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    recognition_session.get_providers.return_value = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    model = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(
            text_det_model=SimpleNamespace(
                runner=SimpleNamespace(session=detection_session)
            ),
            text_rec_model=SimpleNamespace(
                runner=SimpleNamespace(session=recognition_session)
            ),
        )
    )
    paddle_constructor = MagicMock(return_value=model)
    fake_module = SimpleNamespace(PaddleOCR=paddle_constructor)

    with (
        patch.dict(sys.modules, {"paddleocr": fake_module}),
        patch.object(
            ocr_paddle,
            "onnxruntime_engine_config",
            return_value=engine_config,
        ),
    ):
        created_model = ocr_paddle._create_model(
            *model_dirs,
            execution_provider="directml",
        )

    assert created_model is model
    paddle_constructor.assert_called_once_with(
        text_detection_model_name="test_detection",
        text_detection_model_dir=str(model_dirs[0]),
        text_recognition_model_name="test_recognition",
        text_recognition_model_dir=str(model_dirs[1]),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
        return_word_box=True,
        engine="onnxruntime",
        device="cpu",
        engine_config=engine_config,
    )
    detection_session.disable_fallback.assert_called_once_with()
    recognition_session.disable_fallback.assert_called_once_with()


def test_paddle_constructor_respects_quiet_logging(
    model_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_logger = logging.getLogger("pdftopdfa")
    paddlex_logger = logging.getLogger("paddlex")
    monkeypatch.setattr(project_logger, "level", logging.ERROR)
    monkeypatch.setattr(paddlex_logger, "level", logging.INFO)

    def paddle_constructor(**_kwargs: object) -> object:
        assert not paddlex_logger.isEnabledFor(logging.INFO)
        return object()

    fake_module = SimpleNamespace(PaddleOCR=paddle_constructor)
    with patch.dict(sys.modules, {"paddleocr": fake_module}):
        ocr_paddle._create_model(*model_dirs)


def test_model_load_and_compatibility_errors_are_wrapped(
    model_dirs: tuple[Path, Path],
) -> None:
    fake_module = SimpleNamespace(
        PaddleOCR=MagicMock(side_effect=ValueError("Model name mismatch"))
    )

    with (
        patch.dict(sys.modules, {"paddleocr": fake_module}),
        pytest.raises(OCRError, match="Could not initialize.*Model name mismatch"),
    ):
        ocr_paddle._create_model(*model_dirs)


def test_model_session_is_lazy_reused_and_replaced_for_new_directories(
    model_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first_model = MagicMock()
    second_model = MagicMock()
    options = _options(model_dirs)
    other_dirs = (tmp_path / "other-detection", tmp_path / "other-recognition")
    for model_dir in other_dirs:
        model_dir.mkdir()
        for filename in ocr_paddle._MODEL_FILENAMES:
            (model_dir / filename).write_bytes(b"other model artifact")
    other_options = _options(other_dirs)

    with patch.object(
        ocr_paddle,
        "_create_model",
        side_effect=[first_model, second_model],
    ) as create_model:
        assert ocr_paddle._get_model(options) is first_model
        assert ocr_paddle._get_model(options) is first_model
        assert ocr_paddle._get_model(other_options) is second_model

    assert create_model.call_count == 2
    first_model.close.assert_called_once_with()


def test_model_session_is_replaced_when_execution_provider_changes(
    model_dirs: tuple[Path, Path],
) -> None:
    cpu_model = MagicMock()
    directml_model = MagicMock()
    cpu_options = _options(model_dirs)
    directml_options = _options(model_dirs, execution_provider="directml")

    with patch.object(
        ocr_paddle,
        "_create_model",
        side_effect=[cpu_model, directml_model],
    ) as create_model:
        assert ocr_paddle._get_model(cpu_options) is cpu_model
        assert ocr_paddle._get_model(cpu_options) is cpu_model
        assert ocr_paddle._get_model(directml_options) is directml_model
        assert ocr_paddle._get_model(directml_options) is directml_model

    assert create_model.call_args_list[0].args == (*model_dirs, "cpu")
    assert create_model.call_args_list[1].args == (*model_dirs, "directml")
    cpu_model.close.assert_called_once_with()
    directml_model.close.assert_not_called()


def test_release_model_cache_closes_loaded_model(
    model_dirs: tuple[Path, Path],
) -> None:
    model = MagicMock()

    with patch.object(ocr_paddle, "_create_model", return_value=model):
        assert ocr_paddle._get_model(_options(model_dirs)) is model

    ocr_paddle._release_model_cache()

    model.close.assert_called_once_with()
    assert ocr_paddle._cached_model is None
    assert ocr_paddle._cached_pair is None
    assert ocr_paddle._cached_fingerprint is None
    assert ocr_paddle._cached_execution_provider is None


def test_explicit_validation_reloads_same_path_after_artifact_replacement(
    model_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first_model = MagicMock()
    second_model = MagicMock()
    options = _options(model_dirs)

    with patch.object(
        ocr_paddle,
        "_create_model",
        side_effect=[first_model, second_model],
    ) as create_model:
        assert ocr_paddle._get_model(options) is first_model

        artifact = model_dirs[0] / "inference.onnx"
        original_stat = artifact.stat()
        replacement = tmp_path / "replacement.onnx"
        replacement.write_bytes(b"x" * original_stat.st_size)
        os.utime(
            replacement,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        replacement.replace(artifact)

        ocr_paddle.validate_model_directories(*model_dirs)
        assert ocr_paddle._get_model(options) is second_model

    assert create_model.call_count == 2
    first_model.close.assert_called_once_with()


def test_failed_same_path_validation_keeps_loaded_model(
    model_dirs: tuple[Path, Path],
) -> None:
    model = MagicMock()
    options = _options(model_dirs)

    with patch.object(ocr_paddle, "_create_model", return_value=model):
        assert ocr_paddle._get_model(options) is model
        artifact = model_dirs[0] / "inference.yml"
        artifact.unlink()

        with pytest.raises(OCRError, match="missing"):
            ocr_paddle.validate_model_directories(*model_dirs)

    assert ocr_paddle._cached_model is model
    model.close.assert_not_called()


def test_parallel_predictions_share_one_session_and_are_serialized(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    state_lock = threading.Lock()
    active_predictions = 0
    maximum_active = 0

    class InstrumentedModel:
        def predict(
            self,
            input_file: str,
            *,
            return_word_box: bool,
            text_det_limit_side_len: int,
            text_det_limit_type: str,
        ):
            nonlocal active_predictions, maximum_active
            assert input_file == str(page_image)
            assert return_word_box is True
            assert text_det_limit_side_len == 1600
            assert text_det_limit_type == "max"
            with state_lock:
                active_predictions += 1
                maximum_active = max(maximum_active, active_predictions)
            time.sleep(0.01)
            with state_lock:
                active_predictions -= 1
            return [_result()]

    model = InstrumentedModel()
    options = _options(model_dirs)
    with (
        patch.object(
            ocr_paddle,
            "_create_model",
            return_value=model,
        ) as create_model,
        ThreadPoolExecutor(max_workers=6) as executor,
    ):
        results = list(
            executor.map(
                lambda _index: ocr_paddle._predict(page_image, options),
                range(12),
            )
        )

    assert all(result["rec_texts"] == ["Hello world"] for result in results)
    assert create_model.call_count == 1
    assert maximum_active == 1


def test_recognize_image_auto_reuses_existing_prediction_results(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    model = SimpleNamespace(
        predict=MagicMock(
            return_value=[
                _result(
                    texts=["First", "Second"],
                    scores=[0.9, 0.8],
                    polygons=[
                        [[10, 20], [100, 20], [100, 40], [10, 40]],
                        [[10, 50], [100, 50], [100, 70], [10, 70]],
                    ],
                    boxes=[[10, 20, 100, 40], [10, 50, 100, 70]],
                )
            ]
        )
    )

    with patch.object(ocr_paddle, "_create_model", return_value=model) as create_model:
        first = ocr_paddle.recognize_image(
            page_image,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
        )
        second = ocr_paddle.recognize_image(
            page_image,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
        )

    assert first == second == [("First", 0.9), ("Second", 0.8)]
    assert create_model.call_count == 1
    assert model.predict.call_count == 2


def test_ocr_session_owns_one_model_for_multiple_images(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    model = MagicMock()
    model.predict.side_effect = [
        [_result(texts=["First"])],
        [_result(texts=["Second"])],
    ]

    with patch.object(ocr_paddle, "_create_model", return_value=model) as create_model:
        with OCRSession(
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
        ) as session:
            create_model.assert_not_called()
            first = session.recognize_image(page_image)
            second = session.recognize_image(page_image)
            model.close.assert_not_called()
        session.close()

    assert first == [("First", 0.95)]
    assert second == [("Second", 0.95)]
    create_model.assert_called_once_with(*model_dirs, "cpu")
    model.close.assert_called_once_with()
    assert ocr_paddle._cached_model is None


def test_ocr_session_closes_its_model_once_after_inference_error(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    model = MagicMock()
    model.predict.side_effect = [
        [_result(texts=["First"])],
        RuntimeError("inference failed"),
    ]

    with (
        patch.object(ocr_paddle, "_create_model", return_value=model) as create_model,
        pytest.raises(OCRError, match="inference failed"),
    ):
        with OCRSession(
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
        ) as session:
            assert session.recognize_image(page_image) == [("First", 0.95)]
            session.recognize_image(page_image)

    create_model.assert_called_once_with(*model_dirs, "cpu")
    model.close.assert_called_once_with()
    assert ocr_paddle._cached_model is None


def test_recognize_image_single_line_bypasses_detection_and_reuses_model(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    text_rec_model = MagicMock(
        side_effect=[
            [{"rec_text": "Whole image", "rec_score": 0.91}],
            [{"rec_text": "Whole image", "rec_score": 0.91}],
        ]
    )
    model = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(text_rec_model=text_rec_model),
        predict=MagicMock(),
    )

    with patch.object(ocr_paddle, "_create_model", return_value=model) as create_model:
        for _index in range(2):
            assert ocr_paddle.recognize_image(
                page_image,
                detection_model_dir=model_dirs[0],
                recognition_model_dir=model_dirs[1],
                layout="single_line",
            ) == [("Whole image", 0.91)]

    assert create_model.call_count == 1
    assert text_rec_model.call_count == 2
    model.predict.assert_not_called()


def test_allowed_characters_mask_ctc_logits_before_decoding(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    class Decoder:
        character = ["blank", "A", "B"]

        @staticmethod
        def get_ignored_tokens() -> list[int]:
            return [0]

        def __call__(
            self,
            prediction: list[np.ndarray],
            **_kwargs: object,
        ) -> tuple[list[str], list[float]]:
            logits = prediction[0]
            indices = logits.argmax(axis=-1)[0]
            selected = np.concatenate(([True], indices[1:] != indices[:-1]))
            selected &= indices != 0
            text = "".join(self.character[index] for index in indices[selected])
            score = float(logits.max(axis=-1)[0][selected].mean())
            return [text], [score]

    class TextRecognitionModel:
        def __init__(self) -> None:
            self.post_op = Decoder()
            self.logits = np.array(
                [[[0.1, 0.8, 0.9], [0.9, 0.05, 0.05]]],
                dtype=np.float32,
            )

        def __call__(
            self,
            _input_file: str,
            *,
            return_word_box: bool,
        ) -> list[dict[str, object]]:
            assert return_word_box is False
            texts, scores = self.post_op(
                [self.logits],
                return_word_box=return_word_box,
            )
            return [{"rec_text": texts[0], "rec_score": scores[0]}]

    text_rec_model = TextRecognitionModel()
    decoder = text_rec_model.post_op
    model = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(text_rec_model=text_rec_model),
        predict=MagicMock(),
    )

    with patch.object(ocr_paddle, "_create_model", return_value=model):
        restricted = ocr_paddle.recognize_image(
            page_image,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
            layout="single_line",
            allowed_characters="A",
        )
        unrestricted = ocr_paddle.recognize_image(
            page_image,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
            layout="single_line",
        )

    assert restricted == [("A", pytest.approx(0.8))]
    assert unrestricted == [("B", pytest.approx(0.9))]
    assert text_rec_model.post_op is decoder
    assert np.isfinite(text_rec_model.logits).all()


def test_allowed_characters_also_restrict_auto_layout_decoding(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    decoder = MagicMock()
    decoder.character = ["blank", "A", "B"]
    decoder.get_ignored_tokens.return_value = [0]

    def decode(
        prediction: list[np.ndarray],
        **_kwargs: object,
    ) -> tuple[list[str], list[float]]:
        assert np.isneginf(prediction[0][..., 2]).all()
        return ["A"], [0.8]

    decoder.side_effect = decode
    text_rec_model = SimpleNamespace(post_op=decoder)

    class Model:
        paddlex_pipeline = SimpleNamespace(text_rec_model=text_rec_model)

        @staticmethod
        def predict(_input_file: str, **_kwargs: object) -> list[dict[str, object]]:
            texts, scores = text_rec_model.post_op(
                [np.array([[[0.1, 0.8, 0.9]]], dtype=np.float32)]
            )
            return [_result(texts=texts, scores=scores)]

    with patch.object(ocr_paddle, "_create_model", return_value=Model()):
        results = ocr_paddle.recognize_image(
            page_image,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
            allowed_characters="A",
        )

    assert results == [("A", 0.8)]
    assert text_rec_model.post_op is decoder


def test_latin_character_filter_keeps_accents_numbers_and_punctuation() -> None:
    for character in ("A", "ä", "ß", "é", "1", "€", "—"):
        assert ocr_paddle._has_only_latin_letters(character)

    for character in ("𩡥", "中", "Ж", "あ", "\uf140"):
        assert not ocr_paddle._has_only_latin_letters(character)


@pytest.mark.parametrize(
    ("languages", "expected_text"),
    [
        pytest.param(["de"], "A", id="latin"),
        pytest.param(["ch"], "𩡥", id="chinese"),
    ],
)
def test_generate_ocr_restricts_non_latin_letters_for_latin_languages(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    languages: list[str],
    expected_text: str,
) -> None:
    class Decoder:
        character = ["blank", "A", "𩡥"]

        @staticmethod
        def get_ignored_tokens() -> list[int]:
            return [0]

        def __call__(
            self,
            prediction: list[np.ndarray],
            **_kwargs: object,
        ) -> tuple[list[str], list[float]]:
            logits = prediction[0]
            index = int(logits.argmax(axis=-1)[0, 0])
            return [self.character[index]], [float(logits[0, 0, index])]

    decoder = Decoder()
    text_rec_model = SimpleNamespace(post_op=decoder)

    class Model:
        paddlex_pipeline = SimpleNamespace(text_rec_model=text_rec_model)

        @staticmethod
        def predict(_input_file: str, **_kwargs: object) -> list[dict[str, object]]:
            texts, scores = text_rec_model.post_op(
                [np.array([[[0.1, 0.8, 0.9]]], dtype=np.float32)]
            )
            return [_result(texts=texts, scores=scores)]

    with patch.object(ocr_paddle, "_create_model", return_value=Model()):
        _page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs, languages),
        )

    assert plain_text == expected_text
    assert text_rec_model.post_op is decoder


def test_generate_ocr_maps_lines_words_geometry_and_metadata(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["Hello world", "Second"],
        scores=[0.95, 0.75],
        polygons=[
            [[-10, 20], [130, 20], [130, 40], [-10, 40]],
            [[20, 60], [110, 65], [108, 85], [18, 80]],
        ],
        boxes=[[-10, 20, 130, 40], [18, 60, 110, 85]],
        text_word=[["Hello", "world"], ["Second"]],
        text_word_region=[
            [
                [[-5, 20], [55, 20], [55, 40], [-5, 40]],
                [[60, 20], [130, 20], [130, 40], [60, 40]],
            ],
            [[[20, 60], [110, 65], [108, 85], [18, 80]]],
        ],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs, ["de", "en"]),
            page_number=7,
        )

    assert page.ocr_class == OcrClass.PAGE
    assert page.bbox == BoundingBox(left=0, top=0, right=400, bottom=200)
    assert page.dpi == pytest.approx(300, abs=0.1)
    assert page.page_number == 7
    assert page.language == "de+en"
    assert plain_text == "Hello world\nSecond"
    assert [line.text for line in page.children] == ["Hello world", "Second"]
    assert page.children[0].bbox == BoundingBox(left=0, top=20, right=130, bottom=40)
    assert page.children[0].confidence == 0.95
    assert [word.text for word in page.children[0].children] == ["Hello", "world"]
    assert page.children[0].children[0].bbox == BoundingBox(
        left=0,
        top=20,
        right=55,
        bottom=40,
    )
    assert page.children[1].textangle == pytest.approx(-math.degrees(math.atan2(5, 90)))


def test_column_detection_prefers_structural_gap_over_amount_columns() -> None:
    regions = [
        ocr_paddle._LayoutRegion(10, 10, 130, 20),
        ocr_paddle._LayoutRegion(10, 30, 130, 40),
        ocr_paddle._LayoutRegion(160, 10, 180, 20),
        ocr_paddle._LayoutRegion(160, 30, 180, 40),
        ocr_paddle._LayoutRegion(230, 10, 350, 20),
        ocr_paddle._LayoutRegion(230, 30, 350, 40),
        ocr_paddle._LayoutRegion(380, 10, 395, 20),
        ocr_paddle._LayoutRegion(380, 30, 395, 40),
    ]

    assert ocr_paddle._column_regions(regions, 400, 200) == [
        ocr_paddle._LayoutRegion(0, 0, 205, 200),
        ocr_paddle._LayoutRegion(205, 0, 400, 200),
    ]


def test_layout_orders_detected_columns_without_rerunning_ocr(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    with patch.object(
        ocr_paddle,
        "_predict",
        return_value=_two_column_result(),
    ) as predict:
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs, layout=True),
        )

    assert [line.text for line in page.children] == [
        "Left top",
        "Left bottom",
        "Right top",
        "Right bottom",
    ]
    assert plain_text == "Left top\nLeft bottom\nRight top\nRight bottom"
    predict.assert_called_once()


@pytest.mark.parametrize(
    (
        "line_polygon",
        "word_regions",
        "expected_word_polygons",
        "expected_textangle",
    ),
    [
        pytest.param(
            [[20, 20], [220, 20], [220, 60], [20, 60]],
            [
                [[20, 20], [100, 20], [100, 60], [20, 60]],
                [[120, 20], [220, 20], [220, 60], [120, 60]],
            ],
            [
                [(20, 20), (100, 20), (100, 60), (20, 60)],
                [(120, 20), (220, 20), (220, 60), (120, 60)],
            ],
            0.0,
            id="straight",
        ),
        pytest.param(
            [[20, 20], [120, 20], [120, 140], [20, 140]],
            [
                [[20, 20], [60, 20], [60, 140], [20, 140]],
                [[70, 20], [120, 20], [120, 140], [70, 140]],
            ],
            [
                [(20, 20), (60, 20), (60, 140), (20, 140)],
                [(70, 20), (120, 20), (120, 140), (70, 140)],
            ],
            0.0,
            id="near-square-horizontal",
        ),
        pytest.param(
            [[20, 20], [220, 40], [216, 80], [16, 60]],
            [
                [[20, 20], [100, 20], [100, 80], [20, 80]],
                [[120, 20], [220, 20], [220, 80], [120, 80]],
            ],
            [
                [(20, 20), (100, 28), (96, 68), (16, 60)],
                [(120, 30), (220, 40), (216, 80), (116, 70)],
            ],
            -math.degrees(math.atan2(20, 200)),
            id="skewed",
        ),
        pytest.param(
            [[220, 20], [260, 20], [240, 180], [200, 180]],
            [
                [[220, 20], [220, 84], [260, 84], [260, 20]],
                [[220, 100], [220, 180], [260, 180], [260, 100]],
            ],
            [
                [(220, 20), (260, 20), (252, 84), (212, 84)],
                [(210, 100), (250, 100), (240, 180), (200, 180)],
            ],
            -math.degrees(math.atan2(160, -20)),
            id="rotated-90",
        ),
    ],
)
def test_generate_ocr_projects_recognition_spans_onto_line_polygon(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    line_polygon: list[list[float]],
    word_regions: list[list[list[float]]],
    expected_word_polygons: list[list[tuple[float, float]]],
    expected_textangle: float,
) -> None:
    xs = [point[0] for point in line_polygon]
    ys = [point[1] for point in line_polygon]
    result = _result(
        texts=["First second"],
        polygons=[line_polygon],
        boxes=[[min(xs), min(ys), max(xs), max(ys)]],
        text_word=[["First", "second"]],
        text_word_region=[word_regions],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    line = page.children[0]
    assert plain_text == "First second"
    assert line.text == "First second"
    _assert_polygon(line.poly, [tuple(point) for point in line_polygon])
    assert line.baseline == Baseline(slope=0.0, intercept=0.0)
    assert line.textangle == pytest.approx(expected_textangle)
    assert [word.text for word in line.children] == ["First", "second"]
    assert all(word.bbox != line.bbox for word in line.children)
    for word, expected_polygon in zip(
        line.children,
        expected_word_polygons,
        strict=True,
    ):
        _assert_polygon(word.poly, expected_polygon)


def test_generate_ocr_keeps_narrow_single_character_horizontal(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["I"],
        polygons=[[[100, 20], [110, 20], [110, 50], [100, 50]]],
        boxes=[[100, 20, 110, 50]],
        text_word=[["I"]],
        text_word_region=[[[[100, 20], [100, 50], [110, 50], [110, 20]]]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    assert page.children[0].textangle == 0.0


def test_generate_ocr_projects_steep_line_along_its_long_edges(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    angle = math.radians(80)
    direction = (100 * math.cos(angle), 100 * math.sin(angle))
    normal = (-20 * math.sin(angle), 20 * math.cos(angle))
    line_polygon = [
        (100, 20),
        (100 + direction[0], 20 + direction[1]),
        (
            100 + direction[0] + normal[0],
            20 + direction[1] + normal[1],
        ),
        (100 + normal[0], 20 + normal[1]),
    ]
    line_start = line_polygon[0][1]
    line_end = line_polygon[2][1]
    boundaries = (0.0, 0.4, 0.5, 1.0)
    word_regions = [
        [
            (line_polygon[0][0], line_start + (line_end - line_start) * start),
            (line_polygon[0][0], line_start + (line_end - line_start) * end),
            (line_polygon[1][0], line_start + (line_end - line_start) * end),
            (line_polygon[1][0], line_start + (line_end - line_start) * start),
        ]
        for start, end in zip(boundaries[::2], boundaries[1::2], strict=True)
    ]
    result = _result(
        texts=["First second"],
        polygons=[line_polygon],
        boxes=[[80, 20, 118, 122]],
        text_word=[["First", "second"]],
        text_word_region=[word_regions],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    line = page.children[0]
    assert line.textangle == pytest.approx(-80)
    expected_boundaries = ((0.0, 0.4), (0.5, 1.0))
    for word, (start, end) in zip(
        line.children,
        expected_boundaries,
        strict=True,
    ):
        expected_polygon = [
            (
                line_polygon[0][0] + direction[0] * start,
                line_polygon[0][1] + direction[1] * start,
            ),
            (
                line_polygon[0][0] + direction[0] * end,
                line_polygon[0][1] + direction[1] * end,
            ),
            (
                line_polygon[3][0] + direction[0] * end,
                line_polygon[3][1] + direction[1] * end,
            ),
            (
                line_polygon[3][0] + direction[0] * start,
                line_polygon[3][1] + direction[1] * start,
            ),
        ]
        _assert_polygon(word.poly, expected_polygon)


def test_generate_ocr_projects_before_clipping_at_page_edge(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["Visible"],
        polygons=[[[-20, 20], [180, 60], [170, 100], [-30, 60]]],
        boxes=[[-30, 20, 180, 100]],
        text_word=[["Visible"]],
        text_word_region=[[[[-10, 20], [70, 20], [70, 100], [-10, 100]]]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    word = page.children[0].children[0]
    _assert_polygon(
        word.poly,
        [(0, 22), (70, 38), (60, 78), (0, 62)],
    )


def test_generate_ocr_derives_orientation_from_full_line_polygon(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["Edge"],
        polygons=[[[-200, 20], [20, 30], [20, 70], [-200, 60]]],
        boxes=[[-200, 20, 20, 70]],
        text_word=[["Edge"]],
        text_word_region=[[[[-10, 20], [20, 20], [20, 70], [-10, 70]]]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    line = page.children[0]
    _assert_polygon(line.poly, [(0, 20), (20, 30), (20, 70), (0, 60)])
    assert line.textangle == pytest.approx(-math.degrees(math.atan2(20, 440)))


def test_cpu_and_directml_produce_identical_ocr_output(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["Hello world", "Second line"],
        scores=[0.95, 0.8],
        polygons=[
            [[10, 20], [130, 20], [130, 40], [10, 40]],
            [[20, 60], [150, 60], [150, 80], [20, 80]],
        ],
        boxes=[[10, 20, 130, 40], [20, 60, 150, 80]],
        text_word=[["Hello", "world"], ["Second", "line"]],
        text_word_boxes=[
            [[10, 20, 55, 40], [60, 20, 130, 40]],
            [[20, 60, 85, 80], [90, 60, 150, 80]],
        ],
    )

    outputs = []
    for execution_provider in ("cpu", "directml"):
        with patch.object(ocr_paddle, "_predict", return_value=result):
            outputs.append(
                ocr_paddle.PaddleOcrEngine.generate_ocr(
                    page_image,
                    _options(
                        model_dirs,
                        ["de", "en"],
                        execution_provider=execution_provider,
                    ),
                    page_number=7,
                )
            )

    cpu_page, cpu_plain_text = outputs[0]
    directml_page, directml_plain_text = outputs[1]
    assert directml_plain_text == cpu_plain_text
    assert directml_page == cpu_page


def test_cpu_and_directml_real_inference_produce_identical_ocr_output(
    tmp_path: Path,
) -> None:
    """Compare real providers when DirectML hardware and test models are available."""
    detection_dir = os.environ.get("PDFTOPDFA_TEST_OCR_DETECTION_MODEL_DIR")
    recognition_dir = os.environ.get("PDFTOPDFA_TEST_OCR_RECOGNITION_MODEL_DIR")
    if not detection_dir or not recognition_dir:
        pytest.skip("DirectML OCR model directories are not configured")

    try:
        import onnxruntime
    except ImportError:
        pytest.skip("ONNX Runtime is not installed")
    if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
        pytest.skip("DmlExecutionProvider is not available")

    image_path = tmp_path / "directml-parity.png"
    image = Image.new("RGB", (1200, 300), "white")
    font_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pdftopdfa"
        / "resources"
        / "fonts"
        / "LiberationSans-Regular.ttf"
    )
    ImageDraw.Draw(image).text(
        (50, 80),
        "DIRECTML OCR PARITY 123",
        fill="black",
        font=ImageFont.truetype(font_path, 80),
    )
    image.save(image_path, dpi=(300, 300))

    model_dirs = Path(detection_dir), Path(recognition_dir)
    outputs = []
    for execution_provider in ("cpu", "directml"):
        try:
            outputs.append(
                ocr_paddle.PaddleOcrEngine.generate_ocr(
                    image_path,
                    _options(model_dirs, execution_provider=execution_provider),
                    page_number=1,
                )
            )
        except OCRError as exc:
            if execution_provider == "directml" and "refusing CPU fallback" in str(exc):
                pytest.skip("No usable DirectML device is available")
            raise

    cpu_page, cpu_plain_text = outputs[0]
    directml_page, directml_plain_text = outputs[1]
    assert cpu_plain_text.strip()
    assert directml_plain_text == cpu_plain_text
    assert directml_page == cpu_page


def test_generate_ocr_accepts_axis_aligned_word_boxes(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        text_word=[["Hello", "world"]],
        text_word_boxes=[[[10, 20, 60, 40], [65, 20, 130, 40]]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    assert [word.text for word in page.children[0].children] == ["Hello", "world"]
    assert page.children[0].children[1].bbox == BoundingBox(
        left=65,
        top=20,
        right=130,
        bottom=40,
    )


def test_generate_ocr_groups_space_tokens_and_attaches_punctuation(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["AI systems,"],
        polygons=[[[20, 20], [220, 40], [216, 80], [16, 60]]],
        boxes=[[16, 20, 220, 80]],
        text_word=[["AI", " ", "systems", ","]],
        text_word_region=[
            [
                [[20, 20], [60, 20], [60, 80], [20, 80]],
                [[60, 20], [80, 20], [80, 80], [60, 80]],
                [[80, 20], [180, 20], [180, 80], [80, 80]],
                [[180, 20], [190, 20], [190, 80], [180, 80]],
            ]
        ],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    assert [word.text for word in page.children[0].children] == ["AI", "systems,"]
    assert page.children[0].children[1].bbox == BoundingBox(
        left=76,
        top=26,
        right=190,
        bottom=77,
    )
    _assert_polygon(
        page.children[0].children[1].poly,
        [(80, 26), (190, 37), (186, 77), (76, 66)],
    )


def test_generate_ocr_combines_vertical_tokens_without_losing_orientation(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["AB,"],
        polygons=[[[100, 20], [140, 20], [140, 180], [100, 180]]],
        boxes=[[100, 20, 140, 180]],
        text_word=[["A", "B", ","]],
        text_word_region=[
            [
                [[100, 20], [100, 70], [140, 70], [140, 20]],
                [[100, 70], [100, 120], [140, 120], [140, 70]],
                [[100, 120], [100, 140], [140, 140], [140, 120]],
            ]
        ],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    words = page.children[0].children
    assert [word.text for word in words] == ["AB,"]
    _assert_polygon(
        words[0].poly,
        [(100, 20), (140, 20), (140, 140), (100, 140)],
    )


def test_generate_ocr_scales_raster_coordinates_to_ocrmypdf_page_dpi(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    page_context = SimpleNamespace(
        pageinfo=SimpleNamespace(
            dpi=SimpleNamespace(to_scalar=lambda: 150.0),
        ),
        get_path=lambda _name: page_image,
    )
    ocr_paddle.filter_ocr_image(
        page_context,
        Image.new("RGB", (400, 200), "white"),
    )

    result = _result(
        texts=["Scale"],
        polygons=[[[20, 20], [220, 30], [210, 90], [10, 60]]],
        boxes=[[10, 20, 220, 90]],
        text_word=[["Scale"]],
        text_word_region=[[[[20, 20], [220, 20], [220, 90], [20, 90]]]],
    )
    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    assert page.dpi == 150.0
    assert page.bbox == BoundingBox(left=0, top=0, right=200, bottom=100)
    assert page.children[0].bbox == BoundingBox(
        left=5,
        top=10,
        right=110,
        bottom=45,
    )
    _assert_polygon(
        page.children[0].poly,
        [(10, 10), (110, 15), (105, 45), (5, 30)],
    )
    assert page.children[0].baseline.slope == pytest.approx(0.0492610837)
    assert page.children[0].baseline.intercept == pytest.approx(-4.975185951)
    _assert_polygon(
        page.children[0].children[0].poly,
        [(10, 10), (110, 15), (105, 45), (5, 30)],
    )
    assert ocr_paddle._coordinate_dpi_by_image == {}


def test_filter_ocr_image_supplies_dpi_for_vector_only_force_page(
    tmp_path: Path,
) -> None:
    page_image = tmp_path / "vector-force.png"
    Image.new("RGB", (600, 600), "white").save(page_image, dpi=(600, 600))

    class VectorPageInfo:
        _dpi: Resolution | None = None

        @property
        def dpi(self) -> Resolution:
            return self._dpi or Resolution(0.0, 0.0)

    pageinfo = VectorPageInfo()
    page_context = SimpleNamespace(
        pageinfo=pageinfo,
        get_path=lambda _name: page_image,
    )
    image = Image.new("RGB", (600, 600), "white")
    image.info["dpi"] = (600.0, 600.0)

    ocr_paddle.filter_ocr_image(page_context, image)

    assert pageinfo.dpi.to_scalar() == 600.0
    assert ocr_paddle._coordinate_dpi_by_image == {page_image.resolve(): 600.0}


def test_vector_only_force_page_completes_without_zero_dpi(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
) -> None:
    input_path = tmp_path / "vector.pdf"
    output_path = tmp_path / "forced.pdf"
    with Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(72, 72))
        font = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name.Helvetica,
        )
        page.obj[Name.Resources] = Dictionary(Font=Dictionary(F1=font))
        page.obj[Name.Contents] = pdf.make_stream(
            b"BT /F1 10 Tf 5 36 Td (Vector text) Tj ET"
        )
        pdf.save(input_path)

    with patch.object(
        ocr_paddle,
        "_predict",
        return_value=_result(texts=[], scores=[], polygons=[], boxes=[]),
    ):
        apply_ocr(
            input_path,
            output_path,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
            force=True,
        )

    with Pdf.open(output_path) as pdf:
        assert len(pdf.pages) == 1


def test_rotated_page_grafts_polygon_text_layer(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
) -> None:
    input_path = tmp_path / "rotated.pdf"
    output_path = tmp_path / "rotated-ocr.pdf"
    with Pdf.new() as pdf:
        image = pdf.make_stream(b"\x80")
        image[Name.Type] = Name.XObject
        image[Name.Subtype] = Name.Image
        image[Name.Width] = 1
        image[Name.Height] = 1
        image[Name.ColorSpace] = Name.DeviceGray
        image[Name.BitsPerComponent] = 8
        page = pdf.add_blank_page(page_size=(72, 144))
        page.obj[Name.Rotate] = 90
        page.obj[Name.Resources] = Dictionary(XObject=Dictionary(Im0=image))
        page.obj[Name.Contents] = pdf.make_stream(b"q 72 0 0 144 0 0 cm /Im0 Do Q")
        pdf.save(input_path)

    result = _result(
        texts=["Rotate test"],
        polygons=[[[100, 100], [500, 120], [498, 220], [98, 200]]],
        boxes=[[98, 100, 500, 220]],
        text_word=[["Rotate", "test"]],
        text_word_region=[
            [
                [[100, 100], [300, 100], [300, 220], [100, 220]],
                [[320, 100], [500, 100], [500, 220], [320, 220]],
            ]
        ],
    )
    with patch.object(ocr_paddle, "_predict", return_value=result):
        apply_ocr(
            input_path,
            output_path,
            detection_model_dir=model_dirs[0],
            recognition_model_dir=model_dirs[1],
        )

    assert "".join(extract_text(output_path).split()) == "Rotatetest"
    with Pdf.open(output_path) as pdf:
        page = pdf.pages[0]
        assert int(page.Rotate) == 90
        forms = [
            xobject
            for name, xobject in page.Resources.XObject.items()
            if str(name).startswith("/OCR-")
        ]
        assert len(forms) == 1
        assert [float(value) for value in forms[0].BBox] == [0, 0, 144, 72]
        page_matrix = next(
            instruction.operands
            for instruction in parse_content_stream(page)
            if str(instruction.operator) == "cm"
        )
        assert [float(value) for value in page_matrix] == pytest.approx(
            [0, 1, -1, 0, 72, 0]
        )

        form_instructions = list(parse_content_stream(forms[0]))
        line_matrix = next(
            instruction.operands
            for instruction in form_instructions
            if str(instruction.operator) == "cm"
        )
        assert [float(value) for value in line_matrix] == pytest.approx(
            [0.998752, -0.049938, 0.049938, 0.998752, 11.40, 48.02],
            abs=1e-6,
        )
        word_positions = [
            [float(value) for value in instruction.operands]
            for instruction in form_instructions
            if str(instruction.operator) == "Td"
        ]
        assert len(word_positions) == 2
        assert word_positions[0] == pytest.approx([0, 0], abs=0.01)
        assert word_positions[1] == pytest.approx([26.43, 0], abs=0.01)


def test_missing_word_regions_split_line_at_recognized_spaces(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["one  longer"],
        polygons=[[[20, 20], [220, 40], [216, 80], [16, 60]]],
        boxes=[[16, 20, 220, 80]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    line = page.children[0]
    assert plain_text == "one  longer"
    assert line.text == "one  longer"
    assert [word.text for word in line.children] == ["one", "longer"]
    _assert_polygon(
        line.children[0].poly,
        [
            (20, 20),
            (20 + 200 * 3 / 11, 20 + 20 * 3 / 11),
            (16 + 200 * 3 / 11, 60 + 20 * 3 / 11),
            (16, 60),
        ],
    )
    _assert_polygon(
        line.children[1].poly,
        [
            (20 + 200 * 5 / 11, 20 + 20 * 5 / 11),
            (220, 40),
            (216, 80),
            (16 + 200 * 5 / 11, 60 + 20 * 5 / 11),
        ],
    )


@pytest.mark.parametrize(
    ("word_texts", "word_regions"),
    [
        (["Hello"], [[10, 20, 130, 40]]),
        (["Hello", "world"], [[10, 20, 130, 40]]),
        (["Goodbye", "world"], [[10, 20, 60, 40], [65, 20, 130, 40]]),
    ],
)
def test_word_mismatch_uses_proportional_fallback_words(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    word_texts: list[str],
    word_regions: list[list[int]],
) -> None:
    result = _result(
        text_word=[word_texts],
        text_word_boxes=[word_regions],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    line = page.children[0]
    assert [word.text for word in line.children] == ["Hello", "world"]
    assert line.children[0].bbox.right == pytest.approx(10 + 120 * 5 / 11)
    assert line.children[1].bbox.left == pytest.approx(10 + 120 * 6 / 11)
    assert all(word.bbox != line.bbox for word in line.children)


def test_invalid_word_region_preserves_other_recognition_boundaries(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        text_word=[["Hello", "world"]],
        text_word_boxes=[[[10, 20, 10, 40], [80, 20, 130, 40]]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    words = page.children[0].children
    assert [word.text for word in words] == ["Hello", "world"]
    assert words[0].bbox.right == pytest.approx(10 + 120 * 5 / 11)
    assert words[1].bbox == BoundingBox(left=80, top=20, right=130, bottom=40)


def test_invalid_geometry_is_discarded_and_valid_geometry_is_clipped(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        polygons=[
            [[0, 0], [math.nan, 0], [20, 20], [0, 20]],
            [[10, 10], [10, 10], [10, 20], [10, 20]],
            [[0, 0], [10, 10], [20, 20], [30, 30]],
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[350, 180], [450, 180], [450, 250], [350, 250]],
        ],
        boxes=[
            [0, 0, 20, 20],
            [10, 10, 10, 20],
            [0, 0, 30, 30],
            [0, 0, 10, 10],
            [350, 180, 450, 250],
        ],
        texts=["not finite", "degenerate", "collinear", "five points", "kept"],
        scores=[0.5, 0.5, 0.5, 0.5, float("inf")],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    assert plain_text == "kept"
    assert [line.text for line in page.children] == ["kept"]
    assert page.children[0].bbox == BoundingBox(
        left=350,
        top=180,
        right=400,
        bottom=200,
    )
    assert page.children[0].confidence is None


def test_empty_result_produces_empty_page(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    with patch.object(
        ocr_paddle,
        "_predict",
        return_value=_result(texts=[], scores=[], polygons=[], boxes=[]),
    ):
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    assert page.children == []
    assert plain_text == ""


@pytest.mark.parametrize("text", [None, b"line", 1])
def test_non_string_recognition_text_fails_closed(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    text: object,
) -> None:
    result = _result()
    result["rec_texts"] = [text]

    with (
        patch.object(ocr_paddle, "_predict", return_value=result),
        pytest.raises(OCRError, match=r"rec_texts\[0\].*string"),
    ):
        ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )


@pytest.mark.parametrize(
    "result",
    [
        {
            "rec_texts": ["line"],
            "rec_scores": [],
            "rec_polys": [],
            "rec_boxes": [],
        },
        {
            "rec_texts": ["line"],
            "rec_scores": [0.9],
            "rec_polys": [[[0, 0], [30, 0], [30, 10], [0, 10]]],
        },
    ],
)
def test_inconsistent_top_level_arrays_fail_closed(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    result: dict[str, object],
) -> None:
    with (
        patch.object(ocr_paddle, "_predict", return_value=result),
        pytest.raises(OCRError, match="missing|inconsistent"),
    ):
        ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )


def test_inconsistent_optional_word_arrays_use_proportional_fallback_words(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        text_word=[],
        text_word_boxes=[[[10, 20, 130, 40]]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    line = page.children[0]
    assert [word.text for word in line.children] == ["Hello", "world"]
    assert all(word.bbox != line.bbox for word in line.children)


@pytest.mark.parametrize(
    ("word_texts", "word_regions"),
    [
        ("not an array", [[[10, 20, 130, 40]]]),
        ([["Hello", "world"]], "not an array"),
    ],
)
def test_malformed_optional_word_arrays_use_proportional_fallback_words(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    word_texts: object,
    word_regions: object,
) -> None:
    result = _result(
        text_word=word_texts,
        text_word_boxes=word_regions,
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    line = page.children[0]
    assert [word.text for word in line.children] == ["Hello", "world"]
    assert all(word.bbox != line.bbox for word in line.children)


@pytest.mark.parametrize(
    ("languages", "metadata"),
    [(["en"], "en"), (["de"], "de"), (["de", "en"], "de+en")],
)
def test_language_metadata_uses_pp_ocr_codes(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    languages: list[str],
    metadata: str,
) -> None:
    assert validate_ocr_languages(languages) == languages
    with patch.object(
        ocr_paddle,
        "_predict",
        return_value=_result(texts=[], scores=[], polygons=[], boxes=[]),
    ):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs, languages),
        )
    assert page.language == metadata


@pytest.mark.parametrize("language", ["eng", "deu", "unknown"])
def test_legacy_and_unknown_languages_are_rejected(language: str) -> None:
    with pytest.raises(ValueError, match="Unsupported PaddleOCR language"):
        validate_ocr_languages([language])


def _rotated_polygon(angle: float, top: float) -> list[list[float]]:
    width = 120.0
    height = 20.0
    radians = math.radians(angle)
    dx = width * math.cos(radians)
    dy = width * math.sin(radians)
    return [
        [30.0, top],
        [30.0 + dx, top + dy],
        [30.0 + dx, top + dy + height],
        [30.0, top + height],
    ]


def _ocr_page_context(
    ocr_image: Path,
    options: SimpleNamespace,
    *,
    pageno: int = 0,
    textareas: tuple[tuple[float, float, float, float], ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        pageinfo=SimpleNamespace(
            dpi=SimpleNamespace(to_scalar=lambda: 300.0),
            get_textareas=lambda **_kwargs: iter(textareas),
        ),
        options=options,
        pageno=pageno,
        get_path=lambda name: ocr_image if name == "ocr.png" else None,
    )


def test_zero_deskew_reuses_prediction_once_for_exact_ocr_image(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
) -> None:
    work_folder = tmp_path / "ocrmypdf.io.zero"
    work_folder.mkdir()
    page_image = work_folder / "000001_rasterize.png"
    ocr_image = work_folder / "000001_ocr.png"
    Image.new("RGB", (400, 200), "white").save(page_image, dpi=(300, 300))
    Image.new("RGB", (400, 200), "white").save(ocr_image, dpi=(300, 300))
    result = _result(
        text_word=[["Hello", "world"]],
        text_word_region=[
            [
                [[10, 20], [65, 20], [65, 40], [10, 40]],
                [[70, 20], [130, 20], [130, 40], [70, 40]],
            ]
        ],
    )
    model = MagicMock()
    model.predict.return_value = [result]
    options = _options(model_dirs, deskew=True)
    page_context = _ocr_page_context(ocr_image, options)

    with patch.object(ocr_paddle, "_get_model", return_value=model):
        correction = ocr_paddle.PaddleOcrEngine.get_deskew(page_image, options)
        ocr_paddle.filter_ocr_image(
            page_context,
            Image.new("RGB", (400, 200), "white"),
        )
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            ocr_image,
            options,
        )

    assert correction == 0.0
    assert model.predict.call_count == 1
    assert plain_text == "Hello world"
    assert page.children[0].bbox == BoundingBox(
        left=10,
        top=20,
        right=130,
        bottom=40,
    )
    assert [word.bbox for word in page.children[0].children] == [
        BoundingBox(left=10, top=20, right=65, bottom=40),
        BoundingBox(left=70, top=20, right=130, bottom=40),
    ]
    assert ocr_paddle._pending_deskew_results == {}
    assert ocr_paddle._prediction_result_by_image == {}


def test_zero_deskew_repredicts_when_ocrmypdf_masks_existing_text(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
) -> None:
    work_folder = tmp_path / "ocrmypdf.io.masked"
    work_folder.mkdir()
    page_image = work_folder / "000001_rasterize.png"
    ocr_image = work_folder / "000001_ocr.png"
    Image.new("RGB", (400, 200), "white").save(page_image, dpi=(300, 300))
    Image.new("RGB", (400, 200), "white").save(ocr_image, dpi=(300, 300))
    model = MagicMock()
    model.predict.side_effect = [
        [_result(texts=["Deskew raster"])],
        [_result(texts=["Masked OCR raster"])],
    ]
    options = _options(model_dirs, deskew=True)
    page_context = _ocr_page_context(
        ocr_image,
        options,
        textareas=((0.0, 0.0, 10.0, 10.0),),
    )

    with patch.object(ocr_paddle, "_get_model", return_value=model):
        assert ocr_paddle.PaddleOcrEngine.get_deskew(page_image, options) == 0.0
        ocr_paddle.filter_ocr_image(
            page_context,
            Image.new("RGB", (400, 200), "white"),
        )
        _page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            ocr_image,
            options,
        )

    assert plain_text == "Masked OCR raster"
    assert model.predict.call_count == 2
    assert ocr_paddle._pending_deskew_results == {}
    assert ocr_paddle._prediction_result_by_image == {}


def test_filter_skips_textarea_check_without_cached_deskew(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
) -> None:
    ocr_image = tmp_path / "000001_ocr.png"
    get_textareas = MagicMock(side_effect=AssertionError("unexpected text scan"))
    page_context = _ocr_page_context(ocr_image, _options(model_dirs))
    page_context.pageinfo.get_textareas = get_textareas

    ocr_paddle.filter_ocr_image(
        page_context,
        Image.new("RGB", (400, 200), "white"),
    )

    get_textareas.assert_not_called()


def test_zero_deskew_cache_is_scoped_to_run_and_page(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
) -> None:
    runs = []
    results_by_folder = {}
    for name, text in (("first", "First page"), ("second", "Second page")):
        work_folder = tmp_path / f"ocrmypdf.io.{name}"
        work_folder.mkdir()
        raster_image = work_folder / "000001_rasterize.png"
        ocr_image = work_folder / "000001_ocr.png"
        Image.new("RGB", (400, 200), "white").save(
            raster_image,
            dpi=(300, 300),
        )
        Image.new("RGB", (400, 200), "white").save(ocr_image, dpi=(300, 300))
        options = _options(model_dirs, deskew=True)
        runs.append((raster_image, ocr_image, options))
        results_by_folder[work_folder.resolve()] = _result(texts=[text])

    def predict(input_file: str, **_kwargs: object) -> list[dict[str, object]]:
        return [results_by_folder[Path(input_file).resolve().parent]]

    after_deskew = threading.Barrier(2)
    after_filter = threading.Barrier(2)
    model = MagicMock()
    model.predict.side_effect = predict

    def process_page(
        run: tuple[Path, Path, SimpleNamespace],
    ) -> str:
        raster_image, ocr_image, options = run
        assert ocr_paddle.PaddleOcrEngine.get_deskew(raster_image, options) == 0.0
        after_deskew.wait(timeout=5)
        ocr_paddle.filter_ocr_image(
            _ocr_page_context(ocr_image, options),
            Image.new("RGB", (400, 200), "white"),
        )
        after_filter.wait(timeout=5)
        _page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            ocr_image,
            options,
        )
        return plain_text

    with (
        patch.object(ocr_paddle, "_get_model", return_value=model),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        plain_texts = list(executor.map(process_page, runs))

    assert plain_texts == ["First page", "Second page"]
    assert model.predict.call_count == 2
    assert ocr_paddle._pending_deskew_results == {}
    assert ocr_paddle._prediction_result_by_image == {}


def test_aborted_zero_deskew_does_not_leak_into_non_deskew_run(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
) -> None:
    work_folder = tmp_path / "ocrmypdf.io.reused"
    work_folder.mkdir()
    raster_image = work_folder / "000001_rasterize.png"
    ocr_image = work_folder / "000001_ocr.png"
    Image.new("RGB", (400, 200), "white").save(raster_image, dpi=(300, 300))
    Image.new("RGB", (400, 200), "white").save(ocr_image, dpi=(300, 300))
    deskew_options = _options(model_dirs, deskew=True)
    non_deskew_options = _options(model_dirs)
    model = MagicMock()
    model.predict.side_effect = [
        [_result(texts=["Aborted page"])],
        [_result(texts=["Current page"])],
    ]

    with patch.object(ocr_paddle, "_get_model", return_value=model):
        assert (
            ocr_paddle.PaddleOcrEngine.get_deskew(
                raster_image,
                deskew_options,
            )
            == 0.0
        )
        ocr_paddle.filter_ocr_image(
            _ocr_page_context(ocr_image, non_deskew_options),
            Image.new("RGB", (400, 200), "white"),
        )
        _page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            ocr_image,
            non_deskew_options,
        )

    assert plain_text == "Current page"
    assert model.predict.call_count == 2
    assert ocr_paddle._pending_deskew_results == {}
    assert ocr_paddle._prediction_result_by_image == {}


def test_nonzero_deskew_runs_ocr_prediction_again(
    tmp_path: Path,
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    ocr_image = tmp_path / "ocr.png"
    Image.new("RGB", (400, 200), "white").save(ocr_image, dpi=(300, 300))
    polygons = [_rotated_polygon(3.0, 40), _rotated_polygon(3.0, 100)]
    result = _result(
        texts=["one", "two"],
        scores=[0.9, 0.9],
        polygons=polygons,
        boxes=[[30, 40, 160, 70], [30, 100, 160, 130]],
    )
    model = MagicMock()
    model.predict.return_value = [result]
    page_context = SimpleNamespace(
        pageinfo=SimpleNamespace(
            dpi=SimpleNamespace(to_scalar=lambda: 300.0),
        ),
        get_path=lambda name: ocr_image if name == "ocr.png" else None,
    )
    options = _options(model_dirs)

    with patch.object(ocr_paddle, "_get_model", return_value=model):
        correction = ocr_paddle.PaddleOcrEngine.get_deskew(page_image, options)
        ocr_paddle.filter_ocr_image(
            page_context,
            Image.new("RGB", (400, 200), "white"),
        )
        page, plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            ocr_image,
            options,
        )

    assert correction == pytest.approx(3.0, abs=0.01)
    assert model.predict.call_count == 2
    assert plain_text == "one\ntwo"
    assert [line.poly for line in page.children] == [
        [tuple(point) for point in polygon] for polygon in polygons
    ]
    assert ocr_paddle._prediction_result_by_image == {}


@pytest.mark.parametrize("angle", [-3.0, 3.0])
def test_deskew_preserves_pillow_correction_sign(
    model_dirs: tuple[Path, Path],
    page_image: Path,
    angle: float,
) -> None:
    polygons = [_rotated_polygon(angle, 40), _rotated_polygon(angle, 100)]
    result = _result(
        texts=["one", "two"],
        scores=[0.9, 0.9],
        polygons=polygons,
        boxes=[[30, 40, 160, 70], [30, 100, 160, 130]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        correction = ocr_paddle.PaddleOcrEngine.get_deskew(
            page_image,
            _options(model_dirs),
        )

    assert correction == pytest.approx(angle, abs=0.01)


def test_deskew_includes_straight_lines_in_median(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    angles = [0.0, 0.0, 0.0, 3.0, 3.0]
    result = _result(
        texts=["one", "two", "three", "four", "five"],
        scores=[0.9] * len(angles),
        polygons=[
            _rotated_polygon(angle, 20 + index * 30)
            for index, angle in enumerate(angles)
        ],
        boxes=[[30, 20 + index * 30, 160, 40 + index * 30] for index in range(5)],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        correction = ocr_paddle.PaddleOcrEngine.get_deskew(
            page_image,
            _options(model_dirs),
        )

    assert correction == 0.0


def test_deskew_ignores_tiny_median_correction(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["one", "two"],
        scores=[0.9, 0.9],
        polygons=[_rotated_polygon(0.04, 40), _rotated_polygon(0.04, 100)],
        boxes=[[30, 40, 160, 70], [30, 100, 160, 130]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        correction = ocr_paddle.PaddleOcrEngine.get_deskew(
            page_image,
            _options(model_dirs),
        )

    assert correction == 0.0


def test_deskew_requires_two_long_lines_within_ten_degrees(
    model_dirs: tuple[Path, Path],
    page_image: Path,
) -> None:
    result = _result(
        texts=["usable", "too steep", "too short"],
        scores=[0.9, 0.9, 0.9],
        polygons=[
            _rotated_polygon(3, 40),
            _rotated_polygon(11, 90),
            [[20, 140], [25, 140], [25, 150], [20, 150]],
        ],
        boxes=[[30, 40, 160, 70], [30, 90, 160, 120], [20, 140, 25, 150]],
    )

    with patch.object(ocr_paddle, "_predict", return_value=result):
        correction = ocr_paddle.PaddleOcrEngine.get_deskew(
            page_image,
            _options(model_dirs),
        )

    assert correction == 0.0


def test_plugin_blocks_tesseract_and_registers_compatibility_only() -> None:
    plugin_manager = MagicMock()

    ocr_paddle.initialize(plugin_manager)
    options = ocr_paddle.register_options()

    plugin_manager.set_blocked.assert_called_once_with(
        "ocrmypdf.builtin_plugins.tesseract_ocr"
    )
    assert set(options) == {"paddle", "tesseract"}
    compatibility = options["tesseract"]()
    assert compatibility.pagesegmode is None
    assert compatibility.downsample_above == 32767
    assert compatibility.downsample_large_images is True
    assert not any(
        callable(getattr(compatibility, name, None))
        for name in ("version", "languages", "generate_ocr", "generate_hocr")
    )


def test_initialized_plugin_manager_has_no_tesseract_hooks() -> None:
    from ocrmypdf._plugin_manager import get_plugin_manager

    with patch(
        "subprocess.Popen",
        side_effect=AssertionError("OCR plugin initialization started a process"),
    ):
        manager = get_plugin_manager(
            ["pdftopdfa.ocr_paddle", "pdftopdfa.ocr_rotation_fix"]
        )
        plugin_name = "ocrmypdf.builtin_plugins.tesseract_ocr"
        blocked_plugin = manager.pluggy_manager.get_plugin(plugin_name)
        assert blocked_plugin is not None

        manager.initialize(plugin_manager=manager.pluggy_manager)

    assert manager.pluggy_manager.is_blocked(plugin_name)
    assert manager.pluggy_manager.get_plugin(plugin_name) is None
    assert manager.pluggy_manager.get_hookcallers(blocked_plugin) is None


def test_plugin_selects_only_paddle_engine() -> None:
    assert ocr_paddle.get_ocr_engine(SimpleNamespace(ocr_engine="tesseract")) is None
    assert isinstance(
        ocr_paddle.get_ocr_engine(SimpleNamespace(ocr_engine="paddle")),
        ocr_paddle.PaddleOcrEngine,
    )
    assert isinstance(
        ocr_paddle.get_ocr_engine(None),
        ocr_paddle.PaddleOcrEngine,
    )
