# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the offline PaddleOCR engine."""

from __future__ import annotations

import hashlib
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

import pytest
from ocrmypdf.helpers import Resolution
from ocrmypdf.hocrtransform import BoundingBox, OcrClass
from pikepdf import Dictionary, Name, Pdf
from PIL import Image

import pdftopdfa.ocr_paddle as ocr_paddle
from pdftopdfa.exceptions import OCRError
from pdftopdfa.ocr import apply_ocr, validate_ocr_languages


@pytest.fixture(autouse=True)
def _isolated_model_cache():
    """Keep the process-wide Paddle session isolated between tests."""
    ocr_paddle._reset_model_cache_for_tests()
    yield
    ocr_paddle._reset_model_cache_for_tests()


def _artifact(data: bytes) -> ocr_paddle._Artifact:
    return ocr_paddle._Artifact(
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


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
        specs[kind] = ocr_paddle._ModelSpec(
            name=f"test_{kind}",
            artifacts={
                filename: _artifact(data) for filename, data in artifacts.items()
            },
        )
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
) -> SimpleNamespace:
    detection_dir, recognition_dir = model_dirs
    return SimpleNamespace(
        paddle=SimpleNamespace(
            detection_model_dir=detection_dir,
            recognition_model_dir=recognition_dir,
        ),
        languages=languages or ["en"],
        ocr_engine="paddle",
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


def test_pinned_model_contract_matches_plan() -> None:
    """The accepted external artifacts are immutable and reviewable."""
    assert ocr_paddle._DETECTION_MODEL.name == "PP-OCRv6_medium_det"
    assert ocr_paddle._DETECTION_MODEL.artifacts == {
        "inference.onnx": ocr_paddle._Artifact(
            size=62_032_837,
            sha256=("eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1"),
        ),
        "inference.yml": ocr_paddle._Artifact(
            size=886,
            sha256=("7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871"),
        ),
    }
    assert ocr_paddle._RECOGNITION_MODEL.name == "PP-OCRv6_medium_rec"
    assert ocr_paddle._RECOGNITION_MODEL.artifacts == {
        "inference.onnx": ocr_paddle._Artifact(
            size=76_554_979,
            sha256=("9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba"),
        ),
        "inference.yml": ocr_paddle._Artifact(
            size=150_580,
            sha256=("991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129"),
        ),
    }


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


def test_model_validation_hashes_until_session_is_initialized(
    model_dirs: tuple[Path, Path],
) -> None:
    expected = tuple(path.resolve() for path in model_dirs)

    assert ocr_paddle.validate_model_directories(*model_dirs) == expected
    with patch.object(ocr_paddle, "_sha256", wraps=ocr_paddle._sha256) as sha256:
        assert ocr_paddle.validate_model_directories(*model_dirs) == expected
    assert sha256.call_count == 4

    ocr_paddle._cached_model = object()

    with patch.object(
        ocr_paddle,
        "_sha256",
        side_effect=AssertionError("unchanged artifacts must use the validation cache"),
    ):
        assert ocr_paddle.validate_model_directories(*model_dirs) == expected


def test_model_validation_rehashes_same_size_tampering_before_initialization(
    model_dirs: tuple[Path, Path],
) -> None:
    detection_dir, recognition_dir = model_dirs
    assert ocr_paddle.validate_model_directories(*model_dirs)

    artifact = detection_dir / "inference.onnx"
    original_stat = artifact.stat()
    artifact.write_bytes(b"x" * original_stat.st_size)
    os.utime(
        artifact,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    with pytest.raises(OCRError, match="failed SHA-256 verification"):
        ocr_paddle.validate_model_directories(detection_dir, recognition_dir)


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


def test_model_validation_rejects_wrong_size(
    model_dirs: tuple[Path, Path],
) -> None:
    detection_dir, recognition_dir = model_dirs
    (detection_dir / "inference.onnx").write_bytes(b"wrong size")

    with pytest.raises(OCRError, match=r"has size \d+, expected \d+"):
        ocr_paddle.validate_model_directories(detection_dir, recognition_dir)


def test_model_validation_rejects_wrong_hash(
    model_dirs: tuple[Path, Path],
) -> None:
    detection_dir, recognition_dir = model_dirs
    artifact = recognition_dir / "inference.yml"
    artifact.write_bytes(b"x" * artifact.stat().st_size)

    with pytest.raises(OCRError, match="failed SHA-256 verification"):
        ocr_paddle.validate_model_directories(detection_dir, recognition_dir)


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


def test_model_session_is_lazy_reused_and_replaced_after_artifact_change(
    model_dirs: tuple[Path, Path],
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
        assert ocr_paddle._get_model(options) is first_model

        artifact = model_dirs[0] / "inference.yml"
        stat = artifact.stat()
        os.utime(
            artifact,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )

        assert ocr_paddle._get_model(options) is second_model

    assert create_model.call_count == 2
    first_model.close.assert_called_once_with()


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
        text_word=[["AI", " ", "systems", ","]],
        text_word_boxes=[
            [
                [10, 20, 30, 40],
                [31, 20, 35, 40],
                [36, 20, 100, 40],
                [101, 20, 105, 40],
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
        left=36,
        top=20,
        right=105,
        bottom=40,
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

    with patch.object(ocr_paddle, "_predict", return_value=_result()):
        page, _plain_text = ocr_paddle.PaddleOcrEngine.generate_ocr(
            page_image,
            _options(model_dirs),
        )

    assert page.dpi == 150.0
    assert page.bbox == BoundingBox(left=0, top=0, right=200, bottom=100)
    assert page.children[0].bbox == BoundingBox(
        left=5,
        top=10,
        right=65,
        bottom=20,
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


@pytest.mark.parametrize(
    ("word_texts", "word_regions"),
    [
        (["Hello"], [[10, 20, 130, 40]]),
        (["Hello", "world"], [[10, 20, 130, 40]]),
        (["Goodbye", "world"], [[10, 20, 60, 40], [65, 20, 130, 40]]),
        (["Hello", "world"], [[10, 20, 10, 40], [65, 20, 130, 40]]),
    ],
)
def test_word_mismatch_falls_back_to_full_line(
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
    assert len(line.children) == 1
    assert line.children[0].text == "Hello world"
    assert line.children[0].bbox == line.bbox


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


def test_inconsistent_optional_word_arrays_fall_back_to_full_line(
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
    assert len(line.children) == 1
    assert line.children[0].text == line.text
    assert line.children[0].bbox == line.bbox


@pytest.mark.parametrize(
    ("word_texts", "word_regions"),
    [
        ("not an array", [[[10, 20, 130, 40]]]),
        ([["Hello", "world"]], "not an array"),
    ],
)
def test_malformed_optional_word_arrays_fall_back_to_full_line(
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
    assert [word.text for word in line.children] == [line.text]


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
