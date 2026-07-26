# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PaddleOCR engine plugin for OCRmyPDF."""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
import threading
from argparse import SUPPRESS
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ocrmypdf
from ocrmypdf.helpers import Resolution
from ocrmypdf.hocrtransform import BoundingBox, OcrClass, OcrElement
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from PIL import Image
from pydantic import BaseModel

from .exceptions import OCRError
from .ocr import PADDLE_OCR_LANGUAGES

if TYPE_CHECKING:
    import pluggy
    from ocrmypdf._jobcontext import PageContext
    from ocrmypdf._options import OcrOptions

logger = logging.getLogger(__name__)

_TESSERACT_PLUGIN = "ocrmypdf.builtin_plugins.tesseract_ocr"
_MODEL_FILENAMES = frozenset({"inference.onnx", "inference.yml"})
_MIN_DESKEW_ANGLE = 0.05
_MAX_DESKEW_ANGLE = 10.0
_TEXT_DETECTION_LIMIT_SIDE_LEN = 1600


@dataclass(frozen=True)
class _Artifact:
    size: int
    sha256: str


@dataclass(frozen=True)
class _ModelSpec:
    name: str
    artifacts: dict[str, _Artifact]


_DETECTION_MODEL = _ModelSpec(
    name="PP-OCRv6_medium_det",
    artifacts={
        "inference.onnx": _Artifact(
            size=62_032_837,
            sha256="eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
        ),
        "inference.yml": _Artifact(
            size=886,
            sha256="7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871",
        ),
    },
)
_RECOGNITION_MODEL = _ModelSpec(
    name="PP-OCRv6_medium_rec",
    artifacts={
        "inference.onnx": _Artifact(
            size=76_554_979,
            sha256="9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
        ),
        "inference.yml": _Artifact(
            size=150_580,
            sha256="991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
        ),
    },
)


class _PaddleOptions(BaseModel):
    """Private OCRmyPDF options used to carry model directories to workers."""

    detection_model_dir: Path | None = None
    recognition_model_dir: Path | None = None


class _TesseractCompatibilityOptions(BaseModel):
    """Fields OCRmyPDF 17.8.1 reads even when Tesseract is blocked."""

    pagesegmode: int | None = None
    downsample_above: int = 32767
    downsample_large_images: bool = True


_model_lock = threading.RLock()
_prediction_lock = threading.RLock()
_cached_pair: tuple[Path, Path] | None = None
_cached_signature: tuple[tuple[int, int, int], ...] | None = None
_cached_model: Any | None = None
_coordinate_dpi_lock = threading.Lock()
_coordinate_dpi_by_image: dict[Path, float] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_signature(
    model_dir: Path,
    spec: _ModelSpec,
    *,
    verify_hashes: bool,
) -> tuple[tuple[int, int, int], ...]:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise OCRError(f"{spec.name} model directory does not exist: {model_dir}")

    try:
        entries = {entry.name for entry in model_dir.iterdir()}
    except OSError as exc:
        raise OCRError(f"Could not inspect {spec.name} model directory: {exc}") from exc

    if entries != _MODEL_FILENAMES:
        missing = sorted(_MODEL_FILENAMES - entries)
        unexpected = sorted(entries - _MODEL_FILENAMES)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise OCRError(
            f"{spec.name} model directory must contain exactly inference.onnx "
            f"and inference.yml ({'; '.join(details)})"
        )

    signature: list[tuple[int, int, int]] = []
    for filename in sorted(_MODEL_FILENAMES):
        path = model_dir / filename
        if path.is_symlink() or not path.is_file():
            raise OCRError(f"{spec.name} model artifact is not a regular file: {path}")
        try:
            stat = path.stat()
        except OSError as exc:
            raise OCRError(
                f"Could not inspect {spec.name} model artifact: {exc}"
            ) from exc

        expected = spec.artifacts[filename]
        if stat.st_size != expected.size:
            raise OCRError(
                f"{spec.name} {filename} has size {stat.st_size}, "
                f"expected {expected.size}"
            )
        if verify_hashes:
            try:
                actual_hash = _sha256(path)
            except OSError as exc:
                raise OCRError(
                    f"Could not read {spec.name} model artifact: {exc}"
                ) from exc
            if actual_hash != expected.sha256:
                raise OCRError(f"{spec.name} {filename} failed SHA-256 verification")
        signature.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(signature)


def _resolve_and_validate_model_directories(
    detection_model_dir: Path,
    recognition_model_dir: Path,
) -> tuple[Path, Path]:
    try:
        unresolved_pair = (
            Path(detection_model_dir).expanduser(),
            Path(recognition_model_dir).expanduser(),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise OCRError(f"Invalid PaddleOCR model directory: {exc}") from exc

    if any(model_dir.is_symlink() for model_dir in unresolved_pair):
        raise OCRError("PaddleOCR model directories must not be symbolic links")

    try:
        pair = tuple(model_dir.resolve() for model_dir in unresolved_pair)
    except (OSError, RuntimeError) as exc:
        raise OCRError(f"Invalid PaddleOCR model directory: {exc}") from exc

    global _cached_signature
    with _model_lock:
        quick_signature = (
            *_model_signature(pair[0], _DETECTION_MODEL, verify_hashes=False),
            *_model_signature(pair[1], _RECOGNITION_MODEL, verify_hashes=False),
        )
        if (
            _cached_model is not None
            and pair == _cached_pair
            and quick_signature == _cached_signature
        ):
            return pair

        verified_signature = (
            *_model_signature(pair[0], _DETECTION_MODEL, verify_hashes=True),
            *_model_signature(pair[1], _RECOGNITION_MODEL, verify_hashes=True),
        )
        _replace_cached_model(pair, verified_signature)
        return pair


def validate_model_directories(
    detection_model_dir: Path | None,
    recognition_model_dir: Path | None,
) -> tuple[Path, Path]:
    """Validate the exact offline PP-OCRv6 Medium model artifacts."""
    if detection_model_dir is None or recognition_model_dir is None:
        raise OCRError(
            "Both PaddleOCR detection and recognition model directories are required"
        )
    with _prediction_lock:
        return _resolve_and_validate_model_directories(
            detection_model_dir,
            recognition_model_dir,
        )


def _replace_cached_model(
    pair: tuple[Path, Path],
    signature: tuple[tuple[int, int, int], ...],
) -> None:
    global _cached_model, _cached_pair, _cached_signature

    if _cached_model is not None:
        close = getattr(_cached_model, "close", None)
        if callable(close):
            close()
    _cached_model = None
    _cached_pair = pair
    _cached_signature = signature


def _create_model(detection_model_dir: Path, recognition_model_dir: Path) -> Any:
    try:
        from paddleocr import PaddleOCR

        paddlex_logger = logging.getLogger("paddlex")
        paddlex_logger.setLevel(
            max(paddlex_logger.getEffectiveLevel(), logger.getEffectiveLevel())
        )
        return PaddleOCR(
            text_detection_model_name=_DETECTION_MODEL.name,
            text_detection_model_dir=str(detection_model_dir),
            text_recognition_model_name=_RECOGNITION_MODEL.name,
            text_recognition_model_dir=str(recognition_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=_TEXT_DETECTION_LIMIT_SIDE_LEN,
            text_det_limit_type="max",
            return_word_box=True,
            engine="onnxruntime",
            device="cpu",
            engine_config={"providers": ["CPUExecutionProvider"]},
        )
    except Exception as exc:
        raise OCRError(f"Could not initialize PaddleOCR: {exc}") from exc


def _get_model(options: OcrOptions) -> Any:
    paddle_options = options.paddle
    pair = validate_model_directories(
        paddle_options.detection_model_dir,
        paddle_options.recognition_model_dir,
    )

    global _cached_model
    with _model_lock:
        if _cached_model is None:
            _cached_model = _create_model(*pair)
        return _cached_model


def _predict(input_file: Path, options: OcrOptions) -> Any:
    try:
        with _prediction_lock:
            results = list(
                _get_model(options).predict(
                    str(input_file),
                    return_word_box=True,
                    text_det_limit_side_len=_TEXT_DETECTION_LIMIT_SIDE_LEN,
                    text_det_limit_type="max",
                )
            )
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(f"PaddleOCR inference failed for {input_file}: {exc}") from exc

    if len(results) != 1:
        raise OCRError(f"PaddleOCR returned {len(results)} results for one page image")
    result = results[0]
    try:
        if "rec_texts" not in result and "res" in result:
            result = result["res"]
    except (KeyError, TypeError):
        pass
    return result


def _field(result: Any, name: str) -> Any:
    try:
        return result[name]
    except (KeyError, TypeError) as exc:
        raise OCRError(f"PaddleOCR result is missing {name}") from exc


def _sequence_length(value: Any, name: str) -> int:
    if isinstance(value, str | bytes):
        raise OCRError(f"PaddleOCR result field {name} must be a sequence")
    try:
        return len(value)
    except TypeError as exc:
        raise OCRError(f"PaddleOCR result field {name} must be a sequence") from exc


def _parallel_line_data(result: Any) -> tuple[Any, Any, Any, Any]:
    values = tuple(
        _field(result, name)
        for name in ("rec_texts", "rec_scores", "rec_polys", "rec_boxes")
    )
    lengths = tuple(
        _sequence_length(value, name)
        for value, name in zip(
            values,
            ("rec_texts", "rec_scores", "rec_polys", "rec_boxes"),
            strict=True,
        )
    )
    if len(set(lengths)) != 1:
        raise OCRError(
            "PaddleOCR returned inconsistent line arrays: "
            f"rec_texts={lengths[0]}, rec_scores={lengths[1]}, "
            f"rec_polys={lengths[2]}, rec_boxes={lengths[3]}"
        )
    return values


def _optional_word_data(result: Any, line_count: int) -> tuple[Any, Any] | None:
    try:
        words = result["text_word"]
    except (KeyError, TypeError):
        return None

    regions = None
    for name in ("text_word_region", "text_word_boxes"):
        try:
            candidate = result[name]
        except (KeyError, TypeError):
            continue
        try:
            candidate_length = _sequence_length(candidate, name)
        except OCRError:
            return None
        if candidate_length != line_count:
            return None
        if regions is None:
            regions = candidate

    try:
        words_length = _sequence_length(words, "text_word")
    except OCRError:
        return None
    if words_length != line_count:
        return None
    if regions is None:
        return None
    return words, regions


def _clipped_polygon(
    value: Any,
    width: int,
    height: int,
    *,
    scale: float = 1.0,
) -> tuple[list[tuple[float, float]], BoundingBox] | None:
    try:
        if len(value) != 4:
            return None
        polygon = []
        for point in value:
            if len(point) != 2:
                return None
            x = float(point[0])
            y = float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            polygon.append(
                (
                    min(max(x, 0.0), float(width)) * scale,
                    min(max(y, 0.0), float(height)) * scale,
                )
            )
    except (TypeError, ValueError, IndexError):
        return None

    left = min(point[0] for point in polygon)
    top = min(point[1] for point in polygon)
    right = max(point[0] for point in polygon)
    bottom = max(point[1] for point in polygon)
    area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                polygon,
                [*polygon[1:], polygon[0]],
                strict=True,
            )
        )
    )
    if right <= left or bottom <= top or area <= 1e-6:
        return None
    return polygon, BoundingBox(left=left, top=top, right=right, bottom=bottom)


def _word_polygon(value: Any) -> Any:
    try:
        if len(value) != 4:
            return ()
        try:
            if all(len(point) == 2 for point in value):
                return value
        except TypeError:
            pass
        left, top, right, bottom = value
        return (
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        )
    except (TypeError, ValueError):
        return ()


def _line_angle(polygon: list[tuple[float, float]]) -> float:
    first, second = polygon[:2]
    image_angle = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
    return -image_angle


def _confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _fallback_word(
    text: str,
    polygon: list[tuple[float, float]],
    bbox: BoundingBox,
    confidence: float | None,
    language: str,
) -> OcrElement:
    return OcrElement(
        ocr_class=OcrClass.WORD,
        bbox=bbox,
        poly=polygon,
        text=text,
        confidence=confidence,
        direction="ltr",
        language=language,
    )


def _combined_word_geometry(
    geometries: list[tuple[list[tuple[float, float]], BoundingBox]],
) -> tuple[list[tuple[float, float]], BoundingBox]:
    if len(geometries) == 1:
        return geometries[0]

    left = min(geometry[1].left for geometry in geometries)
    top = min(geometry[1].top for geometry in geometries)
    right = max(geometry[1].right for geometry in geometries)
    bottom = max(geometry[1].bottom for geometry in geometries)
    bbox = BoundingBox(left=left, top=top, right=right, bottom=bottom)
    return (
        [(left, top), (right, top), (right, bottom), (left, bottom)],
        bbox,
    )


def _words_for_line(
    text: str,
    line_polygon: list[tuple[float, float]],
    line_bbox: BoundingBox,
    confidence: float | None,
    language: str,
    word_texts: Any,
    word_regions: Any,
    width: int,
    height: int,
    scale: float,
) -> list[OcrElement]:
    try:
        if len(word_texts) != len(word_regions) or not word_texts:
            raise ValueError
        tokens = [str(token) for token in word_texts]
        if any(not token for token in tokens):
            raise ValueError
        if "".join("".join(tokens).split()) != "".join(text.split()):
            raise ValueError

        normalized_text = " ".join(text.split())
        if all(not any(character.isspace() for character in token) for token in tokens):
            if " ".join(tokens) == normalized_text:
                words = []
                for token, region in zip(tokens, word_regions, strict=True):
                    geometry = _clipped_polygon(
                        _word_polygon(region),
                        width,
                        height,
                        scale=scale,
                    )
                    if geometry is None:
                        raise ValueError
                    polygon, bbox = geometry
                    words.append(
                        OcrElement(
                            ocr_class=OcrClass.WORD,
                            bbox=bbox,
                            poly=polygon,
                            text=token,
                            confidence=confidence,
                            direction="ltr",
                            language=language,
                        )
                    )
                return words

        grouped_words: list[
            tuple[str, list[tuple[list[tuple[float, float]], BoundingBox]]]
        ] = []
        current_text = ""
        current_geometries: list[tuple[list[tuple[float, float]], BoundingBox]] = []

        def finish_word() -> None:
            nonlocal current_text, current_geometries
            if current_text:
                grouped_words.append((current_text, current_geometries))
            current_text = ""
            current_geometries = []

        for token, region in zip(tokens, word_regions, strict=True):
            if token[:1].isspace():
                finish_word()

            content = token.strip()
            if content:
                geometry = _clipped_polygon(
                    _word_polygon(region),
                    width,
                    height,
                    scale=scale,
                )
                if geometry is None:
                    raise ValueError
                current_text += content
                current_geometries.append(geometry)

            if token[-1:].isspace():
                finish_word()
        finish_word()

        if not grouped_words:
            raise ValueError
        if " ".join(word[0] for word in grouped_words) != " ".join(text.split()):
            raise ValueError

        words = []
        for token, geometries in grouped_words:
            polygon, bbox = _combined_word_geometry(geometries)
            words.append(
                OcrElement(
                    ocr_class=OcrClass.WORD,
                    bbox=bbox,
                    poly=polygon,
                    text=token,
                    confidence=confidence,
                    direction="ltr",
                    language=language,
                )
            )
        return words
    except (TypeError, ValueError):
        return [
            _fallback_word(
                text,
                line_polygon,
                line_bbox,
                confidence,
                language,
            )
        ]


def _image_properties(input_file: Path) -> tuple[int, int, float]:
    try:
        with Image.open(input_file) as image:
            width, height = image.size
            dpi_value = image.info.get("dpi", (300.0, 300.0))
    except Exception as exc:
        raise OCRError(f"Could not inspect OCR page image {input_file}: {exc}") from exc

    if isinstance(dpi_value, tuple):
        dpi = float(dpi_value[0])
    else:
        dpi = float(dpi_value)
    if not math.isfinite(dpi) or dpi <= 0:
        dpi = 300.0
    else:
        dpi = float(round(dpi))
    return width, height, dpi


def _take_coordinate_scale(input_file: Path, raster_dpi: float) -> tuple[float, float]:
    key = input_file.resolve()
    with _coordinate_dpi_lock:
        coordinate_dpi = _coordinate_dpi_by_image.pop(key, raster_dpi)
    if not math.isfinite(coordinate_dpi) or coordinate_dpi <= 0:
        coordinate_dpi = raster_dpi
    return coordinate_dpi, coordinate_dpi / raster_dpi


def _language_metadata(options: OcrOptions) -> str:
    return "+".join(options.languages)


class PaddleOcrEngine(OcrEngine):
    """OCRmyPDF engine backed by offline PP-OCRv6 Medium ONNX models."""

    @staticmethod
    def version() -> str:
        return version("paddleocr")

    @staticmethod
    def creator_tag(options: OcrOptions) -> str:
        return f"PaddleOCR {PaddleOcrEngine.version()} (PP-OCRv6 Medium)"

    def __str__(self) -> str:
        return f"PaddleOCR {self.version()} (PP-OCRv6 Medium)"

    @staticmethod
    def languages(options: OcrOptions) -> set[str]:
        return set(PADDLE_OCR_LANGUAGES)

    @staticmethod
    def get_orientation(
        input_file: Path,
        options: OcrOptions,
    ) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options: OcrOptions) -> float:
        width, height, _dpi = _image_properties(input_file)
        _texts, _scores, polygons, _boxes = _parallel_line_data(
            _predict(input_file, options)
        )
        minimum_length = max(20.0, min(width, height) * 0.05)
        angles = []
        for value in polygons:
            geometry = _clipped_polygon(value, width, height)
            if geometry is None:
                continue
            polygon, _bbox = geometry
            first, second = polygon[:2]
            edge_length = math.hypot(
                second[0] - first[0],
                second[1] - first[1],
            )
            angle = -_line_angle(polygon)
            if (
                edge_length >= minimum_length
                and _MIN_DESKEW_ANGLE <= abs(angle) <= _MAX_DESKEW_ANGLE
            ):
                angles.append(angle)
        return statistics.median(angles) if len(angles) >= 2 else 0.0

    @staticmethod
    def supports_generate_ocr() -> bool:
        return True

    @staticmethod
    def generate_ocr(
        input_file: Path,
        options: OcrOptions,
        page_number: int = 0,
    ) -> tuple[OcrElement, str]:
        width, height, raster_dpi = _image_properties(input_file)
        coordinate_dpi, scale = _take_coordinate_scale(input_file, raster_dpi)
        result = _predict(input_file, options)
        texts, scores, polygons, _boxes = _parallel_line_data(result)
        word_data = _optional_word_data(result, len(texts))
        language = _language_metadata(options)

        lines = []
        plain_text = []
        for index in range(len(texts)):
            text = texts[index]
            if not isinstance(text, str):
                raise OCRError(
                    f"PaddleOCR result field rec_texts[{index}] must be a string"
                )
            geometry = _clipped_polygon(
                polygons[index],
                width,
                height,
                scale=scale,
            )
            if not text or geometry is None:
                continue

            plain_text.append(text)
            polygon, bbox = geometry
            confidence = _confidence(scores[index])
            if word_data is None:
                words = [
                    _fallback_word(
                        text,
                        polygon,
                        bbox,
                        confidence,
                        language,
                    )
                ]
            else:
                word_texts, word_regions = word_data
                words = _words_for_line(
                    text,
                    polygon,
                    bbox,
                    confidence,
                    language,
                    word_texts[index],
                    word_regions[index],
                    width,
                    height,
                    scale,
                )

            lines.append(
                OcrElement(
                    ocr_class=OcrClass.LINE,
                    bbox=bbox,
                    poly=polygon,
                    text=text,
                    confidence=confidence,
                    children=words,
                    direction="ltr",
                    language=language,
                    textangle=_line_angle(polygon),
                )
            )

        page = OcrElement(
            ocr_class=OcrClass.PAGE,
            bbox=BoundingBox(
                left=0,
                top=0,
                right=width * scale,
                bottom=height * scale,
            ),
            children=lines,
            direction="ltr",
            language=language,
            dpi=coordinate_dpi,
            page_number=page_number,
        )
        return page, "\n".join(plain_text)

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        raise NotImplementedError("PaddleOCR uses OCRmyPDF's direct OcrElement API")

    @staticmethod
    def generate_pdf(
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        raise NotImplementedError("PaddleOCR requires the fpdf2 renderer")


def _reset_model_cache_for_tests() -> None:
    """Clear and close cached PaddleOCR state for isolated tests."""
    global _cached_model, _cached_pair, _cached_signature
    with _prediction_lock, _model_lock:
        if _cached_model is not None:
            close = getattr(_cached_model, "close", None)
            if callable(close):
                close()
        _cached_model = None
        _cached_pair = None
        _cached_signature = None
    with _coordinate_dpi_lock:
        _coordinate_dpi_by_image.clear()


@ocrmypdf.hookimpl(tryfirst=True)
def initialize(plugin_manager: pluggy.PluginManager) -> None:
    """Block OCRmyPDF's built-in Tesseract plugin before dependency checks."""
    plugin_manager.set_blocked(_TESSERACT_PLUGIN)


@ocrmypdf.hookimpl
def register_options() -> dict[str, type[BaseModel]]:
    """Register private model paths and OCRmyPDF's compatibility namespace."""
    return {
        "paddle": _PaddleOptions,
        "tesseract": _TesseractCompatibilityOptions,
    }


@ocrmypdf.hookimpl
def add_options(parser: Any) -> None:
    """Add hidden API-only arguments that carry model paths to worker options."""
    parser.add_argument(
        "--paddle-detection-model-dir",
        type=Path,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--paddle-recognition-model-dir",
        type=Path,
        help=SUPPRESS,
    )


@ocrmypdf.hookimpl
def filter_ocr_image(page: PageContext, image: Image.Image) -> None:
    """Record OCRmyPDF's coordinate DPI without modifying the OCR raster."""
    coordinate_dpi = float(page.pageinfo.dpi.to_scalar())
    if not math.isfinite(coordinate_dpi) or coordinate_dpi <= 0:
        image_dpi = image.info.get("dpi", (600.0, 600.0))
        if isinstance(image_dpi, tuple):
            raster_dpi = float(image_dpi[0])
        else:
            raster_dpi = float(image_dpi)
        if not math.isfinite(raster_dpi) or raster_dpi <= 0:
            raster_dpi = 600.0
        coordinate_dpi = float(round(raster_dpi))
        # OCRmyPDF 17.8.1 reports zero DPI for vector-only pages and has no
        # public setter, but its fpdf2 grafter requires a positive PageInfo DPI.
        page.pageinfo._dpi = Resolution(coordinate_dpi, coordinate_dpi)

    key = page.get_path("ocr.png").resolve()
    with _coordinate_dpi_lock:
        _coordinate_dpi_by_image[key] = coordinate_dpi


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine(options: OcrOptions | None) -> PaddleOcrEngine | None:
    """Select PaddleOCR only when OCRmyPDF requests the paddle engine."""
    if options is not None and options.ocr_engine != "paddle":
        return None
    return PaddleOcrEngine()
