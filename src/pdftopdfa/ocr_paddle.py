# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PaddleOCR engine plugin for OCRmyPDF."""

from __future__ import annotations

import logging
import math
import re
import statistics
import threading
from argparse import SUPPRESS
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import ocrmypdf
from ocrmypdf.helpers import Resolution
from ocrmypdf.hocrtransform import Baseline, BoundingBox, OcrClass, OcrElement
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from PIL import Image
from pydantic import BaseModel

from ._ocr_runtime import _MODEL_FILENAMES as _MODEL_FILENAMES
from ._ocr_runtime import (
    _format_exception_message,
    _ModelSpec,
    _validate_model_directory,
    execution_provider_base,
    onnxruntime_engine_config,
    require_execution_provider,
    validate_ocr_execution_provider,
)
from .exceptions import OCRError
from .ocr import PADDLE_OCR_LANGUAGES

if TYPE_CHECKING:
    import pluggy
    from ocrmypdf._jobcontext import PageContext
    from ocrmypdf._options import OcrOptions

logger = logging.getLogger(__name__)

_TESSERACT_PLUGIN = "ocrmypdf.builtin_plugins.tesseract_ocr"
_MIN_DESKEW_ANGLE = 0.05
_MAX_DESKEW_ANGLE = 10.0
_TEXT_DETECTION_LIMIT_SIDE_LEN = 1600


@dataclass(frozen=True)
class _CachedPrediction:
    result: Any
    model_pair: tuple[Path, Path, str]
    image_size: tuple[int, int]


_DETECTION_MODEL = _ModelSpec(name="PP-OCRv6_medium_det")
_RECOGNITION_MODEL = _ModelSpec(name="PP-OCRv6_medium_rec")


@dataclass(frozen=True)
class _LayoutRegion:
    left: int
    top: int
    right: int
    bottom: int


class _PaddleOptions(BaseModel):
    """Private OCRmyPDF options used to carry model directories to workers."""

    detection_model_dir: Path | None = None
    recognition_model_dir: Path | None = None
    execution_provider: str = "cpu"
    layout: bool = False


class _TesseractCompatibilityOptions(BaseModel):
    """Fields OCRmyPDF 17.8.1 reads even when Tesseract is blocked."""

    pagesegmode: int | None = None
    downsample_above: int = 32767
    downsample_large_images: bool = True


_model_lock = threading.RLock()
_prediction_lock = threading.RLock()
_cached_pair: tuple[Path, Path] | None = None
_cached_fingerprint: tuple[tuple[int, int, int, int, int, int], ...] | None = None
_cached_model: Any | None = None
_cached_execution_provider: str | None = None
_pending_deskew_results: dict[tuple[Path, int], _CachedPrediction] = {}
_prediction_result_by_image: dict[Path, _CachedPrediction] = {}
_coordinate_dpi_lock = threading.Lock()
_coordinate_dpi_by_image: dict[Path, float] = {}


def _resolve_and_validate_model_directories(
    detection_model_dir: Path | None,
    recognition_model_dir: Path | None,
    *,
    recheck: bool,
) -> tuple[Path, Path]:
    if detection_model_dir is None or recognition_model_dir is None:
        raise OCRError(
            "Both PaddleOCR detection and recognition model directories are required"
        )
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

    with _model_lock:
        if not recheck and pair == _cached_pair and _cached_fingerprint is not None:
            return pair

        fingerprint = (
            *_validate_model_directory(pair[0], _DETECTION_MODEL),
            *_validate_model_directory(pair[1], _RECOGNITION_MODEL),
        )
        if pair != _cached_pair or fingerprint != _cached_fingerprint:
            _replace_cached_model(pair, fingerprint)
        return pair


def validate_model_directories(
    detection_model_dir: Path | None,
    recognition_model_dir: Path | None,
) -> tuple[Path, Path]:
    """Validate the offline PP-OCRv6 Medium model directory structure."""
    with _prediction_lock:
        return _resolve_and_validate_model_directories(
            detection_model_dir,
            recognition_model_dir,
            recheck=True,
        )


def _replace_cached_model(
    pair: tuple[Path, Path],
    fingerprint: tuple[tuple[int, int, int, int, int, int], ...],
) -> None:
    global _cached_execution_provider, _cached_fingerprint, _cached_model, _cached_pair

    if _cached_model is not None:
        close = getattr(_cached_model, "close", None)
        if callable(close):
            close()
    _cached_model = None
    _cached_execution_provider = None
    _cached_pair = pair
    _cached_fingerprint = fingerprint


def _create_model(
    detection_model_dir: Path,
    recognition_model_dir: Path,
    execution_provider: str = "cpu",
) -> Any:
    try:
        from paddleocr import PaddleOCR

        paddlex_logger = logging.getLogger("paddlex")
        paddlex_logger.setLevel(
            max(paddlex_logger.getEffectiveLevel(), logger.getEffectiveLevel())
        )
        model = PaddleOCR(
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
            engine_config=onnxruntime_engine_config(execution_provider),
        )
        if execution_provider_base(execution_provider) == "directml":
            try:
                pipeline = model.paddlex_pipeline
                require_execution_provider(
                    pipeline.text_det_model.runner.session,
                    execution_provider,
                )
                require_execution_provider(
                    pipeline.text_rec_model.runner.session,
                    execution_provider,
                )
            except Exception:
                close = getattr(model, "close", None)
                if callable(close):
                    close()
                raise
        return model
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(
            f"Could not initialize PaddleOCR: {_format_exception_message(exc)}"
        ) from exc


def _get_model(options: OcrOptions) -> Any:
    paddle_options = options.paddle
    execution_provider = validate_ocr_execution_provider(
        getattr(paddle_options, "execution_provider", "cpu")
    )
    global _cached_execution_provider, _cached_model
    with _prediction_lock:
        pair = _resolve_and_validate_model_directories(
            paddle_options.detection_model_dir,
            paddle_options.recognition_model_dir,
            recheck=False,
        )
        with _model_lock:
            if (
                _cached_model is not None
                and _cached_execution_provider != execution_provider
            ):
                close = getattr(_cached_model, "close", None)
                if callable(close):
                    close()
                _cached_model = None
            if _cached_model is None:
                _cached_model = _create_model(*pair, execution_provider)
                _cached_execution_provider = execution_provider
            return _cached_model


def _model_pair_for_cache(options: OcrOptions) -> tuple[Path, Path, str] | None:
    try:
        paddle_options = options.paddle
        if (
            paddle_options.detection_model_dir is None
            or paddle_options.recognition_model_dir is None
        ):
            return None
        pair = (
            Path(paddle_options.detection_model_dir).expanduser().resolve(),
            Path(paddle_options.recognition_model_dir).expanduser().resolve(),
        )
        execution_provider = validate_ocr_execution_provider(
            getattr(paddle_options, "execution_provider", "cpu")
        )
        return (*pair, execution_provider)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _page_cache_key(path: Path) -> tuple[Path, int] | None:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    page_prefix, separator, _name = resolved.name.partition("_")
    if not separator or not page_prefix.isascii() or not page_prefix.isdecimal():
        return None
    try:
        page_number = int(page_prefix)
    except ValueError:
        return None
    if page_number <= 0:
        return None
    return resolved.parent, page_number


def _discard_stale_prediction_results() -> None:
    stale_pages = [key for key in _pending_deskew_results if not key[0].is_dir()]
    for key in stale_pages:
        _pending_deskew_results.pop(key, None)

    stale_images = [
        path for path in _prediction_result_by_image if not path.parent.is_dir()
    ]
    for path in stale_images:
        _prediction_result_by_image.pop(path, None)


def _ocr_image_masks_text(page: PageContext) -> bool:
    """Return conservatively whether OCRmyPDF will blank existing text areas."""
    try:
        return any(
            page.pageinfo.get_textareas(
                visible=None,
                corrupt=None,
            )
        )
    except Exception:
        return True


@contextmanager
def _allowed_character_decoder(
    text_rec_model: Any,
    allowed_characters: str | None,
) -> Iterator[None]:
    if allowed_characters is None:
        yield
        return

    decoder = text_rec_model.post_op
    try:
        characters = decoder.character
        ignored_tokens = set(decoder.get_ignored_tokens())
    except AttributeError as exc:
        raise OCRError("PP-OCRv6 CTC decoder is unavailable") from exc

    allowed = set(allowed_characters)
    masked_classes = np.array(
        [
            index not in ignored_tokens and character not in allowed
            for index, character in enumerate(characters)
        ],
        dtype=bool,
    )

    def decode(prediction: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            logits = np.asarray(prediction[0])
        except (IndexError, TypeError) as exc:
            raise OCRError("PP-OCRv6 CTC decoder received invalid logits") from exc
        if logits.ndim == 0 or logits.shape[-1] != len(characters):
            raise OCRError("PP-OCRv6 CTC decoder received incompatible logits")

        masked_logits = np.array(logits, copy=True)
        masked_logits[..., masked_classes] = -np.inf
        prediction = list(prediction)
        prediction[0] = masked_logits
        return decoder(prediction, *args, **kwargs)

    text_rec_model.post_op = decode
    try:
        yield
    finally:
        text_rec_model.post_op = decoder


def _predict(
    input_file: Path,
    options: OcrOptions,
    *,
    allowed_characters: str | None = None,
) -> Any:
    try:
        with _prediction_lock:
            _discard_stale_prediction_results()
            cached_prediction = _prediction_result_by_image.pop(
                input_file.resolve(),
                None,
            )
            if (
                allowed_characters is None
                and cached_prediction is not None
                and cached_prediction.model_pair == _model_pair_for_cache(options)
            ):
                return cached_prediction.result
            model = _get_model(options)
            if allowed_characters is None:
                results = list(
                    model.predict(
                        str(input_file),
                        return_word_box=True,
                        text_det_limit_side_len=_TEXT_DETECTION_LIMIT_SIDE_LEN,
                        text_det_limit_type="max",
                    )
                )
            else:
                with _allowed_character_decoder(
                    model.paddlex_pipeline.text_rec_model,
                    allowed_characters,
                ):
                    results = list(
                        model.predict(
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


def _text_confidence_results(
    texts: Any,
    scores: Any,
) -> list[tuple[str, float]]:
    text_count = _sequence_length(texts, "rec_texts")
    score_count = _sequence_length(scores, "rec_scores")
    if text_count != score_count:
        raise OCRError(
            "PaddleOCR returned inconsistent recognition arrays: "
            f"rec_texts={text_count}, rec_scores={score_count}"
        )

    results = []
    for index, (text, score) in enumerate(zip(texts, scores, strict=True)):
        if not isinstance(text, str):
            raise OCRError(
                f"PaddleOCR result field rec_texts[{index}] must be a string"
            )
        confidence = _confidence(score)
        if confidence is None:
            raise OCRError(
                f"PaddleOCR result field rec_scores[{index}] is not a valid confidence"
            )
        results.append((text, confidence))
    return results


def recognize_image(
    input_path: str | Path,
    *,
    detection_model_dir: Path,
    recognition_model_dir: Path,
    ocr_execution_provider: str = "cpu",
    layout: str = "auto",
    allowed_characters: str | None = None,
) -> list[tuple[str, float]]:
    """Recognize text and confidence values in one image with PP-OCRv6."""
    if layout not in {"auto", "single_line"}:
        raise ValueError("layout must be 'auto' or 'single_line'")
    if allowed_characters is not None and not isinstance(allowed_characters, str):
        raise TypeError("allowed_characters must be a string or None")

    input_file = Path(input_path)
    options = SimpleNamespace(
        paddle=_PaddleOptions(
            detection_model_dir=detection_model_dir,
            recognition_model_dir=recognition_model_dir,
            execution_provider=ocr_execution_provider,
        )
    )

    if layout == "auto":
        result = _predict(
            input_file,
            options,
            allowed_characters=allowed_characters,
        )
        return _text_confidence_results(
            _field(result, "rec_texts"),
            _field(result, "rec_scores"),
        )

    try:
        with _prediction_lock:
            model = _get_model(options)
            text_rec_model = model.paddlex_pipeline.text_rec_model
            with _allowed_character_decoder(text_rec_model, allowed_characters):
                results = list(
                    text_rec_model(
                        str(input_file),
                        return_word_box=False,
                    )
                )
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(f"PaddleOCR inference failed for {input_file}: {exc}") from exc

    if len(results) != 1:
        raise OCRError(f"PaddleOCR returned {len(results)} results for one image")
    result = results[0]
    text = _field(result, "rec_text")
    if isinstance(text, tuple):
        text = text[0]
    return _text_confidence_results(
        [text],
        [_field(result, "rec_score")],
    )


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


def _polygon_geometry(
    value: Any,
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    clip: bool = True,
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
            if clip:
                x = min(max(x, 0.0), float(width))
                y = min(max(y, 0.0), float(height))
            polygon.append(
                (
                    x * scale,
                    y * scale,
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


def _word_regions_are_vertical(polygon: list[tuple[float, float]]) -> bool:
    width = max(point[0] for point in polygon) - min(point[0] for point in polygon)
    height = max(point[1] for point in polygon) - min(point[1] for point in polygon)
    return width <= 0.0 or height / width > 1.5


def _line_uses_side_edges(polygon: list[tuple[float, float]]) -> bool:
    crop_width = max(
        math.dist(polygon[0], polygon[1]),
        math.dist(polygon[3], polygon[2]),
    )
    crop_height = max(
        math.dist(polygon[0], polygon[3]),
        math.dist(polygon[1], polygon[2]),
    )
    return crop_width <= 0.0 or crop_height / crop_width >= 1.5


def _text_layer_geometry(
    source_polygon: list[tuple[float, float]],
    rendered_polygon: list[tuple[float, float]],
    text: str,
) -> tuple[float, Baseline]:
    # A single narrow glyph has no reliable vertical signal; prefer horizontal.
    if len(text.strip()) > 1 and _line_uses_side_edges(source_polygon):
        first_start, first_end = source_polygon[0], source_polygon[3]
        second_start, second_end = source_polygon[1], source_polygon[2]
        baseline_start, baseline_end = source_polygon[0], source_polygon[3]
        opposite_start, opposite_end = source_polygon[1], source_polygon[2]
    else:
        first_start, first_end = source_polygon[0], source_polygon[1]
        second_start, second_end = source_polygon[3], source_polygon[2]
        baseline_start, baseline_end = source_polygon[3], source_polygon[2]
        opposite_start, opposite_end = source_polygon[0], source_polygon[1]

    dx = (first_end[0] - first_start[0]) + (second_end[0] - second_start[0])
    dy = (first_end[1] - first_start[1]) + (second_end[1] - second_start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        dx = first_end[0] - first_start[0]
        dy = first_end[1] - first_start[1]
        length = math.hypot(dx, dy)
    if length <= 1e-6:
        return _line_angle(source_polygon), Baseline()

    unit_x = dx / length
    unit_y = dy / length
    normal_x = -unit_y
    normal_y = unit_x
    if (
        (baseline_start[0] + baseline_end[0]) / 2.0
        - (opposite_start[0] + opposite_end[0]) / 2.0
    ) * normal_x + (
        (baseline_start[1] + baseline_end[1]) / 2.0
        - (opposite_start[1] + opposite_end[1]) / 2.0
    ) * normal_y < 0.0:
        normal_x = -normal_x
        normal_y = -normal_y

    local_points = [
        (
            point[0] * unit_x + point[1] * unit_y,
            point[0] * normal_x + point[1] * normal_y,
        )
        for point in rendered_polygon
    ]
    baseline_points = [
        (
            point[0] * unit_x + point[1] * unit_y,
            point[0] * normal_x + point[1] * normal_y,
        )
        for point in (baseline_start, baseline_end)
    ]
    baseline_dx = baseline_points[1][0] - baseline_points[0][0]
    slope = (
        (baseline_points[1][1] - baseline_points[0][1]) / baseline_dx
        if abs(baseline_dx) > 1e-6
        else 0.0
    )
    if abs(slope) <= 1e-12:
        slope = 0.0
    left = min(point[0] for point in local_points)
    bottom = max(point[1] for point in local_points)
    baseline_at_left = baseline_points[0][1] + slope * (left - baseline_points[0][0])
    intercept = baseline_at_left - bottom
    if abs(intercept) <= 1e-6:
        intercept = 0.0
    textangle = -math.degrees(math.atan2(unit_y, unit_x))
    return textangle, Baseline(
        slope=slope,
        intercept=intercept,
    )


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
    *,
    uses_side_edges: bool,
) -> tuple[list[tuple[float, float]], BoundingBox]:
    if len(geometries) == 1:
        return geometries[0]

    if uses_side_edges:
        polygon = [
            geometries[0][0][0],
            geometries[0][0][1],
            geometries[-1][0][2],
            geometries[-1][0][3],
        ]
    else:
        polygon = [
            geometries[0][0][0],
            geometries[-1][0][1],
            geometries[-1][0][2],
            geometries[0][0][3],
        ]
    left = min(point[0] for point in polygon)
    top = min(point[1] for point in polygon)
    right = max(point[0] for point in polygon)
    bottom = max(point[1] for point in polygon)
    bbox = BoundingBox(left=left, top=top, right=right, bottom=bottom)
    return polygon, bbox


def _line_slice(
    polygon: list[tuple[float, float]],
    start: float,
    end: float,
) -> tuple[list[tuple[float, float]], BoundingBox]:
    def interpolate(
        first: tuple[float, float],
        second: tuple[float, float],
        position: float,
    ) -> tuple[float, float]:
        return (
            first[0] + (second[0] - first[0]) * position,
            first[1] + (second[1] - first[1]) * position,
        )

    if _line_uses_side_edges(polygon):
        word_polygon = [
            interpolate(polygon[0], polygon[3], start),
            interpolate(polygon[1], polygon[2], start),
            interpolate(polygon[1], polygon[2], end),
            interpolate(polygon[0], polygon[3], end),
        ]
    else:
        word_polygon = [
            interpolate(polygon[0], polygon[1], start),
            interpolate(polygon[0], polygon[1], end),
            interpolate(polygon[3], polygon[2], end),
            interpolate(polygon[3], polygon[2], start),
        ]

    return word_polygon, BoundingBox(
        left=min(point[0] for point in word_polygon),
        top=min(point[1] for point in word_polygon),
        right=max(point[0] for point in word_polygon),
        bottom=max(point[1] for point in word_polygon),
    )


def _projected_line_slice(
    line_polygon: list[tuple[float, float]],
    start: float,
    end: float,
    width: int,
    height: int,
    scale: float,
) -> tuple[list[tuple[float, float]], BoundingBox] | None:
    polygon, _bbox = _line_slice(line_polygon, start, end)
    return _polygon_geometry(
        polygon,
        width,
        height,
        scale=scale,
    )


def _proportional_word_geometry(
    source_polygon: list[tuple[float, float]],
    line_polygon: list[tuple[float, float]],
    start: float,
    end: float,
    width: int,
    height: int,
    scale: float,
) -> tuple[list[tuple[float, float]], BoundingBox]:
    geometry = _projected_line_slice(
        source_polygon,
        start,
        end,
        width,
        height,
        scale,
    )
    return geometry or _line_slice(line_polygon, start, end)


def _project_word_region(
    line_polygon: list[tuple[float, float]],
    value: Any,
    width: int,
    height: int,
    scale: float,
) -> tuple[list[tuple[float, float]], BoundingBox] | None:
    geometry = _polygon_geometry(
        _word_polygon(value),
        width,
        height,
        clip=False,
    )
    if geometry is None:
        return None
    _polygon, bbox = geometry

    if _word_regions_are_vertical(line_polygon):
        line_start = line_polygon[0][1]
        line_end = line_polygon[2][1]
        region_start = bbox.top
        region_end = bbox.bottom
    else:
        line_start = line_polygon[0][0]
        line_end = line_polygon[1][0]
        region_start = bbox.left
        region_end = bbox.right

    line_length = line_end - line_start
    if abs(line_length) <= 1e-6:
        return None
    start, end = sorted(
        (
            (region_start - line_start) / line_length,
            (region_end - line_start) / line_length,
        )
    )
    start = min(max(start, 0.0), 1.0)
    end = min(max(end, 0.0), 1.0)
    if end - start <= 1e-6:
        return None
    return _projected_line_slice(
        line_polygon,
        start,
        end,
        width,
        height,
        scale,
    )


def _fallback_words(
    text: str,
    source_polygon: list[tuple[float, float]],
    line_polygon: list[tuple[float, float]],
    line_bbox: BoundingBox,
    confidence: float | None,
    language: str,
    width: int,
    height: int,
    scale: float,
) -> list[OcrElement]:
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return [
            _fallback_word(
                text,
                line_polygon,
                line_bbox,
                confidence,
                language,
            )
        ]

    words = []
    for match in matches:
        start = match.start() / len(text)
        end = match.end() / len(text)
        geometry = _proportional_word_geometry(
            source_polygon,
            line_polygon,
            start,
            end,
            width,
            height,
            scale,
        )
        words.append(
            _fallback_word(
                match.group(),
                *geometry,
                confidence,
                language,
            )
        )
    return words


def _words_for_line(
    text: str,
    source_polygon: list[tuple[float, float]],
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
                matches = list(re.finditer(r"\S+", text))
                if len(matches) != len(tokens):
                    raise ValueError
                words = []
                for token, region, match in zip(
                    tokens,
                    word_regions,
                    matches,
                    strict=True,
                ):
                    geometry = _project_word_region(
                        source_polygon,
                        region,
                        width,
                        height,
                        scale,
                    )
                    if geometry is None:
                        geometry = _proportional_word_geometry(
                            source_polygon,
                            line_polygon,
                            match.start() / len(text),
                            match.end() / len(text),
                            width,
                            height,
                            scale,
                        )
                    polygon, bbox = geometry
                    words.append(
                        _fallback_word(
                            token,
                            polygon,
                            bbox,
                            confidence,
                            language,
                        )
                    )
                return words

        grouped_words: list[
            tuple[
                str,
                list[tuple[list[tuple[float, float]], BoundingBox]] | None,
            ]
        ] = []
        current_text = ""
        current_geometries: list[tuple[list[tuple[float, float]], BoundingBox]] = []
        current_geometry_valid = True

        def finish_word() -> None:
            nonlocal current_geometry_valid, current_geometries, current_text
            if current_text:
                grouped_words.append(
                    (
                        current_text,
                        current_geometries if current_geometry_valid else None,
                    )
                )
            current_text = ""
            current_geometries = []
            current_geometry_valid = True

        for token, region in zip(tokens, word_regions, strict=True):
            if token[:1].isspace():
                finish_word()

            content = token.strip()
            if content:
                geometry = _project_word_region(
                    source_polygon,
                    region,
                    width,
                    height,
                    scale,
                )
                if geometry is None:
                    current_geometry_valid = False
                else:
                    current_geometries.append(geometry)
                current_text += content

            if token[-1:].isspace():
                finish_word()
        finish_word()

        if not grouped_words:
            raise ValueError
        if " ".join(word[0] for word in grouped_words) != " ".join(text.split()):
            raise ValueError
        matches = list(re.finditer(r"\S+", text))
        if len(matches) != len(grouped_words):
            raise ValueError

        words = []
        for (token, geometries), match in zip(
            grouped_words,
            matches,
            strict=True,
        ):
            if token != match.group():
                raise ValueError
            if geometries is None:
                polygon, bbox = _proportional_word_geometry(
                    source_polygon,
                    line_polygon,
                    match.start() / len(text),
                    match.end() / len(text),
                    width,
                    height,
                    scale,
                )
            else:
                polygon, bbox = _combined_word_geometry(
                    geometries,
                    uses_side_edges=_line_uses_side_edges(source_polygon),
                )
            words.append(
                _fallback_word(
                    token,
                    polygon,
                    bbox,
                    confidence,
                    language,
                )
            )
        return words
    except (TypeError, ValueError):
        return _fallback_words(
            text,
            source_polygon,
            line_polygon,
            line_bbox,
            confidence,
            language,
            width,
            height,
            scale,
        )


def _region_from_geometry(
    value: Any,
    width: int,
    height: int,
) -> _LayoutRegion | None:
    geometry = _polygon_geometry(
        _word_polygon(value),
        width,
        height,
    )
    if geometry is None:
        return None
    _polygon, bbox = geometry
    left = max(0, min(width, math.floor(bbox.left)))
    top = max(0, min(height, math.floor(bbox.top)))
    right = max(0, min(width, math.ceil(bbox.right)))
    bottom = max(0, min(height, math.ceil(bbox.bottom)))
    if right - left < 2 or bottom - top < 2:
        return None
    return _LayoutRegion(left, top, right, bottom)


def _result_text_regions(
    result: Any,
    width: int,
    height: int,
) -> list[_LayoutRegion]:
    texts, _scores, polygons, _boxes = _parallel_line_data(result)
    word_data = _optional_word_data(result, len(texts))
    regions = []
    if word_data is not None:
        _word_texts, word_regions = word_data
        for line_regions in word_regions:
            try:
                values = list(line_regions)
            except TypeError:
                continue
            for value in values:
                region = _region_from_geometry(value, width, height)
                if region is not None:
                    regions.append(region)
    if regions:
        return regions

    for value in polygons:
        region = _region_from_geometry(value, width, height)
        if region is not None:
            regions.append(region)
    return regions


def _column_regions(
    content_regions: list[_LayoutRegion],
    width: int,
    height: int,
) -> list[_LayoutRegion]:
    if len(content_regions) < 4:
        return [_LayoutRegion(0, 0, width, height)]

    intervals = sorted((region.left, region.right) for region in content_regions)
    merged: list[list[int]] = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])

    minimum_gap = max(12, round(width * 0.04))
    minimum_span = width * 0.15
    candidates = []
    for first, second in zip(merged, merged[1:]):
        gap = second[0] - first[1]
        if gap < minimum_gap:
            continue
        split = (first[1] + second[0]) // 2
        left_regions = [
            region
            for region in content_regions
            if (region.left + region.right) / 2 < split
        ]
        right_regions = [
            region
            for region in content_regions
            if (region.left + region.right) / 2 >= split
        ]
        if len(left_regions) < 2 or len(right_regions) < 2:
            continue
        candidates.append((gap, split))

    if not candidates:
        return [_LayoutRegion(0, 0, width, height)]

    largest_gap = max(gap for gap, _split in candidates)
    separators = [split for gap, split in candidates if gap >= largest_gap * 0.7]

    boundaries = [0, *sorted(separators), width]
    columns = [
        _LayoutRegion(left, 0, right, height)
        for left, right in zip(boundaries, boundaries[1:])
        if right > left
    ]
    if len(columns) <= 1:
        return [_LayoutRegion(0, 0, width, height)]

    for column in columns:
        matches = [
            region
            for region in content_regions
            if column.left <= (region.left + region.right) / 2 < column.right
        ]
        if len(matches) < 2:
            return [_LayoutRegion(0, 0, width, height)]
        content_span = max(region.right for region in matches) - min(
            region.left for region in matches
        )
        if content_span < minimum_span:
            return [_LayoutRegion(0, 0, width, height)]
    return columns


def _line_column_index(
    line: OcrElement,
    columns: list[_LayoutRegion],
) -> int:
    if line.bbox is None:
        return 0
    center = (line.bbox.left + line.bbox.right) / 2
    for index, column in enumerate(columns):
        if column.left <= center < column.right:
            return index
    return min(
        range(len(columns)),
        key=lambda index: abs(
            center - (columns[index].left + columns[index].right) / 2
        ),
    )


def _sort_lines(
    lines: list[OcrElement],
    columns: list[_LayoutRegion],
) -> list[OcrElement]:
    return sorted(
        lines,
        key=lambda line: (
            _line_column_index(line, columns),
            line.bbox.top if line.bbox is not None else math.inf,
            line.bbox.left if line.bbox is not None else math.inf,
        ),
    )


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


def _lines_from_result(
    result: Any,
    width: int,
    height: int,
    scale: float,
    language: str,
) -> list[OcrElement]:
    texts, scores, polygons, _boxes = _parallel_line_data(result)
    word_data = _optional_word_data(result, len(texts))
    lines = []
    for index in range(len(texts)):
        text = texts[index]
        if not isinstance(text, str):
            raise OCRError(
                f"PaddleOCR result field rec_texts[{index}] must be a string"
            )
        source_geometry = _polygon_geometry(
            polygons[index],
            width,
            height,
            clip=False,
        )
        geometry = _polygon_geometry(
            polygons[index],
            width,
            height,
            scale=scale,
        )
        if not text or source_geometry is None or geometry is None:
            continue

        source_polygon, _source_bbox = source_geometry
        polygon, bbox = geometry
        scaled_source_polygon = [(x * scale, y * scale) for x, y in source_polygon]
        textangle, baseline = _text_layer_geometry(
            scaled_source_polygon,
            polygon,
            text,
        )
        confidence = _confidence(scores[index])
        if word_data is None:
            words = _fallback_words(
                text,
                source_polygon,
                polygon,
                bbox,
                confidence,
                language,
                width,
                height,
                scale,
            )
        else:
            word_texts, word_regions = word_data
            words = _words_for_line(
                text,
                source_polygon,
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
                baseline=baseline,
                textangle=textangle,
            )
        )
    return lines


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
        page_key = _page_cache_key(input_file)
        with _prediction_lock:
            _discard_stale_prediction_results()
            if page_key is not None:
                _pending_deskew_results.pop(page_key, None)

        width, height, _dpi = _image_properties(input_file)
        result = _predict(input_file, options)
        _texts, _scores, polygons, _boxes = _parallel_line_data(result)
        minimum_length = max(20.0, min(width, height) * 0.05)
        angles = []
        for value in polygons:
            geometry = _polygon_geometry(value, width, height)
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
        correction = statistics.median(angles) if len(angles) >= 2 else 0.0
        model_pair = _model_pair_for_cache(options)
        if correction == 0.0 and page_key is not None and model_pair is not None:
            with _prediction_lock:
                _pending_deskew_results[page_key] = _CachedPrediction(
                    result=result,
                    model_pair=model_pair,
                    image_size=(width, height),
                )
        return correction

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
        language = _language_metadata(options)
        layout = bool(getattr(options.paddle, "layout", False))
        result = _predict(input_file, options)
        lines = _lines_from_result(
            result,
            width,
            height,
            scale,
            language,
        )
        if layout:
            content_regions = _result_text_regions(result, width, height)
            columns = _column_regions(content_regions, width, height)
            scaled_columns = [
                _LayoutRegion(
                    round(column.left * scale),
                    round(column.top * scale),
                    round(column.right * scale),
                    round(column.bottom * scale),
                )
                for column in columns
            ]
            lines = _sort_lines(lines, scaled_columns)

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
        return page, "\n".join(line.text for line in lines)

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


def _release_model_cache() -> None:
    """Close and clear cached PaddleOCR state."""
    global _cached_execution_provider, _cached_fingerprint, _cached_model, _cached_pair
    with _prediction_lock, _model_lock:
        if _cached_model is not None:
            close = getattr(_cached_model, "close", None)
            if callable(close):
                close()
        _cached_model = None
        _cached_pair = None
        _cached_fingerprint = None
        _cached_execution_provider = None
        _pending_deskew_results.clear()
        _prediction_result_by_image.clear()
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
    parser.add_argument(
        "--paddle-execution-provider",
        default="cpu",
        help=SUPPRESS,
    )
    parser.add_argument(
        "--paddle-layout",
        action="store_true",
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
    with _prediction_lock:
        _discard_stale_prediction_results()
        _prediction_result_by_image.pop(key, None)
        page_key = _page_cache_key(key)
        pending = (
            _pending_deskew_results.pop(page_key, None)
            if page_key is not None
            else None
        )
        options = getattr(page, "options", None)
        expected_page = getattr(page, "pageno", None)
        page_number_matches = page_key is not None and (
            expected_page is None
            or isinstance(expected_page, int)
            and page_key[1] == expected_page + 1
        )
        if (
            pending is not None
            and options is not None
            and bool(getattr(options, "deskew", False))
            and not bool(getattr(options, "clean", False))
            and not bool(getattr(options, "remove_background", False))
            and pending.model_pair == _model_pair_for_cache(options)
            and pending.image_size == image.size
            and page_number_matches
            and not _ocr_image_masks_text(page)
        ):
            _prediction_result_by_image[key] = pending
    with _coordinate_dpi_lock:
        _coordinate_dpi_by_image[key] = coordinate_dpi


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine(options: OcrOptions | None) -> PaddleOcrEngine | None:
    """Select PaddleOCR only when OCRmyPDF requests the paddle engine."""
    if options is not None and options.ocr_engine != "paddle":
        return None
    return PaddleOcrEngine()
