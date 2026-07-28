# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Offline document-orientation normalization for OCR page rotation."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import shutil
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath
from typing import Any

from ._ocr_runtime import (
    _format_exception_message,
    execution_provider_base,
    onnxruntime_engine_config,
    require_execution_provider,
    validate_ocr_execution_provider,
)
from .exceptions import OCRError

logger = logging.getLogger(__name__)

MODEL_NAME = "PP-LCNet_x1_0_doc_ori"
ORIENTATION_BATCH_SIZE = 8
ORIENTATION_CONFIDENCE_THRESHOLD = 0.80
ORIENTATION_RETRY_MIN_SCORE = 0.50
ORIENTATION_RENDER_SCALE = 1.5

_ALLOWED_ANGLES = frozenset({0, 90, 180, 270})
_MODEL_RESOURCE_PARTS = (
    "resources",
    "models",
    MODEL_NAME,
)
_MODEL_MANIFEST = "manifest.json"

_model: Any | None = None
_model_execution_provider: str | None = None
_model_dir: Path | None = None
_model_lock = threading.RLock()
_prediction_lock = threading.Lock()
_resource_stack = ExitStack()
atexit.register(_resource_stack.close)


@dataclass(frozen=True)
class OrientationResult:
    """Result of orientation analysis and optional PDF rotation."""

    page_number: int
    correction_angle: int
    score: float
    previous_rotate: int
    final_rotate: int
    applied: bool


@dataclass(frozen=True)
class _Prediction:
    """Raw orientation result returned by PaddleOCR for one rendered page."""

    page_number: int
    correction_angle: int
    score: float


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_directory(model_dir: Path) -> None:
    """Validate the bundled model manifest and every required artifact."""
    manifest_path = model_dir / _MODEL_MANIFEST
    if not manifest_path.is_file():
        raise OCRError(
            f"Bundled Paddle orientation manifest is missing: {manifest_path}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OCRError(
            f"Bundled Paddle orientation manifest is invalid: {manifest_path}"
        ) from exc

    if manifest.get("model_name") != MODEL_NAME:
        raise OCRError("Bundled Paddle orientation manifest has the wrong model name")
    if manifest.get("engine") != "onnxruntime":
        raise OCRError("Bundled Paddle orientation manifest has the wrong engine")

    model_files = manifest.get("files")
    if not isinstance(model_files, dict) or not model_files:
        raise OCRError("Bundled Paddle orientation manifest contains no model files")

    for relative_name, expected_hash in model_files.items():
        relative_path = PurePosixPath(str(relative_name))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise OCRError(
                "Bundled Paddle orientation manifest contains an unsafe file path"
            )
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise OCRError(
                f"Bundled Paddle orientation manifest has an invalid hash for "
                f"{relative_name}"
            )

        artifact_path = model_dir.joinpath(*relative_path.parts)
        if not artifact_path.is_file():
            raise OCRError(
                f"Bundled Paddle orientation model file is missing: {artifact_path}"
            )
        if _sha256(artifact_path) != expected_hash.lower():
            raise OCRError(
                f"Bundled Paddle orientation model file is corrupt: {artifact_path}"
            )


def _resolve_model_directory() -> Path:
    """Materialize and validate the package-internal model directory once."""
    global _model_dir

    with _model_lock:
        if _model_dir is not None:
            return _model_dir

        try:
            resource = files("pdftopdfa")
            for part in _MODEL_RESOURCE_PARTS:
                resource = resource.joinpath(part)
            model_dir = Path(_resource_stack.enter_context(as_file(resource)))
            _validate_model_directory(model_dir)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(
                f"Could not access the bundled Paddle orientation model: {exc}"
            ) from exc

        _model_dir = model_dir
        return model_dir


def _create_model(execution_provider: str = "cpu") -> Any:
    """Create a PaddleOCR classifier from the bundled ONNX files."""
    try:
        from paddleocr import DocImgOrientationClassification

        model = DocImgOrientationClassification(
            model_name=MODEL_NAME,
            model_dir=str(_resolve_model_directory()),
            engine="onnxruntime",
            device="cpu",
            engine_config=onnxruntime_engine_config(execution_provider),
        )
        if execution_provider_base(execution_provider) == "directml":
            try:
                session = model.paddlex_predictor.runner.session
                require_execution_provider(session, execution_provider)
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
            "Could not load the bundled Paddle orientation model: "
            f"{_format_exception_message(exc)}"
        ) from exc


def _get_model(execution_provider: str = "cpu") -> Any:
    """Return the process-wide lazy PaddleOCR classifier for a provider."""
    global _model, _model_execution_provider

    execution_provider = validate_ocr_execution_provider(execution_provider)
    with _model_lock:
        if _model is None or _model_execution_provider != execution_provider:
            if _model is not None:
                close = getattr(_model, "close", None)
                if callable(close):
                    close()
                _model = None
                _model_execution_provider = None
            _model = _create_model(execution_provider)
            _model_execution_provider = execution_provider
        return _model


def _release_model_cache() -> None:
    """Close and clear the model singleton."""
    global _model, _model_execution_provider

    with _model_lock:
        if _model is not None:
            close = getattr(_model, "close", None)
            if callable(close):
                close()
        _model = None
        _model_execution_provider = None


def _parse_prediction(result: Any, page_number: int) -> _Prediction:
    """Validate and parse one PaddleOCR result."""
    try:
        labels = result["label_names"]
        scores = result["scores"]
        if len(labels) != 1 or len(scores) != 1:
            raise ValueError("expected exactly one orientation prediction")
        correction_angle = int(labels[0])
        score = float(scores[0])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise OCRError(
            f"Paddle orientation returned invalid output for page {page_number}"
        ) from exc

    if correction_angle not in _ALLOWED_ANGLES:
        raise OCRError(
            f"Paddle orientation returned unsupported angle {correction_angle} "
            f"for page {page_number}"
        )
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise OCRError(
            f"Paddle orientation returned invalid score {score!r} "
            f"for page {page_number}"
        )

    return _Prediction(page_number, correction_angle, score)


def _predict_batch(
    images: list[Any],
    page_numbers: list[int],
    *,
    execution_provider: str = "cpu",
) -> list[_Prediction]:
    """Run one serialized PaddleOCR batch and validate its output."""
    try:
        with _prediction_lock:
            results = list(
                _get_model(execution_provider).predict(
                    images,
                    batch_size=len(images),
                )
            )
    except OCRError:
        raise
    except Exception as exc:
        first_page = page_numbers[0]
        last_page = page_numbers[-1]
        raise OCRError(
            f"Paddle orientation inference failed for pages "
            f"{first_page}-{last_page}: {exc}"
        ) from exc

    if len(results) != len(images):
        raise OCRError(
            "Paddle orientation returned a different number of results than inputs"
        )
    return [
        _parse_prediction(result, page_number)
        for result, page_number in zip(results, page_numbers, strict=True)
    ]


def _refine_low_confidence_predictions(
    images: list[Any],
    predictions: list[_Prediction],
    *,
    execution_provider: str = "cpu",
) -> list[_Prediction]:
    """Retry uncertain pages with rotated and spatial Paddle consensus."""
    import numpy as np

    retry_indices = [
        index
        for index, prediction in enumerate(predictions)
        if ORIENTATION_RETRY_MIN_SCORE
        <= prediction.score
        < ORIENTATION_CONFIDENCE_THRESHOLD
    ]
    if not retry_indices:
        return predictions

    retry_images: list[Any] = []
    retry_metadata: list[tuple[int, int]] = []
    for index in retry_indices:
        for pre_rotation in (90, 180, 270):
            retry_images.append(
                np.ascontiguousarray(np.rot90(images[index], k=pre_rotation // 90))
            )
            retry_metadata.append((index, pre_rotation))

    candidates = {index: [predictions[index]] for index in retry_indices}
    for start in range(0, len(retry_images), ORIENTATION_BATCH_SIZE):
        end = start + ORIENTATION_BATCH_SIZE
        batch_metadata = retry_metadata[start:end]
        batch_predictions = _predict_batch(
            retry_images[start:end],
            [predictions[index].page_number for index, _ in batch_metadata],
            execution_provider=execution_provider,
        )
        for prediction, (index, pre_rotation) in zip(
            batch_predictions,
            batch_metadata,
            strict=True,
        ):
            candidates[index].append(
                _Prediction(
                    page_number=prediction.page_number,
                    correction_angle=(pre_rotation + prediction.correction_angle) % 360,
                    score=prediction.score,
                )
            )

    refined = predictions.copy()
    for index, page_candidates in candidates.items():
        counts = {
            angle: sum(
                candidate.correction_angle == angle for candidate in page_candidates
            )
            for angle in _ALLOWED_ANGLES
        }
        winning_angle = max(counts, key=lambda angle: counts[angle])
        if counts[winning_angle] < 3:
            continue
        winning_score = max(
            candidate.score
            for candidate in page_candidates
            if candidate.correction_angle == winning_angle
        )
        refined[index] = _Prediction(
            page_number=predictions[index].page_number,
            correction_angle=winning_angle,
            score=winning_score,
        )

    spatial_indices = [
        index
        for index in retry_indices
        if refined[index].score < ORIENTATION_CONFIDENCE_THRESHOLD
    ]
    spatial_images: list[Any] = []
    spatial_metadata: list[int] = []
    for index in spatial_indices:
        image = images[index]
        height, width = image.shape[:2]
        top_band = int(height * 0.45)
        bottom_band = int(height * 0.55)
        left_band = int(width * 0.55)
        right_band = int(width * 0.45)
        variants = (
            image[:top_band, :],
            image[bottom_band:, :],
            image[:, :left_band],
            image[:, right_band:],
            image[:bottom_band, :left_band],
            image[:bottom_band, right_band:],
            image[top_band:, :left_band],
            image[top_band:, right_band:],
        )
        for variant in variants:
            spatial_images.append(np.ascontiguousarray(variant))
            spatial_metadata.append(index)

    spatial_candidates = {index: [] for index in spatial_indices}
    for start in range(0, len(spatial_images), ORIENTATION_BATCH_SIZE):
        end = start + ORIENTATION_BATCH_SIZE
        batch_indices = spatial_metadata[start:end]
        batch_predictions = _predict_batch(
            spatial_images[start:end],
            [predictions[index].page_number for index in batch_indices],
            execution_provider=execution_provider,
        )
        for prediction, index in zip(
            batch_predictions,
            batch_indices,
            strict=True,
        ):
            if prediction.score >= ORIENTATION_CONFIDENCE_THRESHOLD:
                spatial_candidates[index].append(prediction)

    for index, page_candidates in spatial_candidates.items():
        counts = {
            angle: sum(
                candidate.correction_angle == angle for candidate in page_candidates
            )
            for angle in _ALLOWED_ANGLES
        }
        winning_angle = max(counts, key=lambda angle: counts[angle])
        if counts[winning_angle] < 3:
            continue
        winning_score = max(
            candidate.score
            for candidate in page_candidates
            if candidate.correction_angle == winning_angle
        )
        refined[index] = _Prediction(
            page_number=predictions[index].page_number,
            correction_angle=winning_angle,
            score=winning_score,
        )

    return refined


def _classify_pdf_pages(
    pdf_path: Path,
    *,
    execution_provider: str = "cpu",
) -> list[_Prediction]:
    """Render and classify every visible page in a PDF."""
    try:
        import numpy as np
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        predictions: list[_Prediction] = []
        batch_images: list[Any] = []
        batch_page_numbers: list[int] = []
        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                bitmap = None
                try:
                    bitmap = page.render(scale=ORIENTATION_RENDER_SCALE)
                    pil_image = bitmap.to_pil().convert("RGB")
                    rgb = np.asarray(pil_image)
                    bgr = np.ascontiguousarray(rgb[..., ::-1])
                    batch_images.append(bgr)
                    batch_page_numbers.append(page_index + 1)
                finally:
                    if bitmap is not None:
                        close_bitmap = getattr(bitmap, "close", None)
                        if callable(close_bitmap):
                            close_bitmap()
                    close_page = getattr(page, "close", None)
                    if callable(close_page):
                        close_page()

                if len(batch_images) == ORIENTATION_BATCH_SIZE:
                    batch_predictions = _predict_batch(
                        batch_images,
                        batch_page_numbers,
                        execution_provider=execution_provider,
                    )
                    predictions.extend(
                        _refine_low_confidence_predictions(
                            batch_images,
                            batch_predictions,
                            execution_provider=execution_provider,
                        )
                    )
                    batch_images = []
                    batch_page_numbers = []

            if batch_images:
                batch_predictions = _predict_batch(
                    batch_images,
                    batch_page_numbers,
                    execution_provider=execution_provider,
                )
                predictions.extend(
                    _refine_low_confidence_predictions(
                        batch_images,
                        batch_predictions,
                        execution_provider=execution_provider,
                    )
                )
        finally:
            close_pdf = getattr(pdf, "close", None)
            if callable(close_pdf):
                close_pdf()
        return predictions
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(
            f"Could not render PDF pages for Paddle orientation: {exc}"
        ) from exc


def _effective_page_rotate(page: Any) -> int:
    """Return a page's effective inherited PDF /Rotate value."""
    current = page.obj
    visited: set[tuple[int, int]] = set()

    while current is not None:
        objgen = getattr(current, "objgen", None)
        if isinstance(objgen, tuple):
            if objgen in visited:
                break
            visited.add(objgen)

        if "/Rotate" in current:
            return int(current.get("/Rotate", 0) or 0) % 360
        current = current.get("/Parent")

    return 0


def _corrected_page_rotate(existing_rotate: int, correction_angle: int) -> int:
    """Map Paddle's counter-clockwise correction to PDF clockwise rotation."""
    return (existing_rotate - correction_angle) % 360


def normalize_pdf_orientation(
    input_path: Path,
    output_path: Path,
    *,
    execution_provider: str = "cpu",
) -> list[OrientationResult]:
    """Normalize all confidently detected page orientations into a PDF copy."""
    execution_provider = validate_ocr_execution_provider(execution_provider)
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise OCRError("Paddle orientation input and output paths must differ")

    predictions = _classify_pdf_pages(
        input_path,
        execution_provider=execution_provider,
    )

    try:
        import pikepdf

        results: list[OrientationResult] = []
        changed_pages: list[int] = []
        with pikepdf.open(input_path) as pdf:
            if len(pdf.pages) != len(predictions):
                raise OCRError(
                    "PDF page count changed during Paddle orientation analysis"
                )

            for page, prediction in zip(pdf.pages, predictions, strict=True):
                previous_rotate = _effective_page_rotate(page)
                final_rotate = previous_rotate
                applied = False

                if prediction.score < ORIENTATION_CONFIDENCE_THRESHOLD:
                    logger.warning(
                        "Paddle orientation confidence below %.2f on page %s: "
                        "angle=%s, score=%.4f; leaving page unchanged",
                        ORIENTATION_CONFIDENCE_THRESHOLD,
                        prediction.page_number,
                        prediction.correction_angle,
                        prediction.score,
                    )
                else:
                    final_rotate = _corrected_page_rotate(
                        previous_rotate,
                        prediction.correction_angle,
                    )
                    if final_rotate != previous_rotate:
                        page.Rotate = final_rotate
                        applied = True
                        changed_pages.append(prediction.page_number)

                results.append(
                    OrientationResult(
                        page_number=prediction.page_number,
                        correction_angle=prediction.correction_angle,
                        score=prediction.score,
                        previous_rotate=previous_rotate,
                        final_rotate=final_rotate,
                        applied=applied,
                    )
                )

            if changed_pages:
                pdf.save(output_path)
            else:
                shutil.copy2(input_path, output_path)
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(f"Could not apply Paddle page orientations: {exc}") from exc

    logger.info(
        "Paddle orientation analyzed %s page(s) and rotated %s page(s)%s",
        len(results),
        len(changed_pages),
        (
            f": {', '.join(str(page) for page in changed_pages)}"
            if changed_pages
            else ""
        ),
    )
    return results
