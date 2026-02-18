# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ocrmypdf plugin for OpenCV-based image preprocessing.

This plugin implements the ``filter_ocr_image`` hook to denoise and
binarize page images before they are sent to Tesseract.
"""

import logging

import numpy as np
import ocrmypdf
from PIL import Image

logger = logging.getLogger(__name__)


@ocrmypdf.hookimpl
def filter_ocr_image(page, image):  # noqa: ARG001 – page required by hook spec
    """Preprocess the OCR image using OpenCV denoising and adaptive thresholding.

    Args:
        page: ocrmypdf page context (unused).
        image: PIL Image of the rendered page.

    Returns:
        Preprocessed PIL Image.
    """
    import cv2

    img = np.array(image)

    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    result = Image.fromarray(binary)
    result.info = image.info.copy()
    return result
