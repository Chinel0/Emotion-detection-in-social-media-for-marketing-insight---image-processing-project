"""
Image preprocessing utilities for emotion detection.

Provides functions for loading, resizing, normalizing, and enhancing
images, as well as face detection and region-of-interest extraction
using OpenCV.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from typing import Optional


# Haar-cascade XML bundled with OpenCV
_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_image(source: str | np.ndarray) -> np.ndarray:
    """Load an image from a file path or accept a NumPy array.

    Parameters
    ----------
    source:
        Path to an image file **or** a BGR/RGB NumPy array.

    Returns
    -------
    np.ndarray
        Image as a BGR NumPy array (uint8).

    Raises
    ------
    FileNotFoundError
        If *source* is a path that does not exist.
    ValueError
        If the file cannot be decoded as an image.
    """
    if isinstance(source, np.ndarray):
        return source.copy()

    if not os.path.isfile(source):
        raise FileNotFoundError(f"Image file not found: {source}")

    img = cv2.imread(source)
    if img is None:
        raise ValueError(f"Could not decode image: {source}")
    return img


def resize_image(
    image: np.ndarray,
    width: int = 224,
    height: int = 224,
    keep_aspect_ratio: bool = False,
) -> np.ndarray:
    """Resize an image to the specified dimensions.

    Parameters
    ----------
    image:
        Input BGR image as a NumPy array.
    width, height:
        Target dimensions in pixels.
    keep_aspect_ratio:
        When *True* the image is letterboxed to fit the target size
        while preserving its original aspect ratio.

    Returns
    -------
    np.ndarray
        Resized BGR image.
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        scale = min(width / w, height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, image.shape[2] if image.ndim == 3 else 1),
                          dtype=image.dtype)
        y_off = (height - new_h) // 2
        x_off = (width - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to the [0, 1] range.

    Parameters
    ----------
    image:
        Input image as a NumPy array (uint8 or float).

    Returns
    -------
    np.ndarray
        Float64 array with values in [0, 1].
    """
    return image.astype(np.float64) / 255.0


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Parameters
    ----------
    image:
        Input BGR image.

    Returns
    -------
    np.ndarray
        Single-channel grayscale image.
    """
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_image(
    image: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.2,
    sharpness: float = 1.5,
) -> np.ndarray:
    """Apply brightness, contrast, and sharpness enhancement.

    Enhancement is performed with Pillow for better quality.

    Parameters
    ----------
    image:
        Input BGR NumPy array.
    brightness:
        Factor ≥ 0 (1.0 = original). Values > 1 brighten the image.
    contrast:
        Factor ≥ 0 (1.0 = original). Values > 1 increase contrast.
    sharpness:
        Factor ≥ 0 (1.0 = original). Values > 1 sharpen the image.

    Returns
    -------
    np.ndarray
        Enhanced BGR image.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if brightness != 1.0:
        pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    if contrast != 1.0:
        pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    if sharpness != 1.0:
        pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def detect_faces(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> list[tuple[int, int, int, int]]:
    """Detect faces in an image using a Haar-cascade classifier.

    Parameters
    ----------
    image:
        Input BGR image.
    scale_factor:
        How much the image size is reduced at each scale.
    min_neighbors:
        Minimum number of neighbours each candidate rectangle should
        retain to be considered a valid detection.
    min_size:
        Minimum object size.

    Returns
    -------
    list of (x, y, w, h)
        Bounding boxes of detected faces.
    """
    cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
    gray = convert_to_grayscale(image)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    if len(faces) == 0:
        return []
    return [tuple(map(int, face)) for face in faces]


def extract_face_regions(
    image: np.ndarray,
    padding: float = 0.1,
) -> list[np.ndarray]:
    """Return cropped face sub-images.

    An optional *padding* fraction is applied around each detected
    bounding box so that facial-feature models have surrounding context.

    Parameters
    ----------
    image:
        Input BGR image.
    padding:
        Fractional padding applied to each side of the bounding box
        (e.g. 0.1 means 10 % of the box width/height on each side).

    Returns
    -------
    list of np.ndarray
        Cropped BGR face images. Empty list when no faces are detected.
    """
    faces = detect_faces(image)
    h_img, w_img = image.shape[:2]
    regions = []
    for x, y, w, h in faces:
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        regions.append(image[y1:y2, x1:x2])
    return regions


def preprocess_for_detection(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Full preprocessing pipeline for an emotion-detection model.

    Steps: resize → enhance contrast/sharpness → return BGR uint8.

    Parameters
    ----------
    image:
        Input BGR image.
    target_size:
        (width, height) for the output image.

    Returns
    -------
    np.ndarray
        Preprocessed BGR image ready for the detection model.
    """
    resized = resize_image(image, width=target_size[0], height=target_size[1])
    enhanced = enhance_image(resized, contrast=1.2, sharpness=1.3)
    return enhanced
