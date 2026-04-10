"""
Core emotion detection module.

Wraps DeepFace to provide per-image and batch emotion analysis,
returning structured results that downstream modules can consume.
"""

import os
import logging
import numpy as np
import cv2
from typing import Any

from image_preprocessor import load_image, preprocess_for_detection, detect_faces

logger = logging.getLogger(__name__)

# Canonical emotion labels returned by DeepFace
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# Suppress TensorFlow/Keras verbosity by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _import_deepface():
    """Lazy-import DeepFace so the rest of the module loads without TF."""
    from deepface import DeepFace  # noqa: PLC0415
    return DeepFace


class EmotionDetectionResult:
    """Container for the emotion analysis result of a single image.

    Attributes
    ----------
    emotions : dict[str, float]
        Mapping of emotion label → confidence score (0–100).
    dominant_emotion : str
        Emotion with the highest confidence.
    face_count : int
        Number of faces detected in the image.
    faces : list[dict]
        Raw per-face DeepFace output (may be empty when no face found).
    error : str or None
        Error message when analysis failed, ``None`` on success.
    """

    def __init__(
        self,
        emotions: dict[str, float],
        dominant_emotion: str,
        face_count: int,
        faces: list[dict],
        error: str | None = None,
    ):
        self.emotions = emotions
        self.dominant_emotion = dominant_emotion
        self.face_count = face_count
        self.faces = faces
        self.error = error

    @property
    def success(self) -> bool:
        """Return *True* when analysis completed without errors."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serializable dictionary representation."""
        return {
            "emotions": self.emotions,
            "dominant_emotion": self.dominant_emotion,
            "face_count": self.face_count,
            "success": self.success,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"EmotionDetectionResult(dominant={self.dominant_emotion!r}, "
            f"faces={self.face_count}, success={self.success})"
        )


def _zero_emotions() -> dict[str, float]:
    return {e: 0.0 for e in EMOTIONS}


def _build_result_from_deepface(raw_faces: list[dict]) -> EmotionDetectionResult:
    """Convert raw DeepFace output into an :class:`EmotionDetectionResult`."""
    if not raw_faces:
        return EmotionDetectionResult(
            emotions=_zero_emotions(),
            dominant_emotion="neutral",
            face_count=0,
            faces=[],
        )

    # Aggregate emotion scores across all detected faces (simple average).
    aggregated: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    valid_faces = 0
    for face in raw_faces:
        face_emotions: dict[str, float] = face.get("emotion", {})
        if face_emotions:
            for emotion in EMOTIONS:
                aggregated[emotion] += face_emotions.get(emotion, 0.0)
            valid_faces += 1

    if valid_faces > 0:
        aggregated = {e: v / valid_faces for e, v in aggregated.items()}

    dominant = max(aggregated, key=aggregated.__getitem__)
    return EmotionDetectionResult(
        emotions=aggregated,
        dominant_emotion=dominant,
        face_count=len(raw_faces),
        faces=raw_faces,
    )


def detect_emotions(
    source: str | np.ndarray,
    preprocess: bool = True,
    enforce_detection: bool = False,
) -> EmotionDetectionResult:
    """Detect emotions in a single image.

    Parameters
    ----------
    source:
        File path to an image **or** a BGR NumPy array.
    preprocess:
        When *True* (default) the image is passed through
        :func:`image_preprocessor.preprocess_for_detection` before analysis.
    enforce_detection:
        When *False* (default) DeepFace does not raise an exception when no
        face is found; the result will have ``face_count == 0``.

    Returns
    -------
    EmotionDetectionResult
        Structured result.  Check :attr:`EmotionDetectionResult.success`
        before using the scores.
    """
    try:
        image = load_image(source)
    except (FileNotFoundError, ValueError) as exc:
        return EmotionDetectionResult(
            emotions=_zero_emotions(),
            dominant_emotion="neutral",
            face_count=0,
            faces=[],
            error=str(exc),
        )

    DeepFace = _import_deepface()

    if preprocess:
        image = preprocess_for_detection(image)

    try:
        raw = DeepFace.analyze(
            img_path=image,
            actions=["emotion"],
            enforce_detection=enforce_detection,
            silent=True,
        )
        # DeepFace.analyze returns a list of face dicts
        if isinstance(raw, dict):
            raw = [raw]
        return _build_result_from_deepface(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DeepFace analysis failed: %s", exc)
        return EmotionDetectionResult(
            emotions=_zero_emotions(),
            dominant_emotion="neutral",
            face_count=0,
            faces=[],
            error=str(exc),
        )


def detect_emotions_batch(
    sources: list[str | np.ndarray],
    preprocess: bool = True,
    enforce_detection: bool = False,
) -> list[EmotionDetectionResult]:
    """Detect emotions in a list of images.

    Parameters
    ----------
    sources:
        List of file paths and/or BGR NumPy arrays.
    preprocess:
        Passed through to :func:`detect_emotions`.
    enforce_detection:
        Passed through to :func:`detect_emotions`.

    Returns
    -------
    list of EmotionDetectionResult
        One result per input source, in the same order.
    """
    return [
        detect_emotions(src, preprocess=preprocess, enforce_detection=enforce_detection)
        for src in sources
    ]


def get_emotion_summary(results: list[EmotionDetectionResult]) -> dict[str, Any]:
    """Compute aggregate statistics from a list of detection results.

    Only successful results (``result.success is True``) are included.

    Parameters
    ----------
    results:
        Collection of :class:`EmotionDetectionResult` instances.

    Returns
    -------
    dict with keys:
        ``total_images``, ``analyzed_images``, ``total_faces``,
        ``average_emotions`` (dict of emotion → mean score),
        ``dominant_emotion_counts`` (dict of emotion → occurrence count),
        ``overall_dominant_emotion`` (most frequent dominant emotion).
    """
    successful = [r for r in results if r.success]
    if not successful:
        return {
            "total_images": len(results),
            "analyzed_images": 0,
            "total_faces": 0,
            "average_emotions": _zero_emotions(),
            "dominant_emotion_counts": {e: 0 for e in EMOTIONS},
            "overall_dominant_emotion": "neutral",
        }

    avg_emotions: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    dominant_counts: dict[str, int] = {e: 0 for e in EMOTIONS}
    total_faces = 0

    for result in successful:
        for emotion, score in result.emotions.items():
            avg_emotions[emotion] = avg_emotions.get(emotion, 0.0) + score
        dominant_counts[result.dominant_emotion] = (
            dominant_counts.get(result.dominant_emotion, 0) + 1
        )
        total_faces += result.face_count

    n = len(successful)
    avg_emotions = {e: v / n for e, v in avg_emotions.items()}
    overall_dominant = max(dominant_counts, key=dominant_counts.__getitem__)

    return {
        "total_images": len(results),
        "analyzed_images": n,
        "total_faces": total_faces,
        "average_emotions": avg_emotions,
        "dominant_emotion_counts": dominant_counts,
        "overall_dominant_emotion": overall_dominant,
    }
