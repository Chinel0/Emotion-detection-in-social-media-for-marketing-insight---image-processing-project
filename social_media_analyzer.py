"""
Social media image analyzer.

Provides a high-level pipeline for processing collections of social-media
images: loading, preprocessing, emotion detection, and per-image / aggregate
reporting.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import cv2

from image_preprocessor import load_image
from emotion_detector import (
    detect_emotions,
    detect_emotions_batch,
    get_emotion_summary,
    EmotionDetectionResult,
    EMOTIONS,
)

logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@dataclass
class ImageAnalysisResult:
    """Result for a single social-media image.

    Attributes
    ----------
    image_path : str or None
        Source path (``None`` for in-memory arrays).
    emotion_result : EmotionDetectionResult
        Emotion detection output.
    metadata : dict
        Arbitrary metadata supplied by the caller (e.g. post ID, timestamp).
    """

    image_path: str | None
    emotion_result: EmotionDetectionResult
    metadata: dict = field(default_factory=dict)

    @property
    def dominant_emotion(self) -> str:
        return self.emotion_result.dominant_emotion

    @property
    def success(self) -> bool:
        return self.emotion_result.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "dominant_emotion": self.dominant_emotion,
            "emotion_result": self.emotion_result.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class BatchAnalysisReport:
    """Aggregated report for a batch of social-media images.

    Attributes
    ----------
    results : list[ImageAnalysisResult]
        Per-image results.
    summary : dict
        Aggregate statistics from :func:`emotion_detector.get_emotion_summary`.
    """

    results: list[ImageAnalysisResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return len(self.results)

    @property
    def successful_analyses(self) -> int:
        return sum(1 for r in self.results if r.success)

    def emotion_distribution(self) -> dict[str, float]:
        """Return the average emotion scores across all successful analyses."""
        return self.summary.get("average_emotions", {e: 0.0 for e in EMOTIONS})

    def dominant_emotion_counts(self) -> dict[str, int]:
        """Return how many images had each emotion as dominant."""
        return self.summary.get("dominant_emotion_counts", {e: 0 for e in EMOTIONS})

    def overall_dominant_emotion(self) -> str:
        return self.summary.get("overall_dominant_emotion", "neutral")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "successful_analyses": self.successful_analyses,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }


def _collect_image_paths(directory: str) -> list[str]:
    """Return all image file paths found directly in *directory*."""
    paths = []
    for entry in sorted(os.listdir(directory)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            paths.append(os.path.join(directory, entry))
    return paths


def analyze_image(
    source: str | np.ndarray,
    metadata: dict | None = None,
) -> ImageAnalysisResult:
    """Analyze a single social-media image for emotional content.

    Parameters
    ----------
    source:
        File path or BGR NumPy array.
    metadata:
        Optional metadata to attach to the result (e.g. ``{"post_id": 42}``).

    Returns
    -------
    ImageAnalysisResult
    """
    path = source if isinstance(source, str) else None
    emotion_result = detect_emotions(source)
    return ImageAnalysisResult(
        image_path=path,
        emotion_result=emotion_result,
        metadata=metadata or {},
    )


def analyze_images(
    sources: list[str | np.ndarray],
    metadata_list: list[dict] | None = None,
) -> BatchAnalysisReport:
    """Analyze a list of social-media images.

    Parameters
    ----------
    sources:
        List of file paths and/or BGR NumPy arrays.
    metadata_list:
        Optional list of metadata dicts aligned with *sources*.

    Returns
    -------
    BatchAnalysisReport
    """
    if metadata_list is None:
        metadata_list = [{} for _ in sources]

    emotion_results = detect_emotions_batch(sources)
    results = [
        ImageAnalysisResult(
            image_path=src if isinstance(src, str) else None,
            emotion_result=er,
            metadata=meta,
        )
        for src, er, meta in zip(sources, emotion_results, metadata_list)
    ]

    summary = get_emotion_summary([r.emotion_result for r in results])
    return BatchAnalysisReport(results=results, summary=summary)


def analyze_directory(
    directory: str,
    recursive: bool = False,
) -> BatchAnalysisReport:
    """Analyze all images in a directory.

    Parameters
    ----------
    directory:
        Path to the folder containing images.
    recursive:
        When *True* also search sub-directories.

    Returns
    -------
    BatchAnalysisReport

    Raises
    ------
    NotADirectoryError
        If *directory* does not exist or is not a directory.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    if recursive:
        paths: list[str] = []
        for root, _dirs, files in os.walk(directory):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS:
                    paths.append(os.path.join(root, f))
    else:
        paths = _collect_image_paths(directory)

    if not paths:
        logger.warning("No images found in directory: %s", directory)

    metadata_list = [{"image_path": p} for p in paths]
    return analyze_images(paths, metadata_list)


def filter_results_by_emotion(
    report: BatchAnalysisReport,
    emotion: str,
    min_score: float = 0.0,
) -> list[ImageAnalysisResult]:
    """Filter analysis results by a specific emotion threshold.

    Parameters
    ----------
    report:
        A :class:`BatchAnalysisReport` to filter.
    emotion:
        Emotion label (e.g. ``"happy"``).
    min_score:
        Minimum score [0–100] for the emotion.

    Returns
    -------
    list[ImageAnalysisResult]
        Subset of results where the emotion score ≥ *min_score*.
    """
    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion {emotion!r}. Choose from: {EMOTIONS}")
    return [
        r for r in report.results
        if r.success and r.emotion_result.emotions.get(emotion, 0.0) >= min_score
    ]
