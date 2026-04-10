"""
Unit tests for the emotion detection pipeline.

Tests cover:
  - image_preprocessor    : loading, resizing, normalization, face detection
  - emotion_detector      : EmotionDetectionResult, get_emotion_summary
  - social_media_analyzer : BatchAnalysisReport, filter helpers
  - marketing_insights    : sentiment mapping, insight/strategy generation

DeepFace model inference is mocked so tests run without GPU/network access.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import cv2

# Ensure the project root is on the path regardless of how tests are invoked.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from image_preprocessor import (
    load_image,
    resize_image,
    normalize_image,
    convert_to_grayscale,
    enhance_image,
    detect_faces,
    extract_face_regions,
    preprocess_for_detection,
)
from emotion_detector import (
    EmotionDetectionResult,
    EMOTIONS,
    get_emotion_summary,
    _zero_emotions,
    _build_result_from_deepface,
    detect_emotions,
    detect_emotions_batch,
)
from social_media_analyzer import (
    ImageAnalysisResult,
    BatchAnalysisReport,
    analyze_image,
    analyze_images,
    analyze_directory,
    filter_results_by_emotion,
)
from marketing_insights import (
    emotion_to_sentiment,
    get_content_recommendations,
    generate_insights,
    generate_campaign_strategy,
    format_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bgr_image(h: int = 100, w: int = 100, channels: int = 3) -> np.ndarray:
    """Create a solid blue BGR image."""
    return np.full((h, w, channels), (255, 100, 50), dtype=np.uint8)


def _make_gray_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a solid gray image."""
    return np.full((h, w), 128, dtype=np.uint8)


def _make_emotion_result(
    dominant: str = "happy",
    face_count: int = 1,
    error: str | None = None,
) -> EmotionDetectionResult:
    emotions = {e: 0.0 for e in EMOTIONS}
    emotions[dominant] = 80.0
    return EmotionDetectionResult(
        emotions=emotions,
        dominant_emotion=dominant,
        face_count=face_count,
        faces=[],
        error=error,
    )


# ---------------------------------------------------------------------------
# image_preprocessor tests
# ---------------------------------------------------------------------------

class TestLoadImage(unittest.TestCase):
    def test_accepts_numpy_array(self):
        img = _make_bgr_image()
        result = load_image(img)
        np.testing.assert_array_equal(result, img)

    def test_copy_returned_for_array(self):
        img = _make_bgr_image()
        result = load_image(img)
        self.assertIsNot(result, img)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_image("/nonexistent/path/image.jpg")

    def test_loads_saved_png(self):
        img = _make_bgr_image(60, 60)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            cv2.imwrite(tmp_path, img)
            loaded = load_image(tmp_path)
            self.assertEqual(loaded.shape, img.shape)
        finally:
            os.unlink(tmp_path)


class TestResizeImage(unittest.TestCase):
    def test_default_resize(self):
        img = _make_bgr_image(50, 80)
        result = resize_image(img, width=32, height=32)
        self.assertEqual(result.shape[:2], (32, 32))

    def test_keep_aspect_ratio(self):
        img = _make_bgr_image(100, 200)
        result = resize_image(img, width=100, height=100, keep_aspect_ratio=True)
        self.assertEqual(result.shape[:2], (100, 100))


class TestNormalizeImage(unittest.TestCase):
    def test_output_range(self):
        img = _make_bgr_image()
        norm = normalize_image(img)
        self.assertGreaterEqual(norm.min(), 0.0)
        self.assertLessEqual(norm.max(), 1.0)

    def test_dtype(self):
        img = _make_bgr_image()
        self.assertEqual(normalize_image(img).dtype, np.float64)


class TestConvertToGrayscale(unittest.TestCase):
    def test_bgr_to_gray(self):
        img = _make_bgr_image()
        gray = convert_to_grayscale(img)
        self.assertEqual(gray.ndim, 2)

    def test_already_gray_unchanged(self):
        img = _make_gray_image()
        result = convert_to_grayscale(img)
        np.testing.assert_array_equal(result, img)


class TestEnhanceImage(unittest.TestCase):
    def test_returns_same_shape(self):
        img = _make_bgr_image()
        enhanced = enhance_image(img)
        self.assertEqual(enhanced.shape, img.shape)

    def test_identity_at_1(self):
        img = _make_bgr_image()
        result = enhance_image(img, brightness=1.0, contrast=1.0, sharpness=1.0)
        # Pillow round-trips introduce minor rounding; allow small tolerance
        self.assertTrue(np.allclose(result.astype(float), img.astype(float), atol=2))


class TestDetectFaces(unittest.TestCase):
    def test_no_face_in_plain_image(self):
        img = _make_bgr_image(100, 100)
        faces = detect_faces(img)
        self.assertIsInstance(faces, list)

    def test_returns_list_of_tuples(self):
        img = _make_bgr_image()
        faces = detect_faces(img)
        for face in faces:
            self.assertIsInstance(face, tuple)
            self.assertEqual(len(face), 4)


class TestExtractFaceRegions(unittest.TestCase):
    def test_no_faces_returns_empty(self):
        img = _make_bgr_image()
        regions = extract_face_regions(img)
        self.assertIsInstance(regions, list)


class TestPreprocessForDetection(unittest.TestCase):
    def test_output_size(self):
        img = _make_bgr_image(300, 400)
        processed = preprocess_for_detection(img, target_size=(224, 224))
        self.assertEqual(processed.shape[:2], (224, 224))

    def test_output_dtype(self):
        img = _make_bgr_image()
        processed = preprocess_for_detection(img)
        self.assertEqual(processed.dtype, np.uint8)


# ---------------------------------------------------------------------------
# emotion_detector tests
# ---------------------------------------------------------------------------

class TestEmotionDetectionResult(unittest.TestCase):
    def test_success_when_no_error(self):
        r = _make_emotion_result("happy")
        self.assertTrue(r.success)

    def test_failure_when_error_set(self):
        r = _make_emotion_result(error="some error")
        self.assertFalse(r.success)

    def test_to_dict_keys(self):
        r = _make_emotion_result("happy")
        d = r.to_dict()
        for key in ("emotions", "dominant_emotion", "face_count", "success", "error"):
            self.assertIn(key, d)

    def test_repr(self):
        r = _make_emotion_result("sad")
        self.assertIn("sad", repr(r))


class TestBuildResultFromDeepface(unittest.TestCase):
    def test_empty_faces(self):
        r = _build_result_from_deepface([])
        self.assertEqual(r.face_count, 0)
        self.assertEqual(r.dominant_emotion, "neutral")

    def test_single_face(self):
        raw = [{"emotion": {"happy": 70.0, "neutral": 20.0, "sad": 10.0,
                            "angry": 0.0, "disgust": 0.0, "fear": 0.0, "surprise": 0.0}}]
        r = _build_result_from_deepface(raw)
        self.assertEqual(r.dominant_emotion, "happy")
        self.assertAlmostEqual(r.emotions["happy"], 70.0)

    def test_two_faces_averaged(self):
        raw = [
            {"emotion": {"happy": 80.0, "neutral": 20.0, "sad": 0.0,
                         "angry": 0.0, "disgust": 0.0, "fear": 0.0, "surprise": 0.0}},
            {"emotion": {"happy": 60.0, "neutral": 40.0, "sad": 0.0,
                         "angry": 0.0, "disgust": 0.0, "fear": 0.0, "surprise": 0.0}},
        ]
        r = _build_result_from_deepface(raw)
        self.assertAlmostEqual(r.emotions["happy"], 70.0)
        self.assertEqual(r.face_count, 2)


class TestGetEmotionSummary(unittest.TestCase):
    def test_empty_results(self):
        summary = get_emotion_summary([])
        self.assertEqual(summary["analyzed_images"], 0)

    def test_all_failed(self):
        results = [_make_emotion_result(error="err") for _ in range(3)]
        summary = get_emotion_summary(results)
        self.assertEqual(summary["analyzed_images"], 0)

    def test_basic_summary(self):
        results = [
            _make_emotion_result("happy"),
            _make_emotion_result("happy"),
            _make_emotion_result("sad"),
        ]
        summary = get_emotion_summary(results)
        self.assertEqual(summary["analyzed_images"], 3)
        self.assertEqual(summary["overall_dominant_emotion"], "happy")
        self.assertEqual(summary["dominant_emotion_counts"]["happy"], 2)
        self.assertEqual(summary["dominant_emotion_counts"]["sad"], 1)

    def test_total_faces_counted(self):
        results = [_make_emotion_result("happy", face_count=2) for _ in range(3)]
        summary = get_emotion_summary(results)
        self.assertEqual(summary["total_faces"], 6)


class TestDetectEmotions(unittest.TestCase):
    def test_bad_path_returns_error_result(self):
        result = detect_emotions("/nonexistent/image.jpg")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    @patch("emotion_detector._import_deepface")
    def test_successful_detection(self, mock_import):
        mock_df = MagicMock()
        mock_df.analyze.return_value = [
            {"emotion": {e: (90.0 if e == "happy" else 0.0) for e in EMOTIONS}}
        ]
        mock_import.return_value = mock_df

        img = _make_bgr_image()
        result = detect_emotions(img)
        self.assertTrue(result.success)
        self.assertEqual(result.dominant_emotion, "happy")

    @patch("emotion_detector._import_deepface")
    def test_deepface_exception_handled(self, mock_import):
        mock_df = MagicMock()
        mock_df.analyze.side_effect = RuntimeError("model error")
        mock_import.return_value = mock_df

        img = _make_bgr_image()
        result = detect_emotions(img)
        self.assertFalse(result.success)
        self.assertIn("model error", result.error)


class TestDetectEmotionsBatch(unittest.TestCase):
    @patch("emotion_detector._import_deepface")
    def test_batch_length_matches(self, mock_import):
        mock_df = MagicMock()
        mock_df.analyze.return_value = [
            {"emotion": {e: 0.0 for e in EMOTIONS}}
        ]
        mock_import.return_value = mock_df

        images = [_make_bgr_image() for _ in range(4)]
        results = detect_emotions_batch(images)
        self.assertEqual(len(results), 4)


# ---------------------------------------------------------------------------
# social_media_analyzer tests
# ---------------------------------------------------------------------------

class TestImageAnalysisResult(unittest.TestCase):
    def test_dominant_emotion_property(self):
        er = _make_emotion_result("angry")
        iar = ImageAnalysisResult(image_path=None, emotion_result=er)
        self.assertEqual(iar.dominant_emotion, "angry")

    def test_success_property(self):
        er = _make_emotion_result()
        iar = ImageAnalysisResult(image_path=None, emotion_result=er)
        self.assertTrue(iar.success)

    def test_to_dict(self):
        er = _make_emotion_result("happy")
        iar = ImageAnalysisResult(image_path="/img.jpg", emotion_result=er, metadata={"id": 1})
        d = iar.to_dict()
        self.assertEqual(d["image_path"], "/img.jpg")
        self.assertIn("emotion_result", d)


class TestBatchAnalysisReport(unittest.TestCase):
    def _make_report(self):
        results = [
            ImageAnalysisResult(image_path=None, emotion_result=_make_emotion_result("happy")),
            ImageAnalysisResult(image_path=None, emotion_result=_make_emotion_result("sad")),
            ImageAnalysisResult(image_path=None, emotion_result=_make_emotion_result(error="err")),
        ]
        summary = get_emotion_summary([r.emotion_result for r in results])
        return BatchAnalysisReport(results=results, summary=summary)

    def test_total_images(self):
        self.assertEqual(self._make_report().total_images, 3)

    def test_successful_analyses(self):
        self.assertEqual(self._make_report().successful_analyses, 2)

    def test_overall_dominant(self):
        report = self._make_report()
        # happy and sad each appear once; result depends on tie-breaking
        self.assertIn(report.overall_dominant_emotion(), EMOTIONS)

    def test_emotion_distribution_keys(self):
        report = self._make_report()
        dist = report.emotion_distribution()
        for e in EMOTIONS:
            self.assertIn(e, dist)


class TestAnalyzeImage(unittest.TestCase):
    @patch("social_media_analyzer.detect_emotions")
    def test_returns_image_analysis_result(self, mock_detect):
        mock_detect.return_value = _make_emotion_result("happy")
        result = analyze_image(_make_bgr_image())
        self.assertIsInstance(result, ImageAnalysisResult)

    @patch("social_media_analyzer.detect_emotions")
    def test_metadata_attached(self, mock_detect):
        mock_detect.return_value = _make_emotion_result()
        result = analyze_image(_make_bgr_image(), metadata={"post_id": 99})
        self.assertEqual(result.metadata["post_id"], 99)


class TestAnalyzeImages(unittest.TestCase):
    @patch("social_media_analyzer.detect_emotions_batch")
    def test_batch_report(self, mock_batch):
        mock_batch.return_value = [_make_emotion_result("happy") for _ in range(3)]
        report = analyze_images([_make_bgr_image() for _ in range(3)])
        self.assertIsInstance(report, BatchAnalysisReport)
        self.assertEqual(report.total_images, 3)


class TestAnalyzeDirectory(unittest.TestCase):
    def test_not_a_directory_raises(self):
        with self.assertRaises(NotADirectoryError):
            analyze_directory("/nonexistent/path")

    @patch("social_media_analyzer.detect_emotions_batch")
    def test_empty_directory(self, mock_batch):
        mock_batch.return_value = []
        with tempfile.TemporaryDirectory() as tmpdir:
            report = analyze_directory(tmpdir)
        self.assertEqual(report.total_images, 0)

    @patch("social_media_analyzer.detect_emotions_batch")
    def test_discovers_images(self, mock_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                path = os.path.join(tmpdir, f"img_{i}.jpg")
                cv2.imwrite(path, _make_bgr_image())
            mock_batch.return_value = [_make_emotion_result() for _ in range(3)]
            report = analyze_directory(tmpdir)
        self.assertEqual(report.total_images, 3)


class TestFilterResultsByEmotion(unittest.TestCase):
    def _make_report(self):
        results = [
            ImageAnalysisResult(
                image_path=None,
                emotion_result=EmotionDetectionResult(
                    emotions={**{e: 0.0 for e in EMOTIONS}, "happy": 85.0},
                    dominant_emotion="happy", face_count=1, faces=[],
                ),
            ),
            ImageAnalysisResult(
                image_path=None,
                emotion_result=EmotionDetectionResult(
                    emotions={**{e: 0.0 for e in EMOTIONS}, "happy": 30.0},
                    dominant_emotion="neutral", face_count=1, faces=[],
                ),
            ),
        ]
        return BatchAnalysisReport(results=results, summary={})

    def test_filter_happy_high_threshold(self):
        report = self._make_report()
        filtered = filter_results_by_emotion(report, "happy", min_score=80.0)
        self.assertEqual(len(filtered), 1)

    def test_filter_happy_low_threshold(self):
        report = self._make_report()
        filtered = filter_results_by_emotion(report, "happy", min_score=0.0)
        self.assertEqual(len(filtered), 2)

    def test_invalid_emotion_raises(self):
        report = self._make_report()
        with self.assertRaises(ValueError):
            filter_results_by_emotion(report, "confusion")


# ---------------------------------------------------------------------------
# marketing_insights tests
# ---------------------------------------------------------------------------

class TestEmotionToSentiment(unittest.TestCase):
    def test_happy_positive(self):
        self.assertEqual(emotion_to_sentiment("happy"), "positive")

    def test_surprise_positive(self):
        self.assertEqual(emotion_to_sentiment("surprise"), "positive")

    def test_neutral(self):
        self.assertEqual(emotion_to_sentiment("neutral"), "neutral")

    def test_negative_emotions(self):
        for e in ("sad", "angry", "disgust", "fear"):
            self.assertEqual(emotion_to_sentiment(e), "negative", msg=e)


class TestGetContentRecommendations(unittest.TestCase):
    def test_returns_list(self):
        for e in EMOTIONS:
            recs = get_content_recommendations(e)
            self.assertIsInstance(recs, list)
            self.assertGreater(len(recs), 0)

    def test_unknown_emotion_fallback(self):
        recs = get_content_recommendations("unknown_emotion")
        self.assertIsInstance(recs, list)


class TestGenerateInsights(unittest.TestCase):
    def _make_summary(self, dominant: str = "happy") -> dict:
        counts = {e: 0 for e in EMOTIONS}
        counts[dominant] = 3
        return {
            "total_images": 5,
            "analyzed_images": 3,
            "total_faces": 4,
            "average_emotions": {**{e: 5.0 for e in EMOTIONS}, dominant: 70.0},
            "dominant_emotion_counts": counts,
            "overall_dominant_emotion": dominant,
        }

    def test_keys_present(self):
        insights = generate_insights(self._make_summary())
        for k in (
            "dominant_emotion", "sentiment", "engagement_potential",
            "audience_segment", "content_recommendations",
            "emotion_breakdown", "narrative",
        ):
            self.assertIn(k, insights)

    def test_dominant_emotion_matches(self):
        insights = generate_insights(self._make_summary("sad"))
        self.assertEqual(insights["dominant_emotion"], "sad")
        self.assertEqual(insights["sentiment"], "negative")

    def test_narrative_contains_brand(self):
        insights = generate_insights(self._make_summary(), brand_name="TestBrand")
        self.assertIn("TestBrand", insights["narrative"])

    def test_empty_summary_no_crash(self):
        insights = generate_insights({})
        self.assertIn("dominant_emotion", insights)


class TestGenerateCampaignStrategy(unittest.TestCase):
    def _make_insights(self, dominant: str = "happy") -> dict:
        return {
            "dominant_emotion": dominant,
            "sentiment": emotion_to_sentiment(dominant),
            "engagement_potential": "high",
            "content_recommendations": ["rec1", "rec2"],
        }

    def test_keys_present(self):
        strategy = generate_campaign_strategy(self._make_insights())
        for k in (
            "campaign_goal", "dominant_emotion", "sentiment",
            "recommended_tone", "content_types", "kpis",
            "risk_level", "action_items",
        ):
            self.assertIn(k, strategy)

    def test_risk_high_for_angry(self):
        strategy = generate_campaign_strategy(self._make_insights("angry"))
        self.assertEqual(strategy["risk_level"], "high")

    def test_risk_low_for_happy(self):
        strategy = generate_campaign_strategy(self._make_insights("happy"))
        self.assertEqual(strategy["risk_level"], "low")

    def test_campaign_goal_preserved(self):
        strategy = generate_campaign_strategy(
            self._make_insights(), campaign_goal="lead generation"
        )
        self.assertEqual(strategy["campaign_goal"], "lead generation")


class TestFormatReport(unittest.TestCase):
    def _make_insights(self) -> dict:
        return generate_insights(
            {
                "total_images": 5,
                "analyzed_images": 4,
                "total_faces": 6,
                "average_emotions": {e: 10.0 for e in EMOTIONS},
                "dominant_emotion_counts": {**{e: 0 for e in EMOTIONS}, "happy": 4},
                "overall_dominant_emotion": "happy",
            }
        )

    def test_returns_string(self):
        report = format_report(self._make_insights())
        self.assertIsInstance(report, str)

    def test_contains_section_headers(self):
        report = format_report(self._make_insights())
        self.assertIn("NARRATIVE", report)
        self.assertIn("EMOTION BREAKDOWN", report)
        self.assertIn("CONTENT RECOMMENDATIONS", report)

    def test_strategy_section_present_when_supplied(self):
        insights = self._make_insights()
        strategy = generate_campaign_strategy(insights)
        report = format_report(insights, strategy=strategy)
        self.assertIn("CAMPAIGN STRATEGY", report)

    def test_no_strategy_section_by_default(self):
        report = format_report(self._make_insights())
        self.assertNotIn("CAMPAIGN STRATEGY", report)


if __name__ == "__main__":
    unittest.main()
