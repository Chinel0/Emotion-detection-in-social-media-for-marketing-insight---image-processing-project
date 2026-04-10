# Emotion Detection in Social Media for Marketing Insight

An image-processing and computer vision project that analyses emotional
expressions in social media images and converts the results into actionable
marketing insights.

---

## Overview

This project explores how **image processing** and **computer vision**
techniques can be applied to analyse the emotional content of social-media
images.  The pipeline:

1. **Preprocesses** images (resize, enhance, face extraction via OpenCV
   Haar-cascade classifiers).
2. **Detects emotions** in each image using
   [DeepFace](https://github.com/serengil/deepface) (angry, disgust, fear,
   happy, sad, surprise, neutral).
3. **Aggregates** per-image results into batch statistics.
4. **Generates marketing insights** – sentiment, engagement potential,
   audience segment, content recommendations, and campaign strategy.

---

## Project Structure

```
.
├── image_preprocessor.py    # Image loading, resizing, enhancement, face detection
├── emotion_detector.py      # Core emotion detection (wraps DeepFace)
├── social_media_analyzer.py # Batch analysis pipeline for social-media images
├── marketing_insights.py    # Marketing insights & campaign strategy generator
├── requirements.txt         # Python dependencies
└── tests/
    └── test_emotion_detection.py  # Unit tests (66 tests)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow ≥ 2.16 requires the `tf-keras` compatibility shim:
> ```bash
> pip install tf-keras
> ```

---

## Quick Start

### Analyse a single image

```python
from social_media_analyzer import analyze_image
from marketing_insights import generate_insights, format_report

result = analyze_image("path/to/social_post.jpg")
print(result.dominant_emotion)          # e.g. "happy"
print(result.emotion_result.emotions)   # {"happy": 87.3, "neutral": 6.1, ...}
```

### Analyse a folder of images and generate a report

```python
from social_media_analyzer import analyze_directory
from marketing_insights import generate_insights, generate_campaign_strategy, format_report

report = analyze_directory("images/")

insights  = generate_insights(report.summary, brand_name="AcmeCo")
strategy  = generate_campaign_strategy(insights, campaign_goal="brand awareness")
print(format_report(insights, strategy))
```

Sample output:

```
============================================================
  EMOTION DETECTION – MARKETING INSIGHTS REPORT
============================================================

  Dominant Emotion     : Happy
  Overall Sentiment    : Positive
  Engagement Potential : High
  Audience Segment     : Satisfied customers / brand advocates

  NARRATIVE
  ---------
  Analysis of 10 social media image(s) (14 face(s) detected across 10 image(s))
  indicates a predominantly **happy** emotional tone for AcmeCo. ...

  EMOTION BREAKDOWN
  -----------------
  happy      72.4%  ██████████████
  neutral    12.1%  ██
  surprise    8.3%  █
  ...

  CONTENT RECOMMENDATIONS
  -----------------------
  1. Feature this content in brand highlight reels.
  2. Use happy imagery in ad creatives for maximum resonance.
  3. Encourage user-generated content campaigns to amplify positive sentiment.

  CAMPAIGN STRATEGY
  -----------------
  Goal          : brand awareness
  Tone          : celebratory and upbeat
  Risk Level    : Low
  ...
============================================================
```

---

## Module Reference

### `image_preprocessor`

| Function | Description |
|---|---|
| `load_image(source)` | Load from path or NumPy array |
| `resize_image(image, width, height)` | Resize with optional aspect-ratio preservation |
| `normalize_image(image)` | Scale pixels to [0, 1] |
| `convert_to_grayscale(image)` | BGR → grayscale |
| `enhance_image(image, ...)` | Brightness / contrast / sharpness enhancement |
| `detect_faces(image)` | Haar-cascade face detection |
| `extract_face_regions(image)` | Crop detected face regions |
| `preprocess_for_detection(image)` | Full pipeline (resize + enhance) |

### `emotion_detector`

| Symbol | Description |
|---|---|
| `EMOTIONS` | Tuple of supported emotion labels |
| `detect_emotions(source)` | Analyse a single image |
| `detect_emotions_batch(sources)` | Analyse multiple images |
| `get_emotion_summary(results)` | Aggregate statistics |
| `EmotionDetectionResult` | Result container (emotions, dominant, face count) |

### `social_media_analyzer`

| Symbol | Description |
|---|---|
| `analyze_image(source)` | Analyse one image with metadata support |
| `analyze_images(sources)` | Analyse a list of images |
| `analyze_directory(directory)` | Analyse all images in a folder |
| `filter_results_by_emotion(report, emotion, min_score)` | Filter by emotion threshold |
| `BatchAnalysisReport` | Aggregated report container |

### `marketing_insights`

| Function | Description |
|---|---|
| `emotion_to_sentiment(emotion)` | Map emotion → positive/neutral/negative |
| `get_content_recommendations(dominant)` | Return content strategy bullet points |
| `generate_insights(summary)` | Full insight dict from a summary |
| `generate_campaign_strategy(insights)` | Campaign strategy from insights |
| `format_report(insights, strategy)` | Human-readable formatted report |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

66 tests cover preprocessing utilities, emotion detection logic, batch
analysis, and the marketing insights generator.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python-headless` | Image I/O, face detection, preprocessing |
| `numpy` | Numerical array operations |
| `Pillow` | Image enhancement |
| `deepface` | Pre-trained emotion detection model |
| `tensorflow` | DeepFace backend |

---

## License

See [LICENSE](LICENSE).
