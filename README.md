# Emotion Detection in Social Media Images for Marketing Insight

### An Image Processing and Computer Vision Project
**Author:** Chinelo Lydia Nweke

---

## 1. Overview

This project investigates how computer vision and deep learning can be used to detect emotional expressions in facial images, and how those detections can be aggregated into insights that are useful for marketing analytics. Rather than relying on engagement metrics alone (likes, shares, comments), the project asks whether the *emotional content* of an image — as perceived through a viewer's face, or as conveyed by the people depicted in marketing material — can be measured systematically and turned into a quantifiable signal.

The project is built as an end-to-end pipeline: it starts from a public, labeled facial-expression dataset, performs systematic data cleaning and exploratory data analysis (EDA), trains a custom convolutional neural network (CNN) to classify emotions, evaluates that model rigorously, and finally wraps it in a real-world inference pipeline that can take arbitrary social-media-style photographs, detect faces in them, classify the emotion expressed, and summarize the results as a marketing-relevant insight.

## 2. Research Question and Hypothesis

**Research Question:**
> How can image processing and computer vision techniques be used to detect emotional expressions in social media images, and how can this information support marketing analytics and content evaluation?

**Hypothesis:**
> A CNN-based emotion classifier trained on the FER2013 dataset will produce systematically different emotion distributions between owned-brand content creators (Ballerina Farm) and sponsored-influencer content (Nara Smith), with owned-brand posts showing higher proportions of happy expressions, because product-focused content is designed to evoke active positivity, whereas sponsored lifestyle content defaults to a composed neutral aesthetic. However, the model's performance will be constrained by domain shift between the laboratory-conditioned training data and real-world Instagram imagery, providing a natural boundary for conclusions and directions for further work.

## 3. Objectives

- Detect human faces in arbitrary, real-world images.
- Classify the emotional expression of each detected face into one of three marketing-relevant categories: **happy**, **neutral**, or **sad**.
- Analyze the resulting distribution of emotions across a batch of images.
- Translate that distribution into a data-driven takeaway that a marketing team could act on (for example, whether a campaign's visuals are predominantly evoking positive, neutral, or negative affect).

## 4. Dataset

**Source:** FER2013 (Facial Expression Recognition 2013), obtained from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

**Original structure:** FER2013 ships as pre-split `train/` and `test/` directories, each containing seven class subfolders — `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. Every image is a pre-cropped, **48×48 grayscale** facial close-up.

**Known limitation of the dataset:** FER2013 is a widely used benchmark, but it is also widely documented to contain label noise — independent estimates put *human* labeling agreement on the dataset at roughly **65–68%**. This means a portion of the "ground truth" labels are themselves debatable even to a human rater. This is an important caveat to carry into any discussion of model accuracy: a model trained on FER2013 cannot realistically be expected to exceed the consistency of the humans who labeled it, and any confusion-matrix errors should be interpreted with this ceiling in mind.

**Why the project narrows down to three emotions:** Although FER2013 provides seven emotion classes, the project deliberately restricts its final classification target to **happy, neutral, and sad**. The justification is twofold:
1. *Marketing relevance* — a marketing analyst is primarily interested in whether an audience's reaction to visual content skews positive, neutral, or negative. Finer-grained categories such as "fear", "disgust", or "surprise" are far less common in everyday social-media photography and are harder to act on from a campaign-strategy standpoint.
2. *Class viability* — as the EDA below shows, some of the seven original classes (notably `disgust`) are extremely under-represented, which would make a reliable seven-way classifier difficult to train and evaluate without heavy oversampling or augmentation that risks distorting the data.

## 5. Phase 1 — Dataset Loading, Cleaning, and Exploratory Data Analysis (EDA)

This phase establishes a trustworthy foundation for everything downstream: confirming the data is what it claims to be, removing defects, and characterizing its statistical properties so that later preprocessing decisions (image size, normalization constants, class weighting) are evidence-based rather than guessed.

### 5.1 Class counts (original 7-class structure)
The raw FER2013 `train/` and `test/` directories were enumerated and counted per class to confirm the dataset's well-known imbalance — `happy` is the largest class, `disgust` is dramatically smaller than every other class, and the remaining five classes sit at broadly comparable, moderate sizes. This imbalance directly motivated the later decision to compute and apply class weights during training.

### 5.2 Corrupted / unreadable image check
Every image in the dataset was opened and validated to detect files that were truncated, unreadable, or otherwise corrupted. **No corrupted images were found** — the dataset's files all loaded successfully, so no removal was necessary at this step.

### 5.3 Image size and aspect-ratio distribution
A sample of images was inspected to confirm their dimensions. This confirmed that **every image is exactly 48×48 pixels**, i.e. perfectly square with a 1:1 aspect ratio and no inconsistencies — which is expected for FER2013, but is exactly the kind of assumption that should be verified rather than taken on faith before it is baked into a preprocessing pipeline.

### 5.4 Per-channel mean and standard deviation
Pixel-intensity statistics were computed across a representative sample of the training set (after normalizing pixel values to the [0, 1] range). The resulting statistics were:

- **Mean ≈ 0.5077**
- **Standard deviation ≈ 0.2551**

These two constants are not cosmetic — they are used directly in Phase 2 as the normalization parameters (`(pixel / 255.0 - MEAN) / STD`) applied to every image that flows through the training, validation, test, and inference pipelines. Computing them empirically (rather than assuming the conventional `0.5 / 0.5` placeholders) keeps the normalization faithful to the actual data distribution.

### 5.5 Sample grid per class (visual spot-check)
A grid of sample images was displayed for each of the seven classes. This qualitative check served two purposes: (1) confirming that the class-folder labels matched what a human would intuitively call that emotion, and (2) building familiarity with the kind of low-resolution, tightly-cropped, grayscale imagery the model would be trained on — a useful mental anchor for later understanding *why* the model behaves differently on full-color, full-scene social media photographs.

### 5.6 Exact-duplicate detection and removal (within-split)
Every image file was hashed (MD5) and compared against every other file in the same split to find byte-for-byte duplicates — a known issue in FER2013 caused by the way the dataset was scraped and assembled.

- **Duplicates detected in the training set: 1,236**
- **Duplicates removed: 1,236**
- **Training images remaining after deduplication: 27,473** (down from an original 28,709 raw training images)

Removing exact duplicates matters because duplicated samples can silently inflate a class's effective weight and can leak between training and validation splits, both of which would make reported performance look better than it actually is.

### 5.7 Train/test overlap detection and removal (cross-split leakage check)
A second, more critical hashing pass compared every test-set image against every training-set image to detect cross-split duplicates — i.e., images that appear in *both* the training and the held-out test set. This is one of the most serious threats to a trustworthy evaluation: if the model has effectively "seen" a test image during training, its test-set accuracy is no longer a fair estimate of generalization.

- An initial cross-split comparison flagged **531** training/test hash collisions.
- After re-hashing against the cleaned (post-deduplication) training set, **568 overlapping images were removed from the test set**.
- A final confirmation pass verified that **zero overlap remains** between the training and test sets.

This step is one of the most methodologically important parts of the project: it guarantees that the test-set metrics reported in Phase 6 reflect genuine generalization rather than memorization.

### 5.8 Manifest creation
A CSV manifest recording the file path, split (`train`/`test`), and class label for every surviving image was generated. This manifest acts as the canonical, de-duplicated, leak-free index that all downstream phases (pipeline construction, class-weight computation, training) read from — ensuring that every phase of the project operates on an identical, reproducible view of the data.

### 5.9 Class-weight computation (7-class baseline)
Using `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", ...)`, per-class weights were computed from the cleaned training manifest so that under-represented classes contribute proportionally more to the loss during training. The resulting seven-class weights were approximately:

| Class | Weight |
|---|---|
| angry | 1.02 |
| disgust | 10.30 |
| fear | 1.01 |
| happy | 0.55 |
| neutral | 0.81 |
| sad | 0.83 |
| surprise | 1.47 |

The dramatically higher weight for `disgust` (≈10×) is a direct, quantitative reflection of how severely under-represented that class is — and is itself further justification for excluding it (along with the other rarely-marketing-relevant classes) from the final 3-class target.

### 5.10 Face-detection coverage check
As a sanity check relevant to Phase 7 (where faces must be detected in *uncropped* real-world photos), OpenCV's Haar cascade face detector was run over a sample of the FER2013 images themselves to roughly measure what proportion contain a detectable face. This served as an early indicator of how reliably Haar cascades behave on this style of imagery, foreshadowing the detector-tuning work that would later prove necessary in Phase 7.

### 5.11 Final target selection
Based on the class-count imbalance (5.1), the disgust class's extreme rarity and resulting weight (5.9), and the marketing-relevance argument outlined in Section 4, the project formally narrows its classification target to three classes: **happy, neutral, sad**. All subsequent phases — pipeline construction, augmentation, model architecture, training, and evaluation — operate on this 3-class problem.

### 5.12 Additional visual and statistical analysis
Beyond the checks above, the EDA phase also produced: a bar-chart visualization of class distribution (both before and after narrowing to 3 classes), an additional sample-image display for the final target classes, a focused image-property analysis (confirming consistency of size, channel count, and intensity range across the cleaned dataset), and a written summary consolidating these findings for reporting purposes — effectively the same evidence base this README section is now distilling.

## 6. Phase 2 — Preprocessing and Data Pipeline

This phase converts the cleaned, analyzed dataset into a live training/validation/test pipeline. Concretely, it:

- Resizes every image to **48×48** (already the native size, but enforced explicitly for robustness against any future dataset substitutions).
- Converts every image to **grayscale** (matching FER2013's native format and keeping the input channel count consistent with the EDA-derived normalization statistics).
- **Normalizes** every pixel using the empirically computed constants from Section 5.4 — `MEAN ≈ 0.5077`, `STD ≈ 0.2551` — via a custom `preprocessing_function` passed into Keras's `ImageDataGenerator`.
- Restricts the generator to the **three target emotions** (`happy`, `neutral`, `sad`) identified in Section 5.11, by pointing it only at those class subfolders.
- Builds three separate generators — **train**, **validation**, and **test** — using `ImageDataGenerator(validation_split=...)` to carve a validation subset directly out of the training directory, while the held-out (and now leak-free, per Section 5.7) test directory remains completely untouched until final evaluation.
- **Recomputes class weights for just the 3 target classes**, since the 7-class weights from Section 5.9 are no longer applicable once `disgust`, `fear`, `angry`, and `surprise` are excluded. The 3-class weights came out only mildly imbalanced — confirming that `happy`, `neutral`, and `sad` are reasonably comparable in size relative to one another, unlike the original 7-class split.

## 7. Phase 3 — Data Augmentation

Augmentation is folded directly into the training generator built in Phase 2 — the standard approach when using `ImageDataGenerator`, since it applies transformations on-the-fly to each training batch and never touches the validation or test generators (preserving the integrity of those evaluation sets). The augmentations used are deliberately conservative, chosen to reflect realistic variation in how a face might appear in a photo without distorting the emotional signal itself:

- `rotation_range = 10` — small random rotations (±10°), simulating a slightly tilted head or camera angle.
- `zoom_range = 0.1` — slight random zoom in/out, simulating differences in framing and distance from the camera.
- `horizontal_flip = True` — random horizontal mirroring, on the reasoning that a mirrored face still conveys the same emotion.

Because the 3-class imbalance identified in Phase 2 turned out to be mild, augmentation here is used primarily as a **robustness/regularization** measure — helping the model generalize across small variations in framing, lighting, and orientation — rather than as a corrective tool for severe class skew (which would call for more aggressive oversampling techniques).

## 8. Phase 4 — Model Architecture

**Decision point — custom CNN vs. transfer learning:** The project deliberately builds a custom CNN from scratch rather than fine-tuning a pretrained network (e.g., VGG, ResNet, MobileNet). The reasoning: FER2013 images are 48×48 grayscale, while essentially all popular pretrained backbones expect roughly 224×224 RGB input. Adapting a pretrained network to this dataset would require upscaling small grayscale images by nearly 5× in each spatial dimension and tiling the single grayscale channel into three — adding substantial preprocessing complexity and computational cost while offering little obvious benefit, since the low-level visual features these networks learned from ImageNet (natural-scene textures, colors, object parts) have limited relevance to small grayscale facial close-ups. A compact, purpose-built CNN is therefore the more defensible choice, and is small enough to train in a reasonable time even on CPU hardware.

**Architecture summary:** the network consists of three sequential convolutional blocks of increasing depth, followed by a dense classification head:

- **Block 1:** `Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPooling2D → Dropout`
- **Block 2:** `Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPooling2D → Dropout`
- **Block 3:** `Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPooling2D → Dropout`
- **Classifier head:** `Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(3, activation="softmax")`

Design rationale for each component:
- The **doubled convolution per block** (`Conv → BN → Conv → BN`) lets the network build richer local feature representations before each spatial downsampling step.
- **Batch normalization** after every convolution stabilizes and accelerates training by normalizing intermediate activations.
- **MaxPooling** progressively reduces spatial resolution, letting deeper layers operate on increasingly abstract, larger-receptive-field features.
- **Dropout** at multiple stages (within each block and in the dense head, at a relatively high 0.5 rate before the output layer) combats overfitting — an important safeguard given the relatively small, low-resolution input and the noted label-noise ceiling of the dataset.
- The **increasing filter counts** (32 → 64 → 128) follow the standard CNN convention of trading spatial resolution for representational depth as the network goes deeper.
- The final **3-unit softmax** layer directly reflects the narrowed-down classification target (`happy`, `neutral`, `sad`).

## 9. Phase 5 — Training

The model is compiled with:
- **Optimizer:** Adam (learning rate = 1e-3)
- **Loss function:** categorical cross-entropy (the standard choice for multi-class, single-label softmax classification)
- **Metric:** accuracy

Three callbacks are attached to make training both efficient and robust:
- **EarlyStopping** — monitors validation loss and halts training once it stops improving, restoring the best-performing weights. This prevents both wasted computation and overfitting from training too long.
- **ModelCheckpoint** — saves the best-performing model to disk during training (based on validation accuracy), guaranteeing that the best version of the model — not simply the last — is preserved regardless of what happens in later epochs.
- **ReduceLROnPlateau** — reduces the learning rate when validation loss plateaus, allowing the optimizer to take smaller, more precise steps as training approaches convergence.

Training was configured for up to **50 epochs**, with the explicit expectation (documented in the notebook itself) that `EarlyStopping` would very likely halt training before all 50 epochs completed. The **class weights recomputed for the 3-class problem in Phase 2** were passed into `model.fit(...)`, ensuring the loss function accounts for the (mild) remaining class imbalance. Two saved artifacts result from this phase: the best checkpoint (selected by validation accuracy during training) and the final model state at the point training stopped.

## 10. Phase 6 — Evaluation

Evaluation is performed exclusively on the **held-out test set** — the same set that was confirmed in Section 5.7 to have zero overlap with the training data, ensuring the resulting metrics are a fair measure of generalization rather than memorization. The evaluation methodology comprises four complementary analyses:

1. **Accuracy / loss curves** — plotting training vs. validation accuracy and loss across epochs to visually assess whether the model converged, and whether (and at what point) it began to overfit.
2. **Confusion matrix** — a 3×3 matrix (for `happy` / `neutral` / `sad`) showing exactly which emotions the model tends to confuse with one another. This is particularly informative given the documented ~65–68% human-labeling-agreement ceiling of FER2013: some of the "errors" a confusion matrix surfaces may reflect genuinely ambiguous source images rather than model weakness.
3. **Classification report** — precision, recall, and F1-score computed per class, offering a more granular view than overall accuracy alone (which can be misleading under class imbalance).
4. **Error analysis** — a direct visual inspection of a sample of misclassified test images, paired with their true and predicted labels. This qualitative step is often the most revealing: it can expose whether errors cluster around genuinely ambiguous expressions, low-quality source images, or a systematic model bias toward a particular class.

**Results from the executed notebook run:**

Training halted at **epoch 37** via `EarlyStopping`, with best weights restored from **epoch 29**. The model was evaluated on 4,109 held-out test images.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| happy | 0.910 | 0.882 | 0.896 | 1,717 |
| neutral | 0.619 | 0.772 | 0.687 | 1,189 |
| sad | 0.737 | 0.589 | 0.654 | 1,203 |
| **overall accuracy** | | | **0.764** | **4,109** |

**Misclassified: 968 / 4,109 images (23.6%)**

Key observations from these metrics:
- **Happy** is the most reliably classified class (F1 = 0.896), with both precision and recall comfortably above 0.88 — reflecting the dataset's strong happy-class representation and the relative visual distinctiveness of smiling expressions.
- **Neutral** shows high recall (0.772) but lower precision (0.619), meaning the model over-predicts neutral — faces from other classes are frequently pulled toward the neutral category, consistent with neutral being the "default" prediction under ambiguity.
- **Sad** has acceptable precision (0.737) but noticeably lower recall (0.589), meaning many genuinely sad faces are misclassified, most likely as neutral — which makes intuitive sense given how subtle the visual difference between a sad and a composed-neutral expression can be in low-resolution grayscale images.
- The 76.4% overall accuracy sits meaningfully above random chance (33.3% for 3 classes), and is broadly consistent with the documented ~65–68% *human* labeling-agreement ceiling for FER2013, leaving relatively limited headroom for further improvement without addressing the underlying data-quality constraint.

## 11. Phase 7 — Face Detection and Emotion Inference Pipeline

This phase is where the *"social media images"* half of the research question is addressed directly. FER2013 supplies the model with pre-cropped, perfectly centered, 48×48 grayscale faces — but a real social media photograph is full-color, full-scene, arbitrarily sized, and may contain anything from zero to many faces at varying positions and scales. The inference pipeline therefore has to perform every preprocessing step that the FER2013 dataset curators had already done for us, automatically, on raw input:

```
load image → detect face (Haar cascade) → crop → resize to 48×48 → grayscale → normalize → predict emotion
```

**Demo image set:** A small batch of real, social-media-style photographs (clearly showing faces) was collected and placed in a `demo_images/` directory located one level above the project folder (resolved by the notebook as `../demo_images/`). The final demo batch used for this phase and Phase 8 contained **29 images**.

**Decision point — Haar cascades vs. a deep-learning face detector:** OpenCV's Haar cascade classifier (`detectMultiScale`) was chosen for face detection because it is lightweight, requires no additional model downloads or GPU acceleration, and integrates natively with OpenCV — keeping the pipeline simple and fully reproducible on commodity hardware. The trade-off, documented honestly below, is that Haar cascades are known to be more prone to false positives than modern deep-learning-based detectors (e.g., MTCNN, RetinaFace), particularly on higher-resolution, visually busy images.

**Detector tuning — a worked example of iterative refinement:** During development, the pipeline initially produced far more predictions than there were images (e.g., 205 predicted "faces" across only 29 images), with seemingly contradictory emotion labels for the same photograph. Investigation revealed the cause: the default `detectMultiScale` parameters (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(40, 40)`) were triggering on non-face regions — textures, shadows, and background patterns that Haar cascades can mistake for faces, especially in higher-resolution real-world photos very different from FER2013's tightly-cropped training imagery. This was resolved in two complementary steps:

1. **Stricter detector parameters** — increasing `minNeighbors` to `8` and `minSize` to `(80, 80)` made the detector substantially more conservative, reducing the false-positive count from 205 down to 57 detections across the 29 images.
2. **A "largest detected face" heuristic** — for single-portrait images (the expected case for this kind of demo set), the genuine face is virtually always the largest detected region; any remaining smaller false-positive detections in the same image can therefore be safely discarded by keeping only the bounding box with the greatest area (`w * h`). Applying this heuristic brought the result down to a clean **29 predictions for 29 images** — exactly one face per photo, as expected.

It is worth noting explicitly (as this caused some initial concern during development) that **all of this tuning happens entirely at the detection stage and requires zero retraining of the CNN** — `detectMultiScale`'s parameters and the largest-face heuristic operate purely on *where* a face is found in the raw image, completely independently of the already-trained emotion-classification model that runs afterward on the cropped result.

The finalized per-image pipeline:
1. Load the raw image (color, arbitrary resolution).
2. Run the tuned Haar cascade detector (`minNeighbors=8`, `minSize=(80, 80)`) to obtain candidate face bounding boxes.
3. Keep only the largest candidate box (by pixel area).
4. Crop the image to that bounding box.
5. Resize the crop to 48×48.
6. Convert to grayscale.
7. Normalize using the same `MEAN ≈ 0.5077` / `STD ≈ 0.2551` constants established in Section 5.4 and used consistently throughout training (Section 6).
8. Feed the processed crop into the trained CNN and obtain the predicted emotion label and confidence score.
9. Annotate the original image with the detected bounding box and predicted label for visual verification.

## 12. Phase 8 — Marketing Insights

This final phase closes the loop on the research question by running the complete pipeline from Phase 7 over the full demo batch, aggregating every prediction into a structured table (`df_insights`, containing the image filename, predicted emotion, and confidence score for each detected face), and translating the resulting distribution into a marketing-relevant interpretation.

**Demonstrated results (29-image demo batch):**

| Predicted emotion | Count |
|---|---|
| Neutral | 17 |
| Sad | 8 |
| Happy | 4 |

**Interpretation:** The demo batch's emotional distribution skews heavily toward **neutral**, with sad expressions outnumbering happy ones. From a marketing-analytics standpoint, a result like this would typically be read as a signal worth investigating further — for example, prompting questions about whether the visual content being analyzed is genuinely failing to evoke positive affect in its subjects/viewers, or whether the skew is partly an artifact of (a) the composition of the particular demo set used, (b) the previously discussed ~65–68% labeling-agreement ceiling inherited from FER2013's training labels, or (c) a "neutral bias" — a tendency of models trained on datasets like FER2013 to default to the majority-leaning, least-committal class (`neutral`) when an expression is subtle, ambiguous, or not strongly emotive, rather than the mistake reflecting a true absence of positive sentiment in the subject. Distinguishing between these explanations — genuine signal vs. dataset/model artifact — is precisely the kind of question that a larger-scale follow-up study (see Section 14) would be designed to answer.

## 13. Domain Shift: A Key Methodological Consideration

A recurring theme across Phases 7 and 8 — and one worth foregrounding explicitly in any write-up — is the **domain shift** between the curated training distribution (FER2013: small, grayscale, pre-cropped, front-facing, studio-style facial close-ups) and the real-world target distribution (social media photographs: full-color, arbitrary resolution and composition, variable lighting and pose, often containing multiple subjects or cluttered backgrounds). Every design decision in Phase 7 — from the choice of a robust-but-imperfect face detector, to its iterative tuning, to the empirical normalization constants carried over from Phase 1 — exists specifically to bridge this gap as faithfully as possible. Framing the project's results with this domain shift in mind is essential: any drop in apparent reliability when moving from FER2013's test set (Phase 6) to real social media images (Phases 7–8) is not necessarily a flaw in the model itself, but an expected and well-documented consequence of applying a model trained on one visual domain to a meaningfully different one.

## 14. Phase 9 — Real-World Brand Case Study: Ballerina Farm vs. Nara Smith

To directly address the second half of the research question — *"how can this support marketing analytics?"* — the inference pipeline developed in Phases 7 and 8 was applied to a curated set of real Instagram posts from two brands operating in the same niche (**homesteading lifestyle**) but with structurally different content strategies.

### 14.1 Brand Selection Rationale

| Brand | Instagram handle | Content strategy |
|---|---|---|
| Ballerina Farm | @ballerinafarmstore | Owned product brand — sells and promotes its own farm produce, protein powder, and food products |
| Nara Smith | @naraaziza | Sponsored-influencer — promotes third-party brands (Calvin Klein, Skims, Miu Miu, Burberry, H&M) |

This contrast was selected because the "owned vs. sponsored" axis represents a structurally meaningful difference in how emotion is likely to be deployed visually: a brand that owns its product has a strong commercial incentive to portray joyful, aspirational imagery, whereas a lifestyle influencer promoting multiple external brands tends to project a composed, editorial aesthetic rather than overt positivity.

### 14.2 Data Collection

- **26 unique posts** were collected in total: approximately 13 per brand (after deduplication — several posts were captured as multiple screenshot frames and reconciled using a `post_group` deduplication key to prevent double-counting engagement).
- Screenshots were taken directly from Instagram, capturing the post image alongside publicly visible engagement metrics (likes, comments, reposts, sends/saves).
- Faces were cropped from each post using the same tuned Haar cascade pipeline from Phase 7 (`minNeighbors=8`, `minSize=(80, 80)`, keep-largest-face heuristic), with a 60% bounding-box margin expansion for natural portrait framing.
- 3 images that failed automated face detection (non-frontal angles, eyes closed, face partially obscured) were cropped manually and added to the dataset, consistent with the Phase 7 limitation discussion.
- Engagement data was compiled into a structured CSV (`engagement_data.csv`) with a `post_group` field to handle duplicate crops from the same underlying post.

### 14.3 Inference and Emotion Results

Each cropped face image was passed through the trained CNN (Phase 5/6 model). The raw predicted class indices were mapped to the correct 3-class labels (`happy`, `neutral`, `sad`). Results per unique post, after deduplication:

| Brand | Happy | Neutral | No face detected |
|---|---|---|---|
| Ballerina Farm | 44% | 56% | 3 posts |
| Nara Smith | 0% | 100% | 2 posts |

Notable observations:
- All four of Ballerina Farm's "happy" predictions came from **product-focused posts** (Farmer Hydrate relaunch, colostrum product shot, Chef JR collaboration) — content where the brand controls the visual tone and has a clear commercial incentive to evoke positivity.
- Ballerina Farm's lifestyle/people posts (ballerinas, couples, guest appearances) predicted **neutral** — the same register as all of Nara Smith's content.
- Nara Smith's feed predicted uniformly neutral across every detected face, consistent with the composed, high-fashion aesthetic common to sponsored lifestyle content.
- Ballerina Farm's two highest-engagement posts (491,000 and 107,000 likes) had **no face detected** — suggesting their viral content is scenery- or product-driven rather than portrait-driven, a notable content-strategy finding in itself.

### 14.4 Engagement Comparison (Median, Unique Posts)

| Metric | Ballerina Farm | Nara Smith |
|---|---|---|
| Likes (median) | 2,281 | 772 |
| Comments (median) | 46 | 241 |
| Reposts (median) | 6 | 288 |
| Sends (median) | 81 | N/A (not visible) |

Ballerina Farm achieves higher passive engagement (likes/saves), while Nara Smith drives substantially more active engagement (comments, reposts). This divergence aligns with the emotional register difference: posts that feel aspirational and warm tend to be saved/liked, while composed editorial posts invite more sharing and discussion.

### 14.5 Domain Shift Revisited

Under domain shift, the model maps nearly all real-world social media faces to `neutral` — which is expected and consistent with Phase 13's discussion. The analytically useful signal is therefore not the absolute label but the **relative difference between brands**: Ballerina Farm's product content registers as happy at a meaningful rate (44%) while Nara Smith's does not (0%). This brand-level differentiation persists even under the domain-shift constraint, which is the primary justification for treating the result as interpretable rather than noise.

### 14.6 Output Files

| File | Contents |
|---|---|
| `Homestead_brand/engagement_data.csv` | 38 rows (all files), engagement metrics, `post_group` deduplication key |
| `Homestead_brand/predictions_all.csv` | Full merged table of all crops with predicted emotions |
| `Homestead_brand/predictions_unique_posts.csv` | Deduplicated, one row per post — the analysis-ready table |
| `Homestead_brand/brand_comparison.png` | Two-panel chart: emotion distribution by brand + engagement metrics by brand |

## 15. Suggested Further Work

The brand case study in Phase 9 establishes a proof-of-concept pipeline but operates at a small scale (26 unique posts). Meaningful extensions would include:

- **Larger sample**: 50–100 posts per brand to support statistically robust comparisons and significance testing.
- **Correlation analysis**: With a larger dataset, compute Pearson or Spearman correlations between predicted emotion and individual engagement metrics to test whether happiness in a post's face predicts higher likes or saves.
- **Deep-learning face detector**: Replacing the Haar cascade with a modern detector (e.g., MTCNN or RetinaFace) would substantially reduce the number of missed faces, particularly on non-frontal or partially-obscured angles that currently return null predictions.
- **Domain adaptation**: Fine-tuning the CNN on a small labeled set of social media facial images would reduce the domain-shift bias identified in Phase 13 and Phase 9, potentially unlocking more precise per-image emotion labels rather than the coarse brand-level patterns currently observed.
- **Multi-brand, multi-niche study**: Extending beyond two brands and one niche (homesteading) to test whether the owned-vs-sponsored emotional register difference generalizes across other content categories (beauty, fitness, food, fashion).

## 16. Project Workflow Summary

```
Phase 1  Dataset loading, cleaning (corruption check, dedup, leakage removal), and EDA
            |
Phase 2  Preprocessing & data pipeline (resize, grayscale, normalize, 3-class generators)
            |
Phase 3  Data augmentation (rotation, zoom, horizontal flip — applied on-the-fly to training data)
            |
Phase 4  Model architecture (custom 3-block CNN with batch norm and dropout)
            |
Phase 5  Training (Adam optimizer, categorical cross-entropy, class weights, callbacks)
            |
Phase 6  Evaluation (accuracy/loss curves, confusion matrix, classification report, error analysis)
            |
Phase 7  Face detection & inference pipeline (Haar cascade, detector tuning, full preprocessing chain)
            |
Phase 8  Marketing insights (aggregation of predictions into a marketing-relevant takeaway)
            |
Phase 9  Real-world brand case study — Ballerina Farm vs. Nara Smith
          (Instagram data collection → face cropping → engagement CSV → inference → brand comparison charts)
```

## 17. Technology Stack

| Tool / Library | Role in the Project |
|---|---|
| Python | Core programming language |
| OpenCV | Image loading, face detection (Haar cascades), image manipulation |
| TensorFlow / Keras | CNN construction, training, evaluation, and inference |
| NumPy & Pandas | Numerical computation, dataset manifests, results aggregation |
| Matplotlib | Visualization of class distributions, training curves, confusion matrices, and sample/result images |
| scikit-learn | Class-weight computation, classification metrics (precision/recall/F1, confusion matrix) |

## 18. Data Sources

- **FER2013** — training and evaluation dataset. Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- **Instagram (Phase 9)** — screenshots collected manually from the public Instagram accounts @ballerinafarmstore and @naraaziza, used solely for academic analysis. Engagement metrics (likes, comments, reposts, sends) read directly from the Instagram UI at the time of collection.
