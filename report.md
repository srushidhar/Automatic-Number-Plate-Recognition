# Project Report: Automatic Number Plate Recognition (ANPR)

**Course:** Computer Vision — Evaluated Project
**Domain:** Classical Image Processing / Optical Character Recognition

---

## 1. Abstract

This project implements an end-to-end Automatic Number Plate Recognition
(ANPR) system that detects and reads vehicle license plates from images and
video using purely classical computer-vision techniques, combined with
Tesseract OCR for text extraction. The system is delivered as a
command-line tool supporting single-image, batch, video, and live webcam
modes. On front-facing, moderately clear inputs, the pipeline reliably
localizes the plate region and extracts its alphanumeric text.

## 2. Introduction & Motivation

License plate recognition is a foundational computer-vision task with
real-world applications in toll collection, parking management, traffic law
enforcement, and access control. While modern production ANPR systems
typically rely on deep-learning object detectors (e.g. YOLO) trained on
large annotated plate datasets, a classical image-processing approach —
edge detection, contour analysis, and geometric filtering — can achieve
reasonable accuracy without any training data or GPU, making it a good
vehicle for demonstrating core CV concepts taught in this course: filtering,
edge detection, morphology, contour geometry, and perspective transforms.

## 3. Objectives

1. Automatically localize the license-plate region in an image of a vehicle
   without any pretrained detection model.
2. Correct perspective distortion in the localized region.
3. Extract the plate's alphanumeric text using OCR.
4. Provide a fully CLI-executable tool that works on single images, folders
   of images, video files, and a live webcam feed.

## 4. Related Work

- **Classical approaches**: edge-map + contour filtering (as used here),
  and Sobel/vertical-edge-density methods that exploit the high density of
  vertical edges created by plate characters.
- **Learning-based approaches**: Haar-cascade plate detectors, and modern
  deep detectors (YOLO/SSD) fine-tuned on plate datasets, typically paired
  with a CRNN or Tesseract for the OCR stage.
- This project follows the classical family, since it requires no labeled
  training data and remains fully explainable step-by-step — a good fit for
  a self-contained course project.

## 5. Methodology

### 5.1 Preprocessing
The input frame is resized to a fixed working width for consistent
processing speed, converted to grayscale, and denoised with a bilateral
filter, which smooths flat regions while preserving strong edges (important
because plate boundaries and characters are edge-rich).

### 5.2 Edge Detection & Morphology
Canny edge detection produces a binary edge map. A rectangular structuring
element is then used with morphological closing to bridge small gaps between
character edges, merging each plate's text into a single dense, contiguous
blob that stands out from the surrounding, sparser edges of the rest of the
vehicle.

### 5.3 Contour Extraction & Geometric Filtering
All external contours are extracted and ranked by area. Each candidate is
filtered by:
- **Area bounds** (relative to image area) to reject noise and oversized
  regions,
- **Aspect ratio** (~1.8–6.0), matching common single- and double-line plate
  proportions,
- **Rectangular solidity** (contour area / bounding-rectangle area), to
  favor solid, plate-shaped blobs over irregular ones.

Each surviving candidate receives a heuristic confidence score combining
solidity, relative size, and closeness to a 4-vertex polygon approximation.

### 5.4 Perspective Correction
The best-scoring candidate's rotated bounding box corners are used to
compute a perspective transform, warping the plate to a flat, top-down
rectangle regardless of the vehicle's viewing angle in the original photo.

### 5.5 OCR Enhancement & Recognition
The warped crop is upscaled (to give Tesseract enough resolution),
Gaussian-blurred slightly, and binarized with Otsu's method. A small
morphological closing reconnects any character strokes broken by
thresholding, and a white border is added (Tesseract performs poorly when
text touches the image edge). Tesseract (`--psm 7`, single text line) then
reads the plate, restricted to an alphanumeric character whitelist.

### 5.6 Post-processing & Validation
Raw OCR output is cleaned (uppercased, whitespace/non-whitelisted characters
stripped) and validated with a plausibility check (length bounds, presence
of both letters and digits) to reject obviously incorrect readings, and
duplicate detections across overlapping candidate regions are merged.

## 6. Implementation

- **Language / libraries**: Python 3, OpenCV (`opencv-python`), NumPy,
  `pytesseract` (Tesseract OCR wrapper).
- **Architecture**: `anpr/preprocessing.py` (localization + deskew),
  `anpr/ocr.py` (text extraction + validation), `anpr/pipeline.py`
  (orchestration), `main.py` (CLI: image / batch / video / webcam modes).
- **No training required**: the entire system runs on CPU with no model
  downloads, keeping it fully reproducible from a clean checkout — a
  deliberate design choice to satisfy the "executable via command line,
  no prior context needed" evaluation requirement.
- **Testing**: `tests/test_pipeline.py` contains unit tests for the resize
  utility, candidate localization on a synthetic image, OCR text cleanup,
  plausibility validation, and an end-to-end smoke test of the full
  pipeline.

## 7. Results

On a synthetically generated test image containing a clearly rendered plate
(`generate_sample.py`), the pipeline correctly localizes the plate region
(bounding box tightly around the plate) and recovers the plate string with
high fidelity via OCR. On batch runs over a folder of images, the tool
produces a `results.csv` summary alongside annotated copies of each input
image with the detected plate boxed and its recognized text overlaid.

*(When evaluating with real vehicle photographs, insert sample annotated
outputs and a short accuracy table — e.g. N images tested, M correctly
localized, K read with exactly correct text — here.)*

### Qualitative observations
- Performs best on reasonably front-on shots with good contrast between the
  plate and its surroundings.
- Detection is robust to moderate rotation thanks to the perspective-warp
  step.
- OCR accuracy is the main bottleneck on lower-resolution or stylized-font
  plates; the upscale + Otsu-threshold step meaningfully improves this over
  running Tesseract on the raw crop.

## 8. Limitations & Future Work

- **False positives**: other rectangular, high-contrast regions (bumper
  stickers, grille sections) can occasionally be misidentified as plates;
  the plausibility filter mitigates but does not eliminate this.
- **Non-Latin / stylized fonts**: Tesseract's default English-trained model
  struggles with some regional plate fonts; a custom-trained OCR model would
  improve this.
- **Extreme angles / occlusion / low light**: not explicitly handled;
  a learned plate detector (e.g. a small YOLO model fine-tuned on a plate
  dataset) would be more robust than contour-based localization in these
  conditions, at the cost of requiring training data and a GPU.
- **Future work**: add a lightweight trained plate detector as a fallback
  when the classical localizer finds no plausible candidates, and add a
  plate-format-specific regex/checksum validation layer per country/region.

## 9. Conclusion

This project demonstrates a complete, dependency-light ANPR pipeline built
entirely from classical computer-vision primitives — filtering, edge
detection, morphology, contour analysis, and perspective transforms — paired
with OCR for text recognition. The approach requires no training data or
GPU, is fully reproducible from the command line, and achieves reliable
results on well-conditioned inputs, while clearly delineating where a
learning-based approach would be needed for more challenging real-world
conditions.

## 10. References

1. OpenCV Documentation — https://docs.opencv.org/
2. Tesseract OCR — https://github.com/tesseract-ocr/tesseract
3. R. Szeliski, *Computer Vision: Algorithms and Applications*, Springer.
4. Canny, J., "A Computational Approach to Edge Detection," IEEE PAMI, 1986.
