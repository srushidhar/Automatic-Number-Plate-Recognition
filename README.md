# Automatic Number Plate Recognition (ANPR)

A classical computer-vision pipeline that locates a vehicle's license plate
in an image or video and reads the plate text using OCR — no deep-learning
training or GPU required.

**Course:** Computer Vision — Evaluated Project

## How it works (pipeline overview)

1. **Preprocessing** — grayscale conversion, bilateral filtering (denoise
   while preserving edges), Canny edge detection, and morphological closing
   to merge character strokes into solid, contiguous blobs.
2. **Plate localization** — contour extraction followed by geometric
   filtering (area, aspect ratio ~1.8–6.0, rectangular solidity) to shortlist
   plate-like regions; candidates are ranked by a heuristic score.
3. **Perspective correction** — the best candidate's four corners are used to
   perform a perspective (deskew) warp, producing a straightened, top-down
   crop of the plate.
4. **OCR enhancement** — the crop is upscaled, thresholded (Otsu binarization),
   and lightly cleaned up morphologically to maximize OCR accuracy.
5. **Text recognition** — [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
   (via `pytesseract`) reads the plate text, restricted to an
   alphanumeric whitelist; results are cleaned and validated (length,
   digit + letter presence) before being reported.

See `report.md` for the full project report (motivation, methodology,
results, and limitations).

## Repository structure

```
.
├── main.py                 # CLI entry point
├── generate_sample.py      # creates a synthetic test image (for quick smoke tests)
├── anpr/
│   ├── __init__.py
│   ├── preprocessing.py    # edge detection, contour filtering, deskew
│   ├── ocr.py               # Tesseract wrapper + text cleanup/validation
│   └── pipeline.py         # ties preprocessing + OCR together
├── tests/
│   └── test_pipeline.py    # unit tests (unittest)
├── sample_images/          # put your test images here
├── output/                 # annotated results are written here
├── requirements.txt
├── report.md
└── README.md
```

## 1. Environment setup

Requires **Python 3.9+**.

### 1.1 Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 1.2 Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
```

### 1.3 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Install the Tesseract OCR engine (system binary)

`pytesseract` is a Python wrapper — it needs the actual Tesseract binary
installed on your system.

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt-get update && sudo apt-get install -y tesseract-ocr` |
| macOS (Homebrew) | `brew install tesseract` |
| Windows | Download the installer from the [UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your `PATH` |

Verify the install:

```bash
tesseract --version
```

## 2. Running the project

All usage is via the command line through `main.py`.

### 2.1 Quick smoke test (no external images needed)

Generates a synthetic car+plate image and runs the full pipeline on it:

```bash
python generate_sample.py
python main.py --mode image --input sample_images/synthetic_car.jpg --output output/synthetic_result.jpg
```

Expected console output:
```
[OK] synthetic_car.jpg -> MP09AB1234
    Annotated image saved to: output/synthetic_result.jpg
```

### 2.2 Single real image

Place a photo of a vehicle (ideally a reasonably front-on shot of the plate)
into `sample_images/`, then:

```bash
python main.py --mode image --input sample_images/car1.jpg --output output/car1_result.jpg
```

### 2.3 Batch mode (process a whole folder)

```bash
python main.py --mode batch --input sample_images/ --output output/
```

Writes one annotated image per input plus a summary `output/results.csv`
with columns `filename, detected_plate_text(s)`.

### 2.4 Video file

```bash
python main.py --mode video --input sample_images/traffic.mp4 --output output/traffic_result.mp4
```

For performance, detection runs every 3rd frame by default (configurable via
the `process_every_n` parameter in `main.py`); intermediate frames reuse the
last annotation.

### 2.5 Live webcam

```bash
python main.py --mode webcam
```

Press `q` in the display window to quit.

## 3. Running the tests

```bash
python -m unittest discover -s tests -v
```

## 4. Configuration notes

- **Plate aspect-ratio range** and **contour area thresholds** are defined in
  `anpr/preprocessing.py::find_plate_candidates` — tune these if your plates
  follow a very different regional format (e.g. square EU/moto plates).
- **OCR character whitelist** is defined in `anpr/ocr.py::_TESS_CONFIG` —
  extend it if your plate format includes additional symbols.
- The pipeline is intentionally **classical CV / non-deep-learning**, so it
  runs on CPU with no model downloads and no training data required. This
  keeps the project fully reproducible and runnable by an evaluator with a
  fresh machine.

## 5. Known limitations

- Designed and tuned for roughly front-on / slightly angled shots of a single
  vehicle; heavy perspective, motion blur, or very low resolution reduces
  accuracy.
- Contour-based localization can occasionally pick up other rectangular,
  high-contrast regions (e.g. bumper stickers, headlight housings) as false
  positives — the OCR plausibility check (`ocr.py::is_plausible_plate`)
  filters out most of these by rejecting non-alphanumeric-looking results.
- Tesseract's out-of-the-box accuracy on stylized or non-Latin plate fonts is
  limited; for production use, a fine-tuned OCR model or a trained plate
  detector (e.g. YOLO) would improve robustness. This is discussed further
  in `report.md`.

## License

Submitted as coursework. Feel free to reuse for learning purposes.
