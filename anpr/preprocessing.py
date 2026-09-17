"""
preprocessing.py
-----------------
Classical image-processing utilities used to localize likely license-plate
regions in a vehicle image before OCR is attempted.

Pipeline (all pure OpenCV / NumPy, no pretrained deep nets required):
    1. Resize for consistent processing speed.
    2. Convert to grayscale + bilateral filter (denoise, keep edges sharp).
    3. Canny edge detection.
    4. Morphological closing to merge character strokes into solid blobs.
    5. Contour extraction + geometric filtering (aspect ratio, area, solidity)
       to shortlist rectangular plate-like regions.
    6. Perspective correction (deskew) of the best candidate.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlateCandidate:
    box: np.ndarray          # 4x2 array of corner points (in resized-image coords)
    rect_image: np.ndarray   # cropped & deskewed BGR plate image
    score: float             # heuristic confidence score (higher = more plate-like)


def resize_keep_aspect(image: np.ndarray, target_width: int = 900) -> np.ndarray:
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / float(w)
    return cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def preprocess_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_plate(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    rect = _order_points(box.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB), 1)

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB), 1)

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_plate_candidates(image: np.ndarray, max_candidates: int = 5) -> List[PlateCandidate]:
    """Return a ranked list of plate-like regions found in `image`."""
    closed = preprocess_edges(image)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    img_area = image.shape[0] * image.shape[1]
    candidates: List[PlateCandidate] = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if area < 0.001 * img_area or area > 0.35 * img_area:
            continue

        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)

        # Typical plate aspect ratios (single/double line) roughly 2:1 to 5.5:1
        if not (1.8 <= aspect <= 6.0):
            continue

        box = cv2.boxPoints(rect)
        rect_area = rw * rh
        solidity = area / rect_area if rect_area > 0 else 0
        if solidity < 0.5:
            continue

        # Heuristic score: prefer larger, more rectangular (higher solidity),
        # plate-ratio-typical regions, and fewer vertices (~4 in approx).
        vertex_penalty = abs(len(approx) - 4) * 0.05
        score = (solidity * 0.6) + (min(area / img_area, 0.15) / 0.15 * 0.3) - vertex_penalty

        warped = _warp_plate(image, box)
        if warped.shape[0] < 10 or warped.shape[1] < 30:
            continue

        candidates.append(PlateCandidate(box=box, rect_image=warped, score=score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def enhance_plate_for_ocr(plate_img: np.ndarray) -> np.ndarray:
    """Upscale + threshold a cropped plate image to improve OCR accuracy."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img
    h, w = gray.shape[:2]
    scale = 3 if max(h, w) < 300 else 2
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure dark text on light background (OCR engines generally prefer this)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Small closing pass to reconnect character strokes broken by thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Pad with white border -- Tesseract performs poorly when text touches edges
    thresh = cv2.copyMakeBorder(thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return thresh
