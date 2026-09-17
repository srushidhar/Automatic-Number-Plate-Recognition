"""
pipeline.py
-----------
End-to-end Automatic Number Plate Recognition (ANPR) pipeline for a single
image. Combines classical CV plate localization with OCR text extraction,
and returns a structured result plus an annotated visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from . import preprocessing as pp
from . import ocr as ocr_mod


@dataclass
class PlateResult:
    text: str
    confidence_score: float
    box: np.ndarray


@dataclass
class FrameResult:
    annotated_image: np.ndarray
    plates: List[PlateResult] = field(default_factory=list)


def process_image(image: np.ndarray, max_candidates: int = 5) -> FrameResult:
    resized = pp.resize_keep_aspect(image)
    candidates = pp.find_plate_candidates(resized, max_candidates=max_candidates)

    annotated = resized.copy()
    plates: List[PlateResult] = []

    for cand in candidates:
        binarized = pp.enhance_plate_for_ocr(cand.rect_image)
        try:
            text = ocr_mod.read_plate_text(binarized)
        except RuntimeError as e:
            # Surface the setup error once, on the caller's terms.
            raise e

        if not text or not ocr_mod.is_plausible_plate(text):
            continue

        plates.append(PlateResult(text=text, confidence_score=cand.score, box=cand.box))

        box_int = cand.box.astype(int)
        cv2.drawContours(annotated, [box_int], 0, (0, 255, 0), 2)
        x, y = box_int[0]
        cv2.putText(
            annotated, text, (int(x), max(int(y) - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

    # De-duplicate near-identical readings (e.g. from overlapping candidates)
    seen = set()
    unique_plates = []
    for p in plates:
        if p.text not in seen:
            seen.add(p.text)
            unique_plates.append(p)

    return FrameResult(annotated_image=annotated, plates=unique_plates)
