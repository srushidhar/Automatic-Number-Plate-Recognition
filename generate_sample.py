#!/usr/bin/env python3
"""
generate_sample.py
-------------------
Creates a synthetic "car with license plate" test image so the pipeline can
be exercised end-to-end without needing to source a real photograph. This is
only a convenience for quick smoke-testing -- for the project report, run the
pipeline on real photographs placed in sample_images/.
"""

import os
import cv2
import numpy as np

OUT_PATH = "sample_images/synthetic_car.jpg"


def main():
    os.makedirs("sample_images", exist_ok=True)

    canvas = np.full((480, 900, 3), (90, 90, 90), dtype=np.uint8)  # gray "road/background"

    # Car body (simple rectangle silhouette)
    cv2.rectangle(canvas, (200, 150), (700, 350), (40, 40, 180), -1)
    cv2.rectangle(canvas, (250, 100), (650, 160), (60, 60, 200), -1)  # roof

    # Plate background (white) + border
    plate_tl, plate_br = (380, 300), (560, 350)
    cv2.rectangle(canvas, plate_tl, plate_br, (255, 255, 255), -1)
    cv2.rectangle(canvas, plate_tl, plate_br, (0, 0, 0), 2)

    # Plate text
    cv2.putText(
        canvas, "MP09AB1234", (plate_tl[0] + 8, plate_br[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
    )

    cv2.imwrite(OUT_PATH, canvas)
    print(f"Synthetic test image written to {OUT_PATH}")


if __name__ == "__main__":
    main()
