"""
Basic smoke tests. Run with:  python -m pytest tests/ -v
(or simply: python tests/test_pipeline.py)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from anpr import preprocessing as pp
from anpr import ocr as ocr_mod
from anpr.pipeline import process_image


def _make_synthetic_plate_image():
    canvas = np.full((480, 900, 3), (90, 90, 90), dtype=np.uint8)
    cv2.rectangle(canvas, (200, 150), (700, 350), (40, 40, 180), -1)
    cv2.rectangle(canvas, (380, 300), (560, 350), (255, 255, 255), -1)
    cv2.rectangle(canvas, (380, 300), (560, 350), (0, 0, 0), 2)
    cv2.putText(canvas, "MP09AB1234", (388, 335),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


class TestPreprocessing(unittest.TestCase):
    def test_resize_keep_aspect(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        out = pp.resize_keep_aspect(img, target_width=200)
        self.assertEqual(out.shape[1], 200)
        self.assertAlmostEqual(out.shape[0] / out.shape[1], 200 / 400, places=2)

    def test_find_plate_candidates_on_synthetic_image(self):
        img = _make_synthetic_plate_image()
        candidates = pp.find_plate_candidates(img)
        self.assertGreater(len(candidates), 0, "Should find at least one plate-like region")


class TestOCR(unittest.TestCase):
    def test_clean_text_strips_noise(self):
        self.assertEqual(ocr_mod.clean_text(" mp09ab1234\n"), "MP09AB1234")

    def test_is_plausible_plate(self):
        self.assertTrue(ocr_mod.is_plausible_plate("MP09AB1234"))
        self.assertFalse(ocr_mod.is_plausible_plate("AB"))       # too short
        self.assertFalse(ocr_mod.is_plausible_plate("12345678")) # no letters


class TestFullPipeline(unittest.TestCase):
    def test_process_image_returns_result_object(self):
        img = _make_synthetic_plate_image()
        result = process_image(img)
        self.assertIsNotNone(result.annotated_image)
        self.assertEqual(result.annotated_image.shape[1], 900)
        # OCR result content isn't asserted strictly (font-dependent),
        # but the pipeline should run end-to-end without raising.


if __name__ == "__main__":
    unittest.main()
