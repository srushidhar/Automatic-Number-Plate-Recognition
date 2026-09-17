"""
ocr.py
------
Wraps Tesseract OCR (via pytesseract) with plate-specific configuration and
lightweight post-processing to clean up recognized text.
"""

from __future__ import annotations

import re
import numpy as np

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

# Whitelist covers most Latin-alphabet plate formats (letters + digits + a
# couple of common separators). Adjust per country/format if needed.
_TESS_CONFIG = (
    "--oem 3 --psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
)

_CLEAN_RE = re.compile(r"[^A-Z0-9-]")


def read_plate_text(plate_img_binarized: np.ndarray) -> str:
    """Run OCR on a preprocessed (binarized) plate crop and return cleaned text."""
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract "
            "and install the Tesseract binary (see README)."
        )
    raw = pytesseract.image_to_string(plate_img_binarized, config=_TESS_CONFIG)
    return clean_text(raw)


def clean_text(raw: str) -> str:
    text = raw.strip().upper().replace(" ", "")
    text = _CLEAN_RE.sub("", text)
    return text


def is_plausible_plate(text: str, min_len: int = 4, max_len: int = 12) -> bool:
    """Reject obviously-wrong OCR output (too short/long, no digits at all, etc.)."""
    if not (min_len <= len(text) <= max_len):
        return False
    has_digit = any(ch.isdigit() for ch in text)
    has_alpha = any(ch.isalpha() for ch in text)
    return has_digit and has_alpha
