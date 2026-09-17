#!/usr/bin/env python3
"""
main.py
-------
Command-line interface for the Automatic Number Plate Recognition (ANPR)
project.

Usage:
    # Single image
    python main.py --mode image --input sample_images/car1.jpg --output output/car1_result.jpg

    # All images in a folder
    python main.py --mode batch --input sample_images/ --output output/

    # Video file
    python main.py --mode video --input sample_images/traffic.mp4 --output output/traffic_result.mp4

    # Live webcam (press 'q' to quit)
    python main.py --mode webcam
"""

import argparse
import csv
import os
import sys
import time

import cv2

from anpr.pipeline import process_image


def run_image(input_path: str, output_path: str) -> None:
    image = cv2.imread(input_path)
    if image is None:
        sys.exit(f"[ERROR] Could not read image: {input_path}")

    result = process_image(image)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result.annotated_image)

    if result.plates:
        print(f"[OK] {os.path.basename(input_path)} -> "
              f"{', '.join(p.text for p in result.plates)}")
    else:
        print(f"[--] {os.path.basename(input_path)} -> no plate detected")

    print(f"    Annotated image saved to: {output_path}")


def run_batch(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext))
    if not files:
        sys.exit(f"[ERROR] No images found in {input_dir}")

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filename", "detected_plate_text(s)"])

        for fname in files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, f"annotated_{fname}")
            image = cv2.imread(in_path)
            if image is None:
                print(f"[SKIP] Unreadable file: {fname}")
                continue

            result = process_image(image)
            cv2.imwrite(out_path, result.annotated_image)
            plate_texts = ";".join(p.text for p in result.plates) or "NOT_DETECTED"
            writer.writerow([fname, plate_texts])
            print(f"[OK] {fname} -> {plate_texts}")

    print(f"\nSummary written to: {csv_path}")


def run_video(input_path: str, output_path: str, process_every_n: int = 3) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_annotated = None
    seen_plates = set()
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OCR is the expensive step; run detection every Nth frame and reuse
        # the last annotation in between to keep throughput reasonable.
        if frame_idx % process_every_n == 0:
            result = process_image(frame)
            last_annotated = result.annotated_image
            for p in result.plates:
                if p.text not in seen_plates:
                    seen_plates.add(p.text)
                    print(f"[frame {frame_idx}] New plate detected: {p.text}")
            out_frame = cv2.resize(last_annotated, (width, height))
        else:
            out_frame = frame

        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s.")
    print(f"Unique plates seen: {sorted(seen_plates) or 'none'}")
    print(f"Output video saved to: {output_path}")


def run_webcam() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Could not open webcam.")

    print("Press 'q' to quit.")
    frame_idx = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 == 0:
            result = process_image(frame)
            last_annotated = result.annotated_image

        display = last_annotated if last_annotated is not None else frame
        cv2.imshow("ANPR - press q to quit", display)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Automatic Number Plate Recognition (ANPR)")
    parser.add_argument("--mode", choices=["image", "batch", "video", "webcam"], required=True)
    parser.add_argument("--input", help="Path to input image, folder, or video")
    parser.add_argument("--output", help="Path to output image, folder, or video")
    args = parser.parse_args()

    if args.mode == "image":
        if not args.input:
            sys.exit("--input is required for --mode image")
        out = args.output or "output/result.jpg"
        run_image(args.input, out)

    elif args.mode == "batch":
        if not args.input:
            sys.exit("--input is required for --mode batch")
        out = args.output or "output/"
        run_batch(args.input, out)

    elif args.mode == "video":
        if not args.input:
            sys.exit("--input is required for --mode video")
        out = args.output or "output/result.mp4"
        run_video(args.input, out)

    elif args.mode == "webcam":
        run_webcam()


if __name__ == "__main__":
    main()
