#!/usr/bin/env python3
"""
Real-Time Face Recognition Script

This script performs real-time face recognition using:
- MediaPipe for face detection
- LBPH for face recognition

Usage:
    python recognize_faces.py

Press 'q' to quit.
"""

import cv2
import os
import sys
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer

# Configuration
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
TARGET_SIZE = (200, 200)  # Must match training size
CONFIDENCE_THRESHOLD = 80.0  # Lower is better for LBPH


def run_recognition():
    """Main function for real-time face recognition."""
    print("=" * 50)
    print("  REAL-TIME FACE RECOGNITION")
    print("  MediaPipe Detection + LBPH Recognition")
    print("=" * 50)
    print()

    # Initialize detector
    print("Initializing face detector (MediaPipe)...")
    detector = FaceDetector(min_detection_confidence=0.5)

    # Initialize recognizer and load model
    print("Initializing face recognizer (LBPH)...")
    recognizer = FaceRecognizer(threshold=CONFIDENCE_THRESHOLD)

    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at '{MODEL_PATH}'")
        print("Please run these steps first:")
        print("  1. python collect_faces.py  (collect training data)")
        print("  2. python train_model.py    (train the model)")
        sys.exit(1)

    if not recognizer.load_model(MODEL_PATH, LABEL_MAP_PATH):
        print("Error: Failed to load model!")
        sys.exit(1)

    print(f"\nLoaded model with {len(recognizer.label_map)} known person(s):")
    for label_id, name in recognizer.label_map.items():
        print(f"  - {name}")

    # Open webcam
    print("\nStarting webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam!")
        sys.exit(1)

    print("\nRecognition running! Press 'q' to quit.")
    print("-" * 50)

    frame_count = 0
    fps = 0
    fps_start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame!")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction

        # Detect faces
        bboxes = detector.detect_faces(frame)

        names = []
        confidences = []
        colors = []

        for bbox in bboxes:
            # Extract face ROI
            face_roi = detector.extract_face_roi(frame, bbox, TARGET_SIZE)

            if face_roi is not None:
                # Recognize face
                name, confidence, is_known = recognizer.predict_with_name(face_roi)
                names.append(name)

                # Convert confidence to percentage (inverted for display)
                # LBPH: lower = better, so we invert for intuitive display
                if is_known:
                    conf_percent = max(0, 100 - confidence)
                    colors.append((0, 255, 0))  # Green for known
                else:
                    conf_percent = 0
                    colors.append((0, 0, 255))  # Red for unknown

                confidences.append(conf_percent)
            else:
                names.append("Error")
                confidences.append(0)
                colors.append((0, 165, 255))  # Orange for error

        # Draw results
        for i, (x, y, w, h) in enumerate(bboxes):
            color = colors[i] if i < len(colors) else (255, 255, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Prepare label
            if i < len(names):
                label = f"{names[i]}"
                if i < len(confidences) and confidences[i] > 0:
                    label += f" ({confidences[i]:.0f}%)"

                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                cv2.rectangle(
                    frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1
                )
                cv2.putText(
                    frame, label, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness
                )

        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:
            fps_end_time = cv2.getTickCount()
            fps = frame_count / (
                (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            )
            frame_count = 0
            fps_start_time = cv2.getTickCount()

        # Draw info overlay
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Faces: {len(bboxes)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Show frame
        cv2.imshow("Face Recognition - MediaPipe + LBPH", frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nQuitting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("Recognition stopped.")


if __name__ == "__main__":
    run_recognition()
