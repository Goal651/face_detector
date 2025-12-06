#!/usr/bin/env python3
"""
Face Data Collection Script

This script captures face images from your webcam for training the LBPH recognizer.
It uses MediaPipe for face detection to ensure good quality face crops.

Usage:
    python collect_faces.py

Follow the prompts to enter your name and collect face images.
"""

import cv2
import os
import sys
import time
from face_detector import FaceDetector

# Configuration
DATASET_DIR = "dataset"
NUM_IMAGES = 50  # Number of images to collect per person
TARGET_SIZE = (200, 200)  # Size of face images
CAPTURE_DELAY = 0.1  # Seconds between captures


def create_dataset_dir(person_name: str) -> str:
    """Create directory for person's face images."""
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir


def get_existing_count(person_dir: str) -> int:
    """Get count of existing images in directory."""
    if not os.path.exists(person_dir):
        return 0
    return len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])


def collect_faces():
    """Main function to collect face images."""
    print("=" * 50)
    print("  FACE DATA COLLECTION SCRIPT")
    print("  Using MediaPipe for Face Detection")
    print("=" * 50)
    print()

    # Get person's name
    person_name = input("Enter the person's name (no spaces): ").strip()
    if not person_name:
        print("Error: Name cannot be empty!")
        sys.exit(1)

    # Replace spaces with underscores
    person_name = person_name.replace(" ", "_")

    # Create dataset directory
    person_dir = create_dataset_dir(person_name)
    existing_count = get_existing_count(person_dir)

    if existing_count > 0:
        print(f"\nFound {existing_count} existing images for '{person_name}'")
        choice = input("Do you want to [a]dd more or [r]eplace all? (a/r): ").lower()
        if choice == "r":
            # Remove existing images
            for f in os.listdir(person_dir):
                if f.endswith(".jpg"):
                    os.remove(os.path.join(person_dir, f))
            existing_count = 0
            print("Existing images removed.")

    print(f"\nCollecting {NUM_IMAGES} face images for '{person_name}'")
    print("Position your face in the camera and move slightly between captures.")
    print("Press 'q' to quit early, 'p' to pause/resume")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    # Initialize face detector
    detector = FaceDetector(min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam!")
        sys.exit(1)

    collected = 0
    paused = False
    last_capture_time = 0

    print("\nCollecting faces...")

    while collected < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame!")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        display_frame = frame.copy()

        # Detect faces
        bboxes = detector.detect_faces(frame)

        # Draw UI
        status_color = (0, 255, 0) if not paused else (0, 165, 255)
        status_text = f"Collected: {collected}/{NUM_IMAGES}" if not paused else "PAUSED"
        cv2.putText(
            display_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2,
        )
        cv2.putText(
            display_frame,
            f"Person: {person_name}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if len(bboxes) == 0:
            cv2.putText(
                display_frame,
                "No face detected - position your face",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        elif len(bboxes) > 1:
            cv2.putText(
                display_frame,
                "Multiple faces - only show one face",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

        # Draw bounding boxes
        for x, y, w, h in bboxes:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), status_color, 2)

        # Capture logic
        current_time = time.time()
        if (
            not paused
            and len(bboxes) == 1
            and current_time - last_capture_time >= CAPTURE_DELAY
        ):

            # Extract face ROI
            face_roi = detector.extract_face_roi(frame, bboxes[0], TARGET_SIZE)

            if face_roi is not None:
                # Save face image
                img_path = os.path.join(person_dir, f"{existing_count + collected}.jpg")
                cv2.imwrite(img_path, face_roi)
                collected += 1
                last_capture_time = current_time

                # Flash effect
                cv2.rectangle(
                    display_frame,
                    (0, 0),
                    (frame.shape[1], frame.shape[0]),
                    (255, 255, 255),
                    10,
                )

        # Progress bar
        progress = int((collected / NUM_IMAGES) * 400)
        cv2.rectangle(
            display_frame,
            (10, frame.shape[0] - 30),
            (10 + progress, frame.shape[0] - 10),
            (0, 255, 0),
            -1,
        )
        cv2.rectangle(
            display_frame,
            (10, frame.shape[0] - 30),
            (410, frame.shape[0] - 10),
            (255, 255, 255),
            2,
        )

        cv2.imshow("Face Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nCollection cancelled by user.")
            break
        elif key == ord("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # Final status
    print("\n" + "=" * 50)
    if collected > 0:
        print(f"SUCCESS! Collected {collected} images for '{person_name}'")
        print(f"Images saved to: {person_dir}/")
        print("\nNext step: Run 'python train_model.py' to train the model")
    else:
        print("No images collected.")
    print("=" * 50)


if __name__ == "__main__":
    collect_faces()
