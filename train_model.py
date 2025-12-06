#!/usr/bin/env python3
"""
LBPH Model Training Script

This script trains the LBPH face recognizer using the collected face dataset.
It scans the dataset directory and creates a trained model.

Usage:
    python train_model.py
"""

import cv2
import os
import sys
import numpy as np
from face_recognizer import FaceRecognizer

# Configuration
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")


def load_dataset():
    """
    Load face images and labels from the dataset directory.

    Returns:
        Tuple of (faces_list, labels_list, label_map)
    """
    faces = []
    labels = []
    label_map = {}  # Maps label_id -> person_name
    current_label = 0

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found!")
        print("Please run 'python collect_faces.py' first to collect training data.")
        sys.exit(1)

    # Get list of person directories
    persons = [
        d
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    if len(persons) == 0:
        print(f"Error: No person directories found in '{DATASET_DIR}'!")
        print("Please run 'python collect_faces.py' first to collect training data.")
        sys.exit(1)

    print(f"Found {len(persons)} person(s) in dataset:")

    for person_name in sorted(persons):
        person_dir = os.path.join(DATASET_DIR, person_name)

        # Get all jpg images for this person
        image_files = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]

        if len(image_files) == 0:
            print(f"  - {person_name}: No images found, skipping...")
            continue

        print(f"  - {person_name}: {len(image_files)} images (label={current_label})")

        # Add to label map
        label_map[current_label] = person_name

        # Load images
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                faces.append(img)
                labels.append(current_label)

        current_label += 1

    return faces, labels, label_map


def train_model():
    """Main function to train the LBPH model."""
    print("=" * 50)
    print("  LBPH MODEL TRAINING SCRIPT")
    print("=" * 50)
    print()

    # Load dataset
    print("Loading dataset...")
    faces, labels, label_map = load_dataset()

    if len(faces) == 0:
        print("Error: No face images found in dataset!")
        sys.exit(1)

    print(f"\nTotal images loaded: {len(faces)}")
    print(f"Unique persons: {len(label_map)}")

    # Create and train recognizer
    print("\nTraining LBPH recognizer...")
    recognizer = FaceRecognizer(threshold=80.0)
    recognizer.set_label_map(label_map)
    recognizer.train(faces, labels)

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    print("\nSaving model...")
    recognizer.save_model(MODEL_PATH, LABEL_MAP_PATH)

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")
    print(f"\nTrained on {len(label_map)} person(s):")
    for label_id, name in label_map.items():
        count = labels.count(label_id)
        print(f"  - {name}: {count} images")
    print("\nNext step: Run 'python recognize_faces.py' for real-time recognition")
    print("=" * 50)


if __name__ == "__main__":
    train_model()
