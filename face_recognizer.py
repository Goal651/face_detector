"""
Face Recognition Module using LBPH (Local Binary Pattern Histograms)

This module provides face recognition capabilities using OpenCV's LBPH recognizer.
LBPH is a classical computer vision approach that doesn't require deep learning.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional


class FaceRecognizer:
    """
    Face recognizer using OpenCV's LBPH (Local Binary Pattern Histograms).

    LBPH works by:
    1. Dividing the face image into local regions
    2. Computing LBP for each pixel (comparing with neighbors)
    3. Creating histograms of LBP values for each region
    4. Concatenating histograms to form a feature vector
    5. Comparing feature vectors using chi-square distance
    """

    def __init__(
        self,
        radius: int = 1,
        neighbors: int = 8,
        grid_x: int = 8,
        grid_y: int = 8,
        threshold: float = 80.0,
    ):
        """
        Initialize the LBPH face recognizer.

        Args:
            radius: Radius of circular LBP operator
            neighbors: Number of neighbors for LBP
            grid_x: Number of horizontal cells in the grid
            grid_y: Number of vertical cells in the grid
            threshold: Confidence threshold (lower = better match in LBPH)
        """
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=radius,
            neighbors=neighbors,
            grid_x=grid_x,
            grid_y=grid_y,
            threshold=threshold,
        )
        self.label_map: Dict[int, str] = {}  # Maps label_id -> person_name
        self.threshold = threshold
        self.is_trained = False

    def train(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """
        Train the recognizer on face images.

        Args:
            faces: List of grayscale face images (should be same size)
            labels: List of integer labels corresponding to each face
        """
        if len(faces) == 0:
            raise ValueError("No training faces provided")
        if len(faces) != len(labels):
            raise ValueError("Number of faces must match number of labels")

        # Convert to numpy arrays
        faces_array = [np.array(face, dtype=np.uint8) for face in faces]
        labels_array = np.array(labels, dtype=np.int32)

        self.recognizer.train(faces_array, labels_array)
        self.is_trained = True
        print(f"Model trained on {len(faces)} face images")

    def predict(self, face: np.ndarray) -> Tuple[int, float]:
        """
        Predict the identity of a face.

        Args:
            face: Grayscale face image

        Returns:
            Tuple of (label_id, confidence)
            Note: For LBPH, lower confidence = better match
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first or load a model.")

        label, confidence = self.recognizer.predict(face)
        return label, confidence

    def predict_with_name(self, face: np.ndarray) -> Tuple[str, float, bool]:
        """
        Predict identity and return the person's name.

        Args:
            face: Grayscale face image

        Returns:
            Tuple of (name, confidence, is_known)
            - name: Person's name or "Unknown"
            - confidence: Recognition confidence (lower = better for LBPH)
            - is_known: True if confidence is below threshold
        """
        label, confidence = self.predict(face)

        # Check if confidence is good enough (lower is better in LBPH)
        if confidence < self.threshold:
            name = self.label_map.get(label, f"Person_{label}")
            return name, confidence, True
        else:
            return "Unknown", confidence, False

    def set_label_map(self, label_map: Dict[int, str]) -> None:
        """
        Set the mapping from label IDs to person names.

        Args:
            label_map: Dictionary mapping integer labels to string names
        """
        self.label_map = label_map

    def save_model(self, model_path: str, label_map_path: str) -> None:
        """
        Save the trained model and label map to disk.

        Args:
            model_path: Path to save the LBPH model (.yml file)
            label_map_path: Path to save the label map (.json file)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save LBPH model
        self.recognizer.save(model_path)
        print(f"Model saved to: {model_path}")

        # Save label map as JSON
        # Convert int keys to strings for JSON compatibility
        label_map_str = {str(k): v for k, v in self.label_map.items()}
        with open(label_map_path, "w") as f:
            json.dump(label_map_str, f, indent=2)
        print(f"Label map saved to: {label_map_path}")

    def load_model(self, model_path: str, label_map_path: str) -> bool:
        """
        Load a trained model and label map from disk.

        Args:
            model_path: Path to the LBPH model (.yml file)
            label_map_path: Path to the label map (.json file)

        Returns:
            True if loading was successful, False otherwise
        """
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        if not os.path.exists(label_map_path):
            print(f"Label map file not found: {label_map_path}")
            return False

        try:
            # Load LBPH model
            self.recognizer.read(model_path)
            print(f"Model loaded from: {model_path}")

            # Load label map
            with open(label_map_path, "r") as f:
                label_map_str = json.load(f)
            # Convert string keys back to integers
            self.label_map = {int(k): v for k, v in label_map_str.items()}
            print(f"Label map loaded: {list(self.label_map.values())}")

            self.is_trained = True
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def update(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """
        Update the model with additional training data (incremental learning).

        Args:
            faces: List of new grayscale face images
            labels: List of integer labels for the new faces
        """
        if not self.is_trained:
            # If not trained, just do initial training
            self.train(faces, labels)
            return

        faces_array = [np.array(face, dtype=np.uint8) for face in faces]
        labels_array = np.array(labels, dtype=np.int32)

        self.recognizer.update(faces_array, labels_array)
        print(f"Model updated with {len(faces)} new face images")


if __name__ == "__main__":
    # Simple test of the recognizer
    print("LBPH Face Recognizer Module")
    print("=" * 40)
    print("This module provides face recognition using LBPH.")
    print("\nHow LBPH works:")
    print("1. Computes Local Binary Patterns for each pixel")
    print("2. Creates histograms of LBP values per region")
    print("3. Compares histograms using Chi-Square distance")
    print("\nUse collect_faces.py to gather training data")
    print("Use train_model.py to train the LBPH model")
    print("Use recognize_faces.py for real-time recognition")
