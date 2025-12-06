"""
Face Detection Module using MediaPipe

This module provides face detection capabilities using Google's MediaPipe library.
It detects faces in video frames and returns bounding box coordinates.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Face detector using MediaPipe's Face Detection solution.

    This class wraps MediaPipe's face detection to provide a simple interface
    for detecting faces and extracting face regions from video frames.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the face detector.

        Args:
            min_detection_confidence: Minimum confidence threshold for face detection (0.0-1.0)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (within 2m), 1 for full-range
            min_detection_confidence=min_detection_confidence,
        )

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame and return bounding boxes.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        bboxes = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                # Get bounding box from relative coordinates
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                if width > 0 and height > 0:
                    bboxes.append((x, y, width, height))

        return bboxes

    def extract_face_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (200, 200),
    ) -> Optional[np.ndarray]:
        """
        Extract face region of interest (ROI) from frame and convert to grayscale.

        Args:
            frame: BGR image (OpenCV format)
            bbox: Bounding box as (x, y, width, height)
            target_size: Size to resize the face ROI to (width, height)

        Returns:
            Grayscale face ROI resized to target_size, or None if extraction fails
        """
        x, y, w, h = bbox

        # Extract face region
        face_roi = frame[y : y + h, x : x + w]

        if face_roi.size == 0:
            return None

        # Convert to grayscale (required for LBPH)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Resize to consistent size for recognition
        resized_face = cv2.resize(gray_face, target_size)

        return resized_face

    def draw_detections(
        self,
        frame: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        names: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: BGR image (OpenCV format)
            bboxes: List of bounding boxes
            names: Optional list of names to display
            confidences: Optional list of confidence values
            color: BGR color for the bounding box

        Returns:
            Frame with drawings
        """
        annotated_frame = frame.copy()

        for i, (x, y, w, h) in enumerate(bboxes):
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

            # Prepare label text
            label_parts = []
            if names and i < len(names):
                label_parts.append(names[i])
            if confidences and i < len(confidences):
                label_parts.append(f"{confidences[i]:.1f}%")

            if label_parts:
                label = " - ".join(label_parts)

                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    annotated_frame,
                    (x, y - text_h - 10),
                    (x + text_w + 10, y),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x + 5, y - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )

        return annotated_frame

    def close(self):
        """Release resources."""
        self.face_detection.close()


if __name__ == "__main__":
    # Simple test of face detection
    print("Testing Face Detector...")

    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for webcam

        # Detect faces
        bboxes = detector.detect_faces(frame)

        # Draw detections
        annotated = detector.draw_detections(
            frame, bboxes, names=[f"Face {i+1}" for i in range(len(bboxes))]
        )

        cv2.imshow("Face Detection Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Test complete!")
