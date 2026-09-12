"""Pretrained YOLO person observations for the fire-rescue prototype.

This module detects the COCO ``person`` class only.  A detected person is an
image observation, not evidence of injury, consciousness, or victim status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Union


ImageSource = object
PersonObservation = Dict[str, Union[bool, float, int, Optional[Dict[str, float]]]]


class VisionDependencyError(RuntimeError):
    """Raised when Ultralytics or the requested pretrained checkpoint is unavailable."""


class PersonDetector:
    """Lazy-loading wrapper around a pretrained Ultralytics YOLO detector.

    ``yolov8n.pt`` is the small general-purpose COCO checkpoint selected for
    this first CPU-friendly prototype.  The interface is intentionally limited
    to person observations so a future pose-estimation component can be added
    separately without changing sensor or fusion modules.
    """

    PERSON_CLASS_ID = 0  # COCO class 0 is "person".

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.4) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model: object | None = None

    def detect(self, image: ImageSource) -> PersonObservation:
        """Return standardized person observations for an image path or frame.

        ``image`` may be a local image path, a PIL image, or a NumPy/OpenCV
        image frame accepted by Ultralytics.  The returned bounding box belongs
        to the most confident detected person and uses pixel coordinates:
        ``x_min``, ``y_min``, ``x_max``, and ``y_max``.
        """
        if isinstance(image, (str, Path)) and not Path(image).is_file():
            raise FileNotFoundError(f"Image file was not found: {image}")

        model = self._load_model()
        try:
            results = model.predict(
                source=image,
                conf=self.confidence_threshold,
                classes=[self.PERSON_CLASS_ID],
                verbose=False,
            )
        except Exception as error:
            raise VisionDependencyError(f"YOLO prediction failed: {error}") from error

        if not results or results[0].boxes is None:
            return self._no_person_observation()

        person_detections = []
        boxes = results[0].boxes
        for class_id, confidence, coordinates in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            if int(class_id) != self.PERSON_CLASS_ID:
                continue
            person_detections.append(
                {
                    "confidence": float(confidence),
                    "bounding_box": {
                        "x_min": float(coordinates[0]),
                        "y_min": float(coordinates[1]),
                        "x_max": float(coordinates[2]),
                        "y_max": float(coordinates[3]),
                    },
                }
            )

        if not person_detections:
            return self._no_person_observation()

        best_detection = max(person_detections, key=lambda detection: detection["confidence"])
        best_box = best_detection["bounding_box"]
        assert isinstance(best_box, dict)
        return {
            "person_detected": True,
            "confidence": round(float(best_detection["confidence"]), 4),
            "bounding_box": {key: round(value, 2) for key, value in best_box.items()},
            "person_count": len(person_detections),
        }

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise VisionDependencyError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from error

        try:
            self._model = YOLO(self.model_name)
        except Exception as error:
            raise VisionDependencyError(
                f"Could not load pretrained checkpoint {self.model_name!r}. "
                "Check the model name, local file, or network access for the initial download."
            ) from error
        return self._model

    @staticmethod
    def _no_person_observation() -> PersonObservation:
        return {
            "person_detected": False,
            "confidence": 0.0,
            "bounding_box": None,
            "person_count": 0,
        }


def detect_person(
    image: ImageSource,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.4,
) -> PersonObservation:
    """Detect people in one image without manually constructing a detector."""
    return PersonDetector(model_name, confidence_threshold).detect(image)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people in one image with a pretrained Ultralytics YOLO model."
    )
    parser.add_argument("image", help="Path to a local image file.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint name or path.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Minimum person-detection confidence from 0.0 to 1.0.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the one-image command-line demo."""
    arguments = parse_arguments()
    try:
        observation = detect_person(
            arguments.image,
            model_name=arguments.model,
            confidence_threshold=arguments.confidence,
        )
    except (FileNotFoundError, ValueError, VisionDependencyError) as error:
        print(f"Person detection error: {error}", file=sys.stderr)
        return 1

    print("Person observation (not a victim diagnosis):")
    print(observation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
