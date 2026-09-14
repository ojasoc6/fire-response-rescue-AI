"""Modular fire and smoke vision detection interface for search-and-rescue.

NOTE: This module currently operates in PLACEHOLDER mode. No fine-tuned
or pretrained fire/smoke vision model is loaded or trained yet. Placeholder
outputs are explicitly labeled as such to prevent false claims of active detection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ImageSource = Union[str, Path]
FireSmokeObservation = Dict[str, Any]


class FireSmokeDetector:
    """Modular detector interface for visual fire and smoke observations.

    In the current stage, no trained model is attached by default. The detector
    runs in an explicit PLACEHOLDER mode, returning zeroed detection values
    clearly flagged with detection_source='placeholder'.

    When a trained or fine-tuned model checkpoint (e.g. YOLO trained on D-Fire)
    becomes available, it can be passed via `model_path` or `custom_backend`
    without altering caller contracts.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        custom_backend: Optional[Callable[[ImageSource], FireSmokeObservation]] = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self._custom_backend = custom_backend
        self._model: Optional[Any] = None

        # Check if an actual model file exists
        self.is_model_available = (
            self.model_path is not None and self.model_path.is_file()
        )

    def detect(self, image: ImageSource) -> FireSmokeObservation:
        """Run fire and smoke detection on an image path.

        Returns a standardized observation dictionary containing:
        - fire_detected (bool)
        - fire_confidence (float)
        - smoke_detected (bool)
        - smoke_confidence (float)
        - fire_count (int)
        - smoke_count (int)

        Additional provenance fields:
        - detection_source ('placeholder' | 'model')
        - model_loaded (bool)
        """
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file was not found: {image_path}")

        # If a custom backend/callable is provided, delegate to it
        if self._custom_backend is not None:
            return self._custom_backend(image_path)

        # If an actual model path is provided and exists, execute model-based inference
        if self.is_model_available:
            return self._run_model_inference(image_path)

        # Otherwise return explicit placeholder output
        return self._placeholder_observation()

    def _placeholder_observation(self) -> FireSmokeObservation:
        """Return explicit, zeroed placeholder observation when no model is active."""
        return {
            "fire_detected": False,
            "fire_confidence": 0.0,
            "smoke_detected": False,
            "smoke_confidence": 0.0,
            "fire_count": 0,
            "smoke_count": 0,
            "detection_source": "placeholder",
            "model_loaded": False,
        }

    def _run_model_inference(self, image_path: Path) -> FireSmokeObservation:
        """Run inference using an Ultralytics YOLO model trained for fire/smoke."""
        model = self._load_model()
        results = model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return {
                "fire_detected": False,
                "fire_confidence": 0.0,
                "smoke_detected": False,
                "smoke_confidence": 0.0,
                "fire_count": 0,
                "smoke_count": 0,
                "detection_source": "model",
                "model_loaded": True,
            }

        boxes = results[0].boxes
        names = getattr(results[0], "names", {})

        fire_confidences: List[float] = []
        smoke_confidences: List[float] = []

        for class_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            class_name = str(names.get(int(class_id), "")).lower()
            if "fire" in class_name or "flame" in class_name:
                fire_confidences.append(float(conf))
            elif "smoke" in class_name:
                smoke_confidences.append(float(conf))

        return {
            "fire_detected": len(fire_confidences) > 0,
            "fire_confidence": round(max(fire_confidences), 4) if fire_confidences else 0.0,
            "smoke_detected": len(smoke_confidences) > 0,
            "smoke_confidence": round(max(smoke_confidences), 4) if smoke_confidences else 0.0,
            "fire_count": len(fire_confidences),
            "smoke_count": len(smoke_confidences),
            "detection_source": "model",
            "model_loaded": True,
        }

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.model_path is None or not self.model_path.is_file():
            raise RuntimeError("No valid model file available to load.")

        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        return self._model
    def _run_model_inference_frame(self, frame: Any) -> FireSmokeObservation:
        """Run inference directly on an OpenCV ndarray frame."""
        model = self._load_model()
        results = model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return {
                "fire_detected": False,
                "fire_confidence": 0.0,
                "smoke_detected": False,
                "smoke_confidence": 0.0,
                "fire_count": 0,
                "smoke_count": 0,
                "detection_source": "model",
                "model_loaded": True,
            }
        boxes = results[0].boxes
        names = getattr(results[0], "names", {})
        fire_confidences: List[float] = []
        smoke_confidences: List[float] = []
        for class_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            class_name = str(names.get(int(class_id), "")).lower()
            if "fire" in class_name or "flame" in class_name:
                fire_confidences.append(float(conf))
            elif "smoke" in class_name:
                smoke_confidences.append(float(conf))
        return {
            "fire_detected": len(fire_confidences) > 0,
            "fire_confidence": round(max(fire_confidences), 4) if fire_confidences else 0.0,
            "smoke_detected": len(smoke_confidences) > 0,
            "smoke_confidence": round(max(smoke_confidences), 4) if smoke_confidences else 0.0,
            "fire_count": len(fire_confidences),
            "smoke_count": len(smoke_confidences),
            "detection_source": "model",
            "model_loaded": True,
        }

    def detect_frame(self, frame: Any) -> FireSmokeObservation:
        """Run fire and smoke detection directly on a video frame (ndarray)."""
        if self._custom_backend is not None:
            return self._custom_backend(frame)
        if self.is_model_available:
            return self._run_model_inference_frame(frame)
        return self._placeholder_observation()

def detect_fire_smoke(
    image_path: ImageSource,
    model_path: Optional[Union[str, Path]] = None,
) -> FireSmokeObservation:
    """Convenience helper to run fire/smoke detection on a single image."""
    return FireSmokeDetector(model_path=model_path).detect(image_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run visual fire and smoke detection on an image."
    )
    parser.add_argument("image_path", type=Path, help="Path to input image file.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional path to trained fire/smoke YOLO checkpoint.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    args = parser.parse_args()

    detector = FireSmokeDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
    )

    try:
        observation = detector.detect(args.image_path)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if observation.get("detection_source") == "placeholder":
        print("[NOTE] Running in PLACEHOLDER mode. No fine-tuned fire/smoke model is active.")

    print(json.dumps(observation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
