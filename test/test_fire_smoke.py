"""Unit tests for the fire and smoke vision detection interface."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.fire_smoke_detection import FireSmokeDetector, detect_fire_smoke

SAMPLE_IMAGE = PROJECT_ROOT / "data" / "vision_test" / "standing.jpeg"

REQUIRED_KEYS = {
    "fire_detected",
    "fire_confidence",
    "smoke_detected",
    "smoke_confidence",
    "fire_count",
    "smoke_count",
}


class FireSmokeDetectionSchemaTests(unittest.TestCase):
    def test_placeholder_output_matches_required_schema(self) -> None:
        detector = FireSmokeDetector()
        obs = detector.detect(SAMPLE_IMAGE)

        # Verify all required schema keys are present
        for key in REQUIRED_KEYS:
            self.assertIn(key, obs)

        # Verify types
        self.assertIsInstance(obs["fire_detected"], bool)
        self.assertIsInstance(obs["fire_confidence"], float)
        self.assertIsInstance(obs["smoke_detected"], bool)
        self.assertIsInstance(obs["smoke_confidence"], float)
        self.assertIsInstance(obs["fire_count"], int)
        self.assertIsInstance(obs["smoke_count"], int)

        # Verify placeholder provenance
        self.assertEqual(obs.get("detection_source"), "placeholder")
        self.assertFalse(obs.get("model_loaded"))
        self.assertFalse(obs["fire_detected"])
        self.assertFalse(obs["smoke_detected"])

    def test_convenience_function_returns_valid_schema(self) -> None:
        obs = detect_fire_smoke(SAMPLE_IMAGE)
        self.assertTrue(REQUIRED_KEYS.issubset(obs.keys()))
        self.assertEqual(obs["detection_source"], "placeholder")

    def test_custom_backend_integration(self) -> None:
        def mock_trained_model(image_path: Path) -> dict[str, object]:
            return {
                "fire_detected": True,
                "fire_confidence": 0.88,
                "smoke_detected": True,
                "smoke_confidence": 0.72,
                "fire_count": 1,
                "smoke_count": 2,
                "detection_source": "model",
                "model_loaded": True,
            }

        detector = FireSmokeDetector(custom_backend=mock_trained_model)
        obs = detector.detect(SAMPLE_IMAGE)

        self.assertTrue(obs["fire_detected"])
        self.assertEqual(obs["fire_confidence"], 0.88)
        self.assertTrue(obs["smoke_detected"])
        self.assertEqual(obs["smoke_confidence"], 0.72)
        self.assertEqual(obs["fire_count"], 1)
        self.assertEqual(obs["smoke_count"], 2)
        self.assertEqual(obs["detection_source"], "model")

    def test_missing_image_raises_file_not_found(self) -> None:
        detector = FireSmokeDetector()
        with self.assertRaises(FileNotFoundError):
            detector.detect("non_existent_image.jpeg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
