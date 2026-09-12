"""Deterministic end-to-end tests for the rule-based rescue pipeline."""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.feature_processing import FeatureProcessor, SensorDataValidationError
from fusion.fusion_engine import FusionEngine
from simulation.sensor_simulator import SensorSimulator
from temporal.temporal_features import TemporalFeatureExtractor
from vision.vision_integration import VisionIntegrator


REQUIRED_SENSOR_FIELDS = {
    "gas_level",
    "temperature",
    "flame_intensity",
    "distance",
    "movement",
}


def process_history(raw_history: list[dict[str, float]]) -> list[dict[str, object]]:
    """Produce the processed history consumed by temporal and fusion tests."""
    processor = FeatureProcessor()
    return [processor.process(reading) for reading in raw_history]


def stable_history() -> list[dict[str, object]]:
    return process_history(
        [
            {"gas_level": 150, "temperature": 25, "flame_intensity": 1, "distance": 150, "movement": 0.70},
            {"gas_level": 155, "temperature": 25.5, "flame_intensity": 1, "distance": 149, "movement": 0.68},
            {"gas_level": 148, "temperature": 24.8, "flame_intensity": 2, "distance": 151, "movement": 0.72},
        ]
    )


def worsening_fire_history() -> list[dict[str, object]]:
    return process_history(
        [
            {"gas_level": 320, "temperature": 29, "flame_intensity": 65, "distance": 100, "movement": 0.65},
            {"gas_level": 500, "temperature": 45, "flame_intensity": 75, "distance": 90, "movement": 0.40},
            {"gas_level": 680, "temperature": 65, "flame_intensity": 85, "distance": 80, "movement": 0.15},
            {"gas_level": 850, "temperature": 85, "flame_intensity": 95, "distance": 70, "movement": 0.05},
        ]
    )


class SensorSimulatorTests(unittest.TestCase):
    def test_all_scenarios_provide_the_sensor_contract(self) -> None:
        simulator = SensorSimulator(seed=10)
        for scenario in (
            SensorSimulator.NORMAL,
            SensorSimulator.FIRE,
            SensorSimulator.VICTIM,
            SensorSimulator.FIRE_WITH_VICTIM,
        ):
            with self.subTest(scenario=scenario):
                reading = simulator.generate_reading(scenario)
                self.assertEqual(set(reading), REQUIRED_SENSOR_FIELDS)
                self.assertTrue(all(isinstance(value, float) for value in reading.values()))

    def test_seeded_runs_are_reproducible(self) -> None:
        first = SensorSimulator(seed=99)
        second = SensorSimulator(seed=99)
        first_readings = [first.generate_reading(SensorSimulator.FIRE) for _ in range(3)]
        second_readings = [second.generate_reading(SensorSimulator.FIRE) for _ in range(3)]
        self.assertEqual(first_readings, second_readings)


class FeatureProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = FeatureProcessor()
        self.valid_reading = {
            "gas_level": 650,
            "temperature": 70,
            "flame_intensity": 80,
            "distance": 100,
            "movement": 0.10,
        }

    def test_valid_reading_is_normalized_and_derived(self) -> None:
        features = self.processor.process(self.valid_reading)
        self.assertEqual(features["gas_level"], 650.0)
        self.assertEqual(features["gas_level_normalized"], 0.65)
        self.assertEqual(features["movement_normalized"], 0.1)
        self.assertTrue(features["high_gas"])
        self.assertTrue(features["high_temperature"])
        self.assertTrue(features["flame_detected"])
        self.assertTrue(features["low_movement"])

    def test_invalid_missing_nan_and_infinite_values_are_rejected(self) -> None:
        invalid_inputs = [
            {"gas_level": 1},
            {**self.valid_reading, "gas_level": "not-a-number"},
            {**self.valid_reading, "temperature": math.nan},
            {**self.valid_reading, "flame_intensity": math.inf},
        ]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(SensorDataValidationError):
                    self.processor.process(invalid_input)


class TemporalAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = TemporalFeatureExtractor()

    def test_stable_history_does_not_indicate_worsening(self) -> None:
        temporal = self.extractor.extract(stable_history())
        self.assertFalse(temporal["gas_rising"])
        self.assertFalse(temporal["temperature_rising"])
        self.assertFalse(temporal["flame_persistent"])
        self.assertFalse(temporal["environment_worsening"])

    def test_worsening_fire_history_sets_environmental_indicators(self) -> None:
        temporal = self.extractor.extract(worsening_fire_history())
        self.assertGreater(temporal["gas_slope"], 0)
        self.assertGreater(temporal["temperature_slope"], 0)
        self.assertTrue(temporal["gas_rising"])
        self.assertTrue(temporal["temperature_rising"])
        self.assertTrue(temporal["flame_persistent"])
        self.assertTrue(temporal["environment_worsening"])

    def test_single_reading_history_is_safe(self) -> None:
        temporal = self.extractor.extract(stable_history()[:1])
        self.assertEqual(temporal["gas_slope"], 0.0)
        self.assertEqual(temporal["temperature_slope"], 0.0)
        self.assertFalse(temporal["environment_worsening"])


class FusionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FusionEngine()
        self.extractor = TemporalFeatureExtractor()

    def test_normal_assessment_is_low_none_none(self) -> None:
        history = stable_history()
        assessment = self.engine.fuse(history[-1], self.extractor.extract(history))
        self.assertEqual(assessment["hazard_level"], "LOW")
        self.assertEqual(assessment["victim_state"], "NONE")
        self.assertEqual(assessment["rescue_priority"], "NONE")

    def test_fire_has_elevated_hazard_without_a_victim_claim(self) -> None:
        history = worsening_fire_history()
        assessment = self.engine.fuse(history[-1], self.extractor.extract(history))
        self.assertIn(assessment["hazard_level"], {"HIGH", "CRITICAL"})
        self.assertEqual(assessment["victim_state"], "NONE")
        self.assertEqual(assessment["rescue_priority"], "NONE")

    def test_person_with_vulnerable_evidence_without_hazard_is_potential(self) -> None:
        history = process_history(
            [
                {"gas_level": 150, "temperature": 25, "flame_intensity": 1, "distance": 80, "movement": 0.08},
                {"gas_level": 155, "temperature": 26, "flame_intensity": 1, "distance": 75, "movement": 0.05},
            ]
        )
        assessment = self.engine.fuse(
            history[-1],
            self.extractor.extract(history),
            {"person_detected": True, "confidence": 0.9, "possible_collapsed_pose": True},
        )
        self.assertEqual(assessment["victim_state"], "POTENTIAL")
        self.assertEqual(assessment["rescue_priority"], "LOW")

    def test_fire_person_and_vulnerable_evidence_is_critical(self) -> None:
        history = worsening_fire_history()
        assessment = self.engine.fuse(
            history[-1],
            self.extractor.extract(history),
            {"person_detected": True, "confidence": 0.9, "possible_collapsed_pose": True},
        )
        self.assertEqual(assessment["hazard_level"], "CRITICAL")
        self.assertEqual(assessment["victim_state"], "CRITICAL")
        self.assertEqual(assessment["rescue_priority"], "CRITICAL")

    def test_low_movement_alone_never_claims_a_victim(self) -> None:
        history = process_history(
            [
                {"gas_level": 150, "temperature": 25, "flame_intensity": 1, "distance": 80, "movement": 0.05},
                {"gas_level": 152, "temperature": 25, "flame_intensity": 1, "distance": 78, "movement": 0.04},
            ]
        )
        assessment = self.engine.fuse(history[-1], self.extractor.extract(history))
        self.assertEqual(assessment["victim_state"], "NONE")
        self.assertEqual(assessment["rescue_priority"], "NONE")

    def test_standardized_upright_vision_is_not_vulnerable_pose_evidence(self) -> None:
        history = stable_history()
        vision = {
            "person_detected": True,
            "person_count": 1,
            "confidence": 0.9,
            "bounding_box": {"x_min": 1, "y_min": 1, "x_max": 10, "y_max": 10},
            "posture": "UPRIGHT",
            "possible_vulnerable_pose": False,
        }
        assessment = self.engine.fuse(history[-1], self.extractor.extract(history), vision)
        self.assertEqual(assessment["victim_state"], "POTENTIAL")

    def test_standardized_lying_pose_is_supporting_not_diagnostic_evidence(self) -> None:
        history = stable_history()
        vision = {
            "person_detected": True,
            "person_count": 1,
            "confidence": 0.9,
            "bounding_box": {"x_min": 1, "y_min": 1, "x_max": 10, "y_max": 10},
            "posture": "LYING_CANDIDATE",
            "possible_vulnerable_pose": True,
        }
        assessment = self.engine.fuse(history[-1], self.extractor.extract(history), vision)
        self.assertEqual(assessment["victim_state"], "POTENTIAL")

    def test_standardized_no_person_observation_creates_no_victim_claim(self) -> None:
        history = stable_history()
        vision = {
            "person_detected": False,
            "person_count": 0,
            "confidence": 0.0,
            "bounding_box": None,
            "posture": "UNKNOWN",
            "possible_vulnerable_pose": False,
        }
        assessment = self.engine.fuse(history[-1], self.extractor.extract(history), vision)
        self.assertEqual(assessment["victim_state"], "NONE")


class VisionIntegrationTests(unittest.TestCase):
    def test_matching_pose_for_selected_person_builds_standard_observation(self) -> None:
        class FakeDetector:
            def detect(self, image_path: str) -> dict[str, object]:
                return {
                    "person_detected": True,
                    "person_count": 2,
                    "confidence": 0.95,
                    "bounding_box": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
                }

        def fake_pose_estimator(image_path: str, **_: object) -> dict[str, object]:
            return {
                "persons": [
                    {"posture": "UPRIGHT", "detection_confidence": 0.99,
                     "bounding_box": {"x_min": 0, "y_min": 0, "x_max": 40, "y_max": 40}},
                    {"posture": "LYING_CANDIDATE", "detection_confidence": 0.80,
                     "bounding_box": {"x_min": 55, "y_min": 55, "x_max": 145, "y_max": 145}},
                ]
            }

        observation = VisionIntegrator(FakeDetector(), fake_pose_estimator).analyze_image("image.jpg")
        self.assertTrue(observation["person_detected"])
        self.assertEqual(observation["person_count"], 2)
        self.assertEqual(observation["posture"], "LYING_CANDIDATE")
        self.assertTrue(observation["possible_vulnerable_pose"])
        self.assertEqual(observation["detection_source"], "person_detector+pose")

    def test_pose_fallback_uses_qualified_pose_when_detector_finds_nobody(self) -> None:
        class FakeDetector:
            def detect(self, image_path: str) -> dict[str, object]:
                return {
                    "person_detected": False,
                    "person_count": 0,
                    "confidence": 0.0,
                    "bounding_box": None,
                }

        def fake_pose_estimator(image_path: str, **_: object) -> dict[str, object]:
            return {
                "person_count": 1,
                "persons": [
                    {
                        "posture": "LYING_CANDIDATE",
                        "detection_confidence": 0.70,
                        "visible_keypoints": 12,
                        "bounding_box": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                    }
                ],
            }

        observation = VisionIntegrator(FakeDetector(), fake_pose_estimator).analyze_image("image.jpg")
        self.assertTrue(observation["person_detected"])
        self.assertEqual(observation["confidence"], 0.70)
        self.assertEqual(observation["posture"], "LYING_CANDIDATE")
        self.assertTrue(observation["possible_vulnerable_pose"])
        self.assertEqual(observation["detection_source"], "pose_fallback")

    def test_no_detection_returns_safe_unknown_observation(self) -> None:
        class FakeDetector:
            def detect(self, image_path: str) -> dict[str, object]:
                return {
                    "person_detected": False,
                    "person_count": 0,
                    "confidence": 0.0,
                    "bounding_box": None,
                }

        def fake_pose_estimator(image_path: str, **_: object) -> dict[str, object]:
            return {"person_count": 0, "persons": []}

        observation = VisionIntegrator(FakeDetector(), fake_pose_estimator).analyze_image("image.jpg")
        self.assertFalse(observation["person_detected"])
        self.assertEqual(observation["posture"], "UNKNOWN")
        self.assertFalse(observation["possible_vulnerable_pose"])
        self.assertEqual(observation["detection_source"], "none")


class EndToEndTests(unittest.TestCase):
    def test_main_runs_all_four_scenarios_successfully(self) -> None:
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for scenario in ("NORMAL", "FIRE", "VICTIM", "FIRE_WITH_VICTIM"):
            self.assertIn(f"Scenario: {scenario} (simulated)", completed.stdout)
        self.assertIn("Demonstration completed successfully", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
