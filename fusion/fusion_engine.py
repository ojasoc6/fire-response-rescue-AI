"""Transparent multimodal cross-verification and triage fusion engine.

Implements 4 coordinated evaluation layers:
1. Environmental Threat Cross-Verification (IR Flame, Gas, Vision Fire, Vision Smoke)
2. Victim Triage & Priority (Posture + Environmental Hazard Level)
3. Navigation Advisory (3x Ultrasonic Clearance)
4. Robot Platform Safety (MPU-6050 IMU Acceleration & Tilt)
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


FusionValue = Union[str, List[str]]
FusionResult = Dict[str, FusionValue]


class FusionDataValidationError(ValueError):
    """Raised when fusion inputs do not match the documented contracts."""


class FusionEngine:
    """Coordinate multi-sensor cross-verification, triage, navigation, and platform safety."""

    _SENSOR_FLAGS = ("high_gas", "high_temperature", "flame_detected", "low_movement")
    _TEMPORAL_FIELDS = (
        "environment_worsening",
        "flame_persistent",
        "high_gas_persistence",
        "high_temperature_persistence",
        "flame_detected_persistence",
    )

    def fuse(
        self,
        processed_features: Mapping[str, object],
        temporal_features: Mapping[str, object],
        vision_info: Optional[Mapping[str, object]] = None,
    ) -> FusionResult:
        """Return cross-verified threat, victim triage, navigation advisory, and platform safety.

        Coordinated layers:
        A. Environmental Threat Cross-Verification
        B. Victim Triage & Priority
        C. Navigation Advisory (3x Ultrasonic Clearance)
        D. Robot Platform Safety (MPU-6050 IMU)
        """
        sensor = self._validate_sensor_features(processed_features)
        temporal = self._validate_temporal_features(temporal_features)
        vision = self._normalize_vision(vision_info)

        # Layer A: Environmental Threat Cross-Verification
        hazard_level, hazard_desc, confirming = self._assess_threat(processed_features, vision_info)

        # Layer B: Victim Triage & Priority
        victim_state, priority = self._assess_victim_triage(hazard_level, vision_info, vision)

        # Layer C: Navigation Advisory
        nav_advisory = self._assess_navigation(processed_features, vision_info)

        # Layer D: Robot Platform Safety
        robot_status = self._assess_robot_safety(vision_info)

        decision_factors: List[str] = []
        if confirming:
            decision_factors.append(
                f"Threat: {hazard_desc} ({len(confirming)} confirming modalities: {', '.join(confirming)})"
            )
        else:
            decision_factors.append(f"Threat: {hazard_desc} (No active fire/smoke modalities tripped)")

        if bool(temporal.get("environment_worsening")) or bool(temporal.get("flame_persistent")):
            decision_factors.append("Environmental trend: persistent or worsening hazard conditions")

        decision_factors.append(f"Victim triage: {victim_state} (Priority: {priority})")
        decision_factors.append(f"Navigation advisory: {nav_advisory}")
        decision_factors.append(f"Robot platform safety: {robot_status}")

        return {
            "hazard_level": hazard_level,
            "hazard_desc": hazard_desc,
            "victim_state": victim_state,
            "priority": priority,
            "rescue_priority": priority,
            "nav_advisory": nav_advisory,
            "robot_status": robot_status,
            "decision_factors": decision_factors,
        }

    def _assess_threat(
        self,
        processed_features: Mapping[str, object],
        vision_info: Optional[Mapping[str, object]],
    ) -> Tuple[str, str, List[str]]:
        """Evaluate environmental threat using 4 independent modalities:

        - IR Flame sensor
        - Gas/Smoke sensor (spike >= 300)
        - Vision Fire detector (conf >= 0.50)
        - Vision Smoke detector (conf >= 0.60)
        """
        confirming: List[str] = []

        # 1. IR Flame
        flame_detected_raw = False
        if vision_info and bool(vision_info.get("flame_detected", False)):
            flame_detected_raw = True
        elif bool(processed_features.get("flame_detected", False)):
            flame_detected_raw = True
        elif float(processed_features.get("flame_intensity", 0.0)) >= 60.0:
            flame_detected_raw = True

        if flame_detected_raw:
            confirming.append("IR Flame")

        # 2. Gas Spike (Gas >= 300 or explicit alert)
        gas_spike = False
        if vision_info:
            if float(vision_info.get("gas_raw", 0.0)) >= 300.0:
                gas_spike = True
            elif float(vision_info.get("gas_level", 0.0)) >= 300.0:
                gas_spike = True
            elif bool(vision_info.get("gas_alert", False)):
                gas_spike = True

        if not gas_spike:
            if float(processed_features.get("gas_level", 0.0)) >= 300.0:
                gas_spike = True
            elif float(processed_features.get("gas_raw", 0.0)) >= 300.0:
                gas_spike = True
            elif bool(processed_features.get("high_gas", False)):
                gas_spike = True
            elif bool(processed_features.get("gas_alert", False)):
                gas_spike = True

        if gas_spike:
            confirming.append("Gas Spike (>=300)")

        # 3. Vision Fire (confidence >= 0.50)
        vision_fire = False
        if vision_info:
            fire_conf = float(vision_info.get("fire_confidence", 0.0))
            if fire_conf >= 0.50:
                vision_fire = True
            elif bool(vision_info.get("fire_detected", False)) and fire_conf == 0.0:
                # If boolean is explicitly True without confidence breakdown, default to high confidence
                vision_fire = True

        if vision_fire:
            confirming.append("Vision Fire (>=0.50)")

        # 4. Vision Smoke (confidence >= 0.60)
        vision_smoke = False
        if vision_info:
            smoke_conf = float(vision_info.get("smoke_confidence", 0.0))
            if smoke_conf >= 0.60:
                vision_smoke = True
            elif bool(vision_info.get("smoke_detected", False)) and smoke_conf == 0.0:
                vision_smoke = True

        if vision_smoke:
            confirming.append("Vision Smoke (>=0.60)")

        count = len(confirming)
        if count == 0:
            return "LOW", "CLEAR", confirming
        elif count == 1:
            return "MODERATE", "UNCONFIRMED_ALERT (SINGLE SENSOR)", confirming
        elif count == 2:
            return "HIGH", "ACTIVE_HAZARD_CONFIRMED", confirming
        else:
            return "CRITICAL", "STRUCTURAL_FIRE_CONFIRMED", confirming

    def _assess_victim_triage(
        self,
        hazard_level: str,
        vision_info: Optional[Mapping[str, object]],
        normalized_vision: Mapping[str, Union[bool, float]],
    ) -> Tuple[str, str]:
        """Triage victim state and rescue priority based on posture and environmental hazard.

        Rules:
        - If no person detected: victim_state: NONE, priority: LOW
        - If person detected:
            - Posture == LYING_DOWN: victim_state: CRITICAL_INCAPACITATED, priority: CRITICAL
            - Posture == SITTING and Hazard in [HIGH, CRITICAL]: victim_state: AT_RISK, priority: HIGH
            - Posture == SITTING and Hazard in [LOW, MODERATE]: victim_state: POTENTIAL, priority: MODERATE
            - Posture == STANDING and Hazard in [HIGH, CRITICAL]: victim_state: TRAPPED_MOBILE, priority: HIGH
            - Posture == STANDING and Hazard in [LOW, MODERATE]: victim_state: STABLE_MOBILE, priority: LOW
        """
        person_detected = bool(normalized_vision.get("person_detected", False) or normalized_vision.get("victim_signal", False))
        if not person_detected or not vision_info:
            return "NONE", "LOW"

        posture_str = str(vision_info.get("posture", "")).strip().upper()
        vulnerable = bool(
            vision_info.get("possible_vulnerable_pose", False)
            or vision_info.get("possible_collapsed_pose", False)
            or normalized_vision.get("possible_vulnerable_pose", False)
        )

        is_lying = posture_str in ("LYING_DOWN", "LYING_CANDIDATE", "LYING") or vulnerable
        is_sitting = posture_str in ("SITTING", "SEATED")
        is_standing = posture_str in ("STANDING", "UPRIGHT")

        if is_lying:
            return "CRITICAL_INCAPACITATED", "CRITICAL"
        elif is_sitting:
            if hazard_level in ("HIGH", "CRITICAL"):
                return "AT_RISK", "HIGH"
            else:
                return "POTENTIAL", "MODERATE"
        elif is_standing:
            if hazard_level in ("HIGH", "CRITICAL"):
                return "TRAPPED_MOBILE", "HIGH"
            else:
                return "STABLE_MOBILE", "LOW"
        else:
            # Default / UNKNOWN posture
            if hazard_level in ("HIGH", "CRITICAL"):
                return "AT_RISK", "HIGH"
            else:
                return "POTENTIAL", "MODERATE"

    def _assess_navigation(
        self,
        processed_features: Mapping[str, object],
        vision_info: Optional[Mapping[str, object]],
    ) -> str:
        """Evaluate 3x ultrasonic distance clearance (front, left, right in cm).

        Rules:
        - If front < 0 or all -1.0: SENSOR_OFFLINE
        - If front < 20 and left < 20 and right < 20: TRAPPED_DEAD_END
        - If front < 30:
            - If left >= right and left >= 30: DETOUR_LEFT
            - Else if right > left and right >= 30: DETOUR_RIGHT
            - Else: OBSTACLE_AHEAD_REVERSE
        - If front >= 30: PATH_CLEAR
        """
        front = -1.0
        left = -1.0
        right = -1.0

        if vision_info and "front_cm" in vision_info:
            front = float(vision_info["front_cm"])
        elif "front_cm" in processed_features:
            front = float(processed_features["front_cm"])
        elif "distance" in processed_features:
            front = float(processed_features["distance"])

        if vision_info and "left_cm" in vision_info:
            left = float(vision_info["left_cm"])
        elif "left_cm" in processed_features:
            left = float(processed_features["left_cm"])

        if vision_info and "right_cm" in vision_info:
            right = float(vision_info["right_cm"])
        elif "right_cm" in processed_features:
            right = float(processed_features["right_cm"])

        # Check offline
        if front < 0 or (front == -1.0 and left == -1.0 and right == -1.0):
            return "SENSOR_OFFLINE"

        # Check trapped in dead end
        if front < 20 and 0 <= left < 20 and 0 <= right < 20:
            return "TRAPPED_DEAD_END"

        # Check obstacle ahead
        if front < 30:
            if left >= right and left >= 30:
                return "DETOUR_LEFT"
            elif right > left and right >= 30:
                return "DETOUR_RIGHT"
            else:
                return "OBSTACLE_AHEAD_REVERSE"

        return "PATH_CLEAR"

    def _assess_robot_safety(
        self,
        vision_info: Optional[Mapping[str, object]],
    ) -> str:
        """Evaluate MPU-6050 IMU acceleration and tilt status.

        Rules:
        - If tilt == True or abs(acc_z) < 0.6 or sqrt(acc_x^2 + acc_y^2) > 0.8:
            ROLLOVER_RISK (HALT)
        - Else if sqrt(acc_x^2 + acc_y^2 + acc_z^2) > 2.0:
            IMPACT_DETECTED
        - Else:
            STABLE (LEVEL)
        """
        acc_x = 0.0
        acc_y = 0.0
        acc_z = 1.0
        tilt = False

        if vision_info:
            acc_x = float(vision_info.get("acc_x", 0.0))
            acc_y = float(vision_info.get("acc_y", 0.0))
            acc_z = float(vision_info.get("acc_z", 1.0))
            tilt = bool(vision_info.get("tilt", False))

        xy_mag = math.sqrt(acc_x**2 + acc_y**2)
        total_mag = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        if tilt or abs(acc_z) < 0.6 or xy_mag > 0.8:
            return "ROLLOVER_RISK (HALT)"
        elif total_mag > 2.0:
            return "IMPACT_DETECTED"
        else:
            return "STABLE (LEVEL)"

    def _validate_sensor_features(self, features: Mapping[str, object]) -> Dict[str, bool]:
        if not isinstance(features, Mapping):
            raise FusionDataValidationError("processed_features must be a mapping")
        missing = [field for field in self._SENSOR_FLAGS if field not in features]
        if missing:
            raise FusionDataValidationError(
                f"processed_features is missing: {', '.join(missing)}"
            )
        return {field: bool(features[field]) for field in self._SENSOR_FLAGS}

    def _validate_temporal_features(
        self, features: Mapping[str, object]
    ) -> Dict[str, Union[bool, float]]:
        if not isinstance(features, Mapping):
            raise FusionDataValidationError("temporal_features must be a mapping")
        missing = [field for field in self._TEMPORAL_FIELDS if field not in features]
        if missing:
            raise FusionDataValidationError(
                f"temporal_features is missing: {', '.join(missing)}"
            )

        temporal: Dict[str, Union[bool, float]] = {
            "environment_worsening": bool(features["environment_worsening"]),
            "flame_persistent": bool(features["flame_persistent"]),
        }
        for field in self._TEMPORAL_FIELDS[2:]:
            try:
                value = float(features[field])
            except (TypeError, ValueError) as error:
                raise FusionDataValidationError(f"{field} must be numeric") from error
            if not 0.0 <= value <= 1.0:
                raise FusionDataValidationError(f"{field} must be between 0.0 and 1.0")
            temporal[field] = value
        return temporal

    @staticmethod
    def _normalize_vision(
        vision_info: Optional[Mapping[str, object]],
    ) -> Dict[str, Union[bool, float]]:
        if vision_info is None:
            return {
                "person_detected": False,
                "possible_vulnerable_pose": False,
                "victim_signal": False,
                "confidence": 0.0,
            }
        if not isinstance(vision_info, Mapping):
            raise FusionDataValidationError("vision_info must be a mapping or None")

        try:
            confidence = float(vision_info.get("confidence", 1.0))
        except (TypeError, ValueError) as error:
            raise FusionDataValidationError("vision confidence must be numeric") from error
        if not 0.0 <= confidence <= 1.0:
            raise FusionDataValidationError("vision confidence must be between 0.0 and 1.0")

        # Low-confidence sightings are retained outside fusion until confirmed.
        person_detected = bool(vision_info.get("person_detected", False)) and confidence >= 0.5
        possible_vulnerable_pose = bool(
            vision_info.get(
                "possible_vulnerable_pose",
                vision_info.get("possible_collapsed_pose", False),
            )
        )
        return {
            "person_detected": person_detected,
            "possible_vulnerable_pose": possible_vulnerable_pose,
            "victim_signal": bool(vision_info.get("victim_signal", False)),
            "confidence": confidence,
        }


def fuse_assessment(
    processed_features: Mapping[str, object],
    temporal_features: Mapping[str, object],
    vision_info: Optional[Mapping[str, object]] = None,
) -> FusionResult:
    """Fuse one current reading, temporal context, and optional vision evidence."""
    return FusionEngine().fuse(processed_features, temporal_features, vision_info)


if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from fusion.feature_processing import process_sensor_reading
    from temporal.temporal_features import extract_temporal_features

    scenarios = {
        "NORMAL": {
            "history": [
                {"gas_level": 145, "temperature": 25, "flame_intensity": 1, "distance": 150, "movement": 0.70},
                {"gas_level": 150, "temperature": 25, "flame_intensity": 1, "distance": 150, "movement": 0.68},
                {"gas_level": 148, "temperature": 26, "flame_intensity": 2, "distance": 145, "movement": 0.72},
            ],
            "vision": None,
        },
        "FIRE": {
            "history": [
                {"gas_level": 450, "temperature": 48, "flame_intensity": 65, "distance": 100, "movement": 0.50},
                {"gas_level": 620, "temperature": 60, "flame_intensity": 75, "distance": 90, "movement": 0.35},
                {"gas_level": 790, "temperature": 75, "flame_intensity": 90, "distance": 80, "movement": 0.20},
            ],
            "vision": None,
        },
        "VICTIM": {
            "history": [
                {"gas_level": 150, "temperature": 25, "flame_intensity": 1, "distance": 80, "movement": 0.08},
                {"gas_level": 155, "temperature": 26, "flame_intensity": 1, "distance": 75, "movement": 0.06},
                {"gas_level": 150, "temperature": 25, "flame_intensity": 1, "distance": 70, "movement": 0.05},
            ],
            "vision": {"person_detected": True, "confidence": 0.90, "posture": "LYING_DOWN"},
        },
        "FIRE_WITH_VICTIM": {
            "history": [
                {"gas_level": 480, "temperature": 50, "flame_intensity": 70, "distance": 75, "movement": 0.10},
                {"gas_level": 670, "temperature": 68, "flame_intensity": 85, "distance": 65, "movement": 0.06},
                {"gas_level": 850, "temperature": 90, "flame_intensity": 95, "distance": 55, "movement": 0.03},
            ],
            "vision": {"person_detected": True, "confidence": 0.95, "posture": "LYING_DOWN", "fire_detected": True, "fire_confidence": 0.90},
        },
    }

    engine = FusionEngine()
    for scenario, inputs in scenarios.items():
        processed_history = [process_sensor_reading(reading) for reading in inputs["history"]]
        processed_evidence = processed_history[-1]
        temporal_evidence = extract_temporal_features(processed_history)
        assessment = engine.fuse(processed_evidence, temporal_evidence, inputs["vision"])
        print(f"\n{scenario}")
        print(f"Processed sensor evidence: {processed_evidence}")
        print(f"Temporal evidence: {temporal_evidence}")
        print(f"Assessment: {assessment}")
