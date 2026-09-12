"""Transparent rule-based fusion for the fire-rescue software prototype.

This is deliberately a baseline, not a trained machine-learning model.  It
combines processed environmental evidence, short-window temporal evidence,
and optional vision observations into explainable rescue-assessment outputs.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Union


FusionValue = Union[str, List[str]]
FusionResult = Dict[str, FusionValue]


class FusionDataValidationError(ValueError):
    """Raised when fusion inputs do not match the documented contracts."""


class FusionEngine:
    """Combine processed sensor, temporal, and optional vision evidence."""

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
        """Return hazard, victim, and rescue-priority assessments.

        ``vision_info`` is optional to keep the sensor pipeline independently
        usable.  Supported visual observations are ``person_detected``,
        ``confidence``, ``possible_vulnerable_pose``, and ``victim_signal``.
        A stationary sensor reading alone never creates a victim assessment.
        """
        sensor = self._validate_sensor_features(processed_features)
        temporal = self._validate_temporal_features(temporal_features)
        vision = self._normalize_vision(vision_info)

        hazard_level, hazard_evidence = self._assess_hazard(sensor, temporal)
        victim_state, victim_evidence = self._assess_victim(
            sensor, hazard_level, vision
        )
        rescue_priority, priority_evidence = self._assess_priority(
            hazard_level, victim_state, vision
        )

        return {
            "hazard_level": hazard_level,
            "victim_state": victim_state,
            "rescue_priority": rescue_priority,
            "decision_factors": hazard_evidence + victim_evidence + priority_evidence,
        }

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
        # ``possible_collapsed_pose`` remains accepted for compatibility with
        # earlier callers; new real vision observations use the clearer name.
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

    @staticmethod
    def _assess_hazard(
        sensor: Mapping[str, bool], temporal: Mapping[str, Union[bool, float]]
    ) -> tuple[str, List[str]]:
        current_hazards = sum(
            (sensor["high_gas"], sensor["high_temperature"], sensor["flame_detected"])
        )
        evidence: List[str] = []
        for field, label in (
            ("high_gas", "high gas/smoke indication"),
            ("high_temperature", "high temperature"),
            ("flame_detected", "flame detected"),
        ):
            if sensor[field]:
                evidence.append(label)

        # One cue is medium; two or three simultaneous environmental cues are high.
        if current_hazards == 0:
            hazard_level = "LOW"
        elif current_hazards == 1:
            hazard_level = "MEDIUM"
        else:
            hazard_level = "HIGH"

        persistent_environmental_cues = sum(
            float(temporal[field]) >= 0.6
            for field in (
                "high_gas_persistence",
                "high_temperature_persistence",
                "flame_detected_persistence",
            )
        )
        escalation = bool(temporal["environment_worsening"]) or bool(
            temporal["flame_persistent"]
        ) or persistent_environmental_cues >= 2
        if escalation:
            evidence.append("persistent or worsening environmental conditions")
            hazard_level = {
                "LOW": "MEDIUM",
                "MEDIUM": "HIGH",
                "HIGH": "CRITICAL",
            }[hazard_level]
        return hazard_level, evidence

    @staticmethod
    def _assess_victim(
        sensor: Mapping[str, bool], hazard_level: str, vision: Mapping[str, Union[bool, float]]
    ) -> tuple[str, List[str]]:
        person_or_explicit_signal = bool(vision["person_detected"]) or bool(
            vision["victim_signal"]
        )
        if not person_or_explicit_signal:
            # Low movement remains supporting context, never a victim claim.
            return "NONE", []

        evidence = ["person or explicit victim signal observed"]
        vulnerable_pose_or_stillness = bool(vision["possible_vulnerable_pose"]) or sensor[
            "low_movement"
        ]
        if vulnerable_pose_or_stillness:
            evidence.append("possible vulnerable visual posture or low movement")

        if hazard_level == "CRITICAL" and vulnerable_pose_or_stillness:
            return "CRITICAL", evidence
        # A visual lying candidate or low movement is supporting context, not
        # proof of a victim condition.  It elevates assessment only when an
        # environmental hazard independently corroborates the concern.
        if hazard_level in {"HIGH", "CRITICAL"} or (
            hazard_level == "MEDIUM" and vulnerable_pose_or_stillness
        ):
            return "AT_RISK", evidence
        return "POTENTIAL", evidence

    @staticmethod
    def _assess_priority(
        hazard_level: str, victim_state: str, vision: Mapping[str, Union[bool, float]]
    ) -> tuple[str, List[str]]:
        if victim_state == "NONE":
            return "NONE", []

        if victim_state == "CRITICAL":
            return "CRITICAL", ["critical victim evidence in a critical hazard"]
        if victim_state == "AT_RISK":
            if hazard_level == "CRITICAL":
                return "CRITICAL", ["at-risk person in a critical hazard"]
            if hazard_level in {"HIGH", "MEDIUM"}:
                return "HIGH", ["at-risk person with environmental hazard"]
            return "MEDIUM", ["at-risk person requires assessment"]

        # A person alone has low priority; danger must be corroborated to rise.
        if hazard_level == "CRITICAL":
            return "HIGH", ["potential person in a critical hazard"]
        if hazard_level == "HIGH":
            return "MEDIUM", ["potential person with high environmental hazard"]
        if hazard_level == "MEDIUM":
            return "LOW", ["potential person with moderate environmental hazard"]
        return "LOW", ["potential person observed"]


def fuse_assessment(
    processed_features: Mapping[str, object],
    temporal_features: Mapping[str, object],
    vision_info: Optional[Mapping[str, object]] = None,
) -> FusionResult:
    """Fuse one current reading, temporal context, and optional vision evidence."""
    return FusionEngine().fuse(processed_features, temporal_features, vision_info)


if __name__ == "__main__":
    # Permit direct execution as well as ``python -m fusion.fusion_engine``.
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
            "vision": {"person_detected": True, "confidence": 0.90, "possible_collapsed_pose": True},
        },
        "FIRE_WITH_VICTIM": {
            "history": [
                {"gas_level": 480, "temperature": 50, "flame_intensity": 70, "distance": 75, "movement": 0.10},
                {"gas_level": 670, "temperature": 68, "flame_intensity": 85, "distance": 65, "movement": 0.06},
                {"gas_level": 850, "temperature": 90, "flame_intensity": 95, "distance": 55, "movement": 0.03},
            ],
            "vision": {"person_detected": True, "confidence": 0.95, "possible_collapsed_pose": True},
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
