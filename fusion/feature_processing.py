"""Validation and lightweight feature preparation for simulated sensor data.

The processor accepts the five-field dictionary produced by
``simulation.sensor_simulator.SensorSimulator``.  It has no model dependency:
the normalized and derived values are intended for the later fusion layer.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Union


FeatureValue = Union[float, bool]
ProcessedFeatures = Dict[str, FeatureValue]


class SensorDataValidationError(ValueError):
    """Raised when an input does not satisfy the sensor-reading contract."""


class FeatureProcessor:
    """Validate one sensor reading and produce deterministic fusion features."""

    REQUIRED_FIELDS = (
        "gas_level",
        "temperature",
        "flame_intensity",
        "distance",
        "movement",
    )

    # Reference ranges describe the simulated prototype, not hardware limits.
    _NORMALIZATION_RANGES = {
        "gas_level": (0.0, 1000.0),
        "temperature": (0.0, 125.0),
        "flame_intensity": (0.0, 100.0),
        "distance": (0.0, 300.0),
        "movement": (0.0, 1.0),
    }

    def process(self, sensor_data: Mapping[str, object]) -> ProcessedFeatures:
        """Validate and transform a raw sensor reading.

        Raw values are retained under their original names so downstream code
        does not lose useful units.  Normalized values are clipped only in
        their ``*_normalized`` representation; an out-of-reference but finite
        raw measurement is therefore still available for inspection.

        Missing, non-numeric, boolean, NaN, and infinite values raise a clear
        ``SensorDataValidationError`` instead of being silently imputed.
        """
        validated = self._validate(sensor_data)

        gas_level = validated["gas_level"]
        temperature = validated["temperature"]
        flame_intensity = validated["flame_intensity"]
        distance = validated["distance"]
        movement = validated["movement"]

        return {
            # Preserve the sanitized raw sensor readings and their units.
            "gas_level": gas_level,
            "temperature": temperature,
            "flame_intensity": flame_intensity,
            "distance": distance,
            "movement": movement,
            # Normalized values support transparent, comparable fusion rules.
            "gas_level_normalized": self._normalize("gas_level", gas_level),
            "temperature_normalized": self._normalize("temperature", temperature),
            "flame_intensity_normalized": self._normalize(
                "flame_intensity", flame_intensity
            ),
            "distance_normalized": self._normalize("distance", distance),
            "movement_normalized": self._normalize("movement", movement),
            # Simple explainable cues for a future baseline fusion engine.
            "high_gas": gas_level >= 500.0,
            "high_temperature": temperature >= 55.0,
            "flame_detected": flame_intensity >= 20.0,
            "low_movement": movement <= 0.15,
        }

    def _validate(self, sensor_data: Mapping[str, object]) -> Dict[str, float]:
        if not isinstance(sensor_data, Mapping):
            raise SensorDataValidationError("sensor_data must be a mapping")

        missing = [field for field in self.REQUIRED_FIELDS if field not in sensor_data]
        if missing:
            raise SensorDataValidationError(
                f"Missing required sensor fields: {', '.join(missing)}"
            )

        validated: Dict[str, float] = {}
        for field in self.REQUIRED_FIELDS:
            value = sensor_data[field]
            if isinstance(value, bool):
                raise SensorDataValidationError(f"{field} must be numeric, not boolean")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise SensorDataValidationError(f"{field} must be numeric") from error
            if not math.isfinite(numeric_value):
                raise SensorDataValidationError(f"{field} must be finite (not NaN or infinity)")
            validated[field] = numeric_value
        return validated

    def _normalize(self, field: str, value: float) -> float:
        lower, upper = self._NORMALIZATION_RANGES[field]
        clipped_value = min(max(value, lower), upper)
        return round((clipped_value - lower) / (upper - lower), 4)


def process_sensor_reading(sensor_data: Mapping[str, object]) -> ProcessedFeatures:
    """Process one reading without requiring callers to instantiate a class."""
    return FeatureProcessor().process(sensor_data)


if __name__ == "__main__":
    # Direct execution makes ``fusion/`` the import root.  Add the project
    # root so this optional demo can also import the sibling simulation module.
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from simulation.sensor_simulator import SensorSimulator

    demo_simulator = SensorSimulator(seed=42)
    demo_processor = FeatureProcessor()
    for demo_scenario in (
        SensorSimulator.NORMAL,
        SensorSimulator.FIRE,
        SensorSimulator.VICTIM,
        SensorSimulator.FIRE_WITH_VICTIM,
    ):
        raw_reading = demo_simulator.generate_reading(demo_scenario)
        print(f"{demo_scenario}: {demo_processor.process(raw_reading)}")
