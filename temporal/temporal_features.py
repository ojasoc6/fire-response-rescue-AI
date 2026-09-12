"""Interpretable temporal features over short processed-sensor histories."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict, List, Union


TemporalValue = Union[float, bool]
TemporalFeatures = Dict[str, TemporalValue]


class TemporalDataValidationError(ValueError):
    """Raised when a processed sensor history cannot be analyzed safely."""


class TemporalFeatureExtractor:
    """Summarize a short observation history for the future fusion layer.

    Slopes are least-squares change per observation (not per wall-clock
    second), because the prototype readings do not yet include timestamps.
    Persistence values are fractions from 0.0 to 1.0 indicating how often a
    condition occurred in the supplied history.
    """

    REQUIRED_FIELDS = (
        "gas_level",
        "temperature",
        "flame_intensity",
        "movement",
    )

    def extract(self, history: Sequence[Mapping[str, object]]) -> TemporalFeatures:
        """Calculate averages, slopes, persistence, and trend indicators.

        A one-reading history is accepted: averages and persistence remain
        meaningful, while all slopes are zero and dynamic trend flags are
        false.  Empty or invalid histories raise a clear validation error.
        """
        readings = self._validate_history(history)

        gas_values = [reading["gas_level"] for reading in readings]
        temperature_values = [reading["temperature"] for reading in readings]
        flame_values = [reading["flame_intensity"] for reading in readings]
        movement_values = [reading["movement"] for reading in readings]

        high_gas_persistence = self._persistence(
            [reading["high_gas"] for reading in readings]
        )
        high_temperature_persistence = self._persistence(
            [reading["high_temperature"] for reading in readings]
        )
        flame_persistence = self._persistence(
            [reading["flame_detected"] for reading in readings]
        )
        low_movement_persistence = self._persistence(
            [reading["low_movement"] for reading in readings]
        )

        gas_slope = self._slope(gas_values)
        temperature_slope = self._slope(temperature_values)
        flame_slope = self._slope(flame_values)
        movement_slope = self._slope(movement_values)

        has_trend_history = len(readings) >= 2
        gas_rising = has_trend_history and gas_slope >= 25.0
        temperature_rising = has_trend_history and temperature_slope >= 2.0
        movement_declining = has_trend_history and movement_slope <= -0.03
        flame_persistent = len(readings) >= 2 and flame_persistence >= 0.6

        # Worsening requires at least two independent environmental signals.
        environmental_signals = (
            gas_rising,
            temperature_rising,
            flame_persistent,
        )
        environment_worsening = sum(environmental_signals) >= 2

        return {
            "gas_recent_average": self._average(gas_values),
            "temperature_recent_average": self._average(temperature_values),
            "flame_intensity_recent_average": self._average(flame_values),
            "movement_recent_average": self._average(movement_values),
            "gas_slope": gas_slope,
            "temperature_slope": temperature_slope,
            "flame_intensity_slope": flame_slope,
            "movement_slope": movement_slope,
            "high_gas_persistence": high_gas_persistence,
            "high_temperature_persistence": high_temperature_persistence,
            "flame_detected_persistence": flame_persistence,
            "low_movement_persistence": low_movement_persistence,
            "gas_rising": gas_rising,
            "temperature_rising": temperature_rising,
            "flame_persistent": flame_persistent,
            "movement_declining": movement_declining,
            "environment_worsening": environment_worsening,
        }

    def _validate_history(
        self, history: Sequence[Mapping[str, object]]
    ) -> List[Dict[str, Union[float, bool]]]:
        if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
            raise TemporalDataValidationError("history must be a sequence of processed readings")
        if not history:
            raise TemporalDataValidationError("history must contain at least one reading")

        validated_readings: List[Dict[str, Union[float, bool]]] = []
        for index, reading in enumerate(history):
            if not isinstance(reading, Mapping):
                raise TemporalDataValidationError(f"history item {index} must be a mapping")
            missing = [field for field in self.REQUIRED_FIELDS if field not in reading]
            if missing:
                raise TemporalDataValidationError(
                    f"history item {index} is missing: {', '.join(missing)}"
                )

            validated: Dict[str, Union[float, bool]] = {}
            for field in self.REQUIRED_FIELDS:
                value = reading[field]
                if isinstance(value, bool):
                    raise TemporalDataValidationError(
                        f"history item {index} field {field} must be numeric"
                    )
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as error:
                    raise TemporalDataValidationError(
                        f"history item {index} field {field} must be numeric"
                    ) from error
                if not math.isfinite(numeric_value):
                    raise TemporalDataValidationError(
                        f"history item {index} field {field} must be finite"
                    )
                validated[field] = numeric_value

            # Use the processor's flags when present; otherwise derive them
            # from the same transparent thresholds for compatibility.
            validated["high_gas"] = bool(reading.get("high_gas", validated["gas_level"] >= 500.0))
            validated["high_temperature"] = bool(
                reading.get("high_temperature", validated["temperature"] >= 55.0)
            )
            validated["flame_detected"] = bool(
                reading.get("flame_detected", validated["flame_intensity"] >= 20.0)
            )
            validated["low_movement"] = bool(
                reading.get("low_movement", validated["movement"] <= 0.15)
            )
            validated_readings.append(validated)
        return validated_readings

    @staticmethod
    def _average(values: List[float]) -> float:
        return round(sum(values) / len(values), 4)

    @staticmethod
    def _slope(values: List[float]) -> float:
        """Return least-squares slope per observation index."""
        if len(values) < 2:
            return 0.0
        mean_index = (len(values) - 1) / 2
        mean_value = sum(values) / len(values)
        numerator = sum(
            (index - mean_index) * (value - mean_value)
            for index, value in enumerate(values)
        )
        denominator = sum((index - mean_index) ** 2 for index in range(len(values)))
        return round(numerator / denominator, 4)

    @staticmethod
    def _persistence(flags: List[bool]) -> float:
        return round(sum(flags) / len(flags), 4)


def extract_temporal_features(
    history: Sequence[Mapping[str, object]],
) -> TemporalFeatures:
    """Extract temporal features without requiring callers to instantiate a class."""
    return TemporalFeatureExtractor().extract(history)


if __name__ == "__main__":
    # Permit both ``python temporal/temporal_features.py`` and ``python -m``.
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from fusion.feature_processing import process_sensor_reading

    stable_raw_readings = [
        {"gas_level": 150, "temperature": 25, "flame_intensity": 1, "distance": 150, "movement": 0.70},
        {"gas_level": 155, "temperature": 25.5, "flame_intensity": 1, "distance": 149, "movement": 0.68},
        {"gas_level": 148, "temperature": 24.8, "flame_intensity": 2, "distance": 151, "movement": 0.72},
        {"gas_level": 152, "temperature": 25.2, "flame_intensity": 1, "distance": 150, "movement": 0.69},
    ]
    worsening_raw_readings = [
        {"gas_level": 320, "temperature": 29, "flame_intensity": 65, "distance": 100, "movement": 0.65},
        {"gas_level": 410, "temperature": 32, "flame_intensity": 72, "distance": 95, "movement": 0.45},
        {"gas_level": 550, "temperature": 37, "flame_intensity": 80, "distance": 90, "movement": 0.25},
        {"gas_level": 690, "temperature": 43, "flame_intensity": 88, "distance": 85, "movement": 0.10},
        {"gas_level": 810, "temperature": 49, "flame_intensity": 90, "distance": 80, "movement": 0.05},
    ]

    for label, raw_readings in (
        ("Sequence A — stable/normal", stable_raw_readings),
        ("Sequence B — worsening fire", worsening_raw_readings),
    ):
        processed_history = [process_sensor_reading(reading) for reading in raw_readings]
        print(f"{label}: {extract_temporal_features(processed_history)}")
