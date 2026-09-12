"""Simulated environmental sensor readings for the fire-rescue prototype.

This module intentionally produces *simulated* readings.  Its five-field
dictionary is the small, hardware-neutral contract that a future ESP32 input
adapter can provide to the rest of the pipeline.
"""

from __future__ import annotations

import random
from typing import Dict, Optional


SensorReading = Dict[str, float]


class SensorSimulator:
    """Generate plausible, simulated readings for predefined rescue scenarios.

    Passing a seed makes the sequence of generated readings reproducible.  A
    single simulator may be used to model a time series: each call returns a
    new deterministic reading when constructed with the same seed and called
    in the same order.
    """

    NORMAL = "NORMAL"
    FIRE = "FIRE"
    VICTIM = "VICTIM"
    FIRE_WITH_VICTIM = "FIRE_WITH_VICTIM"

    _SCENARIOS = {NORMAL, FIRE, VICTIM, FIRE_WITH_VICTIM}

    def __init__(self, seed: Optional[int] = None) -> None:
        self._random = random.Random(seed)

    def generate_reading(self, scenario: str) -> SensorReading:
        """Return one simulated reading for ``scenario``.

        Values use intentionally simple prototype units:
        ``gas_level`` is a simulated MQ-2 smoke/gas indication (0--1000),
        ``temperature`` is an MLX90614-style reading in degrees Celsius,
        ``flame_intensity`` is a simulated flame signal percentage (0--100),
        ``distance`` is a simulated HC-SR04 obstacle distance in centimetres,
        and ``movement`` is normalized MPU6050 activity (0.0--1.0).
        """
        normalized_scenario = self._validate_scenario(scenario)
        ranges = self._ranges_for(normalized_scenario)

        # MQ-2-like gas/smoke indication; higher values suggest smoke or gas.
        gas_level = self._random.uniform(*ranges["gas_level"])
        # MLX90614-like ambient/object temperature in degrees Celsius.
        temperature = self._random.uniform(*ranges["temperature"])
        # Flame sensor signal expressed as a simple simulated intensity percent.
        flame_intensity = self._random.uniform(*ranges["flame_intensity"])
        # HC-SR04-like distance to the nearest obstacle in centimetres.
        distance = self._random.uniform(*ranges["distance"])
        # MPU6050-like normalized activity; low values can indicate stillness.
        movement = self._random.uniform(*ranges["movement"])

        return {
            "gas_level": round(gas_level, 2),
            "temperature": round(temperature, 2),
            "flame_intensity": round(flame_intensity, 2),
            "distance": round(distance, 2),
            "movement": round(movement, 3),
        }

    def _validate_scenario(self, scenario: str) -> str:
        if not isinstance(scenario, str):
            raise TypeError("scenario must be a string")

        normalized = scenario.strip().upper()
        if normalized not in self._SCENARIOS:
            options = ", ".join(sorted(self._SCENARIOS))
            raise ValueError(f"Unknown scenario {scenario!r}. Choose one of: {options}")
        return normalized

    @staticmethod
    def _ranges_for(scenario: str) -> Dict[str, tuple[float, float]]:
        """Return conservative simulated ranges for the requested scenario."""
        if scenario == SensorSimulator.NORMAL:
            return {
                "gas_level": (80.0, 220.0),
                "temperature": (20.0, 31.0),
                "flame_intensity": (0.0, 5.0),
                "distance": (80.0, 300.0),
                "movement": (0.35, 0.95),
            }
        if scenario == SensorSimulator.FIRE:
            return {
                "gas_level": (550.0, 950.0),
                "temperature": (55.0, 110.0),
                "flame_intensity": (65.0, 100.0),
                "distance": (30.0, 180.0),
                "movement": (0.10, 0.60),
            }
        if scenario == SensorSimulator.VICTIM:
            return {
                "gas_level": (100.0, 260.0),
                "temperature": (20.0, 33.0),
                "flame_intensity": (0.0, 5.0),
                "distance": (25.0, 120.0),
                "movement": (0.0, 0.12),
            }
        # FIRE_WITH_VICTIM combines severe environmental cues with stillness.
        return {
            "gas_level": (650.0, 1000.0),
            "temperature": (65.0, 125.0),
            "flame_intensity": (75.0, 100.0),
            "distance": (20.0, 100.0),
            "movement": (0.0, 0.08),
        }


if __name__ == "__main__":
    demo_simulator = SensorSimulator(seed=42)
    for demo_scenario in (
        SensorSimulator.NORMAL,
        SensorSimulator.FIRE,
        SensorSimulator.VICTIM,
        SensorSimulator.FIRE_WITH_VICTIM,
    ):
        print(f"{demo_scenario}: {demo_simulator.generate_reading(demo_scenario)}")
