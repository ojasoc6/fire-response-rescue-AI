"""Command-line demonstration for the fire-rescue rule-based pipeline.

All readings and vision observations printed by this program are simulated
prototype data; they are not measurements from physical robot hardware.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional


_IMPORT_ERROR: Optional[Exception] = None
try:
    from fusion.feature_processing import FeatureProcessor
    from fusion.fusion_engine import FusionEngine
    from simulation.sensor_simulator import SensorSimulator
    from temporal.temporal_features import TemporalFeatureExtractor
    from vision.vision_integration import VisionIntegrator
except Exception as error:
    _IMPORT_ERROR = error


SCENARIOS = ("NORMAL", "FIRE", "VICTIM", "FIRE_WITH_VICTIM")

def run_scenario(
    scenario: str,
    seed: int,
    history_length: int = 5,
    image_path: Optional[str] = None,
) -> Dict[str, object]:
    """Run the full simulated pipeline once and return its intermediate results."""
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"Required pipeline module could not load: {_IMPORT_ERROR}")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose one of: {', '.join(SCENARIOS)}")
    if history_length < 1:
        raise ValueError("history_length must be at least 1")

    simulator = SensorSimulator(seed=seed)
    processor = FeatureProcessor()
    temporal_extractor = TemporalFeatureExtractor()
    fusion_engine = FusionEngine()

    raw_history = [simulator.generate_reading(scenario) for _ in range(history_length)]
    processed_history = [processor.process(reading) for reading in raw_history]
    temporal_features = temporal_extractor.extract(processed_history)
    vision_info = VisionIntegrator().analyze_image(image_path) if image_path else None
    assessment = fusion_engine.fuse(
        processed_history[-1], temporal_features, vision_info
    )

    return {
        "scenario": scenario,
        "current_reading": raw_history[-1],
        "processed_features": processed_history[-1],
        "temporal_features": temporal_features,
        "vision_info": vision_info,
        "assessment": assessment,
    }


def print_demonstration(result: Dict[str, object]) -> None:
    """Print only the evidence needed to explain one pipeline assessment."""
    reading = result["current_reading"]
    temporal = result["temporal_features"]
    assessment = result["assessment"]
    assert isinstance(reading, dict)
    assert isinstance(temporal, dict)
    assert isinstance(assessment, dict)

    print("=" * 64)
    print(f"Scenario: {result['scenario']} (simulated)")
    print(
        "Current sensors: "
        f"gas={reading['gas_level']}, "
        f"temperature={reading['temperature']} C, "
        f"flame={reading['flame_intensity']}%, "
        f"distance={reading['distance']} cm, "
        f"movement={reading['movement']}"
    )
    print(
        "Temporal indicators: "
        f"gas slope={temporal['gas_slope']}, "
        f"temperature slope={temporal['temperature_slope']}, "
        f"gas rising={temporal['gas_rising']}, "
        f"temperature rising={temporal['temperature_rising']}, "
        f"flame persistent={temporal['flame_persistent']}, "
        f"movement declining={temporal['movement_declining']}, "
        f"environment worsening={temporal['environment_worsening']}"
    )
    if result["vision_info"] is not None:
        print(f"Vision observation (not a victim diagnosis): {result['vision_info']}")
    print(f"Hazard level:    {assessment['hazard_level']}")
    print(f"Victim state:    {assessment['victim_state']}")
    print(f"Rescue priority: {assessment['rescue_priority']}")
    factors = assessment["decision_factors"]
    print("Decision factors:")
    if factors:
        for factor in factors:
            print(f"- {factor}")
    else:
        print("- No elevated hazard or explicit victim evidence.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simulated fire-rescue sensor-fusion demonstration."
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Run only one scenario; by default all four are demonstrated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible simulated sensor histories.",
    )
    parser.add_argument(
        "--image",
        help=(
            "Optional local camera image for real person and pose observations. "
            "Without it, the demonstration remains sensor-only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run selected scenario demonstrations and report clean failures."""
    arguments = parse_arguments()
    if _IMPORT_ERROR is not None:
        print(f"Pipeline startup error: required module unavailable: {_IMPORT_ERROR}", file=sys.stderr)
        return 1

    selected_scenarios: List[str] = (
        [arguments.scenario] if arguments.scenario else list(SCENARIOS)
    )
    try:
        for index, scenario in enumerate(selected_scenarios):
            result = run_scenario(
                scenario,
                seed=arguments.seed + index,
                image_path=arguments.image,
            )
            print_demonstration(result)
        print("=" * 64)
        print("Demonstration completed successfully. Sensor data is simulated.")
        return 0
    except (RuntimeError, ValueError, TypeError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
