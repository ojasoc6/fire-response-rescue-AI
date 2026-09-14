"""Utility functions for drawing overlay information on OpenCV frames.

The overlay includes simulation mode banner, device info, FPS, resolution, and
key observation fields (person detection, fire/smoke detection, and fusion
assessment). This is a lightweight helper used by the live‑camera mode.
"""

import cv2
from typing import Mapping, Tuple

def draw_overlay(
    frame,
    vision_info: Mapping[str, object] | None,
    fire_info: Mapping[str, object] | None,
    assessment: Mapping[str, object] | None,
    fps: float,
    resolution: Tuple[int, int],
    device: str,
) -> None:
    """Draw a simple textual overlay on ``frame``.

    The function mutates the frame in‑place; it does **not** return a new
    image. All values are optional – missing dictionaries are skipped.
    """
    overlay_lines = []
    overlay_lines.append("DEMO MODE — SENSOR INPUTS SIMULATED")
    overlay_lines.append(f"Device: {device}")
    overlay_lines.append(f"Resolution: {resolution[0]}x{resolution[1]}")
    overlay_lines.append(f"FPS: {fps:.1f}")

    if vision_info:
        overlay_lines.append(
            f"Person: {'YES' if vision_info.get('person_detected') else 'NO'} "
            f"({vision_info.get('posture', 'N/A')})"
        )
    if fire_info:
        overlay_lines.append(
            f"Fire: {'YES' if fire_info.get('fire_detected') else 'NO'} "
            f"(conf {fire_info.get('fire_confidence', 0):.2f})"
        )
        overlay_lines.append(
            f"Smoke: {'YES' if fire_info.get('smoke_detected') else 'NO'} "
            f"(conf {fire_info.get('smoke_confidence', 0):.2f})"
        )
    if assessment:
        overlay_lines.append(f"Hazard: {assessment.get('hazard_level', 'N/A')}")
        overlay_lines.append(f"Victim: {assessment.get('victim_state', 'N/A')}")
        overlay_lines.append(f"Priority: {assessment.get('rescue_priority', 'N/A')}")

    y0, dy = 20, 20
    for i, line in enumerate(overlay_lines):
        y = y0 + i * dy
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
