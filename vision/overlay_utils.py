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
    sensor_data: Mapping[str, object] | None = None,
) -> None:
    """Draw a simple textual overlay on ``frame``.

    The function mutates the frame in‑place; it does **not** return a new
    image. All values are optional – missing dictionaries are skipped.
    """
    overlay_lines = []
    if sensor_data is not None:
        if sensor_data.get("connected"):
            overlay_lines.append("HARDWARE MODE — ESP32 CONNECTED")
        else:
            overlay_lines.append("HARDWARE MODE — ESP32 CONNECTING...")
    else:
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

    # Telemetry HUD overlay when hardware sensor data is available
    if sensor_data is not None:
        curr_y = y0 + len(overlay_lines) * dy + 10

        f_dist = float(sensor_data.get("front_cm", -1.0))
        l_dist = float(sensor_data.get("left_cm", -1.0))
        r_dist = float(sensor_data.get("right_cm", -1.0))
        dist_text = f"Dist (cm): F:{f_dist:.1f} L:{l_dist:.1f} R:{r_dist:.1f}"
        cv2.putText(frame, dist_text, (10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        curr_y += dy

        gas_raw = int(sensor_data.get("gas_raw", 0))
        gas_alert = bool(sensor_data.get("gas_alert", False))
        gas_color = (0, 0, 255) if gas_alert else (0, 255, 255)
        gas_text = f"Gas: {gas_raw} ({'ALERT' if gas_alert else 'Normal'})"
        cv2.putText(frame, gas_text, (10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gas_color, 2, cv2.LINE_AA)
        curr_y += dy

        flame_det = bool(sensor_data.get("flame_detected", False))
        if flame_det:
            cv2.putText(frame, "Flame: ALERT", (10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Flame: Clear", (10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        curr_y += dy

        acc_x = float(sensor_data.get("acc_x", 0.0))
        acc_y = float(sensor_data.get("acc_y", 0.0))
        acc_z = float(sensor_data.get("acc_z", 1.0))
        tilt = bool(sensor_data.get("tilt", False))
        prefix = f"IMU (g): X:{acc_x:.2f} Y:{acc_y:.2f} Z:{acc_z:.2f} | Tilt: "
        cv2.putText(frame, prefix, (10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        (prefix_w, _), _ = cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        status_text = "TILT DETECTED" if tilt else "LEVEL"
        status_color = (0, 0, 255) if tilt else (0, 255, 0)
        cv2.putText(frame, status_text, (10 + prefix_w, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        curr_y += dy

