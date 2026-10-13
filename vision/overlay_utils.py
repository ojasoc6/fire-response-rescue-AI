"""Utility functions for drawing telemetry and multimodal assessment HUD on OpenCV frames.

Implements an alpha-blended semi-transparent backdrop card and a structured telemetry layout:
- Header: System mode & device status
- Telemetry: Distances (front, left, right), Gas & Flame, IMU & Robot Platform Safety
- Intelligence & Mission Status: Threat Hazard Level & Description, Victim Triage & Priority, Navigation Advisory
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Mapping, Tuple


def draw_overlay(
    frame: np.ndarray,
    vision_info: Mapping[str, object] | None,
    fire_info: Mapping[str, object] | None,
    assessment: Mapping[str, object] | None,
    fps: float,
    resolution: Tuple[int, int],
    device: str,
    sensor_data: Mapping[str, object] | None = None,
) -> None:
    """Draw a semi-transparent HUD card with cross-verified telemetry and mission status.

    Mutates ``frame`` in-place.
    """
    if frame is None or not hasattr(frame, "shape"):
        return

    # 1. Header: System mode & device
    if sensor_data is not None:
        if bool(sensor_data.get("connected", False)):
            header_str = f"HARDWARE MODE - ESP32 CONNECTED | Device: {device} | {fps:.1f} FPS"
        else:
            header_str = f"HARDWARE MODE - ESP32 CONNECTING... | Device: {device} | {fps:.1f} FPS"
    else:
        header_str = f"DEMO MODE - SENSOR INPUTS SIMULATED | Device: {device} | {fps:.1f} FPS"

    # 2. Telemetry: Distances (front, left, right in cm)
    if sensor_data is not None:
        f_val = float(sensor_data.get("front_cm", -1.0))
        l_val = float(sensor_data.get("left_cm", -1.0))
        r_val = float(sensor_data.get("right_cm", -1.0))
        f_str = f"{f_val:.1f}cm" if f_val >= 0 else "OFFLINE"
        l_str = f"{l_val:.1f}cm" if l_val >= 0 else "OFFLINE"
        r_str = f"{r_val:.1f}cm" if r_val >= 0 else "OFFLINE"
    else:
        f_str = "SIMULATED"
        l_str = "SIMULATED"
        r_str = "SIMULATED"
    dist_str = f"Distances: Front: {f_str} | Left: {l_str} | Right: {r_str}"

    # 3. Telemetry: Gas & Flame
    if sensor_data is not None:
        gas_raw = int(sensor_data.get("gas_raw", 0))
        gas_alert = bool(sensor_data.get("gas_alert", False))
        flame_detected = bool(sensor_data.get("flame_detected", False))
    else:
        gas_raw = 120
        gas_alert = False
        flame_detected = False
    gas_status = "ALERT" if gas_alert else "Normal"
    flame_status = "ALERT" if flame_detected else "Clear"
    gas_flame_str = f"Gas & Flame: Gas: {gas_raw} ({gas_status}) | Flame: {flame_status}"

    # 4. Telemetry: IMU & Platform Safety Status
    if sensor_data is not None:
        acc_x = float(sensor_data.get("acc_x", 0.0))
        acc_y = float(sensor_data.get("acc_y", 0.0))
        acc_z = float(sensor_data.get("acc_z", 1.0))
    else:
        acc_x, acc_y, acc_z = 0.0, 0.0, 1.0
    robot_status = str(assessment.get("robot_status", "STABLE (LEVEL)")) if assessment else "STABLE (LEVEL)"
    imu_str = f"IMU (g): X:{acc_x:.2f} Y:{acc_y:.2f} Z:{acc_z:.2f} | {robot_status}"

    # 5. Intelligence: Hazard Level & Description
    hazard_level = str(assessment.get("hazard_level", "LOW")) if assessment else "LOW"
    hazard_desc = str(assessment.get("hazard_desc", "CLEAR")) if assessment else "CLEAR"
    hazard_str = f"Hazard: {hazard_level} ({hazard_desc})"

    # Hazard color coding: Green for LOW, Orange/Yellow for MODERATE, Bright Red for HIGH/CRITICAL
    if hazard_level == "LOW":
        hazard_color = (0, 255, 0)
    elif hazard_level == "MODERATE":
        hazard_color = (0, 215, 255)  # Amber / Yellow
    else:
        hazard_color = (0, 0, 255)  # Bright Red

    # 6. Intelligence: Victim Triage & Priority
    victim_state = str(assessment.get("victim_state", "NONE")) if assessment else "NONE"
    priority = str(assessment.get("priority", assessment.get("rescue_priority", "LOW"))) if assessment else "LOW"
    victim_str = f"Victim: {victim_state} | Priority: {priority}"

    if priority in ("HIGH", "CRITICAL"):
        victim_color = (0, 0, 255)
    elif priority == "MODERATE":
        victim_color = (0, 215, 255)
    else:
        victim_color = (0, 255, 128)

    # 7. Intelligence: Navigation Advisory
    nav_advisory = str(assessment.get("nav_advisory", "PATH_CLEAR")) if assessment else "PATH_CLEAR"
    nav_str = f"Navigation: {nav_advisory}"

    if nav_advisory in ("TRAPPED_DEAD_END", "OBSTACLE_AHEAD_REVERSE", "SENSOR_OFFLINE"):
        nav_color = (0, 0, 255)
    elif nav_advisory in ("DETOUR_LEFT", "DETOUR_RIGHT"):
        nav_color = (0, 215, 255)
    else:
        nav_color = (0, 255, 0)

    # 8. Optional Vision Observation Summary
    p_det = "YES" if (vision_info and vision_info.get("person_detected")) else "NO"
    posture = str(vision_info.get("posture", "N/A")) if vision_info else "N/A"
    f_det = "YES" if (fire_info and fire_info.get("fire_detected")) else "NO"
    s_det = "YES" if (fire_info and fire_info.get("smoke_detected")) else "NO"
    vision_str = f"Vision: Person: {p_det} ({posture}) | Fire: {f_det} | Smoke: {s_det}"

    # Assemble HUD text and colors
    hud_entries = [
        (header_str, (0, 255, 255)),
        (dist_str, (220, 220, 220)),
        (gas_flame_str, (0, 0, 255) if (gas_alert or flame_detected) else (0, 255, 128)),
        (imu_str, (0, 0, 255) if "ROLLOVER" in robot_status or "IMPACT" in robot_status else (0, 255, 128)),
        (hazard_str, hazard_color),
        (victim_str, victim_color),
        (nav_str, nav_color),
        (vision_str, (200, 255, 255)),
    ]

    # Semi-Transparent Backdrop Card (cv2.addWeighted with opacity 0.6)
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = 10, 10
    card_w = min(w_frame - 20, 680)
    card_h = min(h_frame - 20, 18 + len(hud_entries) * 25)
    x2, y2 = x1 + card_w, y1 + card_h

    if y2 > y1 and x2 > x1:
        roi = frame[y1:y2, x1:x2]
        dark_card = np.zeros_like(roi)
        dark_card[:] = (12, 12, 12)
        cv2.addWeighted(dark_card, 0.6, roi, 0.4, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (75, 75, 75), 1)

    # Render HUD text lines onto frame
    start_y = y1 + 22
    for i, (text, color) in enumerate(hud_entries):
        text_y = start_y + i * 25
        cv2.putText(
            frame,
            text,
            (x1 + 12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
