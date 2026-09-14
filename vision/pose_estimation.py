from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

# Cache for loaded pose models to avoid reloading per frame
_POSE_MODEL_CACHE: dict[str, YOLO] = {}

def _load_pose_model(model_name: str) -> YOLO:
    """Load and cache a YOLO pose model.

    Reuses the same model instance across calls to reduce overhead.
    """
    if model_name not in _POSE_MODEL_CACHE:
        _POSE_MODEL_CACHE[model_name] = YOLO(model_name)
    return _POSE_MODEL_CACHE[model_name]


# COCO keypoint indices used by the Ultralytics pose model
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Simple COCO skeleton connections for visualization
SKELETON = [
    (5, 6),       # shoulders
    (5, 11),      # left shoulder -> left hip
    (6, 12),      # right shoulder -> right hip
    (11, 12),     # hips
    (11, 13),     # left hip -> left knee
    (13, 15),     # left knee -> left ankle
    (12, 14),     # right hip -> right knee
    (14, 16),     # right knee -> right ankle
    (5, 7),        # left shoulder -> left elbow
    (7, 9),        # left elbow -> left wrist
    (6, 8),        # right shoulder -> right elbow
    (8, 10),       # right elbow -> right wrist
    (0, 5),        # nose -> left shoulder
    (0, 6),        # nose -> right shoulder
]

DEFAULT_MODEL = "yolov8n-pose.pt"


def _point_available(
    keypoints: list[list[float]],
    confidences: list[float],
    index: int,
    threshold: float = 0.25,
) -> bool:
    """Return True when a keypoint exists and has sufficient confidence."""
    if index >= len(keypoints) or index >= len(confidences):
        return False

    x, y = keypoints[index]
    return (
        math.isfinite(x)
        and math.isfinite(y)
        and confidences[index] >= threshold
    )


def _midpoint(
    keypoints: list[list[float]],
    confidences: list[float],
    first: int,
    second: int,
    threshold: float = 0.25,
) -> tuple[float, float] | None:
    """Return the midpoint of two sufficiently confident keypoints."""
    if not (
        _point_available(keypoints, confidences, first, threshold)
        and _point_available(keypoints, confidences, second, threshold)
    ):
        return None

    x1, y1 = keypoints[first]
    x2, y2 = keypoints[second]

    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _joint_angle_degrees(
    first: tuple[float, float],
    vertex: tuple[float, float],
    second: tuple[float, float],
) -> float | None:
    """Return the angle at ``vertex`` or None for degenerate geometry."""
    first_vector = (first[0] - vertex[0], first[1] - vertex[1])
    second_vector = (second[0] - vertex[0], second[1] - vertex[1])
    first_length = math.hypot(*first_vector)
    second_length = math.hypot(*second_vector)
    if first_length < 1.0 or second_length < 1.0:
        return None

    cosine = (
        first_vector[0] * second_vector[0]
        + first_vector[1] * second_vector[1]
    ) / (first_length * second_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _angle_from_vertical(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Return an orientation from vertical: 0° vertical, 90° horizontal."""
    return math.degrees(
        math.atan2(abs(second[0] - first[0]), abs(second[1] - first[1]))
    )


def _classify_posture(
    keypoints: list[list[float]],
    confidences: list[float],
    bbox: tuple[float, float, float, float],
    keypoint_threshold: float = 0.25,
) -> tuple[str, dict[str, Any]]:
    """
    Classify posture using simple geometric heuristics.

    This is a visual posture heuristic only.
    It is NOT a medical diagnosis and does not indicate consciousness,
    injury, or actual victim status.
    """

    shoulder_center = _midpoint(
        keypoints,
        confidences,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        keypoint_threshold,
    )

    hip_center = _midpoint(
        keypoints,
        confidences,
        LEFT_HIP,
        RIGHT_HIP,
        keypoint_threshold,
    )

    def available_point(index: int) -> tuple[float, float] | None:
        if not _point_available(keypoints, confidences, index, keypoint_threshold):
            return None
        return tuple(keypoints[index])

    left_hip = available_point(LEFT_HIP)
    right_hip = available_point(RIGHT_HIP)
    left_knee = available_point(LEFT_KNEE)
    right_knee = available_point(RIGHT_KNEE)
    left_ankle = available_point(LEFT_ANKLE)
    right_ankle = available_point(RIGHT_ANKLE)

    x_min, y_min, x_max, y_max = bbox
    bbox_width = max(x_max - x_min, 1.0)
    bbox_height = max(y_max - y_min, 1.0)
    bbox_aspect_ratio = bbox_width / bbox_height

    diagnostic: dict[str, Any] = {
        "bbox_aspect_ratio": round(bbox_aspect_ratio, 3),
    }

    # Derive leg geometry from complete left/right hip-knee-ankle chains.
    # Extended knees plus vertically oriented legs support an upright pose;
    # flexed knees and horizontal thighs support a seated pose.
    knee_angles: list[float] = []
    thigh_angles_from_vertical: list[float] = []
    shin_angles_from_vertical: list[float] = []
    for hip, knee, ankle in (
        (left_hip, left_knee, left_ankle),
        (right_hip, right_knee, right_ankle),
    ):
        if hip is None or knee is None or ankle is None:
            continue
        knee_angle = _joint_angle_degrees(hip, knee, ankle)
        if knee_angle is not None:
            knee_angles.append(knee_angle)
            thigh_angles_from_vertical.append(_angle_from_vertical(hip, knee))
            shin_angles_from_vertical.append(_angle_from_vertical(knee, ankle))

    if knee_angles:
        diagnostic["average_knee_angle_deg"] = round(
            sum(knee_angles) / len(knee_angles), 1
        )
        diagnostic["average_thigh_angle_from_vertical_deg"] = round(
            sum(thigh_angles_from_vertical) / len(thigh_angles_from_vertical), 1
        )
        diagnostic["average_shin_angle_from_vertical_deg"] = round(
            sum(shin_angles_from_vertical) / len(shin_angles_from_vertical), 1
        )

    # ------------------------------------------------------------------
    # 1. Body orientation and torso compactness identify lying candidates.
    # ------------------------------------------------------------------
    if shoulder_center is not None and hip_center is not None:
        torso_angle = _angle_from_vertical(shoulder_center, hip_center)
        torso_length = _distance(shoulder_center, hip_center)

        diagnostic["torso_angle_from_vertical_deg"] = round(torso_angle, 1)
        diagnostic["torso_length_px"] = round(torso_length, 1)

        shoulder_width = (
            _distance(tuple(keypoints[LEFT_SHOULDER]), tuple(keypoints[RIGHT_SHOULDER]))
            if _point_available(keypoints, confidences, LEFT_SHOULDER, keypoint_threshold)
            and _point_available(keypoints, confidences, RIGHT_SHOULDER, keypoint_threshold)
            else None
        )
        torso_compactness = None
        if shoulder_width is not None and shoulder_width >= 1.0:
            torso_compactness = torso_length / shoulder_width
            diagnostic["torso_to_shoulder_width_ratio"] = round(
                torso_compactness, 3
            )

        # Horizontal torso needs a supporting wide body envelope.  The box is
        # evidence only; it never classifies posture by itself.
        if torso_angle >= 55 and bbox_aspect_ratio >= 1.15:
            diagnostic["reason"] = "horizontal_torso_with_wide_body_envelope"
            return "LYING_CANDIDATE", diagnostic

        # A supine/prone person may be viewed from above.  In that case the
        # torso can appear vertical in the image, but it is unusually compact
        # relative to shoulder width while the legs remain extended.
        average_knee_angle = (
            sum(knee_angles) / len(knee_angles) if knee_angles else None
        )
        if (
            torso_compactness is not None
            and torso_compactness <= 0.85
            and bbox_aspect_ratio >= 1.05
            and average_knee_angle is not None
            and average_knee_angle >= 145
        ):
            diagnostic["reason"] = "compact_torso_with_extended_legs_and_wide_envelope"
            return "LYING_CANDIDATE", diagnostic

        # ------------------------------------------------------------------
        # 2. Distinguish standing from sitting for a mostly vertical torso.
        # ------------------------------------------------------------------
        if torso_angle <= 35 and average_knee_angle is not None:
            average_thigh_angle = sum(thigh_angles_from_vertical) / len(
                thigh_angles_from_vertical
            )
            average_shin_angle = sum(shin_angles_from_vertical) / len(
                shin_angles_from_vertical
            )

            if (
                average_knee_angle >= 150
                and average_thigh_angle <= 30
                and average_shin_angle <= 30
            ):
                diagnostic["reason"] = "vertical_torso_with_extended_vertical_legs"
                return "UPRIGHT", diagnostic

            if average_knee_angle <= 130 and average_thigh_angle >= 35:
                diagnostic["reason"] = "vertical_torso_with_flexed_knees_and_nonvertical_thighs"
                return "SITTING", diagnostic

    # ------------------------------------------------------------------
    # 3. Ambiguous or incomplete geometry must not force a posture label.
    # ------------------------------------------------------------------
    diagnostic["reason"] = "ambiguous_or_insufficient_multi_signal_geometry"
    return "UNKNOWN", diagnostic


def _serialize_keypoints(
    keypoints: list[list[float]],
    confidences: list[float],
) -> list[dict[str, float]]:
    """Convert keypoints into a JSON-friendly representation."""
    output: list[dict[str, float]] = []

    for index, point in enumerate(keypoints):
        x, y = point

        confidence = (
            float(confidences[index])
            if index < len(confidences)
            else 0.0
        )

        output.append(
            {
                "index": index,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "confidence": round(confidence, 4),
            }
        )

    return output


def estimate_pose(
    image_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    detection_confidence: float = 0.25,
    keypoint_threshold: float = 0.25,
    save_annotated: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run pretrained human pose estimation on an image.

    Returns observations for each detected person.

    This function reports visual posture observations only.
    It does not diagnose victims, injuries, or consciousness.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file was not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"OpenCV could not read the image: {image_path}"
        )

    # Load the pretrained pose model.
    model = YOLO(model_name)

    results = model(
        image,
        conf=detection_confidence,
        verbose=False,
    )

    if not results:
        raise RuntimeError("The pose model returned no results.")

    result = results[0]

    observations: list[dict[str, Any]] = []
    annotated = image.copy()

    if result.keypoints is None or result.boxes is None:
        observations_output = {
            "pose_detected": False,
            "person_count": 0,
            "persons": [],
        }

        if save_annotated:
            final_output_path = _save_annotated_image(
                annotated,
                image_path,
                output_path,
            )
            observations_output["annotated_image"] = str(
                final_output_path
            )

        return observations_output

    keypoints_xy = result.keypoints.xy.cpu().tolist()

    keypoints_conf = (
        result.keypoints.conf.cpu().tolist()
        if result.keypoints.conf is not None
        else []
    )

    boxes_xyxy = result.boxes.xyxy.cpu().tolist()
    box_conf = (
        result.boxes.conf.cpu().tolist()
        if result.boxes.conf is not None
        else []
    )

    person_count = len(keypoints_xy)

    for person_index in range(person_count):
        person_keypoints = keypoints_xy[person_index]

        if person_index < len(keypoints_conf):
            person_keypoint_conf = keypoints_conf[person_index]
        else:
            person_keypoint_conf = [0.0] * len(person_keypoints)

        bbox = (
            tuple(boxes_xyxy[person_index])
            if person_index < len(boxes_xyxy)
            else (0.0, 0.0, 0.0, 0.0)
        )

        posture, posture_diagnostics = _classify_posture(
            person_keypoints,
            person_keypoint_conf,
            bbox,
            keypoint_threshold,
        )

        visible_keypoints = sum(
            confidence >= keypoint_threshold
            for confidence in person_keypoint_conf
        )

        detection_conf = (
            float(box_conf[person_index])
            if person_index < len(box_conf)
            else 0.0
        )

        person_observation = {
            "person_index": person_index,
            "person_detected": True,
            "detection_confidence": round(detection_conf, 4),
            "posture": posture,
            "visible_keypoints": visible_keypoints,
            "bounding_box": {
                "x_min": round(float(bbox[0]), 2),
                "y_min": round(float(bbox[1]), 2),
                "x_max": round(float(bbox[2]), 2),
                "y_max": round(float(bbox[3]), 2),
            },
            "keypoints": _serialize_keypoints(
                person_keypoints,
                person_keypoint_conf,
            ),
            "posture_diagnostics": posture_diagnostics,
        }

        observations.append(person_observation)

        # --------------------------------------------------------------
        # Draw bounding box.
        # --------------------------------------------------------------
        x_min, y_min, x_max, y_max = map(int, bbox)

        cv2.rectangle(
            annotated,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2,
        )

        # --------------------------------------------------------------
        # Draw keypoints.
        # --------------------------------------------------------------
        for point_index, point in enumerate(person_keypoints):
            if point_index >= len(person_keypoint_conf):
                continue

            confidence = person_keypoint_conf[point_index]

            if confidence < keypoint_threshold:
                continue

            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                annotated,
                (x, y),
                4,
                (0, 0, 255),
                -1,
            )

        # --------------------------------------------------------------
        # Draw skeleton connections.
        # --------------------------------------------------------------
        for first, second in SKELETON:
            if (
                first >= len(person_keypoints)
                or second >= len(person_keypoints)
                or first >= len(person_keypoint_conf)
                or second >= len(person_keypoint_conf)
            ):
                continue

            if (
                person_keypoint_conf[first] < keypoint_threshold
                or person_keypoint_conf[second] < keypoint_threshold
            ):
                continue

            p1 = (
                int(person_keypoints[first][0]),
                int(person_keypoints[first][1]),
            )

            p2 = (
                int(person_keypoints[second][0]),
                int(person_keypoints[second][1]),
            )

            cv2.line(
                annotated,
                p1,
                p2,
                (255, 0, 0),
                2,
            )

        # --------------------------------------------------------------
        # Draw posture label.
        # --------------------------------------------------------------
        label = f"{posture} ({detection_conf:.2f})"

        label_y = max(y_min - 10, 25)

        cv2.putText(
            annotated,
            label,
            (x_min, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    observations_output = {
        "pose_detected": len(observations) > 0,
        "person_count": len(observations),
        "persons": observations,
    }

    if save_annotated:
        final_output_path = _save_annotated_image(
            annotated,
            image_path,
            output_path,
        )

        observations_output["annotated_image"] = str(
            final_output_path
        )

    return observations_output


def _save_annotated_image(
    image,
    image_path: Path,
    output_path: str | Path | None,
) -> Path:
    """Save the annotated image and return the final path."""

    if output_path is None:
        output_path = (
            image_path.parent
            / f"{image_path.stem}_pose.jpg"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Could not save annotated image to: {output_path}"
        )

    return output_path


def estimate_pose_frame(
    frame: Any,
    model_name: str = DEFAULT_MODEL,
    detection_confidence: float = 0.25,
    keypoint_threshold: float = 0.25,
    save_annotated: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run pose estimation on an OpenCV ndarray (frame).

    Mirrors ``estimate_pose`` but operates directly on an image array to avoid
    disk I/O. Returns the same observation dict structure. ``save_annotated``
    is currently unsupported for frames; provide ``output_path`` only when
    ``save_annotated=True``.
    """
    if not hasattr(frame, "shape"):
        raise TypeError("Frame must be an image array (e.g., NumPy ndarray).")
    model = _load_pose_model(model_name)
    results = model(frame, conf=detection_confidence, verbose=False)
    if not results:
        raise RuntimeError("The pose model returned no results.")
    result = results[0]
    observations: list[dict[str, Any]] = []
    annotated = frame.copy() if save_annotated else None
    if result.keypoints is None or result.boxes is None:
        observations_output = {"pose_detected": False, "person_count": 0, "persons": []}
        if save_annotated:
            raise NotImplementedError("Annotated image saving not supported for frame input.")
        return observations_output
    keypoints_xy = result.keypoints.xy.cpu().tolist()
    keypoints_conf = (
        result.keypoints.conf.cpu().tolist()
        if result.keypoints.conf is not None
        else []
    )
    boxes_xyxy = result.boxes.xyxy.cpu().tolist()
    box_conf = (
        result.boxes.conf.cpu().tolist()
        if result.boxes.conf is not None
        else []
    )
    person_count = len(keypoints_xy)
    for person_index in range(person_count):
        person_keypoints = keypoints_xy[person_index]
        person_keypoint_conf = (
            keypoints_conf[person_index]
            if person_index < len(keypoints_conf)
            else [0.0] * len(person_keypoints)
        )
        bbox = (
            tuple(boxes_xyxy[person_index])
            if person_index < len(boxes_xyxy)
            else (0.0, 0.0, 0.0, 0.0)
        )
        posture, posture_diagnostics = _classify_posture(
            person_keypoints,
            person_keypoint_conf,
            bbox,
            keypoint_threshold,
        )
        visible_keypoints = sum(conf >= keypoint_threshold for conf in person_keypoint_conf)
        detection_conf = (
            float(box_conf[person_index])
            if person_index < len(box_conf)
            else 0.0
        )
        person_observation = {
            "person_index": person_index,
            "person_detected": True,
            "detection_confidence": round(detection_conf, 4),
            "posture": posture,
            "visible_keypoints": visible_keypoints,
            "bounding_box": {
                "x_min": round(float(bbox[0]), 2),
                "y_min": round(float(bbox[1]), 2),
                "x_max": round(float(bbox[2]), 2),
                "y_max": round(float(bbox[3]), 2),
            },
            "keypoints": _serialize_keypoints(person_keypoints, person_keypoint_conf),
            "posture_diagnostics": posture_diagnostics,
        }
        observations.append(person_observation)
        if save_annotated:
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            for idx, pt in enumerate(person_keypoints):
                if idx >= len(person_keypoint_conf):
                    continue
                conf = person_keypoint_conf[idx]
                if conf < keypoint_threshold:
                    continue
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            for first, second in SKELETON:
                if (
                    first >= len(person_keypoints)
                    or second >= len(person_keypoints)
                    or first >= len(person_keypoint_conf)
                    or second >= len(person_keypoint_conf)
                ):
                    continue
                if (
                    person_keypoint_conf[first] < keypoint_threshold
                    or person_keypoint_conf[second] < keypoint_threshold
                ):
                    continue
                p1 = (int(person_keypoints[first][0]), int(person_keypoints[first][1]))
                p2 = (int(person_keypoints[second][0]), int(person_keypoints[second][1]))
                cv2.line(annotated, p1, p2, (255, 0, 0), 2)
            label = f"{posture} ({detection_conf:.2f})"
            label_y = max(y_min - 10, 25)
            cv2.putText(annotated, label, (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    observations_output = {
        "pose_detected": len(observations) > 0,
        "person_count": len(observations),
        "persons": observations,
    }
    if save_annotated:
        if output_path is None:
            raise ValueError("output_path required when save_annotated=True for frames.")
        final_output_path = _save_annotated_image(annotated, Path("frame_input"), output_path)
        observations_output["annotated_image"] = str(final_output_path)
    return observations_output

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrained human pose estimation. "
            "Posture labels are visual heuristics, not medical diagnoses."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ultralytics pose checkpoint (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Person detection confidence threshold",
    )

    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.25,
        help="Keypoint confidence threshold",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the annotated output image",
    )

    args = parser.parse_args()

    try:
        result = estimate_pose(
            image_path=args.image,
            model_name=args.model,
            detection_confidence=args.conf,
            keypoint_threshold=args.keypoint_conf,
            save_annotated=True,
            output_path=args.output,
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    except Exception as exc:
        print(f"Pose estimation error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
