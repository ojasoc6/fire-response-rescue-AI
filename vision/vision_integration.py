"""Adapt real person and pose model outputs into one fusion-safe observation.

All returned values are visual observations.  In particular,
``possible_vulnerable_pose`` only means that pose geometry produced a
``LYING_CANDIDATE`` label; it is not a diagnosis or proof of injury.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional

from vision.person_detection import PersonDetector


VisionObservation = Dict[str, object]


class VisionIntegrationError(RuntimeError):
    """Raised when detector and pose observations cannot be combined safely."""


class VisionIntegrator:
    """Create a standardized, person-only visual observation for fusion."""

    POSE_FALLBACK_MIN_CONFIDENCE = 0.50
    POSE_FALLBACK_MIN_VISIBLE_KEYPOINTS = 6

    def __init__(
        self,
        person_detector: Optional[object] = None,
        pose_estimator: Optional[Callable[..., Mapping[str, object]]] = None,
    ) -> None:
        self._person_detector = person_detector or PersonDetector()
        self._pose_estimator = pose_estimator

    def analyze_image(self, image_path: str | Path) -> VisionObservation:
        """Run real person detection and pose estimation on one image path.

        Person detection and pose estimation are complementary sources.  The
        detector's highest-confidence person remains the main observation when
        available.  When it is not, a sufficiently confident pose result with
        enough visible keypoints can supply a visual fallback observation.
        """
        detector_observation: Mapping[str, object] = self._no_person_observation()
        try:
            detector_observation = self._person_detector.detect(image_path)
        except Exception as error:
            detector_error = error
        else:
            detector_error = None

        try:
            pose_result = self._get_pose_estimator()(image_path, save_annotated=False)
        except Exception as error:
            if detector_observation.get("person_detected", False):
                # A detector-only observation remains useful when pose is
                # unavailable, but carries no posture conclusion.
                return self._detector_only_observation(detector_observation)
            message = "Pose estimation failed"
            if detector_error is not None:
                message += f" after person detection also failed: {detector_error}"
            raise VisionIntegrationError(f"{message}: {error}") from error

        pose_persons = pose_result.get("persons", [])
        qualifying_pose_persons = self._qualifying_pose_persons(pose_persons)

        if detector_observation.get("person_detected", False):
            selected_box = detector_observation.get("bounding_box")
            if not isinstance(selected_box, Mapping):
                raise VisionIntegrationError("Detected person is missing a valid bounding box")

            pose_person = self._select_matching_pose_person(selected_box, pose_persons)
            posture = str(pose_person.get("posture", "UNKNOWN")) if pose_person else "UNKNOWN"
            return {
                "person_detected": True,
                "person_count": int(detector_observation.get("person_count", 1)),
                "confidence": float(detector_observation.get("confidence", 0.0)),
                "bounding_box": dict(selected_box),
                "posture": posture,
                "possible_vulnerable_pose": posture == "LYING_CANDIDATE",
                "detection_source": "person_detector+pose" if pose_person else "person_detector",
            }

        if qualifying_pose_persons:
            pose_person = max(
                qualifying_pose_persons,
                key=lambda person: float(person.get("detection_confidence", 0.0)),
            )
            return {
                "person_detected": True,
                "person_count": max(1, int(pose_result.get("person_count", len(qualifying_pose_persons)))),
                "confidence": float(pose_person["detection_confidence"]),
                "bounding_box": dict(pose_person["bounding_box"]),
                "posture": str(pose_person.get("posture", "UNKNOWN")),
                "possible_vulnerable_pose": pose_person.get("posture") == "LYING_CANDIDATE",
                "detection_source": "pose_fallback",
            }

        return self._no_person_observation()

    def _get_pose_estimator(self) -> Callable[..., Mapping[str, object]]:
        if self._pose_estimator is None:
            # Keep the normal sensor-only entry point independent of OpenCV and
            # the pose-model import until an image is explicitly requested.
            from vision.pose_estimation import estimate_pose

            self._pose_estimator = estimate_pose
        return self._pose_estimator

    @staticmethod
    def _select_matching_pose_person(
        selected_box: Mapping[str, object], persons: object
    ) -> Optional[Mapping[str, object]]:
        if not isinstance(persons, list):
            return None
        candidates = [person for person in persons if isinstance(person, Mapping)]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda person: (
                _intersection_over_union(selected_box, person.get("bounding_box")),
                float(person.get("detection_confidence", 0.0)),
            ),
        )

    def _qualifying_pose_persons(self, persons: object) -> list[Mapping[str, object]]:
        """Return pose people safe enough to act as a detector fallback."""
        if not isinstance(persons, list):
            return []
        qualifying: list[Mapping[str, object]] = []
        for person in persons:
            if not isinstance(person, Mapping):
                continue
            box = person.get("bounding_box")
            try:
                confidence = float(person.get("detection_confidence", 0.0))
                visible_keypoints = int(person.get("visible_keypoints", 0))
            except (TypeError, ValueError):
                continue
            if (
                isinstance(box, Mapping)
                and confidence >= self.POSE_FALLBACK_MIN_CONFIDENCE
                and visible_keypoints >= self.POSE_FALLBACK_MIN_VISIBLE_KEYPOINTS
            ):
                qualifying.append(person)
        return qualifying

    @staticmethod
    def _detector_only_observation(
        detector_observation: Mapping[str, object],
    ) -> VisionObservation:
        box = detector_observation.get("bounding_box")
        if not isinstance(box, Mapping):
            raise VisionIntegrationError("Detected person is missing a valid bounding box")
        return {
            "person_detected": True,
            "person_count": int(detector_observation.get("person_count", 1)),
            "confidence": float(detector_observation.get("confidence", 0.0)),
            "bounding_box": dict(box),
            "posture": "UNKNOWN",
            "possible_vulnerable_pose": False,
            "detection_source": "person_detector",
        }

    @staticmethod
    def _no_person_observation(person_count: int = 0) -> VisionObservation:
        return {
            "person_detected": False,
            "person_count": person_count,
            "confidence": 0.0,
            "bounding_box": None,
            "posture": "UNKNOWN",
            "possible_vulnerable_pose": False,
            "detection_source": "none",
        }


def _intersection_over_union(
    first: Mapping[str, object], second: object
) -> float:
    """Return IoU for two serialized boxes, or 0 for invalid geometry."""
    if not isinstance(second, Mapping):
        return 0.0
    try:
        first_x_min, first_y_min, first_x_max, first_y_max = (
            float(first[key]) for key in ("x_min", "y_min", "x_max", "y_max")
        )
        second_x_min, second_y_min, second_x_max, second_y_max = (
            float(second[key]) for key in ("x_min", "y_min", "x_max", "y_max")
        )
    except (KeyError, TypeError, ValueError):
        return 0.0

    intersection_width = max(
        0.0, min(first_x_max, second_x_max) - max(first_x_min, second_x_min)
    )
    intersection_height = max(
        0.0, min(first_y_max, second_y_max) - max(first_y_min, second_y_min)
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_x_max - first_x_min) * max(
        0.0, first_y_max - first_y_min
    )
    second_area = max(0.0, second_x_max - second_x_min) * max(
        0.0, second_y_max - second_y_min
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0
