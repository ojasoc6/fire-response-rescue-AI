"""Batch evaluation for the vision integration pipeline.

Runs person detection and pose estimation on image datasets, saves
structured evaluation results to CSV, and displays summary metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.vision_integration import VisionIntegrator


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "vision_test" / "vision_eval"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "vision_test" / "vision_eval_results.csv"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

CSV_FIELDNAMES = [
    "filename",
    "person_detected",
    "person_count",
    "confidence",
    "posture",
    "possible_vulnerable_pose",
    "detection_source",
]


def discover_image_files(directory: Path) -> List[Path]:
    """Discover all supported image files in a directory sorted by name."""
    if not directory.is_dir():
        return []
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def evaluate_images(
    image_dir: Path,
    output_csv: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Run vision integration evaluation on all images in image_dir.

    Returns the evaluation rows and any recorded image-level errors.
    """
    image_paths = discover_image_files(image_dir)
    integrator = VisionIntegrator()

    rows: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    for image_path in image_paths:
        try:
            obs = integrator.analyze_image(image_path)
            rows.append(
                {
                    "filename": image_path.name,
                    "person_detected": bool(obs.get("person_detected", False)),
                    "person_count": int(obs.get("person_count", 0)),
                    "confidence": float(obs.get("confidence", 0.0)),
                    "posture": str(obs.get("posture", "UNKNOWN")),
                    "possible_vulnerable_pose": bool(
                        obs.get("possible_vulnerable_pose", False)
                    ),
                    "detection_source": str(obs.get("detection_source", "none")),
                }
            )
        except Exception as error:
            errors[image_path.name] = str(error)
            rows.append(
                {
                    "filename": image_path.name,
                    "person_detected": False,
                    "person_count": 0,
                    "confidence": 0.0,
                    "posture": "UNKNOWN",
                    "possible_vulnerable_pose": False,
                    "detection_source": "error",
                }
            )

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "person_detected": row["person_detected"],
                    "person_count": row["person_count"],
                    "confidence": f"{row['confidence']:.4f}",
                    "posture": row["posture"],
                    "possible_vulnerable_pose": row["possible_vulnerable_pose"],
                    "detection_source": row["detection_source"],
                }
            )

    return rows, errors


def print_evaluation_table(rows: List[Dict[str, Any]]) -> None:
    """Print a clean, aligned table of evaluation results to the terminal."""
    if not rows:
        print("No images found to evaluate.")
        return

    col_widths = {
        "filename": max(len("filename"), max(len(r["filename"]) for r in rows)),
        "person_detected": len("person_detected"),
        "person_count": len("person_count"),
        "confidence": len("confidence"),
        "posture": max(len("posture"), max(len(r["posture"]) for r in rows)),
        "possible_vulnerable_pose": len("possible_vulnerable_pose"),
        "detection_source": max(
            len("detection_source"), max(len(r["detection_source"]) for r in rows)
        ),
    }

    header = (
        f"{'filename':<{col_widths['filename']}} | "
        f"{'person_detected':<{col_widths['person_detected']}} | "
        f"{'person_count':<{col_widths['person_count']}} | "
        f"{'confidence':<{col_widths['confidence']}} | "
        f"{'posture':<{col_widths['posture']}} | "
        f"{'possible_vulnerable_pose':<{col_widths['possible_vulnerable_pose']}} | "
        f"{'detection_source':<{col_widths['detection_source']}}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for r in rows:
        detected_str = str(r["person_detected"])
        count_str = str(r["person_count"])
        conf_str = f"{r['confidence']:.4f}"
        vulnerable_str = str(r["possible_vulnerable_pose"])

        print(
            f"{r['filename']:<{col_widths['filename']}} | "
            f"{detected_str:<{col_widths['person_detected']}} | "
            f"{count_str:<{col_widths['person_count']}} | "
            f"{conf_str:<{col_widths['confidence']}} | "
            f"{r['posture']:<{col_widths['posture']}} | "
            f"{vulnerable_str:<{col_widths['possible_vulnerable_pose']}} | "
            f"{r['detection_source']:<{col_widths['detection_source']}}"
        )
    print(separator)


def print_summary(rows: List[Dict[str, Any]], errors: Dict[str, str]) -> None:
    """Print the final summary metrics."""
    total_images = len(rows)
    detected_count = sum(1 for r in rows if r["person_detected"])
    not_detected_count = sum(1 for r in rows if not r["person_detected"])
    upright_count = sum(1 for r in rows if r["posture"] == "UPRIGHT")
    sitting_count = sum(1 for r in rows if r["posture"] == "SITTING")
    lying_count = sum(1 for r in rows if r["posture"] == "LYING_CANDIDATE")
    unknown_count = sum(1 for r in rows if r["posture"] == "UNKNOWN")
    fallback_count = sum(1 for r in rows if r["detection_source"] == "pose_fallback")

    print("\nSummary:")
    print(f"total images: {total_images}")
    print(f"images with person detected: {detected_count}")
    print(f"images with no person detected: {not_detected_count}")
    print(f"number classified as UPRIGHT: {upright_count}")
    print(f"number classified as SITTING: {sitting_count}")
    print(f"number classified as LYING_CANDIDATE: {lying_count}")
    print(f"number classified as UNKNOWN: {unknown_count}")
    print(f"number using pose_fallback: {fallback_count}")

    if errors:
        print("\nImage errors encountered:")
        for filename, err in errors.items():
            print(f"- {filename}: {err}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch evaluate vision integration on test images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Path to folder containing test images (default: data/vision_test/vision_eval)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output CSV file (default: data/vision_test/vision_eval_results.csv)",
    )
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        print(f"Error: Directory not found: {args.image_dir}", file=sys.stderr)
        return 1

    rows, errors = evaluate_images(args.image_dir, args.output_csv)
    print_evaluation_table(rows)
    print_summary(rows, errors)
    print(f"\nResults saved to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
