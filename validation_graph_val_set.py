"""Evaluate the two ball detectors on the SoccerNet validation sequences.

The validation set is the same one used in ``main.py``: SoccerNet tracking
sequences (e.g. ``D:/football-tracker/test/test/SNMOT-136``).  Each sequence
contains ``img1/`` (frames), ``gt/gt.txt`` (MOT rows
``frame, track_id, x, y, w, h, ...`` in pixels) and ``gameinfo.ini``, which
says which track IDs are balls.  Only ball tracks are used as ground truth.
"""

from __future__ import annotations

import argparse
import configparser
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from bytetracker import FootballByteTracker
from non_neural import NonNeuralMethod
from rcnn_data_handler import RCNNDataHandler


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def intersection_area(first: Box, second: Box) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def ball_track_ids(sequence_dir: Path) -> set[int]:
    game_info = configparser.ConfigParser()
    if not game_info.read(sequence_dir / "gameinfo.ini"):
        raise FileNotFoundError(f"Missing gameinfo.ini in {sequence_dir}")

    num_tracklets = game_info.getint("Sequence", "num_tracklets")
    return {
        track_id
        for track_id in range(1, num_tracklets + 1)
        if "ball" in game_info.get("Sequence", f"trackletID_{track_id}")
    }


def read_sequence_labels(sequence_dir: Path) -> dict[int, list[Box]]:
    """Return ``{frame_number: [ball boxes]}`` for one SoccerNet sequence."""
    ball_ids = ball_track_ids(sequence_dir)
    labels: dict[int, list[Box]] = defaultdict(list)

    gt_path = sequence_dir / "gt" / "gt.txt"
    for line in gt_path.read_text().splitlines():
        values = line.split(",")
        if len(values) < 6:
            continue
        frame, track_id = int(values[0]), int(values[1])
        if track_id not in ball_ids:
            continue
        x, y, width, height = map(float, values[2:6])
        labels[frame].append(Box(x, y, x + width, y + height))
    return labels


def load_validation_set(data_root: Path, sequences: list[str]) -> list[tuple[str, list[tuple[Path, list[Box]]]]]:
    """Return ``[(sequence_name, [(frame_path, ball_boxes), ...]), ...]``."""
    validation_set = []
    for sequence in sequences:
        sequence_dir = data_root / sequence
        frame_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No validation frames found in {sequence_dir / 'img1'}")
        labels = read_sequence_labels(sequence_dir)
        samples = [(path, labels.get(int(path.stem), [])) for path in frame_paths]
        validation_set.append((sequence, samples))
    return validation_set


def prediction_boxes(prediction: dict) -> list[Box]:
    tensor = prediction.get("boxes")
    if tensor is None:
        return []
    return [Box(*map(float, box)) for box in tensor.detach().cpu().numpy()]


def tracked_boxes(tracks) -> list[Box]:
    if tracks is None:
        return []
    return [Box(*map(float, box)) for box in tracks.xyxy]


def match_boxes(
    predictions: Iterable[Box],
    labels: Iterable[Box],
    minimum_intersection: float,
) -> tuple[list[tuple[Box, Box, float]], list[Box], list[Box]]:
    predictions = list(predictions)
    labels = list(labels)
    candidates = [
        (intersection_area(prediction, label), prediction_index, label_index)
        for prediction_index, prediction in enumerate(predictions)
        for label_index, label in enumerate(labels)
        if intersection_area(prediction, label) >= minimum_intersection
    ]
    candidates.sort(reverse=True)

    used_predictions = set()
    used_labels = set()
    matches = []
    for overlap, prediction_index, label_index in candidates:
        if prediction_index in used_predictions or label_index in used_labels:
            continue
        used_predictions.add(prediction_index)
        used_labels.add(label_index)
        matches.append((predictions[prediction_index], labels[label_index], overlap))

    unmatched_predictions = [
        prediction for index, prediction in enumerate(predictions)
        if index not in used_predictions
    ]
    unmatched_labels = [
        label for index, label in enumerate(labels)
        if index not in used_labels
    ]
    return matches, unmatched_predictions, unmatched_labels


def evaluate_detector(
    detector_name: str,
    detector,
    make_tracker,
    validation_set: list[tuple[str, list[tuple[Path, list[Box]]]]],
    minimum_intersection: float,
) -> dict:
    counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    true_positive_overlap_percentages = []
    total_intersection = 0.0
    total_label_area = 0.0

    # Fresh tracker per sequence so tracks do not leak between clips.
    samples = [
        (sequence_index, frame_path, labels)
        for sequence_index, (_, sequence_samples) in enumerate(validation_set)
        for frame_path, labels in sequence_samples
    ]

    start_time = time.perf_counter()
    total_frames = len(samples)
    progress_interval = max(1, total_frames // 100)
    current_sequence = None
    tracker = None

    print(f"\n[{detector_name}] Validating {total_frames} frames...", flush=True)
    for frame_index, (sequence_index, frame_path, labels) in enumerate(samples, 1):
        if sequence_index != current_sequence:
            current_sequence = sequence_index
            tracker = make_tracker()
        detection = detector(str(frame_path))[0]
        predictions = tracked_boxes(tracker.update(detection))
        matches, unmatched_predictions, unmatched_labels = match_boxes(
            predictions, labels, minimum_intersection
        )

        counts["true_positive"] += len(matches)
        counts["false_positive"] += len(unmatched_predictions)
        counts["false_negative"] += len(unmatched_labels)
        if not predictions and not labels:
            counts["true_negative"] += 1

        for _, label, overlap in matches:
            total_intersection += overlap
            total_label_area += label.area
            true_positive_overlap_percentages.append(100.0 * overlap / label.area)

        if frame_index == total_frames or frame_index % progress_interval == 0:
            elapsed = time.perf_counter() - start_time
            frames_per_second = frame_index / elapsed if elapsed else 0.0
            remaining = total_frames - frame_index
            eta = remaining / frames_per_second if frames_per_second else 0.0
            print(
                f"\r[{detector_name}] {frame_index}/{total_frames} "
                f"({100.0 * frame_index / total_frames:6.2f}%) "
                f"elapsed={elapsed:7.1f}s ETA={eta:7.1f}s",
                end="",
                flush=True,
            )
    print(flush=True)

    return {
        "name": detector_name,
        "counts": counts,
        "tp_area_overlap_mean": (
            float(np.mean(true_positive_overlap_percentages))
            if true_positive_overlap_percentages else 0.0
        ),
        "area_overlap_recall": (
            100.0 * total_intersection / total_label_area
            if total_label_area else 0.0
        ),
        "any_overlap_precision": (
            counts["true_positive"] / (counts["true_positive"] + counts["false_positive"])
            if counts["true_positive"] + counts["false_positive"] else 0.0
        ),
        "any_overlap_recall": (
            counts["true_positive"] / (counts["true_positive"] + counts["false_negative"])
            if counts["true_positive"] + counts["false_negative"] else 0.0
        ),
    }


def load_detectors(neural_model: str, non_neural_model: str):
    neural = RCNNDataHandler()
    neural.load_model(str(Path(neural_model)))

    non_neural = NonNeuralMethod()
    non_neural.load_model(str(Path(non_neural_model)))
    make_tracker = lambda: FootballByteTracker(football_label=1)
    return [
        ("neural", neural.detect, make_tracker),
        ("non-neural", non_neural.detect, make_tracker),
    ]


def print_results(results: list[dict]) -> None:
    for result in results:
        counts = result["counts"]
        print(f"\n{result['name']}")
        print("  % area overlap (intersection / ground-truth area):")
        print(f"    mean TP overlap: {result['tp_area_overlap_mean']:.2f}%")
        print(f"    area-overlap recall: {result['area_overlap_recall']:.2f}%")
        print("  any overlap:")
        print(
            f"    TP={counts['true_positive']}  FP={counts['false_positive']}  "
            f"TN={counts['true_negative']}  FN={counts['false_negative']}"
        )
        print(f"    precision: {100 * result['any_overlap_precision']:.2f}%")
        print(f"    recall: {100 * result['any_overlap_recall']:.2f}%")


def plot_results(results: list[dict], output_path: Path) -> None:
    names = [result["name"] for result in results]
    x = np.arange(len(names))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(
        x - width / 2,
        [result["tp_area_overlap_mean"] for result in results],
        width,
        label="Mean TP area overlap (%)",
    )
    axes[0].bar(
        x + width / 2,
        [100 * result["any_overlap_recall"] for result in results],
        width,
        label="Any-overlap recall (%)",
    )
    axes[0].set_ylabel("Percentage")
    axes[0].set_title("Overlap measurements")
    axes[0].set_xticks(x, names)
    axes[0].legend()

    categories = ["true_positive", "false_positive", "true_negative", "false_negative"]
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(categories))
    for offset, category in zip(offsets, categories):
        axes[1].bar(
            x + offset,
            [result["counts"][category] for result in results],
            width,
            label=category,
        )
    axes[1].set_title("Any-overlap confusion counts")
    axes[1].set_xticks(x, names)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"\nSaved graph to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/football-tracker/test/test"))
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["SNMOT-136", "SNMOT-150"],
        help="SoccerNet sequence folders (inside --data-root) to validate on.",
    )
    parser.add_argument(
        "--neural-model",
        default="1epoch_updated",
        help="Model stem used by RCNNDataHandler.load_model.",
    )
    parser.add_argument(
        "--non-neural-model",
        default="3percent_32_32_size_40_pc2",
        help="Model stem used by NonNeuralMethod.load_model.",
    )
    parser.add_argument("--minimum-intersection", type=float, default=1.0)
    parser.add_argument("--plot", type=Path, default=Path("validation_metrics.png"))
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    validation_set = load_validation_set(args.data_root, args.sequences)
    for sequence, samples in validation_set:
        ball_frames = sum(1 for _, labels in samples if labels)
        print(f"{sequence}: {len(samples)} frames, {ball_frames} with a labelled ball")

    detectors = load_detectors(args.neural_model, args.non_neural_model)
    results = [
        evaluate_detector(
            name,
            detector,
            make_tracker,
            validation_set,
            args.minimum_intersection,
        )
        for name, detector, make_tracker in detectors
    ]
    print_results(results)
    plot_results(results, args.plot)


if __name__ == "__main__":
    main()