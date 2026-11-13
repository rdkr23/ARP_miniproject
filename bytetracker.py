import numpy as np
import supervision as sv
from charset_normalizer import detect
from trackers import ByteTrackTracker


class FootballByteTracker:
    def __init__(self, football_label=1, track_activation_threshold=0.7):
        self.football_label = football_label

        self.tracker = ByteTrackTracker(
            lost_track_buffer=8,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.01,
            track_activation_threshold=track_activation_threshold
        )

    def update(self, prediction):
        detections = prediction_to_bytetrack(prediction, self.football_label)

        self.tracker.update(detections)

        tracks = self.tracker.tracked_objects

        if not tracks:
            return None

        return tracks


def prediction_to_bytetrack(prediction, football_label):
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    mask = labels == football_label

    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if len(boxes) == 0:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels.astype(np.int32),
    )