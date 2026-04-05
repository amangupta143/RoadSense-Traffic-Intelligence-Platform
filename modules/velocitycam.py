"""
VelocityCam — FINAL STABLE VERSION
- Browser-compatible video output (mp4v)
- Accurate perspective-based speed
- Stable tracking + edge-case handling
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Optional, Dict

import supervision as sv
from ultralytics import YOLO


# ---------------- CONFIG ---------------- #
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 960
YOLO_MODEL = "yolov8n.pt"

VEHICLE_CLASSES = {2, 3, 5, 7}

# 🔥 MUST match real-world visible road distance (meters)
TARGET_WIDTH = 25
TARGET_HEIGHT = 100   # adjust if speed is wrong

DEFAULT_SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

DEFAULT_TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])


# ---------------- VIEW TRANSFORM ---------------- #
class ViewTransformer:
    def __init__(self, source, target):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32)
        )

    def transform(self, pts):
        if pts.size == 0:
            return pts
        pts = pts.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2)


# ---------------- MAIN FUNCTION ---------------- #
def run_speed_vision(
    input_path: str,
    output_path: str,
    source_polygon: Optional[np.ndarray] = None,
    model_path: str = YOLO_MODEL
) -> Dict:

    source = source_polygon if source_polygon is not None else DEFAULT_SOURCE
    target = DEFAULT_TARGET

    model = YOLO(model_path)

    video_info = sv.VideoInfo.from_video_path(input_path)
    frames = sv.get_video_frames_generator(input_path)

    tracker = sv.ByteTrack(frame_rate=video_info.fps)
    polygon_zone = sv.PolygonZone(source)
    transformer = ViewTransformer(source, target)

    history = defaultdict(lambda: deque(maxlen=int(video_info.fps)))
    speeds = {}

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.6,
        text_thickness=2,
        text_position=sv.Position.BOTTOM_CENTER
    )

    # ✅ FIXED: Browser-safe codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        float(video_info.fps),
        (int(video_info.width), int(video_info.height))
    )

    if not writer.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open. Check codec support.")

    for frame in frames:

        # -------- DETECTION -------- #
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        if len(detections) == 0:
            writer.write(frame)
            continue

        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[np.isin(detections.class_id, list(VEHICLE_CLASSES))]

        # polygon filter (safe)
        try:
            detections = detections[polygon_zone.trigger(detections)]
        except:
            pass

        # -------- TRACKING -------- #
        detections = detections.with_nms(IOU_THRESHOLD)
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is None:
            writer.write(frame)
            continue

        # -------- SPEED CALC -------- #
        anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        world_pts = transformer.transform(anchors)

        for tid, (_, y) in zip(detections.tracker_id, world_pts):
            history[int(tid)].append(float(y))

        labels = []

        for tid in detections.tracker_id:
            tid = int(tid)
            pts = history[tid]

            if len(pts) < video_info.fps / 2:
                labels.append(f"#{tid}")
                continue

            distance = abs(pts[-1] - pts[0])  # meters (after transform)

            time_sec = len(pts) / video_info.fps

            if time_sec <= 0:
                labels.append(f"#{tid}")
                continue

            speed = (distance / time_sec) * 3.6
            speed = float(round(speed, 1))

            speeds[tid] = speed
            labels.append(f"#{tid} {speed} km/h")

        # -------- DRAW -------- #
        frame_out = frame.copy()
        frame_out = box_annotator.annotate(frame_out, detections)
        frame_out = label_annotator.annotate(frame_out, detections, labels)

        writer.write(frame_out)

    writer.release()

    # -------- OUTPUT -------- #
    speed_values = list(speeds.values())

    return {
        "vehicles_tracked": int(len(speeds)),
        "avg_speed_kmh": float(round(sum(speed_values) / len(speed_values), 1)) if speed_values else 0.0,
        "max_speed_kmh": float(round(max(speed_values), 1)) if speed_values else 0.0,
        "speed_data": [
            {"vehicle_id": int(k), "speed_kmh": float(v)}
            for k, v in speeds.items()
        ]
    }