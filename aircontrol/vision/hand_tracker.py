import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import logging
import time
import os
import sys
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

from aircontrol.config import CONFIG

logger = logging.getLogger(__name__)


def _model_path() -> str:
    candidates = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "models",
                "hand_landmarker.task",
            )
        )
    ]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "models", "hand_landmarker.task"))
        candidates.append(os.path.join(meipass, "hand_landmarker.task"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


MODEL_PATH = _model_path()


@dataclass
class HandLandmarks:
    landmarks: np.ndarray
    handedness: str
    score: float
    center: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]


class HandTracker:
    def __init__(self, config: Optional[object] = None):
        self.config = config or CONFIG
        self.gesture_config = self.config.gesture

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"MediaPipe hand landmarker model not found at {MODEL_PATH}")

        # NOTE: mediapipe 1.0.x crashes on Apple Silicon (M4) inside
        # DrishtiMetalHelper ("Service is unavailable") even when CPU is
        # forced. Pin mediapipe==0.10.35 (see requirements.txt), which works.
        # Build BaseOptions defensively across 0.10.x / 1.x API differences.
        base_options_kwargs = {"model_asset_path": MODEL_PATH}
        try:
            base_options_kwargs["delegate"] = mp_python.BaseOptions.Delegate.CPU
        except Exception:
            pass
        base_options = mp_python.BaseOptions(**base_options_kwargs)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._latest_result = None
        self._result_ready = False

        # Per-hand histories keyed by MediaPipe handedness label
        # ("Left"/"Right") so one hand's state never overwrites the other's.
        self._histories: Dict[str, Dict[str, list]] = {}
        self._max_history = 10

    def _hist(self, label: str) -> Dict[str, list]:
        entry = self._histories.get(label)
        if entry is None:
            entry = {"landmarks": [], "centers": []}
            self._histories[label] = entry
        return entry

    def _result_callback(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        self._latest_result = result
        self._result_ready = True

    def process(self, frame: np.ndarray) -> List[HandLandmarks]:
        """Detect up to 2 hands. Returns one HandLandmarks per hand
        (possibly empty). Never uses screen position for identity --
        handedness comes straight from MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        self._result_ready = False
        self.landmarker.detect_async(mp_image, timestamp_ms)

        timeout = 0.1
        start = time.time()
        while not self._result_ready and time.time() - start < timeout:
            time.sleep(0.001)

        if self._latest_result is None:
            self._clear_history()
            return []

        result = self._latest_result

        if not result.hand_landmarks:
            self._clear_history()
            return []

        h, w = frame.shape[:2]
        hands: List[HandLandmarks] = []
        seen_labels = set()
        for hand_landmarks, handedness_info in zip(
            result.hand_landmarks, result.handedness or []
        ):
            label = (
                handedness_info[0].category_name
                if handedness_info
                else "Unknown"
            )
            score = handedness_info[0].score if handedness_info else 1.0
            if label in seen_labels:
                # Degenerate duplicate label: keep the higher-score hand so
                # per-hand state keyed by label can never collide.
                prev = next(h for h in hands if h.handedness == label)
                if score <= prev.score:
                    continue
                hands.remove(prev)
            seen_labels.add(label)

            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
            )

            center_x = np.mean(landmarks[:, 0]) * w
            center_y = np.mean(landmarks[:, 1]) * h

            x_coords = landmarks[:, 0] * w
            y_coords = landmarks[:, 1] * h
            bbox = (
                int(np.min(x_coords)),
                int(np.min(y_coords)),
                int(np.max(x_coords)),
                int(np.max(y_coords)),
            )

            hist = self._hist(label)
            hist["landmarks"].append(landmarks)
            hist["centers"].append((center_x, center_y, time.time()))
            if len(hist["landmarks"]) > self._max_history:
                hist["landmarks"].pop(0)
            if len(hist["centers"]) > self._max_history:
                hist["centers"].pop(0)

            hands.append(
                HandLandmarks(
                    landmarks=landmarks,
                    handedness=label,
                    score=score,
                    center=(center_x, center_y),
                    bounding_box=bbox,
                )
            )

        # Drop histories for labels no longer visible so stale hands cannot
        # leak into a later appearance.
        for label in list(self._histories):
            if label not in seen_labels:
                del self._histories[label]

        return hands

    def get_smoothed_landmarks(self, label: Optional[str] = None) -> Optional[np.ndarray]:
        hist = self._histories.get(label, {}) if label else None
        landmark_history = hist.get("landmarks", []) if hist else []
        if len(landmark_history) < self.gesture_config.smoothing_frames:
            return landmark_history[-1] if landmark_history else None

        recent = landmark_history[-self.gesture_config.smoothing_frames :]
        return np.mean(recent, axis=0)

    def get_center_history(self, label: Optional[str] = None) -> List[Tuple[float, float, float]]:
        if label:
            return list(self._histories.get(label, {}).get("centers", []))
        merged: List[Tuple[float, float, float]] = []
        for hist in self._histories.values():
            merged.extend(hist["centers"])
        return merged

    def get_landmark_history(self, label: Optional[str] = None) -> List[np.ndarray]:
        if label:
            return list(self._histories.get(label, {}).get("landmarks", []))
        merged: List[np.ndarray] = []
        for hist in self._histories.values():
            merged.extend(hist["landmarks"])
        return merged

    def _clear_history(self) -> None:
        self._histories.clear()

    def draw_landmarks(self, frame: np.ndarray, hand: HandLandmarks) -> np.ndarray:
        h, w = frame.shape[:2]
        landmarks = hand.landmarks.copy()
        landmarks[:, 0] *= w
        landmarks[:, 1] *= h
        landmarks = landmarks.astype(np.int32)

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for connection in connections:
            start_idx, end_idx = connection
            pt1 = tuple(landmarks[start_idx])
            pt2 = tuple(landmarks[end_idx])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        for pt in landmarks:
            cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1)

        return frame

    def close(self) -> None:
        if self.landmarker:
            self.landmarker.close()
        self._clear_history()