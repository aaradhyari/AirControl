import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import logging
import time
import os
import sys
from typing import Optional, List, Tuple
from dataclasses import dataclass

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
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._latest_result = None
        self._result_ready = False

        self._landmark_history: List[np.ndarray] = []
        self._center_history: List[Tuple[float, float, float]] = []
        self._max_history = 10

    def _result_callback(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        self._latest_result = result
        self._result_ready = True

    def process(self, frame: np.ndarray) -> Optional[HandLandmarks]:
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
            return None

        result = self._latest_result

        if not result.hand_landmarks:
            self._clear_history()
            return None

        hand_landmarks = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name if result.handedness else "Right"
        score = result.handedness[0][0].score if result.handedness else 1.0

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
        )

        h, w = frame.shape[:2]
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

        self._landmark_history.append(landmarks)
        self._center_history.append((center_x, center_y, time.time()))
        if len(self._landmark_history) > self._max_history:
            self._landmark_history.pop(0)
        if len(self._center_history) > self._max_history:
            self._center_history.pop(0)

        return HandLandmarks(
            landmarks=landmarks,
            handedness=handedness,
            score=score,
            center=(center_x, center_y),
            bounding_box=bbox,
        )

    def get_smoothed_landmarks(self) -> Optional[np.ndarray]:
        if len(self._landmark_history) < self.gesture_config.smoothing_frames:
            return self._landmark_history[-1] if self._landmark_history else None

        recent = self._landmark_history[-self.gesture_config.smoothing_frames :]
        return np.mean(recent, axis=0)

    def get_center_history(self) -> List[Tuple[float, float, float]]:
        return self._center_history.copy()

    def get_landmark_history(self) -> List[np.ndarray]:
        return self._landmark_history.copy()

    def _clear_history(self) -> None:
        self._landmark_history.clear()
        self._center_history.clear()

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