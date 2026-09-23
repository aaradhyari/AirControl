import cv2
import logging
import threading
import time
from typing import Optional, Callable, Tuple
import numpy as np

from aircontrol.config import CONFIG, CameraConfig

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CONFIG.camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._fps_start = time.time()
        self._current_fps = 0.0

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        self.frame_callback = callback

    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        self.error_callback = callback

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return True

            try:
                self.cap = cv2.VideoCapture(self.config.index)
                if not self.cap.isOpened():
                    self._handle_error("Failed to open camera")
                    return False

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

                logger.info(
                    f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
                )

                self.running = True
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                return True

            except Exception as e:
                self._handle_error(f"Camera start error: {e}")
                return False

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                return
            self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        logger.info("Camera stopped")

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / self.config.target_fps

        while self.running:
            loop_start = time.time()

            with self._lock:
                if not self.cap or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()

            if not ret or frame is None:
                self._handle_error("Failed to read frame")
                time.sleep(0.1)
                continue

            self._last_frame = frame
            self._frame_count += 1

            if self.frame_callback:
                try:
                    self.frame_callback(frame)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._update_fps()

    def _update_fps(self) -> None:
        now = time.time()
        if now - self._fps_start >= 1.0:
            self._current_fps = self._frame_count / (now - self._fps_start)
            self._frame_count = 0
            self._fps_start = now

    def get_fps(self) -> float:
        return self._current_fps

    def get_last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame.copy() if self._last_frame is not None else None

    def is_running(self) -> bool:
        return self.running

    def _handle_error(self, message: str) -> None:
        logger.error(message)
        if self.error_callback:
            try:
                self.error_callback(message)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")


CameraConfig = type(CONFIG.camera)