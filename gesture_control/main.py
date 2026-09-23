import os

# Silence MediaPipe/TF C++ INFO+WARNING spam. Must be set before mediapipe
# is first imported (it chain-imports via hand_tracker below).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import logging
import threading
import time
import signal
import argparse
from typing import Optional

from gesture_control.config import CONFIG
from gesture_control.camera.camera import Camera
from gesture_control.vision.hand_tracker import HandTracker
from gesture_control.vision.gesture_recognizer import GestureRecognizer, GestureType, GestureState, GestureResult
from gesture_control.actions.media import play_pause
from gesture_control.actions.desktop import switch_space
from gesture_control.actions.fullscreen import toggle_fullscreen
from gesture_control.actions.volume import volume_up, volume_down
from gesture_control.menubar.app import run_menu_bar_app, GestureMenuBarApp

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class GestureControlApp:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        if debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        self.camera = Camera()
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()

        self.menu_app: Optional[GestureMenuBarApp] = None
        self.running = False
        self.enabled = False
        self.camera_error = False

        self._processing_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last_debug_print = 0.0
        self._frame_count = 0
        self._fps_start = time.time()

    def start(self) -> None:
        logger.info("Gesture Control started")

        access_ok = _check_accessibility()
        if not access_ok:
            logger.warning(
                "Accessibility permission MISSING -- Space switching and "
                "other key simulations will not work. Grant it at System "
                "Settings > Privacy & Security > Accessibility, then "
                "restart this app."
            )

        self.menu_app = run_menu_bar_app(
            enabled_callback=self._on_enabled_change,
            quit_callback=self._on_quit,
            toggle_callback=self._on_toggle,
        )
        self.menu_app.set_accessibility_ok(access_ok)

        self.camera.set_frame_callback(self._process_frame)
        self.camera.set_error_callback(self._on_camera_error)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()

        self.menu_app.run()

    def _signal_handler(self, signum, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _on_enabled_change(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            if enabled:
                self.camera_error = False
                if not self.camera.is_running():
                    self.camera.start()
                self.gesture_recognizer.reset()
                logger.info("Gesture recognition enabled")
            else:
                self.camera.stop()
                logger.info("Gesture recognition disabled")

    def _on_toggle(self, enabled: bool) -> None:
        self._on_enabled_change(enabled)

    def _on_quit(self) -> None:
        logger.info("Quit requested")
        self.stop()

    def _on_camera_error(self, error: str) -> None:
        logger.error(f"Camera error: {error}")
        self.camera_error = True
        if self.menu_app:
            self.menu_app.set_camera_available(False)

    def _processing_loop(self) -> None:
        while self.running:
            time.sleep(0.01)

    def _process_frame(self, frame) -> None:
        if not self.enabled:
            return

        try:
            hand = self.hand_tracker.process(frame)
            result = self.gesture_recognizer.process(hand)

            self._frame_count += 1
            self._maybe_print_debug(hand, result)

            if result.gesture != GestureType.NONE and result.state == GestureState.ACTION_EXECUTED:
                self._execute_action(result.gesture)
                if self.menu_app:
                    self.menu_app.show_feedback(result.gesture)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

    def _maybe_print_debug(self, hand, result: GestureResult) -> None:
        if not self.debug_mode:
            return

        now = time.time()
        if now - self._last_debug_print < 0.5:
            return

        self._last_debug_print = now

        hand_status = "YES" if hand else "NO"
        gesture_name = result.gesture.value if result.gesture != GestureType.NONE else "NONE"
        state_name = result.state.value
        confidence = f"{result.confidence:.2f}" if result.confidence > 0 else "0.00"
        cooldown = f"{result.cooldown_remaining:.1f}s" if result.cooldown_remaining > 0 else "0.0s"

        fps = self.camera.get_fps()

        print(
            f"\rHand: {hand_status} | Gesture: {gesture_name} | Confidence: {confidence} | "
            f"FPS: {fps:.1f} | State: {state_name} | Cooldown: {cooldown}",
            end="",
            flush=True,
        )

    def _execute_action(self, gesture: GestureType) -> None:
        logger.info(f"Action: {gesture.value.upper()}")

        try:
            if gesture == GestureType.OPEN_PALM:
                play_pause()
            elif gesture == GestureType.SWIPE_LEFT:
                switch_space("left")
            elif gesture == GestureType.SWIPE_RIGHT:
                switch_space("right")
            elif gesture == GestureType.FIST:
                toggle_fullscreen()
            elif gesture == GestureType.THUMB_UP:
                volume_up()
            elif gesture == GestureType.THUMB_DOWN:
                volume_down()
        except Exception as e:
            logger.error(f"Action execution failed: {e}")

    def stop(self) -> None:
        if not self.running:
            return

        self.running = False
        logger.info("Stopping Gesture Control...")

        self.camera.stop()
        self.hand_tracker.close()

        if self.menu_app:
            self.menu_app.stop()

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        logger.info("Gesture Control stopped")


def _check_accessibility() -> bool:
    """True if this process is trusted for Accessibility (required for
    System Events key simulation and CGEvent posting)."""
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception as e:
        logger.debug(f"Accessibility check unavailable: {e}")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Gesture Control - macOS menu-bar hand gesture recognition")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with console output")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no macOS actions)")
    args = parser.parse_args()

    if args.test:
        CONFIG.app.debug_mode = True
        logger.info("Running in TEST MODE - no macOS actions will be executed")

    app = GestureControlApp(debug_mode=args.debug or CONFIG.app.debug_mode)

    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    finally:
        app.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())