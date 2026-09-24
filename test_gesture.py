#!/usr/bin/env python3
"""
Test/Debug mode for AirControl.
Runs gesture recognition without executing macOS actions.
Shows real-time debug information.
"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import cv2
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aircontrol.config import CONFIG
from aircontrol.camera.camera import Camera
from aircontrol.vision.hand_tracker import HandTracker
from aircontrol.vision.gesture_recognizer import GestureRecognizer, GestureType, GestureState

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_test_mode(show_video: bool = False) -> None:
    CONFIG.app.debug_mode = True

    camera = Camera()
    hand_tracker = HandTracker()
    gesture_recognizer = GestureRecognizer()

    print("=" * 60)
    print("AIRCONTROL - TEST MODE (no macOS actions fire)")
    print("=" * 60)
    print("Show one hand for normal gestures, or hold LEFT FIST + swipe")
    print("RIGHT hand for app-switch mode.")
    print("Press 'q' to quit, 's' to save a frame.")
    print("=" * 60)
    print()

    frame_count = 0
    fps_start = time.time()
    current_fps = 0.0
    last_frame = {"frame": None}
    infer_ms = {"ms": 0.0}

    def process_frame(frame):
        nonlocal frame_count, fps_start, current_fps

        last_frame["frame"] = frame
        t0 = time.time()
        hands = hand_tracker.process(frame)
        frame_result = gesture_recognizer.process_frame(hands)
        infer_ms["ms"] = (time.time() - t0) * 1000.0

        frame_count += 1
        if time.time() - fps_start >= 1.0:
            current_fps = frame_count / (time.time() - fps_start)
            frame_count = 0
            fps_start = time.time()

        print_debug_info(
            hands, frame_result, current_fps, infer_ms["ms"], gesture_recognizer
        )

        if show_video:
            display_frame = frame.copy()
            for hand in hands:
                display_frame = hand_tracker.draw_landmarks(display_frame, hand)
            cv2.imshow("AirControl - Test Mode", display_frame)

    camera.set_frame_callback(process_frame)

    if not camera.start():
        logger.error("Failed to start camera")
        return

    try:
        while True:
            if show_video:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if last_frame["frame"] is not None:
                        cv2.imwrite(f"debug_frame_{int(time.time())}.jpg", last_frame["frame"])
                        print("Frame saved")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        hand_tracker.close()
        if show_video:
            cv2.destroyAllWindows()
        print("\nTest mode stopped.")


def print_debug_info(hands, frame_result, fps: float, infer_ms: float,
                     recognizer: GestureRecognizer) -> None:
    snap = recognizer.get_debug_snapshot()
    left = snap.get("left", {})
    right = snap.get("right", {})

    def fmt(side):
        if not side.get("detected"):
            return "no"
        return f"yes {side.get('gesture')}({side.get('conf')})"

    legacy = frame_result.legacy
    app_event = frame_result.app_event
    event_str = app_event.kind if app_event else "-"

    print(
        f"\rL:[{fmt(left)}] R:[{fmt(right)}] Mode:{frame_result.mode} "
        f"FistHold:{snap.get('fist_hold_ms', 0):.0f}ms Ev:{event_str} "
        f"Legacy:{legacy.gesture.value} FPS:{fps:.1f} Infer:{infer_ms:.1f}ms",
        end="",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="AirControl Test Mode")
    parser.add_argument("--video", action="store_true", help="Show video window with landmarks")
    args = parser.parse_args()

    run_test_mode(show_video=args.video)


if __name__ == "__main__":
    main()