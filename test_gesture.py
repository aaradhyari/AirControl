#!/usr/bin/env python3
"""
Test/Debug mode for Gesture Control.
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

sys.path.insert(0, '/Users/aaradhya-rai/GestureControl')

from gesture_control.config import CONFIG
from gesture_control.camera.camera import Camera
from gesture_control.vision.hand_tracker import HandTracker
from gesture_control.vision.gesture_recognizer import GestureRecognizer, GestureType, GestureState

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
    print("GESTURE CONTROL - TEST MODE")
    print("=" * 60)
    print("Show your hand to the camera.")
    print("Recognized gestures will be displayed below.")
    print("Press 'q' to quit, 's' to save a frame.")
    print("=" * 60)
    print()

    frame_count = 0
    fps_start = time.time()
    current_fps = 0.0
    last_frame = {"frame": None}

    def process_frame(frame):
        nonlocal frame_count, fps_start, current_fps

        last_frame["frame"] = frame
        hand = hand_tracker.process(frame)
        result = gesture_recognizer.process(hand)

        frame_count += 1
        if time.time() - fps_start >= 1.0:
            current_fps = frame_count / (time.time() - fps_start)
            frame_count = 0
            fps_start = time.time()

        print_debug_info(hand, result, current_fps, gesture_recognizer)

        if show_video:
            display_frame = frame.copy()
            if hand:
                display_frame = hand_tracker.draw_landmarks(display_frame, hand)
            cv2.imshow("Gesture Control - Test Mode", display_frame)

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


def print_debug_info(hand, result, fps: float, recognizer: GestureRecognizer) -> None:
    hand_status = "YES" if hand else "NO"
    gesture_name = result.gesture.value.upper() if result.gesture != GestureType.NONE else "NONE"
    state_name = result.state.value
    confidence = f"{result.confidence:.2f}" if result.confidence > 0 else "0.00"
    cooldown = f"{result.cooldown_remaining:.1f}s" if result.cooldown_remaining > 0 else "0.0s"
    hold_time = f"{result.hold_time:.2f}s" if result.hold_time > 0 else "0.00s"

    if hand:
        finger_states = recognizer.get_finger_states()
        fingers = []
        for name, state in finger_states.items():
            status = "↑" if state.extended else "↓"
            fingers.append(f"{name[0]}:{status}")
        finger_str = " ".join(fingers)
    else:
        finger_str = "N/A"

    print(
        f"\rHand: {hand_status} | Fingers: {finger_str} | "
        f"Gesture: {gesture_name:15s} | Conf: {confidence} | "
        f"FPS: {fps:5.1f} | State: {state_name:18s} | "
        f"Hold: {hold_time} | Cooldown: {cooldown}",
        end="",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Gesture Control Test Mode")
    parser.add_argument("--video", action="store_true", help="Show video window with landmarks")
    args = parser.parse_args()

    run_test_mode(show_video=args.video)


if __name__ == "__main__":
    main()