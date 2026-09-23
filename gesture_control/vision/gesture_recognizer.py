import time
import logging
import math
from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from gesture_control.config import CONFIG
from gesture_control.vision.hand_tracker import HandLandmarks

logger = logging.getLogger(__name__)


class GestureType(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"


class GestureState(Enum):
    IDLE = "idle"
    HAND_DETECTED = "hand_detected"
    GESTURE_CANDIDATE = "gesture_candidate"
    GESTURE_CONFIRMED = "gesture_confirmed"
    ACTION_EXECUTED = "action_executed"
    COOLDOWN = "cooldown"
    WAIT_FOR_RELEASE = "wait_for_release"


@dataclass
class GestureResult:
    gesture: GestureType = GestureType.NONE
    confidence: float = 0.0
    state: GestureState = GestureState.IDLE
    hold_time: float = 0.0
    cooldown_remaining: float = 0.0


@dataclass
class FingerState:
    extended: bool = False
    confidence: float = 0.0


class GestureRecognizer:
    def __init__(self, config: Optional[object] = None):
        self.config = config or CONFIG
        self.gesture_config = self.config.gesture

        self._state = GestureState.IDLE
        self._candidate_gesture: Optional[GestureType] = None
        self._candidate_start_time: float = 0.0
        self._candidate_start_center: Tuple[float, float] = (0.0, 0.0)
        self._recent_static: List[GestureType] = []
        self._last_action_time: float = 0.0
        self._last_gesture: Optional[GestureType] = None
        self._hand_present = False
        self._release_confirmed = False
        self._needs_release = False
        self._swipe_locked = False
        self._center_history: List[Tuple[float, float, float]] = []
        self._last_landmarks: Optional[np.ndarray] = None

        self._finger_states: dict = {
            "thumb": FingerState(),
            "index": FingerState(),
            "middle": FingerState(),
            "ring": FingerState(),
            "pinky": FingerState(),
        }

    def process(self, hand: Optional[HandLandmarks]) -> GestureResult:
        current_time = time.time()

        if hand is None:
            return self._handle_no_hand(current_time)

        self._hand_present = True
        self._release_confirmed = False

        landmarks = hand.landmarks
        self._analyze_fingers(landmarks, hand.handedness)

        static_gesture = self._recognize_static_gesture()
        swipe_gesture = self._recognize_swipe(hand)

        result = self._update_state_machine(
            current_time, static_gesture, swipe_gesture, hand
        )
        return result

    def _handle_no_hand(self, current_time: float) -> GestureResult:
        self._needs_release = False
        self._swipe_locked = False
        if self._state == GestureState.WAIT_FOR_RELEASE:
            self._state = GestureState.IDLE
            self._candidate_gesture = None
            self._release_confirmed = True
        elif self._state in (GestureState.ACTION_EXECUTED, GestureState.COOLDOWN):
            # Hand vanished right after an action: once the cooldown lapses
            # there is nothing left to wait for, go straight back to IDLE.
            # (Without this the machine sat in ACTION_EXECUTED forever and
            # no static gesture could ever fire a second time.)
            cooldown_remaining = (
                self.gesture_config.action_cooldown
                - (current_time - self._last_action_time)
            )
            if cooldown_remaining <= 0:
                self._state = GestureState.IDLE
                self._candidate_gesture = None
                self._release_confirmed = True
            else:
                self._state = GestureState.COOLDOWN

        self._hand_present = False

        cooldown_remaining = max(
            0.0, self.gesture_config.action_cooldown - (current_time - self._last_action_time)
        )

        return GestureResult(
            gesture=GestureType.NONE,
            confidence=0.0,
            state=self._state,
            hold_time=0.0,
            cooldown_remaining=cooldown_remaining,
        )

    def _analyze_fingers(
        self, landmarks: np.ndarray, handedness: str
    ) -> None:
        wrist = landmarks[0]

        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        finger_mcps = [2, 5, 9, 13, 17]

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]

        for i, name in enumerate(finger_names):
            tip = landmarks[finger_tips[i]]
            pip = landmarks[finger_pips[i]]
            mcp = landmarks[finger_mcps[i]]

            if name == "thumb":
                extended, conf = self._analyze_thumb(landmarks, handedness)
            else:
                extended, conf = self._analyze_finger(tip, pip, mcp, wrist)

            self._finger_states[name].extended = extended
            self._finger_states[name].confidence = conf

    def _analyze_thumb(
        self, landmarks: np.ndarray, handedness: str
    ) -> Tuple[bool, float]:
        tip = landmarks[4]
        ip = landmarks[3]
        mcp = landmarks[2]
        wrist = landmarks[0]

        thumb_vec = tip - ip
        palm_vec = mcp - wrist

        if handedness == "Right":
            cross_z = thumb_vec[0] * palm_vec[1] - thumb_vec[1] * palm_vec[0]
            extended = cross_z > 0
        else:
            cross_z = thumb_vec[0] * palm_vec[1] - thumb_vec[1] * palm_vec[0]
            extended = cross_z < 0

        tip_to_mcp = np.linalg.norm(tip[:2] - mcp[:2])
        ip_to_mcp = np.linalg.norm(ip[:2] - mcp[:2])
        confidence = min(1.0, tip_to_mcp / max(ip_to_mcp, 1e-6))

        return extended, confidence

    def _analyze_finger(
        self, tip: np.ndarray, pip: np.ndarray, mcp: np.ndarray, wrist: np.ndarray
    ) -> Tuple[bool, float]:
        tip_to_wrist = np.linalg.norm(tip[:2] - wrist[:2])
        pip_to_wrist = np.linalg.norm(pip[:2] - wrist[:2])
        mcp_to_wrist = np.linalg.norm(mcp[:2] - wrist[:2])

        extended = tip_to_wrist > pip_to_wrist > mcp_to_wrist
        confidence = min(1.0, (tip_to_wrist - pip_to_wrist) / max(pip_to_wrist - mcp_to_wrist, 1e-6))

        return extended, max(0.0, confidence)

    def _recognize_static_gesture(self) -> Tuple[GestureType, float]:
        thumb_ext = self._finger_states["thumb"].extended
        index_ext = self._finger_states["index"].extended
        middle_ext = self._finger_states["middle"].extended
        ring_ext = self._finger_states["ring"].extended
        pinky_ext = self._finger_states["pinky"].extended

        fingers_extended = [index_ext, middle_ext, ring_ext, pinky_ext]
        extended_count = sum(fingers_extended)

        thumb_conf = self._finger_states["thumb"].confidence
        finger_conf = np.mean([self._finger_states[n].confidence for n in ["index", "middle", "ring", "pinky"]])

        if extended_count == 4 and thumb_ext:
            return GestureType.OPEN_PALM, min(thumb_conf, finger_conf)

        if extended_count == 0 and not thumb_ext:
            return GestureType.FIST, 1.0 - finger_conf

        if thumb_ext and extended_count <= 1:
            thumb_tip_y = self._get_thumb_direction()
            if thumb_tip_y < -0.15:
                return GestureType.THUMB_UP, thumb_conf
            elif thumb_tip_y > 0.15:
                return GestureType.THUMB_DOWN, thumb_conf

        return GestureType.NONE, 0.0

    def _get_thumb_direction(self) -> float:
        landmarks = self._get_last_landmarks()
        if landmarks is None:
            return 0.0

        tip = landmarks[4]
        ip = landmarks[3]
        mcp = landmarks[2]

        thumb_vec = tip - ip
        return -thumb_vec[1]

    def _get_last_landmarks(self) -> Optional[np.ndarray]:
        return self._last_landmarks

    def _recognize_swipe(self, hand: HandLandmarks) -> Tuple[GestureType, float]:
        history = self._get_center_history()
        if len(history) < 3:
            return GestureType.NONE, 0.0

        current_time = time.time()
        window_start = current_time - self.gesture_config.swipe_window

        recent = [(x, y, t) for x, y, t in history if t >= window_start]
        if len(recent) < 3:
            return GestureType.NONE, 0.0

        if self._swipe_locked:
            # One motion = one swipe: stay locked until the hand calms down.
            xs = [p[0] for p in recent]
            peak = max((abs(b - a) for a, b in zip(xs, xs[1:])), default=0.0)
            if peak <= 15.0:
                self._swipe_locked = False
            else:
                return GestureType.NONE, 0.0

        x_start = recent[0][0]
        x_end = recent[-1][0]
        y_start = recent[0][1]
        y_end = recent[-1][1]
        t_start = recent[0][2]
        t_end = recent[-1][2]

        dx = x_end - x_start
        dy = y_end - y_start
        dt = t_end - t_start

        if dt < 0.1:
            return GestureType.NONE, 0.0

        velocity = abs(dx) / dt
        vertical_disp = abs(dy)

        if velocity < self.gesture_config.min_swipe_velocity:
            return GestureType.NONE, 0.0

        if vertical_disp > self.gesture_config.max_vertical_displacement:
            return GestureType.NONE, 0.0

        if abs(dx) < self.gesture_config.min_swipe_distance:
            return GestureType.NONE, 0.0

        # Direction consistency: most *moving* frame-to-frame x-steps must
        # share the swipe's sign, otherwise this is jitter/drift, not a
        # swipe. Zero steps (repeated centers from duplicate/stale frames --
        # the camera runs ~15 FPS while the callback runs faster) are
        # neutral, not votes against.
        xs = [p[0] for p in recent]
        steps = [b - a for a, b in zip(xs, xs[1:])]
        sign = -1.0 if dx < 0 else 1.0
        moving = [s for s in steps if abs(s) > 1e-6]
        if len(moving) < 2:
            return GestureType.NONE, 0.0
        consistent = sum(1 for s in moving if s * sign > 0)
        if consistent / len(moving) < 0.6:
            return GestureType.NONE, 0.0

        # Discontinuity veto: a big time gap between consecutive points
        # means the hand vanished and reappeared elsewhere (teleport), not
        # a swipe. Tolerates a couple of dropped frames at 15 FPS.
        times = [p[2] for p in recent]
        gaps = [t2 - t1 for t1, t2 in zip(times, times[1:])]
        if gaps and max(gaps) > 0.25:
            return GestureType.NONE, 0.0

        if dx < 0:
            confidence = min(1.0, abs(dx) / self.gesture_config.min_swipe_distance)
            return GestureType.SWIPE_LEFT, confidence
        else:
            confidence = min(1.0, abs(dx) / self.gesture_config.min_swipe_distance)
            return GestureType.SWIPE_RIGHT, confidence

    def _get_center_history(self) -> List[Tuple[float, float, float]]:
        return self._center_history

    def _update_state_machine(
        self,
        current_time: float,
        static_gesture: Tuple[GestureType, float],
        swipe_gesture: Tuple[GestureType, float],
        hand: HandLandmarks,
    ) -> GestureResult:
        static_type, static_conf = static_gesture
        swipe_type, swipe_conf = swipe_gesture

        self._center_history.append((hand.center[0], hand.center[1], current_time))
        if len(self._center_history) > 10:
            self._center_history.pop(0)

        self._last_landmarks = hand.landmarks

        cooldown_elapsed = current_time - self._last_action_time
        in_cooldown = cooldown_elapsed < self.gesture_config.action_cooldown

        if self._state == GestureState.ACTION_EXECUTED:
            # An action just fired -- the hold is over, enter cooldown so a
            # subsequent gesture can start fresh afterwards.
            self._state = GestureState.COOLDOWN
            self._candidate_gesture = None

        if swipe_type != GestureType.NONE and swipe_conf >= self.gesture_config.confidence_threshold:
            if not in_cooldown:
                return self._execute_action(current_time, swipe_type, swipe_conf)

        if static_type != GestureType.NONE and static_conf >= self.gesture_config.confidence_threshold:
            if static_type == self._last_gesture and self._needs_release:
                # Same gesture still held after firing: locked until the hand
                # disappears. This is the anti-repeat rule.
                pass
            else:
                if self._candidate_gesture != static_type:
                    # New (or first) gesture: start timing immediately from
                    # ANY state, so swipe -> palm flows without dropping the
                    # hand first. The cooldown gate below still applies.
                    self._candidate_gesture = static_type
                    self._candidate_start_time = current_time
                    self._candidate_start_center = hand.center
                    if self._state != GestureState.GESTURE_CANDIDATE:
                        self._state = (
                            GestureState.HAND_DETECTED
                            if self._state == GestureState.IDLE
                            else GestureState.GESTURE_CANDIDATE
                        )

                if self._state in (GestureState.HAND_DETECTED, GestureState.GESTURE_CANDIDATE):
                    moved = abs(hand.center[0] - self._candidate_start_center[0]) + abs(
                        hand.center[1] - self._candidate_start_center[1]
                    )
                    if moved > self.gesture_config.static_stillness_px:
                        # Hand is travelling, not holding -- restart the hold
                        # clock (a swipe in progress must not bank palm time).
                        self._candidate_start_time = current_time
                        self._candidate_start_center = hand.center
                        hold_duration = 0.0
                    else:
                        hold_duration = current_time - self._candidate_start_time
                    # Rolling classification memory, deliberately NOT reset
                    # when the candidate flips: a palm/fist/palm flicker must
                    # fail the stability gate below, while a genuinely held
                    # gesture dominates the window and passes.
                    self._recent_static.append(static_type)
                    if len(self._recent_static) > 15:
                        self._recent_static.pop(0)
                    if hold_duration >= self.gesture_config.hold_time:
                        stability = self._recent_static.count(static_type) / max(
                            1, len(self._recent_static)
                        )
                        if (
                            stability >= self.gesture_config.static_stability
                            and not in_cooldown
                        ):
                            return self._execute_action(current_time, static_type, static_conf)
                        # else: hold complete but classification flapping or
                        # cooling down -- stay a candidate, no fire.

                    return GestureResult(
                        gesture=static_type,
                        confidence=static_conf,
                        state=self._state,
                        hold_time=hold_duration,
                        cooldown_remaining=self.gesture_config.action_cooldown - cooldown_elapsed,
                    )

        if self._state == GestureState.COOLDOWN:
            if not in_cooldown:
                self._state = GestureState.WAIT_FOR_RELEASE

        if self._state == GestureState.WAIT_FOR_RELEASE:
            if self._release_confirmed:
                self._state = GestureState.IDLE
                self._candidate_gesture = None

        return GestureResult(
            gesture=GestureType.NONE,
            confidence=0.0,
            state=self._state,
            hold_time=0.0,
            cooldown_remaining=max(0.0, self.gesture_config.action_cooldown - cooldown_elapsed),
        )

    def _execute_action(
        self, current_time: float, gesture: GestureType, confidence: float
    ) -> GestureResult:
        self._last_action_time = current_time
        self._last_gesture = gesture
        self._state = GestureState.ACTION_EXECUTED
        self._candidate_gesture = None
        self._needs_release = True
        if gesture in (GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT):
            # Forget the motion that just fired so the same trajectory
            # cannot retrigger (or reverse-fire) on subsequent frames,
            # and lock until the hand calms down: one motion, one swipe.
            self._center_history = []
            self._swipe_locked = True

        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            state=GestureState.ACTION_EXECUTED,
            hold_time=0.0,
            cooldown_remaining=self.gesture_config.action_cooldown,
        )

    def get_state(self) -> GestureState:
        return self._state

    def get_finger_states(self) -> dict:
        return self._finger_states.copy()

    def reset(self) -> None:
        self._state = GestureState.IDLE
        self._candidate_gesture = None
        self._recent_static = []
        self._last_action_time = 0.0
        self._last_gesture = None
        self._hand_present = False
        self._release_confirmed = False
        self._needs_release = False
        self._swipe_locked = False
        for state in self._finger_states.values():
            state.extended = False
            state.confidence = 0.0