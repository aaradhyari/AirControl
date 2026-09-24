"""Synthetic two-hand verification for the app-switch feature.

Drives GestureRecognizer.process_frame with crafted landmark sets through
the REAL classification pipeline (no monkeypatching of gesture logic).
Covers: lone fist paths, modifier hold, left/right swipes, release, grace,
handedness-by-label (not x), and legacy backward compatibility.
"""
import sys
import time

sys.path.insert(0, "/Users/aaradhya-rai/GestureControl")

import numpy as np
from aircontrol.config import Config
from aircontrol.vision.hand_tracker import HandLandmarks
from aircontrol.vision.gesture_recognizer import (
    GestureRecognizer,
    GestureType,
    GestureState,
    AppSwitchState,
)

PASS = []
FAIL = []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name, extra)


def base_landmarks():
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.5, 0.8, 0)
    return lm


def fist_landmarks():
    lm = base_landmarks()
    lm[2] = (0.44, 0.70, 0)
    lm[3] = (0.45, 0.65, 0)
    lm[4] = (0.45, 0.65, 0)  # curled thumb: zero vector -> not extended
    for i, x in enumerate([0.44, 0.50, 0.56, 0.62]):
        m, p, t = 5 + 4 * i, 6 + 4 * i, 8 + 4 * i
        lm[m] = (x, 0.52, 0)
        lm[p] = (x, 0.58, 0)
        lm[t] = (x, 0.72, 0)  # tip nearer wrist than pip -> folded, conf 0
    return lm


def palm_landmarks():
    lm = base_landmarks()
    lm[2] = (0.42, 0.70, 0)
    lm[3] = (0.34, 0.62, 0)
    lm[4] = (0.26, 0.55, 0)
    for i, x in enumerate([0.46, 0.52, 0.58, 0.64]):
        m, p, t = 5 + 4 * i, 6 + 4 * i, 8 + 4 * i
        lm[m] = (x, 0.55, 0)
        lm[p] = (x, 0.40, 0)
        lm[t] = (x, 0.20, 0)
    return lm


def hand(lm_fn, label, x=320.0, score=0.95):
    w = 60.0
    return HandLandmarks(
        landmarks=lm_fn(),
        handedness=label,
        score=score,
        center=(x, 240.0),
        bounding_box=(int(x - w / 2), 200, int(x + w / 2), 280),
    )


def make_rec():
    cfg = Config()
    cfg.gesture.hold_time = 0.1
    cfg.gesture.action_cooldown = 0.15
    cfg.gesture.confidence_threshold = 0.5
    cfg.gesture.app_switch_hold_time = 0.1
    cfg.gesture.app_switch_release_grace = 0.12
    return GestureRecognizer(config=cfg)


def feed(rec, hands, n=1, dt=0.025):
    """Returns (legacy_fires, app_events)."""
    fires, events = [], []
    for _ in range(n):
        fr = rec.process_frame(hands)
        time.sleep(dt)
        if fr.legacy.state == GestureState.ACTION_EXECUTED:
            fires.append(fr.legacy.gesture)
        if fr.app_event is not None:
            events.append(fr.app_event)
    return fires, events


# --- 0. recipes classify as intended -------------------------------------
rec = make_rec()
rec.process_frame([hand(fist_landmarks, "Left")])
check("left fist classified FIST",
      rec._hands["Left"].last_static == GestureType.FIST,
      str(rec._hands["Left"].last_static))
rec = make_rec()
rec.process_frame([hand(palm_landmarks, "Right")])
check("right palm classified OPEN_PALM",
      rec._hands["Right"].last_static == GestureType.OPEN_PALM)

# --- 1. lone RIGHT fist -> legacy fullscreen path, no app-switch ---------
rec = make_rec()
fires, events = feed(rec, [hand(fist_landmarks, "Right")], n=12)
check("right fist fires legacy FIST once", fires == [GestureType.FIST], str(fires))
check("right fist: no app-switch events", events == [])
check("right fist: mode stays normal", rec.get_debug_snapshot()["mode"] == "normal")

# --- 2. lone LEFT fist -> ACTIVE, no legacy fullscreen, release -> exit + consummate
rec = make_rec()
fires, events = feed(rec, [hand(fist_landmarks, "Left")], n=12)
check("left fist: no legacy fullscreen while held", GestureType.FIST not in fires, str(fires))
check("left fist: enter event", [e.kind for e in events] == ["enter"],
      str([e.kind for e in events]))
check("left fist: mode app_switch",
      rec.get_debug_snapshot()["mode"] == "app_switch")
fires2, events2 = feed(rec, [], n=10)
check("release: exit event",
      [e.kind for e in events2] == ["exit"], str([e.kind for e in events2]))
check("release w/o swipes: consummate fullscreen",
      events2 and events2[0].consummate_fullscreen is True)
check("after release: mode normal",
      rec.get_debug_snapshot()["mode"] == "normal")

# --- 3. left fist + right swipe left -> exactly one prev ------------------
rec = make_rec()
feed(rec, [hand(fist_landmarks, "Left")], n=10)  # arm + activate
prev_events = []
t = time.time()
for i in range(12):  # right hand sweeps 420 -> 150
    x = 420 - i * 24
    fr = rec.process_frame([hand(fist_landmarks, "Left", x=200.0),
                            hand(palm_landmarks, "Right", x=x)])
    time.sleep(0.025)
    if fr.app_event is not None:
        prev_events.append(fr.app_event.kind)
    assert all(g != GestureType.FIST or True for g in []), ""
check("right swipe left -> one prev", prev_events == ["prev"], str(prev_events))

# --- 4. continued motion must not refire; stop, swipe right -> one next --
more = []
for i in range(10):  # keep moving left (same motion continues)
    x = 150 - i * 10
    fr = rec.process_frame([hand(fist_landmarks, "Left", x=200.0),
                            hand(palm_landmarks, "Right", x=x)])
    time.sleep(0.025)
    if fr.app_event is not None:
        more.append(fr.app_event.kind)
check("continued motion: no refire", more == [], str(more))
calm = []
for i in range(8):  # hand calms (unlock), then sweeps right
    x = 60 + i * 4
    fr = rec.process_frame([hand(fist_landmarks, "Left", x=200.0),
                            hand(palm_landmarks, "Right", x=x)])
    time.sleep(0.025)
    if fr.app_event is not None:
        calm.append(fr.app_event.kind)
for i in range(12):
    x = 100 + i * 26
    fr = rec.process_frame([hand(fist_landmarks, "Left", x=200.0),
                            hand(palm_landmarks, "Right", x=x)])
    time.sleep(0.025)
    if fr.app_event is not None:
        calm.append(fr.app_event.kind)
check("second distinct swipe -> one next", calm == ["next"], str(calm))

# --- 5. release after swipes -> exit WITHOUT consummate ------------------
fires5, events5 = feed(rec, [], n=10)
check("release after swipes: exit, no consummate",
      len(events5) == 1 and events5[0].kind == "exit"
      and events5[0].consummate_fullscreen is False,
      str([(e.kind, e.consummate_fullscreen) for e in events5]))

# --- 6. right swipe with NO left fist -> legacy space swipe, no app event
rec = make_rec()
got = []
for i in range(12):
    x = 420 - i * 24
    fr = rec.process_frame([hand(palm_landmarks, "Right", x=x)])
    time.sleep(0.025)
    if fr.legacy.state == GestureState.ACTION_EXECUTED:
        got.append(("legacy", fr.legacy.gesture.value))
    if fr.app_event is not None:
        got.append(("app", fr.app_event.kind))
check("lone right swipe -> legacy SWIPE_LEFT only",
      ("legacy", "swipe_left") in got and not any(k == "app" for k, _ in got), str(got))

# --- 7. grace: brief left dropout keeps mode; long dropout exits ----------
rec = make_rec()
feed(rec, [hand(fist_landmarks, "Left")], n=10)
feed(rec, [], n=3)  # ~0.075s < 0.12 grace
check("brief dropout: still app_switch",
      rec.get_debug_snapshot()["mode"] == "app_switch")
feed(rec, [], n=8)  # total > grace
check("long dropout: exits to normal",
      rec.get_debug_snapshot()["mode"] == "normal")

# --- 8. handedness by LABEL, not x: left-labeled fist on screen right ----
rec = make_rec()
fires, events = feed(rec, [hand(fist_landmarks, "Left", x=520.0)], n=10)
check("label rules: right-side LEFT fist still arms modifier",
      [e.kind for e in events] == ["enter"] and GestureType.FIST not in fires,
      str([e.kind for e in events]))

# --- 9. legacy single right palm still fires (backward compat) ------------
rec = make_rec()
fires, events = feed(rec, [hand(palm_landmarks, "Right")], n=12)
check("legacy right palm fires once", fires == [GestureType.OPEN_PALM], str(fires))

print()
print(f"PASSED {len(PASS)}  FAILED {len(FAIL)}")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
print("ALL TWO-HAND TESTS PASSED")
