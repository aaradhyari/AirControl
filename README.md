# AirControl

<img src="assets/aircontrol-logo.png" alt="AirControl logo" width="360" />

A native macOS menu-bar app that watches your MacBook's built-in camera, recognizes hand gestures locally, and drives system functions — media, Spaces, fullscreen, volume. No terminal window, no cloud, no recording.

## Gestures

| Gesture | Action |
|---------|--------|
| ✋ Open palm (hold ~0.5s) | Play / Pause media |
| 👈 Swipe left | Previous Space |
| 👉 Swipe right | Next Space |
| ✊ Fist (hold ~0.5s) | Toggle fullscreen |
| 👍 Thumb up (hold ~0.5s) | Volume up (+5%) |
| 👎 Thumb down (hold ~0.5s) | Volume down (−5%) |

One presentation = one action. Holding a gesture never repeats it; a different gesture works immediately without dropping your hand first.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.11–3.13
- Camera permission (for your terminal, or for `AirControl.app` once packaged)
- Accessibility permission (for Space switching and simulated keys)

## Quick start

```bash
git clone <your-repo-url> AirControl
cd AirControl

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MediaPipe hand model (~8 MB, cached, downloaded once)
mkdir -p models
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

python -m aircontrol.main
```

> **Note:** `mediapipe` is pinned to `==0.10.35` on purpose — 1.0.x aborts on Apple Silicon inside `DrishtiMetalHelper` at landmarker creation. Do not relax this without testing on-device.

Grant permissions when macOS asks (or pre-grant in System Settings → Privacy & Security → **Camera** and **Accessibility**). The app warns at startup and shows `Accessibility: Required` in its menu if the grant is missing.

## Running

```bash
python -m aircontrol.main           # menu-bar app (stays in foreground of this shell)
python -m aircontrol.main --debug   # verbose per-frame state
python -m aircontrol.main --test    # recognition runs, no macOS actions fire
python test_gesture.py                   # calibration console: hand / fingers / gesture / FPS / state
python test_gesture.py --video           # same, plus landmark overlay window
```

Menu-bar icon: **●** enabled · **○** disabled · **⚠** camera unavailable. The menu lists every gesture mapping plus Camera / Status / Accessibility rows, and shows a brief overlay each time a gesture fires.

## How it works

**Pipeline:** `camera` (threaded OpenCV capture, 640×480) → `vision/hand_tracker.py` (MediaPipe Tasks `HandLandmarker`, CPU delegate, 21 landmarks) → `vision/gesture_recognizer.py` (temporal state machine) → `actions/*` → native macOS APIs.

**Static gestures** (palm, fist, thumbs) require: hold ≥ `hold_time`, hand still within `static_stillness_px`, and classification stable across ≥ `static_stability` of recent frames. Finger state comes from landmark geometry (tip vs PIP vs MCP distances, cross-product thumb test), never pixel positions.

**Swipes** use hand-center trajectory over `swipe_window`: ≥ `min_swipe_distance` px horizontal, ≥ `min_swipe_velocity` px/s, ≤ `max_vertical_displacement` px vertical, ≥60% of moving steps in one direction, no time-gap teleports, one fire per motion (locked until the hand calms).

**Anti-repeat:** `IDLE → HAND_DETECTED → GESTURE_CANDIDATE → ACTION_EXECUTED → COOLDOWN → WAIT_FOR_RELEASE → IDLE`, enforced per gesture — same held gesture is locked until the hand disappears; `action_cooldown` (1.0s) rate-limits everything.

**macOS actions:**

| Action | Mechanism |
|--------|-----------|
| Play/Pause | `NX_KEYTYPE_PLAY` system key event (works with Music, Spotify, browser YouTube) |
| Spaces | System Events `key code 123/124 using {control down}` — raw HID injection is ignored by the Dock's hotkey interceptor, so this path is required |
| Fullscreen | `⌘ + Ctrl + F` via Quartz `CGEvent` |
| Volume | `osascript` output-volume ± `volume_step` |

## Configuration

Everything lives in `aircontrol/config.py` (dataclasses, no other files to touch):

```python
CameraConfig:   index=0, width=640, height=480, target_fps=30
GestureConfig:  hold_time=0.45, action_cooldown=1.0, swipe_window=0.5,
                min_swipe_distance=120, min_swipe_velocity=300.0,
                max_vertical_displacement=80, static_stillness_px=40.0,
                static_stability=0.7, volume_step=5,
                confidence_threshold=0.7, smoothing_frames=3
ShortcutConfig: left_space=("ctrl","left"), right_space=("ctrl","right"),
                fullscreen=("cmd","ctrl","f")
AppConfig:      enable_visual_feedback=True, feedback_duration=0.8
```

If your Mission Control shortcuts differ from `Ctrl+←/→`, update `left_space`/`right_space` to match.

## Project layout

```
aircontrol/
├── main.py                 # entry point, permission checks, orchestration
├── config.py               # all tunables (see above)
├── camera/camera.py        # threaded capture, callbacks, clean teardown
├── vision/
│   ├── hand_tracker.py     # MediaPipe landmarker wrapper + histories
│   └── gesture_recognizer.py  # fingers, swipes, state machine, debounce
├── actions/
│   ├── media.py            # NX_KEYTYPE_PLAY system event
│   ├── desktop.py          # Spaces via System Events (Quartz fallback)
│   ├── fullscreen.py       # ⌘CtrlF via Quartz
│   └── volume.py           # system volume via osascript
└── menubar/app.py          # native AppKit menu bar, status, feedback
test_gesture.py             # safe calibration mode (no side effects)
models/                    # hand_landmarker.task (downloaded, git-ignored)
```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `not authorized to capture video` | Grant Camera to your terminal; rerun |
| Gestures detect, spaces don't switch | Grant Accessibility, restart; confirm physical `Ctrl+←/→` switches Spaces and `Move left/right a space` is ticked under Keyboard Shortcuts → Mission Control |
| Palms/fists alternate rapidly | Fixed by the stability gate — if you still see it, raise `static_stability` or `hold_time` |
| Swipes never fire | Make them brisk: ~⅓ of the frame in <0.5s; slow drifts are rejected by design |
| `DrishtiMetalHelper` abort at startup | You installed mediapipe 1.x — `pip install "mediapipe==0.10.35"` |
| High CPU | Lower `target_fps` / resolution; processing idles when no hand is visible |

## Packaging as AirControl.app

`pip install pyinstaller`, then e.g.:

```bash
pyinstaller --noconfirm --clean --windowed \
  --name "AirControl" \
  --icon=assets/AirControl.icns \
  --collect-all mediapipe \
  --add-data "models/hand_landmarker.task:models" \
  --osx-bundle-identifier=com.aircontrol.app \
  aircontrol/main.py
```

`--collect-all mediapipe` is required — without it the bundle crashes at
startup with `No module named 'mediapipe.tasks.c'`. Then stamp the bundle:

```bash
PLIST="dist/AirControl.app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :NSCameraUsageDescription string 'AirControl needs camera access to recognize hand gestures locally on your Mac.'" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string 0.1.0" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleVersion string 0.1.0" "$PLIST"
codesign --force --deep -s - dist/AirControl.app
```

`LSUIElement` makes it menu-bar-only (no Dock icon / Terminal). On first launch, grant Camera + Accessibility to `AirControl.app` itself.

## Releases

Versions live in `aircontrol/__init__.py` (`__version__`) and the bundle's `Info.plist`. To cut a release:

```bash
# 1. bump __version__, commit
# 2. tag and push
git tag -a v0.1.0 -m "AirControl 0.1.0"
git push origin v0.1.0
# 3. zip the app and attach it to a GitHub Release for the tag
ditto -c -k --sequesterRsrc dist/AirControl.app AirControl-0.1.0-mac-arm64.zip
```

## Privacy

All frames are processed in-memory on your Mac and discarded — nothing is saved, recorded, or uploaded. (MediaPipe's failed `clearcut` telemetry lines in the log are blocked analytics pings, not image data.)

## License

MIT — see [LICENSE](LICENSE).
