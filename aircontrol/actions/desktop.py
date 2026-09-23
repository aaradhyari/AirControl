import logging
import subprocess
import time
from typing import Tuple

from aircontrol.config import CONFIG

logger = logging.getLogger(__name__)


def switch_space(direction: str) -> bool:
    shortcut = _get_shortcut(direction)
    if not shortcut:
        return False

    # Primary path: System Events. Raw CGEvents posted to the HID tap are
    # delivered to apps but IGNORED by the Dock's space-switch hotkey
    # interceptor on recent macOS (it filters non-hardware event sources),
    # so Quartz injection alone cannot switch Spaces. System Events
    # synthesis goes through the honored path.
    if _send_system_events(shortcut):
        return True
    logger.warning("System Events failed, trying Quartz")
    return _send_key_combination(shortcut)


def _get_shortcut(direction: str) -> Tuple[str, ...]:
    if direction == "left":
        return CONFIG.shortcuts.left_space
    elif direction == "right":
        return CONFIG.shortcuts.right_space
    return ()


def _send_key_combination(keys: Tuple[str, ...]) -> bool:
    try:
        key_map = {
            "left": 123,
            "right": 124,
            "ctrl": 0x3B,
            "cmd": 0x37,
            "alt": 0x3A,
            "shift": 0x38,
        }

        key_codes = []
        flags = 0

        from Quartz import (
            CGEventCreateKeyboardEvent,
            CGEventPost,
            kCGHIDEventTap,
            CGEventSetFlags,
            kCGEventFlagMaskControl,
            kCGEventFlagMaskCommand,
            kCGEventFlagMaskAlternate,
            kCGEventFlagMaskShift,
        )

        flag_map = {
            "ctrl": kCGEventFlagMaskControl,
            "cmd": kCGEventFlagMaskCommand,
            "alt": kCGEventFlagMaskAlternate,
            "shift": kCGEventFlagMaskShift,
        }

        for key in keys:
            if key in flag_map:
                flags |= flag_map[key]
            elif key in key_map:
                key_codes.append(key_map[key])

        if not key_codes:
            return False

        for key_code in key_codes:
            event_down = CGEventCreateKeyboardEvent(None, key_code, True)
            event_up = CGEventCreateKeyboardEvent(None, key_code, False)

            if flags:
                CGEventSetFlags(event_down, flags)
                CGEventSetFlags(event_up, flags)

            # Post down and up as separate presses with a small hold,
            # like a physical key -- instantaneous down+up pairs can be
            # missed by the shortcut recognizer.
            CGEventPost(kCGHIDEventTap, event_down)
            time.sleep(0.03)
            CGEventPost(kCGHIDEventTap, event_up)
            time.sleep(0.02)

        return True

    except Exception as e:
        logger.warning(f"Quartz key event failed, trying osascript: {e}")
        return _fallback_osascript(keys)


def _send_system_events(keys: Tuple[str, ...]) -> bool:
    """Switch Spaces via System Events -- the automation path macOS honors
    for the Mission Control shortcut. Requires Accessibility permission;
    on first run macOS prompts, and the call blocks until it is answered,
    hence the generous timeout (the gesture thread simply waits)."""
    try:
        key_map = {
            "left": 123,
            "right": 124,
            "f": 3,
        }

        modifier_map = {
            "ctrl": "control down",
            "cmd": "command down",
            "alt": "option down",
            "shift": "shift down",
        }

        modifiers = []
        key_code = None

        for key in keys:
            if key in modifier_map:
                modifiers.append(modifier_map[key])
            elif key in key_map:
                key_code = key_map[key]

        if key_code is None:
            return False

        modifier_str = " using {" + ", ".join(modifiers) + "}" if modifiers else ""

        script = f'tell application "System Events" to key code {key_code}{modifier_str}'

        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=8.0,
        )
        if result.returncode != 0:
            logger.warning(f"System Events failed: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(
            "System Events timed out -- grant Accessibility permission to "
            "your terminal (System Settings > Privacy & Security > "
            "Accessibility), then retry."
        )
        return False
    except Exception as e:
        logger.error(f"System Events error: {e}")
        return False


def _fallback_osascript(keys: Tuple[str, ...]) -> bool:
    try:
        key_map = {
            "left": "left arrow",
            "right": "right arrow",
        }

        modifier_map = {
            "ctrl": "control down",
            "cmd": "command down",
            "alt": "option down",
            "shift": "shift down",
        }

        modifiers = []
        key_code = None

        for key in keys:
            if key in modifier_map:
                modifiers.append(modifier_map[key])
            elif key in key_map:
                key_code = key_map[key]

        if not key_code:
            return False

        modifier_str = " using {" + ", ".join(modifiers) + "}" if modifiers else ""

        script = f"""
        tell application "System Events"
            key code {_get_key_code(key_code)}{modifier_str}
        end tell
        """

        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"osascript fallback failed: {e}")
        return False


def _get_key_code(key_name: str) -> int:
    codes = {
        "left arrow": 123,
        "right arrow": 124,
    }
    return codes.get(key_name, 123)