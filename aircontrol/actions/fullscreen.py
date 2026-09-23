import logging
import subprocess
import time
from typing import Tuple

from aircontrol.config import CONFIG

logger = logging.getLogger(__name__)


def toggle_fullscreen() -> bool:
    shortcut = CONFIG.shortcuts.fullscreen
    return _send_key_combination(shortcut)


def _send_key_combination(keys: Tuple[str, ...]) -> bool:
    try:
        key_map = {
            "f": 3,
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


def _fallback_osascript(keys: Tuple[str, ...]) -> bool:
    try:
        key_map = {
            "f": "f",
        }

        modifier_map = {
            "ctrl": "control down",
            "cmd": "command down",
            "alt": "option down",
            "shift": "shift down",
        }

        modifiers = []
        key_char = None

        for key in keys:
            if key in modifier_map:
                modifiers.append(modifier_map[key])
            elif key in key_map:
                key_char = key_map[key]

        if not key_char:
            return False

        modifier_str = " using {" + ", ".join(modifiers) + "}" if modifiers else ""

        script = f"""
        tell application "System Events"
            keystroke "{key_char}"{modifier_str}
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