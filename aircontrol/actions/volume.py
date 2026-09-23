import logging
import subprocess
from typing import Tuple

from aircontrol.config import CONFIG

logger = logging.getLogger(__name__)


def set_volume(level: int) -> bool:
    level = max(0, min(100, level))
    try:
        script = f"set volume output volume {level}"
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Set volume failed: {e}")
        return False


def get_volume() -> int:
    try:
        script = "output volume of (get volume settings)"
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as e:
        logger.error(f"Get volume failed: {e}")
    return 50


def volume_up(step: int = None) -> bool:
    if step is None:
        step = CONFIG.gesture.volume_step
    current = get_volume()
    return set_volume(current + step)


def volume_down(step: int = None) -> bool:
    if step is None:
        step = CONFIG.gesture.volume_step
    current = get_volume()
    return set_volume(current - step)


def mute() -> bool:
    try:
        script = "set volume output muted true"
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Mute failed: {e}")
        return False


def unmute() -> bool:
    try:
        script = "set volume output muted false"
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Unmute failed: {e}")
        return False