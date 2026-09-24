import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def list_targets() -> List[Dict]:
    """Front-to-back ordered switch candidates from the on-screen window
    list. Each target is {"pid": int, "name": str}. Visual stacking order
    approximates recent use, like the system switcher."""
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )

        raw = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
    except Exception as e:
        logger.error(f"Window list failed: {e}")
        return []

    targets: List[Dict] = []
    seen_pids = set()
    try:
        from AppKit import NSRunningApplication, NSApplicationActivationPolicyRegular
        regular_policy = NSApplicationActivationPolicyRegular
    except Exception:
        NSRunningApplication = None
        regular_policy = 0
    try:
        for info in raw or []:
            pid = info.get("kCGWindowOwnerPID")
            name = info.get("kCGWindowOwnerName", "")
            if not pid or pid in seen_pids or not name:
                continue
            if NSRunningApplication is not None:
                running = NSRunningApplication.runningApplicationWithProcessIdentifier_(
                    int(pid)
                )
                if running is None or running.activationPolicy() != regular_policy:
                    continue
            seen_pids.add(pid)
            targets.append({"pid": int(pid), "name": str(name)})
    except Exception as e:
        logger.error(f"Window list parse failed: {e}")
        return []
    return targets


def frontmost_pid() -> Optional[int]:
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else None
    except Exception as e:
        logger.error(f"Frontmost app query failed: {e}")
        return None


def activate_pid(pid: int) -> bool:
    try:
        from AppKit import NSWorkspace, NSApplicationActivateIgnoringOtherApps
        from Foundation import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is None:
            return False
        return bool(
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        )
    except Exception as e:
        logger.error(f"Activate pid {pid} failed: {e}")
        return False


def step(direction: str) -> Optional[int]:
    """Activate the next/previous app in visual order. Returns the newly
    selected index, or None if there is nothing to switch to."""
    targets = list_targets()
    if len(targets) < 2:
        return None
    current = frontmost_pid()
    try:
        current_index = next(
            i for i, t in enumerate(targets) if t["pid"] == current
        )
    except StopIteration:
        current_index = 0 if direction == "next" else len(targets) - 1
        if activate_pid(targets[current_index]["pid"]):
            return current_index
        return None

    delta = 1 if direction == "next" else -1
    new_index = (current_index + delta) % len(targets)
    if activate_pid(targets[new_index]["pid"]):
        return new_index
    return None
