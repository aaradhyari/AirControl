import logging
import threading
from typing import Optional, List, Dict, Any

import objc
from AppKit import (
    NSObject,
    NSPanel,
    NSView,
    NSTextField,
    NSImageView,
    NSScreen,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
    NSBackingStoreBuffered,
    NSFloatingWindowLevel,
    NSFont,
    NSColor,
    NSTimer,
    NSMakeRect,
    NSWorkspace,
    NSRunningApplication,
    NSImage,
)
from Foundation import NSRunLoop, NSDefaultRunLoopMode

logger = logging.getLogger(__name__)


class SwitcherModel:
    """Thread-safe overlay state. Mutated from the camera thread, rendered
    by the overlay's main-thread timer -- AppKit is never touched off the
    main thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {"visible": False, "apps": [], "selected": 0}

    def update(
        self,
        visible: bool,
        apps: Optional[List[Dict]] = None,
        selected: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._state["visible"] = visible
            if apps is not None:
                self._state["apps"] = list(apps)
            if selected is not None:
                self._state["selected"] = selected

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = dict(self._state)
            snap["apps"] = list(snap["apps"])
            return snap


class AppSwitcherOverlay(NSObject):
    ROW_HEIGHT = 46.0
    WIDTH = 430.0
    MAX_ROWS = 8

    def initWithModel_(self, model: SwitcherModel):
        self = objc.super(AppSwitcherOverlay, self).init()
        if self is None:
            return None
        self._model = model
        self._panel: Optional[NSPanel] = None
        self._rows: List[NSView] = []
        self._row_pids: List[int] = []
        self._selected: int = -1
        self._timer: Optional[NSTimer] = None
        self._content: Optional[NSView] = None
        return self

    def start(self) -> None:
        try:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.1, self, "sync:", None, True
            )
            NSRunLoop.currentRunLoop().addTimer_forMode_(self._timer, NSDefaultRunLoopMode)
        except Exception as e:
            logger.error(f"Switcher overlay timer failed: {e}")

    def close(self) -> None:
        try:
            if self._timer is not None:
                self._timer.invalidate()
                self._timer = None
            if self._panel is not None:
                self._panel.orderOut_(None)
                self._panel = None
        except Exception as e:
            logger.error(f"Switcher overlay close failed: {e}")

    def sync_(self, timer) -> None:
        try:
            snap = self._model.snapshot()
            if not snap["visible"]:
                if self._panel is not None:
                    self._panel.orderOut_(None)
                    self._panel = None
                    self._rows = []
                    self._row_pids = []
                    self._selected = -1
                return
            apps = snap["apps"][: self.MAX_ROWS]
            pids = [a["pid"] for a in apps]
            if self._panel is None:
                self._build_(apps)
            elif pids != self._row_pids:
                self._build_(apps)
            self._highlight_(snap["selected"])
            if self._panel is not None:
                self._panel.orderFrontRegardless()
        except Exception as e:
            logger.error(f"Switcher overlay sync failed: {e}")

    def _build_(self, apps: List[Dict]) -> None:
        if self._panel is not None:
            self._panel.orderOut_(None)
        n = max(1, len(apps))
        height = n * self.ROW_HEIGHT + 64.0
        screen = NSScreen.mainScreen()
        frame = screen.frame() if screen is not None else NSMakeRect(0, 0, 1440, 900)
        x = frame.origin.x + (frame.size.width - self.WIDTH) / 2.0
        y = frame.origin.y + (frame.size.height - height) / 2.0 + 60.0

        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, self.WIDTH, height),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        self._panel.setLevel_(NSFloatingWindowLevel)
        self._panel.setIgnoresMouseEvents_(True)
        self._panel.setOpaque_(False)
        self._panel.setHasShadow_(True)
        self._panel.setBackgroundColor_(NSColor.clearColor())

        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, self.WIDTH, height))
        content.setWantsLayer_(True)
        layer = content.layer()
        layer.setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.92).CGColor()
        )
        layer.setCornerRadius_(16.0)
        self._panel.setContentView_(content)
        self._content = content

        header = NSTextField.alloc().initWithFrame_(
            NSMakeRect(20, height - 44, self.WIDTH - 40, 24)
        )
        header.setStringValue_("APP SWITCH  ·  hold ✊  ·  swipe 👉👈  ·  release to exit")
        header.setBezeled_(False)
        header.setDrawsBackground_(False)
        header.setEditable_(False)
        header.setSelectable_(False)
        header.setTextColor_(NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.55))
        header.setFont_(NSFont.systemFontOfSize_(12))
        content.addSubview_(header)

        self._rows = []
        self._row_pids = []
        for i, app_info in enumerate(apps):
            row_y = height - 64 - (i + 1) * self.ROW_HEIGHT
            row = NSView.alloc().initWithFrame_(
                NSMakeRect(12, row_y, self.WIDTH - 24, self.ROW_HEIGHT - 6)
            )
            row.setWantsLayer_(True)
            row.layer().setCornerRadius_(10.0)
            content.addSubview_(row)

            icon_view = NSImageView.alloc().initWithFrame_(NSMakeRect(8, 5, 30, 30))
            icon_view.setImage_(self._iconForPid_(app_info["pid"]))
            icon_view.setImageScaling_(2)  # NSImageScaleProportionallyUpOrDown
            row.addSubview_(icon_view)

            label = NSTextField.alloc().initWithFrame_(
                NSMakeRect(48, 4, self.WIDTH - 24 - 60, 32)
            )
            label.setStringValue_(str(app_info.get("name", "?")))
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setTextColor_(NSColor.whiteColor())
            label.setFont_(NSFont.systemFontOfSize_(15))
            row.addSubview_(label)

            self._rows.append(row)
            self._row_pids.append(app_info["pid"])
        self._selected = -1

    def _highlight_(self, selected: int) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        for i, row in enumerate(self._rows):
            if i == selected:
                row.layer().setBackgroundColor_(
                    NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.20).CGColor()
                )
            else:
                row.layer().setBackgroundColor_(NSColor.clearColor().CGColor())

    def _iconForPid_(self, pid: int) -> NSImage:
        try:
            running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if running is not None and running.icon() is not None:
                return running.icon()
        except Exception:
            pass
        try:
            return NSImage.imageNamed_("NSApplicationIcon")
        except Exception:
            return NSImage.alloc().init()
