import logging
import threading
import time
from typing import Optional, Callable
from enum import Enum

import objc
from AppKit import (
    NSApplication,
    NSStatusBar,
    NSStatusItem,
    NSMenu,
    NSMenuItem,
    NSImage,
    NSBundle,
    NSObject,
    NSRunLoop,
    NSDefaultRunLoopMode,
    NSTimer,
    NSAttributedString,
    NSFont,
    NSColor,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSOnState,
    NSOffState,
    NSMixedState,
)
from Foundation import NSAutoreleasePool
from PyObjCTools import AppHelper

from gesture_control.config import CONFIG
from gesture_control.vision.gesture_recognizer import GestureType

logger = logging.getLogger(__name__)


class MenuBarIcon(Enum):
    ENABLED = "●"
    DISABLED = "○"
    WARNING = "⚠"


class GestureMenuBarApp(NSObject):
    def init(self):
        self = objc.super(GestureMenuBarApp, self).init()
        if self is None:
            return None

        self._enabled = False
        self._camera_available = True
        self._status_text = "Inactive"
        self._gesture_feedback = ""
        self._feedback_timer = None
        self._update_timer = None

        self._status_item: Optional[NSStatusItem] = None
        self._menu: Optional[NSMenu] = None
        self._enabled_item: Optional[NSMenuItem] = None
        self._camera_item: Optional[NSMenuItem] = None
        self._status_item_menu: Optional[NSMenuItem] = None
        self._access_item: Optional[NSMenuItem] = None
        self._feedback_item: Optional[NSMenuItem] = None

        self._action_callback: Optional[Callable] = None
        self._quit_callback: Optional[Callable] = None
        self._toggle_callback: Optional[Callable] = None

        self._setup_menu_bar()
        return self

    def _setup_menu_bar(self) -> None:
        status_bar = NSStatusBar.systemStatusBar()
        self._status_item = status_bar.statusItemWithLength_(-1)

        self._menu = NSMenu.alloc().init()

        title_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "✋ Gesture Control", None, ""
        )
        title_item.setEnabled_(False)
        self._menu.addItem_(title_item)

        separator = NSMenuItem.separatorItem()
        self._menu.addItem_(separator)

        self._enabled_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "● Enabled", "toggleEnabled:", "e"
        )
        self._enabled_item.setTarget_(self)
        self._menu.addItem_(self._enabled_item)

        separator2 = NSMenuItem.separatorItem()
        self._menu.addItem_(separator2)

        gestures = [
            ("✋ Play / Pause", None, ""),
            ("👈 Previous Desktop", None, ""),
            ("👉 Next Desktop", None, ""),
            ("✊ Fullscreen", None, ""),
            ("👍 Volume Up", None, ""),
            ("👎 Volume Down", None, ""),
        ]

        for title, action, key in gestures:
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
            item.setEnabled_(False)
            self._menu.addItem_(item)

        separator3 = NSMenuItem.separatorItem()
        self._menu.addItem_(separator3)

        self._camera_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Camera: Built-in", None, ""
        )
        self._camera_item.setEnabled_(False)
        self._menu.addItem_(self._camera_item)

        self._status_item_menu = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Status: ● Active", None, ""
        )
        self._status_item_menu.setEnabled_(False)
        self._menu.addItem_(self._status_item_menu)

        self._access_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Accessibility: Granted", None, ""
        )
        self._access_item.setEnabled_(False)
        self._menu.addItem_(self._access_item)

        separator4 = NSMenuItem.separatorItem()
        self._menu.addItem_(separator4)

        disable_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Disable Gestures", "toggleEnabled:", "d"
        )
        disable_item.setTarget_(self)
        self._menu.addItem_(disable_item)

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit", "quitApp:", "q"
        )
        quit_item.setTarget_(self)
        self._menu.addItem_(quit_item)

        self._status_item.setMenu_(self._menu)
        self._update_status_icon()

    def _update_status_icon(self) -> None:
        if not self._status_item:
            return

        if not self._camera_available:
            icon_text = MenuBarIcon.WARNING.value
        elif self._enabled:
            icon_text = MenuBarIcon.ENABLED.value
        else:
            icon_text = MenuBarIcon.DISABLED.value

        attr_string = NSAttributedString.alloc().initWithString_attributes_(
            icon_text,
            {
                NSFontAttributeName: NSFont.systemFontOfSize_(14),
                NSForegroundColorAttributeName: NSColor.whiteColor() if self._enabled else NSColor.grayColor(),
            },
        )
        self._status_item.button().setAttributedTitle_(attr_string)

    def _update_menu_state(self) -> None:
        if not self._menu:
            return

        if self._enabled:
            self._enabled_item.setTitle_("● Enabled")
            self._enabled_item.setState_(NSOnState)
            self._status_item_menu.setTitle_("Status: ● Active")
            if self._camera_item:
                self._camera_item.setTitle_("Camera: Built-in")
        else:
            self._enabled_item.setTitle_("○ Disabled")
            self._enabled_item.setState_(NSOffState)
            self._status_item_menu.setTitle_("Status: Inactive")
            if self._camera_item:
                self._camera_item.setTitle_("Camera: Off")

        if not self._camera_available:
            self._camera_item.setTitle_("Camera: Unavailable")
            self._status_item_menu.setTitle_("Status: Error")

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._update_status_icon()
        self._update_menu_state()
        if self._toggle_callback:
            try:
                self._toggle_callback(enabled)
            except Exception as e:
                logger.error(f"Toggle callback error: {e}")

    def set_camera_available(self, available: bool) -> None:
        self._camera_available = available
        self._update_status_icon()
        self._update_menu_state()

    def set_accessibility_ok(self, ok: bool) -> None:
        if self._access_item:
            self._access_item.setTitle_(
                "Accessibility: Granted" if ok else "Accessibility: Required"
            )

    def show_feedback(self, gesture: GestureType) -> None:
        if not CONFIG.app.enable_visual_feedback:
            return

        gesture_names = {
            GestureType.OPEN_PALM: "PLAY / PAUSE",
            GestureType.SWIPE_LEFT: "PREVIOUS DESKTOP",
            GestureType.SWIPE_RIGHT: "NEXT DESKTOP",
            GestureType.FIST: "FULLSCREEN",
            GestureType.THUMB_UP: "VOLUME UP",
            GestureType.THUMB_DOWN: "VOLUME DOWN",
        }

        name = gesture_names.get(gesture, "UNKNOWN")
        self._gesture_feedback = f"{gesture.value}  {name}"

        if self._feedback_timer:
            self._feedback_timer.invalidate()

        self._feedback_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            CONFIG.app.feedback_duration,
            self,
            "clearFeedback:",
            None,
            False,
        )

        self._update_feedback_menu()

    def clearFeedback_(self, timer) -> None:
        self._gesture_feedback = ""
        self._update_feedback_menu()

    def _update_feedback_menu(self) -> None:
        pass

    def set_action_callback(self, callback: Callable) -> None:
        self._action_callback = callback

    def set_quit_callback(self, callback: Callable) -> None:
        self._quit_callback = callback

    def set_toggle_callback(self, callback: Callable) -> None:
        self._toggle_callback = callback

    def toggleEnabled_(self, sender) -> None:
        self.set_enabled(not self._enabled)

    def quitApp_(self, sender) -> None:
        if self._quit_callback:
            self._quit_callback()

    def run(self) -> None:
        AppHelper.runEventLoop()

    def stop(self) -> None:
        AppHelper.stopEventLoop()


def run_menu_bar_app(
    enabled_callback: Callable[[bool], None],
    quit_callback: Callable[[], None],
    toggle_callback: Callable[[bool], None],
) -> GestureMenuBarApp:
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)

    menu_app = GestureMenuBarApp.alloc().init()
    menu_app.set_action_callback(enabled_callback)
    menu_app.set_quit_callback(quit_callback)
    menu_app.set_toggle_callback(toggle_callback)

    return menu_app