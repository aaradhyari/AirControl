import logging

logger = logging.getLogger(__name__)

NX_KEYTYPE_PLAY = 16

_KEY_DOWN = 0xA
_KEY_UP = 0xB


def play_pause() -> bool:
    return _send_media_key(NX_KEYTYPE_PLAY)


def _send_media_key(key_code: int) -> bool:
    """Post a system-wide media-key press (works with Music, Spotify,
    browser YouTube, ...). Uses NSSystemDefined events with
    NX_SUBTYPE_AUX_CONTROL_BUTTONS -- the same events a physical
    Play/Pause key generates. No osascript, so nothing to hang."""
    try:
        from AppKit import NSEvent
        import AppKit
        from Quartz import CGEventPost, kCGHIDEventTap

        ns_system_defined = getattr(AppKit, "NSSystemDefined", 14)
        nx_subtype_aux = getattr(AppKit, "NX_SUBTYPE_AUX_CONTROL_BUTTONS", 8)

        def make_event(key_state: int):
            return NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                ns_system_defined,
                (0, 0),
                0,
                0,
                0,
                None,
                nx_subtype_aux,
                (key_code << 16) | (key_state << 8),
                -1,
            )

        event_down = make_event(_KEY_DOWN)
        event_up = make_event(_KEY_UP)
        if event_down is None or event_up is None:
            logger.error("Failed to create media key events")
            return False

        CGEventPost(kCGHIDEventTap, event_down.CGEvent())
        CGEventPost(kCGHIDEventTap, event_up.CGEvent())
        return True
    except Exception as e:
        logger.error(f"Media key failed: {e}")
        return False


def next_track() -> bool:
    return False


def previous_track() -> bool:
    return False