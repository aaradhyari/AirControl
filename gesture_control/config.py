from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30


@dataclass
class GestureConfig:
    hold_time: float = 0.45
    action_cooldown: float = 1.0
    swipe_window: float = 0.5
    min_swipe_distance: int = 120
    min_swipe_velocity: float = 300.0
    max_vertical_displacement: int = 80
    static_stillness_px: float = 40.0
    static_stability: float = 0.7
    volume_step: int = 5
    confidence_threshold: float = 0.7
    smoothing_frames: int = 3


@dataclass
class ShortcutConfig:
    left_space: Tuple[str, ...] = ("ctrl", "left")
    right_space: Tuple[str, ...] = ("ctrl", "right")
    fullscreen: Tuple[str, ...] = ("cmd", "ctrl", "f")
    play_pause: Tuple[str, ...] = ("play_pause",)
    volume_up: Tuple[str, ...] = ("volume_up",)
    volume_down: Tuple[str, ...] = ("volume_down",)


@dataclass
class AppConfig:
    enable_visual_feedback: bool = True
    feedback_duration: float = 0.8
    debug_mode: bool = False
    launch_at_login: bool = False


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    shortcuts: ShortcutConfig = field(default_factory=ShortcutConfig)
    app: AppConfig = field(default_factory=AppConfig)


CONFIG = Config()