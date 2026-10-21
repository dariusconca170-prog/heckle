"""Typed application configuration.

Configuration precedence: constructor args > ``TWITCHSIM_*`` env vars > ``.env``
> ``config/default.toml`` > field defaults. Paths are resolved relative to the
repository root, not the process working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "default.toml"
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "data" / "media"
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


class Settings(BaseSettings):
    """Application settings shared by the API, the clock and the LLM harness."""

    model_config = SettingsConfigDict(
        env_prefix="TWITCHSIM_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        toml_file=DEFAULT_CONFIG_FILE,
        extra="ignore",
    )

    app_name: str = "Twitch Simulator"
    version: str = "0.1.0"

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    data_dir: Path = PROJECT_ROOT / "data"
    media_dir: Path = DEFAULT_MEDIA_DIR
    frontend_dist: Path = DEFAULT_FRONTEND_DIST

    # File playback: the read head is t_play + lookahead_s. M0.5 replaces the
    # value from measured latency; see data/bakeoff.md.
    lookahead_s: float = 5.5
    seek_threshold_s: float = 1.25
    clock_broadcast_hz: float = 4.0

    # llama.cpp sidecar (used by scripts/bakeoff.py and the app).
    llama_server: str = "llama-server"
    llama_host: str = "127.0.0.1"
    llama_port: int = 8080
    model_profile: Path = PROJECT_ROOT / "data" / "model_profile.json"
    llm_timeout_s: float = 180.0

    # M1 perception. A 512 px long edge is enough for the observer to name the
    # action while keeping vision prefill near 1.5 s on the 4B.
    perception_enabled: bool = True
    ffmpeg: str = "ffmpeg"
    frame_max_edge: int = 512
    frame_timeout_s: float = 15.0
    perception_interval_s: float = 3.0
    perception_scene_change_interval_s: float = 0.5
    perception_boost_s: float = 3.0
    perception_duty_cap: float = 0.35
    perception_error_backoff_s: float = 5.0

    # M3 chat dynamics. These are playback-time behaviours, not wall timers.
    dynamics_enabled: bool = True
    chat_target_buffer: int = 20
    chat_low_buffer: int = 12
    chat_critical_buffer: int = 5
    chat_batch_size: int = 10

    # M4 persistence. Messages and scene events land in SQLite; structured
    # persona memory is derived at stream end and every five minutes of
    # playback (crash safety). Memory rows are capped per viewer (PLAN.md §8).
    persistence_enabled: bool = True
    db_path: Path = PROJECT_ROOT / "data" / "twitchsim.db"
    memory_enabled: bool = True
    memory_top_n: int = 6
    memory_prompt_limit: int = 3
    memory_per_viewer: int = 10
    memory_max_total: int = 400
    memory_per_viewer_cap: int = 500
    memory_checkpoint_s: float = 300.0

    # M5 threads and channel address.
    channel_name: str = "streamer"

    # M6 polls.
    polls_enabled: bool = True

    # M7 clips.
    clips_enabled: bool = True
    clips_dir: Path = PROJECT_ROOT / "data" / "clips"
    clip_window_s: float = 10.0
    clip_baseline_s: float = 300.0
    clip_z_threshold: float = 3.0
    clip_preroll_s: float = 15.0
    clip_postroll_s: float = 5.0
    clip_min_gap_s: float = 45.0
    clip_top_n: int = 12

    # M8 microphone.
    mic_enabled: bool = False
    mic_sample_rate: int = 16000
    mic_chunk_s: float = 3.0
    mic_silence_threshold: float = 0.012
    mic_stt_model: str = "nemo-parakeet-tdt-0.6b-v3"

    # M9 screen capture.
    screen_capture_enabled: bool = False
    screen_fps: int = 12
    screen_hotkey: str = "f9"
    screen_blocklist: tuple[str, ...] = ("1password", "keepass", "bitwarden", "bank")

    media_extensions: tuple[str, ...] = (
        ".mp4",
        ".mkv",
        ".webm",
        ".mov",
        ".avi",
        ".m4v",
        ".ts",
        ".mpg",
        ".mpeg",
    )

    @field_validator(
        "data_dir",
        "media_dir",
        "frontend_dist",
        "model_profile",
        "db_path",
        "clips_dir",
        mode="after",
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (PROJECT_ROOT / value).resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


Settings.model_rebuild()
