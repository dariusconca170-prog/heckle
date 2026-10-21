from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    (root / "sample.mp4").write_bytes(b"0123456789" * 100)
    return root


@pytest.fixture
def settings(tmp_path: Path, media_root: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        media_dir=media_root,
        frontend_dist=tmp_path / "dist",
        model_profile=tmp_path / "model_profile.json",
        db_path=tmp_path / "data" / "twitchsim.db",
        clips_dir=tmp_path / "clips",
        lookahead_s=5.5,
        seek_threshold_s=1.25,
    )


@pytest.fixture
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
