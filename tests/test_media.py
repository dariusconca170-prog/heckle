from __future__ import annotations

import pytest

from backend.media import (
    MediaError,
    MediaLibrary,
    MediaTokenStore,
    RangeNotSatisfiable,
    parse_range,
)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("bytes=0-9", (0, 9)),
        ("bytes=10-", (10, 99)),
        ("bytes=-10", (90, 99)),
        ("bytes=0-1000", (0, 99)),
        ("BYTES=5-5", (5, 5)),
    ],
)
def test_parse_range(header: str | None, expected: tuple[int, int] | None) -> None:
    assert parse_range(header, 100) == expected


@pytest.mark.parametrize(
    "header",
    ["bytes=100-", "bytes=5-2", "bytes=-0", "bytes=0-1,2-3", "bytes=abc-def", "bytes="],
)
def test_parse_range_rejects_invalid(header: str) -> None:
    with pytest.raises(RangeNotSatisfiable):
        parse_range(header, 100)


def test_media_library_lists_and_resolves(tmp_path) -> None:
    root = tmp_path / "media"
    (root / "nested").mkdir(parents=True)
    (root / "a.mp4").write_bytes(b"a")
    (root / "nested" / "b.mkv").write_bytes(b"b")
    (root / "notes.txt").write_text("not video")
    library = MediaLibrary(root, (".mp4", ".mkv"))
    items = library.list()
    assert [item.id for item in items] == ["a.mp4", "nested/b.mkv"]
    assert library.resolve("nested/b.mkv").name == "b"


def test_media_library_rejects_escape(tmp_path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    (tmp_path / "outside.mp4").write_bytes(b"x")
    library = MediaLibrary(root, (".mp4",))
    with pytest.raises(MediaError):
        library.resolve("../outside.mp4")
    with pytest.raises(MediaError):
        library.resolve(str(tmp_path / "outside.mp4"))
    with pytest.raises(MediaError):
        library.resolve("missing.mp4")


def test_token_store(tmp_path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    (root / "a.mp4").write_bytes(b"a")
    library = MediaLibrary(root, (".mp4",))
    item = library.resolve("a.mp4")
    store = MediaTokenStore()
    token = store.issue(item)
    assert store.resolve(token) == item
    assert store.issue(item) == token
    store.revoke(token)
    assert store.resolve(token) is None
