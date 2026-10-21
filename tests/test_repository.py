from __future__ import annotations

from pathlib import Path

from backend.db.repo import MemoryRow, Repository

CONTRACT = {
    "streams": ["id", "title", "category", "source_path", "started_at", "ended_at", "meta_json"],
    "viewers": [
        "id",
        "handle",
        "archetype",
        "created_at",
        "chattiness",
        "toxicity",
        "style_json",
        "active",
    ],
    "viewer_memory": ["viewer_id", "stream_id", "kind", "key", "value", "weight", "updated_at"],
    "messages": [
        "id",
        "stream_id",
        "t_playback",
        "author_id",
        "tier",
        "text",
        "reply_to",
        "mentions_json",
        "emotes_json",
        "kind",
        "scene_ref_json",
        "created_at",
    ],
    "events": ["id", "stream_id", "t_playback", "type", "payload_json"],
    "polls": ["id", "stream_id", "question", "options_json", "opened_at", "closes_at", "status"],
    "poll_votes": ["poll_id", "viewer_id", "option_index", "t_playback"],
    "clips": [
        "id",
        "stream_id",
        "start_t",
        "end_t",
        "path",
        "thumb_path",
        "reason",
        "score",
    ],
}


def _repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "history" / "twitchsim.db")


def test_schema_matches_plan_contract(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        for table, columns in CONTRACT.items():
            assert repo.table_columns(table) == columns, table


def test_stream_and_messages_round_trip(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        repo.create_stream(
            "s1",
            title="first run",
            category="horror",
            source_path="/media/sample.mp4",
            meta={"media_id": "sample.mp4"},
        )
        repo.add_message(
            "s1",
            t_playback=12.5,
            handle="lore_dump",
            text="just like last stream",
            archetype="lore-keeper",
            tier="core",
            created_at=1000.0,
            scene_ref={"scene_t": 12.0},
        )
        repo.add_message(
            "s1",
            t_playback=13.0,
            handle="xX_deadlift_Xx",
            text="W",
            archetype="hype",
            tier="core",
            created_at=1000.5,
        )
        repo.add_event(
            "s1",
            t_playback=12.0,
            type_="fail",
            payload={"kind": "fail", "t": 12.0, "intensity": 0.9, "detail": "falls in cave"},
        )

        messages = repo.messages("s1")
        assert [message["text"] for message in messages] == ["just like last stream", "W"]
        assert messages[0]["handle"] == "lore_dump"
        assert messages[0]["scene_ref"] == {"scene_t": 12.0}
        assert repo.message_count("s1") == 2

        transcript = repo.transcript("s1")
        assert {row["handle"] for row in transcript} == {"lore_dump", "xX_deadlift_Xx"}

        events = repo.events("s1")
        assert events[0]["type"] == "fail"
        assert events[0]["payload"]["detail"] == "falls in cave"

        streams = repo.streams()
        assert streams[0]["id"] == "s1"
        assert streams[0]["message_count"] == 2
        assert streams[0]["meta"] == {"media_id": "sample.mp4"}
        assert streams[0]["ended_at"] is None

        repo.finish_stream("s1", meta={"messages": 2})
        finished = repo.stream("s1")
        assert finished is not None
        assert finished["ended_at"]
        assert finished["meta"]["messages"] == 2

        assert repo.viewer_count() == 2
        assert repo.stats()["messages"] == 2


def test_memory_upsert_load_and_prune(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        repo.create_stream("s1", title="one")
        repo.save_memory(
            [
                MemoryRow(
                    handle="lore_dump",
                    kind="lore",
                    key="fail:12",
                    value="fail: falls in cave",
                    weight=0.4,
                    stream_id="s1",
                ),
                MemoryRow(
                    handle="lore_dump",
                    kind="catchphrase",
                    key="just like last stream",
                    value="just like last stream",
                    weight=0.7,
                    stream_id="s1",
                ),
            ]
        )
        # Same key, lower weight: value refreshes but weight never regresses.
        repo.save_memory(
            [
                MemoryRow(
                    handle="lore_dump",
                    kind="lore",
                    key="fail:12",
                    value="fail: falls in a very dark cave",
                    weight=0.2,
                    stream_id="s1",
                )
            ]
        )
        rows = repo.load_memory(["lore_dump"], per_viewer=6)["lore_dump"]
        assert len(rows) == 2
        lore = next(row for row in rows if row.kind == "lore")
        assert lore.value == "fail: falls in a very dark cave"
        assert lore.weight == 0.4
        # Highest weight first.
        assert rows[0].kind == "catchphrase"

        assert repo.load_memory(["nobody"]) == {}
        assert repo.load_memory([]) == {}

        for index in range(6):
            repo.save_memory(
                [
                    MemoryRow(
                        handle="lore_dump",
                        kind="opinion",
                        key=f"topic{index}",
                        value=f"likes topic{index}",
                        weight=0.1 * index,
                    )
                ]
            )
        assert len(repo.memory_for("lore_dump", limit=3)) == 3
        assert repo.prune_memory(per_viewer=4) >= 0
        assert len(repo.memory_for("lore_dump", limit=50)) == 4
