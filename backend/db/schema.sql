-- SQLite schema for the simulator. Column names are the contract from
-- PLAN.md §8; indexes are additive and may be tuned, but tables and columns
-- must not be renamed without updating the plan.
--
-- Timestamps: streams/viewers/viewer_memory use ISO-8601 UTC text so a human
-- can read the file; messages/events/polls/clips use playback seconds plus an
-- epoch float for wall time, because everything that drives chat is playback
-- time, never wall time.

CREATE TABLE IF NOT EXISTS streams (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS viewers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT NOT NULL UNIQUE,
    archetype TEXT NOT NULL DEFAULT 'normie',
    created_at TEXT NOT NULL,
    chattiness REAL NOT NULL DEFAULT 1.0,
    toxicity REAL NOT NULL DEFAULT 0.0,
    style_json TEXT NOT NULL DEFAULT '{}',
    active INTEGER NOT NULL DEFAULT 1
);

-- Structured persona memory, not raw chat. Kinds: opinion, catchphrase,
-- grudge, lore, preference. At most ~500 rows per viewer (pruned on write).
CREATE TABLE IF NOT EXISTS viewer_memory (
    viewer_id INTEGER NOT NULL REFERENCES viewers(id) ON DELETE CASCADE,
    stream_id TEXT,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_viewer_memory_key
    ON viewer_memory(viewer_id, kind, key);
CREATE INDEX IF NOT EXISTS idx_viewer_memory_weight
    ON viewer_memory(viewer_id, weight DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    t_playback REAL NOT NULL,
    author_id INTEGER NOT NULL REFERENCES viewers(id),
    tier TEXT NOT NULL DEFAULT 'core',
    text TEXT NOT NULL,
    reply_to INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    mentions_json TEXT NOT NULL DEFAULT '[]',
    emotes_json TEXT NOT NULL DEFAULT '[]',
    kind TEXT NOT NULL DEFAULT 'chat',
    scene_ref_json TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_stream_t
    ON messages(stream_id, t_playback);
CREATE INDEX IF NOT EXISTS idx_messages_stream_author
    ON messages(stream_id, author_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    t_playback REAL NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_stream_t
    ON events(stream_id, t_playback);

CREATE TABLE IF NOT EXISTS polls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    options_json TEXT NOT NULL DEFAULT '[]',
    opened_at REAL NOT NULL,
    closes_at REAL,
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS poll_votes (
    poll_id INTEGER NOT NULL REFERENCES polls(id) ON DELETE CASCADE,
    viewer_id INTEGER NOT NULL REFERENCES viewers(id) ON DELETE CASCADE,
    option_index INTEGER NOT NULL,
    t_playback REAL NOT NULL,
    PRIMARY KEY (poll_id, viewer_id)
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    start_t REAL NOT NULL,
    end_t REAL NOT NULL,
    path TEXT NOT NULL DEFAULT '',
    thumb_path TEXT,
    reason TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_clips_stream
    ON clips(stream_id, start_t);
