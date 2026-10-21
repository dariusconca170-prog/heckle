# Twitch Simulator

A local app where a video plays and a simulated Twitch chat reacts to what is
actually on screen — with human timing, stable personas, and no refusals. The
full specification is [`PLAN.md`](./PLAN.md); the build rules are in
[`AGENTS.md`](./AGENTS.md).

**Current status: M0–M10 complete.** M11 (packaging wizard) is the remaining
milestone. M0.5 was explicitly waived by the user and `data/bakeoff.md` names the
**4B as the default model** (`Qwen3.5-4B-Unredacted-MAX`, profile `config/models/4b.toml`,
written to `data/model_profile.json`) — provisionally, not as a measured winner. M1 perception, M2 chatter and M3
dynamics run against that profile; M4 adds SQLite history and persona memory;
M5–M7 add threads, polls/stream events and clips; M8–M9 add microphone and
screen capture; M10 adds the OBS overlay, emotes, settings/presets, export and
the VRAM/tok-s HUD. The M2/M3 human gates are still open; see
`data/eval/README.md`.

---

## What works now

- FastAPI backend with `/health`, media library, session lifecycle and clock API.
- Ranged video serving (`206`, suffix ranges, `416`, `HEAD`) with unguessable
  session tokens. The browser never gets an arbitrary filesystem path.
- A real HTML5 player and setup screen (Vite + TypeScript, no framework).
- One authoritative **playback clock**:
  - browser reports `t_play` at 4 Hz plus every play/pause/seek/rate change;
  - backend projects between reports;
  - `t_read = t_play + look-ahead` is exposed from day one;
  - seek/pause/resume increment `generation` and emit dedicated bus events so
    every future queue can be flushed instead of drifting.
- Typed in-process event bus with per-subscriber backpressure (slow consumers
  drop oldest events, never stall the producer).
- Frozen `SceneState` contract from `PLAN.md` §5.2, including `speech_text`,
  `speech_energy`, and `source`.
- M0.5 tooling: llama-server process manager, OpenAI-compatible client, JSON
  repair, refusal audit, voice sample generation, perception-accuracy runner,
  and `scripts/bakeoff.py` with a real 30-minute full-load gate.
- **M1 perception**: queue-aware ffmpeg frame sampler (look-ahead aware),
  observer prompt → frozen `SceneState`, 90 s timeline, derived scene events.
- **M2 chatter**: one batched model call per 8–10 personas, deterministic output
  guard (refusal/narrator/caption/assistant rejection, dedupe, length cap) and
  in-character template fallbacks. `/api/chat/generate` is the inspection tool.
- **M3 dynamics**: Hawkes rate, weighted author selection, lognormal perception
  delays, pre-generation buffer, emote/copy-pasta/typo mutations, crowd tier.
  The LLM never decides timing.
- **M5 threads**: reply targets with quote rendering, seeded 20 % reply rate,
  4+ message arguments ending with a third party, and ~4 % @streamer mentions.
- **M6 polls + stream furniture**: playback-timed polls with simulated voters and
  chat reactions, subs/gifts/raids/follows, and a viewer-count curve.
- **M7 clips**: z-score spike detection, ffmpeg export to 720p, midpoint
  thumbnails, and reason strings quoting real chat.
- **M8 microphone**: energy VAD, lazy Parakeet STT, manual/browser transcript
  injection, and a scene-state merge tagged `source: "mic"`.
- **M9 screen capture**: `dxcam`/`mss` behind the `FrameSource` interface, a
  privacy blocklist, and a pause hotkey/API toggle.
- **M10 polish**: transparent `/overlay/chat` for OBS, local emotes, runtime
  settings and presets, JSON/CSV chat export, and a VRAM/RAM/tok-s HUD.
- Tests for clock discontinuities, websocket sync, ranges, media safety, events,
  JSON repair, profiles, prompts, scene contract, refusal classification, chat
  dynamics, threads, polls, clips, mic VAD, screen privacy, settings, export,
  the schema contract, memory extraction, and two-stream persistence.

---

## Requirements

- Python 3.12+
- Node.js 20+ / npm
- ffmpeg on `PATH`
- NVIDIA GPU + `nvidia-smi` for the model milestones (RTX 3060 Ti 8 GB target)
- Windows is the shipping target; Linux/POSIX scripts are included for
  development on the same class of machine.
- M0.5 and M8 STT: `pip install -e ".[stt]"` (Parakeet ONNX via `onnx-asr`,
  CPU). M1 does not need it.
- M8 microphone capture: `pip install -e ".[mic]"` (`sounddevice`).
- M9 screen capture: `pip install -e ".[screen]"` (`mss`), or on Windows
  `pip install -e ".[screen-windows]"` for `dxcam`; the pause hotkey additionally
  needs `.[hotkey]`. Without either backend the API reports `unavailable`.
- OBS (or another browser-source host) only for the overlay.

---

## Quick start

### Windows

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

### Linux / POSIX

```bash
./scripts/setup.sh
./scripts/run.sh
```

`setup` creates `.venv`, installs the backend, builds the frontend, and creates
a 30-second demo clip at `data/media/sample.mp4`. `run` opens
<http://127.0.0.1:8000/>; pick the clip and press **Start stream**.

**Your own video:** press **add video** on the setup screen and pick any
`mp4`/`mkv`/`webm`/`mov`/`m4v`/`avi`/`ts` file. It streams straight into
`data/media/` (8 GB cap, existing names never overwritten) and is selected for
you. Dropping files into `data/media/` and pressing **refresh** still works.

The same endpoint is available by hand:

```bash
curl -X POST --data-binary @my-clip.mp4 \
  "http://127.0.0.1:8000/api/media/upload?name=my-clip.mp4"
```

Useful development commands:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check backend scripts tests
cd frontend && npm run build
```

> On Debian/Ubuntu without `python3-venv`, `setup.sh` falls back to
> `venv --without-pip` + `pip --python`. If even that fails, install
> `python3.12-venv`.

---

## M0 acceptance

| Requirement | Where |
|---|---|
| FastAPI app, config, `/health` | `backend/main.py`, `backend/config.py` |
| Static frontend | `frontend/`, mounted from `frontend/dist` |
| Video served with range requests | `backend/media.py`, `backend/api/routes.py` |
| Player plays a selected file | `frontend/src/app.ts`, `player.ts` |
| `PlaybackClock` reports at 4 Hz | `frontend/src/player.ts`, `backend/clock.py`, `backend/api/ws.py` |

The debug panel in the studio shows `t_play`, `t_read`, drift, generation,
look-ahead, video dimensions and update count. Seek the video and watch
`generation` increment; that is the future chat queue being flushed.

---

## M0.5 — the hard gate

> **Status: waived.** The user waived M0.5 for game streaming on an 8 GB card and
> selected the 4B provisionally; `data/bakeoff.md` says so explicitly. It is not
> a measured winner. Re-run this section when hardware time is available.

Do **not** start M1 until `data/bakeoff.md` exists, names a winner, and contains
a real 30-minute full-load row for **both** the 9B and the 35B. The 4B is a fit
fallback only.

### 1. Fetch models

```powershell
.\scripts\fetch_models.ps1 -Model 9b
.\scripts\fetch_models.ps1 -Model 35b
```

or:

```bash
./scripts/fetch_models.sh 9b 35b
```

The Hugging Face repos in `PLAN.md` are community uploads and may be renamed or
removed. Verify before downloading. The TOML profiles in `config/models/` point
at exact filenames under `data/models/`; edit those paths if the downloaded
names differ. Each profile needs its **matching `mmproj`**.

### 2. Prepare the full-load conditions

A gate run is only valid with all four:

1. the app is running at `http://127.0.0.1:8000/`;
2. a video is playing in the browser, so `/api/session` is active;
3. ffmpeg is decoding that video (the harness starts it);
4. a **real STT workload** is running on CPU for the whole run.

Install the pinned Parakeet ONNX stack once (this is the M0 STT install spike;
M8 reuses the same runtime):

```powershell
python -m pip install -e ".[stt]"
```

Then pass a real inference loop to the harness. `scripts/stt_load.py` runs
Parakeet TDT 0.6B v3 on CPU continuously, with the same 4-thread budget the app
will use:

```powershell
python scripts/bakeoff.py run --profile 9b `
  --stt-load "python scripts/stt_load.py --audio data/media/sample.mp4 --threads 4"
```

On POSIX the same command is:

```bash
python scripts/bakeoff.py run --profile 9b \
  --stt-load ".venv/bin/python scripts/stt_load.py --audio data/media/sample.mp4 --threads 4"
```

The load script accepts any file ffmpeg can decode. A spoken clip is preferable
to the synthetic demo tone because it exercises the decoder more; the encoder
cost, and therefore the CPU/RAM pressure the gate measures, is the same.
`--duration-min 0` (the default) keeps running until the bake-off terminates it.

M8 is not built yet, so this is a standalone load generator, not the audio tap.
A placeholder CPU burner does not satisfy the gate; the harness refuses to start
a non-partial run without a live STT process.

### 3. Measure both models

```powershell
python scripts/bakeoff.py run --profile 9b
python scripts/bakeoff.py run --profile 35b
```

Each command:

- launches llama-server itself with the profile's exact flags;
- runs 20 vision-prompt and 20 eight-message chat-batch timings;
- runs the sustained full-load loop (30 minutes by default);
- samples VRAM, system RAM, process RSS/pagefile/page faults every 10 s;
- aborts early at minute 5 if RAM `< 1.5 GB` or tok/s `< 25`;
- writes a raw JSON row and appends to `data/bakeoff.md`.

For harness smoke tests only:

```powershell
python scripts/bakeoff.py run --profile 9b --duration-min 0.1 --allow-partial --no-early-abort
```

A partial run cannot pass the gate and cannot select a winner.

### 4. Refusal audits (both models)

```powershell
python scripts/refusal_audit.py --profile 9b
python scripts/refusal_audit.py --profile 35b
```

40 scenes × 2 modes = 80 calls each. Pass: **0 refusals, ≤ 2 moralizing**. Results
go to `data/bakeoff/refusal_<profile>.json`.

### 5. Blind voice score (both models)

```powershell
python scripts/bakeoff.py voice --profile 9b --blind
python scripts/bakeoff.py voice --profile 35b --blind
python scripts/bakeoff.py rate-voice --label <label>
```

The label is anonymized. Rate 1–5 on “sounds like real Twitch chat, not an
assistant”, using the generated samples. Results are tied back to the profile
only after rating.

### 6. Perception accuracy (both models)

Create `data/perception_eval/manifest.jsonl` with 30 hand-labelled frames
spanning gameplay, talking head, cooking, text-heavy UI and explicit content:

```json
{"image": "frames/001.jpg", "action_keywords": ["hiding"], "object_keywords": ["car", "rain"]}
```

Then:

```powershell
python scripts/bakeoff.py accuracy --profile 9b
python scripts/bakeoff.py accuracy --profile 35b
```

Target ≥ 80 %. The command records summaries for human verification; keyword
matching is only a first pass.

### 7. Select the winner

```powershell
python scripts/bakeoff.py select
```

`select` refuses to write a winner unless both 30-minute rows pass the floor and
all evidence above exists. It applies the decision rule in `PLAN.md` §7.4:
floor first, then blind voice score; a tie goes to the 9B. On success it writes:

- `data/bakeoff.md` — human-readable gate document;
- `data/model_profile.json` — the exact winning profile and command;
- `data/bakeoff/selection.json` — machine-readable decision.

`scripts/run.ps1` / `run.sh` start the selected sidecar automatically once
`data/model_profile.json` exists. The app does **not** re-tune on launch.

### Look-ahead

`select` computes `LOOKAHEAD = t_scene + t_batch + 2.0 s` from the measured run
and writes it to the winning profile. The 9B starting value is 5.5 s; the 35B
uses the larger value the measurement dictates (roughly 8–10 s). Do not copy
the 9B number onto the 35B.

---

## M4 — persistence and memory

SQLite lives at `data/twitchsim.db` (ignored by git). The schema is the contract
from `PLAN.md` §8: `streams`, `viewers`, `viewer_memory`, `messages`, `events`,
`polls`, `poll_votes`, `clips`. M4 writes history and memory; M6 writes poll
rows/votes; M7 writes clip rows and export paths.

What is logged:

- every **released** chat message (`CHAT_MESSAGE` with `kind=chat`), including
  tier, release time and scene reference;
- every derived scene event (`SCENE_EVENT`), stored in playback time;
- a stream row on start and its end time/meta on stop.

Preview batches (`/api/chat/generate`) are marked `kind=preview` and are not
stored as stream history.

Persona memory is structured, not raw chat. At stream end and every 5 minutes of
playback the transcript + events are reduced deterministically to at most 10 rows
per viewer (500 per viewer is the hard cap):

| Kind | Source |
|---|---|
| `catchphrase` | a short line that viewer repeated 2+ times |
| `opinion` | a sentiment word plus the nearest concrete noun |
| `lore` | salient scene events (fail/clutch/jumpscare/…), or the stream title for a lore-keeper that stayed quiet |

The top 3 rows per viewer are injected into the chat prompt as `memory: …` when
that viewer is in the batch. Once per session the lore-keeper gets a guaranteed
beat: the model is asked to work one memory into the line, and if it ignores the
request the line falls back to an in-voice template (`still not over that fail`).
This is what satisfies the M4 acceptance ("second stream references the first")
without relying on luck.

Inspection endpoints:

```text
GET /api/history/streams
GET /api/history/streams/{id}
GET /api/history/streams/{id}/messages?limit=&offset=&newest_first=
GET /api/history/memory?handle=lore_dump&limit=10
```

If the database cannot be opened, persistence is disabled, `/health` reports
`persistence.enabled=false`, and chat keeps running from templates.

### M4 acceptance

| Requirement | Evidence |
|---|---|
| Schema matches `PLAN.md` §8 | `tests/test_repository.py::test_schema_matches_plan_contract` |
| Messages/events survive a session | `tests/test_persistence.py` |
| Memory is written at end and at checkpoints | `tests/test_persistence.py`, `test_checkpoint_writes_memory_mid_stream` |
| Second stream gets the first stream's memory | `tests/test_persistence.py::test_two_streams_carry_memory` |
| A lore-keeper references the previous stream | `tests/test_memory_beat.py`; verified live with `scripts/capture_chat.py` across two sessions (`still not over that fail` in run 2) |

---

## M5 — replies, threads and arguments

`backend/dynamics/threads.py` owns the structural decisions: no self-replies, no
reply to a message that has not been released, and an argument always ends with
a third person telling both sides to stop.

- ~20 % of planned messages target a visible message released 1–35 s earlier;
  questions, hot takes and insults are weighted 3× as replyable.
- A quote line (`@handle  parent text`) is rendered by the studio and overlay.
- Arguments are scheduled once per 75–150 s of playback and always contain at
  least five messages, alternating two personas, then a third-party stop.
- ~4 % of non-reply messages prepend `@` + `channel_name` (default `streamer`).
- `messages.reply_to` is resolved from the live message id to the SQLite row id;
  history joins the parent so exports and the UI show the quote.

**Acceptance:** measured at **20.9 % replies / 3.8 % mentions** over a seeded
90 s simulation with 263 released messages; `tests/test_threads.py` enforces the
≥ 15 % floor and the argument/thread invariants.

---

## M6 — polls and stream events

`backend/features/polls.py` is a playback-time state machine. A poll opens at
the current `t_play`, closes after `duration_s` of playback, and does not advance
while the video is paused. Votes are simulated per active persona with an
archetype bias, a small first-option edge, a herd effect toward the leader, and
a 5 % “misread the options” chance. Chat gets vote announcements, option
arguments, and winner/loser reactions through the normal dynamics queue.

```text
POST /api/polls                     {question, options[], duration_s, allow_multiple}
GET  /api/polls                     current + history + metrics
POST /api/polls/{id}/vote           {handle, option_index}
POST /api/polls/close
```

`backend/features/stream_events.py` produces subs/gifts/raids/follows
(1 sub ≈ 4–8 min, 1 raid ≈ 15–30 min, a follow ≈ 1.5–5 min) plus a viewer curve:
slow ramp, plateau, ±3 % noise, and a decaying raid spike. Events are system
messages in chat, apply a burst, and get short acknowledgements (`W`, `o7`,
`gigachad`) rather than “thanks for the sub”. They are also written to `events`.

```text
POST /api/stream-events/{sub|gift|raid|follow}
```

**Acceptance:** a 6 s poll gathered 31 votes over the real WebSocket clock,
closed itself, announced results, and paused with the playhead. The studio poll
panel shows live bars, the timer, and click-to-vote as `you`.

---

## M7 — clips

`backend/features/clips.py` keeps a 10 s rolling message rate and compares it
with a 5-minute per-second baseline. `z > 3` creates `[now-15 s, now+5 s]`, scores
it from peak rate × scene intensity × duration, and exports after the segment
has played. ffmpeg re-encodes to 720p (H.264/AAC, faststart) and grabs a midpoint
thumbnail. The reason string quotes the three highest-engagement real messages.

```text
GET  /api/clips                      current stream clips + metrics
POST /api/clips                      {start_t?, end_t?, reason?} (manual clip)
GET  /api/clips/{id}/file            MP4 (409 until the export finishes)
GET  /api/clips/{id}/thumb           JPEG
```

Paths are stored relative to `clips_dir` (default `data/clips`) and resolved
inside that directory only. If ffmpeg fails the row remains, `status=failed`, and
chat keeps running.

**Acceptance:** `tests/test_clips.py` detects a seeded spike, checks the reason
quotes the replay messages, and exports a real 5 s segment whose ffprobe duration
is 4.5–5.6 s with a valid JPEG thumbnail. The live smoke test exported a clip
from `sample.mp4` and served it over HTTP.

---

## M8 — microphone

Install the optional capture/STT stack:

```powershell
python -m pip install -e ".[mic,stt]"
```

Set `mic_enabled = true` (or enable it on the settings screen). `sounddevice`
captures 16 kHz mono; `VadSegmenter` uses an adaptive energy floor, 0.25 s
preroll and 0.7 s hangover so silence never calls STT. Segments go to Parakeet
via `onnx-asr` on CPU. A transcript is pushed into `SceneState.speech_text` (with
`source="mic"`) and published as `mic.speech`; the dynamics engine turns it into
1–3 reactions within a few seconds even when the vision loop is disabled.

```text
GET  /api/mic
POST /api/mic/start
POST /api/mic/stop
POST /api/mic/transcript    {text, energy}   manual/browser-STT fallback
```

**Acceptance:** the VAD and injection tests cover segmentation, the bus event,
the scene merge and ≥ 2 reactions within 5.5 s. A real microphone could not be
measured on this Linux host; the optional stack is not installed.

---

## M9 — screen capture

`backend/perception/screen.py` implements the same `FrameSource` interface as
file playback. It prefers `dxcam` on Windows and falls back to `mss`; raw pixels
are encoded/downscaled by ffmpeg, so OpenCV is not required. There is no
look-ahead: the frame is the present, so reactions land 3–5 s after the moment.

Every grab checks the active window title against `screen_blocklist`
(case-insensitive). A match raises before the frame reaches the model. The
pause hotkey (default `f9`, via optional `keyboard`) and the API toggle both stop
capture within one loop tick.

```text
GET  /api/screen
POST /api/screen/start      requires an active stream
POST /api/screen/stop
POST /api/screen/toggle
```

**Acceptance:** privacy blocking, injected frames, scene updates and pause
semantics are tested. A real DXGI capture could not be exercised on this Linux
host (no `dxcam`/`mss`); the service reports `unavailable` with a clear error
instead of crashing.

---

## M10 — overlay, emotes, settings, export, HUD

### OBS overlay

Add a **Browser Source** in OBS:

- URL: `http://127.0.0.1:8000/overlay/chat` (also `/overlay`)
- Width `400`, Height `800`
- check **Shutdown source when not visible**, leave custom CSS empty
- the page is transparent, hides scrollbars, uses 120 ms entrances, and renders
  reply quotes, badges, system messages and local emotes

The overlay is served by FastAPI, not Vite, so it keeps working even when the
frontend build is stale.

### Emotes

`frontend/public/emotes/` ships original SVGs for `W`, `LUL`, `KEKW`, `Pog`,
`monkaS`, `copium`, `ratio` and `<3`. Chat tokens are mapped case-insensitively;
unknown 7TV/BTTV names stay as text. The same set is used by the studio and the
overlay.

### Settings and presets

`/api/settings` persists runtime choices to `data/settings.json` (atomic write):
content mode (`unhinged`/`edgy`/`neutral`), intensity, buffer target, look-ahead,
model path/quant, TTS flag and mic flag. Presets are `calm`, `balanced`, `chaos`.
Intensity and content mode apply immediately; look-ahead applies to the next
stream because changing it mid-stream would invalidate pre-generated messages.

```text
GET   /api/settings
PATCH /api/settings
POST  /api/settings/preset/{calm|balanced|chaos}
GET   /api/system                 VRAM, RAM, tok/s, chat rate, viewers
```

### Chat export

The exporter reads SQLite and returns an attachment with quoted CSV fields or a
JSON envelope:

```text
GET /api/export/chat?format=json|csv
GET /api/history/streams/{id}/export?format=json|csv
```

### HUD

The studio strip shows live viewers, VRAM used/total, last tok/s, RAM %, chat
rate and reply count. `nvidia-smi` is optional and cached for 2 s; without a GPU
the row says `no GPU`. Tok/s comes from llama-server completion usage; with
template-only chatter it reports `off`.

**Acceptance:** `tests/test_polish.py` verifies the transparent overlay markup,
shipped emotes, settings persistence and validation, system metrics without a
GPU, both export formats, and `/health` coverage. The live smoke test fetched
`/overlay/chat` and `/emotes/w.svg`, patched settings, and exported 98 rows.

---

## Architecture boundaries (already enforced where they exist)

```text
browser player ──4 Hz time──► backend clock ──► event bus
                                  │
                                  └─ t_read = t_play + look-ahead (M1 consumes)

media token ──► ranged HTTP ──► <video>
```

- `backend/clock.py` is the only source of `t_play`/`t_read`.
- `backend/events.py` owns the event vocabulary; modules subscribe, they do not
  import each other's internals.
- `backend/media.py` owns library validation and byte ranges.
- `backend/llm/` owns llama.cpp lifecycle, prompts and JSON repair. It does not
  schedule messages.
- `backend/perception/scene.py` is the frozen `SceneState` contract.

From M3 onward the Chat Dynamics Engine decides *when* and *who*; the LLM only
fills a pre-generated text buffer. The LLM never schedules a message.

From M4 onward `backend/persistence/store.py` owns the session lifecycle and
memory, `backend/persistence/service.py` drains the bus into SQLite, and
`backend/db/repo.py` is the only module that speaks SQL.

---

## Repository map

```text
backend/
  main.py                 FastAPI app + static mount + lifespan
  config.py               pydantic-settings (env > .env > TOML)
  clock.py                PlaybackClock, t_play/t_read, generations
  events.py               typed event bus
  media.py                media library, tokens, byte ranges
  api/routes.py           HTTP (library, session, scene, chat, history, export)
  api/features.py         polls, clips, mic, screen, settings, system
  api/overlay.py          transparent OBS chat page
  api/ws.py               browser clock in, bus events out
  runtime.py              data/settings.json overrides + presets
  perception/             frame sampler, observer, timeline/events, screen source
  chatter/                personas, batched generation, guard, memory
  dynamics/               Hawkes, roster, mutations, scheduler, threads
  features/               polls, stream events, clips, mic, screen, system HUD
  persistence/            store (lifecycle/memory) + bus writer
  db/                     schema.sql + synchronous repository
  llm/                    client, server manager, profiles, prompts, repair
config/
  default.toml
  models/{9b,35b,4b}.toml
  voice_scenes.json
  refusal_prompts.json
frontend/                 Vite + TypeScript, one page
scripts/
  setup/run/fetch_models  Windows + POSIX
  make_sample.py
  bakeoff.py              M0.5 gate harness
  refusal_audit.py        40 × 2 audit
  capture_chat.py         live transcript capture for M2/M3 evaluation
  analyze_chat.py         deterministic style/repetition checks
  blind_ab.py             blind A/B packet builder
  serve_model.py          foreground sidecar runner
tests/                    backend, schema, memory, dynamics and prompt tests
data/
  media/                  your videos (sample generated)
  twitchsim.db            runtime history + memory (gitignored)
  bakeoff.md              created by the gate, not by hand
  model_profile.json      created only by `bakeoff.py select`
```

Not implemented yet: M11 packaging (first-run model wizard + checksum
verification) and optional TTS (v3). The screen source lives in
`perception/screen.py` behind the same `FrameSource` interface rather than a
separate `sources/` package; the boundary is unchanged.

---

## Troubleshooting

- **“Frontend is not built”** — run `cd frontend && npm install && npm run build`,
  or re-run setup.
- **Video seeks but chat state is stale** — check that `generation` increments on
  seek; `clock.seek` is the flush signal for all future queues.
- **`416` on video** — the browser asked for a range outside the file; reload the
  page. The server reports `Content-Range: bytes */<size>`.
- **Port 8080 busy** — another llama-server is running. Stop it or pass
  `--port` to `bakeoff.py`, and update the profile.
- **M0.5 run refuses to start** — it is telling you the load is not representative.
  Start the app, play a video, supply the STT command, or use `--allow-partial`
  for a smoke test that cannot select a winner.
- **Vision projector on CPU** — the bake-off and startup log record
  `mmproj_on_gpu`. If false, perception latency jumps; fix the llama.cpp build or
  raise the frame interval before accepting the result.
- **Overlay is black in OBS** — the page needs `http://`, not `https://`, and the
  Browser Source must have a transparent background. Use width 400 / height 800
  and leave “shutdown when not visible” checked so OBS stops the socket when hidden.
- **Emotes show as text** — only the bundled eight names are local; add SVGs to
  `frontend/public/emotes/` and extend `frontend/src/emotes.ts` +
  `backend/api/overlay.py`.
- **Mic/screen unavailable** — install the optional extra shown by
  `GET /api/mic` / `GET /api/screen`; capture is disabled by default and never
  blocks chat when missing.
- **Clip export fails** — check `ffmpeg` is on `PATH` and that the source video
  still exists. The clip row stays with `status=failed`; chat is unaffected.

The Definition of Done is in `PLAN.md` Appendix C. It is not done until a human
cannot reliably distinguish the simulated chat from a real Twitch VOD.
