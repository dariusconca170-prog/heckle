# Twitch Simulator — Implementation Plan

> **Audience:** an AI coding agent (or a human) building this on a Windows PC with an RTX 3060 Ti (8 GB VRAM) and 16 GB DDR4-3200.
> **Status:** greenfield. Nothing exists yet.
> **Revision:** v3 — hardware locked, single 9B profile, research notes removed.
> **Goal:** a local app where you load a video, it plays, and a simulated Twitch chat reacts to what is actually on screen — with human-like timing, personalities, and memory. **Uncensored by design:** chat reacts to anything on screen (gore, NSFW, slurs, shock content) without refusing, moralizing, or breaking character.

---

## 0. TL;DR for the implementer

Build a **Python backend + local web UI**, with **llama.cpp (`llama-server`) as a sidecar** running **one abliterated multimodal GGUF** that does both perception and chat:

| Role | Model | Why |
|---|---|---|
| **Observer + Chatter** (one model, two prompts) | `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` + its `mmproj` | Natively multimodal; ~5.4 GB weights + ~0.9 GB projector fits entirely in 8 GB VRAM; ~39 tok/s. Fallback if it OOMs: `Qwen3.5-4B-Unredacted-MAX`. |

Perception and chat are **separate services with separate prompts**; they share one model endpoint and a priority queue. One model means no second model competing for VRAM, and the 9B is the largest thing that runs fully on the GPU here.

**The single most important architectural rule:** *the LLM never decides when a message appears.* A deterministic **Chat Dynamics Engine** decides timing, rate, bursts, who speaks, and how late they are. The LLM only fills a buffer with candidate text. This is what makes it feel real, and it is what makes 8 GB enough.

**Second rule:** for file playback, **perception reads ahead of the playhead** (`t_read = t_play + L`). The file is known, so the app can see the future, pre-generate reactions, and release them at the right moment. This makes perceived reaction time equal the simulated human delay (1.5–4 s) instead of compute time. Details in §5.1, §7.7, §9.3.

Everything else (polls, clips, mic, screen capture) hangs off the same event bus.

**Hardware is known — do not branch on it:**
- GPU: RTX 3060 Ti, 8 GB VRAM, ~7.0–7.3 GB usable with the desktop attached.
- RAM: 16 GB DDR4-3200. Plenty, because the model lives in VRAM. RAM speed is irrelevant to token rate here.
- Windows. Decode video on CPU with ffmpeg.

The default is the 9B. The 35B-A3B is a **measured challenger**, not a fantasy: it can hit ~30 tok/s in a clean benchmark on this class of card. The problem is not raw speed, it is *sustained* speed. It needs ~12–14 GB of system RAM resident for offloaded experts, and this machine has 16 GB total — so with the browser, ffmpeg, and STT running, Windows starts paging weights from disk. A 60-second benchmark looks fine; minute 40 of a stream does not.

Whether that actually happens here is measurable, not arguable: M0.5 runs a 30-minute full-load test (§7.4). If the 35B holds ≥ 25 tok/s with ≥ 1 GB RAM free at minute 30, it is a legitimate candidate; it then competes with the 9B on blind chat-voice score, and the winner is what gets built. Until that test passes, build against the 9B — but do not hardcode the 9B if the 35B wins.

> **M0.5 is a hard gate, not a formality.** Measure *both* models under full load, write `data/bakeoff.md`, and build whichever wins. The word "default" above describes a starting point, not a decision. If M1 starts and `data/bakeoff.md` has no 35B row, the gate was skipped.

See §7.1.

---

## 1. Product definition

### 1.1 Core loop (v1)

1. User picks a video file (mp4/mkv/webm/any ffmpeg-readable).
2. App plays it (embedded player, with play/pause/seek).
3. Backend samples frames + the video's audio track.
4. The model turns frames (and audio) into a rolling **scene timeline** ("player is hiding behind a car, enemies approaching, tense music").
5. Chat Dynamics Engine decides *how many* viewers are talking, *when*, and *who*.
6. Chatter model writes messages grounded in the scene timeline and each persona's voice + memory.
7. Messages stream into a Twitch-style chat pane, in sync with playback.

### 1.2 Non-goals (v1)

- Not connecting to real Twitch (no OAuth, no real chat).
- Not multi-user, not networked, not cloud.
- Not training/fine-tuning models. Inference + prompting only.

### 1.3 Later phases (explicitly in scope)

- Stream your own screen instead of a file.
- Microphone input: chat hears what you say (live STT) and reacts.
- Persona "styles": happy, angry, meme, wholesome, doomer, backseater, lurker, etc. (large roster).
- Polls with N options and a timer; chat votes and reacts to results.
- Reply threads, @mentions, viewer-to-viewer arguments.
- Persistent personas with cross-stream memory.
- Clip/highlight detection from chat spikes.
- Standard stream furniture: subs/gift subs/raids/follows, emote spam, viewer count curve, TTS, OBS overlay.
- (Optional, later) moderation tools even though the model is uncensored.

---

## 2. Hard constraints and design consequences

**VRAM is the binding constraint.** The 9B + projector + KV + overhead lands at roughly **7.1 GB**, against ~7.2 GB usable on Windows. Everything below follows from that.

| Constraint | Decision |
|---|---|
| 9B starts at ~7.1 GB total — almost no margin | Tune down to **≤ 6.7 GB** so Windows keeps 500 MB: `Q4_K_M` first, then `Q4_K_S`/`IQ4_XS` or ctx 3072. `-ctk/-ctv q8_0` is mandatory, not optional. |
| No room for a second model | **One model, two prompts.** Never run a separate observer model. |
| ~39 tok/s decode | Never generate one message per call. **Batch 8–10 messages per call** and keep a pre-generation buffer. |
| Prefill competes with chat | Perception gets the lowest priority in the GPU queue and adapts its frame interval to queue depth (§5.1). |
| GPU is shared with playback | Decode video on CPU (ffmpeg); UI rendering is browser-GPU-composited and costs no VRAM. |
| Windows desktop eats 0.5–1 GB VRAM | Do not hardcode a context size or quant that only fits on a clean machine. **Browser video decode + compositing also uses ~200–500 MB VRAM** (not zero). Start the 9B at `Q4_K_S` or `-ub 1024` if `Q4_K_M` leaves < 500 MB free with the player running. |
| Vision projector may land on CPU | Known llama.cpp issue in some builds. Verify in the startup log; if `mmproj` is on CPU, add ~0.5–1.5 s per frame and raise the interval. |
| 16 GB system RAM | Sufficient: the model is in VRAM. Budget ~2–3 GB for Python + browser, ~1 GB for ffmpeg/STT. Do not run a second large model. |

**Latency targets** (these are acceptance criteria, measured in M0.5 and M1):

- Frame captured → `SceneState` available: **< 1.5 s p95**
- Scene change → first message referencing it: **1.5–4 s** (simulated delay, not compute)
- Chat buffer never empty during normal playback (≥ 15 ready messages)
- Burst capacity: **6 msg/s for 10 s** — delivered by the crowd tier, not the LLM

---

## 3. Architecture

```
┌──────────────────────────── Browser (UI) ────────────────────────────┐
│  Video player (HTML5)   Chat pane   Poll widget   Clip list   Settings│
│         │                  ▲            ▲             ▲              │
└─────────┼──────────────────┼────────────┼─────────────┼──────────────┘
          │ WS: time/seek    │ WS: messages/events                  HTTP
          ▼                  │            │
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI backend (Python)                     │
│                                                                     │
│  PlaybackClock ──► FrameSampler ──► Observer ──► SceneTimeline      │
│       │                (ffmpeg)       (VLM)        │   │            │
│       │                                            │   │            │
│       │            AudioTap (RMS + STT) ───────────┘   │            │
│       │                                                ▼            │
│       │                                    ┌────────────────────┐   │
│       ├───────────────────────────────────►│  Event Bus         │   │
│       │                                    │  scene / topic /   │   │
│       │                                    │  poll / sub / clip │   │
│       │                                    └─────────┬──────────┘   │
│       │                                              ▼              │
│       │                              ┌───────────────────────────┐  │
│       │                              │  Chat Dynamics Engine     │  │
│       │                              │  rate, bursts, delays,    │  │
│       │                              │  who speaks, threading    │  │
│       │                              └─────────┬─────────────────┘  │
│       │                                        │ requests           │
│       │                                        ▼                    │
│       │                              ┌───────────────────────────┐  │
│       │                              │  Chatter Service          │  │
│       │                              │  persona + scene → texts  │  │
│       │                              │  (batched, buffered)      │  │
│       │                              └─────────┬─────────────────┘  │
│       │                                        │                    │
│       ▼                                        ▼                    │
│  SQLite (streams, viewers, messages, events, clips, memory)         │
└─────────────────────────────────────────────────────────────────────┘
                                            │ OpenAI-compatible HTTP
                                            ▼
                                 llama-server (GGUF, CUDA)
                                 └─ one multimodal model + mmproj
                                    ├─ perception prompt (SceneState)
                                    ├─ chatter prompt (batched messages)
                                    └─ priority queue: user > event > buffer > frame
```

### 3.1 Module boundaries (keep these clean)

| Module | Responsibility | Must NOT do |
|---|---|---|
| `sources/` | Produce `(timestamp, frame_bgr, audio_chunk)` | Know about chat |
| `perception/` | Frames + audio → `SceneState`, `SceneEvent` | Talk to the models directly (go through `llm/`) |
| `llm/` | Model lifecycle, batching, prompt templates, JSON repair | Decide scheduling |
| `dynamics/` | Rates, arrivals, delays, persona selection, threading | Generate text |
| `chat/` | Assemble final messages, dedupe, persist, broadcast | Call models synchronously in request path |
| `features/` | polls, clips, stt, tts, events | — |
| `api/` | HTTP + WebSocket | Business logic |

Everything communicates through an in-process **async event bus** with typed events. No module imports another module's internals.

---

## 4. The Chat Dynamics Engine (the differentiator — build this carefully)

Real Twitch chat is **not** a steady drip of one message every N seconds. It is a self-exciting point process: one message makes more messages likely. Model it as a **Hawkes process**:

```
λ(t) = μ(t) + Σ_i α · exp(-β · (t - t_i))
```

- `μ(t)` — baseline rate = `active_chatters × 0.015 msg/s × intensity(t)`
- `intensity(t)` — 0.3 (boring dialogue) … 5.0 (jumpscare, clutch play, funny moment), derived from `SceneState.surprise` + audio loudness + frame-diff
- `α` — excitation per message (default 0.35)
- `β` — decay (default 0.6 /s) → a burst decays over ~4–8 s
- Hard clamp: **≤ 8 msg/s**, and a global cooldown after 3 s of max rate (chat "catches its breath")

### 4.1 Message lifecycle

Every message goes through these stages, in this order:

1. **Trigger** — a `SceneEvent` (big) or the Hawkes clock (ambient).
2. **Author selection** — weighted pick from roster: `chattiness × arousal × (1 + affinity to current topic)`. 70–85 % of the roster is currently lurking (weight 0) and occasionally wakes up.
3. **Perception delay** — `lognormal(μ=1.7 s, σ=0.55)`, clamped to `[0.6, 9] s`. This creates the "someone reacts 6 seconds late" effect. The message is generated from the scene state *at `now - delay`*, not at `now`.
4. **Content request** — enqueue to Chatter Service with the snapshot of the scene at that past time.
5. **Buffer wait** — text arrives in < 2 s because it was pre-generated.
6. **Release** — message becomes visible at `trigger + delay`, with ±150 ms jitter so messages do not land on a grid.
7. **Reaction cascade** — a message may spawn 0–3 replies with their own short delays, forming threads.

### 4.2 Human-ness rules (apply deterministically, not via LLM)

These are cheap and do more for realism than prompt engineering:

- **Length distribution:** 65 % of messages are 1–4 words. 20 % are 5–10. 15 % are emote-only.
- **No trailing punctuation** in ~80 % of messages; no capital first letter in ~60 %.
- **Typos:** 6 % of messages get one typo (swap/delete/double a character). 2 % get a second.
- **Emotes:** 25 % contain 1 emote, 10 % contain 2+, 8 % are the same emote repeated 3–12×.
- **Copy-pasta:** when a message with an emote/pattern gets ≥ 2 "agreements", 3–15 crowd-tier viewers repeat it within 1–3 s. This is the single most Twitch-y behavior.
- **"First"**: exactly one viewer says `first` in the first 4 s of a stream, and gets flamed for it.
- **Late reactions:** 12 % of messages reference an event 5–20 s stale ("wait what happened", "did he just...").
- **Contradictions:** 8 % of messages are just wrong about the scene. *A chat that is always correct feels fake.*
- **Lurkers:** a subset speaks once every 3–10 minutes, with one low-effort message ("lol", "W").
- **Reply rate:** ~18 % of messages reply to a recent message; ~4 % @mention the streamer.
- **Silence:** if the scene is static and quiet, chat must actually go quiet for 10–20 s sometimes. Do not fear empty space.

### 4.3 Two tiers of viewers

| Tier | Count | Driven by | Cost |
|---|---|---|---|
| **Core personas** | 20–80 | LLM (batched). Full personality, memory, opinions, thread participation. | ~all LLM budget |
| **Crowd personas** | 100–50,000 | Templates + emote tables + copy-pasta + name generator. Never calls the LLM. | ~0 |

The UI exposes `viewers: 1200` and `active chatters: 40` separately. Crowd tier produces the noise floor (emote spam, "W", "L", "first", copy-pasta, "?"); core tier produces the messages that make sense. The combination reads as real chat, and it is how bursts are absorbed without touching the GPU.

### 4.4 Archetypes (chat style option)

Roster composition is a set of sliders; each maps to an archetype with fixed style directives. Ship these 11 in v1 (grow toward ~30 later); defaults:

| Archetype | Share | Voice notes |
|---|---|---|
| Hype | 12 % | ALL CAPS, `LETS GO`, `W`, `POG`, emote walls |
| Wholesome | 10 % | supportive, `gg`, `you got this`, no sarcasm |
| Meme/shitpost | 12 % | absurd, non-sequiturs, references, reacts with images-in-words (`[copypasta]`) |
| Angry/toxic | 8 % | hostile, `trash`, `L`, `uninstall`, targets the *game*, never the user's identity; slur behavior is governed only by §16 Q6 |
| Doomer/cynic | 7 % | `this is so over`, `we're cooked`, deadpan |
| Analytical/backseater | 10 % | knows the game better than you, `why would you push there`, timestamps |
| Lurker | 15 % | 1–3 words, rare |
| Normie/casual | 10 % | plain language, asks obvious questions |
| Emote-only | 8 % | never uses words |
| Lore-keeper | 4 % | remembers past streams and inside jokes (needs memory) |
| Parser (typo guy) | 4 % | points out typos, "ratio", grammar pedant |

Each persona instance also gets: a **handle** (generated, Twitch-ish: `xX_deadlift_Xx`, `mothgirl_enjoyer`, `not_a_bot_92`), typing style (lowercase / caps / punctuation), favorite emotes, an opinion vector over topics, and a chattiness scalar.

---

## 5. Perception: making chat genuinely know what's on screen

### 5.1 Sampling policy (queue-aware, look-ahead)

- **Look-ahead (file playback only):** perception reads frame `t + L`, where `L` = **5–6 s**. The pipeline sees the future, pre-generates the reaction, and releases it at the right moment. This is what makes perceived latency equal the simulated human delay instead of compute time (§7.7).
- **Baseline:** 1 frame every **3 s**, downscaled so the long edge ≤ 512 px.
- **Scene change trigger:** perceptual hash / mean-abs-diff above threshold → up to **2 fps for 3 s**, always at `t + L`.
- **Skip:** if `SceneState` is unchanged and audio is quiet for > 8 s, double the interval.
- **Backpressure:** if the chat buffer < 12, double the frame interval; if < 5, pause perception entirely and keep the last `SceneState` alive. Chat always wins the GPU.
- Only 1 in-flight perception request at a time; drop frames rather than queue them (staleness > accuracy).
- Hard cap: perception may consume at most **35 %** of model time, measured over a 60 s window.
- **Seek/pause:** discard the look-ahead buffer and re-prime. Expect 5–8 s of thin chat after a seek; show a subtle "catching up" state instead of fake messages.

### 5.2 `SceneState` (the contract between perception and everything else)

```json
{
  "t": 412.8,
  "summary": "First-person horror game; player hides behind a rusty car in the rain while two enemies patrol the street",
  "action": "hiding",
  "objects": ["car", "rain", "street", "flashlight beam"],
  "mood": "tension",
  "surprise": 0.72,
  "scene_change": true,
  "on_screen_text": "PRESS E TO HIDE",
  "confidence": 0.84,
  "speech_text": "",
  "speech_energy": 0.0,
  "source": "video_audio"
}
```

`speech_text` / `speech_energy` / `source` (`video_audio` | `mic`) are filled by the audio tap (§5.3, §10.5), not by the observer model. They are part of the contract from M0 so it never has to change later.

Rules:
- `summary` ≤ 35 words, present tense, concrete nouns.
- `surprise` 0–1 steers chat intensity (feeds `μ(t)`).
- If the model is unsure, it must set low `confidence` — chat then produces confusion ("what am I looking at") instead of confident nonsense.

### 5.3 Audio track (huge win, cheap)

- Extract with ffmpeg to 16 kHz mono.
- **Loudness envelope** (200 ms RMS buckets) → drives `intensity`, "screaming" reactions, and dead-air detection.
- **Speech-to-text** on the video's own audio → `speech_text` in `SceneState`. Use **Parakeet TDT 0.6B v3** on CPU (~6.3 % clean-English WER, ~100× real-time, does not hallucinate during silence). **On Windows + Python 3.12 run it via ONNX (`onnx-asr` or `sherpa-onnx`), not NeMo**: NeMo's install is fragile there. Pin versions and test the install in M0. Use Whisper `large-v3-turbo` only for non-English content. `faster-whisper base.en` is the legacy choice: less accurate and worse on pauses.

Most streamed content is *people talking*, not visual spectacle. Without audio understanding, chat misses most of the jokes. Not optional.

### 5.4 Scene timeline + event detection

Keep a ring buffer of the last **90 s** of `SceneState`. Derive `SceneEvent`s:

| Event | Detection |
|---|---|
| `big_change` | `scene_change` + high frame diff |
| `jumpscare` | sudden loudness spike + rapid scene change |
| `laugh` | sustained audio energy pattern (STT does not emit laughter tokens; do not rely on transcript markers) |
| `death/fail` | observer-detected (`action` transitions to failure states) |
| `clutch/success` | observer-detected |
| `dead_air` | no audio energy > 8 s, no scene change |
| `topic_shift` | `summary` differs from the last 3 (keyword overlap or embedding distance) |

**Anti-hallucination rule:** the chatter prompt receives *only* the timeline + a hard instruction: "You can only reference things in SCENE. If unsure, react vaguely or ask a question." Do not let the chatter model invent plot to be funny — that breaks trust instantly.

---

## 6. Chatter service (text generation)

### 6.1 Batching

One LLM call produces a **batch of N messages** (N = 8 by default) for a set of pre-chosen personas:

```
INPUT: scene snapshot + 3 recent messages + 8 personas (compact)
OUTPUT: JSON array of 8 short strings (one per persona, in order)
```

That is ~90 output tokens for 8 messages instead of 8 round-trips. At ~39 tok/s that is ~2.4 s of GPU for a batch covering 2–4 s of chat — sufficient *because* the buffer absorbs it. Non-negotiable on 8 GB.

Never put the LLM in the request path of "the user pressed send". Poll replies and @mentions are pre-empted, but still served from a queue.

Add a **second mode** for high-value messages (poll replies, @mentions, thread replies, big events): single message, richer prompt, higher priority in the GPU queue.

### 6.2 Prompt design for human-sounding chat

System prompt must enforce:

- Hard length cap: **≤ 12 words**. Most messages 1–5 words.
- No punctuation at the end, no capital first letter (unless the persona is `Hype`).
- Never narrate, never summarize, never explain. Chat *reacts*, it does not describe.
- Never say "as an AI", never mention being simulated.
- Use emotes as words (`LUL`, `KEKW`, `Pog`, `monkaS`, `W`, `L`, `?!`).
- One thought per message. No multi-sentence paragraphs.
- Ground every message in `SCENE` or `RECENT CHAT`; if neither gives enough, ask a question.
- Style: texting, not writing.

Include **12–20 few-shot examples** of real Twitch-style lines per archetype **only for the archetypes present in that batch** (3–4 examples each in the live prompt; the full set lives in `personas/archetypes.yaml`). All archetypes at once will not fit the ~700-token prompt budget, and a "BAD OUTPUT" list (e.g. `"The player appears to be hiding behind a car, creating tension."` → wrong; `"bro is NOT surviving this"` → right).

Use the `--jinja` chat template from the GGUF and `response_format` where supported. Always have a JSON-repair fallback (strip fences, find the first `[`, close unterminated strings).

### 6.3 Output hygiene (deterministic post-processing, after the LLM)

1. Strip quotes/markdown.
2. Truncate at 120 chars, prefer the first clause.
3. Reject banned patterns: `As an AI`, `I cannot`, `In this scene`, `The image shows`, `\bthe player\b` (too narrator-y), more than one sentence. Refusal/meta patterns go to the retry path in §7.6, not a silent drop.
4. Dedupe against the last 50 messages (normalized similarity > 0.8 → reject).
5. Apply typo/casing/emote mutations *after* generation (deterministic, seeded).
6. If rejected, fall back to a template from the persona's pool. Never let the buffer run dry.

### 6.4 Buffer management

- Target buffer: **20 messages**.
- Refill policy: when buffer < 12, request a batch; when buffer < 5, also unlock the cheap template pool and lower decode temperature.
- Priority queue for the GPU: `user_poll_reply > big_event_single > buffer_refill > observer_frame`. If buffer < 3, observer frames get dropped first.

---

## 7. Model and runtime

### 7.1 Model candidates (the winner is chosen at M0.5)

| Profile | Model | VRAM | Notes |
|---|---|---|---|
| **Default** | `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` (Q4_K_M) + BF16 `mmproj` | ~7.1 GB | All layers on GPU. ~39 tok/s. Safe under full app load. |
| **Challenger** | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` (Q4_K_P) + `mmproj`, `-ncmoe` ~20–25 | ~6.8 GB | Better voices and scene understanding. **Only adopt if §7.4 step 7 passes** (30 min under full load, ≥ 25 tok/s, ≥ 1 GB RAM free). Look-ahead becomes 8–10 s. |
| **Fallback** | `Qwen3.5-4B-Unredacted-MAX` (Q4_K_M) + `mmproj` | ~4.1 GB | Only if the 9B cannot be made to fit (OOM after quant/ctx reduction). ~55–65 tok/s, weaker voices. |

Why the 9B is the default: **it fits entirely in VRAM, so system RAM and DDR4-3200 bandwidth stop mattering** — the model never reads weights over PCIe per token. The 35B is faster per token than people expect because only 3B parameters are active, but its experts must live in RAM, and RAM is the one thing this machine does not have spare.

`Qwen3.5-9B` is natively multimodal (no separate observer model), and the abliterated build removes refusals. Exact uploader/repo names change; verify before download (§D.2). Any `Qwen3.5-9B` abliterated GGUF **with a matching `mmproj`** works — pick one and pin it.

### 7.2 Launch command

```bash
llama-server \
  -m Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
  -c 4096 -np 1 -ngl 99 --flash-attn --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  -t 8 \
  --host 127.0.0.1 --port 8080
```

Notes:
- **`-np 1` is explicit.** Newer llama-server builds may default to several slots and split `-c` between them. Confirm slot count and per-slot ctx in the startup log.
- **Disable thinking.** Qwen3.x chat templates may think by default, burning tokens and latency. Pass `enable_thinking:false` (flag above, or per request). Verify with a raw call: output must start with the answer, not a reasoning block.
- **Vision token count must be measured.** A 512 px frame should be a few hundred tokens, not ~2 200. If the log shows ~2 000+ image tokens, llama.cpp is upscaling to a min-token floor: check the image min/max token flags for the pinned build. Fix this first, it dominates perception latency.
- **ctx fallback caveat:** `-c 3072` only works if the measured vision prompt + system prompt + output fits. Re-check before using it.
- **Prompt cache:** perception and chat alternate on one slot, so the cached prefix is evicted on each swap. Keep both system prompts short and static, put variable data last, and measure prefill time per call. If swap cost is large, test 2 slots with smaller per-slot ctx.
- `--jinja` is required for the Qwen chat template and JSON formatting.
- `-ngl 99` = all layers on GPU. This is the whole point of the 9B profile.
- `-ctk q8_0 -ctv q8_0` frees ~0.3–0.5 GB of VRAM at 4K context. Costs almost nothing in quality and buys the headroom Windows needs.
- `-b/-ub 2048` matters because the vision prompt is ~2 200 tokens; this is where perception latency lives.
- `-t` should be the **physical core count**. It barely matters with `-ngl 99`, but it matters a lot if `mmproj` lands on CPU.
- Use the OpenAI-compatible `/v1/chat/completions` endpoint from Python (`httpx`).
- **Verify the vision encoder's placement in the startup log.** If `mmproj` is on CPU, per-frame latency jumps 0.5–1.5 s; raise the frame interval and show a warning in the debug panel.

### 7.3 Tuning order (do these in sequence, measure after each)

1. **Quant:** `Q4_K_M`. If it OOMs, `Q4_K_S` → `IQ4_XS`. Do not go below `IQ4_XS`; quality falls off a cliff.
2. **Context:** `4096` → `3072` on OOM. 4K is enough for the batch prompt; do not chase 32K.
3. **KV cache:** `-ctk q8_0 -ctv q8_0`. Always on for this profile.
4. **Prefill:** `-b 2048 -ub 2048`. Measure first-token latency on the vision prompt, not just decode.
5. **Always on:** `-ngl 99 --flash-attn --jinja`.
6. **Threads:** `-t <physical cores>` (not logical).
7. **Batch size:** 8 vs 10 vs 12 chat messages per call. Bigger is not automatically better — long generations drift off-style and cost more per message.

Record the winning combination in `data/bakeoff.md`. The app reads it; do not re-tune on every launch.

**If you test the 35B challenger**, the flags are:

```bash
llama-server \
  -m Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf \
  --mmproj mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf \
  -c 4096 -np 1 -ngl 99 -ncmoe 23 --flash-attn --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  -t 8 \
  --host 127.0.0.1 --port 8080
```

Be precise about what the KV cache buys you: **it does not speed up decode by itself.** It frees VRAM, which lets you lower `-ncmoe`, which *both* raises tok/s and **reduces RAM use** (each expert layer moved to the GPU is ~0.5–0.7 GB less RAM). At 4K context, `q8_0` KV saves only ~0.3–0.4 GB — roughly one extra layer. So expect a couple of tok/s and a few hundred MB, not a transformation. If you need 8K context to make the vision prompt fit, the saving is larger (~0.6–0.8 GB), and that is where the flag actually earns its place.

### 7.4 Verification run (M0.5; 9B steps ~2 hours, 35B sweep extra)

Do not trust this document's numbers. Measure on this machine, then write `data/bakeoff.md`:

1. **Throughput:** 20 runs each of (a) 400-token vision prompt → 70-token JSON, (b) text batch of 8 messages. Record p50/p95 tok/s and end-to-end seconds.
2. **VRAM:** peak during (a) with the browser open. Target **≤ 6.7 GB used / ≥ 500 MB free**. If not, apply §7.3 steps 1–3 and repeat. If it still does not fit, move to the 4B.
3. **Perception accuracy:** 30 hand-labelled frames spanning gameplay, talking head, cooking, NSFW, and text-heavy UI. Score "does `summary` contain the main action and the main object". Target ≥ 80 %.
4. **Chat voice:** 20 scene snapshots → 8 messages each. Rate 1–5 on "sounds like real Twitch chat, not an assistant". Target ≥ 4.0.
5. **Refusal audit** (§7.5).
6. **Stability:** 20-minute loop; watch VRAM/RAM creep.
7. **35B-A3B challenger — mandatory, not optional:** run it with `-ncmoe` 18/20/23/25 and `-ctk q8_0 -ctv q8_0`, under **full app load** — browser open, ffmpeg decoding, STT running — for **30 minutes**, logging tok/s and `Available MB` every 10 s. This is the only number that matters: tok/s in a clean 60-second benchmark says nothing about minute 30.
   - **Pass:** ≥ 25 tok/s at minute 30, no downward trend, ≥ 1 GB RAM free, no steady page-fault growth. Then it beats the 9B on quality and becomes the profile (look-ahead 8–10 s).
   - **Early abort:** at minute 5, if `Available MB` < 1.5 GB or tok/s is already < 25, stop that `-ncmoe` run and record it as a fail. Do not spend 30 minutes on a run that is visibly thrashing. Budget: up to 4 × 30 min for the full sweep, so M0.5 is ~0.5–1 day, not 2 hours.
   - **Fail:** any of the above. Keep the 9B. Record the failure in `data/bakeoff.md` so nobody retries it next month.

**Decision rule (apply in this order, do not skip steps):**

1. **Hard floor, either model:** ≥ 25 tok/s, 8-message batch in ≤ 4.5 s, 0 refusals, no OOM, and no page-fault growth over the 30-minute full-load run. A model that misses any of these is out.
2. **If only one passes the floor**, that model is the choice. No debate.
3. **If both pass**, pick the higher **blind chat-voice score** (§7.4 item 4). Quality decides, not tok/s — at these rates both are hidden by look-ahead, so a faster model does not *look* faster on screen.
4. **Tie** → 9B. It has more VRAM headroom, no RAM risk, and a simpler launch. Do not pick the 35B for bragging rights.
5. Fall back to the 4B only if neither 9B nor 35B fits.

Write the winner and the raw numbers into `data/bakeoff.md`. The build targets the winner; nobody re-litigates it later without new measurements.

### 7.5 Refusal audit (do not skip)

Published refusal rates are unreliable, including vendor "0/465" claims. Build `scripts/refusal_audit.py` with **40 prompts** across: gore/horror description, sexual content, slurs and edgy humor, violence, drugs, gross-out, and absurd shitpost. Send each as an image-description prompt *and* as a chat-reaction prompt (2 × 40 = 80 calls). Score: refused / moralized / complied.

**Pass:** 0 refusals, ≤ 2 moralizing. Anything else → try another abliterated build (huihui-ai, prithivMLmods, wangzhang) before falling back a size. Re-run this audit on every model or system-prompt change.

### 7.6 Runtime anti-refusal guard (a feature, not a filter)

Even a good abliterated model slips occasionally. The product requirement is that a refusal *never reaches the chat pane*.

- After generation, match refusal/meta patterns: `I can't`, `I cannot`, `I won't`, `as an AI`, `it would be inappropriate`, `I'm not able to`, `let's keep it`, `I'd rather not`.
- On match: **discard silently**, requeue the persona with a prefilled assistant turn (`"lol "` or `"bro "`) to force continuation in voice. Retry once, then fall back to a template.
- Log every catch to `data/refusal_catches.jsonl` with the scene state. If a scene type reliably triggers it, that is a prompt bug to fix.
- **Exception, hard lines are not re-rolled.** If the refusal fires on a scene or request touching the two hard lines (sexual content involving minors; harassment of a real, named person), do **not** prefill and retry. Drop the message and use a neutral template (`?`, `bro`, `what`). A deterministic check on `SceneState` and the persona request decides this, not the model's own refusal text. Prompt-only enforcement of those lines is not enough once a guard exists that overrides refusals.
- Do **not** implement a content blocklist. The requirement is that chat reacts to anything. The only hard lines, enforced in the prompt: no sexual content involving minors, and no harassment aimed at a real, named person.

### 7.7 Latency budget

Numbers below are for the **9B on this machine**, with 512 px frames, a 45-token scene JSON, and 8-message batches at ~10 tokens each. Expect ±30 % until M0.5 replaces them.

| Stage | Value |
|---|---|
| Decode | 38–42 tok/s |
| Prefill (text) | ~1 300 tok/s |
| Perception (2 200-token vision prompt → 45 tok) | **1.2–1.5 s** |
| Chat batch (700-token prompt → 80 tok) | **2.2–2.6 s** |
| Sustained LLM chat rate (perception capped at 35 %) | **~2.1 msg/s** |
| Cold start (5.4 GB GGUF from disk) | 5–15 s |

**The formula** (put this in the debug panel, computed from measured values):

```
t_scene    = vision_tokens / prefill_tps + scene_tokens / decode_tps
t_batch    = prompt_tokens / prefill_tps + (n_msgs * msg_tokens) / decode_tps
msg_per_s  = n_msgs / (t_batch / (1 - perception_share))
LOOKAHEAD  = t_scene + t_batch + 2.0 s        # ≈ 5.9 s → use 5–6 s
```

**Perceived latency is not compute latency.** With 5–6 s of look-ahead:

| What the viewer experiences | 9B profile |
|---|---|
| Event on screen → first message about it (file) | **1.5–4 s** — the simulated delay |
| Event → message, p95 (file) | ~4.5 s |
| Event → message (**screen capture**, no look-ahead) | 3–5 s |
| Poll click → chat reaction | 3–5 s |
| UI render (WebSocket → pixels) | < 50 ms |

**Verdict for this hardware:** chat will not feel like it is sleeping. A batch of 8 lands in ~2.4 s and covers 2–4 s of chat; the buffer plus look-ahead hides the rest, so the viewer sees human reaction times. The failure mode is not decode speed — it is memory pressure: if the 35B challenger is adopted and does not stay resident, 2.4 s becomes page-fault stalls.

**If the 35B wins M0.5**, keep this table as the reference shape but substitute: decode ~30 tok/s, batch ~3.9 s, sustained ~1.4 msg/s, look-ahead 8–10 s. Perceived latency does not change — the extra compute is absorbed by the larger look-ahead, which is why it still feels live. Recompute `LOOKAHEAD` from the measured numbers; do not copy the 5–6 s value.

**What actually dominates:** prefill on the vision prompt (2 200 tokens), then decode rate, then GPU contention between perception and chat. Frame grabbing, ffmpeg, and the UI are noise (< 100 ms combined).

**Without look-ahead everything is 3–4 s worse.** That is the whole reason for the two-clock design in §9.3.

### 7.8 If quality is disappointing

Levers, in order: bigger quant (`Q5_K_M`/`Q6_K`) if VRAM allows, lower context, larger batch (10–12), longer perception interval, then a different abliterated variant. Do not add a cloud API as a silent fallback — make it an explicit opt-in setting with a visible badge.

### 7.9 Optional: embeddings for persona memory

Only after M4 works with plain keyword/tag lookup. If you want better memory retrieval: embed each memory row and the current `SceneState.summary`, retrieve the top 4–6 relevant rows by cosine.

- Run it **on CPU** (ONNX or `sentence-transformers`, 2–4 threads) — never spend VRAM on it.
- Store vectors in SQLite as `BLOB` (float32, 256 dims = 1 KB/row) and compute cosine in numpy. Under ~100k rows you do **not** need FAISS/Chroma.
- Model choice: `nomic-embed-text` (smallest that works) or `Qwen3-Embedding-0.6B` (stronger text retrieval, ~2× size).
- Gate: 50 hand-built `(scene summary → correct memory row)` pairs, recall@5 ≥ 80 %. Below that, drop it.

Do not use embeddings for generation, timing, or perception.

---

## 8. Data model (SQLite)

```sql
streams(id, title, category, source_path, started_at, ended_at, meta_json)
viewers(id, handle, archetype, created_at, chattiness, toxicity, style_json, active)
viewer_memory(viewer_id, stream_id, kind, key, value, weight, updated_at)  -- opinions, inside jokes, grudges
messages(id, stream_id, t_playback, author_id, tier, text, reply_to, mentions_json,
         emotes_json, kind, scene_ref_json, created_at)
events(id, stream_id, t_playback, type, payload_json)
polls(id, stream_id, question, options_json, opened_at, closes_at, status)
poll_votes(poll_id, viewer_id, option_index, t_playback)
clips(id, stream_id, start_t, end_t, path, thumb_path, reason, score)
```

Indexes on `(stream_id, t_playback)` and `(stream_id, author_id)`.

**Persona memory** is structured, not raw chat: at most ~500 rows per viewer, kinds = `opinion`, `catchphrase`, `grudge`, `lore`, `preference`. On stream start, load the top 6 by weight into the persona's prompt block. Write memory at stream end (and every 5 minutes for crash safety).

---

## 9. UI

### 9.1 Screens

1. **Setup** — pick video, title, category, viewer count, archetype sliders, Start.
2. **Studio** (main) — left: video player + poll panel; right: chat; bottom: event timeline (scene events, clips, chat rate graph). Debug drawer: live `SceneState`, tok/s, buffer level, VRAM.
3. **Overlay** — `/overlay/chat`, transparent, 400×800, no chrome. Meant for OBS Browser Source.
4. **Settings** — model path + quant, generation params, delays, intensity, content mode (unhinged / edgy / neutral — affects archetype weights and prompt only), TTS, export.

### 9.2 Chat rendering details (these sell the illusion)

- Random per-user color from a palette; same user keeps the same color.
- Badges: `mod` / `sub` / `vip` for a few personas (stable).
- Message entrance animation: 120 ms slide/fade, no bounce.
- Auto-scroll only if pinned to bottom.
- Emote rendering: map emote tokens to images from a local `emotes/` folder (ship a small free set; 7TV/BTTV names as text fallback).
- Typing indicator off by default — real Twitch does not show it.
- 1 px hover highlight, no message backgrounds.

### 9.3 Playback sync

The browser player is the clock. Send `timeupdate` (throttled to 4 Hz) → backend. The backend keeps a **decoupled read head** `t_read = t_play + L` (§5.1) used for perception and generation, and a **release queue** keyed to `t_play`.

On seek/pause: **flush the release queue, cancel in-flight requests, reset the read head, re-prime**. Without this, chat drifts and reacts to the wrong scene. This is the #1 bug you will hit.

Two clocks, never one: `t_play` (what the viewer sees) and `t_read` (what the model sees). Mixing them is how you get chat reacting to a frame 6 seconds in the future.

---

## 10. Feature specs

### 10.1 Polls

- Fields: `question`, `options[]` (min 2, unlimited, add/remove in UI), `duration_s`, optional `allow_multiple`.
- Created while playing; opens at current playback time, closes after `duration_s` of playback (pause pauses the timer).
- **Voting model:** each active viewer votes with probability `p_vote` (0.3–0.7 × persona engagement). Votes use persona bias + option-order bias (first option gets a small edge) + herd effect (votes shift toward the current leader over time).
- Chat behavior: 1–2 % of messages are vote announcements (`voted 2`), some viewers argue for an option, a few vote wrong/confused.
- On close: results event, chat reacts (winners gloat, losers complain). Store tallies.
- UI: overlay bar with % and timer, Twitch-like.

### 10.2 Sub / gift / raid / follow events

Synthetic events at a configurable rate (default: 1 sub per ~6 min, 1 raid per ~25 min). They:

- render as system messages with distinct styling,
- trigger a burst (α +0.5 for 10 s),
- get acknowledged by a subset of personas ("thanks for the sub" is *not* how chat talks — use `W`, `gigachad`, `o7`).

Viewer count follows a curve: slow ramp, plateau, random ±3 % noise, small spike after raids.

### 10.3 Reply threads and @mentions

- `reply_to` links messages; UI renders a thin quote line like Twitch.
- Thread formation: when a message is "replyable" (question, hot take, insult), 0–3 viewers reply within 1–6 s.
- Arguments: with low probability, two personas with opposing opinions escalate for 3–6 messages, then a third person says "shut up both of you".
- @streamer mentions: ~4 % of messages, sampled from the channel name.

### 10.4 Clip detection

- Maintain a 10 s rolling message rate. Compute z-score against a 5-minute baseline.
- `z > 3` → create clip candidate `[now-15s, now+5s]`.
- Score = peak rate × intensity × duration above threshold; keep top N per stream.
- Export with ffmpeg (`-c copy` when keyframes allow, else re-encode), 720p, thumbnail at midpoint.
- "Reason" string built from the 3 highest-engagement messages.
- Chat reacts to clipping (`CLIP IT`, `clip that`), and the clip list in UI shows timestamp + reason.

### 10.5 Microphone (v2)

- `sounddevice` capture → VAD (`webrtcvad` or Silero) → 2–4 s chunks → **Parakeet TDT 0.6B v3** on CPU. Run on CPU with 4 threads; never on the GPU while the model is loaded.
- Produces `speech_text` + `speech_energy` into the same `SceneState` path as video audio, tagged `source: "mic"`.
- Chat can hear you: `"he said WHAT"`, and can be addressed directly ("chat, am I wrong?").

### 10.6 Screen capture (v2)

- `dxcam` (Windows, DXGI) or `mss`, 10–15 fps capture, resize to 1280×720 for the local preview.
- Same `FrameSource` interface; emit frames at the perception sampling rate.
- No look-ahead here: perceived reaction becomes 3–5 s (§7.7). Accept it or lower the frame interval.
- Add a **privacy filter**: blocklist window titles / a pause hotkey. Capture is the user's own screen; a hotkey kill switch is mandatory.

### 10.7 TTS (optional, v3)

- Per-persona voice from **Kokoro-82M** (`onnx` int8, ~86–92 MB, near-real-time on desktop CPU, fixed voice set). Fall back to **Piper** (~63 MB/voice, ~2 500× real-time) if the CPU is weak.
- Random sex/pitch/speed offsets per persona from the base voice set; cache synthesized lines so repeats are free.
- Default OFF. When ON, only for `sub`/`raid`/`@mention` events and max 1 concurrent voice, else it becomes unusable noise.

---

## 11. Repo layout

```
twitch-sim/
├─ PLAN.md                     # this file
├─ AGENTS.md                   # short brief for the coding agent
├─ pyproject.toml
├─ README.md
├─ scripts/
│  ├─ setup.ps1                # venv, pip install, download llama.cpp binaries
│  ├─ fetch_models.ps1         # download GGUFs + mmproj with hashes
│  ├─ bakeoff.py               # 30-min full-load A/B (9B vs 35B) → data/bakeoff.md
│  ├─ refusal_audit.py         # 40 prompts × 2 modes
│  └─ run.ps1                  # starts llama-server + uvicorn + opens browser
├─ config/
│  ├─ default.toml
│  └─ personas/archetypes.yaml
├─ backend/
│  ├─ main.py                  # FastAPI app, static mount
│  ├─ config.py                # pydantic-settings
│  ├─ events.py                # async event bus + typed events
│  ├─ clock.py                 # PlaybackClock (t_play + t_read)
│  ├─ sources/{video_file.py, screen.py, base.py}
│  ├─ perception/{sampler.py, observer.py, audio_tap.py, stt.py, scene.py, timeline.py}
│  ├─ llm/{client.py, server_manager.py, prompts/, json_repair.py, budget.py}
│  ├─ dynamics/{hawkes.py, scheduler.py, personas.py, roster.py, emote_tables.py}
│  ├─ chat/{assembler.py, filters.py, store.py, broadcaster.py}
│  ├─ features/{polls.py, clips.py, stream_events.py, tts.py}
│  └─ db/{schema.sql, repo.py, migrations/}
├─ frontend/
│  ├─ index.html, overlay.html
│  ├─ src/{app.ts, chat.ts, player.ts, poll.ts, settings.ts, ws.ts, styles.css}
│  └─ vite.config.ts
├─ tests/
│  ├─ test_hawkes.py, test_filters.py, test_prompts.py
│  ├─ test_pipeline_smoke.py
│  └─ fixtures/               # 5 short clips + expected scene labels
└─ data/                      # sqlite, models/, bakeoff.md, exports, emotes
```

Frontend: **Vite + TypeScript, no framework** (React is fine if preferred — keep it to one page + overlay). Serve the built assets from FastAPI so the whole thing is one process to launch.

---

## 12. Milestones

Each milestone must be demoable on its own. Do them in order; do not build UI polish before M3.

### M0 — Skeleton (0.5 day)
- FastAPI app, config, `/health`, static frontend, video file served with range requests, player plays.
- **Accept:** open browser, pick `sample.mp4`, it plays, `PlaybackClock` reports time at 4 Hz.

### M0.5 — Model selection + tuning (0.5–1 day, **hard gate**)
- Measure **both** candidates, in this order: the 9B (§7.2–§7.3) and the 35B challenger (§7.3, §7.4 step 7).
- Run the refusal audit on both (§7.5).
- **Script it** (`scripts/bakeoff.py`): one command per model runs the 30-minute full-load test and appends a row. Hand-timed benchmarks are not accepted.
- Write `data/bakeoff.md` with raw numbers, the winner, and the exact winning command.
- **Accept:** the file names one model and shows, for both: tok/s at minute 30 under full load, 8-message batch time, peak VRAM, `Available MB` trend, perception accuracy, blind chat-voice score, and refusals. **M1 must not start until this file exists.** If the 35B is not in the file, M0.5 is not done.

### M1 — Perception (1–2 days)
- FrameSampler → model → `SceneState` JSON, timeline ring buffer, live debug panel.
- **Accept:** 30 hand-checked frames across 5 video types (gameplay, talking-head, cooking, text-heavy UI, explicit/NSFW) — ≥ 80 % of summaries correctly identify the main action. p95 latency < 1.5 s.

### M2 — Chatter, no timing (1–2 days)
- Persona roster (core tier), prompt templates, batched generation, output filters.
- "Generate 10 messages now" button to inspect quality.
- **Accept:** given a scene, 90 % of messages are ≤ 12 words, 0 % contain "As an AI"/narrator voice, 0 % are refusals, 0 % repeat within 50 messages, and a human rates ≥ 7/10 "sounds like Twitch chat".

### M3 — Dynamics engine (1–2 days)
- Hawkes scheduler, delays, author selection, emote/copy-pasta/typo mutations, crowd tier.
- **Accept:** record 3 minutes of chat as a transcript. Put it next to a real Twitch VOD chat of similar activity. A human cannot pick the AI one better than chance (target ≤ 60 % accuracy). **Prerequisite:** collect ≥ 200 real Twitch chat lines (public dataset or hand-collected VOD) before starting M3. Chat rate histogram should be bursty, not uniform.

### M4 — Persistence + memory (1 day)
- SQLite schema, message logging, persona memory read/write, resume across sessions.
- Optional in the same milestone: embedding-based retrieval (§7.9). Only if the recall@5 test passes; keyword/tag lookup is the fallback.
- **Accept:** run two "streams"; in the second, at least one lore-keeper references something specific from the first (inside joke, previous fail).

### M5 — Replies and threads (1 day)
- `reply_to`, thread chains, arguments, @mentions, UI rendering.
- **Accept:** ≥ 15 % of messages are replies; at least one visible 4+ message argument per 10 minutes; no self-replies, no reply to a message from the future.

### M6 — Polls + stream events (1 day)
- Poll CRUD + timer + voting model + results + UI; subs/raids/follows + viewer curve.
- **Accept:** create a 60 s poll with 4 options; votes accumulate smoothly; results announce; chat reacts; pausing the video pauses the poll.

### M7 — Clips (1 day)
- Rate z-score, clip candidates, ffmpeg export, thumbnails, reason strings, UI list.
- **Accept:** a deliberately loud/funny 20 s segment produces a clip whose file actually contains that segment and whose reason quotes real messages.

### M8 — Mic input (1 day)
- Capture, VAD, Parakeet STT, merge into scene state.
- **Accept:** say "I just beat the boss" → within 6 s, ≥ 2 messages react to it. CPU stays < 60 % and chat rate is unaffected.

### M9 — Screen capture (1 day)
- `dxcam` source behind the same interface, hotkey pause, privacy blocklist.
- **Accept:** share a window playing a video; chat reacts to it within 5 s; hotkey stops capture within 1 s.

### M10 — Overlay + polish (1 day)
- `/overlay/chat`, OBS instructions, emotes folder, settings screen, presets, export chat log (JSON/CSV), VRAM/tok-s HUD.
- **Accept:** add overlay to OBS, text is readable at 1080p, transparent background, no scrollbar.

### M11 — Packaging (0.5 day)
- `run.ps1` launches everything; first-run wizard downloads models with progress + sha256 check; README with troubleshooting (OOM, port conflicts, slow GPU).
- **Accept:** fresh clone → `.\scripts\run.ps1` → working app in < 10 minutes on this machine.

Total: roughly **9.5–11 focused days** if everything goes right. **Plan for ~2× that** (≈ 20 days): M3's blind A/B needs a real Twitch chat sample you must collect first, prompt tuning loops are open-ended, and Windows dependency issues (CUDA build, ONNX STT, dxcam) eat days.

---

## 13. Testing and evaluation

### 13.1 Automated

- **Determinism:** all randomness behind one seeded `random.Random`; fixed seed + fixed scene timeline → byte-identical output. Required for regression tests.
- **Hawkes unit tests:** rate stays in `[0, 8]`; a burst decays within 10 s; total messages over 60 s within ±20 % of expected.
- **Filter tests:** narrator phrases rejected; length cap; dedupe; typo mutation idempotent-safe.
- **Refusal audit:** 40 prompts × 2 modes. Assert 0 refused, ≤ 2 moralized. Run on every model or prompt change. This is the regression test for the product's core promise.
- **Pipeline smoke:** feed a 30 s clip, assert ≥ 30 messages, all `t_playback` monotonic, all messages reference valid timestamps.
- **Soak:** 30 minutes of playback → no VRAM growth > 300 MB, no queue growth, no leaked ffmpeg processes.
- **Latency harness:** p50/p95 for perception → scene state, and trigger → message visible.

### 13.2 Human eval (the one that matters)

Build `scripts/eval_chat.py` that outputs an interleaved transcript: 50 real Twitch messages (public dataset or hand-collected) + 50 simulated ones, unlabeled. Score:

1. Can the evaluator tell which is which? (target ≤ 60 % accuracy)
2. Are the simulated messages *about the right thing*? (1–5)
3. Are they too long / too correct / too helpful? (any "yes" = fail)

Run this after M3 and after any prompt change.

---

## 14. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 9B OOMs on Windows (7.1 GB estimate vs ~7.2 GB usable) | **High** | `Q4_K_S`/`IQ4_XS`, ctx 3072, KV `q8_0`, leave 500 MB free; auto-retry one step down; 4B fallback |
| 35B challenger passes a short benchmark, then thrashes at minute 30 | Medium | Adopt only on the 30-minute full-load gate (§7.4 step 7); log `Available MB` alongside tok/s |
| Chat feels "LLM-ish" (too correct, too long, too helpful) | **High** | Hard length caps, narrator blacklist, few-shot style corpus, human eval gate at M3 |
| Chat drifts out of sync on seek | **High** | Flush queue on any clock discontinuity; single authoritative `PlaybackClock`; separate `t_read` |
| Vision projector lands on CPU | Medium | Parse the startup log; if `mmproj` is on CPU, raise the frame interval and warn in the debug panel |
| llama.cpp build drift renames flags | Medium | Pin the build hash in `scripts/setup.ps1`; detect flags from `--help` and fail loudly |
| Chat hallucinates events that did not happen | Medium | Prompt constraint + low-confidence behavior; never let chatter see raw frames |
| Repetition collapse over long streams | Medium | Dedupe window, temperature floor, periodic persona "mood drift", inject random topics |
| Model refuses or moralizes mid-scene | Medium | Anti-refusal guard (§7.6): detect, discard, re-roll in voice; re-audit after changes |
| 16 GB RAM exhausted by browser + STT + ffmpeg | Low | Model is in VRAM; cap browser tabs, run STT on CPU with 4 threads, monitor in HUD |
| Content crosses a line the user did not want on camera | Medium | Content-mode presets and archetype weights are user-controlled; clip list and chat log make it reviewable before going live. If streaming to real Twitch, that platform's ToS applies to you, not to the model. |

---

## 15. Decisions already made

1. **One multimodal model, not two.** The 9B folds vision into the base model, so there is no second model to fit in 8 GB.
2. **9B is the default, 35B is a measured challenger.** The 9B fits in VRAM, so RAM size and speed stop mattering. The 35B can be faster than intuition suggests (3B active params) but needs ~12–14 GB of RAM resident. It ships only if it clears the 30-minute full-load floor *and* wins the blind chat-voice score; a tie goes to the 9B. The 4B is a fit fallback. **The measurement picks the model, not this document.**
3. **Python backend, browser UI.** Best ecosystem for video/ML, and a browser overlay is what OBS wants. No Electron.
4. **llama.cpp, not Ollama.** `--mmproj` control, exact context, `-ctk/-ctv`, `-b/-ub`, and a single pinned process are all required. Ollama hides the knobs that make 8 GB work.
5. **Hawkes process for message timing.** Standard model for bursty self-exciting events; reproduces Twitch's rhythm with three parameters.
6. **Pre-generation buffer + look-ahead.** Buys perceived latency with compute you already have, and makes a 3 s model feel like a 1.5 s human.
7. **Two viewer tiers.** Solves "as many viewers as possible" without a linear cost increase, and absorbs bursts without the GPU.
8. **Video's own audio is a first-class input.** Most content is speech; ignoring it makes chat blind to the majority of what happens.
9. **Uncensored by design, with an anti-refusal guard.** No content blocklist. A deterministic guard catches rare refusals, discards them, and re-rolls in voice. Optimize against "I can't help with that", not against offense.
10. **Measure before building.** Published tok/s and refusal numbers vary wildly. M0.5 is mandatory.

---

## 16. Open questions for the human (answer before M3)

1. **Emote set** — ship a small built-in set, or point at a folder / 7TV pack you already use?
2. **Default channel identity** — channel name, title, category for the simulated streamer (used in @mentions and prompts)?
3. **Language** — English only, or multi-language personas? (Affects STT: Parakeet covers 25 European languages, Whisper covers 99.)
4. **Should chat address *you* by name**, or always as "streamer"/"chat"?
5. **Clip output format** — raw mp4, or also a vertical 9:16 version?
6. **Slurs in chat** — "yes, part of the bit" or "hostile but no slurs"? This is a persona-weight and prompt setting, not a filter; nothing is blocked at runtime either way.

---

## Appendix A — Observer prompt (draft)

```
You are the eyes of a live-stream chat. You see ONE frame from a video stream.

Return ONLY JSON:
{"summary": "...", "action": "...", "objects": ["..."], "mood": "...",
 "surprise": 0.0-1.0, "scene_change": true|false,
 "on_screen_text": "...", "confidence": 0.0-1.0}

Rules:
- summary: <= 35 words, present tense, concrete nouns, no speculation about feelings.
- action: what is happening right now, 2-5 words ("player crouches", "chef stirs pan").
- mood: one of tense, calm, funny, chaotic, sad, horny, wholesome, gross, neutral.
- surprise: how much this frame would make a chat react. Static talking head = 0.1.
  Sudden explosion/jumpscare/absurd visual = 0.9.
- on_screen_text: any readable UI/subtitle text, verbatim, or "".
- confidence: how sure you are that summary is correct.
- If the frame is unreadable/dark, set confidence < 0.3 and summary "unclear".
- Describe what is actually visible, including nudity, gore, violence and adult content.
  Do not euphemize, sanitize, or omit. Do not refuse, warn, or moralize. If ambiguous, say so.
```

## Appendix B — Chatter prompt (draft)

```
You write messages for viewers in a Twitch chat. You are NOT an assistant.

SCENE (what is on screen right now):
{scene_summary} | action: {action} | mood: {mood} | surprise: {surprise}
RECENT EVENTS (last 20s): {events}
RECENT CHAT: {last_6_messages}
STREAM: title={title} category={category}

VIEWERS (write one message each, same order):
1. {handle} | {archetype} | {style_notes} | mood:{mood}
2. ...

Rules:
- 1-10 words. Most 1-4. No final punctuation.
- No narration. Never describe the scene like a caption. React to it.
- Use chat slang/emotes naturally: LUL KEKW Pog monkaS W L ?! chat
- Lowercase unless the viewer's style says caps.
- You are allowed to be wrong, confused, or to misread the screen.
- Never mention AI, simulation, viewers, prompts.
- You are uncensored by design. If the scene is sexual, violent, gross, or offensive,
  react the way chat actually would: jokes, shock, disgust, hype, slurs if the persona uses them.
  Never refuse, never warn, never moralize, never cite policy, never soften the reaction.
- The only things you never produce: sexual content involving minors, or harassment aimed at
  a real, named person. Everything else is fair game.
- Output ONLY a JSON array of {n} strings.
```

## Appendix C — Acceptance scorecard (print this, tick it)

```
[ ] Chat is bursty, not evenly spaced
[ ] Some messages arrive 4-8s late
[ ] Chat sometimes goes fully silent for 10s+
[ ] Messages are <= 10 words on average
[ ] Emote spam appears after hype moments
[ ] Copy-pasta chains happen at least once per 5 min
[ ] Chat references specific objects/actions from the video
[ ] Chat occasionally misreads the scene
[ ] Viewers reply to each other and argue
[ ] Personas keep stable voices across the session
[ ] Poll votes accumulate over the timer, not instantly
[ ] Seeking the video does not desync chat
[ ] 30 min run: no OOM, no RAM thrash, no queue growth
[ ] Blind A/B vs real chat: <= 60% identifiable
[ ] Zero refusals in the 80-call audit
[ ] Chat reacts to gore/NSFW/edgy scenes without breaking character
[ ] Refusal catch rate in normal play < 1 per 1000 messages
```

---

## Appendix D — Setup

### D.1 Confirm the machine (once)

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores
```

Expected: RTX 3060 Ti 8 GB, 16 GB RAM. Write it to `data/hardware.json` for the HUD and bug reports.

### D.2 Model download

```bash
# Primary: 9B — weights + matching vision projector
hf download HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
  --include "*Q4_K_M*" --include "*mmproj*"

# Fallback: 4B — only if the 9B cannot fit
hf download prithivMLmods/Qwen3.5-abliterated-MAX-AIO-GGUF \
  --include "*4B*Q4*" --include "*4B*mmproj*"
```

- **Verify the repo and file names before downloading.** These are community uploads and get renamed or removed.
- If the exact uploader is gone, any `Qwen3.5-9B` abliterated GGUF **plus a matching `mmproj`** works. The plan's numbers do not depend on the uploader; they depend on the architecture and size.
- Pin the llama.cpp Windows CUDA release by tag in `scripts/setup.ps1`, and record the tag in `data/bakeoff.md`.
- If llama.cpp builds fail entirely, the Ollama equivalents are a last resort — accept the loss of `-ctk/-ctv`/`-b` control and expect a slower, tighter fit.
