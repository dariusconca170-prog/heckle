# AGENTS.md

Read `PLAN.md` in full before writing code. It is the spec.

**Project:** local "Twitch simulator" — a video plays, an AI chat reacts to it in near real time with human-like timing and personalities, uncensored by design.
**Target machine:** Windows, RTX 3060 Ti (8 GB VRAM), **16 GB DDR4-3200**, Python 3.12, one pinned llama.cpp server (`-np 1`, thinking disabled), SQLite, Vite/TS frontend.

## Model (selected by measurement at M0.5)

- **Default:** `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` (Q4_K_M) + matching `mmproj`. ~7.1 GB VRAM, ~39 tok/s, all layers on GPU.
- **Challenger:** `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` with `-ncmoe`. Faster than it looks (~30 tok/s in a clean bench) but needs ~12–14 GB of RAM resident. **Only adopt it if `PLAN.md` §7.4 step 7 passes** (abort a run early if RAM free < 1.5 GB or < 25 tok/s at minute 5) — 30 minutes under full app load, ≥ 25 tok/s at minute 30, ≥ 1 GB RAM free, no page-fault growth. If both models pass the floor, the higher blind chat-voice score wins; a tie goes to the 9B.
- **Fallback only if the 9B cannot fit:** `Qwen3.5-4B-Unredacted-MAX`.

The 9B is the default because it fits entirely in VRAM, so system RAM size and DDR4 speed are irrelevant to token rate. The 35B's constraint is RAM capacity under load, not tok/s. Measure it; do not assume either way — and **build whichever wins the measurement**, not whichever is written first here.

Run **M0.5 model selection + tuning** (`PLAN.md` §7.3–§7.5) and record real numbers in `data/bakeoff.md`. Do not build on an unmeasured guess.

## Build order

M0 skeleton → **M0.5 model selection** → M1 perception → M2 chatter → **M3 dynamics** → M4 persistence → M5 threads → M6 polls/events → M7 clips → M8 mic → M9 screen capture → M10 overlay → M11 packaging.
Each milestone has acceptance criteria in `PLAN.md` §12. Do not skip M3, and do not build UI polish before it.

**M0.5 is a hard gate.** Measure **both** the 9B and the 35B under full load, then write `data/bakeoff.md`. M1 does not start until that file names a winner. The 9B is the starting point, not the decision — if the 35B clears the floor and wins the blind voice score, build the 35B.

## Hard rules

1. **The LLM never schedules messages.** Timing lives in `backend/dynamics/`.
2. **Never generate one message per LLM call.** Batch 8–10. Pre-generate ahead of the playback clock.
3. **One model, two prompts.** Perception and chat share one server and a priority queue; chat always wins the GPU.
4. **Chat is grounded in `SceneState`**, but is allowed to be confused/wrong sometimes (8%).
5. **Empty chat is allowed.** Never pad silence.
6. **A refusal never reaches the UI.** Catch refusal/moralizing patterns, discard, re-roll with a prefilled in-voice turn. No content blocklist. **Exception:** never re-roll a refusal on the two hard lines (minors in sexual content, harassment of a real named person); drop and use a template (`PLAN.md` §7.6).
7. **Two clocks.** Perception reads ahead (`t_read = t_play + L`); messages release against `t_play`. `L` = 5–6 s for the 9B, 8–10 s if the 35B wins. Without look-ahead, file playback is 3–4 s late for no reason (`PLAN.md` §7.7, §9.3).
8. `sources/` (file vs screen) must be interchangeable behind one interface.
9. Every feature degrades gracefully when the GPU is busy: templates, not exceptions.
10. Log VRAM, RAM, tok/s, buffer level, and message rate from day one.
11. **Never ship a model that was not A/B-tested on this machine under full load.** `data/bakeoff.md` is the only authority on which model to use.

## Contracts you may not change without asking

- `SceneState` JSON — `PLAN.md` §5.2 (includes `speech_text`, `speech_energy`, `source`)
- SQLite schema — `PLAN.md` §8
- Event types on the bus — `PLAN.md` §3, §5.4

## Definition of done (v1)

`.\scripts\run.ps1` → picks a video → it plays → chat is contextually correct, bursty, in-character, never refuses, and a human cannot reliably tell it from real Twitch chat. See the scorecard in `PLAN.md` Appendix C.
