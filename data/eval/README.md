# Evaluation artifacts

These files are evidence, not a gate pass. The human ratings and blind A/B
still have to be run by a person.

## M3 transcript

```text
data/eval/ai_chat_3min.jsonl          one released message per line
data/eval/ai_chat_3min.txt            visible text only, one line per message
data/eval/ai_chat_3min.scenes.jsonl   scene observations during the run
data/eval/ai_chat_3min.levels.jsonl   buffer/rate samples
data/eval/ai_chat_3min.summary.json   run counters, bins, sources, end state
```

Reproduce a capture against the local backend and the 180 s clip:

```bash
.venv/bin/python scripts/capture_chat.py --duration 180 \
  --media big_buck_bunny_180s.mp4 --out data/eval/ai_chat_3min
```

## Deterministic checks

```bash
.venv/bin/python scripts/analyze_chat.py \
  --ai data/eval/ai_chat_3min.jsonl \
  --reference data/reference/twitch_chat_200.txt \
  --out data/eval/chat_quality_3min
```

This writes `chat_quality_3min.md` and `.json`. It checks length, guard
reasons, model-message repeats within 50, burst shape and source mix. It cannot
measure "sounds like Twitch chat"; that is the human M2 rating.

## Blind A/B

```bash
.venv/bin/python scripts/blind_ab.py
```

This writes `blind_ab.md` (240 lines per panel, visible text only) and a hidden
answer key under `data/eval/keys/`. A reviewer reads the packet and answers A or
B, then runs:

```bash
.venv/bin/python scripts/score_blind_ab.py --answer A --reviewer <name>
```

The M3 acceptance target is that the reviewer cannot identify the simulator
better than chance (≤ 60% accuracy). One answer is not the gate.

**Known limitation:** the AI transcript is for Big Buck Bunny, while the
reference corpus comes from a different Twitch stream. Content mismatch can
reveal the AI panel. A valid blind result needs a real chat sample from a stream
showing similar content. The current packet is useful for style review, not a
clean pass/fail.

## Pending human work

- M2: rate ≥ 7/10 "sounds like Twitch chat" on a scene batch.
- M3: blind A/B identification on matched content.
- M1: 30 hand-labelled frames across gameplay, talking-head, cooking, text UI
  and explicit content. The runner exists in `scripts/bakeoff.py`; the frames
  and labels do not.
