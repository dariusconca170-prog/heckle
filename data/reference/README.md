# Twitch chat reference corpus

`twitch_chat_200.txt` and `twitch_chat_200.jsonl` contain 240 real Twitch chat
lines collected from the public Hugging Face dataset
[`S1lver404/twitch_chat`](https://huggingface.co/datasets/S1lver404/twitch_chat)
on 2026-10-20.

This is the M3 prerequisite from `PLAN.md` §12: **≥ 200 real Twitch chat lines**
to compare against the simulator's transcript. It is a local style/reference
sample only. No messages are used for training, and the collector drops
commands, bot output, URLs, and duplicate lines. The source dataset does not
ship an explicit license file; treat this copy as local evaluation material,
not a redistributable asset.

Reproduce or refresh:

```bash
.venv/bin/python scripts/fetch_twitch_reference.py --target 240
```

The 240 number is a buffer over the 200-line minimum. The M3 blind comparison
still needs a human judge; this file is the reference side, not proof of an
acceptance pass.
