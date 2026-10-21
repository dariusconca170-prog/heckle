# Chat style check

Deterministic checks only. The blind human A/B is separate and this file
does not claim it passed.

| Metric | AI transcript | Real reference |
| --- | ---: | ---: |
| Messages | 342 | 240 |
| Mean words | 3.16 | 3.15 |
| Median words | 3.0 | 2.0 |
| p90 words | 6 | 8 |
| Max words | 10 | 16 |
| 1–4 words % | 78.1 | 78.3 |
| ≤ 12 words % | 100.0 | 98.8 |
| No trailing punctuation % | 98.0 | 96.7 |
| Lowercase start % | 69.0 | 78.3 |
| Emote-only % | 7500 | n/a |
| Contains known emote % | 19900 | n/a |

## AI transcript internals

- Message rate: **1.90 msg/s** over the captured window
- Handles: 79
- Sources: `{"template": 70, "crowd": 193, "model": 79}`
- 5 s bins: `[{"start_s": 0, "count": 5}, {"start_s": 5, "count": 3}, {"start_s": 10, "count": 8}, {"start_s": 15, "count": 11}, {"start_s": 20, "count": 13}, {"start_s": 25, "count": 4}, {"start_s": 30, "count": 10}, {"start_s": 35, "count": 13}, {"start_s": 40, "count": 10}, {"start_s": 45, "count": 13}, {"start_s": 50, "count": 6}, {"start_s": 55, "count": 7}, {"start_s": 60, "count": 12}, {"start_s": 65, "count": 15}, {"start_s": 70, "count": 13}, {"start_s": 75, "count": 13}, {"start_s": 80, "count": 11}, {"start_s": 85, "count": 8}, {"start_s": 90, "count": 15}, {"start_s": 95, "count": 8}, {"start_s": 100, "count": 9}, {"start_s": 105, "count": 8}, {"start_s": 110, "count": 10}, {"start_s": 115, "count": 7}, {"start_s": 120, "count": 7}, {"start_s": 125, "count": 15}, {"start_s": 130, "count": 12}, {"start_s": 135, "count": 7}, {"start_s": 140, "count": 8}, {"start_s": 145, "count": 5}, {"start_s": 150, "count": 8}, {"start_s": 155, "count": 12}, {"start_s": 160, "count": 8}, {"start_s": 165, "count": 14}, {"start_s": 170, "count": 8}, {"start_s": 175, "count": 6}, {"start_s": 180, "count": 0}]`
- Max silent gap: 3.4 s; gaps > 10 s: 0
- Bursts (5 s bin ≥ 8): 26; largest bin: 15

## Guard and repetition

- All messages: guard reasons `{"ok": 342}`
- Model messages: guard reasons `{"ok": 79}`
- Model repeats within 50: 26 (32.91% of 79)
- All-message repeats within 50: 92 (26.9%) — copy-pasta is expected to repeat

## Reference notes

- Reference guard reasons: `{"ok": 236, "too_long": 3, "refusal": 1}`
- The reference corpus has no per-line timestamps and no known emote dictionary,
  so rate and emote comparisons are not possible from it.

## Capture summary

```json
{
  "session_id": "2cac1edc968ab08e",
  "media_id": "big_buck_bunny_180s.mp4",
  "duration_s": 180.0,
  "viewer_count": 1284,
  "messages": 342,
  "message_rate": 1.9,
  "scenes": 41,
  "catching_up_events": 1,
  "dynamics": {
    "enabled": true,
    "status": "end",
    "session_id": "2cac1edc968ab08e",
    "buffer_level": 19,
    "target_buffer": 20,
    "low_buffer": 12,
    "critical_buffer": 5,
    "rate": 2.391,
    "intensity": 1.687,
    "excitation": 1.404,
    "cooling_down": false,
    "active_chatters": 39,
    "released": 342,
    "generated": 359,
    "templates": 72,
    "copies": 0,
    "dropped": 0,
    "late_released": 0,
    "last_error": "",
    "scene_t": 176.31,
    "next_refill_in_s": 0.0
  },
  "perception": {
    "status": "end",
    "session_id": "2cac1edc968ab08e",
    "samples": 41,
    "errors": 0,
    "last_error": "",
    "last_latency_s": 2.2469,
    "last_t": 176.31,
    "next_sample_in_s": 0.0,
    "duty_cap": 0.35,
    "deferred_for_chat": 393
  },
  "bins_5s": [
    {
      "start_s": 0,
      "count": 5
    },
    {
      "start_s": 5,
      "count": 3
    },
    {
      "start_s": 10,
      "count": 8
    },
    {
      "start_s": 15,
      "count": 11
    },
    {
      "start_s": 20,
      "count": 13
    },
    {
      "start_s": 25,
      "count": 4
    },
    {
      "start_s": 30,
      "count": 10
    },
    {
      "start_s": 35,
      "count": 13
    },
    {
      "start_s": 40,
      "count": 10
    },
    {
      "start_s": 45,
      "count": 13
    },
    {
      "start_s": 50,
      "count": 6
    },
    {
      "start_s": 55,
      "count": 7
    },
    {
      "start_s": 60,
      "count": 12
    },
    {
      "start_s": 65,
      "count": 15
    },
    {
      "start_s": 70,
      "count": 13
    },
    {
      "start_s": 75,
      "count": 13
    },
    {
      "start_s": 80,
      "count": 11
    },
    {
      "start_s": 85,
      "count": 8
    },
    {
      "start_s": 90,
      "count": 15
    },
    {
      "start_s": 95,
      "count": 8
    },
    {
      "start_s": 100,
      "count": 9
    },
    {
      "start_s": 105,
      "count": 8
    },
    {
      "start_s": 110,
      "count": 10
    },
    {
      "start_s": 115,
      "count": 7
    },
    {
      "start_s": 120,
      "count": 7
    },
    {
      "start_s": 125,
      "count": 15
    },
    {
      "start_s": 130,
      "count": 12
    },
    {
      "start_s": 135,
      "count": 7
    },
    {
      "start_s": 140,
      "count": 8
    },
    {
      "start_s": 145,
      "count": 5
    },
    {
      "start_s": 150,
      "count": 8
    },
    {
      "start_s": 155,
      "count": 12
    },
    {
      "start_s": 160,
      "count": 8
    },
    {
      "start_s": 165,
      "count": 14
    },
    {
      "start_s": 170,
      "count": 8
    },
    {
      "start_s": 175,
      "count": 6
    }
  ],
  "sources": {
    "template": 70,
    "crowd": 193,
    "model": 79
  },
  "handles": 79
}
```
