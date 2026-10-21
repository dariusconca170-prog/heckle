# M0.5 model bake-off

- Machine: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- Python: `3.12.3`
- Updated: `2026-10-20T16:51:30+00:00`
- Status: **PROVISIONAL — M0.5 waived; not a measured winner**
- Winner: **4b (provisional)**

The gate from `PLAN.md` §7.4: 30 minutes under full app load (browser + active
playback session, ffmpeg decoding, STT running), ≥ 25 tok/s at minute 30,
8-message batch p95 ≤ 4.5 s, ≥ 1 GB RAM free, ≥ 500 MB VRAM free, no page-fault
growth, no crash. A model that misses any line is out. If both pass, the higher
blind chat-voice score wins; a tie goes to the 9B.

## Runs

| Profile | Started | Duration | Full load | tok/s @ end | Chat batch p95 | VRAM peak | VRAM free min | RAM free min | mmproj | Floor | Voice | Refusals | Accuracy | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `9b` | 2026-10-20T16:41:17+00:00 | 0.1 min | yes | — | — | — | — | — | ? | FAIL | — | — refused / — moralized | — | launch failed |
| `4b` | 2026-10-20T16:48:53+00:00 | 0.5 min | yes | 57.7 | 2.4 s | 6831.0 MB | 1003.0 MB | 7455.6 MB | ? | FAIL | — | — refused / — moralized | — | — |

## Failure details

- `9b`: llama-server exited with code 1 during startup.
Log: /home/darius/Desktop/twitch sim/data/logs/llama-server-9b-20261020T164117Z.log
$ /home/darius/Desktop/llamacpp/llama-b11540/llama-server -m "/home/darius/Desktop/twitch sim/data/models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" --mmproj "/home/darius/Desktop/twitch sim/data/models/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf" -c 4096 -np 1 -ngl 99 --alias twitch-sim-9b --host 127.0.0.1 --port 8080 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 8 --flash-attn on --jinja --chat-template-kwargs {"enable_thinking":false}

0.00.081.013 W Setting 'enable_thinking' via --chat-template-kwargs is deprecated. Use --reasoning on / --reasoning off instead.
0.00.081.212 I srv  llama_server: initializing ...
0.00.081.235 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.081.890 W srv  llama_server: security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)
0.00.083.887 I srv    load_model: loading model '/home/darius/Desktop/twitch sim/data/models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
0.01.279.891 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.02.184.855 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4810.28 MiB on device 0: cudaMalloc failed: out of memory
0.02.184.874 E ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA0 buffer of size 5043947520
0.02.407.913 E llama_model_load: error loading model: unable to allocate CUDA0 buffer
0.02.407.921 E llama_model_load_from_file_impl: failed to load model
0.02.407.931 E cmn  common_init_: failed to load model '/home/darius/Desktop/twitch sim/data/models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
0.02.407.940 E srv    load_model: failed to load model, '/home/darius/Desktop/twitch sim/data/models/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
0.02.407.944 I srv    operator(): operator(): cleaning up before exit...
0.02.408.714 E srv  llama_server: exiting due to model loading error

## Run files

- `4b`: `/home/darius/Desktop/twitch sim/data/bakeoff/4b-run-20261020T164923Z.json`
- `9b`: `/home/darius/Desktop/twitch sim/data/bakeoff/9b-run-20261020T164121Z.json`

## Evidence still required

- `scripts/refusal_audit.py` results for both 9B and 35B (0 refused, ≤ 2 moralized).
- blind chat-voice rating for both (`bakeoff.py voice` then `rate-voice`).
- 30-frame perception accuracy for both (`bakeoff.py accuracy`, ≥ 80 %).
- `bakeoff.py select` refuses to name a winner until every line above exists.

## Decision

**4b** — User selected the 4B for 8 GB game streaming (Geometry Dash + OBS + browser). M0.5 measurement was explicitly waived; this is provisional, not a measured winner. Look-ahead updated from the 4B smoke: vision ~1.53 s + batch p95 ~2.41 s + 2 s.

> Provisional: M0.5 was explicitly waived by the user. These are not measured gate results and must be replaced by `bakeoff.py select` after real 9B/35B/4B evidence exists.

Winning launch command:

```bash
llama-server -m /home/darius/Desktop/twitch sim/data/models/Qwen3.5-4B-Unredacted-MAX.Q4_K_M.gguf --mmproj /home/darius/Desktop/twitch sim/data/models/Qwen3.5-4B-Unredacted-MAX.mmproj-Q8_0.gguf -c 4096 -np 1 -ngl 99 --alias twitch-sim-4b --host 127.0.0.1 --port 8080 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 8 --flash-attn on --jinja --chat-template-kwargs {"enable_thinking":false}
```

Look-ahead for the app: **5.8 s**. The app reads `data/model_profile.json`; do not re-tune on launch.
