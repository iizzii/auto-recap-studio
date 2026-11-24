# ASSUMPTIONS.md

Architectural assumptions made during the build. Each choice takes the cleanest
default; override via `config.yaml` or CLI flags.

## Runtime & Environment

1. **GitHub Actions `ubuntu-latest`** is the sole execution target. 2 vCPUs,
   7 GB RAM, ~14 GB free disk on `/tmp`. No GPU.
2. **Python 3.11+** via `actions/setup-python`. All typing uses 3.11 features.
3. **FFmpeg** is installed via `apt-get` (version 5.x+ on Ubuntu 22.04/24.04).
   The `ass` subtitle filter and `lavfi` audio sources are available.
4. **No local execution.** The operator never runs Python locally. All
   interaction is via the GitHub Web UI, Actions artifacts, Telegram, and
   YouTube Studio.

## Media & Sources

5. Source videos are **legally cleared** before pipeline entry. The pipeline
   enforces this via `rights.json` but cannot determine legality itself.
6. Source video is assumed to be a single video stream, any codec FFmpeg can
   decode. No multi-angle or multi-episode containers.
7. Subtitle `.srt` / `.vtt` files are UTF-8. Non-Latin scripts are accepted
   but the pipeline is optimized for English narration.
8. Maximum source file size defaults to 6 GB, well within ephemeral disk.
9. **Source audio is stripped by default** (`source_audio_mode: "off"`). The
   recap uses only TTS + synthesized BGM. Ducked source audio is opt-in and
   requires explicit soundtrack clearance in `rights.json`.

## LLM & TTS

10. **Gemini 2.5 Flash** is the default LLM via `google-genai`. The
    `openai_compatible` fallback allows any OpenAI-API-compatible endpoint.
11. **`edge-tts`** (Microsoft Edge TTS) is the default voice backend. It is
    free, requires no API key, but is rate-limited. Retries with exponential
    backoff handle transient 429s.
12. Mock backends (`--mock-llm`, `--mock-tts`, `--mock-vision`) produce
    deterministic synthetic outputs for CI. The mock TTS uses a 280 Hz sine
    tone to distinguish from the 1000 Hz synthetic source.
13. LLM structured output is requested as JSON. If the model returns markdown
    fences or preamble, the parser strips them before validation.

## Video Processing

14. **`blurred_letterbox`** is the only v1 layout mode. 16:9 source is shown
    at full width centered on a blurred/darkened background filling 9:16.
    `smart_crop` is deferred.
15. **30 fps** is the universal frame rate. All frame arithmetic is integer.
16. Shot durations are 2.0–4.0 seconds (hard max 4.5s). Timeline uses pure
    integer-frame math with zero cumulative drift.
17. `beat_gap_frames` (2 frames = 66.67ms at 30fps) are included INSIDE each
    beat's `audio_frames` (except the final beat). This guarantees
    `sum(beat.audio_frames) == sum(shot.frames) == total_frames`.
18. Intermediate clips are encoded `ultrafast crf 16` for speed; final output
    is `veryfast crf 20` with `movflags +faststart`.

## Captions

19. **`faster-whisper` `base.en`** aligns TTS audio to word-level timestamps.
    In mock mode, timestamps are derived arithmetically.
20. Font: Montserrat ExtraBold (OFL) is preferred; DejaVu Sans Bold is the
    non-fatal fallback.
21. Karaoke captions use `.ass` format burned in via FFmpeg's `ass` filter.

## Publishing

22. **All publishing is private by default.** Public YouTube upload requires
    explicit `approval.json` approval AND the weekly cap allows it.
23. YouTube Data API v3 uploads via refresh-token OAuth. Unverified API
    projects may have uploads forced to `private` by YouTube; this is detected
    and reported honestly.
24. Telegram delivery uses the Bot API. Videos over 50 MB send a link instead.
25. R2/S3 presigned URLs expire after 7 days.

## Security

26. Workflow inputs are NEVER interpolated in `run:` shell blocks. All inputs
    pass through `env:` mappings.
27. Secrets are never printed or echoed.
28. Downloaded files are treated as untrusted. No interpreter execution from
    download directories.
29. `rights.json` MUST exist before any download begins. Missing or invalid
    rights cause immediate exit with code 3.

## Dependencies

30. `opencv-python-headless` is used (not `opencv-python`) to avoid X11 deps.
31. `scipy` is NOT used. BGM/SFX are generated via FFmpeg `lavfi`.
32. `espeak-ng` is NOT installed or used.
33. All dependencies are pinned in `requirements.txt`.

## Queue & Batching

34. `queue.json` is browser-editable. Items are processed in order, one at a
    time, respecting the weekly upload cap.
35. `auto_approve: true` in queue items triggers a loud warning but is honored.

## Testing

36. CI smoke tests use zero secrets. The synthetic test video is 45s with
    color bars, moving shapes, and a 1000 Hz tone.
37. `hypothesis` property-based tests verify frame-sum invariants across
    wide beat-duration ranges.
