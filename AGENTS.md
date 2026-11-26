# Agent Operating Rules (`auto-recap-studio`)

## Dual-Device `HOST_MODE`
- **`PERSONAL_MAC`**: `.venv` and `.env` present. Run `.venv/bin/ruff`, `.venv/bin/pytest`, and `.venv/bin/python` locally for fast verification before pushing. Local media/BGM caches live in `.local_cache/`.
- **`OFFICE_MAC`** (or if `.env`/`.venv` is absent): **ZERO local execution**. Use terminal ONLY for `git` and `gh` CLI; run all tests/renders on GitHub Actions `ubuntu-latest`.

## Token-Saving Rules
- Never dump large files (>150 lines); use `rg`/`grep` and targeted line ranges.
- Never read binary/media files (`*.mp4`, `*.mp3`, `*.wav`, `*.jpg`), `.venv/`, `.local_cache/`, `__pycache__/`, or `scout_reports/`.
- Use concise checks:
  - `ruff check --output-format=concise .`
  - `pytest tests/unit -q --tb=short --maxfail=3`
- Pipe verbose FFmpeg/CLI logs to `/tmp/*.log` and inspect only the last 25 lines on failure.
