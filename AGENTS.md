# Agent Operating Rules (`auto-recap-studio`)

CURRENT_HOST_MODE = OFFICE_MAC

## Dual-Device `HOST_MODE`
- **`OFFICE_MAC` (ACTIVE — MANDATORY ZERO LOCAL EXECUTION):**
  - **STRICTLY FORBIDDEN LOCALLY:** Do **NOT** create or use `.venv` locally. Do **NOT** run `python`, `python3`, `pip`, `pytest`, `ruff`, `ffmpeg`, `yt-dlp`, or download any media/assets locally on this machine.
  - **ALLOWED LOCALLY:** Reading/editing repository files directly in the workspace, and running `git` (`status`, `add`, `commit`, `push`, `pull`) and GitHub CLI `gh` (`workflow run`, `run list`, `run watch`, `run view --log`, `run view --log-failed`, `run download` ONLY for tiny text/JSON logs if needed—never media).
  - **ALL EXECUTION HAPPENS ON GITHUB ACTIONS (`ubuntu-latest`):** Run all linting (`ruff`), unit tests (`pytest`), 3D host asset generation (`generate_host_assets.yml`), and full 3-part video renders (`render_series.yml`) exclusively via GitHub Actions workflows.
- **`PERSONAL_MAC`** (only when explicitly switched back and `.venv`/`.env` present): Local verification via `.venv/bin/ruff`, `.venv/bin/pytest`, and `.venv/bin/python`. Local media/BGM caches live in `.local_cache/`.

## Engineering & Debugging Rules
- Read complete relevant files before editing so edge cases, schema contracts, and config keys are never missed.
- Inspect complete GitHub Actions error tracebacks via `gh run view --log-failed` / `--log` when debugging CI or render runs.
- Never read binary/media files (`*.mp4`, `*.mp3`, `*.wav`, `*.jpg`), `.venv/`, `.local_cache/`, `__pycache__/`, or `scout_reports/`.
- Never ship silent mock fallbacks when `mock_llm=False`; fail loudly with exact status code, model name, and error snippet.
