# License-Safe 9:16 Recap Video Pipeline (GitHub Actions)

An automated, zero-local Python 3.11+ pipeline that converts legally cleared source videos (Public Domain, CC0, CC BY, CC BY-SA, owned, or licensed) into **1080x1920, 30 fps, 2:00–2:58 vertical recap Shorts/Reels** entirely inside GitHub Actions.

---

## Legal Scope & Responsibilities

- **You are responsible for source clearance.** Every project requires `projects/<slug>/rights.json` before any download begins (missing or invalid files exit with code `3`).
- **Public Domain caveats:** Public domain status varies by country (e.g., US vs. EU/UK copyright terms). Verify your target distribution territories and document `pd_basis` + `pd_verified_by_human: true`.
- **Soundtrack rights:** Source audio is stripped (`-an`) by default (`source_audio_mode: "off"`). Never enable `"ducked"` mode unless `soundtrack_rights_note` confirms separate music clearance.
- **No copyright-evasion transforms:** No `hflip`, pitch-shifting, or hidden titles. Mandatory attribution and `"Narration generated with text-to-speech."` are appended programmatically.
- **YouTube Unverified API Notice:** Unverified Google Cloud / YouTube Data API v3 projects may have uploads forced to `private` by YouTube regardless of `privacyStatus`. The pipeline detects and reports this honestly.
- **GitHub Actions Terms:** Respect GitHub Actions usage limits and do not use runners for unauthorized scraping or copyrighted redistribution.

---

## One-Time GitHub Setup (Browser Only)

1. Push or upload this repository to GitHub (private repo recommended).
2. Go to **Settings -> Secrets and variables -> Actions -> New repository secret** and add:
   - **Required for live LLM:** `GEMINI_API_KEY` (or `OPENAI_API_KEY`)
   - **Optional voice:** `ELEVENLABS_API_KEY` (default `edge-tts` needs no key)
   - **Optional R2 storage:** `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_OUTPUT_BUCKET`
   - **Optional Google Drive:** `GDRIVE_SERVICE_ACCOUNT_JSON`
   - **Optional Telegram delivery:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - **Optional YouTube upload:** `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`

---

## Running the Zero-Secret Smoke Test

1. Open the **Actions** tab in GitHub.
2. Select **CI Smoke Test** -> **Run workflow**.
3. Wait for `lint-and-unit` and `e2e-synthetic` to turn green, then download the `ci-final-short` artifact (`final_short.mp4`).

---

## Creating `rights.json` via Browser

1. Open **Actions -> Rights Check -> Run workflow**.
2. Set `create_template: true` and run to print a ready-to-paste `rights.json` template in the workflow summary.
3. In GitHub Web UI, click **Add file -> Create new file**, name it `projects/<your_slug>/rights.json`, paste & fill in your license details, and commit.
4. Re-run **Rights Check** with `slug: <your_slug>` and `create_template: false` to verify it passes.

---

## Two-Stage Browser QC (`plan_only` -> Edit -> `render_only`)

1. Run **Actions -> Recap Pipeline -> Run workflow** with `mode: plan_only`.
2. Open `projects/<slug>/02_script_manifest.json` in GitHub Web UI, click the pencil icon to edit narration/titles/windows, and commit.
3. Re-run **Recap Pipeline** with `mode: render_only` (skips LLM calls and honors your edits).
4. Review `projects/<slug>/04_storyboard.jpg`, Telegram preview, or the downloaded artifact (`retention-days: 2`).

---

## Human Approval Gate & Weekly Cap

- Public YouTube uploads (`youtube_public`) are **blocked and downgraded to private** unless `projects/<slug>/approval.json` has `"approved": true` AND `max_public_uploads_per_week` (default `3`) has not been reached.
- Use `queue.json` + **Batch Queue** workflow for scheduled/queued runs.

---

## Exit Codes

| Code | Meaning |
| :--- | :--- |
| `2` | Invalid CLI/workflow input or insufficient disk space |
| `3` | Rights Guard failure (missing/invalid `rights.json` or IA mismatch) |
| `4` | No subtitles available and auto-transcribe disabled/failed |
| `5` | Script validation failed after retries (see `debug/` artifact) |
| `6` | Voiceover exceeds `max_duration` even after `1.18x` atempo |
| `7` | QA Gate failure (resolution, sync, black/silence, or audio leak) |
| `8` | Time budget exhausted (resume with `render_only`) |
| `9` | All requested publish targets failed |

---

## Running Elsewhere

See [docs/run_elsewhere.md](docs/run_elsewhere.md) and [docker/Dockerfile](docker/Dockerfile) to run on Oracle Cloud Free Tier, a VPS, or Docker.
