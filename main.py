#!/usr/bin/env python3
"""Recap-pipeline CLI – 9:16 vertical recap video generator."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import typer
import yaml

from pipeline import (
    audio_mixer,
    caption_engine,
    chapter_commentator,
    chapter_composer,
    chapter_slicer,
    clip_selector,
    ffutil,
    history,
    host_animator,
    publisher,
    qa_gate,
    report,
    rights_guard,
    script_generator,
    source_fetcher,
    srt_parser,
    story_scout,
    transcriber,
    tts_engine,
    video_composer,
)
from pipeline.budget import TimeBudget
from pipeline.models import ApprovalRecord, Beat, ProjectManifest, QueueFile, RunReport

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("recap")

app = typer.Typer(
    name="recap-pipeline",
    help="9:16 vertical recap video pipeline.",
    add_completion=False,
    no_args_is_help=True,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load *config_path*, merging onto ``config.yaml`` if it is an override."""
    base_path = Path("config.yaml")
    cfg: dict[str, Any] = {}
    if base_path.exists() and config_path.resolve() != base_path.resolve():
        cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    override = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(cfg, override)


def _ensure_approval_file(project_dir: Path) -> None:
    approval_path = project_dir / "approval.json"
    if not approval_path.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(
            ApprovalRecord().model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )


@app.command()
def rights(
    slug: str = typer.Option(..., "--slug", help="Project slug to validate."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Path to config YAML."),
) -> None:
    """Validate the rights record for a project slug."""
    cfg = _load_config(config)
    strict = bool(cfg.get("rights", {}).get("strict_rights_check", True))
    project_dir = Path("projects") / slug
    record = rights_guard.validate_rights(slug, project_dir, strict=strict)
    rights_guard.capture_evidence(record, project_dir, strict=strict)
    typer.echo(f"Rights OK for '{slug}'.")


@app.command()
def run(
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Config YAML path."),
    slug: str = typer.Option(..., "--slug", help="Project slug."),
    video_source: str = typer.Option(..., "--video-source", help="Video URL or local path."),
    srt_source: Optional[str] = typer.Option(None, "--srt-source", help="Subtitle URL or path."),
    mode: str = typer.Option("full_auto", "--mode", help="plan_only | render_only | full_auto"),
    style: str = typer.Option("commentary", "--style", help="commentary | recap"),
    publish_target: str = typer.Option("all_private", "--publish-target", help="Delivery target."),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Use mock LLM responses."),
    mock_tts: bool = typer.Option(False, "--mock-tts", help="Use mock TTS audio."),
    mock_vision: bool = typer.Option(False, "--mock-vision", help="Use mock vision scoring."),
    force: bool = typer.Option(False, "--force", help="Regenerate existing script manifest."),
    split_part: int = typer.Option(0, "--split-part", help="0=full, 1..N=part N."),
    total_parts: int = typer.Option(2, "--total-parts", help="Total parts when split_part > 0."),
    max_duration: Optional[int] = typer.Option(None, "--max-duration", help="Max duration sec."),
    allow_reproduce: bool = typer.Option(False, "--allow-reproduce", help="Allow re-producing published story."),
) -> None:
    """Execute the 9:16 recap pipeline."""
    t0 = time.monotonic()
    cfg = _load_config(config)
    if max_duration is not None:
        cfg.setdefault("runtime", {})["max_duration_sec"] = max_duration

    rt = cfg.get("runtime", {})
    budget = TimeBudget(float(rt.get("time_budget_minutes", 55)))
    project_dir = Path("projects") / slug
    work_dir = Path(str(rt.get("work_dir", "/tmp/recap_work"))) / slug
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rep = RunReport(slug=slug, status="failed", exit_code=1)

    # 1. Rights Guard (MUST run before any download)
    strict = bool(cfg.get("rights", {}).get("strict_rights_check", True))
    rights_rec = rights_guard.validate_rights(slug, project_dir, strict=strict)

    # 1b. Published History Duplicate Check (skip for synthetic ci_synth)
    if slug != "ci_synth":
        history.check_no_duplicate_or_exit(
            slug=slug,
            work_title=rights_rec.work_title,
            source_url=video_source,
            allow_reproduce=allow_reproduce,
        )
    rights_guard.capture_evidence(rights_rec, project_dir, strict=strict)
    run_rep.stages_completed.append("rights_guard")

    if mode in ("plan_only", "full_auto"):
        _ensure_approval_file(project_dir)

    # 2. Source Fetcher & Subtitles
    budget.check_or_exit("source_fetcher")
    source_path, sha256 = source_fetcher.fetch_source(video_source, work_dir, cfg)
    run_rep.source_sha256 = sha256
    run_rep.stages_completed.append("source_fetcher")

    src_dur = ffutil.duration(source_path)
    src_cfg = cfg.get("source", {})
    cfg_head = float(src_cfg.get("exclude_head_sec", 20))
    allowed_start = float(rights_rec.exclude_head_sec) if rights_rec.exclude_head_sec is not None else cfg_head
    allowed_start = min(allowed_start, max(0.0, src_dur - 10.0))
    tail_pct = float(src_cfg.get("exclude_tail_pct", 8))
    allowed_end = max(allowed_start + 5.0, src_dur * (1.0 - tail_pct / 100.0))
    if split_part > 0 and src_dur <= 900:
        log.warning(
            "Source duration %.1fs <= 15 min: shorts perform best as a single complete video (split_part=0).",
            src_dur,
        )
    if split_part > 0 and total_parts > 1:
        ranges = compute_series_ranges(allowed_start, allowed_end, total_parts)
        allowed_range = ranges[min(split_part - 1, len(ranges) - 1)]
    else:
        allowed_range = (allowed_start, allowed_end)

    # Dispatch to uncut chapter_commentary mode when configured (skip for ci_synth)
    default_mode = str(cfg.get("pipeline", {}).get("default_mode", "chapter_commentary"))
    if slug != "ci_synth" and (mode == "chapter_commentary" or (mode == "full_auto" and default_mode == "chapter_commentary")):
        final_mp4 = run_chapter_commentary(
            cfg=cfg,
            slug=slug,
            source_path=source_path,
            rights_rec=rights_rec,
            allowed_range=allowed_range,
            split_part=split_part,
            total_parts=total_parts,
            project_dir=project_dir,
            work_dir=work_dir,
            out_dir=out_dir,
            mock_llm=mock_llm,
            mock_tts=mock_tts,
        )
        run_rep.stages_completed.append("chapter_commentary")
        manifest_stub = ProjectManifest(
            project_slug=slug,
            style="commentary",
            part_number=split_part,
            title=f"{rights_rec.work_title} #shorts",
            banner_text="WATCH KIKO REACT",
            description=f"{rights_rec.work_title} commentary with Kiko.",
            hashtags=["#shorts", "#animation"],
            source_title=rights_rec.work_title,
            attribution_block=rights_rec.attribution_text or "",
            beats=[
                Beat(
                    beat_id=1,
                    narration="Kiko reacts to the movie.",
                    source_window_start_sec=allowed_range[0],
                    source_window_end_sec=allowed_range[1],
                    visual_cue="movie scene",
                    mood="hook",
                    kind="commentary",
                )
            ],
        )
        pub_res = publisher.publish_outputs(
            slug=slug,
            publish_target=publish_target,
            final_mp4=final_mp4,
            thumb_jpg=work_dir / "thumbnail.jpg",
            manifest=manifest_stub,
            rights=rights_rec,
            project_dir=project_dir,
            out_dir=out_dir,
            config=cfg,
        )
        run_rep.publish_results = pub_res
        run_rep.output_files = [str(p) for p in sorted(out_dir.iterdir())]
        run_rep.status = "success"
        run_rep.exit_code = 0
        run_rep.duration_sec = round(time.monotonic() - t0, 2)
        report.write_run_report(run_rep, project_dir, out_dir)
        report.write_github_summary(run_rep)
        h = history.load()
        history.upsert_rendered(
            h, slug=slug, work_title=rights_rec.work_title,
            source_url=video_source, source_sha256=run_rep.source_sha256 or "",
        )
        history.save(h)
        return

    srt_path = transcriber.resolve_subtitles(source_path, srt_source, work_dir, cfg, budget)
    segments = srt_parser.merge_segments(srt_parser.parse_srt(srt_path))
    speech_density = srt_parser.compute_speech_density(segments, bucket_sec=60.0)
    run_rep.stages_completed.append("subtitles")

    # 3. Script Generation / Load
    budget.check_or_exit("script_generator")
    manifest, script_warnings = script_generator.generate_or_load_script(
        slug=slug,
        project_dir=project_dir,
        work_dir=work_dir,
        segments=segments,
        rights=rights_rec,
        allowed_range=allowed_range,
        config=cfg,
        mode=mode,
        style=style,
        split_part=split_part,
        mock_llm=mock_llm,
        force=force,
        budget=budget,
        source_path=source_path,
    )
    run_rep.warnings.extend(script_warnings)
    run_rep.stages_completed.append("script")

    if mode == "plan_only":
        run_rep.status = "success"
        run_rep.exit_code = 0
        run_rep.duration_sec = round(time.monotonic() - t0, 2)
        report.write_run_report(run_rep, project_dir, out_dir)
        report.write_github_summary(run_rep)
        typer.echo(f"Plan ready at {project_dir / '02_script_manifest.json'}")
        return

    # 4. TTS & Timeline
    budget.check_or_exit("tts_engine")
    stitched_voice = tts_engine.synthesize_beats(
        manifest.beats, work_dir, cfg, mock_tts=mock_tts
    )
    run_rep.stages_completed.append("tts")

    # 5. Clip Selection + Vision
    budget.check_or_exit("clip_selector")
    shots = clip_selector.select_shots(
        manifest=manifest,
        source_path=source_path,
        project_dir=project_dir,
        work_dir=work_dir,
        allowed_range=allowed_range,
        speech_density=speech_density,
        config=cfg,
        mock_vision=mock_vision,
    )
    manifest.shots = shots
    run_rep.stages_completed.append("clip_selector")

    # 6. Captions
    budget.check_or_exit("caption_engine")
    ass_path, match_ratio = caption_engine.generate_captions(
        slug=slug,
        beats=manifest.beats,
        stitched_voice=stitched_voice,
        work_dir=work_dir,
        config=cfg,
        mock_tts=mock_tts,
    )
    run_rep.stages_completed.append("captions")

    # 7. Audio Mix & Video Compose
    budget.check_or_exit("audio_mixer")
    master_audio = audio_mixer.mix_audio(stitched_voice, manifest.beats, work_dir, cfg)
    run_rep.stages_completed.append("audio_mixer")

    budget.check_or_exit("video_composer")
    final_mp4, intermediates = video_composer.compose_video(
        manifest=manifest,
        shots=shots,
        source_path=source_path,
        master_audio=master_audio,
        ass_path=ass_path,
        work_dir=work_dir,
        config=cfg,
        work_year=rights_rec.year,
    )
    run_rep.stages_completed.append("video_composer")

    # 8. QA Gate
    qa_rep = qa_gate.run_qa_gate(
        final_mp4=final_mp4,
        shots=shots,
        intermediates=intermediates,
        caption_match_ratio=match_ratio,
        out_dir=out_dir,
        config=cfg,
    )
    run_rep.warnings.extend(qa_rep.get("warnings", []))
    run_rep.stages_completed.append("qa_gate")

    # 9. Publish
    pub_res = publisher.publish_outputs(
        slug=slug,
        publish_target=publish_target,
        final_mp4=final_mp4,
        source_path=source_path,
        manifest=manifest,
        rights=rights_rec,
        shots=shots,
        project_dir=project_dir,
        out_dir=out_dir,
        config=cfg,
    )
    run_rep.publish_results = pub_res
    run_rep.output_files = [str(p) for p in sorted(out_dir.iterdir())]
    run_rep.stages_completed.append("publish")

    run_rep.status = "success"
    run_rep.exit_code = 0
    run_rep.duration_sec = round(time.monotonic() - t0, 2)
    report.write_run_report(run_rep, project_dir, out_dir)
    report.write_github_summary(run_rep)

    if slug != "ci_synth":
        h = history.load()
        history.upsert_rendered(
            h,
            slug=slug,
            work_title=rights_rec.work_title,
            source_url=video_source,
            source_sha256=run_rep.source_sha256 or "",
        )
        history.save(h)


@app.command("queue-next")
def queue_next(
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Config YAML path."),
    queue_file: Path = typer.Option(Path("queue.json"), "--queue-file", help="Queue JSON path."),
) -> None:
    """Select and process the next pending item in queue.json."""
    cfg = _load_config(config)
    q = QueueFile(**json.loads(queue_file.read_text(encoding="utf-8")))
    pending = next((item for item in q.items if item.status == "pending"), None)
    if pending is None:
        typer.echo("No pending items in queue.json.")
        return

    if pending.auto_approve:
        warn = f"WARNING: queue item '{pending.slug}' has auto_approve=true!"
        log.warning(warn)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
        if summary_path:
            with Path(summary_path).open("a", encoding="utf-8") as fh:
                fh.write(f"> [!WARNING]\n> {warn}\n")

    if pending.publish_target == "youtube_public":
        cap = int(cfg.get("publishing", {}).get("max_public_uploads_per_week", 3))
        used = publisher.count_weekly_public_uploads(Path("projects"), queue_file)
        if used >= cap:
            pending.status = "deferred"
            pending.error = f"Weekly cap reached ({used}/{cap})"
            queue_file.write_text(q.model_dump_json(indent=2) + "\n", encoding="utf-8")
            typer.echo(f"Deferred '{pending.slug}': weekly cap reached.")
            return

    try:
        run(
            config=config,
            slug=pending.slug,
            video_source=pending.video_source_url,
            srt_source=pending.srt_source_url,
            mode="full_auto",
            style=pending.style,
            publish_target=pending.publish_target,
            split_part=pending.split_part,
            max_duration=pending.max_duration,
        )
        pending.status = "done"
        pending.completed_on = dt.datetime.now(dt.timezone.utc).isoformat()
    except SystemExit as exc:
        pending.status = "failed"
        pending.error = f"Exit code {exc.code}"
        queue_file.write_text(q.model_dump_json(indent=2) + "\n", encoding="utf-8")
        sys.exit(int(exc.code) if isinstance(exc.code, int) else 1)
    except Exception as exc:
        pending.status = "failed"
        pending.error = str(exc)
        queue_file.write_text(q.model_dump_json(indent=2) + "\n", encoding="utf-8")
        raise

    queue_file.write_text(q.model_dump_json(indent=2) + "\n", encoding="utf-8")


@app.command()
def status() -> None:
    """Show status of queued items and project manifests."""
    q_path = Path("queue.json")
    if q_path.exists():
        q = QueueFile(**json.loads(q_path.read_text(encoding="utf-8")))
        typer.echo(f"Queue items: {len(q.items)}")
        for item in q.items:
            typer.echo(f"  - {item.slug}: {item.status}")
    proj_root = Path("projects")
    if proj_root.exists():
        for p in sorted(proj_root.iterdir()):
            if p.is_dir():
                has_rights = (p / "rights.json").exists()
                has_script = (p / "02_script_manifest.json").exists()
                typer.echo(f"Project {p.name}: rights={has_rights}, script={has_script}")


@app.command()
def scout(
    query: str = typer.Option("", "--query", help="Optional theme/keyword."),
    candidate_count: int = typer.Option(3, "--candidate-count", help="Number of candidates to score."),
    license_scope: str = typer.Option("cc_only", "--license-scope", help="cc_only | cc_and_pd"),
    providers: str = typer.Option(
        "blender_peertube,wikimedia_commons,sepiasearch",
        "--providers",
        help="Comma-separated HD providers.",
    ),
    auto_produce: bool = typer.Option(False, "--auto-produce", help="Auto-render #1 winner if CC0/CC-BY."),
    publish_target: str = typer.Option("telegram", "--publish-target", help="Delivery target."),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Use deterministic mock LLM scores."),
) -> None:
    """Discover, validate, and score top-N free-to-use stories."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    rep = story_scout.run_scout(
        query=query,
        candidate_count=candidate_count,
        license_scope=license_scope,
        providers=[p.strip() for p in providers.split(",") if p.strip()],
        mock_llm=mock_llm,
        mock_http=False,
        gemini_api_key=gemini_key,
        auto_produce=auto_produce,
    )
    if rep.get("error"):
        typer.echo(f"Scout error: {rep['error']}")
        raise typer.Exit(code=1)

    typer.echo(f"Winner: {rep['winner_slug']} (score={rep['winner']['total_score']}/100)")
    for c in rep.get("candidates", []):
        typer.echo(
            f"  #{c['rank']} {c['title']} ({c['license_key']}) -> "
            f"tech={c['tech_score']} + story={c['story_score']} = {c['total_score']}/100"
        )

    if auto_produce and not rep.get("is_pd"):
        w = rep["winner"]
        typer.echo(f"Auto-producing #1 winner: {rep['winner_slug']}...")
        run(
            config=Path("config.yaml"),
            slug=rep["winner_slug"],
            video_source=w["video_url"],
            srt_source=w.get("srt_url") or None,
            mode="full_auto",
            style="commentary",
            publish_target=publish_target,
            mock_llm=mock_llm,
            mock_tts=False,
            mock_vision=False,
            force=True,
            split_part=0,
            max_duration=175,
            allow_reproduce=False,
        )


@app.command(name="mark-published")
def mark_published(
    slug: str = typer.Option(..., "--slug", help="Project slug."),
    platform: str = typer.Option(..., "--platform", help="youtube | instagram | both"),
    youtube_url: str = typer.Option("", "--youtube-url", help="YouTube URL."),
    instagram_url: str = typer.Option("", "--instagram-url", help="Instagram URL."),
) -> None:
    """Mark a project as published on YouTube and/or Instagram."""
    h = history.load()
    if platform in ("youtube", "both"):
        history.mark_published(h, slug=slug, platform="youtube", url=youtube_url)
    if platform in ("instagram", "both"):
        history.mark_published(h, slug=slug, platform="instagram", url=instagram_url)
    history.save(h)

    cal_path = Path("release_calendar_365.json")
    if cal_path.exists():
        from pipeline.models import ReleaseCalendar365
        cal = ReleaseCalendar365(**json.loads(cal_path.read_text(encoding="utf-8")))
        changed = False
        for s in cal.slots:
            if s.base_slug == slug or s.slot_id == slug:
                s.status = "published"
                changed = True
        if changed:
            cal_path.write_text(cal.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(f"Marked '{slug}' as published on {platform}.")


@app.command(name="history")
def show_history() -> None:
    """Print published/rendered story history summary."""
    h = history.load()
    typer.echo(f"Total tracked stories: {len(h.records)}")
    for r in h.records:
        yt = r.platforms.get("youtube")
        ig = r.platforms.get("instagram")
        yt_s = "YT:yes" if yt and yt.published else "YT:no"
        ig_s = "IG:yes" if ig and ig.published else "IG:no"
        typer.echo(f"  - {r.slug} ({r.work_title}) [{r.status}] {yt_s} {ig_s}")


def compute_series_ranges(
    allowed_start: float, allowed_end: float, parts: int
) -> list[tuple[float, float]]:
    """Partition [allowed_start, allowed_end] into N strictly chronological, non-overlapping slices."""
    n = max(1, parts)
    slice_len = max(5.0, (allowed_end - allowed_start) / n)
    return [
        (
            round(allowed_start + k * slice_len, 2),
            round(min(allowed_end, allowed_start + (k + 1) * slice_len), 2),
        )
        for k in range(n)
    ]


@app.command(name="run-series")
def run_series(
    slug: str = typer.Option(..., "--slug", help="Base project slug."),
    video_source: str = typer.Option(..., "--video-source", help="Video URL or local path."),
    parts: int = typer.Option(3, "--parts", help="Number of chronological parts."),
    srt_source: Optional[str] = typer.Option(None, "--srt-source", help="Optional subtitle URL."),
    style: str = typer.Option("commentary", "--style", help="commentary | recap"),
    publish_target: str = typer.Option("artifact_only", "--publish-target", help="Delivery target."),
    force: bool = typer.Option(True, "--force", help="Force regenerate."),
) -> None:
    """Render an N-part chronological series sequentially."""
    for k in range(1, parts + 1):
        typer.echo(f"Rendering {slug} Part {k}/{parts}...")
        run(
            config=Path("config.yaml"),
            slug=slug,
            video_source=video_source,
            srt_source=srt_source,
            mode="full_auto",
            style=style,
            publish_target=publish_target,
            mock_llm=False,
            mock_tts=False,
            mock_vision=False,
            force=force,
            split_part=k,
            total_parts=parts,
            max_duration=175,
            allow_reproduce=True,
        )


def run_chapter_commentary(
    cfg: dict[str, Any],
    slug: str,
    source_path: Path,
    rights_rec: Any,
    allowed_range: tuple[float, float],
    split_part: int,
    total_parts: int,
    project_dir: Path,
    work_dir: Path,
    out_dir: Path,
    *,
    mock_llm: bool = False,
    mock_tts: bool = False,
) -> Path:
    """Execute uncut chapter_commentary mode with Kiko host booth."""
    plan_path = project_dir / "chapters_plan.json"
    if plan_path.exists() and split_part > 1:
        from pipeline.models import ChapterSlice
        slices = [ChapterSlice(**x) for x in json.loads(plan_path.read_text(encoding="utf-8"))]
    else:
        slices = chapter_slicer.plan_chapters(
            source_path, allowed_range[0], allowed_range[1], total_parts if split_part > 0 else 0, cfg
        )
        project_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps([s.model_dump() for s in slices], indent=2) + "\n", encoding="utf-8")

    idx = max(0, min(len(slices) - 1, (split_part - 1) if split_part > 0 else 0))
    ch_slice = slices[idx]
    sheet = chapter_slicer.extract_labeled_contact_sheet(
        source_path, ch_slice.start_sec, ch_slice.end_sec,
        float(cfg.get("chapter_commentary", {}).get("contact_sheet_interval_sec", 3.0)),
        work_dir / f"chapter_{ch_slice.part_number}_sheet.jpg",
    )
    script = chapter_commentator.generate_chapter_commentary(
        sheet, ch_slice, rights_rec.work_title, cfg, mock_llm=mock_llm
    )
    full_wav, timings_json, ass_path = chapter_commentator.synthesize_timed_commentary(
        script, ch_slice.duration_sec, work_dir, cfg, mock_tts=mock_tts
    )
    timings = json.loads(timings_json.read_text(encoding="utf-8"))
    host_mp4 = host_animator.render_host_booth_video(
        full_wav, timings, script.genre, ch_slice.duration_sec, work_dir / "host_booth.mp4", cfg
    )
    final_mp4 = out_dir / "final_short.mp4"
    chapter_composer.compose_chapter_short(
        source_path, ch_slice, host_mp4, full_wav, ass_path,
        rights_rec, script.genre, final_mp4, work_dir, cfg,
    )
    return final_mp4


@app.command(name="build-calendar")
def build_calendar_cmd(
    target_days: int = typer.Option(365, "--target-days", help="Target number of daily slots."),
    include_youtube_cc: bool = typer.Option(False, "--include-youtube-cc", help="Include yt-dlp CC-BY."),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Use deterministic scores."),
) -> None:
    """Build release_calendar_365.json and RELEASE_CALENDAR_365.md."""
    from scripts.build_365_calendar import build_calendar
    cal = build_calendar(
        target_days=target_days,
        include_youtube_cc=include_youtube_cc,
        mock_llm=mock_llm,
    )
    typer.echo(f"Generated {cal.total_slots} calendar slots in release_calendar_365.json.")


@app.command(name="run-next-day")
def run_next_day(
    publish_target: str = typer.Option("artifact_only", "--publish-target", help="Delivery target."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Select next slot without rendering."),
) -> None:
    """Pick the next scheduled day from release_calendar_365.json and render it."""
    from pipeline.models import ReleaseCalendar365, RightsRecord

    cal_path = Path("release_calendar_365.json")
    if not cal_path.exists():
        typer.echo("release_calendar_365.json not found.")
        raise typer.Exit(code=1)

    cal = ReleaseCalendar365(**json.loads(cal_path.read_text(encoding="utf-8")))
    h = history.load()
    is_gha = os.environ.get("GITHUB_ACTIONS") == "true"

    chosen = None
    for slot in sorted(cal.slots, key=lambda s: s.day_number):
        if slot.status != "scheduled":
            continue
        if is_gha and slot.runner_target != "github_actions":
            continue
        rec = history.is_duplicate(h, slug=slot.base_slug, work_title=slot.work_title)
        if rec is not None and history.is_published_anywhere(rec):
            slot.status = "published"
            continue
        if slot.license == "public_domain" and not slot.pd_verified_by_human:
            slot.status = "awaiting_pd_verification"
            typer.echo(f"Day {slot.day_number} ({slot.work_title}) awaiting PD verification: {slot.source_page_url}")
            continue
        chosen = slot
        break

    cal_path.write_text(cal.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if chosen is None:
        typer.echo("No eligible scheduled slots remaining.")
        return

    typer.echo(f"Selected Day {chosen.day_number}: {chosen.work_title} (Part {chosen.part_number}/{chosen.total_parts})")
    if dry_run:
        return

    proj_dir = Path("projects") / chosen.base_slug
    proj_dir.mkdir(parents=True, exist_ok=True)
    if not (proj_dir / "rights.json").exists():
        rr = RightsRecord(
            work_title=chosen.work_title,
            creator=chosen.source_provider,
            year=chosen.year or 2020,
            license=chosen.license,
            license_url=chosen.license_url,
            source_page_url=chosen.source_page_url,
            commercial_use_allowed=True,
            derivatives_allowed=True,
            attribution_text=chosen.attribution_text,
            pd_verified_by_human=chosen.pd_verified_by_human,
            exclude_head_sec=chosen.exclude_head_sec,
            verified_by="calendar_365",
            verified_on=dt.date.today().isoformat(),
        )
        (proj_dir / "rights.json").write_text(rr.model_dump_json(indent=2) + "\n", encoding="utf-8")

    run(
        config=Path("config.yaml"),
        slug=chosen.base_slug,
        video_source=chosen.video_url,
        srt_source=chosen.srt_url,
        mode="full_auto",
        style="commentary",
        publish_target=publish_target,
        mock_llm=False,
        mock_tts=False,
        mock_vision=False,
        force=True,
        split_part=0 if chosen.total_parts == 1 else chosen.part_number,
        total_parts=chosen.total_parts,
        max_duration=175,
        allow_reproduce=True,
    )
    chosen.status = "rendered"
    cal_path.write_text(cal.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
