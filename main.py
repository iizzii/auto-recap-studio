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

import typer
import yaml

from pipeline import (
    audio_mixer,
    caption_engine,
    clip_selector,
    ffutil,
    publisher,
    qa_gate,
    report,
    rights_guard,
    script_generator,
    source_fetcher,
    srt_parser,
    transcriber,
    tts_engine,
    video_composer,
)
from pipeline.budget import TimeBudget
from pipeline.models import ApprovalRecord, QueueFile, RunReport

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
    split_part: int = typer.Option(0, "--split-part", help="0=full, 1=part1, 2=part2."),
    max_duration: Optional[int] = typer.Option(None, "--max-duration", help="Max duration sec."),
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
    head_excl = float(src_cfg.get("exclude_head_sec", 20))
    tail_pct = float(src_cfg.get("exclude_tail_pct", 8))
    allowed_start = min(head_excl, src_dur * 0.2)
    allowed_end = max(allowed_start + 5.0, src_dur * (1.0 - tail_pct / 100.0))
    allowed_range = (allowed_start, allowed_end)

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


if __name__ == "__main__":
    app()
