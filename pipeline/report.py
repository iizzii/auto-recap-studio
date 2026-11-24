"""Run report persistence and GitHub Actions $GITHUB_STEP_SUMMARY rendering."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pipeline.models import RunReport


def write_run_report(
    report: RunReport,
    project_dir: Path,
    out_dir: Path | None = None,
) -> Path:
    """Persist ``run_report.json`` to *project_dir* (and *out_dir* if provided)."""
    project_dir.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump_json(indent=2) + "\n"
    proj_path = project_dir / "run_report.json"
    proj_path.write_text(payload, encoding="utf-8")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_report.json").write_text(payload, encoding="utf-8")
    return proj_path


def write_github_summary(report: RunReport) -> None:
    """Append a Markdown summary of *report* to ``$GITHUB_STEP_SUMMARY`` if set."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_path:
        return

    lines = [
        f"## Recap Pipeline Report: `{report.slug}`",
        "",
        f"- **Status:** `{report.status}` (exit code `{report.exit_code}`)",
        f"- **Duration:** `{report.duration_sec or 0:.2f}s`",
        f"- **Stages:** {', '.join(report.stages_completed) or 'none'}",
    ]
    if report.warnings:
        lines.append("\n### Warnings")
        lines.extend(f"- {w}" for w in report.warnings)
    if report.errors:
        lines.append("\n### Errors")
        lines.extend(f"- {e}" for e in report.errors)
    if report.publish_results:
        lines.append("\n### Publish Results")
        lines.append(f"```json\n{json.dumps(report.publish_results, indent=2)}\n```")

    with Path(summary_path).open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
