"""End-to-end A/V sync, frame-count, and audio-isolation gate for CI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from pipeline import ffutil

WORK_OUTPUT = Path("/tmp/recap_work/ci_synth/output")
FINAL_MP4 = WORK_OUTPUT / "final_short.mp4"
SHOT_LIST = Path("projects/ci_synth/04_shot_list.json")
CI_CONFIG = Path("tests/ci_config.yaml")


def test_final_video_specs_and_sync() -> None:
    """Verify resolution, fps, codec, pixel format, duration, and A/V sync."""
    assert FINAL_MP4.exists(), f"Missing {FINAL_MP4}"
    cfg = yaml.safe_load(CI_CONFIG.read_text(encoding="utf-8"))
    min_dur = float(cfg["runtime"]["min_duration_sec"])
    max_dur = float(cfg["runtime"]["max_duration_sec"])

    info = ffutil.probe(FINAL_MP4)
    v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    a_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")

    assert int(v_stream["width"]) == 1080
    assert int(v_stream["height"]) == 1920
    assert v_stream["codec_name"] == "h264"
    assert v_stream["pix_fmt"] == "yuv420p"

    num, den = (int(x) for x in v_stream["r_frame_rate"].split("/"))
    assert abs((num / den) - 30.0) < 0.05

    v_dur = float(v_stream["duration"])
    a_dur = float(a_stream["duration"])
    assert min_dur <= v_dur <= max_dur
    assert abs(v_dur - a_dur) <= 0.15


def test_shot_frames_match_video_frames() -> None:
    """Verify sum(shot.frames) equals exact video frame count."""
    shots = json.loads(SHOT_LIST.read_text(encoding="utf-8"))
    expected_frames = sum(int(s["frames"]) for s in shots)
    actual_frames = ffutil.frame_count(FINAL_MP4)
    assert actual_frames == expected_frames


def test_no_long_silence_or_black_frames() -> None:
    """Verify no unintended silence gap > 400ms and no black run > 0.3s."""
    sil = ffutil.run(
        ["ffmpeg", "-i", str(FINAL_MP4), "-af", "silencedetect=noise=-50dB:d=0.4", "-f", "null", "-"],
        check=False,
    )
    assert "silence_duration:" not in sil.stderr

    blk = ffutil.run(
        ["ffmpeg", "-i", str(FINAL_MP4), "-vf", "blackdetect=d=0.3:pix_th=0.10", "-f", "null", "-"],
        check=False,
    )
    assert "black_duration:" not in blk.stderr


def test_no_residual_1000hz_source_audio() -> None:
    """Verify 1000 Hz synthetic source tone is absent while 280 Hz mock TTS is present."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(FINAL_MP4),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64)
    assert len(samples) > 48000
    window = samples[: 48000 * 10]
    fft_mag = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), d=1.0 / 48000.0)

    e_280 = float(np.max(fft_mag[(freqs >= 270) & (freqs <= 290)]))
    e_1000 = float(np.max(fft_mag[(freqs >= 990) & (freqs <= 1010)]))
    assert e_280 > e_1000 * 5.0, f"Expected 280Hz >> 1000Hz, got 280={e_280}, 1000={e_1000}"
