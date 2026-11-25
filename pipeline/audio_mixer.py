"""Story-matched light BGM (3-tier license-safe resolver + NumPy musical fallback), SFX, and ducked mixing."""

from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from typing import Any

import numpy as np
import requests

from pipeline import ffutil
from pipeline.models import Beat

log = logging.getLogger(__name__)

_BGM_MANIFEST = Path(__file__).resolve().parent.parent / "assets" / "bgm" / "bgm_manifest.json"
_BGM_CACHE_DIR = Path.home() / ".cache" / "recap_bgm"

_MOOD_BUCKETS = {
    "tense": "suspense",
    "cliffhanger": "suspense",
    "reveal": "suspense",
    "action": "action",
    "hook": "action",
    "emotional": "emotional",
    "calm": "emotional",
    "funny": "playful",
}


def infer_bgm_mood(beats: list[Beat], style: str = "commentary") -> str:
    """Classify the story's dominant musical mood into suspense, action, emotional, or playful."""
    if not beats:
        return "suspense"
    counts: dict[str, int] = {"suspense": 0, "action": 0, "emotional": 0, "playful": 0}
    for b in beats:
        bucket = _MOOD_BUCKETS.get(b.mood, "suspense")
        counts[bucket] += 1
    if counts["playful"] > 0 and counts["playful"] * 2 >= max(counts["suspense"], counts["action"], counts["emotional"]):
        return "playful"
    return max(counts, key=lambda k: counts[k])


def infer_bgm_profile(beats: list[Beat], style: str = "commentary") -> str:
    """Backward-compatible alias mapping to legacy profile names for existing tests."""
    mood = infer_bgm_mood(beats, style=style)
    return {
        "playful": "cartoon_playful",
        "emotional": "emotional_story",
        "suspense": "suspense_action",
        "action": "suspense_action",
    }[mood]


def _synth_bgm_numpy(dur_sec: float, out_wav: Path, mood: str, sr: int = 48000) -> Path:
    """Generate a warm, light, 4-chord musical underscore via pure NumPy with ADSR + stereo width."""
    n_samples = max(sr, int(round(dur_sec * sr)))
    t = np.linspace(0.0, dur_sec, n_samples, endpoint=False, dtype=np.float32)

    progressions = {
        "playful": [(261.63, 329.63, 392.00), (293.66, 349.23, 440.00), (392.00, 493.88, 587.33), (261.63, 329.63, 392.00)],
        "emotional": [(220.00, 261.63, 329.63), (174.61, 220.00, 261.63), (261.63, 329.63, 392.00), (196.00, 246.94, 293.66)],
        "action": [(146.83, 174.61, 220.00), (130.81, 164.81, 196.00), (116.54, 146.83, 174.61), (130.81, 164.81, 196.00)],
        "suspense": [(196.00, 233.08, 293.66), (174.61, 207.65, 261.63), (155.56, 196.00, 233.08), (146.83, 185.00, 220.00)],
    }
    chords = progressions.get(mood, progressions["suspense"])
    chord_dur = 2.5 if mood in ("playful", "action") else 4.0

    audio_l = np.zeros(n_samples, dtype=np.float32)
    audio_r = np.zeros(n_samples, dtype=np.float32)

    chord_idx = (t // chord_dur).astype(np.int32) % len(chords)
    local_t = np.mod(t, chord_dur)

    attack = 0.08
    release = 0.35
    env = np.minimum(local_t / attack, 1.0) * np.minimum((chord_dur - local_t) / release, 1.0)
    env = np.clip(env, 0.0, 1.0)

    pulse_hz = 2.0 if mood in ("playful", "action") else 1.0
    pulse = 0.72 + 0.28 * np.sin(2.0 * np.pi * pulse_hz * t)

    for idx, (f1, f2, f3) in enumerate(chords):
        mask = chord_idx == idx
        if not np.any(mask):
            continue
        tm = t[mask]
        em = env[mask] * pulse[mask]
        pad = (
            0.45 * np.sin(2.0 * np.pi * f1 * tm)
            + 0.28 * np.sin(2.0 * np.pi * f2 * tm)
            + 0.22 * np.sin(2.0 * np.pi * f3 * tm)
            + 0.08 * np.sin(2.0 * np.pi * (f1 * 0.5) * tm)
        ) * em
        pad_r = (
            0.45 * np.sin(2.0 * np.pi * (f1 * 1.0015) * tm)
            + 0.28 * np.sin(2.0 * np.pi * (f2 * 0.9985) * tm)
            + 0.22 * np.sin(2.0 * np.pi * f3 * tm)
            + 0.08 * np.sin(2.0 * np.pi * (f1 * 0.5) * tm)
        ) * em
        audio_l[mask] = pad
        audio_r[mask] = pad_r

    fade_in = min(int(1.0 * sr), n_samples // 4)
    fade_out = min(int(1.5 * sr), n_samples // 4)
    if fade_in > 0:
        ramp_in = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
        audio_l[:fade_in] *= ramp_in
        audio_r[:fade_in] *= ramp_in
    if fade_out > 0:
        ramp_out = np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
        audio_l[-fade_out:] *= ramp_out
        audio_r[-fade_out:] *= ramp_out

    peak = max(float(np.max(np.abs(audio_l))), float(np.max(np.abs(audio_r))), 1e-6)
    scale = 0.35 / peak
    stereo = np.stack([audio_l * scale, audio_r * scale], axis=1)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return out_wav


def _synth_whoosh_sfx(dur_sec: float, out_wav: Path) -> Path:
    """Generate a crisp 0.35 s swept-bandpass noise whoosh at comfortable listening level."""
    actual_dur = min(dur_sec, 0.35)
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=d={actual_dur:.3f}:color=white:amplitude=0.9:seed=42:s=48000",
            "-af",
            (
                f"bandpass=f=1800:width_type=o:w=2.5,"
                f"afade=t=in:st=0:d={actual_dur * 0.25:.3f},"
                f"afade=t=out:st={actual_dur * 0.65:.3f}:d={actual_dur * 0.35:.3f},"
                "volume=0.25"
            ),
            "-ar",
            "48000",
            "-ac",
            "1",
            str(out_wav),
        ]
    )
    return out_wav


def _fetch_ia_bgm_track(mood: str) -> tuple[Path | None, str]:
    """Tier 2: Query Internet Archive for a CC0/CC-BY instrumental track matching mood."""
    cache_dir = _BGM_CACHE_DIR / mood
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = list(cache_dir.glob("*.mp3")) + list(cache_dir.glob("*.ogg"))
    if cached:
        return cached[0], f"Cached IA instrumental ({cached[0].name})"

    try:
        resp = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f"mediatype:audio AND (instrumental OR ambient OR cinematic) AND {mood} AND (licenseurl:cc0 OR licenseurl:creativecommons.org/licenses/by/)",
                "fl[]": "identifier,title,licenseurl",
                "sort[]": "downloads desc",
                "rows": 5,
                "output": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
        for doc in docs:
            lic = str(doc.get("licenseurl", "")).lower()
            if "nc" in lic or "nd" in lic:
                continue
            item_id = doc.get("identifier", "")
            meta = requests.get(f"https://archive.org/metadata/{item_id}", timeout=15).json()
            for f in meta.get("files", []):
                name = str(f.get("name", ""))
                if name.lower().endswith((".mp3", ".ogg")):
                    url = f"https://archive.org/download/{item_id}/{name}"
                    dest = cache_dir / f"{item_id}_{Path(name).name}"
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    attr = f"Music: {doc.get('title', item_id)} ({doc.get('licenseurl', 'CC')})"
                    return dest, attr
    except Exception as exc:
        log.info("Tier 2 IA BGM fetch skipped/failed (%s): %s", mood, exc)
    return None, ""


def _pick_or_synth_bgm(
    dur_sec: float,
    work_dir: Path,
    mood: str,
    allow_network: bool = True,
) -> tuple[Path, str]:
    """3-Tier BGM resolver: Tier 1 manifest -> Tier 2 IA API -> Tier 3 NumPy musical fallback."""
    if _BGM_MANIFEST.exists():
        try:
            data = json.loads(_BGM_MANIFEST.read_text(encoding="utf-8"))
            for track in data.get("tracks", []):
                if track.get("mood") and track.get("mood") != mood:
                    continue
                t_path = Path(track.get("path", ""))
                if t_path.exists():
                    looped = work_dir / "bgm_looped.wav"
                    ffutil.run(
                        [
                            "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(t_path),
                            "-t", f"{dur_sec:.3f}", "-ar", "48000", "-ac", "2", str(looped),
                        ]
                    )
                    return looped, str(track.get("attribution_text", ""))
        except Exception as exc:
            log.warning("Tier 1 BGM manifest error: %s", exc)

    if allow_network:
        ia_path, attr = _fetch_ia_bgm_track(mood)
        if ia_path is not None and ia_path.exists():
            looped = work_dir / "bgm_ia_looped.wav"
            ffutil.run(
                [
                    "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(ia_path),
                    "-t", f"{dur_sec:.3f}", "-ar", "48000", "-ac", "2", str(looped),
                ]
            )
            return looped, attr

    return _synth_bgm_numpy(dur_sec, work_dir / f"bgm_{mood}.wav", mood), ""


def mix_audio(
    voice_master: Path,
    beats: list[Beat],
    work_dir: Path,
    config: dict[str, Any],
    allow_network: bool = True,
) -> Path:
    """Mix voice master + light story-matched BGM (sidechain ducked + vocal pocket EQ) + whoosh SFX."""
    audio_cfg = config.get("audio", {})
    bgm_db = float(audio_cfg.get("bgm_db_below_voice", -21.0))
    whoosh_db = float(audio_cfg.get("sfx_whoosh_db", -16.0))
    whoosh_dur = float(audio_cfg.get("sfx_whoosh_dur_sec", 0.35))
    sfx_enabled = bool(audio_cfg.get("sfx_on_transitions", True))
    target_lufs = float(audio_cfg.get("target_lufs", -14.0))
    true_peak = float(audio_cfg.get("true_peak_db", -1.0))
    sr = int(audio_cfg.get("sample_rate", 48000))
    aac_br = str(audio_cfg.get("aac_bitrate", "192k"))

    voice_dur = ffutil.duration(voice_master)
    mood = infer_bgm_mood(beats)
    log.info("Story BGM mood classified as: %s", mood)

    bgm_wav, _bgm_attr = _pick_or_synth_bgm(voice_dur, work_dir, mood, allow_network=allow_network)

    transition_times: list[float] = []
    if sfx_enabled and len(beats) > 1:
        cur_t = 0.0
        for prev_b, cur_b in zip(beats, beats[1:]):
            cur_t += float(prev_b.audio_duration_sec or 3.0)
            if cur_b.mood != prev_b.mood or (cur_b.kind == "commentary" and prev_b.kind == "story"):
                transition_times.append(cur_t)

    inputs = ["-i", str(voice_master), "-i", str(bgm_wav)]
    filter_parts: list[str] = [
        "[0:a]asplit=2[v_sc][v_mix]",
        f"[1:a]highpass=f=80,equalizer=f=1800:t=q:w=1.2:g=-4,volume={bgm_db}dB[bgm_lvl]",
        "[bgm_lvl][v_sc]sidechaincompress=threshold=0.02:ratio=3.5:attack=15:release=280[bgm_ducked]",
    ]
    mix_labels = ["[v_mix]", "[bgm_ducked]"]

    if transition_times:
        whoosh_wav = _synth_whoosh_sfx(whoosh_dur, work_dir / "whoosh.wav")
        for idx, t_sec in enumerate(transition_times[:6]):
            in_idx = 2 + idx
            inputs.extend(["-i", str(whoosh_wav)])
            delay_ms = max(0, int(round(t_sec * 1000.0)))
            lbl = f"[sfx{idx}]"
            filter_parts.append(
                f"[{in_idx}:a]volume={whoosh_db}dB,adelay={delay_ms}|{delay_ms}{lbl}"
            )
            mix_labels.append(lbl)

    mix_str = "".join(mix_labels)
    filter_parts.append(
        f"{mix_str}amix=inputs={len(mix_labels)}:duration=first:normalize=0,"
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7[out]"
    )

    master_m4a = work_dir / "master_audio.m4a"
    ffutil.run(
        [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]", "-ar", str(sr), "-ac", "2",
            "-c:a", "aac", "-b:a", aac_br,
            str(master_m4a),
        ]
    )
    return master_m4a
