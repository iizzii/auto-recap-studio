"""Genre-adaptive timestamped commentary generator and timed TTS synthesizer."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline import caption_engine, ffutil
from pipeline.models import Beat, ChapterCommentaryScript, ChapterSlice, CommentaryCue

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "chapter_commentary_writer.txt"

_GENRE_VOICES: dict[str, tuple[str, str]] = {
    "comedy": ("en-US-AndrewNeural", "+10%"),
    "horror_comedy": ("en-US-BrianNeural", "+9%"),
    "action_scifi": ("en-US-BrianNeural", "+12%"),
    "emotional_drama": ("en-US-AndrewNeural", "+2%"),
}


def generate_chapter_commentary(
    contact_sheet_path: Path,
    chapter_slice: ChapterSlice,
    work_title: str,
    config: dict[str, Any],
    *,
    subtitles_text: str = "",
    previous_cliffhanger: str = "",
    mock_llm: bool = False,
) -> ChapterCommentaryScript:
    """Generate a ChapterCommentaryScript from a labeled contact sheet."""
    dur = chapter_slice.duration_sec
    if mock_llm:
        emotions = ["shocked", "laughing", "smug", "hyped", "facepalm", "neutral"]
        n_cues = 5
        step = max(2.0, (dur - 2.0) / n_cues)
        cues = [
            CommentaryCue(
                cue_id=i + 1,
                start_sec=round(i * step, 2),
                max_duration_sec=round(max(1.5, step - 0.5), 2),
                text=f"Watch what happens right here in scene {i + 1}!",
                emotion=emotions[i % len(emotions)],  # type: ignore[arg-type]
                pause_emotion=emotions[(i + 1) % len(emotions)],  # type: ignore[arg-type]
            )
            for i in range(n_cues)
        ]
        v_name, v_rate = _GENRE_VOICES["comedy"]
        return ChapterCommentaryScript(
            genre="comedy",
            voice_name=v_name,
            voice_rate=v_rate,
            chapter_title=f"{work_title} (Part {chapter_slice.part_number}/{chapter_slice.total_parts})",
            cliffhanger_summary="Things are about to get wild!" if chapter_slice.part_number < chapter_slice.total_parts else "",
            cues=cues,
        )

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return generate_chapter_commentary(
            contact_sheet_path, chapter_slice, work_title, config, mock_llm=True
        )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        sys_prompt = _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""
        meta = (
            f"WORK_TITLE: {work_title}\n"
            f"PART_NUMBER: {chapter_slice.part_number}/{chapter_slice.total_parts}\n"
            f"CHAPTER_DURATION: {dur:.1f}s\n"
            f"PREVIOUS_CLIFFHANGER: {previous_cliffhanger or 'None'}\n"
            f"SUBTITLES:\n{subtitles_text[:1500]}"
        )
        img = Image.open(contact_sheet_path)
        resp = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[f"{sys_prompt}\n\n{meta}", img],
            config={"response_mime_type": "application/json", "response_schema": ChapterCommentaryScript},
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp.text or "").strip())
        script = ChapterCommentaryScript(**json.loads(raw))
        v_name, v_rate = _GENRE_VOICES.get(script.genre, ("en-US-AndrewNeural", "+10%"))
        script.voice_name = v_name
        script.voice_rate = v_rate
        return script
    except Exception as exc:
        log.warning("Chapter commentary LLM failed (%s); using mock fallback.", exc)
        return generate_chapter_commentary(
            contact_sheet_path, chapter_slice, work_title, config, mock_llm=True
        )


async def _edge_tts_to_mp3(text: str, voice: str, rate: str, out_mp3: Path) -> None:
    import edge_tts
    comm = edge_tts.Communicate(text, voice=voice, rate=rate)
    await comm.save(str(out_mp3))


def _synth_cue_wav(
    text: str, voice: str, rate: str, out_wav: Path, mock_tts: bool = False, sr: int = 48000
) -> float:
    """Synthesize a single cue to 48kHz mono WAV and return its duration in seconds."""
    if mock_tts:
        dur = max(0.8, min(3.0, len(text.split()) * 0.25))
        n = round(dur * sr)
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        tone = 0.25 * np.sin(2.0 * np.pi * 280.0 * t)
        pcm = np.clip(tone * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return dur

    tmp_mp3 = out_wav.with_suffix(".mp3")
    asyncio.run(_edge_tts_to_mp3(text, voice, rate, tmp_mp3))
    ffutil.run(["ffmpeg", "-y", "-i", str(tmp_mp3), "-ar", str(sr), "-ac", "1", str(out_wav)])
    return ffutil.duration(out_wav)


def synthesize_timed_commentary(
    script: ChapterCommentaryScript,
    chapter_duration: float,
    out_dir: Path,
    config: dict[str, Any],
    mock_tts: bool = False,
) -> tuple[Path, Path, Path]:
    """Synthesize cues onto exact chapter_duration timeline -> (full_wav, timings_json, ass_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cc_cfg = config.get("chapter_commentary", {})
    min_gap = float(cc_cfg.get("min_gap_between_lines_sec", 0.35))
    max_speed = float(cc_cfg.get("max_line_speedup", 1.12))
    sr = 48000

    total_samples = max(sr, round(chapter_duration * sr))
    timeline_audio = np.zeros(total_samples, dtype=np.float32)

    placed_timings: list[dict[str, Any]] = []
    cues = sorted(script.cues, key=lambda c: c.start_sec)
    cursor_sec = 0.0

    for idx, cue in enumerate(cues):
        raw_wav = out_dir / f"cue_{cue.cue_id:03d}.wav"
        dur = _synth_cue_wav(cue.text, script.voice_name, script.voice_rate, raw_wav, mock_tts=mock_tts, sr=sr)

        target_start = max(cursor_sec, min(cue.start_sec, chapter_duration - 0.5))
        next_start = cues[idx + 1].start_sec if idx + 1 < len(cues) else chapter_duration
        avail = max(0.4, (next_start - target_start) - min_gap)

        if dur > avail:
            needed_speed = min(max_speed, dur / avail)
            if needed_speed > 1.01:
                sped_wav = out_dir / f"cue_{cue.cue_id:03d}_sped.wav"
                ffutil.run([
                    "ffmpeg", "-y", "-i", str(raw_wav),
                    "-af", f"atempo={needed_speed:.3f}",
                    "-ar", str(sr), "-ac", "1", str(sped_wav),
                ])
                raw_wav = sped_wav
                dur = ffutil.duration(raw_wav)

        with wave.open(str(raw_wav), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

        s_idx = min(total_samples - 1, max(0, round(target_start * sr)))
        e_idx = min(total_samples, s_idx + len(pcm))
        if e_idx > s_idx:
            timeline_audio[s_idx:e_idx] += pcm[: e_idx - s_idx]

        actual_end = round(min(chapter_duration, target_start + (e_idx - s_idx) / sr), 3)
        cursor_sec = actual_end + min_gap
        placed_timings.append({
            "cue_id": cue.cue_id,
            "start_sec": round(target_start, 3),
            "end_sec": actual_end,
            "text": cue.text,
            "emotion": cue.emotion,
            "pause_emotion": cue.pause_emotion,
        })

    full_wav = out_dir / "03_commentary_full.wav"
    out_pcm = np.clip(timeline_audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(full_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_pcm.tobytes())

    timings_json = out_dir / "03_cue_timings.json"
    timings_json.write_text(json.dumps(placed_timings, indent=2) + "\n", encoding="utf-8")

    # Build 05_captions.ass using caption_engine single_word style positioned at seam
    cap_cfg = dict(config.get("captions", {}))
    cap_cfg["mode"] = "single_word"
    cap_cfg["margin_v"] = 985
    fake_beats = [
        Beat(
            beat_id=t["cue_id"],
            narration=t["text"],
            source_window_start_sec=t["start_sec"],
            source_window_end_sec=max(t["start_sec"] + 0.2, t["end_sec"]),
            visual_cue="host commentary",
            mood="hook",
            kind="commentary",
            audio_duration_sec=max(0.2, t["end_sec"] - t["start_sec"]),
        )
        for t in placed_timings
    ]
    ass_path = out_dir / "05_captions.ass"
    if fake_beats:
        gen_ass, _ = caption_engine.generate_captions(
            "chapter", fake_beats, full_wav, out_dir, {"captions": cap_cfg}, mock_tts=True
        )
        # Shift each Dialogue line by its cue's actual start_sec
        lines = gen_ass.read_text(encoding="utf-8").splitlines()
        # Note: mock_tts=True in generate_captions places beats sequentially;
        # for exact cue_timings, we emit single_word events directly per placed_timing
        header_end = next((i + 1 for i, ln in enumerate(lines) if ln.startswith("Format: Layer")), len(lines))
        header = "\n".join(lines[:header_end])
        events: list[str] = []
        for t in placed_timings:
            words = [w.strip("—–- ") for w in re.sub(r"[—–/]+", " ", t["text"]).split() if w.strip("—–- ")]
            if not words:
                continue
            step = (t["end_sec"] - t["start_sec"]) / len(words)
            for wi, w in enumerate(words):
                ws = t["start_sec"] + wi * step
                we = max(ws + 0.08, t["start_sec"] + (wi + 1) * step)
                events.append(
                    f"Dialogue: 0,{caption_engine._ass_time(ws)},{caption_engine._ass_time(we)},Default,,0,0,0,,{w.upper()}"
                )
        ass_path.write_text(header + "\n" + "\n".join(events) + "\n", encoding="utf-8")
    else:
        ass_path.write_text("", encoding="utf-8")

    return full_wav, timings_json, ass_path
