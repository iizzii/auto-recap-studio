#!/usr/bin/env python3
"""Generate 3D Pixar-style Kiko 2.0 host sprite pack (31+ 640x640 RGBA) and 4 dark cinema booths (1080x700)."""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kiko_assets_ai")

SPRITES_DIR = Path("assets/host/sprites")
BOOTHS_DIR = Path("assets/host/booths")
PREVIEW_PATH = Path("assets/host/host_preview_sheet.jpg")

SPRITE_SIZE = 640
BOOTH_SIZE = (1080, 700)

EMOTIONS: tuple[str, ...] = (
    "neutral",
    "laughing",
    "shocked",
    "horrified",
    "smug",
    "hyped",
    "sad",
    "facepalm",
    "disgusted",
    "suspicious",
)
MOUTHS: tuple[str, ...] = ("closed", "half", "open")

GEMINI_IMAGE_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-exp-image-generation",
)

IMAGEN_FALLBACK_MODELS: tuple[str, ...] = (
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-3.0-generate-002",
)

MASTER_PORTRAIT_PROMPT = (
    "3D Pixar and DreamWorks animated feature film character render (waist-up studio portrait) of Kiko 2.0, "
    "an ultra-expressive charismatic ginger-amber and cream tabby cat movie commentator facing the camera "
    "with a lively attentive expression and relaxed happy cat smile. "
    "Richly sculpted fluffy ginger-amber tabby fur with warm cream cheeks and muzzle, a playful messy ginger hair tuft "
    "between large expressive cat ears, huge glossy emerald-green eyes with crisp studio catchlights, cute small pink nose, "
    "wearing a sleek dark charcoal streetwear hoodie with cyan and gold metallic drawcord tips, "
    "matte-black over-ear studio headphones with subtle glowing cyan-teal LED rings, "
    "and a compact matte-black broadcast boom microphone extending cleanly near his left cheek. "
    "Both fluffy ginger-and-cream feline paws visible/usable near the lower chest for expressive acting "
    "(always furry feline cat paws with tiny rounded toe beans — NEVER human hands or fingers). "
    "Cinematic three-point warm key + subtle cyan rim studio lighting, centered waist/chest-up framing "
    "with generous headroom above the ears (at least 18% empty headroom) and hoodie shoulders reaching the bottom edge. "
    "Standing against a 100% solid, completely flat, uniform, shadowless pure neon chroma-key green (#00FF00) background. "
    "NO studio background, NO desk, NO chair, NO props, NO text, NO border, NO watermark."
)

EMOTION_DIRECTIONS: dict[str, str] = {
    "neutral": (
        "friendly relaxed charismatic commentator expression, ears perked upright, bright glossy emerald eyes "
        "looking directly at camera, comfortable relaxed posture in dark charcoal hoodie with one fluffy cat paw resting loosely near chest"
    ),
    "laughing": (
        "hilarious genuine wheezing belly laugh, head tilted slightly back and to the side, eyes squeezed joyfully shut "
        "in happy curved arcs with tiny sparkling happy tears, raised cheeks, one fluffy ginger cat paw clutching his hoodie belly "
        "and the other furry cat paw waving outward in laughter (always furry cat paws, NEVER human hands)"
    ),
    "shocked": (
        "jaw-dropping comedic shock, head slightly pulled back, huge wide emerald eyes with tiny startled pupils, "
        "eyebrows shot high up, ears bolted straight upright, BOTH fluffy ginger-and-cream cat paws flung up near his cheeks "
        "in classic stunned disbelief (always soft furry feline cat paws, NEVER human hands)"
    ),
    "horrified": (
        "dramatic comedic horror and dread, hunched tense hoodie shoulders, wide terrified emerald eyes with tiny pupils, "
        "furrowed worried brows, ears pinned slightly back, BOTH fluffy cat paws gripping the sides of his studio headphones "
        "as he recoils in terror (always soft furry feline cat paws, NEVER human hands)"
    ),
    "smug": (
        "knowing sarcastic smug smirk, head cocked cheekily to one side, one eyebrow arched playfully high, "
        "half-lidded confident emerald eyes glancing side-eye at camera, one fluffy ginger cat paw raised pointing slyly "
        "in an 'I told you so' gesture (always soft furry feline cat paw, NEVER human fingers)"
    ),
    "hyped": (
        "maximum hype excitement, leaning eagerly forward, gleaming sparkling wide emerald eyes, eager raised brows, "
        "ears perked high, BOTH fluffy ginger cat paws pumped high in excited little cat fists near/above ear height "
        "(always soft furry feline cat paws, NEVER human hands)"
    ),
    "sad": (
        "dramatic heartbroken puppy-cat sympathy, head slightly bowed, big glistening glossy teary emerald eyes, "
        "droopy tilted ears, soft downturned brows, BOTH fluffy cat paws clasped gently together in front of his charcoal hoodie chest "
        "(always soft furry feline cat paws, NEVER human hands)"
    ),
    "facepalm": (
        "exasperated comedic disbelief facepalm: head tilted down in cringe, squinting exasperated eyes, "
        "right fluffy orange-ginger and cream feline cat paw (clearly a soft furry cat paw with tiny rounded pink toe beans — "
        "NEVER a human hand or fingers!) slapped squarely onto his forehead/brow while headphones and boom mic stay cleanly intact"
    ),
    "disgusted": (
        "comedic grossed-out 'ew' cringe reaction, head recoiled slightly back and left, nose visibly scrunched up, "
        "asymmetrical disgusted squint and furrowed brow, one fluffy ginger cat paw raised defensively palm-out pushing away in disgust "
        "(clearly a soft furry feline cat paw with pink paw pads — NEVER a human hand)"
    ),
    "suspicious": (
        "sharp detective suspicion, head tilted forward in intense scrutiny, eyes narrowed into sharp suspicious emerald slits "
        "glancing sideways, one eyebrow lowered sharply, one fluffy ginger cat paw raised thoughtfully stroking his chin/muzzle "
        "(always soft furry feline cat paw, NEVER human hands)"
    ),
}

MOUTH_DIRECTIONS: dict[str, str] = {
    "closed": "mouth cleanly CLOSED (expression-matched closed cat mouth or smirk/grimace, zero teeth showing)",
    "half": "mouth slightly PARTED mid-word (small natural half-open oval talking mouth, speaking normally)",
    "open": "mouth WIDE OPEN mid-exclamation or laugh (expressive animated open talking mouth showing tiny cute cat tongue)",
}

BOOTH_PROMPTS: dict[str, str] = {
    "comedy": (
        "3D Pixar animated film dark atmospheric movie-reaction podcast booth background (1080x700 widescreen). "
        "Deep dark charcoal-walnut vertical acoustic slat walls heavily in shadow (dramatically darker than a movie screen), "
        "very subtle dim warm amber tungsten corner rim glow and tiny soft out-of-focus warm practical accent light far right. "
        "Empty shadowed dark center-left space where a bright ginger host sits. "
        "STRICTLY dark moody low-key lighting (low overall brightness). NO bright walls, NO pastel/lavender colors, "
        "NO tile grids, NO floating cartoon circles, NO desk, NO characters, NO cats, NO text, shallow depth of field."
    ),
    "horror_comedy": (
        "3D Pixar animated film dark atmospheric midnight cinema-commentary booth background (1080x700 widescreen). "
        "Deep obsidian charcoal vertical acoustic slat walls in moody darkness (significantly darker than a movie screen), "
        "subtle dim teal-emerald edge accent strip far right and a faint moody midnight-indigo shadow wash. "
        "Empty dark shadowed center-left space where a bright ginger host sits. "
        "STRICTLY dark moody low-key cinema lighting. NO bright purple/lavender walls, NO checkerboard tile grids, "
        "NO floating circles or orbs, NO desk, NO characters, NO cats, NO text, shallow depth of field."
    ),
    "action_scifi": (
        "3D Pixar animated film dark futuristic sci-fi cinema-commentary booth background (1080x700 widescreen). "
        "Deep matte obsidian graphite architectural vertical paneling in dark shadow (significantly darker than a movie screen), "
        "subtle thin dim electric-cyan recessed vertical accent line on the far right edge, soft cinematic dark haze. "
        "Empty dark shadowed center-left space where a bright ginger host sits. "
        "STRICTLY dark low-key lighting. NO bright walls, NO grid wall patterns, NO floating circles, "
        "NO desk, NO characters, NO cats, NO text, shallow depth of field."
    ),
    "emotional_drama": (
        "3D Pixar animated film dark intimate late-night cinema-commentary booth background (1080x700 widescreen). "
        "Deep midnight charcoal-navy velvet acoustic drapery and dark wood slats in soft shadow (significantly darker than a movie screen), "
        "one tiny dim warm golden practical tungsten edge highlight far right. "
        "Empty dark shadowed center-left space where a bright ginger host sits. "
        "STRICTLY dark low-key lighting. NO bright walls, NO grid tiles, NO floating circles, "
        "NO desk, NO characters, NO cats, NO text, shallow depth of field."
    ),
}


def _extract_image_from_response(resp: object) -> Image.Image | None:
    """Pull the first inline image returned by Gemini generate_content."""
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return Image.open(io.BytesIO(inline.data)).convert("RGBA")
    return None


def _discover_image_models(client: object) -> tuple[list[str], list[str]]:
    """Query client.models.list() to discover live Gemini image-gen and Imagen models."""
    gen_models: list[str] = list(GEMINI_IMAGE_MODELS)
    im_models: list[str] = list(IMAGEN_FALLBACK_MODELS)
    try:
        discovered_img: list[str] = []
        discovered_im: list[str] = []
        for m in client.models.list():  # type: ignore[attr-defined]
            raw_name = str(getattr(m, "name", "") or "")
            short = raw_name.removeprefix("models/")
            low = short.lower()
            if "embedding" in low or "veo" in low:
                continue
            if "imagen" in low:
                discovered_im.append(short)
            elif "image" in low or "banana" in low:
                discovered_img.append(short)
        log.info("Discovered Gemini image models: %s | Imagen models: %s", discovered_img, discovered_im)
        for name in discovered_img:
            if name not in gen_models:
                gen_models.append(name)
        for name in discovered_im:
            if name not in im_models:
                im_models.append(name)
    except Exception as exc:
        log.warning("models.list() inspection skipped: %s", str(exc)[:180])
    return gen_models, im_models


def _is_zero_quota_error(msg: str) -> bool:
    low = msg.lower()
    return ("limit: 0" in low) or ("limit:0" in low) or ("free_tier" in low and "quota" in low)


def _generate_image_gemini(
    clients: list[tuple[str, object]],
    prompt: str,
    *,
    gemini_models: list[str],
    imagen_models: list[str],
    dead_pairs: set[tuple[str, str]],
    ref_png_bytes: bytes | None = None,
    aspect_ratio: str = "1:1",
    label: str = "asset",
) -> Image.Image:
    """Generate or edit an image via Gemini native image models across keys, falling back to Imagen."""
    from google.genai import types

    contents: list[object] = []
    if ref_png_bytes is not None:
        contents.append(types.Part.from_bytes(data=ref_png_bytes, mime_type="image/png"))
    contents.append(prompt)

    backoff = (4.0, 10.0, 20.0)
    errors: list[str] = []

    for key_lbl, client in clients:
        for model_name in gemini_models:
            if (key_lbl, model_name) in dead_pairs:
                continue
            for attempt in range(4):
                try:
                    log.info("%s: calling %s (%s, attempt %d/4)", label, model_name, key_lbl, attempt + 1)
                    resp = client.models.generate_content(  # type: ignore[attr-defined]
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["TEXT", "IMAGE"],
                        ),
                    )
                    img = _extract_image_from_response(resp)
                    if img is not None:
                        return img
                    raise RuntimeError(f"Model {model_name} returned no inline image part")
                except Exception as exc:
                    full_msg = str(exc).replace("\n", " ")
                    errors.append(f"{key_lbl}/{model_name}[{attempt + 1}]: {full_msg[:260]}")
                    if "404" in full_msg or "NOT_FOUND" in full_msg or "not supported" in full_msg.lower():
                        log.warning("%s: %s/%s unavailable (404) -> skipping", label, key_lbl, model_name)
                        dead_pairs.add((key_lbl, model_name))
                        break
                    if _is_zero_quota_error(full_msg):
                        log.warning("%s: %s/%s has zero image quota (%s) -> skipping", label, key_lbl, model_name, full_msg[:220])
                        dead_pairs.add((key_lbl, model_name))
                        break
                    if attempt < 3:
                        delay = backoff[attempt] + random.uniform(0.0, 1.5)
                        log.warning("%s: transient error on %s/%s (%s); retrying in %.1fs", label, key_lbl, model_name, full_msg[:180], delay)
                        time.sleep(delay)
                        continue
                    break

    if ref_png_bytes is None:
        for key_lbl, client in clients:
            for im_model in imagen_models:
                if (key_lbl, im_model) in dead_pairs:
                    continue
                try:
                    log.info("%s: fallback Imagen %s (%s)", label, im_model, key_lbl)
                    im_resp = client.models.generate_images(  # type: ignore[attr-defined]
                        model=im_model,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio=aspect_ratio,
                            output_mime_type="image/png",
                        ),
                    )
                    gen_imgs = getattr(im_resp, "generated_images", None) or []
                    if gen_imgs and getattr(gen_imgs[0], "image", None) and gen_imgs[0].image.image_bytes:
                        return Image.open(io.BytesIO(gen_imgs[0].image.image_bytes)).convert("RGBA")
                except Exception as exc:
                    full_msg = str(exc).replace("\n", " ")
                    errors.append(f"{key_lbl}/{im_model}: {full_msg[:220]}")
                    if "404" in full_msg or "400" in full_msg or "403" in full_msg or _is_zero_quota_error(full_msg):
                        dead_pairs.add((key_lbl, im_model))

    raise RuntimeError(f"Gemini image generation unavailable for '{label}': {' | '.join(errors[-4:])}")


def chroma_key_and_align(
    raw_img: Image.Image,
    target_size: int = SPRITE_SIZE,
) -> tuple[Image.Image, float]:
    """Border-seeded chroma-key removal protecting Kiko's emerald eyes, edge despill & 640x640 alignment.

    Returns (aligned_rgba_image, foreground_opaque_ratio).
    """
    rgb = np.array(raw_img.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape
    rf = rgb[:, :, 0].astype(np.float32)
    gf = rgb[:, :, 1].astype(np.float32)
    bf = rgb[:, :, 2].astype(np.float32)
    max_rb = np.maximum(rf, bf)
    g_excess = gf - max_rb

    # Measure actual generated corner chroma color (handles slight #00FF00 drift)
    patch = max(8, min(24, min(h, w) // 20))
    corners = np.concatenate(
        [
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, -patch:].reshape(-1, 3),
            rgb[-patch:, :patch].reshape(-1, 3),
            rgb[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg_med = np.median(corners, axis=0)
    corner_dist = np.sqrt(np.sum((rgb.astype(np.float32) - bg_med.reshape(1, 1, 3)) ** 2, axis=2))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    is_green_chroma = (
        ((hue >= 34) & (hue <= 88) & (sat >= 48) & (val >= 40) & (g_excess > 16.0))
        | ((g_excess > 38.0) & (gf > 90.0))
        | ((corner_dist < 58.0) & (g_excess > 8.0))
    )

    # Border-seeded connected components ONLY -> 100% protects enclosed emerald eyes & teal LEDs!
    _num_labels, labels = cv2.connectedComponents(is_green_chroma.astype(np.uint8), connectivity=4)
    border_labels: set[int] = set()
    border_labels.update(int(x) for x in np.unique(labels[0, :]))
    border_labels.update(int(x) for x in np.unique(labels[:, 0]))
    border_labels.update(int(x) for x in np.unique(labels[:, -1]))
    # Bottom-left and bottom-right corners (outer 22% each side) also seed background
    b_margin = max(4, int(w * 0.22))
    border_labels.update(int(x) for x in np.unique(labels[-1, :b_margin]))
    border_labels.update(int(x) for x in np.unique(labels[-1, -b_margin:]))
    border_labels.discard(0)

    bg_mask = np.isin(labels, list(border_labels))

    # Also allow tiny pure-neon-#00FF00 holes trapped outside the eye/face zone (e.g., mic loop)
    eye_zone = np.zeros((h, w), dtype=bool)
    eye_zone[int(h * 0.20) : int(h * 0.62), int(w * 0.22) : int(w * 0.78)] = True
    pure_neon_hole = (g_excess > 95.0) & (rf < 95.0) & (bf < 95.0) & (~eye_zone)
    bg_mask = bg_mask | pure_neon_hole

    fg_binary = (~bg_mask).astype(np.uint8)
    # Remove tiny stray 1-2px green-screen speckles
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    fg_clean = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, kernel3)

    # Soft anti-aliased edge alpha along 3px outer boundary of foreground
    dist_in = cv2.distanceTransform(fg_clean, cv2.DIST_L2, 3)
    alpha = (fg_clean.astype(np.float32)) * 255.0
    edge_band = (fg_clean == 1) & (dist_in <= 3.2)
    edge_soft = np.clip(1.0 - np.maximum(0.0, g_excess - 10.0) / 52.0, 0.0, 1.0) * 255.0
    alpha = np.where(edge_band, np.minimum(alpha, edge_soft), alpha)

    # Light 1px anti-alias feather + 1px dark-edge cleanup on semi-transparent boundary pixels
    alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 0.75)
    near_edge = (dist_in <= 2.5) & (dist_in > 0.0)
    alpha = np.where(near_edge, alpha_blur, alpha)
    alpha = np.where(bg_mask, 0.0, alpha)

    # Edge green-despill ONLY within 6px of background (keeps emerald eyes 100% rich & vibrant!)
    despill_band = (fg_clean == 1) & (dist_in <= 6.5) & (g_excess > 4.0)
    out_r = rf.copy()
    out_g = gf.copy()
    out_b = bf.copy()
    if np.any(despill_band):
        spill = np.maximum(0.0, gf - max_rb)
        new_g = np.minimum(gf, max_rb + 4.0)
        # Warm neutral energy replacement so ginger fur edges stay silky warm
        out_g = np.where(despill_band, new_g, out_g)
        out_r = np.where(despill_band, np.clip(rf + spill * 0.22, 0.0, 255.0), out_r)
        out_b = np.where(despill_band, np.clip(bf + spill * 0.08, 0.0, 255.0), out_b)

    # 1px dark/halo fringe cleanup on outer translucent edge by pulling inner clean fur color outward
    outer_rim = (alpha > 0.0) & (alpha < 190.0) & (dist_in <= 1.8)
    if np.any(outer_rim):
        inner_rgb = np.stack([out_r, out_g, out_b], axis=2).astype(np.uint8)
        dilated = cv2.dilate(inner_rgb, kernel3, iterations=1).astype(np.float32)
        out_r = np.where(outer_rim, 0.55 * out_r + 0.45 * dilated[:, :, 0], out_r)
        out_g = np.where(outer_rim, 0.55 * out_g + 0.45 * dilated[:, :, 1], out_g)
        out_b = np.where(outer_rim, 0.55 * out_b + 0.45 * dilated[:, :, 2], out_b)

    rgba = np.stack(
        [
            np.clip(out_r, 0, 255).astype(np.uint8),
            np.clip(out_g, 0, 255).astype(np.uint8),
            np.clip(out_b, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        ],
        axis=2,
    )

    # Crop to foreground bounding box & align consistently by chest/shoulder anchor (do NOT squish raised paws!)
    opaque = rgba[:, :, 3] > 32
    opaque_ratio = float(opaque.mean())
    ys, xs = np.where(opaque)
    if len(xs) == 0 or len(ys) == 0:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0)), 0.0

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop_arr = rgba[y0:y1, x0:x1]
    crop_h, crop_w = crop_arr.shape[:2]

    # Measure chest/shoulder horizontal span in the lower torso band (bottom 28% of silhouette)
    band_top = max(0, int(crop_h * 0.72))
    band_opaque = crop_arr[band_top:, :, 3] > 32
    _by, bx = np.where(band_opaque)
    shoulder_w = float(bx.max() - bx.min() + 1) if len(bx) > 8 else float(crop_w)
    shoulder_cx = float((bx.min() + bx.max()) * 0.5) if len(bx) > 8 else float(crop_w) * 0.5

    # Anchor shoulder width to ~62% of canvas so tall poses (hyped fists / shocked ears) keep full torso scale
    target_shoulder_w = target_size * 0.62
    scale_by_shoulder = target_shoulder_w / max(1.0, shoulder_w)
    max_fit_scale = min((target_size * 0.96) / max(1, crop_w), (target_size * 0.94) / max(1, crop_h))
    min_fit_scale = min((target_size * 0.78) / max(1, crop_w), (target_size * 0.78) / max(1, crop_h))
    scale = min(max_fit_scale, max(min_fit_scale, scale_by_shoulder))

    cropped = Image.fromarray(crop_arr, mode="RGBA")
    new_w = max(1, round(cropped.width * scale))
    new_h = max(1, round(cropped.height * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    # Align shoulder center horizontally to canvas center, clamped cleanly inside bounds
    scaled_cx = round(shoulder_cx * scale)
    paste_x = max(0, min(target_size - new_w, (target_size // 2) - scaled_cx))
    paste_y = target_size - new_h
    canvas.alpha_composite(resized, (paste_x, paste_y))
    return canvas, opaque_ratio


def _normalize_dark_booth(img: Image.Image) -> Image.Image:
    """Ensure every 1080x700 cinema booth stays dramatically darker than the movie above."""
    fitted = ImageOps.fit(img.convert("RGB"), BOOTH_SIZE, method=Image.Resampling.LANCZOS)
    arr = np.array(fitted, dtype=np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mean_luma = float(luma.mean())
    if mean_luma > 56.0:
        gain = 46.0 / max(mean_luma, 1.0)
        arr = np.clip(arr * gain, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _generate_valid_sprite(
    clients: list[tuple[str, object]],
    prompt: str,
    *,
    gemini_models: list[str],
    imagen_models: list[str],
    dead_pairs: set[tuple[str, str]],
    ref_png_bytes: bytes | None,
    label: str,
) -> tuple[Image.Image, bytes]:
    """Generate, chroma-key, validate opaque ratio, and return (aligned_640_rgba, raw_png_bytes)."""
    aligned = Image.new("RGBA", (SPRITE_SIZE, SPRITE_SIZE), (0, 0, 0, 0))
    fg_ratio = 0.0
    raw = aligned
    for attempt in range(3):
        raw = _generate_image_gemini(
            clients,
            prompt,
            gemini_models=gemini_models,
            imagen_models=imagen_models,
            dead_pairs=dead_pairs,
            ref_png_bytes=ref_png_bytes,
            aspect_ratio="1:1",
            label=f"{label}_gen{attempt + 1}",
        )
        aligned, fg_ratio = chroma_key_and_align(raw, target_size=SPRITE_SIZE)
        if 0.14 <= fg_ratio <= 0.88:
            buf = io.BytesIO()
            raw.convert("RGB").save(buf, format="PNG")
            return aligned, buf.getvalue()
        log.warning(
            "%s: foreground ratio %.2f out of bounds [0.14, 0.88] on attempt %d/3; regenerating...",
            label,
            fg_ratio,
            attempt + 1,
        )
        time.sleep(2.0)

    if fg_ratio > 0.05:
        buf = io.BytesIO()
        raw.convert("RGB").save(buf, format="PNG")
        return aligned, buf.getvalue()
    raise RuntimeError(f"Sprite '{label}' failed chroma-key validation (fg_ratio={fg_ratio:.3f}).")


def _run_blender_3d_fallback(sprites_dir: Path, booths_dir: Path, samples: int = 28) -> None:
    """Render full 32-sprite Kiko 2.0 + 4 dark 1080x700 booth pack in headless Blender Cycles."""
    blender_bin = shutil.which("blender")
    if not blender_bin:
        raise RuntimeError(
            "All Gemini image models returned 0 quota / 404 AND 'blender' is not installed for 3D fallback."
        )
    script_path = Path(__file__).resolve().parent / "render_kiko_blender_3d.py"
    log.warning(
        "Gemini image generation quota unavailable across configured keys -> "
        "rendering true 3D Pixar-style Kiko 2.0 10-pose pack + 4 dark 1080x700 booths via headless Blender Cycles (%s)...",
        blender_bin,
    )
    cmd = [
        blender_bin,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script_path),
        "--",
        "--sprites-dir",
        str(sprites_dir),
        "--booths-dir",
        str(booths_dir),
        "--samples",
        str(samples),
    ]
    subprocess.run(cmd, check=True)


def _build_preview_sheet(sprites_dir: Path, booths_dir: Path, out_jpg: Path) -> Path:
    """Build a visual verification contact sheet of all Kiko 2.0 sprites and 4 dark cinema booths."""
    sprite_files = sorted(sprites_dir.glob("*.png"))
    booth_files = sorted(booths_dir.glob("*.png"))

    cols = 6
    tile = 210
    s_rows = max(1, (len(sprite_files) + cols - 1) // cols)
    booth_h = 200
    total_w = cols * tile
    total_h = s_rows * tile + (booth_h + 24 if booth_files else 0) + 40

    sheet = Image.new("RGB", (total_w, total_h), (14, 16, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), f"KIKO 2.0 3D ASSET PACK ({len(sprite_files)} sprites + {len(booth_files)} dark 1080x700 booths)", fill=(255, 216, 77))

    for idx, sp in enumerate(sprite_files):
        r, c = divmod(idx, cols)
        x0 = c * tile
        y0 = 36 + r * tile
        draw.rectangle([x0 + 3, y0 + 3, x0 + tile - 4, y0 + tile - 4], fill=(22, 26, 38), outline=(48, 54, 72))
        spr = Image.open(sp).convert("RGBA").resize((tile - 16, tile - 28), Image.Resampling.LANCZOS)
        bg_cell = Image.new("RGBA", (tile - 16, tile - 28), (22, 26, 38, 255))
        bg_cell.alpha_composite(spr, (0, 0))
        sheet.paste(bg_cell.convert("RGB"), (x0 + 8, y0 + 22))
        draw.text((x0 + 10, y0 + 6), sp.stem, fill=(235, 240, 255))

    if booth_files:
        by0 = 44 + s_rows * tile
        bw = total_w // max(1, len(booth_files))
        for idx, bp in enumerate(booth_files):
            bx0 = idx * bw
            b_img = ImageOps.fit(Image.open(bp).convert("RGB"), (bw - 10, booth_h - 24), Image.Resampling.LANCZOS)
            sheet.paste(b_img, (bx0 + 5, by0 + 22))
            draw.text((bx0 + 10, by0 + 4), f"booth: {bp.stem}", fill=(46, 230, 197))

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_jpg, format="JPEG", quality=88)
    log.info("Wrote visual preview sheet -> %s", out_jpg)
    return out_jpg


def _needs_regen(path: Path, expected_size: tuple[int, int], force: bool) -> bool:
    if force or not path.exists():
        return True
    try:
        with Image.open(path) as im:
            return im.size != expected_size
    except Exception:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3D Pixar-style Kiko 2.0 sprites & dark 1080x700 booths.")
    parser.add_argument("--force", action="store_true", help="Regenerate all sprites and booths even if present.")
    parser.add_argument("--blender-samples", type=int, default=26, help="Cycles samples if Blender 3D renderer runs.")
    args = parser.parse_args()

    keys: list[tuple[str, str]] = []
    for env_name in ("GEMINI_API_KEY", "GEMINI_API_KEY1"):
        val = os.environ.get(env_name, "").strip()
        if val and all(val != existing for _, existing in keys):
            keys.append((env_name, val))

    SPRITES_DIR.mkdir(parents=True, exist_ok=True)
    BOOTHS_DIR.mkdir(parents=True, exist_ok=True)

    clients: list[tuple[str, object]] = []
    gemini_models = list(GEMINI_IMAGE_MODELS)
    imagen_models = list(IMAGEN_FALLBACK_MODELS)
    dead_pairs: set[tuple[str, str]] = set()

    if keys:
        from google import genai
        from google.genai import types

        for lbl, k_val in keys:
            c = genai.Client(api_key=k_val, http_options=types.HttpOptions(timeout=120_000))
            clients.append((lbl, c))
        gemini_models, imagen_models = _discover_image_models(clients[0][1])

    # 1. Try Tier 1: Gemini Native Image Generation + Reference-Conditioned Editing
    used_blender_fallback = False
    master_path = SPRITES_DIR / "neutral_closed.png"
    master_raw_cache = Path(tempfile.gettempdir()) / "kiko_v2_master_raw.png"

    try:
        if not clients:
            raise RuntimeError("No GEMINI_API_KEY configured.")

        if _needs_regen(master_path, (SPRITE_SIZE, SPRITE_SIZE), args.force) or not master_raw_cache.exists():
            log.info("Generating Master 3D Pixar Kiko 2.0 portrait (neutral_closed) via Gemini...")
            master_img, master_raw_bytes = _generate_valid_sprite(
                clients,
                MASTER_PORTRAIT_PROMPT,
                gemini_models=gemini_models,
                imagen_models=imagen_models,
                dead_pairs=dead_pairs,
                ref_png_bytes=None,
                label="neutral_closed",
            )
            master_img.save(master_path, format="PNG")
            master_raw_cache.write_bytes(master_raw_bytes)
            time.sleep(1.5)
        else:
            master_raw_bytes = master_raw_cache.read_bytes()
            log.info("Reusing existing 640x640 master sprite %s", master_path)

        # 2. Generate all 10 emotions x closed/open consistently from Master Reference
        for emotion in EMOTIONS:
            for mouth in ("closed", "open"):
                if emotion == "neutral" and mouth == "closed":
                    continue
                out_path = SPRITES_DIR / f"{emotion}_{mouth}.png"
                if not _needs_regen(out_path, (SPRITE_SIZE, SPRITE_SIZE), args.force):
                    log.info("Skipping existing 640x640 sprite %s", out_path.name)
                    continue
                edit_prompt = (
                    "Using this EXACT same 3D Pixar-style ginger-amber tabby cat character Kiko 2.0 from the reference image, "
                    "preserve the IDENTICAL camera distance, head scale, shoulder position, ginger-and-cream tabby fur pattern, "
                    "messy ginger hair tuft between ears, dark charcoal streetwear hoodie with cyan/gold drawcord tips, "
                    "matte-black over-ear headphones with cyan LED rings, black boom microphone, studio lighting, and "
                    "100% solid flat shadowless pure chroma-key green (#00FF00) background. "
                    f"Modify ONLY Kiko's facial expression, head tilt, fluffy cat paw pose, and mouth to match:\n"
                    f"- Expression & Pose: {EMOTION_DIRECTIONS[emotion]}.\n"
                    f"- Mouth: {MOUTH_DIRECTIONS[mouth]}.\n"
                    "Keep clean fur silhouette against pure flat #00FF00 green with NO desk, NO props, NO text."
                )
                spr_img, _ = _generate_valid_sprite(
                    clients,
                    edit_prompt,
                    gemini_models=gemini_models,
                    imagen_models=imagen_models,
                    dead_pairs=dead_pairs,
                    ref_png_bytes=master_raw_bytes,
                    label=f"{emotion}_{mouth}",
                )
                spr_img.save(out_path, format="PNG")
                time.sleep(1.5)

        # 3. Generate natural blink sprite (saved as both neutral_blink.png and blink.png)
        blink_path = SPRITES_DIR / "blink.png"
        neutral_blink_path = SPRITES_DIR / "neutral_blink.png"
        if _needs_regen(neutral_blink_path, (SPRITE_SIZE, SPRITE_SIZE), args.force) or _needs_regen(
            blink_path, (SPRITE_SIZE, SPRITE_SIZE), args.force
        ):
            blink_prompt = (
                "Using this EXACT same 3D Pixar-style ginger tabby cat Kiko 2.0 from the reference image, keep the IDENTICAL "
                "pose, head position, dark charcoal hoodie, fur markings, headphones, boom mic, lighting, and solid flat #00FF00 green background. "
                "Change ONLY the eyes so BOTH eyes are softly gently closed in a natural mid-blink with soft curved dark lashes, "
                "and keep the relaxed friendly closed cat smile."
            )
            blink_img, _ = _generate_valid_sprite(
                clients,
                blink_prompt,
                gemini_models=gemini_models,
                imagen_models=imagen_models,
                dead_pairs=dead_pairs,
                ref_png_bytes=master_raw_bytes,
                label="neutral_blink",
            )
            blink_img.save(neutral_blink_path, format="PNG")
            blink_img.save(blink_path, format="PNG")
            time.sleep(1.5)

        # 4. Generate half-open talking mouth sprites for all 10 emotions
        for emotion in EMOTIONS:
            half_path = SPRITES_DIR / f"{emotion}_half.png"
            if not _needs_regen(half_path, (SPRITE_SIZE, SPRITE_SIZE), args.force):
                continue
            closed_p = SPRITES_DIR / f"{emotion}_closed.png"
            open_p = SPRITES_DIR / f"{emotion}_open.png"
            try:
                half_prompt = (
                    "Using this EXACT same 3D Pixar-style ginger tabby cat Kiko 2.0 from the reference image, keep the IDENTICAL "
                    "camera framing, head position, dark charcoal hoodie, fur markings, headphones, boom mic, lighting, and solid flat #00FF00 green background. "
                    f"Match this expression & pose: {EMOTION_DIRECTIONS[emotion]}. "
                    f"Mouth shape: {MOUTH_DIRECTIONS['half']}. "
                    "Keep clean silhouette on pure flat #00FF00 green with NO text or props."
                )
                half_img, _ = _generate_valid_sprite(
                    clients,
                    half_prompt,
                    gemini_models=gemini_models,
                    imagen_models=imagen_models,
                    dead_pairs=dead_pairs,
                    ref_png_bytes=master_raw_bytes,
                    label=f"{emotion}_half",
                )
                half_img.save(half_path, format="PNG")
                time.sleep(1.2)
            except Exception as exc:
                log.warning("%s_half Gemini generation fell back to smooth blend (%s)", emotion, exc)
                c_im = Image.open(closed_p).convert("RGBA")
                o_im = Image.open(open_p).convert("RGBA")
                Image.blend(c_im, o_im, 0.5).save(half_path, format="PNG")

        # 5. Generate 4 dark moody 3D cinema commentary booths (1080x700)
        for genre, b_prompt in BOOTH_PROMPTS.items():
            b_path = BOOTHS_DIR / f"{genre}.png"
            if not _needs_regen(b_path, BOOTH_SIZE, args.force):
                log.info("Skipping existing booth %s", b_path.name)
                continue
            raw_booth = _generate_image_gemini(
                clients,
                b_prompt,
                gemini_models=gemini_models,
                imagen_models=imagen_models,
                dead_pairs=dead_pairs,
                ref_png_bytes=None,
                aspect_ratio="16:9",
                label=f"booth_{genre}",
            )
            dark_fitted = _normalize_dark_booth(raw_booth)
            dark_fitted.save(b_path, format="PNG")
            time.sleep(1.5)
    except Exception as gemini_exc:
        log.warning("Tier-1 Gemini image pipeline unavailable: %s", gemini_exc)
        _run_blender_3d_fallback(SPRITES_DIR, BOOTHS_DIR, samples=args.blender_samples)
        used_blender_fallback = True

    # 6. Build visual verification sheet & verify exact asset counts
    _build_preview_sheet(SPRITES_DIR, BOOTHS_DIR, PREVIEW_PATH)

    n_sprites = len(list(SPRITES_DIR.glob("*.png")))
    n_booths = len(list(BOOTHS_DIR.glob("*.png")))
    backend = "Blender 3D Cycles" if used_blender_fallback else "Gemini Native Image AI"
    log.info("Generated %d sprites (640x640 RGBA) and %d dark booths (1080x700) via %s.", n_sprites, n_booths, backend)
    if n_sprites < 31 or n_booths < 4:
        raise RuntimeError(f"Asset pack incomplete: {n_sprites} sprites (<31) or {n_booths} booths (<4).")


if __name__ == "__main__":
    main()
