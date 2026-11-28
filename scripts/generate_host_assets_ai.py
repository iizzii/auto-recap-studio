#!/usr/bin/env python3
"""Generate 3D Pixar-style Kiko host sprite pack (25+ 640x640 RGBA) and 4 genre booths (1080x900) via Gemini."""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
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
BOOTH_SIZE = (1080, 900)

EMOTIONS: tuple[str, ...] = (
    "neutral",
    "laughing",
    "shocked",
    "horrified",
    "smug",
    "hyped",
    "sad",
    "facepalm",
)
MOUTHS: tuple[str, ...] = ("closed", "half", "open")

GEMINI_IMAGE_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.0-flash-preview-image-generation",
)

IMAGEN_FALLBACK_MODELS: tuple[str, ...] = (
    "imagen-3.0-generate-002",
    "imagen-4.0-generate-001",
    "imagen-3.0-generate-001",
)

MASTER_PORTRAIT_PROMPT = (
    "3D Pixar and DreamWorks animated feature film style waist-up character portrait of Kiko, "
    "an adorable charismatic orange-ginger and cream tabby cat podcast commentator facing the camera "
    "with a friendly relaxed attentive expression and closed happy cat smile. "
    "Richly sculpted soft fluffy ginger tabby fur with warm cream cheeks, muzzle, and chest fur, "
    "large expressive glossy emerald-green eyes with warm studio catchlights, cute small pink nose, "
    "wearing sleek matte-charcoal over-ear studio headphones with subtle glowing cyan-teal LED trim rings "
    "and a compact matte-black broadcast boom microphone extending cleanly near his left cheek. "
    "Soft cinematic three-point warm studio lighting on the character, centered waist-up framing "
    "with generous headroom above ears and fluffy shoulders reaching the bottom edge. "
    "Standing against a 100% solid, completely flat, uniform, shadowless pure neon chroma-key green (#00FF00) background. "
    "NO studio background, NO desk, NO chair, NO props, NO text, NO border, NO watermark."
)

EMOTION_DIRECTIONS: dict[str, str] = {
    "neutral": "friendly relaxed attentive commentator expression, ears perked upright, bright emerald eyes looking at camera",
    "laughing": "hilarious genuine belly laugh, eyes happily squeezed shut in joyful curved arcs, cheeks raised, whiskers perky, ears slightly tilted back with mirth",
    "shocked": "jaw-dropping comedic shock, emerald eyes super wide open with tiny surprised pupils, eyebrows shot high up, ears bolted straight up",
    "horrified": "dramatic comedic horror and dread, wide terrified emerald eyes, furrowed worried eyebrows, ears pinned slightly back, tense hunched shoulders",
    "smug": "knowing sarcastic smug smirk, one eyebrow arched playfully high, half-lidded confident emerald eyes glancing cheekily at camera",
    "hyped": "maximum hype excitement, gleaming sparkling wide emerald eyes, eager raised eyebrows, energetic leaned-forward posture, ears perked high",
    "sad": "dramatic heartbroken puppy-cat sympathy, big glistening glossy teary emerald eyes, droopy tilted ears, soft downturned brows",
    "facepalm": (
        "exasperated comedic disbelief facepalm: Kiko raises ONE fluffy orange-ginger and cream feline cat paw "
        "(clearly a soft furry cat paw with tiny rounded toe beans/paw edge — NEVER a human hand or fingers!) "
        "up against his forehead/brow while squinting in disbelief, headphones and boom mic still cleanly intact"
    ),
}

MOUTH_DIRECTIONS: dict[str, str] = {
    "closed": "mouth cleanly CLOSED (relaxed cat smile or expression-matched closed mouth, zero teeth showing)",
    "half": "mouth slightly PARTED mid-word (small natural half-open oval talking mouth, speaking normally)",
    "open": "mouth WIDE OPEN mid-exclamation or laugh (expressive animated open talking mouth showing tiny cute cat tongue)",
}

BOOTH_PROMPTS: dict[str, str] = {
    "comedy": (
        "3D Pixar and DreamWorks animated film background render of a cozy vibrant movie-commentary podcast studio booth "
        "for a comedy show. Warm golden amber key lighting, glowing soft out-of-focus string fairy lights on dark wood "
        "acoustic paneling, subtle vintage framed film posters and a red-and-white striped cinema popcorn bucket softly blurred "
        "on a side shelf on the far right. Empty studio space on the left-center where a host will be composited later. "
        "NO characters, NO cats, NO people, NO text, NO logos, cinematic shallow depth of field, rich Pixar color grading."
    ),
    "horror_comedy": (
        "3D Pixar and DreamWorks animated film background render of a spooky-fun midnight movie-commentary studio booth. "
        "Moody deep twilight purple shadows, glowing teal-emerald neon accent light strips along matte acoustic wall tiles, "
        "whimsical tiny glowing forest mushroom nightlight softly blurred on the far right shelf, subtle atmospheric haze. "
        "Empty studio space on the left-center where a host will be composited later. "
        "NO characters, NO cats, NO people, NO text, NO logos, cinematic shallow depth of field, rich Pixar lighting."
    ),
    "action_scifi": (
        "3D Pixar and DreamWorks animated film background render of a sleek futuristic sci-fi movie-reaction command booth. "
        "Brushed dark graphite geometric acoustic panels, glowing electric cyan and cobalt blue recessed light strips, "
        "softly blurred glowing holographic audio visualizer bars on the far right console. "
        "Empty studio space on the left-center where a host will be composited later. "
        "NO characters, NO cats, NO people, NO text, NO logos, cinematic shallow depth of field, crisp Pixar lighting."
    ),
    "emotional_drama": (
        "3D Pixar and DreamWorks animated film background render of an intimate cinematic late-night cinema commentary booth. "
        "Deep midnight indigo velvet acoustic curtains, warm golden practical Edison bulb bokeh lights, rain-dappled window "
        "glow softly out of focus on the far right, warm cozy storytelling atmosphere. "
        "Empty studio space on the left-center where a host will be composited later. "
        "NO characters, NO cats, NO people, NO text, NO logos, cinematic shallow depth of field, lush Pixar lighting."
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


def _generate_image_gemini(
    client: object,
    prompt: str,
    *,
    ref_png_bytes: bytes | None = None,
    aspect_ratio: str = "1:1",
    label: str = "asset",
) -> Image.Image:
    """Generate or edit an image via Gemini native image models, falling back to Imagen if needed."""
    from google.genai import types

    contents: list[object] = []
    if ref_png_bytes is not None:
        contents.append(types.Part.from_bytes(data=ref_png_bytes, mime_type="image/png"))
    contents.append(prompt)

    backoff = (4.0, 10.0, 22.0)
    errors: list[str] = []

    for model_name in GEMINI_IMAGE_MODELS:
        for attempt in range(4):
            try:
                log.info("%s: calling %s (attempt %d/4)", label, model_name, attempt + 1)
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
                msg = str(exc).replace("\n", " ")[:240]
                errors.append(f"{model_name}[{attempt + 1}]: {msg}")
                if "404" in msg or "NOT_FOUND" in msg or "not supported" in msg.lower():
                    log.warning("%s: %s unavailable -> next model", label, model_name)
                    break
                if attempt < 3:
                    delay = backoff[attempt] + random.uniform(0.0, 1.5)
                    log.warning("%s: transient error (%s); retrying in %.1fs", label, msg[:120], delay)
                    time.sleep(delay)
                    continue
                break

    # Fallback to Imagen text-to-image if native Gemini image models are unavailable
    if ref_png_bytes is None:
        for im_model in IMAGEN_FALLBACK_MODELS:
            try:
                log.info("%s: fallback Imagen %s", label, im_model)
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
                errors.append(f"{im_model}: {str(exc)[:180]}")

    raise RuntimeError(f"Image generation failed for '{label}': {' | '.join(errors[-4:])}")


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

    # Light 1px anti-alias feather on boundary only
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

    rgba = np.stack(
        [
            np.clip(out_r, 0, 255).astype(np.uint8),
            np.clip(out_g, 0, 255).astype(np.uint8),
            np.clip(out_b, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        ],
        axis=2,
    )

    # Crop to foreground bounding box & align consistently bottom-anchored inside target_size x target_size
    opaque = rgba[:, :, 3] > 32
    opaque_ratio = float(opaque.mean())
    ys, xs = np.where(opaque)
    if len(xs) == 0 or len(ys) == 0:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0)), 0.0

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = Image.fromarray(rgba[y0:y1, x0:x1], mode="RGBA")

    # Fit cleanly to 91% height, max 92% width, horizontally centered & bottom-anchored
    max_h = int(target_size * 0.91)
    max_w = int(target_size * 0.92)
    scale = min(max_w / max(1, cropped.width), max_h / max(1, cropped.height))
    new_w = max(1, round(cropped.width * scale))
    new_h = max(1, round(cropped.height * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = target_size - new_h
    canvas.alpha_composite(resized, (paste_x, paste_y))
    return canvas, opaque_ratio


def _generate_valid_sprite(
    client: object,
    prompt: str,
    *,
    ref_png_bytes: bytes | None,
    label: str,
) -> tuple[Image.Image, bytes]:
    """Generate, chroma-key, validate opaque ratio, and return (aligned_640_rgba, raw_png_bytes)."""
    for attempt in range(3):
        raw = _generate_image_gemini(
            client,
            prompt,
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

    # Return last best effort only if non-empty
    if fg_ratio > 0.05:
        buf = io.BytesIO()
        raw.convert("RGB").save(buf, format="PNG")
        return aligned, buf.getvalue()
    raise RuntimeError(f"Sprite '{label}' failed chroma-key validation (fg_ratio={fg_ratio:.3f}).")


def _build_preview_sheet(sprites_dir: Path, booths_dir: Path, out_jpg: Path) -> Path:
    """Build a visual verification contact sheet of all Kiko sprites and 4 genre booths."""
    sprite_files = sorted(sprites_dir.glob("*.png"))
    booth_files = sorted(booths_dir.glob("*.png"))

    cols = 5
    tile = 220
    s_rows = max(1, (len(sprite_files) + cols - 1) // cols)
    booth_h = 210
    total_w = cols * tile
    total_h = s_rows * tile + (booth_h + 24 if booth_files else 0) + 40

    sheet = Image.new("RGB", (total_w, total_h), (18, 20, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), f"KIKO 3D ASSET PACK ({len(sprite_files)} sprites + {len(booth_files)} booths)", fill=(255, 216, 77))

    # Draw dark studio backdrop squares behind each RGBA sprite to inspect chroma-key edge quality
    for idx, sp in enumerate(sprite_files):
        r, c = divmod(idx, cols)
        x0 = c * tile
        y0 = 36 + r * tile
        draw.rectangle([x0 + 3, y0 + 3, x0 + tile - 4, y0 + tile - 4], fill=(28, 32, 46), outline=(52, 58, 78))
        spr = Image.open(sp).convert("RGBA").resize((tile - 16, tile - 28), Image.Resampling.LANCZOS)
        bg_cell = Image.new("RGBA", (tile - 16, tile - 28), (28, 32, 46, 255))
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
    parser = argparse.ArgumentParser(description="Generate 3D Pixar-style Kiko sprites & genre booths via Gemini.")
    parser.add_argument("--force", action="store_true", help="Regenerate all sprites and booths even if present.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.error("GEMINI_API_KEY is required to generate 3D host assets.")
        sys.exit(1)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
    SPRITES_DIR.mkdir(parents=True, exist_ok=True)
    BOOTHS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Master Reference Portrait (neutral_closed) on solid #00FF00 green screen
    master_path = SPRITES_DIR / "neutral_closed.png"
    master_raw_cache = Path("/tmp/kiko_master_raw.png")

    if _needs_regen(master_path, (SPRITE_SIZE, SPRITE_SIZE), args.force) or not master_raw_cache.exists():
        log.info("Generating Master 3D Pixar Kiko portrait (neutral_closed)...")
        master_img, master_raw_bytes = _generate_valid_sprite(
            client, MASTER_PORTRAIT_PROMPT, ref_png_bytes=None, label="neutral_closed"
        )
        master_img.save(master_path, format="PNG")
        master_raw_cache.write_bytes(master_raw_bytes)
        time.sleep(1.5)
    else:
        master_raw_bytes = master_raw_cache.read_bytes()
        log.info("Reusing existing 640x640 master sprite %s", master_path)

    # 2. Generate all 8 emotions x closed/open + blink consistently from Master Reference
    for emotion in EMOTIONS:
        for mouth in ("closed", "open"):
            if emotion == "neutral" and mouth == "closed":
                continue
            out_path = SPRITES_DIR / f"{emotion}_{mouth}.png"
            if not _needs_regen(out_path, (SPRITE_SIZE, SPRITE_SIZE), args.force):
                log.info("Skipping existing 640x640 sprite %s", out_path.name)
                continue
            edit_prompt = (
                "Using this EXACT same 3D Pixar-style orange-ginger tabby cat character Kiko from the reference image, "
                "preserve the IDENTICAL camera distance, head scale, shoulder position, ginger-and-cream tabby fur pattern, "
                "matte-charcoal headphones with cyan-teal LED rings, black boom microphone, studio lighting, and "
                "100% solid flat shadowless pure chroma-key green (#00FF00) background. "
                f"Modify ONLY Kiko's facial expression, pose, and mouth to match:\n"
                f"- Expression: {EMOTION_DIRECTIONS[emotion]}.\n"
                f"- Mouth: {MOUTH_DIRECTIONS[mouth]}.\n"
                "Keep clean fur silhouette against pure flat #00FF00 green with NO desk, NO props, NO text."
            )
            spr_img, _ = _generate_valid_sprite(
                client, edit_prompt, ref_png_bytes=master_raw_bytes, label=f"{emotion}_{mouth}"
            )
            spr_img.save(out_path, format="PNG")
            time.sleep(1.5)

    # 3. Generate natural blink sprite
    blink_path = SPRITES_DIR / "blink.png"
    if _needs_regen(blink_path, (SPRITE_SIZE, SPRITE_SIZE), args.force):
        blink_prompt = (
            "Using this EXACT same 3D Pixar-style ginger tabby cat Kiko from the reference image, keep the IDENTICAL "
            "pose, head position, fur markings, headphones, boom mic, lighting, and solid flat #00FF00 green background. "
            "Change ONLY the eyes so BOTH eyes are softly gently closed in a natural mid-blink with soft curved dark lashes, "
            "and keep the relaxed friendly closed cat smile."
        )
        blink_img, _ = _generate_valid_sprite(
            client, blink_prompt, ref_png_bytes=master_raw_bytes, label="blink"
        )
        blink_img.save(blink_path, format="PNG")
        time.sleep(1.5)

    # 4. Generate half-open talking mouth sprites for all 8 emotions (with aligned fallback if API rate-limits)
    for emotion in EMOTIONS:
        half_path = SPRITES_DIR / f"{emotion}_half.png"
        if not _needs_regen(half_path, (SPRITE_SIZE, SPRITE_SIZE), args.force):
            continue
        closed_p = SPRITES_DIR / f"{emotion}_closed.png"
        open_p = SPRITES_DIR / f"{emotion}_open.png"
        try:
            half_prompt = (
                "Using this EXACT same 3D Pixar-style ginger tabby cat Kiko from the reference image, keep the IDENTICAL "
                "camera framing, head position, fur markings, headphones, boom mic, lighting, and solid flat #00FF00 green background. "
                f"Match this expression: {EMOTION_DIRECTIONS[emotion]}. "
                f"Mouth shape: {MOUTH_DIRECTIONS['half']}. "
                "Keep clean silhouette on pure flat #00FF00 green with NO text or props."
            )
            half_img, _ = _generate_valid_sprite(
                client, half_prompt, ref_png_bytes=master_raw_bytes, label=f"{emotion}_half"
            )
            half_img.save(half_path, format="PNG")
            time.sleep(1.2)
        except Exception as exc:
            log.warning("%s_half Gemini generation fell back to smooth blend (%s)", emotion, exc)
            c_im = Image.open(closed_p).convert("RGBA")
            o_im = Image.open(open_p).convert("RGBA")
            Image.blend(c_im, o_im, 0.5).save(half_path, format="PNG")

    # 5. Generate 4 lush 3D Pixar-style genre broadcast booths (1080x900)
    for genre, b_prompt in BOOTH_PROMPTS.items():
        b_path = BOOTHS_DIR / f"{genre}.png"
        if not _needs_regen(b_path, BOOTH_SIZE, args.force):
            log.info("Skipping existing booth %s", b_path.name)
            continue
        raw_booth = _generate_image_gemini(
            client, b_prompt, ref_png_bytes=None, aspect_ratio="4:3", label=f"booth_{genre}"
        )
        fitted = ImageOps.fit(raw_booth.convert("RGB"), BOOTH_SIZE, method=Image.Resampling.LANCZOS)
        fitted.save(b_path, format="PNG")
        time.sleep(1.5)

    # 6. Build visual verification sheet & verify exact asset counts
    _build_preview_sheet(SPRITES_DIR, BOOTHS_DIR, PREVIEW_PATH)

    n_sprites = len(list(SPRITES_DIR.glob("*.png")))
    n_booths = len(list(BOOTHS_DIR.glob("*.png")))
    log.info("Generated %d sprites (640x640 RGBA) and %d booths (1080x900).", n_sprites, n_booths)
    if n_sprites < 25 or n_booths < 4:
        raise RuntimeError(f"Asset pack incomplete: {n_sprites} sprites (<25) or {n_booths} booths (<4).")


if __name__ == "__main__":
    main()
