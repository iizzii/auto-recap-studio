"""Generate expressive 2x-supersampled Kiko cat sprites (512x512 RGBA) and genre booths (1080x690 RGB)."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

EMOTIONS = ("neutral", "laughing", "shocked", "horrified", "smug", "hyped", "sad", "facepalm")
MOUTHS = ("closed", "half", "open")

_SPRITES_DIR = Path("assets/host/sprites")
_BOOTHS_DIR = Path("assets/host/booths")


def _draw_kiko(emotion: str, mouth: str, blink: bool = False) -> Image.Image:
    """Draw Kiko at 1024x1024 and downsample 2x to 512x512 RGBA for crisp anti-aliasing."""
    s = 1024
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Ginger & cream fur palette
    fur = (238, 146, 62, 255)
    fur_dark = (196, 108, 36, 255)
    cream = (255, 238, 212, 255)
    hp_dark = (36, 40, 52, 255)
    hp_accent = (0, 210, 255, 255)

    # Shoulders / torso
    d.rounded_rectangle([260, 700, 764, 1020], radius=140, fill=fur)
    d.ellipse([360, 740, 664, 1010], fill=cream)

    # Ears (tilt slightly on sad/horrified)
    ear_drop = 30 if emotion in ("sad", "horrified") else 0
    d.polygon([(240, 380 + ear_drop), (190, 120 + ear_drop), (410, 240)], fill=fur, outline=fur_dark)
    d.polygon([(784, 380 + ear_drop), (834, 120 + ear_drop), (614, 240)], fill=fur, outline=fur_dark)
    d.polygon([(255, 345 + ear_drop), (225, 175 + ear_drop), (370, 255)], fill=(255, 175, 165, 255))
    d.polygon([(769, 345 + ear_drop), (799, 175 + ear_drop), (654, 255)], fill=(255, 175, 165, 255))

    # Headband & Headphones
    d.arc([165, 145, 859, 800], start=190, end=350, fill=hp_dark, width=44)
    d.rounded_rectangle([140, 390, 235, 620], radius=45, fill=hp_dark, outline=hp_accent, width=6)
    d.rounded_rectangle([789, 390, 884, 620], radius=45, fill=hp_dark, outline=hp_accent, width=6)

    # Head
    d.ellipse([215, 220, 809, 770], fill=fur)
    d.ellipse([295, 470, 729, 755], fill=cream)

    # Eyes
    if blink or emotion == "laughing":
        d.arc([325, 405, 455, 475], start=180, end=360, fill=(30, 25, 25, 255), width=14)
        d.arc([569, 405, 699, 475], start=180, end=360, fill=(30, 25, 25, 255), width=14)
    else:
        ew = 64 if emotion in ("shocked", "horrified") else 52
        for cx in (390, 634):
            d.ellipse([cx - ew, 435 - ew, cx + ew, 435 + ew], fill=(255, 255, 255, 255))
            pw = 22 if emotion in ("shocked", "horrified") else 34
            d.ellipse([cx - pw, 435 - pw, cx + pw, 435 + pw], fill=(28, 32, 42, 255))
            d.ellipse([cx - pw + 8, 435 - pw + 6, cx - pw + 22, 435 - pw + 20], fill=(255, 255, 255, 255))

    # Eyebrows
    if emotion == "smug":
        d.line([(335, 365), (445, 345)], fill=fur_dark, width=14)
        d.line([(579, 375), (689, 365)], fill=fur_dark, width=14)
    elif emotion in ("sad", "horrified"):
        d.line([(335, 375), (445, 345)], fill=fur_dark, width=14)
        d.line([(579, 345), (689, 375)], fill=fur_dark, width=14)
    elif emotion == "hyped":
        d.line([(335, 345), (445, 365)], fill=fur_dark, width=14)
        d.line([(579, 365), (689, 345)], fill=fur_dark, width=14)

    # Tears on sad
    if emotion == "sad":
        d.ellipse([350, 495, 376, 545], fill=(110, 200, 255, 220))
        d.ellipse([648, 495, 674, 545], fill=(110, 200, 255, 220))

    # Nose & whiskers
    d.polygon([(492, 520), (532, 520), (512, 548)], fill=(235, 115, 120, 255))
    for y in (535, 560):
        d.line([(240, y - 10), (355, y)], fill=fur_dark, width=6)
        d.line([(669, y), (784, y - 10)], fill=fur_dark, width=6)

    # Mouth
    if mouth == "closed":
        d.arc([460, 565, 564, 615], start=10, end=170, fill=(45, 25, 25, 255), width=10)
    elif mouth == "half":
        d.ellipse([472, 570, 552, 622], fill=(160, 45, 55, 255), outline=(45, 25, 25, 255), width=8)
    else:
        d.ellipse([458, 562, 566, 655], fill=(160, 45, 55, 255), outline=(45, 25, 25, 255), width=8)
        d.ellipse([482, 618, 542, 650], fill=(245, 130, 145, 255))

    # Boom mic
    d.line([(205, 560), (415, 625)], fill=(25, 28, 35, 255), width=14)
    d.ellipse([405, 608, 445, 648], fill=(45, 50, 62, 255), outline=hp_accent, width=4)

    # Facepalm paw
    if emotion == "facepalm":
        d.rounded_rectangle([430, 320, 605, 500], radius=65, fill=cream, outline=fur_dark, width=8)

    return img.resize((512, 512), Image.Resampling.LANCZOS)


def _draw_booth(genre: str) -> Image.Image:
    """Draw a 1080x690 genre-adaptive theater booth background."""
    w, h = 1080, 690
    palettes = {
        "comedy": ((48, 30, 14), (245, 185, 65), (75, 46, 22)),
        "horror_comedy": ((24, 14, 42), (0, 210, 190), (32, 22, 52)),
        "action_scifi": ((14, 20, 36), (0, 225, 255), (26, 32, 48)),
        "emotional_drama": ((16, 26, 52), (245, 175, 85), (28, 38, 64)),
    }
    bg, accent, desk = palettes.get(genre, palettes["comedy"])
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)

    # Soft spotlights / neon strips
    d.ellipse([120, -80, 520, 320], fill=tuple(min(255, c + 28) for c in bg))
    d.ellipse([580, -60, 960, 300], fill=tuple(min(255, c + 20) for c in bg))
    d.rectangle([0, 0, w, 8], fill=accent)

    # Desk foreground
    d.rectangle([0, 540, w, h], fill=desk)
    d.rectangle([0, 540, w, 548], fill=accent)

    # Popcorn bucket silhouette on comedy
    if genre == "comedy":
        d.polygon([(820, 540), (920, 540), (900, 430), (840, 430)], fill=(220, 50, 50))
        d.ellipse([835, 405, 905, 445], fill=(255, 235, 160))
    return img


def main() -> None:
    _SPRITES_DIR.mkdir(parents=True, exist_ok=True)
    _BOOTHS_DIR.mkdir(parents=True, exist_ok=True)

    for em in EMOTIONS:
        for m in MOUTHS:
            _draw_kiko(em, m).save(_SPRITES_DIR / f"{em}_{m}.png")
    _draw_kiko("neutral", "closed", blink=True).save(_SPRITES_DIR / "blink.png")

    for g in ("comedy", "horror_comedy", "action_scifi", "emotional_drama"):
        _draw_booth(g).save(_BOOTHS_DIR / f"{g}.png")


if __name__ == "__main__":
    main()
