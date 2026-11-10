#!/usr/bin/env python3
"""Render BitakaModz stickers promo clip — 9:16, VO-synced, BG subs + SFX/VFX."""

from __future__ import annotations

import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PRODUCTS = ASSETS / "products"
BRAND = ASSETS / "brand"
SFX = ROOT / "sfx"
AUDIO = ROOT / "audio"
OUT = ROOT / "out"
FRAMES = OUT / "frames"

W, H = 1080, 1920
FPS = 30
DURATION = 19.48  # matches cleaned VO
TOTAL = int(round(DURATION * FPS))

# Timed to Whisper medium + manual BG correction for от това / до това
OT_TOVA_START = 9.62
OT_TOVA_END = 10.90
DO_TOVA_START = 11.10
DO_TOVA_END = 12.40

FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
FONT_FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Corrected Bulgarian captions (spoken intent; Whisper was noisy)
SUBS = [
    (1.00, 3.20, "Променил си визията\nна своята тротинетка"),
    (3.40, 5.30, "но цветът на буквите\nоще е стоков"),
    (5.84, 7.80, "Вече с новите стикери\nза букви от BitakaModz"),
    (7.80, 9.50, "можеш да завършиш визията"),
    (9.50, 10.90, "и от това"),
    (10.98, 12.40, "се получи до това"),
    (12.70, 15.40, "Избери твоя модел и цвят\nот сайта на bitakamodz.com"),
    (15.40, 18.70, "В момента тече нашият\nBlack Friday с огромни намаления"),
]

COLORS = [
    ("#ffe900", "Жълт"),
    ("#ff0000", "Червен"),
    ("#001eff", "Син"),
    ("#c800ff", "Лилав"),
    ("#ffffff", "Бял"),
]


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REG
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype(FONT_FALLBACK, size)


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t * t * t


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def fit_cover(img: Image.Image, size: tuple[int, int], zoom: float = 1.0) -> Image.Image:
    tw, th = size
    tw, th = int(tw * zoom), int(th * zoom)
    fitted = ImageOps.fit(img, (tw, th), method=Image.Resampling.LANCZOS)
    if zoom == 1.0:
        return fitted
    # center-crop back to target
    left = (fitted.width - size[0]) // 2
    top = (fitted.height - size[1]) // 2
    return fitted.crop((left, top, left + size[0], top + size[1]))


def fit_contain(img: Image.Image, box: tuple[int, int], pad: int = 0) -> Image.Image:
    bw, bh = box[0] - pad * 2, box[1] - pad * 2
    copy = img.copy()
    copy.thumbnail((bw, bh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", box, (0, 0, 0, 0))
    canvas.paste(copy, ((box[0] - copy.width) // 2, (box[1] - copy.height) // 2), copy)
    return canvas


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return m


def gradient_bg(t: float) -> Image.Image:
    """Dark charcoal with animated amber/yellow glow (Bitaka brand)."""
    img = Image.new("RGB", (W, H), (8, 8, 10))
    px = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    # drifting glow
    cx = W * (0.35 + 0.3 * math.sin(t * 0.7))
    cy = H * (0.28 + 0.15 * math.cos(t * 0.55))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.clip(1.0 - dist / (W * 0.95), 0, 1) ** 2
    px[:, :, 0] = 8 + glow * 55
    px[:, :, 1] = 8 + glow * 38
    px[:, :, 2] = 10 + glow * 8
    # bottom vignette
    vig = (yy / H) ** 2
    px[:, :, 0] *= 1 - 0.35 * vig
    px[:, :, 1] *= 1 - 0.35 * vig
    px[:, :, 2] *= 1 - 0.45 * vig
    return Image.fromarray(np.clip(px, 0, 255).astype(np.uint8), "RGB")


def paste_center(base: Image.Image, overlay: Image.Image, cy: int, opacity: float = 1.0) -> None:
    if opacity <= 0:
        return
    ov = overlay
    if opacity < 1:
        a = ov.split()[-1].point(lambda p: int(p * opacity))
        ov = ov.copy()
        ov.putalpha(a)
    x = (base.width - ov.width) // 2
    y = cy - ov.height // 2
    base.paste(ov, (x, y), ov)


def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    cy: int,
    fnt: ImageFont.FreeTypeFont,
    fill=(255, 255, 255, 255),
    stroke: int = 0,
    stroke_fill=(0, 0, 0, 255),
    spacing: int = 10,
) -> None:
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = cy - total_h // 2
    for i, line in enumerate(lines):
        x = (W - widths[i]) // 2
        draw.text(
            (x, y),
            line,
            font=fnt,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=stroke_fill,
        )
        y += heights[i] + spacing


def subtitle_block(text: str, progress: float) -> Image.Image:
    """TikTok-style caption with scale-in."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = lerp(0.85, 1.0, ease_out_cubic(progress))
    alpha = int(255 * min(1.0, progress * 3))
    fnt = font(int(54 * scale), bold=True)
    # soft backdrop pill via blur shadow
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    draw_text_centered(sd, text, int(H * 0.78), fnt, fill=(0, 0, 0, 180), stroke=8, stroke_fill=(0, 0, 0, 180))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    layer = Image.alpha_composite(layer, shadow)
    draw = ImageDraw.Draw(layer)
    draw_text_centered(
        draw,
        text,
        int(H * 0.78),
        fnt,
        fill=(255, 240, 80, alpha),
        stroke=4,
        stroke_fill=(0, 0, 0, alpha),
    )
    return layer


def flash_overlay(intensity: float, color=(255, 230, 120)) -> Image.Image:
    a = int(max(0, min(1, intensity)) * 200)
    return Image.new("RGBA", (W, H), (*color, a))


def wipe_mask(progress: float, direction: str = "left") -> Image.Image:
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    if direction == "left":
        x = int(W * ease_in_out(progress))
        d.rectangle((0, 0, x, H), fill=255)
    else:
        x = int(W * (1 - ease_in_out(progress)))
        d.rectangle((x, 0, W, H), fill=255)
    return m


def make_photo_card(photo: Image.Image, label: str, accent: str, zoom: float) -> Image.Image:
    card_w, card_h = 920, 1180
    card = Image.new("RGBA", (card_w, card_h), (18, 18, 22, 255))
    photo_area = fit_cover(photo.convert("RGBA"), (card_w - 40, card_h - 160), zoom=zoom)
    card.paste(photo_area, (20, 20), photo_area)
    # accent bar
    bar = Image.new("RGBA", (card_w, 12), (*tuple(int(accent[i : i + 2], 16) for i in (1, 3, 5)), 255))
    card.paste(bar, (0, card_h - 148), bar)
    draw = ImageDraw.Draw(card)
    fnt = font(48, bold=True)
    bbox = draw.textbbox((0, 0), label, font=fnt)
    tw = bbox[2] - bbox[0]
    draw.text(((card_w - tw) // 2, card_h - 110), label, font=fnt, fill=(255, 255, 255, 255))
    mask = rounded_mask((card_w, card_h), 36)
    card.putalpha(mask)
    # outline
    outline = Image.new("RGBA", (card_w + 8, card_h + 8), (0, 0, 0, 0))
    od = ImageDraw.Draw(outline)
    od.rounded_rectangle((0, 0, card_w + 7, card_h + 7), radius=40, outline=(255, 220, 60, 220), width=4)
    outline.paste(card, (4, 4), card)
    return outline


def color_row(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    n = len(COLORS)
    gap = 24
    size = 96
    total = n * size + (n - 1) * gap
    x0 = (W - total) // 2
    for i, (hex_c, _name) in enumerate(COLORS):
        pulse = 0.5 + 0.5 * math.sin(t * 4 + i)
        r = int(size // 2 + pulse * 4)
        cx = x0 + i * (size + gap) + size // 2
        cy = 80
        rgb = tuple(int(hex_c[j : j + 2], 16) for j in (1, 3, 5))
        draw.ellipse((cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4), fill=(255, 255, 255, 40))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*rgb, 255), outline=(255, 255, 255, 200), width=3)
    return layer


def render_frame(i: int, cache: dict) -> Image.Image:
    t = i / FPS
    bg = gradient_bg(t)
    frame = bg.convert("RGBA")

    logo = cache["logo"]
    logo_mark = cache["logo_mark"]
    stickers = cache["stickers"]
    before = cache["before"]
    after = cache["after"]
    panels = cache["panels"]
    stop = cache["stop"]

    # --- INTRO logo (0 - 1.1s) ---
    if t < 1.15:
        p = ease_out_cubic(t / 0.55)
        scale = lerp(0.6, 1.0, p)
        opacity = min(1.0, t / 0.25) * (1.0 if t < 0.85 else max(0, 1 - (t - 0.85) / 0.3))
        lw = int(720 * scale)
        lh = int(logo.height * (lw / logo.width))
        lg = logo.resize((lw, lh), Image.Resampling.LANCZOS)
        paste_center(frame, lg, int(H * 0.42), opacity)

    # --- PROBLEM / stock lettering (1.0 - 5.5) ---
    if 0.9 < t < 5.7:
        local = t - 0.9
        opacity = ease_out_cubic(min(1, local / 0.35))
        if t > 5.2:
            opacity *= max(0, 1 - (t - 5.2) / 0.5)
        # Ken Burns on stock orange stem (= first attached photo)
        zoom = 1.04 + 0.10 * ease_in_out(min(1, local / 4.5))
        hero = fit_cover(before, (920, 1100), zoom=zoom)
        mask = rounded_mask(hero.size, 28)
        hero.putalpha(mask)
        ang = math.sin(local * 1.1) * 1.2
        hero_r = hero.rotate(ang, resample=Image.Resampling.BICUBIC, expand=True)
        paste_center(frame, hero_r, int(H * 0.40), opacity * 0.98)
        wm = logo_mark.resize(
            (180, int(180 * logo_mark.height / logo_mark.width)),
            Image.Resampling.LANCZOS,
        )
        paste_center(frame, wm, int(H * 0.08), opacity * 0.85)

    # --- PRODUCT REVEAL stickers (5.5 - 9.5) ---
    if 5.4 < t < 9.65:
        local = t - 5.4
        opacity = ease_out_cubic(min(1, local / 0.3))
        if t > 9.25:
            opacity *= max(0, 1 - (t - 9.25) / 0.4)
        zoom = 1.0 + 0.06 * ease_out_cubic(min(1, local / 0.5))
        # pop scale
        pop = lerp(0.88, 1.0, ease_out_cubic(min(1, local / 0.4)))
        st = fit_contain(stickers, (int(980 * pop * zoom), int(980 * pop * zoom)))
        # glow behind
        glow = st.filter(ImageFilter.GaussianBlur(28))
        glow = ImageEnhance.Brightness(glow).enhance(1.6)
        paste_center(frame, glow, int(H * 0.36), opacity * 0.55)
        paste_center(frame, st, int(H * 0.36), opacity)
        # badge
        badge = Image.new("RGBA", (320, 72), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge)
        bd.rounded_rectangle((0, 0, 319, 71), radius=36, fill=(255, 210, 0, int(240 * opacity)))
        bf = font(34, bold=True)
        bb = bd.textbbox((0, 0), "НОВО · -25%", font=bf)
        bd.text(((320 - (bb[2] - bb[0])) // 2, (72 - (bb[3] - bb[1])) // 2 - 4), "НОВО · -25%", font=bf, fill=(10, 10, 10, 255))
        paste_center(frame, badge, int(H * 0.12), opacity)
        # price
        price_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pd = ImageDraw.Draw(price_layer)
        draw_text_centered(
            pd,
            "Стикери  €8.00",
            int(H * 0.62),
            font(52),
            fill=(255, 255, 255, int(255 * opacity)),
        )
        frame = Image.alpha_composite(frame, price_layer)

    # --- ОТ ТОВА (before) ---
    if OT_TOVA_START - 0.08 <= t <= DO_TOVA_START + 0.15:
        local = t - OT_TOVA_START
        appear = ease_out_cubic(min(1, max(0, local) / 0.22))
        zoom = 1.02 + 0.06 * min(1, max(0, local) / 0.8)
        card = make_photo_card(before, "ОТ ТОВА", "#ff7a1a", zoom)
        # slide up + scale
        y = int(H * 0.42 - (1 - appear) * 80)
        paste_center(frame, card, y, appear)
        if 0 <= local <= 0.18:
            frame = Image.alpha_composite(frame, flash_overlay(1 - local / 0.18, (255, 160, 60)))

    # --- ДО ТОВА (after) with wipe transition ---
    if DO_TOVA_START - 0.05 <= t <= 13.0:
        local = t - DO_TOVA_START
        appear = ease_out_cubic(min(1, max(0, local) / 0.25))
        zoom = 1.02 + 0.08 * min(1, max(0, local) / 0.9)
        card = make_photo_card(after, "ДО ТОВА", "#ffe900", zoom)
        y = int(H * 0.42)
        # wipe reveal over before
        if local < 0.35:
            # composite wipe
            base_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            paste_center(base_layer, card, y, 1.0)
            mask = wipe_mask(min(1, local / 0.32), "left")
            masked = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            masked.paste(base_layer, (0, 0), mask)
            frame = Image.alpha_composite(frame, masked)
            # edge streak
            streak_x = int(W * ease_in_out(min(1, local / 0.32)))
            streak = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sd = ImageDraw.Draw(streak)
            sd.rectangle((streak_x - 18, 0, streak_x + 18, H), fill=(255, 240, 120, 180))
            streak = streak.filter(ImageFilter.GaussianBlur(10))
            frame = Image.alpha_composite(frame, streak)
            frame = Image.alpha_composite(frame, flash_overlay(max(0, 1 - local / 0.25), (255, 245, 160)))
        else:
            fade = 1.0 if t < 12.5 else max(0, 1 - (t - 12.5) / 0.5)
            paste_center(frame, card, y, appear * fade)

    # --- CTA / site + colors + BF (12.6 - end) ---
    if t >= 12.55:
        local = t - 12.55
        opacity = ease_out_cubic(min(1, local / 0.35))
        # product mosaic strip
        mosaic = Image.new("RGBA", (1000, 520), (0, 0, 0, 0))
        tiles = [stickers, stop, panels, cache["leds"]]
        for ti, im in enumerate(tiles):
            tile = fit_cover(im.convert("RGBA"), (230, 230), zoom=1.05)
            tm = rounded_mask(tile.size, 20)
            tile.putalpha(tm)
            ox = 20 + ti * 245
            oy = int(20 + math.sin(local * 2 + ti) * 10)
            mosaic.paste(tile, (ox, oy), tile)
        paste_center(frame, mosaic, int(H * 0.34), opacity)

        # logo
        lw = 520
        lh = int(logo.height * (lw / logo.width))
        paste_center(frame, logo.resize((lw, lh), Image.Resampling.LANCZOS), int(H * 0.58), opacity)

        # URL + BF badge
        cta = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cta)
        draw_text_centered(
            cd,
            "bitakamodz.com/product/stikeri",
            int(H * 0.64),
            font(34),
            fill=(255, 255, 255, int(255 * opacity)),
        )
        # BF pill with subtle pulse scale
        pulse = 1.0 + 0.035 * math.sin(t * 6)
        base_pill = Image.new("RGBA", (560, 88), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(base_pill)
        pdraw.rounded_rectangle(
            (0, 0, 559, 87),
            radius=44,
            fill=(255, 210, 0, int(255 * opacity)),
        )
        pf = font(40, bold=True)
        pb = pdraw.textbbox((0, 0), "BLACK FRIDAY  -25%", font=pf)
        pdraw.text(
            ((560 - (pb[2] - pb[0])) // 2, (88 - (pb[3] - pb[1])) // 2 - 4),
            "BLACK FRIDAY  -25%",
            font=pf,
            fill=(10, 10, 10, 255),
        )
        pw, ph = int(560 * pulse), int(88 * pulse)
        pill = base_pill.resize((pw, ph), Image.Resampling.LANCZOS)
        cta.paste(pill, ((W - pw) // 2, int(H * 0.68)), pill)
        frame = Image.alpha_composite(frame, cta)

        # color swatches
        row = color_row(t)
        frame.paste(row, (0, int(H * 0.74)), row)

    # --- Subtitles ---
    for start, end, text in SUBS:
        if start <= t <= end + 0.05:
            prog = min(1.0, (t - start) / 0.18)
            fade_out = 1.0
            if t > end - 0.12:
                fade_out = max(0, (end + 0.05 - t) / 0.17)
            sub = subtitle_block(text, prog)
            if fade_out < 1:
                a = sub.split()[-1].point(lambda p, fo=fade_out: int(p * fo))
                sub = sub.copy()
                sub.putalpha(a)
            frame = Image.alpha_composite(frame, sub)
            break

    # subtle film grain
    rng = np.random.default_rng(i * 9973)
    grain = rng.integers(0, 28, (H, W), dtype=np.uint8)
    grain_img = Image.fromarray(grain, mode="L").convert("RGBA")
    grain_img.putalpha(18)
    frame = Image.alpha_composite(frame, grain_img)

    return frame.convert("RGB")


def build_audio_mix(out_wav: Path) -> None:
    """Mix cleaned VO with timed SFX."""
    vo = AUDIO / "vo_clean.wav"
    ot_ms = int(OT_TOVA_START * 1000)
    do_ms = int(DO_TOVA_START * 1000)
    fc = (
        f"[1]volume=0.55,adelay=80|80[s1];"
        f"[2]volume=0.45,adelay=5400|5400[s2];"
        f"[3]volume=0.55,adelay={ot_ms}|{ot_ms}[s3];"
        f"[4]volume=0.5,adelay={ot_ms}|{ot_ms}[s4];"
        f"[5]volume=0.65,adelay={do_ms}|{do_ms}[s5];"
        f"[6]volume=0.45,adelay={do_ms}|{do_ms}[s6];"
        f"[7]volume=0.35,adelay=12700|12700[s7];"
        f"[8]volume=0.3,adelay=15400|15400[s8];"
        f"[0]volume=1.0[v0];"
        f"[v0][s1][s2][s3][s4][s5][s6][s7][s8]"
        f"amix=inputs=9:duration=first:dropout_transition=0:normalize=0[aout]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(vo),
        "-i",
        str(SFX / "rise.wav"),
        "-i",
        str(SFX / "whoosh.wav"),
        "-i",
        str(SFX / "slam.wav"),
        "-i",
        str(SFX / "whoosh.wav"),
        "-i",
        str(SFX / "impact.wav"),
        "-i",
        str(SFX / "whoosh2.wav"),
        "-i",
        str(SFX / "sparkle.wav"),
        "-i",
        str(SFX / "click.wav"),
        "-filter_complex",
        fc,
        "-map",
        "[aout]",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])


_WORKER_CACHE = None


def _init_worker() -> None:
    global _WORKER_CACHE
    _WORKER_CACHE = {
        "logo": load_rgba(BRAND / "logo-footer-transparent.png"),
        "logo_mark": load_rgba(BRAND / "logo-mark-transparent.png"),
        "stickers": load_rgba(PRODUCTS / "stikeri-6e3ef1ca526baa7b.webp"),
        "before": load_rgba(ASSETS / "before_ot_tova.jpg"),
        "after": load_rgba(ASSETS / "after_do_tova.jpg"),
        "panels": load_rgba(PRODUCTS / "stranichni-paneli-276835a7035e61f5.webp"),
        "stop": load_rgba(PRODUCTS / "zaden-stop-1.webp"),
        "leds": load_rgba(PRODUCTS / "stranichni-led-svetlini-2.webp"),
    }


def _render_one(i: int) -> str:
    assert _WORKER_CACHE is not None
    frame = render_frame(i, _WORKER_CACHE)
    path = FRAMES / f"f_{i:05d}.jpg"
    frame.save(path, quality=92, optimize=True)
    return str(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if FRAMES.exists():
        for p in FRAMES.glob("*.jpg"):
            p.unlink()
    FRAMES.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {TOTAL} frames @ {W}x{H} {FPS}fps...")
    workers = max(2, min(8, (os.cpu_count() or 4)))
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_render_one, i) for i in range(TOTAL)]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 60 == 0 or done == TOTAL:
                print(f"  {done}/{TOTAL}")

    mix = OUT / "audio_mix.wav"
    print("Mixing audio...")
    build_audio_mix(mix)

    mp4 = OUT / "bitakamodz_stikeri_9x16.mp4"
    print("Encoding mp4...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(FRAMES / "f_%05d.jpg"),
            "-i",
            str(mix),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    print("Done:", mp4, mp4.stat().st_size)


if __name__ == "__main__":
    main()
