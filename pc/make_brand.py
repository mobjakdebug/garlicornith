"""Station AI brand images, drawn here (no external artwork):

  chat/assets/logo.bin     PIMG: "PIMG", u32 count, per image <HHI (width, height, offset of RGBA bytes); the
                           mark at 40 and 168 px (straight alpha), cropped to its ink, for the header and the
                           loading / about cards
  chat/sce_sys/icon0.png   512 x 512 home screen icon
  chat/sce_sys/pic1.png    1920 x 1080 home screen background: the mark and the "station ai" wordmark

Run: python make_brand.py   (writes build/brand_preview.png too)

The mark is measured from the logo artwork and rebuilt as geometry, so it stays crisp at every size: two slanted
ribbons ("S"), the two blue dots, the corner tick, the small square cut and — at the two large sizes — the two
hairline ornaments (the arrow at the lower left, the plug at the upper right). Everything lives in a box of
104.12 x 82.13 units, of which the ribbons take the left 100.

Three sizes of detail: 0 draws only the ribbons and the dots (at 48 px the rest is sub-pixel), 1 adds the tick and
the square, 2 adds the ornaments.

The wordmark is set in Inter SemiBold (SIL OFL, pc/brand, licence beside it) — the neo-grotesque of the artwork;
Selawik (what the app's own text is set in) and DejaVu Sans Bold stand in if it is missing. Either way the result is
baked into the PNGs in chat/sce_sys, so a build needs no font at all.
"""

import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SS = 4   # supersampling

WHITE = (255, 255, 255, 255)
BLUE = (0x2C, 0x5A, 0xE2, 255)      # the dots, from the artwork
INK = (0x1C, 0x1C, 0x1C)            # the field the logo was given on
BG_TOP, BG_BOT = (0x0B, 0x0D, 0x11), (0x06, 0x07, 0x09)   # the app's own backdrop (chat/ui.c)

UNIT_W, UNIT_H = 104.12, 82.13

# ---- the geometry, in the mark's units ----------------------------------------------------------------------
BAND_A = [(53.95, 0.0), (91.41, 0.0), (26.12, 61.51), (0.0, 50.86)]        # the upper ribbon, its point at the bottom
BAND_B = [(74.91, 23.02), (100.0, 32.65), (47.77, 82.13), (12.71, 82.13)]   # the lower one, a 180 degree turn of it
TICK = [(26.12, 8.93), (26.12, 16.15), (19.59, 16.15)]                     # the corner tick, upper left
SQUARE = [(19.93, 43.64), (26.12, 43.64), (26.12, 50.86), (19.93, 50.86)]  # the cut in the upper ribbon
DOTS = [((77.32, 35.05), 2.92), ((77.32, 68.73), 2.75)]                    # (centre, radius)

ARROW = (6.19, 68.38, 13.06, 3.78, 25.77)   # tip x, y; head base x, head half-height; shaft end x (the lower left)
PLUG = (80.76, 88.66, 3.44, 98.63, 1.40, 104.12)   # line start, half-disc flat side + radius, ring centre + radius,
                                                   # line end — all on y = 16.15 (the upper right)

# Inter SemiBold is in the repository (SIL OFL, the licence beside it): the wordmark of the artwork is that kind of
# neo-grotesque. Selawik — what the app's own text is set in, and downloaded rather than committed — stands in next,
# and the system's DejaVu after that.
FONT_CANDIDATES = [
    ROOT / "pc" / "brand" / "Inter-SemiBold.ttf",
    ROOT / "pc" / "fonts_libres" / "Selawik" / "selawksb.ttf",
    ROOT / "pc" / "fonts_libres" / "Selawik" / "Selawik-Bold.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
WORDMARK = "station ai"
WORDMARK_TRACKING = -0.022   # of the type size: the artwork's wordmark is set tight
WORDMARK_RATIO = 95 / 538    # the artwork's wordmark: height over width, and Inter needs the 7 % squeeze


def font_at(px):
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), px)
    raise SystemExit("no wordmark font: put selawksb.ttf in pc/fonts_libres/Selawik (see the docstring)")


def mark(px_w, detail=2):
    """The mark, RGBA and transparent, `px_w` pixels wide (the ribbons are 100/104.12 of that)."""
    S = px_w * SS / UNIT_W
    size = (max(1, round(UNIT_W * S)), max(1, round(UNIT_H * S)))
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for poly in (BAND_A, BAND_B):
        d.polygon([(x * S, y * S) for x, y in poly], fill=255)
    if detail >= 1:
        d.polygon([(x * S, y * S) for x, y in TICK], fill=255)
        d.polygon([(x * S, y * S) for x, y in SQUARE], fill=0)   # the cut: alpha goes back to nothing
    if detail >= 2:
        hair = max(1, round(0.72 * S))
        tx, ty, hx, hh, se = ARROW      # the arrowhead points left; its shaft runs into the lower ribbon
        d.polygon([(tx * S, ty * S), (hx * S, (ty - hh) * S), (hx * S, (ty + hh) * S)], fill=255)
        d.rectangle([hx * S, ty * S - hair / 2, se * S, ty * S + hair / 2], fill=255)
        x0, flat, r, ring_c, ring_r, x1 = PLUG
        y = 16.15
        d.rectangle([x0 * S, y * S - hair / 2, x1 * S, y * S + hair / 2], fill=255)
        d.pieslice([(flat - r) * S, (y - r) * S, (flat + r) * S, (y + r) * S], -90, 90, fill=255)
        ring = max(1, round(0.55 * S))
        d.ellipse([(ring_c - ring_r) * S, (y - ring_r) * S, (ring_c + ring_r) * S, (y + ring_r) * S],
                  outline=255, width=ring)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    img.paste(Image.new("RGBA", size, WHITE), (0, 0), mask)
    d = ImageDraw.Draw(img)
    for (cx, cy), r in DOTS:
        d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=BLUE)
    return img.resize((px_w, max(1, round(px_w * UNIT_H / UNIT_W))), Image.LANCZOS)


def wordmark(px_w):
    """The wordmark, RGBA and transparent, its ink exactly `px_w` pixels wide."""
    size = 200
    while True:   # a size that leaves the ink a little wider than asked: the crop below sets it exactly
        f = font_at(size)
        tracking = WORDMARK_TRACKING * size
        widths = [f.getlength(c) for c in WORDMARK]
        total = sum(widths) + tracking * (len(WORDMARK) - 1)
        if total >= px_w * 1.15:
            break
        size = int(size * 1.4)
    canvas = Image.new("L", (int(total + size), int(size * 2)), 0)
    d = ImageDraw.Draw(canvas)
    x = size * 0.5
    for ch, w in zip(WORDMARK, widths):
        d.text((x, size * 0.5), ch, font=f, fill=255)
        x += w + tracking
    box = canvas.getbbox()
    ink = canvas.crop(box)
    img = Image.new("RGBA", ink.size, (0, 0, 0, 0))
    img.paste(Image.new("RGBA", ink.size, WHITE), (0, 0), ink)
    return img.resize((px_w, max(1, round(px_w * WORDMARK_RATIO))), Image.LANCZOS)   # the artwork's proportions


def backdrop(w, h, glow=(0.5, 0.42), glow_r=0.62, grid=48):
    """The app's own backdrop (near-black, a lift of blue behind the mark, the 48 px grid) at any size."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    gx, gy = glow[0] * w, glow[1] * h
    R = glow_r * max(w, h)
    for y in range(h):
        t = y / (h - 1)
        base = [BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * t for i in range(3)]
        for x in range(w):
            g = max(0.0, 1 - math.hypot(x - gx, y - gy) / R) ** 2
            px[x, y] = (int(base[0] + g * 26), int(base[1] + g * 30), int(base[2] + g * 58))
    d = ImageDraw.Draw(img)
    for x in range(0, w, grid):
        d.line([(x, 0), (x, h)], fill=(0x0E, 0x11, 0x16))
    for y in range(0, h, grid):
        d.line([(0, y), (w, y)], fill=(0x0E, 0x11, 0x16))
    return img


def ink(img):
    """The image cropped to what is actually drawn: centring a mark by its box is centring its empty corners."""
    box = img.getbbox()
    return img.crop(box) if box else img


def icon():
    """512 x 512: the mark, on the field the logo was given, its ink 62 % of the tile."""
    W = 512
    img = Image.new("RGBA", (W, W), INK + (255,))
    m = ink(mark(340, detail=1))
    w = round(W * 0.62)
    m = m.resize((w, round(m.height * w / m.width)), Image.LANCZOS)
    img.alpha_composite(m, ((W - m.width) // 2, (W - m.height) // 2))
    return img.convert("RGB")


def background():
    """1920 x 1080: the backdrop, the mark and the wordmark under it — the lockup, centred."""
    W, H = 1920, 1080
    img = backdrop(W, H).convert("RGBA")
    m = ink(mark(360, detail=2))
    wm = wordmark(round(360 * 538 / 291 * 1.0))      # the artwork's ratio: the wordmark is 1.85 x the mark's width
    gap = round(360 * 0.18)
    top = (H - (m.height + gap + wm.height)) // 2 - 20
    img.alpha_composite(m, ((W - m.width) // 2, top))
    img.alpha_composite(wm, ((W - wm.width) // 2, top + m.height + gap))
    return img.convert("RGB")


def main():
    # Cropped to the ink: the app centres these in a square (the header's tile, the loading card's circles),
    # and the mark's own box has empty margin on the right, which put the mark over the tile's border.
    images = [ink(mark(40, detail=0)), ink(mark(168, detail=1))]
    head = b"PIMG" + struct.pack("<I", len(images))
    offset = len(head) + 8 * len(images)
    recs, data = b"", b""
    for im in images:
        recs += struct.pack("<HHI", im.width, im.height, offset + len(data))
        data += im.tobytes()
    (ROOT / "chat" / "assets" / "logo.bin").write_bytes(head + recs + data)
    sce = ROOT / "chat" / "sce_sys"
    sce.mkdir(exist_ok=True)
    ic = icon()
    ic.save(sce / "icon0.png")
    bg = background()
    bg.save(sce / "pic1.png")
    # a sheet to look at: the icon, the splash, and the two sizes the app draws
    prev = Image.new("RGB", (512 + 20 + 960, 540), (16, 16, 17))
    prev.paste(ic, (0, 14))
    prev.paste(bg.resize((960, 540), Image.LANCZOS), (532, 0))
    (ROOT / "build").mkdir(exist_ok=True)
    prev.save(ROOT / "build" / "brand_preview.png")
    sheet = backdrop(1200, 420)
    sheet.paste(images[0], (120, 186), images[0].split()[3])
    sheet.paste(images[1], (260, 126), images[1].split()[3])
    sheet.paste(mark(320, detail=2), (520, 50), mark(320, detail=2).split()[3])
    sheet.save(ROOT / "build" / "brand_marks.png")
    print("logo.bin", len(head + recs + data), "bytes; icon0.png", ic.size, "pic1.png", bg.size)


if __name__ == "__main__":
    main()
