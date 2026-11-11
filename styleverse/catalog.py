"""Self-created synthetic catalog.

Every image is generated programmatically with Pillow — no scraped or
unlicensed images. Provenance is recorded per item (source + license).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .features import COLOR_FAMILIES

CATEGORIES = ["top", "dress", "jacket", "trousers", "skirt"]
PATTERNS = ["solid", "stripes", "checks", "polka"]
SILHOUETTES = ["fitted", "straight", "a-line", "relaxed"]
OCCASIONS = ["casual", "formal", "party", "work"]
CURRENCY = "USD"

PRICE_RANGES = {
    "top": (18, 65),
    "dress": (45, 180),
    "jacket": (60, 180),
    "trousers": (35, 120),
    "skirt": (28, 95),
}

# Silhouette implied by each category's drawn shape.
CATEGORY_SILHOUETTE = {
    "top": "fitted",
    "dress": "a-line",
    "jacket": "straight",
    "trousers": "straight",
    "skirt": "a-line",
}

BG = (225, 225, 228)

# Draw colours: identical to the family centres except white, which is
# drawn brighter than the background so the garment is segmentable.
DRAW_COLORS = {**COLOR_FAMILIES, "white": (250, 250, 250)}


@dataclass
class Product:
    id: str
    title: str
    category: str
    color_family: str
    pattern: str
    silhouette: str
    occasion: str
    price: float
    currency: str
    image_path: str
    source: str = "synthetic (generated in-repo with Pillow)"
    license: str = "CC0 — generated image, no third-party rights"


def _shade(rgb, factor: float) -> tuple:
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)


def _draw_pattern(draw: ImageDraw.ImageDraw, bbox, color, pattern, seed: int):
    """Overlay a pattern inside the garment bounding box."""
    x0, y0, x1, y1 = bbox
    rng = random.Random(seed)
    light = _shade(color, 1.25)
    dark = _shade(color, 0.7)
    if pattern == "stripes":
        y = y0
        while y < y1:
            draw.rectangle([x0, y, x1, min(y + 6, y1)], fill=light)
            y += 14
    elif pattern == "checks":
        step = 14
        for yy in range(y0, y1, step):
            for xx in range(x0, x1, step):
                if ((xx - x0) // step + (yy - y0) // step) % 2 == 0:
                    draw.rectangle([xx, yy, min(xx + step, x1),
                                    min(yy + step, y1)], fill=light)
    elif pattern == "polka":
        for _ in range(40):
            cx = rng.randint(x0 + 4, max(x0 + 4, x1 - 4))
            cy = rng.randint(y0 + 4, max(y0 + 4, y1 - 4))
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=dark)


def _draw_garment(category: str, color: tuple, pattern: str, seed: int) -> Image.Image:
    """Draw a simple garment silhouette on a plain background.

    The pattern is composited through the garment polygon as a mask, so
    it never bleeds onto the background (which would corrupt the
    foreground mask used by attribute extraction).
    """
    img = Image.new("RGB", (128, 160), BG)
    cx = 64
    if category == "top":
        polys = [[(cx - 30, 25), (cx + 30, 25), (cx + 26, 95), (cx - 26, 95)],
                 [(cx - 44, 28), (cx - 30, 28), (cx - 30, 55), (cx - 44, 55)],
                 [(cx + 30, 28), (cx + 44, 28), (cx + 44, 55), (cx + 30, 55)]]
        bbox = (cx - 44, 25, cx + 44, 95)
    elif category == "dress":
        polys = [[(cx - 22, 20), (cx + 22, 20), (cx + 18, 70),
                  (cx + 42, 140), (cx - 42, 140), (cx - 18, 70)]]
        bbox = (cx - 42, 20, cx + 42, 140)
    elif category == "jacket":
        polys = [[(cx - 38, 20), (cx + 38, 20), (cx + 38, 105), (cx - 38, 105)],
                 [(cx - 52, 24), (cx - 38, 24), (cx - 38, 95), (cx - 52, 95)],
                 [(cx + 38, 24), (cx + 52, 24), (cx + 52, 95), (cx + 38, 95)]]
        bbox = (cx - 52, 20, cx + 52, 105)
    elif category == "trousers":
        polys = [[(cx - 28, 15), (cx - 3, 15), (cx - 3, 145), (cx - 28, 145)],
                 [(cx + 3, 15), (cx + 28, 15), (cx + 28, 145), (cx + 3, 145)],
                 [(cx - 28, 15), (cx + 28, 15), (cx + 28, 30), (cx - 28, 30)]]
        bbox = (cx - 28, 15, cx + 28, 145)
    else:  # skirt
        polys = [[(cx - 24, 30), (cx + 24, 30), (cx + 44, 135), (cx - 44, 135)]]
        bbox = (cx - 44, 30, cx + 44, 135)

    draw = ImageDraw.Draw(img)
    for poly in polys:
        draw.polygon(poly, fill=color)
    if category == "jacket":
        draw.line([cx, 20, cx, 105], fill=_shade(color, 0.6), width=2)

    overlay = Image.new("RGB", (128, 160), color)
    _draw_pattern(ImageDraw.Draw(overlay), bbox, color, pattern, seed)
    mask = Image.new("L", (128, 160), 0)
    mask_draw = ImageDraw.Draw(mask)
    for poly in polys:
        mask_draw.polygon(poly, fill=255)
    img.paste(overlay, (0, 0), mask)
    return img


def _occasion_for(category: str, color: str, pattern: str) -> str:
    if category in ("jacket", "trousers") and color in ("black", "white") \
            and pattern == "solid":
        return "formal"
    if category == "dress" and color in ("red", "pink"):
        return "party"
    if color in ("blue", "white") and pattern == "solid":
        return "work"
    return "casual"


def generate_catalog(n: int = 60, out_dir: str | Path = "catalog",
                     seed: int = 7) -> list[Product]:
    """Generate n synthetic products (images + metadata.json)."""
    rng = random.Random(seed)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    products: list[Product] = []
    combos = [(c, col, p) for c in CATEGORIES
              for col in COLOR_FAMILIES for p in PATTERNS]
    rng.shuffle(combos)
    for i in range(n):
        category, color, pattern = combos[i % len(combos)]
        lo, hi = PRICE_RANGES[category]
        price = round(rng.uniform(lo, hi), 2)
        pid = f"sv-{i + 1:03d}"
        img = _draw_garment(category, DRAW_COLORS[color], pattern, seed + i)
        img_path = out / "images" / f"{pid}.jpg"
        img.save(img_path, format="JPEG", quality=88)
        products.append(Product(
            id=pid,
            title=f"{color.title()} {pattern} {category}",
            category=category,
            color_family=color,
            pattern=pattern,
            silhouette=CATEGORY_SILHOUETTE[category],
            occasion=_occasion_for(category, color, pattern),
            price=price,
            currency=CURRENCY,
            image_path=str(img_path),
        ))
    meta = [asdict(p) for p in products]
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return products


def load_catalog(catalog_dir: str | Path = "catalog") -> list[Product]:
    """Load products from a catalog directory's metadata.json."""
    path = Path(catalog_dir) / "metadata.json"
    data = json.loads(path.read_text())
    return [Product(**item) for item in data]
