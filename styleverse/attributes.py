"""Rule-based garment attribute extraction from a query image.

Heuristics only — no learned detector in this reference implementation.
A garment detector/segmenter (e.g. a fine-tuned segmentation model) is the
documented production upgrade. Accuracy is measured on the synthetic catalog
in the evaluation script and reported honestly in the README.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .features import (dominant_color_family, foreground_mask,
                       texture_features)


def _foreground_mask(img: Image.Image) -> np.ndarray:
    return foreground_mask(img)


def _detect_pattern(img: Image.Image) -> str:
    """Stripes / checks / polka / solid from interior gradient structure.

    Gradients are measured only on eroded-foreground pixels so garment
    edges do not pollute the signal. Stripes (horizontal bands) give a
    strong vertical gradient with almost no horizontal gradient; checks
    give both; polka dots give strong gradients in both directions with
    higher total energy than the regular check grid; solid gives ~none.
    Thresholds tuned on the synthetic catalog; see evaluation.
    """
    from .features import foreground_mask

    gray = np.asarray(img.convert("L"), dtype=np.float64)
    m = foreground_mask(img)
    er = m.copy()
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            er &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    mx = er[:, :-1] & er[:, 1:]
    my = er[:-1, :] & er[1:, :]
    mean_gx = float(gx[mx].mean() / 255) if mx.any() else 0.0
    mean_gy = float(gy[my].mean() / 255) if my.any() else 0.0
    if max(mean_gx, mean_gy) < 0.008:
        return "solid"
    if mean_gy > 3 * max(mean_gx, 1e-6):
        return "stripes"
    if mean_gx + mean_gy > 0.022:
        return "polka"
    return "checks"


def _detect_silhouette(mask: np.ndarray) -> str:
    """Width profile of the foreground mask: top vs bottom width."""
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return "straight"
    top = mask[rows[0]:rows[0] + max(1, len(rows) // 4)].any(axis=0)
    mid = mask[len(rows) // 2 - 2:len(rows) // 2 + 2].any(axis=0)
    bot = mask[rows[-max(1, len(rows) // 4):]].any(axis=0)
    w_top = float(top.sum())
    w_mid = float(mid.sum())
    w_bot = float(bot.sum())
    if w_bot > w_top * 1.5 and w_bot > w_mid * 1.2:
        return "a-line"
    if w_top > w_bot * 1.4:
        return "fitted"
    if w_mid > w_top * 1.25 and w_mid > w_bot * 1.25:
        return "relaxed"
    return "straight"


def _detect_category(img: Image.Image, mask: np.ndarray) -> str:
    """Category from mask aspect ratio and width profile.

    Trousers: tall mask with a gap between legs at the bottom.
    Skirt: short mask, bottom much wider than top.
    Dress: tall mask, bottom wider than top (flared).
    Jacket: wide mask with broad shoulders.
    Top: everything else.
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return "top"
    h = len(rows)
    w = len(cols)
    aspect = h / max(1, w)
    silhouette = _detect_silhouette(mask)
    bottom = mask[rows[-max(1, h // 5):]]
    # Two-leg test: middle columns empty in the bottom rows.
    mid_gap = float((~bottom[:, w // 3: 2 * w // 3].any(axis=0)).mean())
    if aspect > 1.35 and mid_gap > 0.55:
        return "trousers"
    if aspect < 0.95 and silhouette == "a-line":
        return "skirt"
    if aspect > 1.15 and silhouette == "a-line":
        return "dress"
    if w > 0.75 * img.width and aspect < 1.25:
        return "jacket"
    return "top"


def _guess_occasion(category: str, color: str, pattern: str) -> str:
    if category in ("jacket", "trousers") and color in ("black", "white") \
            and pattern == "solid":
        return "formal"
    if category == "dress" and color in ("red", "pink"):
        return "party"
    if color in ("blue", "white") and pattern == "solid":
        return "work"
    return "casual"


def extract_attributes(img: Image.Image) -> dict:
    """Return category, colour family, pattern, silhouette, occasion."""
    mask = _foreground_mask(img)
    color = dominant_color_family(img, mask=mask)
    pattern = _detect_pattern(img)
    silhouette = _detect_silhouette(mask)
    category = _detect_category(img, mask)
    return {
        "category": category,
        "color_family": color,
        "pattern": pattern,
        "silhouette": silhouette,
        "occasion": _guess_occasion(category, color, pattern),
    }
