"""Pixel features and image embeddings.

Reference encoder (no model downloads): a concatenated, L2-normalised vector
of a quantised colour histogram, an average-hash (perceptual hash) and a
small texture descriptor. Production upgrade path: CLIP/SigLIP-family
image-text encoders (see README.md).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Named colour families with representative RGB centres.
COLOR_FAMILIES = {
    "red": (198, 40, 40),
    "blue": (40, 90, 180),
    "green": (46, 125, 50),
    "yellow": (230, 200, 40),
    "pink": (225, 130, 170),
    "black": (35, 35, 35),
    "white": (238, 238, 238),
    "brown": (121, 85, 50),
}

EMBED_DIM = 64 + 64 + 6  # histogram + phash + texture

# Catalog background (see catalog.py). Foreground = pixels far from it.
BACKGROUND = np.array([225, 225, 228], dtype=np.float64)


def foreground_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask of pixels that differ from the plain background."""
    arr = np.asarray(_as_rgb(img), dtype=np.float64)
    return np.abs(arr - BACKGROUND).sum(axis=2) > 60


def _as_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def color_histogram(img: Image.Image, bins: int = 4,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Quantised RGB histogram over foreground pixels, normalised."""
    arr = np.asarray(_as_rgb(img), dtype=np.float64)
    flat = arr.reshape(-1, 3)
    if mask is not None:
        flat = flat[mask.reshape(-1)]
    idx = np.clip((flat // (256 // bins)).astype(int), 0, bins - 1)
    code = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
    hist = np.bincount(code, minlength=bins ** 3).astype(np.float64)
    total = hist.sum()
    return hist / total if total else hist


def average_hash(img: Image.Image) -> np.ndarray:
    """8x8 average hash as a 0/1 float vector (perceptual hash)."""
    gray = _as_rgb(img).convert("L").resize((8, 8), Image.LANCZOS)
    px = np.asarray(gray, dtype=np.float64).ravel()
    return (px > px.mean()).astype(np.float64)


def texture_features(img: Image.Image) -> np.ndarray:
    """Small texture descriptor: edge density, row/column variance
    (stripe signal), colourfulness, brightness, contrast."""
    rgb = _as_rgb(img)
    gray = np.asarray(rgb.convert("L"), dtype=np.float64)
    arr = np.asarray(rgb, dtype=np.float64)
    gy = np.abs(np.diff(gray, axis=0)).mean() / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean() / 255.0
    edge_density = float((gx + gy) / 2.0)
    row_var = float(np.var(gray.mean(axis=1)) / 255.0 ** 2 * 10)
    col_var = float(np.var(gray.mean(axis=0)) / 255.0 ** 2 * 10)
    colorfulness = float(arr.std(axis=(0, 1)).mean() / 255.0)
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    return np.array(
        [edge_density, row_var, col_var, colorfulness, brightness, contrast],
        dtype=np.float64,
    )


def embed(img: Image.Image) -> np.ndarray:
    """Concatenate histogram + phash + texture, L2-normalised.

    The colour histogram is computed over foreground pixels only, so the
    shared plain background does not swamp the garment colour. Each block
    is unit-normalised first, then weighted: colour carries the most
    signal for "visually similar", shape second, texture last.
    """
    mask = foreground_mask(img)
    hist = color_histogram(img, mask=mask)
    ph = average_hash(img)
    tex = texture_features(img)

    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n else v

    vec = np.concatenate([2.0 * _unit(hist), 1.0 * _unit(ph),
                          0.5 * _unit(tex)]).astype(np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def dominant_color_family(img: Image.Image,
                          mask: np.ndarray | None = None) -> str:
    """Nearest named colour family to the mean foreground colour."""
    arr = np.asarray(_as_rgb(img), dtype=np.float64).reshape(-1, 3)
    if mask is not None:
        sel = arr[mask.reshape(-1)]
        if len(sel):
            arr = sel
    mean = arr.mean(axis=0)
    best, best_dist = "white", float("inf")
    for name, centre in COLOR_FAMILIES.items():
        dist = float(np.sum((mean - np.array(centre)) ** 2))
        if dist < best_dist:
            best, best_dist = name, dist
    return best
