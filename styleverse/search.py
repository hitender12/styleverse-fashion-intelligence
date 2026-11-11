"""Nearest-neighbour vector search over image embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .catalog import Product
from .features import embed


@dataclass
class SearchResult:
    product: Product
    score: float  # cosine similarity, plus any rerank/feedback bonus
    # (bonuses can push the score above 1.0 — see rerank.py / feedback.py)


class VectorIndex:
    """Brute-force cosine-similarity index (exact; fine at catalog scale)."""

    def __init__(self, products: list[Product]):
        self.products = list(products)
        self.vectors = np.stack([
            embed(Image.open(p.image_path)) for p in self.products
        ]) if products else np.zeros((0, 0))

    def query(self, query_vec: np.ndarray, k: int = 10,
              exclude_ids: set[str] | None = None) -> list[SearchResult]:
        """Top-k by cosine similarity, excluding ids (e.g. the query item)."""
        if not self.products:
            return []
        exclude = exclude_ids or set()
        sims = self.vectors @ query_vec
        order = np.argsort(-sims)
        results: list[SearchResult] = []
        for idx in order:
            product = self.products[int(idx)]
            if product.id in exclude:
                continue
            results.append(SearchResult(product, float(sims[idx])))
            if len(results) >= k:
                break
        return results

    def query_image(self, img: Image.Image, k: int = 10,
                    exclude_ids: set[str] | None = None) -> list[SearchResult]:
        return self.query(embed(img), k=k, exclude_ids=exclude_ids)
