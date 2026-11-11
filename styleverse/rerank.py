"""Constraint-aware reranking.

Hard constraints (budget, category, colour, occasion) are applied AFTER
retrieval and are never buried under a high embedding score: a violating
item is removed, not demoted. Attribute agreement then adds a small,
bounded bonus.
"""

from __future__ import annotations

from dataclasses import dataclass

from .search import SearchResult

ATTR_BONUS = 0.05  # per matching attribute, bounded
MAX_BONUS = 0.15


@dataclass
class Constraints:
    max_price: float | None = None
    category: str | None = None
    color_family: str | None = None
    occasion: str | None = None


def _passes(product, constraints: Constraints) -> bool:
    if constraints.max_price is not None and product.price > constraints.max_price:
        return False
    if constraints.category and product.category != constraints.category:
        return False
    if constraints.color_family and product.color_family != constraints.color_family:
        return False
    if constraints.occasion and product.occasion != constraints.occasion:
        return False
    return True


def _attribute_bonus(product, query_attrs: dict | None) -> float:
    if not query_attrs:
        return 0.0
    matches = sum(
        1 for key in ("category", "color_family", "pattern", "silhouette")
        if query_attrs.get(key) == getattr(product, key, None)
    )
    return min(matches * ATTR_BONUS, MAX_BONUS)


def rerank(results: list[SearchResult], constraints: Constraints,
           query_attrs: dict | None = None) -> list[SearchResult]:
    """Filter by constraints, then re-sort by similarity + attribute bonus."""
    kept = [r for r in results if _passes(r.product, constraints)]
    rescored = [
        SearchResult(r.product, r.score + _attribute_bonus(r.product, query_attrs))
        for r in kept
    ]
    return sorted(rescored, key=lambda r: r.score, reverse=True)
