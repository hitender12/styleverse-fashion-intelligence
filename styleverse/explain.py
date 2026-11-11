"""Explainable recommendations.

Every explanation names only attributes that genuinely match between the
query and the item — never invented reasons.
"""

from __future__ import annotations

from .catalog import Product

_LABELS = {
    "category": "category",
    "color_family": "color family",
    "pattern": "pattern",
    "silhouette": "silhouette",
}


def matched_attributes(query_attrs: dict, product: Product) -> list[str]:
    """Attribute keys where query and product agree (with their value)."""
    matched: list[str] = []
    for key in ("category", "color_family", "pattern", "silhouette"):
        if query_attrs.get(key) == getattr(product, key, None):
            matched.append(f"{_LABELS[key]} ({query_attrs[key]})")
    return matched


def explain(query_attrs: dict, product: Product, score: float) -> str:
    """One-line explanation, e.g. 'matched on silhouette (a-line) and
    color family (blue)'."""
    matched = matched_attributes(query_attrs, product)
    if not matched:
        return ("visually similar overall "
                f"(similarity {score:.2f}); no single attribute matched")
    if len(matched) == 1:
        return f"matched on {matched[0]}"
    return "matched on " + " and ".join(matched)
