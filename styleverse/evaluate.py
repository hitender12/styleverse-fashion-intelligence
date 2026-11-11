"""Reproducible retrieval evaluation.

Labelled queries: each query is a catalog image; the relevant set is all
other catalog items sharing the query's category AND colour family
("visually similar", not "same product" — the query item itself is
excluded from results). Reports recall@5, recall@10, precision@10,
mean latency, and the share of queries with no good match.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .catalog import Product
from .features import embed
from .search import VectorIndex


@dataclass
class EvalMetrics:
    recall_at_5: float
    recall_at_10: float
    precision_at_10: float
    mean_latency_ms: float
    no_good_match_rate: float
    n_queries: int


def relevant_set(query: Product, products: list[Product]) -> set[str]:
    return {
        p.id for p in products
        if p.id != query.id
        and p.category == query.category
        and p.color_family == query.color_family
    }


def run_evaluation(products: list[Product],
                   queries: list[Product] | None = None) -> EvalMetrics:
    """Evaluate the index over labelled queries (defaults: all products)."""
    index = VectorIndex(products)
    queries = queries if queries is not None else list(products)
    recalls5, recalls10, precisions10 = [], [], []
    latencies: list[float] = []
    no_good = 0
    for q in queries:
        relevant = relevant_set(q, products)
        if not relevant:
            no_good += 1
            continue
        start = time.perf_counter()
        results = index.query_image(
            Image.open(q.image_path), k=10, exclude_ids={q.id})
        latencies.append((time.perf_counter() - start) * 1000)
        top5 = {r.product.id for r in results[:5]}
        top10 = {r.product.id for r in results[:10]}
        recalls5.append(len(top5 & relevant) / len(relevant))
        recalls10.append(len(top10 & relevant) / len(relevant))
        precisions10.append(len(top10 & relevant) / 10)
        if not (top10 & relevant):
            no_good += 1
    n = len(recalls5)
    return EvalMetrics(
        recall_at_5=float(np.mean(recalls5)) if n else 0.0,
        recall_at_10=float(np.mean(recalls10)) if n else 0.0,
        precision_at_10=float(np.mean(precisions10)) if n else 0.0,
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        no_good_match_rate=no_good / max(1, len(queries)),
        n_queries=len(queries),
    )


def main() -> None:
    from .catalog import load_catalog
    products = load_catalog("catalog")
    metrics = run_evaluation(products)
    print(f"queries:            {metrics.n_queries}")
    print(f"recall@5:           {metrics.recall_at_5:.3f}")
    print(f"recall@10:          {metrics.recall_at_10:.3f}")
    print(f"precision@10:       {metrics.precision_at_10:.3f}")
    print(f"mean latency:       {metrics.mean_latency_ms:.1f} ms")
    print(f"no good match:      {metrics.no_good_match_rate:.1%}")


if __name__ == "__main__":
    main()
