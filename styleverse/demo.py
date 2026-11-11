"""End-to-end demo: upload a look, retrieve, rerank, explain, feedback."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from styleverse.attributes import extract_attributes
from styleverse.catalog import load_catalog
from styleverse.explain import explain
from styleverse.feedback import SessionFeedback
from styleverse.rerank import Constraints, rerank
from styleverse.search import VectorIndex


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    products = load_catalog(root / "catalog")
    index = VectorIndex(products)

    # Query: use a catalog dress as the uploaded look.
    query_product = next(p for p in products
                         if p.category == "dress" and p.color_family == "red")
    query_img = Image.open(query_product.image_path)
    attrs = extract_attributes(query_img)
    print(f"Query: {query_product.title} (${query_product.price})")
    print(f"Detected attributes: {attrs}")

    results = index.query_image(query_img, k=10, exclude_ids={query_product.id})
    budget = 100.0
    ranked = rerank(results, Constraints(max_price=budget), query_attrs=attrs)
    print(f"\nTop results under ${budget:.0f} budget:")
    for r in ranked[:5]:
        print(f"  - {r.product.title} (${r.product.price}) "
              f"sim={r.score:.2f} — {explain(attrs, r.product, r.score)}")

    # Feedback: mark the top result not relevant and re-rank.
    feedback = SessionFeedback()
    if ranked:
        feedback.mark_not_relevant(ranked[0].product.id)
        print(f"\nMarked '{ranked[0].product.title}' as not relevant.")
        adjusted = feedback.apply(ranked)
        print("Re-ranked top 3:")
        for r in adjusted[:3]:
            print(f"  - {r.product.title} (${r.product.price}) "
                  f"sim={r.score:.2f}")

    # Evaluation summary.
    from styleverse.evaluate import run_evaluation
    metrics = run_evaluation(products)
    print(f"\nEvaluation ({metrics.n_queries} queries): "
          f"recall@5={metrics.recall_at_5:.3f} "
          f"recall@10={metrics.recall_at_10:.3f} "
          f"precision@10={metrics.precision_at_10:.3f} "
          f"latency={metrics.mean_latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
