#!/usr/bin/env python3
"""Reproducible retrieval evaluation. Run from the repo root:

    python evaluation/run_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from styleverse.catalog import load_catalog
from styleverse.evaluate import run_evaluation

if __name__ == "__main__":
    products = load_catalog("catalog")
    m = run_evaluation(products)
    print(f"queries:            {m.n_queries}")
    print(f"recall@5:           {m.recall_at_5:.3f}")
    print(f"recall@10:          {m.recall_at_10:.3f}")
    print(f"precision@10:       {m.precision_at_10:.3f}")
    print(f"mean latency:       {m.mean_latency_ms:.1f} ms")
    print(f"no good match:      {m.no_good_match_rate:.1%}")
