# STYLEVERSE — Architecture

## System overview

```
Upload ──► Privacy ingest (EXIF strip, session scope, TTL)
              │
              ▼
      Attribute extraction (rule-based)
              │                    │
              ▼                    ▼
      Embedding (pixel features)   Catalog metadata
              │                    │
              ▼                    ▼
      Vector index (cosine NN) ──► Candidate retrieval
                                        │
                                        ▼
                              Constraint-aware reranker
                              (budget / category / colour /
                               occasion as hard filters)
                                        │
                                        ▼
                              Explanations + Feedback loop
```

The spec's full architecture (Next.js frontend, FastAPI API, PyTorch
vision service, pgvector/Qdrant, S3 storage, Redis queue) is the production
target. This repo implements the **core library** that the vision service
would wrap: ingestion/privacy, features, search, reranking, explanations,
feedback, and evaluation — all runnable without network or GPUs.

## Modules

| Module | Responsibility |
|---|---|
| `features.py` | Colour histogram (foreground only), average hash, texture descriptor; block-weighted embedding; colour-family centroids |
| `attributes.py` | Rule-based extraction: category (mask aspect + width profile), colour family (masked mean), pattern (interior gradient structure), silhouette (width profile), occasion (heuristic) |
| `catalog.py` | Synthetic catalog generation (Pillow-drawn garments, patterns composited through the garment mask so they never bleed onto the background) + metadata with provenance |
| `search.py` | Exact brute-force cosine-similarity `VectorIndex` |
| `rerank.py` | Hard-constraint filtering after retrieval + bounded attribute bonus |
| `explain.py` | Explanations naming only genuinely matching attributes |
| `feedback.py` | Session-scoped relevant / not relevant / save / hide, applied as score adjustments |
| `privacy.py` | EXIF stripping, upload validation, session TTL + deletion |
| `evaluate.py` | Labelled evaluation: recall@5/10, precision@10, latency, no-good-match rate |
| `demo.py` | End-to-end CLI demo |

## Data model

- `Product`: id, title, category, colour family, pattern, silhouette, occasion, price, currency, image path, source, license.
- `SearchResult`: product + score (cosine similarity plus any rerank/feedback bonus; may exceed 1.0).
- `SessionFeedback`: per-session sets (relevant / not relevant / saved / hidden).
- `Session`: id, directory, created time, TTL; uploads live only inside it.

## Design decisions

- **Foreground-only colour features.** The catalog background is shared, so whole-image histograms would make every item look alike. Histograms and colour means are computed over foreground-masked pixels only.
- **Block-weighted embedding.** Raw concatenation lets the 64-dim average hash dominate the colour histogram; each block is unit-normalised first, then weighted (colour 2.0, shape 1.0, texture 0.5).
- **Constraints as filters, not features.** A budget violation is removed, never demoted — a high embedding score cannot outrank a constraint.
- **Explanations from comparison, not generation.** The explainer diffs query attributes against item attributes; it cannot invent a reason.
- **Feedback is session-scoped.** Nothing persists beyond the session; hiding is a filter, not a deletion.

## Measured behaviour (60-item catalog, seed 7)

- Retrieval: recall@5 0.826, recall@10 1.000, precision@10 0.165 (ceiling ~0.165: average relevant set is 1.27 items), mean latency 4.4 ms.
- Attribute accuracy: colour 1.00, silhouette 0.98, category 0.87, pattern 0.77.

## Production upgrade paths

1. **Encoder:** swap `features.embed()` for a CLIP/SigLIP image encoder; keep the same `VectorIndex` interface. Benchmark on the labelled query set before/after.
2. **Detector:** replace rule-based `extract_attributes()` with a garment detector/segmenter; keep the attribute schema identical so reranking and explanations are unchanged.
3. **Store:** move the brute-force index to pgvector or Qdrant; move metadata to PostgreSQL; images to private S3 buckets with signed URLs.
4. **Serving:** wrap the package in FastAPI (upload/search/feedback endpoints) behind a Redis queue + worker; add the Next.js frontend.
5. **Virtual try-on:** separate research milestone; only with models/assets whose licenses permit the intended use.
