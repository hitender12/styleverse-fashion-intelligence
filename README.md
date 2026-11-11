# STYLEVERSE — Multimodal Fashion Intelligence

> STYLEVERSE is visual fashion search: upload a look, discover visually similar catalog pieces, and refine by budget, category and style attributes.

Upload an outfit image, and Styleverse detects garment attributes, retrieves visually similar products from a catalog, reranks them by your constraints (budget, category, color, occasion), and explains *why* each result matched. Your feedback (relevant / not relevant / save / hide) reshapes the ranking within the session.

## Quickstart

```bash
python -m styleverse catalog    # (re)generate the synthetic catalog (60 items)
python -m styleverse            # end-to-end demo: query, rerank, explain, feedback
python -m styleverse evaluate   # retrieval benchmark
python -m unittest discover -s tests -v
```

Python 3.10+, standard library + numpy + Pillow. No model downloads, no network calls, no keys.

## How it works

1. **Ingest** — uploads are validated (type/size), EXIF metadata is stripped, and files live only inside a session directory with a 24h TTL and a delete path.
2. **Attributes** — rule-based extraction from pixels: category, colour family, pattern, silhouette, occasion.
3. **Embedding** — a reference pixel encoder: foreground colour histogram + average hash + texture descriptor, block-weighted (colour 2.0, shape 1.0, texture 0.5), L2-normalised.
4. **Retrieval** — exact cosine-similarity nearest neighbours over the catalog index.
5. **Reranking** — hard constraints (budget, category, colour, occasion) are applied *after* retrieval and remove violators outright; attribute agreement adds a small bounded bonus.
6. **Explanation** — each result states only attributes that genuinely match, e.g. "matched on color family (red) and pattern (solid)".
7. **Feedback** — relevant (+0.15), not relevant (−0.30), save, hide; applied per session.

## Measured results (60-item synthetic catalog, seed 7)

Retrieval evaluation (`python -m styleverse evaluate`):

| metric | value |
|---|---|
| recall@5 | 0.826 |
| recall@10 | 1.000 |
| precision@10 | 0.165 |
| mean latency | 4.4 ms |
| queries with no good match | 23.3% |

Read precision@10 honestly: the average relevant set is only 1.27 items (14 of 60 queries have *no* other item sharing their category + colour family — a catalog-sparsity artifact), so the precision@10 ceiling for this catalog is ~0.165. recall@10 of 1.000 means every existing relevant item is found in the top 10.

Rule-based attribute extraction accuracy on the same catalog:

| attribute | accuracy |
|---|---|
| colour family | 1.00 |
| silhouette | 0.98 |
| category | 0.87 |
| pattern | 0.77 |

## Honest status

This is a **working reference implementation**, not the production system in the spec:

- **Encoder is pixel features, not CLIP/SigLIP.** No model downloads happen here. CLIP/SigLIP-family image-text encoders are the documented production upgrade path; the `embed()` interface is the seam to swap in.
- **Attribute extraction is rule-based heuristics**, not a learned garment detector/segmenter. It is accurate on the synthetic catalog (above) and will degrade on real photos with cluttered backgrounds, occlusion, or unusual poses.
- **Catalog is synthetic** — 60 garments drawn programmatically with Pillow. No scraped or unlicensed images; every item records source and license (CC0, generated).
- **Search is exact brute-force cosine similarity**, fine at this scale; pgvector/Qdrant is the production path.
- **No web frontend or API server yet** — the package is the core library + CLI + evaluation. FastAPI + Next.js is the next milestone per the spec.
- **No virtual try-on**, no body-size inference, no sensitive-trait inference — deliberately out of scope.

## Project layout

```text
styleverse/           # core package: features, attributes, catalog, search,
                      # rerank, explain, feedback, privacy, evaluate, demo
catalog/              # generated synthetic images + metadata.json
evaluation/           # reproducible evaluation runner
scripts/              # catalog generation script
tests/                # unittest suite (9 tests)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, data model, and upgrade paths.

## License

MIT — see [LICENSE](LICENSE).
