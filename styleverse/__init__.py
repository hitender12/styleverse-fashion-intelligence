"""Styleverse: multimodal fashion intelligence.

Upload a look, discover visually similar catalog pieces, and refine by
budget, category and style attributes. All features are computed from pixels
in this reference implementation; CLIP/SigLIP encoders are the documented
production upgrade path (see README.md).
"""

__version__ = "0.1.0"

from .features import embed, dominant_color_family, COLOR_FAMILIES
from .attributes import extract_attributes
from .catalog import Product, generate_catalog, load_catalog
from .search import VectorIndex, SearchResult
from .rerank import rerank, Constraints
from .explain import explain
from .feedback import SessionFeedback
from .privacy import Session, ingest_upload

__all__ = [
    "embed",
    "dominant_color_family",
    "COLOR_FAMILIES",
    "extract_attributes",
    "Product",
    "generate_catalog",
    "load_catalog",
    "VectorIndex",
    "SearchResult",
    "rerank",
    "Constraints",
    "explain",
    "SessionFeedback",
    "Session",
    "ingest_upload",
]
