"""Session-scoped user feedback.

Actions: relevant, not relevant, save, hide. Feedback affects later ranking
within the same session: relevant items get a boost, not-relevant items are
demoted, hidden items are removed. Nothing persists beyond the session.
"""

from __future__ import annotations

from .search import SearchResult

RELEVANT_BOOST = 0.15
NOT_RELEVANT_PENALTY = 0.30


class SessionFeedback:
    """Per-session feedback store (in-memory; dies with the session)."""

    def __init__(self):
        self.relevant: set[str] = set()
        self.not_relevant: set[str] = set()
        self.saved: set[str] = set()
        self.hidden: set[str] = set()

    def mark_relevant(self, product_id: str) -> None:
        self.relevant.add(product_id)
        self.not_relevant.discard(product_id)

    def mark_not_relevant(self, product_id: str) -> None:
        self.not_relevant.add(product_id)
        self.relevant.discard(product_id)

    def save(self, product_id: str) -> None:
        self.saved.add(product_id)

    def hide(self, product_id: str) -> None:
        self.hidden.add(product_id)

    def apply(self, results: list[SearchResult]) -> list[SearchResult]:
        """Re-rank results using this session's feedback."""
        adjusted: list[SearchResult] = []
        for r in results:
            pid = r.product.id
            if pid in self.hidden:
                continue
            score = r.score
            if pid in self.relevant:
                score += RELEVANT_BOOST
            if pid in self.not_relevant:
                score -= NOT_RELEVANT_PENALTY
            adjusted.append(SearchResult(r.product, score))
        return sorted(adjusted, key=lambda r: r.score, reverse=True)
