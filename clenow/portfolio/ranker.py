"""Rank stocks by momentum score — top N% of the universe."""

from __future__ import annotations


def rank_by_score(
    universe_scores: dict[str, float],
    top_pct: float = 0.20,
) -> list[str]:
    """Return tickers sorted by score descending, keeping the top *top_pct*.

    Args:
        universe_scores: mapping of ticker -> momentum score (can be 0 or negative).
        top_pct: fraction of the universe to keep (e.g. 0.20 = top 20%).

    Returns:
        List of tickers in rank order (highest score first).
        Empty input yields an empty list.
    """
    if not universe_scores:
        return []

    sorted_tickers = sorted(
        universe_scores.keys(),
        key=lambda t: universe_scores[t],
        reverse=True,
    )

    count = max(1, round(len(universe_scores) * top_pct))
    return sorted_tickers[:count]
