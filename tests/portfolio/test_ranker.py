"""Tests for the portfolio ranker module."""

from clenow.portfolio.ranker import rank_by_score


class TestRankByScore:
    """rank_by_score — sort by score descending, take top N%."""

    def test_normal_ranking(self):
        """Stocks sorted by score descending, top 20% returned."""
        scores = {
            "AAPL": 2.5,
            "MSFT": 1.8,
            "GOOG": 1.2,
            "AMZN": 0.9,
            "TSLA": 0.3,
            "META": -0.1,
            "NVDA": -0.5,
            "JPM": -1.0,
            "V": -1.5,
            "JNJ": -2.0,
        }
        result = rank_by_score(scores, top_pct=0.20)
        # 10 * 0.20 = 2 → top 2
        assert result == ["AAPL", "MSFT"]

    def test_empty_scores(self):
        """Empty scores dict returns empty list."""
        assert rank_by_score({}) == []

    def test_all_zero_scores(self):
        """Zero-score stocks still participate in ranking (at the bottom)."""
        scores = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        result = rank_by_score(scores, top_pct=0.50)
        # 4 * 0.5 = 2 → top 2
        assert len(result) == 2

    def test_top_pct_rounding(self):
        """round() is used for top_pct calculation."""
        # 7 * 0.20 = 1.4 → round to 1
        scores = {f"T{i}": float(7 - i) for i in range(7)}
        result = rank_by_score(scores, top_pct=0.20)
        assert len(result) == 1
        assert result[0] == "T0"  # highest score

    def test_single_stock(self):
        """Single stock universe: top 20% = 1 stock."""
        scores = {"AAPL": 1.5}
        result = rank_by_score(scores, top_pct=0.20)
        assert result == ["AAPL"]

    def test_top_pct_one_hundred(self):
        """top_pct=1.0 returns all stocks in rank order."""
        scores = {"A": 3.0, "B": 2.0, "C": 1.0}
        result = rank_by_score(scores, top_pct=1.0)
        assert result == ["A", "B", "C"]

    def test_negative_scores_ranked_correctly(self):
        """Negative scores are still ranked (least negative first)."""
        scores = {"X": -0.5, "Y": -2.0, "Z": -0.1}
        result = rank_by_score(scores, top_pct=0.50)
        # 3 * 0.5 = 1.5 → round to 2
        assert result[0] == "Z"  # -0.1 is highest
        assert result[1] == "X"  # -0.5 is second
