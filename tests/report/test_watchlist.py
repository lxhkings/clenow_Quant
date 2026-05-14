"""Tests for the watchlist report builder."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from clenow.config import Config
from clenow.markets import get_profile
from clenow.report.watchlist import (
    WatchlistRow,
    build_watchlist,
    render_markdown,
)


def _make_prices(tickers: list[str], start: date, end: date, drift: float = 0.001) -> pd.DataFrame:
    """Build a (date, ticker) MultiIndex frame with monotone uptrend per ticker."""
    dates = pd.bdate_range(start, end)
    rows = []
    rng = np.random.default_rng(0)
    for i, t in enumerate(tickers):
        # Different drift per ticker to give distinct scores.
        local_drift = drift + 0.0002 * i
        rets = rng.normal(local_drift, 0.005, len(dates))
        adj = np.cumprod(1 + rets) * 100
        for d, p in zip(dates, adj):
            rows.append((d, t, p, p, 1_000_000_000, p))
    df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close", "raw_close", "volume", "raw_open"])
    return df.set_index(["date", "ticker"]).sort_index()


def test_build_watchlist_returns_rows_sorted_by_score():
    as_of = date(2026, 5, 8)
    universe = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    profile = get_profile("us")
    config = Config(market="US")

    prices = _make_prices(universe, as_of - pd.Timedelta(days=450), as_of)

    dp = MagicMock()
    dp.get_universe.return_value = universe
    dp.load_prices.return_value = prices
    # Regime filter calls load_prices for index_id; return same monotone uptrend.
    # Use ticker = profile.regime_index_id (SP500) row.
    idx_dates = pd.bdate_range(as_of - pd.Timedelta(days=450), as_of)
    idx_adj = np.cumprod(1 + np.full(len(idx_dates), 0.001)) * 1000
    idx_df = pd.DataFrame({
        "date": idx_dates, "adj_close": idx_adj, "raw_close": idx_adj,
        "volume": 0, "raw_open": idx_adj,
    }).set_index("date")
    dp.load_index_prices = MagicMock(return_value=idx_df)

    sector_map = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}
    rows = build_watchlist(as_of, config, profile, dp, sector_map)

    assert all(isinstance(r, WatchlistRow) for r in rows)
    scores = [r.score for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in rows] == list(range(1, len(rows) + 1))


def test_render_markdown_includes_header_and_tables():
    as_of = date(2026, 5, 8)
    profile = get_profile("us")
    config = Config(market="US")
    rows = [
        WatchlistRow(
            rank=1, ticker="NVDA", sector="Tech",
            score=1.234, slope=0.0042, r_squared=0.92,
            annualized_return=1.86, price=920.50, sma100=745.20, dist_pct=23.5,
        ),
        WatchlistRow(
            rank=2, ticker="XOM", sector=None,
            score=0.500, slope=0.0020, r_squared=0.80,
            annualized_return=0.65, price=110.00, sma100=105.00, dist_pct=4.8,
        ),
    ]
    md = render_markdown(rows, as_of, profile, config, total_universe=500)
    assert "强势股名单" in md
    assert "US" in md
    assert "2026-05-08" in md
    assert "NVDA" in md and "Tech" in md
    assert "1.234" in md          # score format
    assert "0.0042" in md         # slope format
    assert "0.92" in md           # r² format
    assert "+186.0%" in md or "+186%" in md  # annualized
    assert "+23.5%" in md         # dist_pct sign
    assert "—" in md              # missing sector dash
    assert "## 顺序表" in md
    assert "## 行业汇总" in md


def test_render_markdown_empty_rows():
    as_of = date(2026, 5, 8)
    profile = get_profile("us")
    config = Config(market="US")
    md = render_markdown([], as_of, profile, config, total_universe=500)
    assert "本日无入选股" in md
    assert "强势股名单" in md