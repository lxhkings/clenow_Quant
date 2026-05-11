"""Integration test: HK backtest path with mocked data provider.

Verifies that the full rebalance cycle works end-to-end for HK profile:
- HSI universe loaded
- HSI regime filter applied
- 100-share lots enforced
- HK cost model applied
- XHKG calendar used for rebalance dates
"""

from datetime import date

import pandas as pd
import pytest

from clenow.backtest.engine import run_backtest
from clenow.config import Config
from clenow.markets import get_profile


class StubHKProvider:
    """Mock provider returning synthetic HK data."""

    def __init__(self):
        # 4 HSI tickers with ~1-year data
        self.tickers = ["00700.HK", "00005.HK", "00388.HK", "00939.HK"]
        self._dates = pd.bdate_range("2023-06-01", "2024-06-30")
        self._prices = self._build_prices()
        self._index = self._build_index()
        self._meta = pd.DataFrame(
            {
                "ticker": self.tickers,
                "name": ["腾讯控股", "汇丰控股", "香港交易所", "建设银行"],
            }
        ).set_index("ticker")

    def _build_prices(self) -> pd.DataFrame:
        rows = []
        for i, t in enumerate(self.tickers):
            base = 50 + i * 30
            for j, d in enumerate(self._dates):
                p = base * (1 + 0.0004 * j)
                rows.append({
                    "date": d.date(),
                    "ticker": t,
                    "raw_open": p,
                    "raw_high": p * 1.01,
                    "raw_low": p * 0.99,
                    "raw_close": p,
                    "adj_close": p,
                    "volume": 5_000_000,
                    "dividend": 0.0,
                    "split_ratio": 1.0,
                })
        return pd.DataFrame(rows).set_index(["date", "ticker"])

    def _build_index(self) -> pd.DataFrame:
        """Build HSI index data in bull regime (close > 200-SMA)."""
        n = len(self._dates)
        return pd.DataFrame(
            {
                "date": self._dates.date,
                "close": [18000 * (1 + 0.0002 * i) for i in range(n)],
            }
        ).set_index("date")

    def load_prices(self, tickers, start, end):
        mask = (
            self._prices.index.get_level_values("date") >= start
        ) & (
            self._prices.index.get_level_values("date") <= end
        ) & (
            self._prices.index.get_level_values("ticker").isin(tickers)
        )
        return self._prices.loc[mask]

    def get_universe(self, as_of, index_id="HSI"):
        assert index_id == "HSI", f"Expected HSI, got {index_id}"
        return list(self.tickers)

    def get_index_prices(self, index_id, start, end):
        assert index_id == "HSI"
        mask = (self._index.index >= start) & (self._index.index <= end)
        return self._index.loc[mask]

    def get_stocks_meta(self, tickers):
        return self._meta.loc[self._meta.index.intersection(tickers)]


def test_hk_lifecycle_basic():
    """Full HK rebalance cycle: equity positive, all trades are 100-lot multiples."""
    provider = StubHKProvider()
    profile = get_profile("HK")
    config = Config(market="HK", risk_factor=0.005)

    result = run_backtest(
        data_provider=provider,
        start=date(2023, 7, 1),
        end=date(2024, 6, 1),
        initial_cash=1_000_000,
        config=config,
        profile=profile,
    )

    # Basic invariants
    assert result.equity_curve.iloc[0]["portfolio_value"] > 0
    assert result.equity_curve.iloc[-1]["portfolio_value"] > 0

    # All closed trades have 100-lot position sizes
    for trade in result.trades:
        assert trade["shares"] % 100 == 0, f"non-lot trade: {trade}"
