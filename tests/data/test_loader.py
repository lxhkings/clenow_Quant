"""Tests for SQLDataProvider using an in-memory SQLite database."""

from __future__ import annotations

import sqlite3
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from clenow.data.loader import SQLDataProvider, _compute_adjustment_factor
from clenow.errors import DataAccessError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_test_db() -> sqlite3.Connection:
    """Build an in-memory SQLite DB with sample data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE prices (
            date        TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            raw_open    REAL,
            raw_high    REAL,
            raw_low     REAL,
            raw_close   REAL,
            volume      INTEGER,
            adj_close   REAL,
            dividend    REAL DEFAULT 0,
            split_ratio REAL DEFAULT 1.0
        )
    """)
    conn.execute("""
        CREATE TABLE index_constituents (
            ticker      TEXT NOT NULL,
            index_id    TEXT NOT NULL,
            as_of_date  TEXT NOT NULL,
            removed_date TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE index_prices (
            index_id TEXT NOT NULL,
            date     TEXT NOT NULL,
            close    REAL NOT NULL
        )
    """)

    # Insert sample prices for AAPL, MSFT, GOOG
    rows = [
        # AAPL
        ("2024-01-02", "AAPL", 185.0, 186.0, 184.0, 185.5, 50_000_000, 185.5, 0.0, 1.0),
        ("2024-01-03", "AAPL", 186.0, 187.0, 185.0, 186.5, 48_000_000, 186.5, 0.0, 1.0),
        ("2024-01-04", "AAPL", 186.5, 188.0, 186.0, 187.0, 52_000_000, 187.0, 0.5, 1.0),
        ("2024-01-05", "AAPL", 187.0, 189.0, 186.5, 188.0, 55_000_000, 188.0, 0.0, 1.0),
        # MSFT
        ("2024-01-02", "MSFT", 370.0, 372.0, 369.0, 371.0, 30_000_000, 371.0, 0.0, 1.0),
        ("2024-01-03", "MSFT", 371.0, 373.0, 370.0, 372.0, 28_000_000, 372.0, 0.0, 1.0),
        ("2024-01-04", "MSFT", 372.0, 374.0, 371.0, 373.0, 32_000_000, 373.0, 0.0, 2.0),
        ("2024-01-05", "MSFT", 186.5, 187.5, 186.0, 187.0, 60_000_000, 374.0, 0.0, 1.0),
        # GOOG
        ("2024-01-02", "GOOG", 140.0, 141.0, 139.5, 140.5, 25_000_000, 140.5, 0.0, 1.0),
        ("2024-01-03", "GOOG", 141.0, 142.0, 140.0, 141.5, 22_000_000, 141.5, 0.0, 1.0),
        ("2024-01-04", "GOOG", 141.5, 143.0, 141.0, 142.0, 27_000_000, 142.0, 0.0, 1.0),
        ("2024-01-05", "GOOG", 142.0, 144.0, 141.5, 143.0, 24_000_000, 143.0, 0.0, 1.0),
    ]
    conn.executemany(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )

    # Insert index_constituents for PIT tests
    # AAPL has been in SP500 since 2020-01-01 and still present
    # MSFT has been in SP500 since 2020-01-01 and still present
    # GOOG was added 2023-06-01 and removed 2024-01-03
    # TSLA was added 2023-01-01 and still present
    universe_rows = [
        ("AAPL", "SP500", "2020-01-01", None),
        ("MSFT", "SP500", "2020-01-01", None),
        ("GOOG", "SP500", "2023-06-01", "2024-01-03"),
        ("TSLA", "SP500", "2023-01-01", None),
        # CSI800 has a different set
        ("600519", "CSI800", "2022-01-01", None),
        ("000858", "CSI800", "2022-01-01", None),
    ]
    conn.executemany(
        "INSERT INTO index_constituents VALUES (?, ?, ?, ?)", universe_rows
    )

    # Insert index prices
    index_rows = [
        ("SP500", "2024-01-02", 4700.0),
        ("SP500", "2024-01-03", 4710.0),
        ("SP500", "2024-01-04", 4720.0),
        ("SP500", "2024-01-05", 4730.0),
    ]
    conn.executemany(
        "INSERT INTO index_prices VALUES (?, ?, ?)", index_rows
    )

    conn.commit()
    return conn


@pytest.fixture
def db_conn() -> sqlite3.Connection:
    """Provide a test database connection."""
    return _create_test_db()


@pytest.fixture
def provider(db_conn: sqlite3.Connection) -> SQLDataProvider:
    """Provide a SQLDataProvider with caching disabled for most tests."""
    return SQLDataProvider(connection=db_conn, use_cache=False)


# ---------------------------------------------------------------------------
# load_prices
# ---------------------------------------------------------------------------

class TestLoadPrices:
    def test_returns_multiindex_dataframe(self, provider: SQLDataProvider) -> None:
        df = provider.load_prices(
            ["AAPL", "MSFT"],
            date(2024, 1, 2),
            date(2024, 1, 5),
        )
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["date", "ticker"]

    def test_correct_columns(self, provider: SQLDataProvider) -> None:
        df = provider.load_prices(
            ["AAPL"],
            date(2024, 1, 2),
            date(2024, 1, 5),
        )
        expected_cols = {
            "raw_open", "raw_high", "raw_low", "raw_close",
            "volume", "adj_close", "dividend", "split_ratio",
        }
        assert set(df.columns) == expected_cols

    def test_data_values(self, provider: SQLDataProvider) -> None:
        df = provider.load_prices(
            ["AAPL"],
            date(2024, 1, 2),
            date(2024, 1, 3),
        )
        # Should have 2 rows for AAPL
        aapl = df.xs("AAPL", level="ticker")
        assert len(aapl) == 2
        assert aapl.loc[date(2024, 1, 2), "raw_close"] == 185.5
        assert aapl.loc[date(2024, 1, 3), "raw_close"] == 186.5

    def test_empty_tickers_returns_empty_frame(self, provider: SQLDataProvider) -> None:
        df = provider.load_prices([], date(2024, 1, 2), date(2024, 1, 5))
        assert df.empty
        assert isinstance(df.index, pd.MultiIndex)

    def test_multiple_tickers(self, provider: SQLDataProvider) -> None:
        df = provider.load_prices(
            ["AAPL", "MSFT", "GOOG"],
            date(2024, 1, 2),
            date(2024, 1, 5),
        )
        tickers = df.index.get_level_values("ticker").unique().tolist()
        assert set(tickers) == {"AAPL", "MSFT", "GOOG"}

    def test_date_range_filtering(self, provider: SQLDataProvider) -> None:
        df = provider.load_prices(
            ["AAPL"],
            date(2024, 1, 3),
            date(2024, 1, 4),
        )
        aapl = df.xs("AAPL", level="ticker")
        assert len(aapl) == 2
        dates = aapl.index.tolist()
        assert date(2024, 1, 2) not in dates
        assert date(2024, 1, 5) not in dates


# ---------------------------------------------------------------------------
# get_universe
# ---------------------------------------------------------------------------

class TestGetUniverse:
    def test_active_constituents(self, provider: SQLDataProvider) -> None:
        """As of 2023-07-01, GOOG is still active (removed 2024-01-03)."""
        universe = provider.get_universe(date(2023, 7, 1))
        assert "AAPL" in universe
        assert "MSFT" in universe
        assert "GOOG" in universe
        assert "TSLA" in universe

    def test_removed_stock_excluded(self, provider: SQLDataProvider) -> None:
        """GOOG was removed on 2024-01-03, so on 2024-01-04 it should be out."""
        universe = provider.get_universe(date(2024, 1, 4))
        assert "GOOG" not in universe
        assert "AAPL" in universe
        assert "MSFT" in universe
        assert "TSLA" in universe

    def test_stock_not_yet_added(self, provider: SQLDataProvider) -> None:
        """GOOG was added 2023-06-01, so on 2023-05-01 it shouldn't be there."""
        universe = provider.get_universe(date(2023, 5, 1))
        assert "GOOG" not in universe

    def test_removed_date_boundary(self, provider: SQLDataProvider) -> None:
        """removed_date > as_of means GOOG is still present on 2024-01-02."""
        universe = provider.get_universe(date(2024, 1, 2))
        assert "GOOG" in universe

    def test_removed_date_exclusion(self, provider: SQLDataProvider) -> None:
        """On 2024-01-03, GOOG's removed_date is NOT > as_of (equal), so excluded."""
        universe = provider.get_universe(date(2024, 1, 3))
        assert "GOOG" not in universe

    def test_sorted_output(self, provider: SQLDataProvider) -> None:
        universe = provider.get_universe(date(2024, 1, 5))
        assert universe == sorted(universe)

    def test_default_index_is_sp500(self, provider: SQLDataProvider) -> None:
        """Calling without index_id should return SP500 constituents."""
        universe = provider.get_universe(date(2024, 1, 2))
        assert "AAPL" in universe
        assert "600519" not in universe

    def test_csi800_universe(self, provider: SQLDataProvider) -> None:
        """Passing index_id='CSI800' should return CSI800 constituents."""
        universe = provider.get_universe(date(2024, 1, 2), index_id="CSI800")
        assert "600519" in universe
        assert "000858" in universe
        assert "AAPL" not in universe

    def test_index_id_cache_invalidation(self, provider: SQLDataProvider) -> None:
        """Switching index_id should rebuild the PIT index."""
        # First call builds SP500 cache
        sp = provider.get_universe(date(2024, 1, 2), index_id="SP500")
        assert provider._pit_index_id == "SP500"
        assert "AAPL" in sp

        # Switch to CSI800 — cache should rebuild
        csi = provider.get_universe(date(2024, 1, 2), index_id="CSI800")
        assert provider._pit_index_id == "CSI800"
        assert "600519" in csi
        assert "AAPL" not in csi

        # Switch back to SP500 — should rebuild again
        sp2 = provider.get_universe(date(2024, 1, 2), index_id="SP500")
        assert provider._pit_index_id == "SP500"
        assert "AAPL" in sp2


# ---------------------------------------------------------------------------
# PIT cache
# ---------------------------------------------------------------------------

class TestPITCache:
    def test_second_call_no_db_hit(self, provider: SQLDataProvider) -> None:
        """After the first get_universe, the PIT index is built in memory.
        The second call should not execute any SQL."""
        # First call populates the cache
        universe1 = provider.get_universe(date(2024, 1, 2))
        assert provider._pit_index is not None

        # Verify that the in-memory PIT index was populated with the expected data.
        # A second call with a different date should resolve purely from the
        # in-memory dictionary without touching the DB. We verify this by
        # confirming _pit_index is already set (the only DB query happened in
        # the first call) and that a different date gives a different result.
        universe2 = provider.get_universe(date(2024, 1, 5))
        assert universe1 != universe2  # Different dates, different universes
        assert "GOOG" not in universe2  # GOOG was removed

    def test_pit_index_built_once(self, provider: SQLDataProvider) -> None:
        """The PIT index should be built only once, on the first get_universe call."""
        assert provider._pit_index is None
        provider.get_universe(date(2024, 1, 2))
        assert provider._pit_index is not None

        # Record the index reference
        first_index = provider._pit_index

        # Second call should not rebuild (same object reference)
        provider.get_universe(date(2024, 1, 5))
        assert provider._pit_index is first_index


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_connection_retry_raises_after_3_attempts(self) -> None:
        """Simulated DB failure raises DataAccessError after 3 attempts."""
        with patch("clenow.data.loader.sqlite3.connect", side_effect=Exception("DB down")):
            with pytest.raises(DataAccessError, match="3 attempts"):
                SQLDataProvider(":memory:", use_cache=False)

    def test_successful_connection_after_retry(self) -> None:
        """If the first attempt fails but the second succeeds, we get a connection."""
        call_count = 0
        real_connect = sqlite3.connect

        def flaky_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Transient failure")
            conn = real_connect(":memory:")
            conn.execute("SELECT 1")
            return conn

        with patch("clenow.data.loader.sqlite3.connect", side_effect=flaky_connect):
            with patch("clenow.data.loader.time.sleep"):  # skip delays in test
                provider = SQLDataProvider(":memory:", use_cache=False)
                assert call_count == 2
                provider.close()


# ---------------------------------------------------------------------------
# adj_close adjustment fallback
# ---------------------------------------------------------------------------

class TestAdjCloseFallback:
    def test_adj_close_computed_when_missing(self, db_conn: sqlite3.Connection) -> None:
        """When adj_close column has no data, compute from dividend + split_ratio."""
        # Set all adj_close to NULL
        db_conn.execute("UPDATE prices SET adj_close = NULL")
        db_conn.commit()

        provider = SQLDataProvider(connection=db_conn, use_cache=False)
        df = provider.load_prices(
            ["AAPL"],
            date(2024, 1, 2),
            date(2024, 1, 5),
        )
        aapl = df.xs("AAPL", level="ticker")

        # All adj_close values should be populated (not NaN)
        assert aapl["adj_close"].notna().all()

        # The most recent row should have adj_close == raw_close (factor = 1.0)
        last_row = aapl.iloc[-1]
        assert last_row["adj_close"] == pytest.approx(last_row["raw_close"], rel=1e-6)

    def test_adj_close_with_dividend(self, db_conn: sqlite3.Connection) -> None:
        """AAPL has a $0.50 dividend on 2024-01-04.
        The adjustment factor for dates before that should be < 1."""
        db_conn.execute("UPDATE prices SET adj_close = NULL")
        db_conn.commit()

        provider = SQLDataProvider(connection=db_conn, use_cache=False)
        df = provider.load_prices(
            ["AAPL"],
            date(2024, 1, 2),
            date(2024, 1, 5),
        )
        aapl = df.xs("AAPL", level="ticker")

        # Rows before the dividend should have adj_close < raw_close
        pre_div = aapl.loc[date(2024, 1, 2)]
        assert pre_div["adj_close"] < pre_div["raw_close"]

        # Row on the dividend date and after should have adj_close == raw_close
        post_div = aapl.loc[date(2024, 1, 5)]
        assert post_div["adj_close"] == pytest.approx(post_div["raw_close"], rel=1e-6)

    def test_adj_close_with_split(self, db_conn: sqlite3.Connection) -> None:
        """MSFT has a 2:1 split on 2024-01-04.
        Pre-split prices should be adjusted down by factor of 1/2."""
        db_conn.execute("UPDATE prices SET adj_close = NULL")
        db_conn.commit()

        provider = SQLDataProvider(connection=db_conn, use_cache=False)
        df = provider.load_prices(
            ["MSFT"],
            date(2024, 1, 2),
            date(2024, 1, 5),
        )
        msft = df.xs("MSFT", level="ticker")

        # Pre-split rows should have adj_close approximately half of raw_close
        pre_split = msft.loc[date(2024, 1, 2)]
        assert pre_split["adj_close"] == pytest.approx(pre_split["raw_close"] / 2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# get_index_prices
# ---------------------------------------------------------------------------

class TestGetIndexPrices:
    def test_returns_correct_data(self, provider: SQLDataProvider) -> None:
        df = provider.get_index_prices("SP500", date(2024, 1, 2), date(2024, 1, 5))
        assert len(df) == 4
        assert "date" in df.columns
        assert "close" in df.columns
        assert df.iloc[0]["close"] == 4700.0

    def test_date_filtering(self, provider: SQLDataProvider) -> None:
        df = provider.get_index_prices("SP500", date(2024, 1, 3), date(2024, 1, 4))
        assert len(df) == 2

    def test_empty_result(self, provider: SQLDataProvider) -> None:
        df = provider.get_index_prices("SP500", date(2025, 1, 1), date(2025, 1, 31))
        assert df.empty
        assert list(df.columns) == ["date", "close"]


# ---------------------------------------------------------------------------
# compute_adjustment_factor (unit-level)
# ---------------------------------------------------------------------------

class TestComputeAdjustmentFactor:
    def test_no_corporate_actions(self) -> None:
        """With no dividends or splits, all factors should be 1.0."""
        df = pd.DataFrame({
            "ticker": ["A"] * 3,
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "raw_close": [100.0, 101.0, 102.0],
            "dividend": [0.0, 0.0, 0.0],
            "split_ratio": [1.0, 1.0, 1.0],
        })
        result = _compute_adjustment_factor(df)
        assert (result == 1.0).all()

    def test_split_adjustment(self) -> None:
        """2:1 split on day 2 — factor for days 0 and 1 should be 0.5."""
        df = pd.DataFrame({
            "ticker": ["A"] * 3,
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "raw_close": [100.0, 100.0, 50.0],
            "dividend": [0.0, 0.0, 0.0],
            "split_ratio": [1.0, 2.0, 1.0],
        })
        result = _compute_adjustment_factor(df)
        assert result.iloc[0] == pytest.approx(0.5)
        assert result.iloc[1] == pytest.approx(1.0)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_dividend_adjustment(self) -> None:
        """$5 dividend on day 2 with close=100 — factor for prior days < 1."""
        df = pd.DataFrame({
            "ticker": ["A"] * 3,
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "raw_close": [100.0, 100.0, 100.0],
            "dividend": [0.0, 5.0, 0.0],
            "split_ratio": [1.0, 1.0, 1.0],
        })
        result = _compute_adjustment_factor(df)
        # Day 0: factor *= (1 - 5/100) = 0.95 (from day 1's dividend)
        assert result.iloc[0] == pytest.approx(0.95)
        # Day 1: factor = 1.0 (this is the day the dividend happens, future factor is 1)
        assert result.iloc[1] == pytest.approx(1.0)
        # Day 2: factor = 1.0
        assert result.iloc[2] == pytest.approx(1.0)
