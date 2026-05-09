"""Tests for the Parquet price-data cache."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from clenow.data.cache import (
    CACHE_DIR,
    _cache_key,
    load_from_cache,
    save_to_cache,
)


def _make_sample_df() -> pd.DataFrame:
    """Create a small price DataFrame with the expected MultiIndex."""
    idx = pd.MultiIndex.from_tuples(
        [
            (date(2024, 1, 2), "AAPL"),
            (date(2024, 1, 3), "AAPL"),
            (date(2024, 1, 2), "MSFT"),
            (date(2024, 1, 3), "MSFT"),
        ],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "raw_open": [185.0, 186.0, 370.0, 371.0],
            "raw_high": [186.0, 187.0, 372.0, 373.0],
            "raw_low": [184.0, 185.0, 369.0, 370.0],
            "raw_close": [185.5, 186.5, 371.0, 372.0],
            "volume": [50_000_000, 48_000_000, 30_000_000, 28_000_000],
            "adj_close": [185.5, 186.5, 371.0, 372.0],
            "dividend": [0.0, 0.0, 0.0, 0.0],
            "split_ratio": [1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def _use_tmp_cache(tmp_path: Path, monkeypatch) -> None:
    """Redirect CACHE_DIR to a temp directory for every test."""
    import clenow.data.cache as cache_mod
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")


class TestCacheKey:
    def test_deterministic(self) -> None:
        tickers = ["AAPL", "MSFT"]
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)
        key1 = _cache_key(tickers, start, end)
        key2 = _cache_key(tickers, start, end)
        assert key1 == key2

    def test_ticker_order_independent(self) -> None:
        key1 = _cache_key(["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 31))
        key2 = _cache_key(["MSFT", "AAPL"], date(2024, 1, 1), date(2024, 1, 31))
        assert key1 == key2

    def test_different_dates_different_key(self) -> None:
        key1 = _cache_key(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
        key2 = _cache_key(["AAPL"], date(2024, 2, 1), date(2024, 2, 28))
        assert key1 != key2


class TestCacheMiss:
    def test_load_returns_none_on_miss(self) -> None:
        result = load_from_cache(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert result is None


class TestCacheHit:
    def test_save_then_load(self) -> None:
        df = _make_sample_df()
        tickers = ["AAPL", "MSFT"]
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)

        save_to_cache(df, tickers, start, end)
        loaded = load_from_cache(tickers, start, end)

        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df)


class TestCacheInvalidation:
    def test_different_date_range_is_miss(self) -> None:
        """A different date range produces a different cache key, so it's a miss."""
        df = _make_sample_df()
        tickers = ["AAPL", "MSFT"]
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)

        save_to_cache(df, tickers, start, end)

        # Requesting a different date range produces a different key → miss
        result = load_from_cache(tickers, date(2024, 1, 1), date(2024, 6, 30))
        assert result is None

    def test_different_tickers_is_miss(self) -> None:
        """Different tickers produce a different cache key, so it's a miss."""
        df = _make_sample_df()
        save_to_cache(df, ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 31))

        result = load_from_cache(["GOOG"], date(2024, 1, 1), date(2024, 1, 31))
        assert result is None


class TestCacheCorruption:
    def test_corrupted_file_returns_none(self, tmp_path) -> None:
        """If a cache file is corrupted, load_from_cache returns None and removes the file."""
        import clenow.data.cache as cache_mod
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)

        # Write garbage to a cache file
        key = _cache_key(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
        cache_file = cache_dir / f"{key}.parquet"
        cache_file.write_text("not a parquet file")

        result = load_from_cache(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
        assert result is None
        # The corrupted file should have been removed
        assert not cache_file.exists()
