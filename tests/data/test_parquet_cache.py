"""Unit tests for ParquetCache."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from clenow.data.parquet_cache import (
    MANIFEST_COLUMNS,
    PRICE_COLUMNS,
    ParquetCache,
)


def _stub_fetcher_empty(tickers, start, end):
    return pd.DataFrame(columns=["date", "ticker", *PRICE_COLUMNS])


def _ticker_rows(ticker: str, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates,
        "ticker": [ticker] * len(dates),
        "raw_open": [100.0] * len(dates),
        "raw_high": [101.0] * len(dates),
        "raw_low": [99.0] * len(dates),
        "raw_close": [100.5] * len(dates),
        "volume": [1_000_000] * len(dates),
        "adj_close": [100.5] * len(dates),
        "dividend": [0.0] * len(dates),
        "split_ratio": [1.0] * len(dates),
    })


class TestManifestReadEmpty:
    def test_returns_empty_dataframe_when_no_manifest(self, tmp_path: Path) -> None:
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        manifest = cache._read_manifest()
        assert list(manifest.columns) == MANIFEST_COLUMNS
        assert len(manifest) == 0


class TestIdentifyGaps:
    def _make_cache(self, tmp_path):
        return ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)

    def _manifest(self, rows):
        return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    def test_ticker_absent_full_range_gap(self, tmp_path):
        cache = self._make_cache(tmp_path)
        manifest = self._manifest([])
        gaps = cache._identify_gaps(
            manifest, ["AAPL"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert gaps == {"AAPL": [(date(2024, 1, 1), date(2024, 1, 31))]}

    def test_full_coverage_no_gap(self, tmp_path):
        cache = self._make_cache(tmp_path)
        manifest = self._manifest([
            ("AAPL", date(2023, 1, 1), date(2025, 1, 1), 500, datetime(2025, 1, 1)),
        ])
        gaps = cache._identify_gaps(
            manifest, ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
        )
        assert gaps == {}

    def test_left_extension_only(self, tmp_path):
        cache = self._make_cache(tmp_path)
        manifest = self._manifest([
            ("AAPL", date(2024, 6, 1), date(2025, 1, 1), 200, datetime(2025, 1, 1)),
        ])
        gaps = cache._identify_gaps(
            manifest, ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
        )
        # gap = (start, manifest_min - 1day)
        assert gaps == {"AAPL": [(date(2024, 1, 1), date(2024, 5, 31))]}

    def test_right_extension_only(self, tmp_path):
        cache = self._make_cache(tmp_path)
        manifest = self._manifest([
            ("AAPL", date(2024, 1, 1), date(2024, 6, 30), 130, datetime(2024, 7, 1)),
        ])
        gaps = cache._identify_gaps(
            manifest, ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
        )
        assert gaps == {"AAPL": [(date(2024, 7, 1), date(2024, 12, 31))]}

    def test_both_sides_extension(self, tmp_path):
        cache = self._make_cache(tmp_path)
        manifest = self._manifest([
            ("AAPL", date(2024, 4, 1), date(2024, 8, 31), 100, datetime(2024, 9, 1)),
        ])
        gaps = cache._identify_gaps(
            manifest, ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
        )
        assert gaps == {
            "AAPL": [
                (date(2024, 1, 1), date(2024, 3, 31)),
                (date(2024, 9, 1), date(2024, 12, 31)),
            ]
        }

    def test_multiple_tickers_mixed_states(self, tmp_path):
        cache = self._make_cache(tmp_path)
        manifest = self._manifest([
            ("AAPL", date(2024, 1, 1), date(2025, 1, 1), 250, datetime(2025, 1, 1)),
            ("MSFT", date(2024, 6, 1), date(2024, 12, 31), 150, datetime(2025, 1, 1)),
            # GOOG absent entirely
        ])
        gaps = cache._identify_gaps(
            manifest, ["AAPL", "MSFT", "GOOG"],
            date(2024, 1, 1), date(2024, 12, 31),
        )
        assert "AAPL" not in gaps  # fully covered
        assert gaps["MSFT"] == [(date(2024, 1, 1), date(2024, 5, 31))]
        assert gaps["GOOG"] == [(date(2024, 1, 1), date(2024, 12, 31))]


class TestWriteTickerAtomic:
    def test_creates_new_file(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        rows = _ticker_rows("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
        cache._write_ticker_atomic("AAPL", rows)
        path = tmp_path / "parquet" / "AAPL.parquet"
        assert path.exists()
        out = pd.read_parquet(path)
        assert len(out) == 2
        assert set(out.columns) == {"date", "ticker", *PRICE_COLUMNS}

    def test_appends_and_dedupes_by_date(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        first = _ticker_rows("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
        second = _ticker_rows("AAPL", [date(2024, 1, 3), date(2024, 1, 4)])  # 1/3 overlaps
        cache._write_ticker_atomic("AAPL", first)
        cache._write_ticker_atomic("AAPL", second)
        out = pd.read_parquet(tmp_path / "parquet" / "AAPL.parquet")
        assert sorted(out["date"].tolist()) == [
            date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
        ]

    def test_no_temp_file_left_behind(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        rows = _ticker_rows("AAPL", [date(2024, 1, 2)])
        cache._write_ticker_atomic("AAPL", rows)
        leftover = list((tmp_path / "parquet").glob("*.tmp"))
        assert leftover == []


class TestUpdateManifestEntries:
    def test_creates_manifest_on_first_write(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        cache._write_ticker_atomic(
            "AAPL", _ticker_rows("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
        )
        cache._update_manifest_entries(["AAPL"])
        manifest = cache._read_manifest()
        assert len(manifest) == 1
        row = manifest.iloc[0]
        assert row["ticker"] == "AAPL"
        assert row["min_date"] == date(2024, 1, 2)
        assert row["max_date"] == date(2024, 1, 3)
        assert row["row_count"] == 2

    def test_updates_existing_entry(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        cache._write_ticker_atomic(
            "AAPL", _ticker_rows("AAPL", [date(2024, 1, 2)])
        )
        cache._update_manifest_entries(["AAPL"])
        cache._write_ticker_atomic(
            "AAPL", _ticker_rows("AAPL", [date(2024, 1, 3), date(2024, 1, 4)])
        )
        cache._update_manifest_entries(["AAPL"])
        manifest = cache._read_manifest()
        row = manifest.iloc[0]
        assert row["min_date"] == date(2024, 1, 2)
        assert row["max_date"] == date(2024, 1, 4)
        assert row["row_count"] == 3

    def test_multiple_tickers(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        cache._write_ticker_atomic(
            "AAPL", _ticker_rows("AAPL", [date(2024, 1, 2)])
        )
        cache._write_ticker_atomic(
            "MSFT", _ticker_rows("MSFT", [date(2024, 1, 3), date(2024, 1, 4)])
        )
        cache._update_manifest_entries(["AAPL", "MSFT"])
        manifest = cache._read_manifest().set_index("ticker")
        assert manifest.loc["AAPL", "min_date"] == date(2024, 1, 2)
        assert manifest.loc["MSFT", "max_date"] == date(2024, 1, 4)


class TestFetchAndPersist:
    def test_no_gaps_no_fetch(self, tmp_path):
        calls = []
        def fetcher(tickers, start, end):
            calls.append((sorted(tickers), start, end))
            return _ticker_rows(tickers[0], [start])
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        cache._fetch_and_persist({})
        assert calls == []

    def test_groups_by_date_range(self, tmp_path):
        calls = []
        def fetcher(tickers, start, end):
            calls.append((sorted(tickers), start, end))
            rows = []
            for t in tickers:
                rows.append(_ticker_rows(t, [start, end]))
            return pd.concat(rows, ignore_index=True)
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        gaps = {
            "AAPL": [(date(2024, 1, 1), date(2024, 1, 31))],
            "MSFT": [(date(2024, 1, 1), date(2024, 1, 31))],   # same range as AAPL
            "GOOG": [(date(2024, 2, 1), date(2024, 2, 28))],   # different range
        }
        cache._fetch_and_persist(gaps)
        # Two unique date ranges → two fetcher calls
        assert len(calls) == 2
        # AAPL+MSFT share one call
        first_call_tickers = next(c[0] for c in calls if c[1] == date(2024, 1, 1))
        assert first_call_tickers == ["AAPL", "MSFT"]
        # GOOG alone
        third_call_tickers = next(c[0] for c in calls if c[1] == date(2024, 2, 1))
        assert third_call_tickers == ["GOOG"]

    def test_multi_gap_per_ticker_dispatches_separately(self, tmp_path):
        calls = []
        def fetcher(tickers, start, end):
            calls.append((sorted(tickers), start, end))
            return pd.concat(
                [_ticker_rows(t, [start, end]) for t in tickers],
                ignore_index=True,
            )
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        gaps = {
            "AAPL": [
                (date(2024, 1, 1), date(2024, 1, 31)),
                (date(2024, 6, 1), date(2024, 6, 30)),
            ],
        }
        cache._fetch_and_persist(gaps)
        assert len(calls) == 2

    def test_persists_files_and_manifest(self, tmp_path):
        def fetcher(tickers, start, end):
            return pd.concat(
                [_ticker_rows(t, [start, end]) for t in tickers],
                ignore_index=True,
            )
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        cache._fetch_and_persist({
            "AAPL": [(date(2024, 1, 2), date(2024, 1, 5))],
        })
        assert (tmp_path / "parquet" / "AAPL.parquet").exists()
        manifest = cache._read_manifest()
        assert manifest.iloc[0]["ticker"] == "AAPL"
        assert manifest.iloc[0]["row_count"] == 2  # _ticker_rows yields exactly 2 dates

    def test_handles_empty_fetcher_response(self, tmp_path):
        """Fetcher returns empty DataFrame - should log warning, no file created."""
        def fetcher(tickers, start, end):
            return pd.DataFrame(columns=["date", "ticker", *PRICE_COLUMNS])  # empty
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        cache._fetch_and_persist({"AAPL": [(date(2024, 1, 1), date(2024, 1, 31))]})
        # No parquet file should be created
        assert not (tmp_path / "parquet" / "AAPL.parquet").exists()
        # Manifest should not have AAPL entry
        manifest = cache._read_manifest()
        assert len(manifest) == 0


class TestQueryViaDuckDB:
    def _seed(self, cache, ticker, dates):
        cache._write_ticker_atomic(ticker, _ticker_rows(ticker, dates))

    def test_returns_only_requested_tickers(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        self._seed(cache, "AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
        self._seed(cache, "MSFT", [date(2024, 1, 2), date(2024, 1, 3)])
        self._seed(cache, "GOOG", [date(2024, 1, 2), date(2024, 1, 3)])

        df = cache._query_via_duckdb(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert sorted(df["ticker"].unique()) == ["AAPL", "MSFT"]

    def test_returns_only_requested_dates(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        self._seed(
            cache, "AAPL",
            [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1)],
        )
        df = cache._query_via_duckdb(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert sorted(df["date"].tolist()) == [date(2024, 1, 2), date(2024, 1, 3)]

    def test_empty_when_cache_dir_empty(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        df = cache._query_via_duckdb(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert df.empty
        assert list(df.columns) == ["date", "ticker", *PRICE_COLUMNS]

    def test_columns_match_schema(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        self._seed(cache, "AAPL", [date(2024, 1, 2)])
        df = cache._query_via_duckdb(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert list(df.columns) == ["date", "ticker", *PRICE_COLUMNS]


class TestLoadEndToEnd:
    def test_cold_load_calls_fetcher_then_caches(self, tmp_path):
        calls = []
        def fetcher(tickers, start, end):
            calls.append(sorted(tickers))
            rows = []
            for t in tickers:
                rows.append(_ticker_rows(t, [date(2024, 1, 2), date(2024, 1, 3)]))
            return pd.concat(rows, ignore_index=True)
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        df = cache.load(["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 31))

        # MultiIndex (date, ticker)
        assert df.index.names == ["date", "ticker"]
        assert list(df.columns) == PRICE_COLUMNS
        assert len(df) == 4  # 2 tickers x 2 dates
        assert calls == [["AAPL", "MSFT"]]

    def test_warm_load_skips_fetcher(self, tmp_path):
        calls = []
        def fetcher(tickers, start, end):
            calls.append(sorted(tickers))
            return pd.concat(
                [_ticker_rows(t, [date(2024, 1, 2), date(2024, 1, 3)])
                 for t in tickers],
                ignore_index=True,
            )
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        cache.load(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))   # warms cache
        cache.load(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))   # warm

        assert len(calls) == 1  # second call did NOT hit fetcher

    def test_partial_load_only_fetches_gap(self, tmp_path):
        calls = []
        def fetcher(tickers, start, end):
            calls.append((sorted(tickers), start, end))
            return pd.concat(
                [_ticker_rows(t, [start, end]) for t in tickers],
                ignore_index=True,
            )
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        # Seed: AAPL covered for Jan only
        cache.load(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
        calls.clear()

        # Now ask for Jan + Feb. Should fetch only Feb gap.
        cache.load(["AAPL"], date(2024, 1, 1), date(2024, 2, 28))
        # Exactly one fetch, for the Feb gap
        assert len(calls) == 1
        _, start, end = calls[0]
        assert start == date(2024, 2, 1)
        assert end == date(2024, 2, 28)

    def test_returns_only_requested_window(self, tmp_path):
        def fetcher(tickers, start, end):
            return pd.concat(
                [_ticker_rows(t, [date(2024, 1, 2), date(2024, 6, 3)])
                 for t in tickers],
                ignore_index=True,
            )
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=fetcher)
        df = cache.load(["AAPL"], date(2024, 1, 1), date(2024, 3, 31))
        assert df.reset_index()["date"].tolist() == [date(2024, 1, 2)]

    def test_empty_tickers_returns_empty_dataframe(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        df = cache.load([], date(2024, 1, 1), date(2024, 1, 31))
        assert df.empty
        assert df.index.names == ["date", "ticker"]


class TestRebuildManifest:
    def test_rebuilds_from_files_after_manifest_lost(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path, db_fetcher=_stub_fetcher_empty)
        cache._write_ticker_atomic(
            "AAPL", _ticker_rows("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
        )
        cache._write_ticker_atomic(
            "MSFT", _ticker_rows("MSFT", [date(2024, 1, 2)])
        )
        cache._update_manifest_entries(["AAPL", "MSFT"])

        # Simulate manifest loss
        cache.manifest_path.unlink()
        assert not cache.manifest_path.exists()

        cache.rebuild_manifest()
        manifest = cache._read_manifest()
        assert sorted(manifest["ticker"].tolist()) == ["AAPL", "MSFT"]
