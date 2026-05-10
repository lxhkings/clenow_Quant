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