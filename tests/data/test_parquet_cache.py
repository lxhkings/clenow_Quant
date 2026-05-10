"""Unit tests for ParquetCache."""

from __future__ import annotations

from datetime import date
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