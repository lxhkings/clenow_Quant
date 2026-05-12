# Backtest Preload Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate repeated 450-day price data loads in the backtest loop by preloading all price data once at startup and serving each iteration from an in-memory slice.

**Architecture:** Add a `preloaded_prices` parameter (optional, backward-compat) to `compute_target_portfolio`. Before the main loop in `run_backtest`, collect the full universe superset and load all data once. Each iteration slices the in-memory DataFrame instead of querying DuckDB. Also precompute the regime filter as a `{date: bool}` dict to skip repeated `get_index_prices` calls.

**Tech Stack:** Python, pandas MultiIndex DataFrame, `pd.Timestamp` for index operations.

---

## File Map

| File | Change |
|------|--------|
| `clenow/backtest/engine.py` | Add `_slice_prices()` helper; add `preloaded_prices` + `bear_regime_cache` params to `compute_target_portfolio`; add `_precompute_bear_regime()` helper; preload in `run_backtest` main loop preamble; pass preloaded into `_detect_delistings`, exec_prices, adv_hist, corp actions |
| `clenow/portfolio/selector.py` | Add `bear_regime_cache: dict[date, bool] | None = None` to `apply_filters`; skip `_is_bear_regime` when cache provided |
| `tests/backtest/test_engine.py` | Tests for `_slice_prices`, `_precompute_bear_regime`, and `compute_target_portfolio` with `preloaded_prices` |
| `tests/portfolio/test_selector.py` | Test `apply_filters` with `bear_regime_cache` |

---

## Task 1: Add `_slice_prices` helper

**Files:**
- Modify: `clenow/backtest/engine.py` (add after imports section, before `_select_one_per_sector`)
- Test: `tests/backtest/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/backtest/test_engine.py — add to existing file

import pandas as pd
from datetime import date
from clenow.backtest.engine import _slice_prices


def _make_prices(tickers, dates):
    """Helper: make a MultiIndex (date, ticker) DataFrame for testing."""
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": pd.Timestamp(d),
                "ticker": t,
                "raw_close": 100.0,
                "raw_open": 99.0,
                "raw_high": 101.0,
                "raw_low": 98.0,
                "adj_close": 100.0,
                "volume": 1_000_000.0,
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df


def test_slice_prices_filters_date_range():
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    tickers = ["AAPL", "MSFT"]
    preloaded = _make_prices(tickers, dates)

    result = _slice_prices(preloaded, ["AAPL"], date(2024, 1, 1), date(2024, 1, 2))

    date_vals = result.index.get_level_values("date")
    ticker_vals = result.index.get_level_values("ticker")
    assert all(d <= pd.Timestamp(date(2024, 1, 2)) for d in date_vals)
    assert set(ticker_vals) == {"AAPL"}


def test_slice_prices_returns_empty_for_missing_ticker():
    dates = [date(2024, 1, 1)]
    preloaded = _make_prices(["AAPL"], dates)

    result = _slice_prices(preloaded, ["TSLA"], date(2024, 1, 1), date(2024, 1, 1))
    assert result.empty


def test_slice_prices_handles_none():
    result = _slice_prices(None, ["AAPL"], date(2024, 1, 1), date(2024, 1, 1))
    assert result.empty
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/backtest/test_engine.py -k "slice_prices" -v --no-cov
```

Expected: `ImportError` or `AttributeError: module 'clenow.backtest.engine' has no attribute '_slice_prices'`

- [ ] **Step 3: Implement `_slice_prices` in `engine.py`**

Add this function after the `_PRICE_LOOKBACK_CALENDAR` constant (around line 51) and before `_select_one_per_sector`:

```python
def _slice_prices(
    preloaded: pd.DataFrame | None,
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Extract date+ticker slice from a preloaded (date, ticker) MultiIndex DataFrame."""
    _empty = pd.DataFrame(columns=["raw_close", "adj_close", "raw_high", "raw_low", "volume"])
    if preloaded is None or preloaded.empty:
        return _empty
    date_idx = preloaded.index.get_level_values("date")
    ticker_idx = preloaded.index.get_level_values("ticker")
    mask = (
        (date_idx >= pd.Timestamp(start))
        & (date_idx <= pd.Timestamp(end))
        & (ticker_idx.isin(set(tickers)))
    )
    sliced = preloaded.loc[mask]
    return sliced if not sliced.empty else _empty
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_engine.py -k "slice_prices" -v --no-cov
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add clenow/backtest/engine.py tests/backtest/test_engine.py
git commit -m "perf: add _slice_prices helper for in-memory price slicing"
```

---

## Task 2: Add `bear_regime_cache` to `apply_filters`

**Files:**
- Modify: `clenow/portfolio/selector.py` — `apply_filters` signature + body
- Test: `tests/portfolio/test_selector.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/portfolio/test_selector.py — add new test class or function

from datetime import date
from unittest.mock import MagicMock
import pandas as pd
from clenow.portfolio.selector import apply_filters
from clenow.config import Config


def _make_prices_for_selector(tickers, n_days=150):
    """Build a minimal all_prices DataFrame for filter tests."""
    rows = []
    base = date(2024, 1, 1)
    from datetime import timedelta
    for i in range(n_days):
        d = base + timedelta(days=i)
        for t in tickers:
            rows.append({
                "date": pd.Timestamp(d),
                "ticker": t,
                "raw_close": 50.0,
                "raw_open": 49.0,
                "raw_high": 51.0,
                "raw_low": 48.0,
                "adj_close": 50.0,
                "volume": 5_000_000.0,
            })
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_apply_filters_uses_bear_regime_cache_skips_db():
    mock_provider = MagicMock()
    config = Config()
    as_of = date(2024, 6, 1)
    prices = _make_prices_for_selector(["AAPL", "MSFT"])

    bear_regime_cache = {as_of: True}  # bear market

    result = apply_filters(
        ranked_tickers=["AAPL", "MSFT"],
        all_prices=prices,
        data_provider=mock_provider,
        as_of=as_of,
        config=config,
        current_positions={},
        bear_regime_cache=bear_regime_cache,
    )

    # Bear market, no existing positions → both filtered out
    assert result == []
    # DB should NOT be called for regime
    mock_provider.get_index_prices.assert_not_called()


def test_apply_filters_bull_regime_cache_allows_new_entries():
    mock_provider = MagicMock()
    config = Config()
    as_of = date(2024, 6, 1)
    prices = _make_prices_for_selector(["AAPL"])

    bear_regime_cache = {as_of: False}  # bull market

    result = apply_filters(
        ranked_tickers=["AAPL"],
        all_prices=prices,
        data_provider=mock_provider,
        as_of=as_of,
        config=config,
        current_positions={},
        bear_regime_cache=bear_regime_cache,
    )

    assert "AAPL" in result
    mock_provider.get_index_prices.assert_not_called()
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/portfolio/test_selector.py -k "bear_regime_cache" -v --no-cov
```

Expected: `TypeError: apply_filters() got an unexpected keyword argument 'bear_regime_cache'`

- [ ] **Step 3: Add `bear_regime_cache` param to `apply_filters`**

In `clenow/portfolio/selector.py`, change `apply_filters` signature and body:

```python
def apply_filters(
    ranked_tickers: list[str],
    all_prices: pd.DataFrame,
    data_provider,
    as_of: date,
    config: Config,
    current_positions: dict[str, Position] | None = None,
    profile: MarketProfile | None = None,
    bear_regime_cache: dict[date, bool] | None = None,
) -> list[str]:
```

Then replace the `_is_bear_regime` call (around line 131) with:

```python
    existing = set(current_positions.keys()) if current_positions else set()

    if bear_regime_cache is not None:
        bear = bear_regime_cache.get(as_of, False)
    else:
        bear = _is_bear_regime(
            data_provider,
            as_of,
            profile.regime_index_id,
            profile.regime_sma_window,
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/portfolio/test_selector.py -v --no-cov
```

Expected: all pass (new tests + existing)

- [ ] **Step 5: Commit**

```bash
git add clenow/portfolio/selector.py tests/portfolio/test_selector.py
git commit -m "perf: add bear_regime_cache to apply_filters to skip DB per iteration"
```

---

## Task 3: Add `_precompute_bear_regime` helper

**Files:**
- Modify: `clenow/backtest/engine.py`
- Test: `tests/backtest/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/backtest/test_engine.py

from clenow.backtest.engine import _precompute_bear_regime


def test_precompute_bear_regime_marks_bear_dates():
    # Create a price series where the last value is below 200-day SMA
    import numpy as np
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Declining prices → last value below SMA
    prices = pd.Series(np.linspace(200, 50, n), index=dates)

    cache = _precompute_bear_regime(prices, sma_window=200)

    # Last date should be bear (price=50 < 200-SMA ≈ 125)
    last_date = dates[-1].date()
    assert last_date in cache
    assert cache[last_date] is True


def test_precompute_bear_regime_marks_bull_dates():
    import numpy as np
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Rising prices → last value above SMA
    prices = pd.Series(np.linspace(50, 200, n), index=dates)

    cache = _precompute_bear_regime(prices, sma_window=200)

    last_date = dates[-1].date()
    assert last_date in cache
    assert cache[last_date] is False


def test_precompute_bear_regime_skips_dates_without_enough_history():
    import numpy as np
    n = 100  # fewer than sma_window=200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = pd.Series(np.ones(n) * 100.0, index=dates)

    cache = _precompute_bear_regime(prices, sma_window=200)

    # Dates before SMA window is complete should be absent
    assert all(v is not None for v in cache.values())  # no None values
    # With only 100 days, no date can have a 200-day SMA → cache is empty
    assert len(cache) == 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/backtest/test_engine.py -k "precompute_bear_regime" -v --no-cov
```

Expected: `ImportError` — `_precompute_bear_regime` not defined yet

- [ ] **Step 3: Implement `_precompute_bear_regime` in `engine.py`**

Add after `_slice_prices` (before `_select_one_per_sector`):

```python
def _precompute_bear_regime(
    index_prices: pd.Series,
    sma_window: int,
) -> dict[date, bool]:
    """Pre-compute bear regime flag for every date in index_prices.

    Returns {date: True_if_bear} only for dates where SMA can be computed
    (i.e., at least sma_window prior data points exist).
    """
    from clenow.signals.regime import is_bear_regime as _is_bear

    if index_prices.empty:
        return {}

    sma = index_prices.rolling(sma_window).mean()
    result: dict[date, bool] = {}
    for ts, price in index_prices.items():
        sma_val = sma.get(ts)
        if sma_val is None or pd.isna(sma_val):
            continue
        d = ts.date() if isinstance(ts, pd.Timestamp) else ts
        result[d] = bool(price < sma_val)
    return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/backtest/test_engine.py -k "precompute_bear_regime" -v --no-cov
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add clenow/backtest/engine.py tests/backtest/test_engine.py
git commit -m "perf: add _precompute_bear_regime for one-time regime cache computation"
```

---

## Task 4: Add `preloaded_prices` + `bear_regime_cache` to `compute_target_portfolio`

**Files:**
- Modify: `clenow/backtest/engine.py` — `compute_target_portfolio`
- Test: `tests/backtest/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/backtest/test_engine.py

from unittest.mock import MagicMock, patch
from datetime import date, timedelta
from clenow.backtest.engine import compute_target_portfolio
from clenow.config import Config
from clenow.types import Position


def _make_full_preloaded(tickers, start, end):
    """Build a realistic preloaded prices DataFrame."""
    from datetime import timedelta
    rows = []
    d = start
    while d <= end:
        for t in tickers:
            rows.append({
                "date": pd.Timestamp(d),
                "ticker": t,
                "raw_close": 100.0,
                "raw_open": 99.0,
                "raw_high": 102.0,
                "raw_low": 98.0,
                "adj_close": 100.0,
                "volume": 10_000_000.0,
                "split_ratio": 1.0,
                "dividend": 0.0,
            })
        d += timedelta(days=1)
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_compute_target_portfolio_uses_preloaded_skips_load_prices():
    mock_provider = MagicMock()
    mock_provider.get_universe.return_value = ["AAPL", "MSFT"]

    as_of = date(2024, 6, 1)
    price_start = as_of - timedelta(days=500)
    preloaded = _make_full_preloaded(["AAPL", "MSFT"], price_start, as_of)

    compute_target_portfolio(
        as_of=as_of,
        current_positions={},
        current_cash=100_000.0,
        config=Config(),
        data_provider=mock_provider,
        preloaded_prices=preloaded,
        bear_regime_cache={as_of: False},
    )

    # load_prices should NOT be called when preloaded_prices is provided
    mock_provider.load_prices.assert_not_called()


def test_compute_target_portfolio_falls_back_to_load_prices_when_no_preload():
    mock_provider = MagicMock()
    mock_provider.get_universe.return_value = []

    compute_target_portfolio(
        as_of=date(2024, 6, 1),
        current_positions={},
        current_cash=100_000.0,
        config=Config(),
        data_provider=mock_provider,
        preloaded_prices=None,  # explicit None → old code path
    )
    # With empty universe, load_prices should also not be called
    mock_provider.load_prices.assert_not_called()
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/backtest/test_engine.py -k "preloaded_skips_load" -v --no-cov
```

Expected: `TypeError: compute_target_portfolio() got unexpected keyword argument 'preloaded_prices'`

- [ ] **Step 3: Add params to `compute_target_portfolio`**

Change the function signature (line ~141 in `engine.py`):

```python
def compute_target_portfolio(
    as_of: date,
    current_positions: dict[str, Position],
    current_cash: float,
    config: Config,
    data_provider: DataProvider,
    sector_mapping: dict[str, str] | None = None,
    select_one_per_sector: bool = False,
    profile: "MarketProfile | None" = None,
    preloaded_prices: pd.DataFrame | None = None,
    bear_regime_cache: dict[date, bool] | None = None,
) -> TargetPortfolio:
```

Replace the price loading block (line ~178-179):

```python
    # Step 2: Compute score and ATR for each ticker
    price_start = as_of - _PRICE_LOOKBACK_CALENDAR
    if preloaded_prices is not None:
        all_prices = _slice_prices(preloaded_prices, universe, price_start, as_of)
    else:
        all_prices = data_provider.load_prices(universe, price_start, as_of)
```

Also thread `bear_regime_cache` into `apply_filters` call (line ~242):

```python
    filtered = apply_filters(
        ranked, all_prices, data_provider, as_of, config,
        current_positions, profile=profile,
        bear_regime_cache=bear_regime_cache,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/backtest/test_engine.py -k "preloaded" -v --no-cov
```

Expected: all pass

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -q -k "not integration and not parquet" --no-cov
```

Expected: 354+ passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add clenow/backtest/engine.py tests/backtest/test_engine.py
git commit -m "perf: add preloaded_prices + bear_regime_cache to compute_target_portfolio"
```

---

## Task 5: Preload in `run_backtest` + pass to each iteration

**Files:**
- Modify: `clenow/backtest/engine.py` — `run_backtest` preamble + main loop
- No new test file (integration test covers this; unit test mocks would defeat the purpose)

- [ ] **Step 1: Add universe superset collection + preload before the main loop**

In `run_backtest()` (around line 405, after `total = len(rebalance_pairs)`):

```python
    total = len(rebalance_pairs)

    # ── Performance preload ────────────────────────────────────────────────
    # Collect full universe superset across all rebalance dates.
    # get_universe() hits the in-memory PIT dict after first build — cheap.
    logger.info("Collecting universe superset for preload...")
    all_universe_tickers: set[str] = set()
    for signal_date, _ in rebalance_pairs:
        all_universe_tickers.update(
            data_provider.get_universe(signal_date, index_id=profile.universe_index_id)
        )

    # Preload all price data for the full backtest period in one shot.
    # Each iteration will slice this in-memory DataFrame instead of querying DuckDB.
    global_price_start = start - _PRICE_LOOKBACK_CALENDAR - timedelta(days=60)
    logger.info(
        "Preloading prices for %d tickers from %s to %s...",
        len(all_universe_tickers), global_price_start, end,
    )
    preloaded_prices = data_provider.load_prices(
        sorted(all_universe_tickers), global_price_start, end
    )
    logger.info("Preload complete: %d rows", len(preloaded_prices))

    # Precompute regime filter for all dates in the backtest.
    regime_index_id = profile.regime_index_id
    sma_window = profile.regime_sma_window
    regime_start = start - timedelta(days=sma_window * 2 + 60)
    logger.info("Precomputing regime filter...")
    _raw_index = data_provider.get_index_prices(regime_index_id, regime_start, end)
    if not _raw_index.empty:
        close_col = (
            "raw_close" if "raw_close" in _raw_index.columns
            else "close" if "close" in _raw_index.columns
            else _raw_index.select_dtypes("number").columns[0]
        )
        bear_regime_cache = _precompute_bear_regime(
            _raw_index[close_col].sort_index(), sma_window
        )
    else:
        bear_regime_cache = {}
    logger.info("Regime cache: %d dates", len(bear_regime_cache))
    # ── End preload ────────────────────────────────────────────────────────
```

- [ ] **Step 2: Pass preloaded data into `compute_target_portfolio`**

Change the `compute_target_portfolio` call (around line 417):

```python
        target = compute_target_portfolio(
            as_of=signal_date,
            current_positions=current_positions,
            current_cash=current_cash,
            config=config,
            data_provider=data_provider,
            sector_mapping=sector_mapping,
            select_one_per_sector=select_one_per_sector,
            profile=profile,
            preloaded_prices=preloaded_prices,
            bear_regime_cache=bear_regime_cache,
        )
```

- [ ] **Step 3: Serve exec_prices from preloaded (removes per-iteration load)**

Replace lines 436-443 (exec_prices load):

```python
        if valuation_tickers:
            exec_lookback = execution_date - timedelta(days=15)
            exec_prices = _slice_prices(
                preloaded_prices, valuation_tickers, exec_lookback, execution_date
            )
        else:
            exec_prices = _empty_prices_frame()
```

- [ ] **Step 4: Serve ADV history from preloaded**

Replace lines 447-448 (adv_hist load):

```python
            adv_lookback_start = execution_date - timedelta(days=30)
            adv_hist = _slice_prices(
                preloaded_prices, order_tickers, adv_lookback_start, execution_date
            )
```

- [ ] **Step 5: Serve delistings check from preloaded**

Change `_detect_delistings` call (line ~411) and pass `preloaded_prices`:

```python
        _detect_delistings(
            tracker, data_provider, signal_date,
            profile=profile, preloaded_prices=preloaded_prices,
        )
```

Then update `_detect_delistings` signature (around line 680) to accept and use preloaded:

```python
def _detect_delistings(
    tracker: PositionTracker,
    data_provider: DataProvider,
    as_of: date,
    profile: "MarketProfile | None" = None,
    preloaded_prices: pd.DataFrame | None = None,
) -> None:
```

Inside `_detect_delistings`, replace the `load_prices` calls (lines 708, 717):

```python
    # Load price data for the current date — use preloaded if available
    if preloaded_prices is not None:
        prices = _slice_prices(preloaded_prices, tickers, as_of - timedelta(days=30), as_of)
    else:
        prices = data_provider.load_prices(tickers, as_of, as_of)
```

And remove the fallback inner `load_prices` call for missing tickers (the slice already covers 30-day lookback):

```python
        if ticker_data is None or ticker_data.empty:
            if preloaded_prices is None:
                # Fallback: look back 30 days when no preloaded available
                lookback_start = as_of - timedelta(days=30)
                hist_prices = data_provider.load_prices([ticker], lookback_start, as_of)
                ticker_data = _get_ticker_series(hist_prices, ticker)
            # If still None with preloaded, the ticker has no data in our range
```

- [ ] **Step 6: Serve corporate actions from preloaded**

In `_apply_corporate_actions` (line ~596), the `data_provider.load_prices` call:

Change the function signature:

```python
def _apply_corporate_actions(
    tracker: PositionTracker,
    data_provider: DataProvider,
    signal_date: date,
    execution_date: date,
    preloaded_prices: pd.DataFrame | None = None,
) -> None:
```

Replace the `load_prices` call (line ~596):

```python
    if preloaded_prices is not None:
        prices = _slice_prices(preloaded_prices, tickers, signal_date, execution_date)
    else:
        prices = data_provider.load_prices(tickers, signal_date, execution_date)
```

Update call site in `run_backtest` main loop (line ~539):

```python
        _apply_corporate_actions(
            tracker, data_provider, signal_date, execution_date,
            preloaded_prices=preloaded_prices,
        )
```

- [ ] **Step 7: Run full test suite**

```bash
uv run pytest tests/ -q -k "not integration and not parquet" --no-cov
```

Expected: 354+ passed

- [ ] **Step 8: Commit**

```bash
git add clenow/backtest/engine.py
git commit -m "perf: preload all price data in run_backtest, serve iterations from memory"
```

---

## Task 6: Verify performance improvement

- [ ] **Step 1: Time the backtest**

```bash
time python run_backtest.py 2>&1 | tail -5
```

Note the wall-clock time.

- [ ] **Step 2: Verify no oscillation**

```bash
python run_backtest.py 2>&1 | grep "pos=" | awk -F'pos=' '{print $2}' | sort -n | uniq -c | head -10
```

Expected: pos values stable (not alternating 20/0/20/0).

- [ ] **Step 3: Verify sector backtest works**

```bash
python run_sector_backtest.py 2>&1 | grep "pos=" | awk -F'pos=' '{print $2}' | sort -n | uniq -c | head -10
```

Expected: pos values ≤ 11 (one per GICS sector).

- [ ] **Step 4: Final commit if all good**

```bash
git add -A
git commit -m "perf: backtest optimization complete — preload + regime cache"
```
