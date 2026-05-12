# Numpy Hot-Path Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate GIL-holding Python loops from the score/ATR pre-compute hot path, enabling ThreadPoolExecutor to achieve real parallelism across all 12 M4 Pro cores.

**Architecture:** Two changes stack together. (1) Vectorize the True Range loop in `compute_atr` using numpy, removing per-row Python overhead. (2) In `run_backtest`, pre-extract per-ticker numpy arrays *before* thread dispatch, then rewrite `_compute_ticker_metrics` to use `np.searchsorted` for window slicing and pass numpy arrays directly to `compute_atr` and `compute_clenow_score` — eliminating pandas `.xs()`, `.loc[]`, `.dropna()`, and column-access overhead from the 451K-iteration hot loop.

**Tech Stack:** numpy (vectorized ops + searchsorted), scipy.stats.linregress (already GIL-releasing), Python's ThreadPoolExecutor.

---

## File Map

| File | Change |
|------|--------|
| `clenow/signals/atr.py` | Vectorize True Range loop; accept numpy via `np.asarray`; Wilder RMA stays Python loop (recursive) but over scalars only |
| `clenow/backtest/engine.py` | Add `_compute_ticker_metrics_np` args structure; rewrite pre-extract to build numpy dict before thread dispatch; rewrite `_compute_ticker_metrics` to use numpy slicing |
| `tests/signals/test_atr.py` | No new tests needed — existing hand-calculated tests cover the refactored logic |
| `tests/backtest/test_engine.py` | Verify `load_prices.call_count == 1` still holds after refactor |

---

## Task 1: Vectorize `compute_atr` True Range loop

**Files:**
- Modify: `clenow/signals/atr.py:54-73`
- Test: `tests/signals/test_atr.py` (existing, no new tests)

- [ ] **Step 1: Run existing ATR tests to establish baseline**

```bash
uv run pytest tests/signals/test_atr.py -v --no-cov 2>&1 | tail -20
```

Expected: all pass. Note the exact count.

- [ ] **Step 2: Replace Python TR loop and Wilder RMA loop in `compute_atr`**

Open `clenow/signals/atr.py`. Replace lines 44–81 with this implementation:

```python
def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> float:
    """Compute the Average True Range using Wilder's RMA.

    Parameters
    ----------
    high : pd.Series
        Daily high prices (raw/unadjusted).
    low : pd.Series
        Daily low prices (raw/unadjusted).
    close : pd.Series
        Daily close prices (raw/unadjusted).
    period : int
        RMA lookback period (default 20).

    Returns
    -------
    float
        The ATR value. Returns 0.0 when:
        - insufficient data (< period + 1 bars)
        - resulting ATR < 0.01 (defensive against illiquid/stopped stocks)
    """
    if len(high) < period + 1:
        return 0.0

    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    # Drop rows where any of H/L/C is NaN
    valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
    h, l, c = h[valid], l[valid], c[valid]

    if len(h) < period + 1:
        return 0.0

    # Vectorized True Range: max(H-L, |H-prevC|, |L-prevC|)
    hl  = h[1:] - l[1:]
    hpc = np.abs(h[1:] - c[:-1])
    lpc = np.abs(l[1:] - c[:-1])
    tr  = np.maximum(hl, np.maximum(hpc, lpc))

    if len(tr) < period:
        return 0.0

    # Wilder's RMA: SMA init then recursive
    # RMA[t] = (RMA[t-1] * (period-1) + TR[t]) / period
    alpha_k = (period - 1) / period
    rma = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        rma = rma * alpha_k + tr[i] / period

    if rma < 0.01:
        return 0.0

    return rma
```

- [ ] **Step 3: Run ATR tests to verify behaviour unchanged**

```bash
uv run pytest tests/signals/test_atr.py -v --no-cov 2>&1 | tail -20
```

Expected: same count as baseline, all pass.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -q --no-cov -k "not integration and not parquet" 2>&1 | tail -5
```

Expected: 364 passed.

- [ ] **Step 5: Commit**

```bash
git add clenow/signals/atr.py
git commit -m "perf: vectorize compute_atr True Range loop using numpy"
```

---

## Task 2: Pre-extract numpy arrays and rewrite `_compute_ticker_metrics`

**Files:**
- Modify: `clenow/backtest/engine.py` — `_compute_ticker_metrics` (lines 124–192) + preload section in `run_backtest` (lines 598–633)

### Part A: Rewrite `_compute_ticker_metrics` to accept numpy dict

- [ ] **Step 1: Replace `_compute_ticker_metrics` in `engine.py`**

Find `_compute_ticker_metrics` (line 124). Replace the entire function with:

```python
def _compute_ticker_metrics(
    args: tuple,
) -> tuple[str, dict[date, float], dict[date, float], dict[date, float]]:
    """Compute Clenow score, ATR, and current price for one ticker across all signal dates.

    Accepts pre-extracted numpy arrays — no pandas in the hot loop.
    Designed for parallel dispatch via ThreadPoolExecutor.

    Returns:
        (ticker, scores, atrs, current_prices) — all keyed by signal date.
    """
    (
        ticker,
        ticker_arrays,          # dict with numpy arrays (dates, adj_close, raw_close, raw_high, raw_low)
        signal_dates,
        score_window,
        atr_period,
        annualization_days,
        gap_threshold,
        lookback_calendar_days,
    ) = args

    dates_np  = ticker_arrays["dates"]       # sorted numpy datetime64[D]
    adj_close = ticker_arrays["adj_close"]   # float64, NaN for missing
    raw_close = ticker_arrays["raw_close"]
    raw_high  = ticker_arrays["raw_high"]
    raw_low   = ticker_arrays["raw_low"]

    scores: dict[date, float] = {}
    atrs: dict[date, float] = {}
    current_prices: dict[date, float] = {}

    lookback_td = np.timedelta64(lookback_calendar_days, "D")

    for signal_date in signal_dates:
        end_ts   = np.datetime64(signal_date, "D")
        start_ts = end_ts - lookback_td

        # O(log N) binary search on sorted array — no pandas, no GIL holding
        lo = int(np.searchsorted(dates_np, start_ts, side="left"))
        hi = int(np.searchsorted(dates_np, end_ts,   side="right"))

        if hi - lo < 2:
            scores[signal_date] = 0.0
            atrs[signal_date]   = 0.0
            continue

        adj_w  = adj_close[lo:hi]
        raw_w  = raw_close[lo:hi]
        high_w = raw_high[lo:hi]
        low_w  = raw_low[lo:hi]

        # compute_clenow_score accepts numpy arrays via np.asarray internally
        scores[signal_date] = compute_clenow_score(
            adj_close=pd.Series(adj_w),
            raw_close=pd.Series(raw_w),
            score_window=score_window,
            annualization_days=annualization_days,
            gap_threshold=gap_threshold,
        )

        atrs[signal_date] = compute_atr(
            high=pd.Series(high_w),
            low=pd.Series(low_w),
            close=pd.Series(raw_w),
            period=atr_period,
        )

        if len(raw_w) > 0 and not np.isnan(raw_w[-1]):
            current_prices[signal_date] = float(raw_w[-1])

    return ticker, scores, atrs, current_prices
```

Note: We keep `pd.Series()` wrappers here temporarily — Task 3 removes them once `compute_clenow_score` accepts numpy directly.

- [ ] **Step 2: Run tests to verify function still works**

```bash
uv run pytest tests/backtest/test_engine.py -q --no-cov -k "not integration" 2>&1 | tail -10
```

Expected: all pass.

### Part B: Pre-extract numpy arrays before thread dispatch

- [ ] **Step 3: Replace the arg-building section in `run_backtest`**

Find the section starting at `# Parallel pre-compute` (around line 598). Replace from there through `_args.append(...)` with:

```python
    # Parallel pre-compute Clenow scores + ATRs for all tickers × all signal dates.
    # Pre-extract per-ticker numpy arrays BEFORE thread dispatch to eliminate
    # pandas .xs()/.sort_index() from the parallel critical path.
    signal_dates = [sd for sd, _ in rebalance_pairs]
    _lookback_days = _PRICE_LOOKBACK_CALENDAR.days

    print("Extracting per-ticker numpy arrays...", flush=True)
    _ticker_arrays: dict[str, dict] = {}
    for _ticker in sorted(all_universe_tickers):
        try:
            td = preloaded_prices.xs(_ticker, level="ticker")
        except KeyError:
            continue
        idx = td.index
        # Normalise index to datetime64[D] for searchsorted compatibility
        if hasattr(idx, "to_numpy"):
            dates_np = idx.to_numpy()
            if dates_np.dtype != np.dtype("datetime64[D]"):
                dates_np = dates_np.astype("datetime64[D]")
        else:
            dates_np = np.array([np.datetime64(d, "D") for d in idx])

        _ticker_arrays[_ticker] = {
            "dates":     dates_np,
            "adj_close": td["adj_close"].to_numpy(dtype=float) if "adj_close" in td.columns else np.array([], dtype=float),
            "raw_close": td["raw_close"].to_numpy(dtype=float) if "raw_close" in td.columns else np.array([], dtype=float),
            "raw_high":  td["raw_high"].to_numpy(dtype=float)  if "raw_high"  in td.columns else np.array([], dtype=float),
            "raw_low":   td["raw_low"].to_numpy(dtype=float)   if "raw_low"   in td.columns else np.array([], dtype=float),
        }
    print(f"Extracted {len(_ticker_arrays)} ticker arrays", flush=True)

    _args = [
        (
            _ticker,
            _ticker_arrays[_ticker],
            signal_dates,
            profile.score_window, config.atr_period,
            profile.annualization_days, config.gap_threshold,
            _lookback_days,
        )
        for _ticker in sorted(_ticker_arrays)
    ]

    n_workers = min(os.cpu_count() or 1, len(_args))
    print(
        f"Pre-computing scores/ATRs: {len(_args)} tickers × {len(signal_dates)} dates "
        f"({n_workers} threads)...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=n_workers) as _pool:
        _results = list(_pool.map(_compute_ticker_metrics, _args))
```

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -q --no-cov -k "not integration and not parquet" 2>&1 | tail -5
```

Expected: 364 passed.

- [ ] **Step 5: Commit**

```bash
git add clenow/backtest/engine.py
git commit -m "perf: pre-extract numpy arrays before thread dispatch, rewrite _compute_ticker_metrics"
```

---

## Task 3: Make `compute_clenow_score` and `compute_atr` accept numpy arrays natively

Wrapping numpy arrays in `pd.Series()` in the hot loop is wasteful. Remove those wrappers by making both functions accept numpy input directly.

**Files:**
- Modify: `clenow/signals/clenow_score.py`
- Modify: `clenow/signals/atr.py` (already accepts via `np.asarray` from Task 1)
- Modify: `clenow/backtest/engine.py` — `_compute_ticker_metrics` (remove `pd.Series()` wrappers)

- [ ] **Step 1: Write failing test to confirm numpy input currently fails**

```python
# In tests/signals/test_atr.py — add temporarily to verify current behaviour
import numpy as np
from clenow.signals.atr import compute_atr

def test_compute_atr_accepts_numpy():
    """compute_atr should work with numpy array inputs."""
    n = 25
    h = np.full(n, 102.0)
    l = np.full(n, 98.0)
    c = np.full(n, 100.0)
    result = compute_atr(h, l, c, period=20)
    assert isinstance(result, float)
    assert result > 0.0
```

```bash
uv run pytest tests/signals/test_atr.py::test_compute_atr_accepts_numpy -v --no-cov 2>&1 | tail -5
```

Expected: PASS (since Task 1 already added `np.asarray`).

- [ ] **Step 2: Add numpy test for `compute_clenow_score`**

Add to `tests/signals/test_clenow_score.py`:

```python
import numpy as np
from clenow.signals.clenow_score import compute_clenow_score

def test_compute_clenow_score_accepts_numpy():
    """compute_clenow_score should accept numpy arrays as well as pandas Series."""
    n = 100
    prices = np.linspace(100, 120, n)  # steady uptrend
    result = compute_clenow_score(
        adj_close=prices,
        raw_close=prices,
        score_window=90,
    )
    assert isinstance(result, float)
    assert result > 0.0  # uptrend should score positive
```

```bash
uv run pytest tests/signals/test_clenow_score.py::test_compute_clenow_score_accepts_numpy -v --no-cov 2>&1 | tail -5
```

Expected: FAIL — `TypeError` because `compute_clenow_score` calls `.iloc[-n:]` on numpy input.

- [ ] **Step 3: Make `compute_clenow_score` accept numpy arrays**

In `clenow/signals/clenow_score.py`, replace the function body to use numpy-native operations when input is an ndarray:

```python
def compute_clenow_score(
    adj_close,
    raw_close,
    score_window: int = 90,
    annualization_days: int = 252,
    gap_threshold: float = 0.15,
) -> float:
    """Compute the Clenow Smooth Momentum score for a price series.

    Accepts both pd.Series and np.ndarray inputs.
    """
    n = score_window

    # Normalise to numpy arrays
    adj_np = np.asarray(adj_close, dtype=float)
    raw_np = np.asarray(raw_close, dtype=float)

    if len(adj_np) < n:
        return 0.0

    adj_w = adj_np[-n:]
    raw_w = raw_np[-n:]

    # Missing data: if > 10% NaN in adj_close window, score = 0
    nan_count = int(np.isnan(adj_w).sum())
    if nan_count > n * 0.10:
        return 0.0

    # Gap detection on raw_close
    raw_valid = raw_w[~np.isnan(raw_w)]
    if len(raw_valid) >= 2:
        log_returns = np.log(raw_valid[1:] / raw_valid[:-1])
        if np.any(np.abs(log_returns) > gap_threshold):
            return 0.0

    # Drop NaN from adj_close for regression
    valid_adj = adj_w[~np.isnan(adj_w)]
    if len(valid_adj) < 2:
        return 0.0

    log_prices = np.log(valid_adj)
    x = np.arange(len(log_prices), dtype=float)

    slope, _intercept, r, _p, _se = linregress(x, log_prices)

    r_squared = r ** 2
    annualized_return = np.exp(slope * annualization_days) - 1
    return float(annualized_return * r_squared)
```

- [ ] **Step 4: Run clenow_score tests**

```bash
uv run pytest tests/signals/test_clenow_score.py -v --no-cov 2>&1 | tail -15
```

Expected: all pass including new numpy test.

- [ ] **Step 5: Remove `pd.Series()` wrappers in `_compute_ticker_metrics`**

In `clenow/backtest/engine.py`, update `_compute_ticker_metrics` — remove `pd.Series()` wrappers from the `compute_clenow_score` and `compute_atr` calls:

```python
        scores[signal_date] = compute_clenow_score(
            adj_close=adj_w,          # numpy array directly
            raw_close=raw_w,
            score_window=score_window,
            annualization_days=annualization_days,
            gap_threshold=gap_threshold,
        )

        atrs[signal_date] = compute_atr(
            high=high_w,              # numpy array directly
            low=low_w,
            close=raw_w,
            period=atr_period,
        )
```

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/ -q --no-cov -k "not integration and not parquet" 2>&1 | tail -5
```

Expected: 364+ passed.

- [ ] **Step 7: Commit**

```bash
git add clenow/signals/clenow_score.py clenow/signals/atr.py clenow/backtest/engine.py tests/signals/test_clenow_score.py tests/signals/test_atr.py
git commit -m "perf: numpy-native inputs for compute_clenow_score and compute_atr — remove pd.Series wrappers from hot path"
```

---

## Task 4: End-to-end timing verification

- [ ] **Step 1: Run backtest with timing output**

```bash
uv run python run_backtest.py --market us 2>&1 | grep -E "Collecting|Preloading|Preload complete|Extract|Pre-comput|complete|Report"
```

Expected output structure:
```
Collecting universe superset...
Preloading N tickers YYYY-MM-DD → YYYY-MM-DD...
Preload complete: X,XXX,XXX rows
Extracting per-ticker numpy arrays...
Extracted NNN ticker arrays
Pre-computing scores/ATRs: NNN tickers × NNN dates (12 threads)...
Pre-compute complete. Starting rebalance loop...
[1/832] ...
...
Report: ./output/report.md
```

- [ ] **Step 2: Verify pos no longer oscillates 20/0**

```bash
uv run python run_backtest.py --market us 2>&1 | grep "pos=" | awk -F'pos=' '{print $2}' | sort -n | uniq -c | head -5
```

Expected: no `0` in position counts (or only at the very start before first position).

- [ ] **Step 3: Run sector backtest**

```bash
uv run python run_sector_backtest.py 2>&1 | grep "pos=" | awk -F'pos=' '{print $2}' | sort -n | uniq -c | head -5
```

Expected: max pos count ≤ 11 (one per GICS sector).
