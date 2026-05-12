# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest tests/ -q                              # All tests
pytest tests/ -x -q                           # Stop on first failure
pytest tests/signals/test_clenow_score.py -v  # Single module
pytest tests/signals/test_clenow_score.py::test_fn_name -v  # Single test
pytest tests/ -k "cn" -v                      # By keyword
```

Run backtest:
```bash
uv run python run_backtest.py --market us
python run_backtest.py                                         # US, 2022-2026, $100k
python run_backtest.py --market cn --start 2023-01-01          # A股
python run_backtest.py --market hk --risk-factor 0.01          # 港股, higher sizing
python run_sector_backtest.py                                  # Sector rotation variant (1 per sector)
```

Live trading CLI (generate order list):
```bash
python -m clenow.live --as-of 2026-05-08 --equity 100000 --market us
python -m clenow.live --as-of 2026-05-08 --market cn --output orders.csv
```

Validate database:
```bash
python scripts/validate_db.py
```

## Architecture

### Core Design Principle

**Backtest engine = Live engine.** `compute_target_portfolio` in `clenow/backtest/engine.py` is a pure, deterministic function shared by both backtest and live CLI. Same code path, no branching. This ensures backtested decisions match live trading decisions.

### Strategy: Clenow Smooth Momentum

From "Stocks on the Move" by Andreas Clenow:

1. **Score calculation** (`signals/clenow_score.py`):
   - 90-day log-linear regression on adjusted close
   - Formula: `score = (exp(slope × 252) - 1) × R²`
   - R² penalizes choppy trends; gaps >15% zero the score

2. **Ranking** (`portfolio/ranker.py`):
   - Sort by score descending, take top 20%

3. **Position sizing** (`portfolio/sizing.py`):
   - ATR-based: `shares = floor((equity × risk_factor) / ATR)`
   - Default `risk_factor=0.001` (conservative), use `0.005` for meaningful positions
   - Single-stock cap: 5% of equity

4. **Double exit rule** (CRITICAL, engine.py):
   - Every rebalance: FIRST check existing positions for 100-day SMA break
   - THEN apply new top 20% list
   - Stock in top 20% but below 100 SMA → forced sell
   - Regime filter (index < 200 SMA): blocks NEW entries only

### Module Responsibilities

```
clenow/
├── signals/          # Clenow score, ATR, regime detection
├── portfolio/        # Ranking, filtering, sizing, position tracking
├── backtest/engine.py   # compute_target_portfolio + run_backtest
├── backtest/executor.py # SimulatedExecutor (fills at raw_open)
├── data/provider.py     # DataProvider Protocol (load_prices, get_universe)
├── data/synology.py     # MariaDB adapter (production)
├── live/cli.py          # Daily decision workflow CLI
└── report/              # Metrics, equity curve, trade log
```

### Data Flow

```
get_universe(as_of) → compute_clenow_score + ATR for all →
rank_by_score(top 20%) → apply_filters(regime/SMA/price/ADV) →
compute_target_positions(ATR sizing) → TargetPortfolio
```

### Key Types

- `TargetPortfolio`: frozen, positions dict `{ticker: shares}`
- `Position`: mutable, tracks entry_price, entry_date, atr_at_entry
- `DataProvider`: Protocol — any DB/broker implementing `load_prices`, `get_universe`
- `Config`: frozen dataclass with strategy parameters; `market` field drives profile lookup
- `MarketProfile`: frozen registry of per-market constants (lot size, cost model, calendar, thresholds)

### Performance Pattern

Functions that process multiple tickers should accept a pre-loaded `all_prices` DataFrame (MultiIndex on date/ticker) rather than querying per ticker. See `apply_filters` in selector.py and `_check_sma_break` in engine.py for the pattern.

### Multi-Market Support

Three markets supported via `--market` flag:
- `us` (default): SP500 universe, NYSE calendar, USD costs
- `cn`: CSI800 universe, XSHG calendar, CNY costs, 100-share lots, ST filter, 涨跌停 detection, suspension/delisting distinction
- `hk`: HSI universe, XHKG calendar, HKD costs, 100-share lots

Each market runs independently — no cross-market portfolios or FX conversion. Market-specific constants live in `clenow/markets/profiles.py:PROFILES`. To add a fourth market: add an entry to PROFILES with appropriate `CostModel` impl (extend `clenow/markets/costs.py`).

Data prerequisites per market (in Synology MariaDB):
- prices table with ticker suffix (`.SH`/`.SZ`/`.HK`) or bare (US)
- index_constituents rows with matching `index_id` (SP500/CSI800/HSI)
- index_prices rows with matching `index_id`
- stocks table with `name` column (used for ST detection on CN)

### Testing

- Hand-calculated fixtures in `tests/signals/` for score/ATR validation
- Integration tests in `tests/backtest/test_lifecycle*.py` run full rebalance cycles (US, CN, HK)
- Tests use `@pytest.mark.integration` for DB-dependent tests
