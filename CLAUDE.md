# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run tests:
```bash
pytest tests/ -q                              # All tests
pytest tests/signals/test_clenow_score.py -v  # Single module
pytest tests/ -x -q                           # Stop on first failure
```

Run backtest:
```bash
python run_backtest.py                        # Default: 2022-2026, $100k, risk_factor=0.005
```

Install dependencies:
```bash
pip install -e ".[dev]"                       # Install with dev dependencies
```

## Architecture

### Core Design Principle

**Backtest engine = Live engine.** `compute_target_portfolio` (engine.py:84-194) is a pure, deterministic function shared by both backtest and live CLI. Same code path, no branching. This ensures backtested decisions match live trading decisions.

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

4. **Double exit rule** (CRITICAL, engine.py:168-177):
   - Every rebalance: FIRST check existing positions for 100-day SMA break
   - THEN apply new top 20% list
   - Stock in top 20% but below 100 SMA → forced sell
   - Regime filter (SP500 < 200 SMA): blocks NEW entries only

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

### Performance Pattern

Functions that process multiple tickers should accept a pre-loaded `all_prices` DataFrame (MultiIndex on date/ticker) rather than querying per ticker. See `apply_filters` in selector.py and `_check_sma_break` in engine.py for the pattern.

### Testing

- 293 tests, 76% coverage
- Hand-calculated fixtures in `tests/signals/` for score/ATR validation
- Integration tests in `tests/backtest/test_lifecycle.py` run full rebalance cycles