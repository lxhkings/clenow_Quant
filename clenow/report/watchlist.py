"""Daily watchlist report builder.

Reuses ClenowMomentum.score/rank/entry_filters to produce the same
ticker selection as the backtest engine, but skips sizing (cash-dependent)
and joins sector + score breakdown for human-readable Markdown output.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import csv
import io

import pandas as pd

from clenow.backtest.engine import get_suspended_tickers
from clenow.data.utils import get_ticker_series
from clenow.signals.clenow_score import compute_clenow_score_components
from clenow.strategy.clenow_momentum import ClenowMomentum

if TYPE_CHECKING:
    from clenow.config import Config
    from clenow.data.provider import DataProvider
    from clenow.markets.profiles import MarketProfile

_LOOKBACK_CALENDAR = timedelta(days=450)


@dataclass(frozen=True)
class WatchlistRow:
    rank: int
    ticker: str
    sector: str | None
    score: float
    slope: float
    r_squared: float
    annualized_return: float
    price: float
    sma100: float
    dist_pct: float


def build_watchlist(
    as_of: date,
    config: "Config",
    profile: "MarketProfile",
    data_provider: "DataProvider",
    sector_map: dict[str, str],
    universe: list[str] | None = None,
) -> list[WatchlistRow]:
    """Compute the ranked, filtered watchlist for as_of.

    Steps:
      1. Load universe + 450d price history.
      2. Score every ticker via ClenowMomentum.score().
      3. Rank top_pct via ClenowMomentum.rank().
      4. Apply entry filters (regime, 100SMA, price, ADV) with
         current_positions={} — bear regime -> empty filtered list (expected).
      5. Drop suspended tickers (CN market chiefly).
      6. For each survivor, compute score components + price/sma100/dist%.
      7. Sort by score desc, assign rank from 1.

    Args:
        universe: Optional explicit universe list. If None, uses get_universe().
    """
    if universe is None:
        universe = data_provider.get_universe(as_of, index_id=profile.universe_index_id)
    if not universe:
        return []

    start = as_of - _LOOKBACK_CALENDAR
    all_prices = data_provider.load_prices(universe, start, as_of)

    strategy = ClenowMomentum()
    scores: dict[str, float] = {}
    for ticker in universe:
        ticker_data = get_ticker_series(all_prices, ticker)
        if ticker_data is None or ticker_data.empty:
            continue
        scores[ticker] = strategy.score(ticker, ticker_data, profile, config)

    ranked = strategy.rank(scores, config)
    filtered = strategy.entry_filters(
        ranked, all_prices, data_provider, as_of, config, profile,
        current_positions={},
    )

    suspended = get_suspended_tickers(filtered, all_prices, as_of, profile)
    filtered = [t for t in filtered if t not in suspended]

    rows_raw: list[WatchlistRow] = []
    for ticker in filtered:
        ticker_data = get_ticker_series(all_prices, ticker)
        if ticker_data is None or ticker_data.empty:
            continue
        adj = ticker_data["adj_close"]
        raw = ticker_data["raw_close"]
        components = compute_clenow_score_components(
            adj_close=adj, raw_close=raw,
            score_window=profile.score_window,
            annualization_days=profile.annualization_days,
            gap_threshold=config.gap_threshold,
        )
        raw_clean = raw.dropna()
        if raw_clean.empty:
            continue
        price = float(raw_clean.iloc[-1])
        if len(raw_clean) < config.stock_sma:
            continue
        sma100 = float(raw_clean.iloc[-config.stock_sma:].mean())
        dist_pct = (price - sma100) / sma100 * 100 if sma100 else 0.0
        rows_raw.append(WatchlistRow(
            rank=0,  # assigned after sort
            ticker=ticker,
            sector=sector_map.get(ticker),
            score=components.score,
            slope=components.slope,
            r_squared=components.r_squared,
            annualized_return=components.annualized_return,
            price=price,
            sma100=sma100,
            dist_pct=dist_pct,
        ))

    rows_raw.sort(key=lambda r: r.score, reverse=True)
    return [
        WatchlistRow(
            rank=i + 1,
            ticker=r.ticker, sector=r.sector,
            score=r.score, slope=r.slope, r_squared=r.r_squared,
            annualized_return=r.annualized_return,
            price=r.price, sma100=r.sma100, dist_pct=r.dist_pct,
        )
        for i, r in enumerate(rows_raw)
    ]


def _fmt_sector(s: str | None) -> str:
    return s if s else "—"


def render_markdown(
    rows: list[WatchlistRow],
    as_of: date,
    profile: "MarketProfile",
    config: "Config",
    total_universe: int,
) -> str:
    """Render a Markdown report with header, ordered table, and sector summary."""
    lines: list[str] = []
    lines.append(f"# 强势股名单 — {config.market.upper()} — {as_of.isoformat()}")
    lines.append("")
    lines.append("**策略参数**")
    lines.append(
        f"- score_window: {profile.score_window} | "
        f"annualization_days: {profile.annualization_days}"
    )
    lines.append(f"- top_pct: {config.top_pct:.0%}")
    lines.append(
        f"- regime: {profile.regime_index_id} < {profile.regime_sma_window}-SMA"
    )
    lines.append(f"- stock SMA filter: {config.stock_sma}-day")
    lines.append(
        f"- min_price: {profile.min_price} | "
        f"min_adv: {profile.min_adv_amount:,.0f}"
    )
    lines.append(f"- 全宇宙: {total_universe} | 入选: {len(rows)}")
    lines.append("")

    if not rows:
        lines.append("> 本日无入选股（regime 熊或全部被过滤）。")
        lines.append("")
        return "\n".join(lines)

    lines.append("## 顺序表（按 score 降序）")
    lines.append("")
    lines.append("| Rank | Ticker | Sector | Score | Slope | R² | 年化 | Price | 100SMA | 距SMA% |")
    lines.append("|------|--------|--------|-------|-------|-----|------|-------|--------|--------|")
    for r in rows:
        lines.append(
            f"| {r.rank} | {r.ticker} | {_fmt_sector(r.sector)} | "
            f"{r.score:.3f} | {r.slope:.4f} | {r.r_squared:.2f} | "
            f"{r.annualized_return * 100:+.1f}% | "
            f"{r.price:.2f} | {r.sma100:.2f} | {r.dist_pct:+.1f}% |"
        )
    lines.append("")

    by_sector: dict[str, list[WatchlistRow]] = defaultdict(list)
    for r in rows:
        by_sector[_fmt_sector(r.sector)].append(r)
    total = len(rows)
    summary = sorted(
        ((sec, items) for sec, items in by_sector.items()),
        key=lambda kv: -len(kv[1]),
    )

    lines.append("## 行业汇总")
    lines.append("")
    lines.append("| Sector | 入选数 | 平均 Score | 占比 |")
    lines.append("|--------|--------|-----------|------|")
    for sector, items in summary:
        avg = sum(it.score for it in items) / len(items)
        pct = len(items) / total * 100
        lines.append(f"| {sector} | {len(items)} | {avg:.3f} | {pct:.1f}% |")
    lines.append("")

    return "\n".join(lines) + "\n"


def render_csv(rows: list[WatchlistRow]) -> str:
    """Render watchlist as CSV for Excel.

    Columns: rank, ticker, sector, score, slope, r_squared,
             annualized_return_pct, price, sma100, dist_pct
    """
    output = io.StringIO()
    fieldnames = [
        "rank", "ticker", "sector", "score", "slope",
        "r_squared", "annualized_return_pct", "price",
        "sma100", "dist_pct",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "rank": r.rank,
            "ticker": r.ticker,
            "sector": r.sector or "",
            "score": round(r.score, 3),
            "slope": round(r.slope, 4),
            "r_squared": round(r.r_squared, 2),
            "annualized_return_pct": round(r.annualized_return * 100, 1),
            "price": round(r.price, 2),
            "sma100": round(r.sma100, 2),
            "dist_pct": round(r.dist_pct, 1),
        })
    return output.getvalue()