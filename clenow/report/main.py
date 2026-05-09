"""Report generation — markdown summary of backtest results.

Generates a human-readable report with metrics, trade statistics,
and equity curve description.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from clenow.report.metrics import compute_metrics
from clenow.report.trades import export_trade_log


def generate_report(backtest_result, output_dir: str) -> str:
    """Generate a markdown report from a BacktestResult.

    Args:
        backtest_result: A BacktestResult with fields: equity_curve,
            trades, final_positions, final_cash, config
        output_dir: Directory to write the report and trade log

    Returns:
        Path to the generated report file
    """
    os.makedirs(output_dir, exist_ok=True)

    equity_curve = backtest_result.equity_curve
    trades = backtest_result.trades

    # Compute metrics
    metrics = compute_metrics(equity_curve, trades)

    # Export trade log
    trade_log_path = os.path.join(output_dir, "trades.csv")
    export_trade_log(trades, trade_log_path)

    # Export equity curve
    equity_path = os.path.join(output_dir, "equity_curve.csv")
    if not equity_curve.empty:
        equity_curve.to_csv(equity_path, index=False)

    # Build report
    lines = []
    lines.append("# Clenow Smooth Momentum — Backtest Report")
    lines.append("")

    # Summary metrics table
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Sharpe Ratio | {metrics['sharpe']:.4f} |")
    lines.append(f"| Sortino Ratio | {_fmt_ratio(metrics['sortino'])} |")
    lines.append(f"| Calmar Ratio | {_fmt_ratio(metrics['calmar'])} |")
    lines.append(f"| Max Drawdown | {metrics['max_drawdown']:.2%} |")
    if metrics["max_drawdown_start"] is not None:
        lines.append(f"| Max Drawdown Start | {metrics['max_drawdown_start']} |")
    if metrics["max_drawdown_end"] is not None:
        lines.append(f"| Max Drawdown End | {metrics['max_drawdown_end']} |")
    lines.append(f"| CAGR | {metrics['cagr']:.2%} |")
    lines.append(f"| Turnover Rate | {metrics['turnover_rate']:.4f} |")
    lines.append(f"| Win Rate | {metrics['win_rate']:.2%} |")
    lines.append(f"| Avg Holding Period | {metrics['avg_holding_period']:.1f} days |")
    lines.append("")

    # Equity curve description
    lines.append("## Equity Curve")
    lines.append("")
    if not equity_curve.empty:
        initial_value = float(equity_curve["portfolio_value"].iloc[0])
        final_value = float(equity_curve["portfolio_value"].iloc[-1])
        start_date = equity_curve["date"].iloc[0]
        end_date = equity_curve["date"].iloc[-1]
        total_return = (final_value / initial_value - 1) if initial_value > 0 else 0.0
        trading_days = len(equity_curve)
        lines.append(f"- **Period**: {start_date} to {end_date}")
        lines.append(f"- **Trading days**: {trading_days}")
        lines.append(f"- **Initial capital**: ${initial_value:,.2f}")
        lines.append(f"- **Final portfolio value**: ${final_value:,.2f}")
        lines.append(f"- **Total return**: {total_return:.2%}")
        lines.append("")
        lines.append(f"Equity curve data exported to: {equity_path}")
    else:
        lines.append("No equity curve data available.")
    lines.append("")

    # Trade statistics
    lines.append("## Trade Statistics")
    lines.append("")
    if trades:
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losing_trades = total_trades - winning_trades
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        # Best and worst trades
        best_trade = max(trades, key=lambda t: t.get("pnl", 0))
        worst_trade = min(trades, key=lambda t: t.get("pnl", 0))

        lines.append(f"- **Total closed trades**: {total_trades}")
        lines.append(f"- **Winning trades**: {winning_trades}")
        lines.append(f"- **Losing trades**: {losing_trades}")
        lines.append(f"- **Total PnL**: ${total_pnl:,.2f}")
        lines.append(f"- **Average PnL per trade**: ${avg_pnl:,.2f}")
        lines.append(
            f"- **Best trade**: {best_trade.get('ticker', 'N/A')} "
            f"${best_trade.get('pnl', 0):,.2f}"
        )
        lines.append(
            f"- **Worst trade**: {worst_trade.get('ticker', 'N/A')} "
            f"${worst_trade.get('pnl', 0):,.2f}"
        )
        lines.append("")
        lines.append(f"Trade log exported to: {trade_log_path}")
    else:
        lines.append("No closed trades recorded.")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    config = backtest_result.config
    lines.append(f"- **Score window**: {config.score_window}")
    lines.append(f"- **ATR period**: {config.atr_period}")
    lines.append(f"- **Regime SMA**: {config.regime_sma}")
    lines.append(f"- **Stock SMA**: {config.stock_sma}")
    lines.append(f"- **Rebalance frequency**: {config.rebalance_freq}")
    lines.append(f"- **Risk factor**: {config.risk_factor}")
    lines.append("")

    # Write report
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


def _fmt_ratio(value: float) -> str:
    """Format a ratio that might be infinity."""
    if value == float("inf") or value == float("-inf"):
        return "inf"
    return f"{value:.4f}"
