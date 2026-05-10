"""Sector rotation backtest — 1 stock per sector with higher position sizing."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pymysql

from clenow.backtest.engine import BacktestResult, run_backtest
from clenow.config import Config
from clenow.data.synology import DEFAULT_DB_CONFIG, SynologyDataProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_sector_mapping() -> dict[str, str]:
    """Get ticker -> sector mapping from latest index_constituents snapshot with sector data."""
    conn = pymysql.connect(**DEFAULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        query = """
            SELECT ticker, sector
            FROM index_constituents
            WHERE sector IS NOT NULL AND sector != ''
              AND snapshot_date = (
                  SELECT MAX(snapshot_date)
                  FROM index_constituents
                  WHERE sector IS NOT NULL AND sector != ''
              )
        """
        cursor.execute(query)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return dict(rows)


def run_sector_backtest(
    start: date = date(2022, 1, 1),
    end: date = date(2026, 5, 9),
    initial_capital: float = 100_000,
) -> BacktestResult:
    """Run backtest with 1 stock per sector and higher position sizing.

    Modifications:
    - 1 stock per sector instead of top 20%
    - max_position_pct = 0.12 (12% per stock, ~11 sectors)
    - risk_factor = 0.02 (higher sizing)
    """
    # Get sector mapping
    sector_mapping = get_sector_mapping()
    logger.info("Loaded sector mapping for %d tickers", len(sector_mapping))

    # Configure for sector rotation
    config = Config(
        risk_factor=0.02,  # Higher sizing (2% of equity risk per position)
        max_position_pct=0.12,  # 12% cap per stock (11 sectors → ~132% max in theory)
        top_pct=1.0,  # Don't filter by top_pct — we'll use sector selection
    )

    # Run backtest
    provider = SynologyDataProvider()
    result = run_backtest(
        start=start,
        end=end,
        initial_cash=initial_capital,
        config=config,
        data_provider=provider,
        sector_mapping=sector_mapping,
        select_one_per_sector=True,
    )

    return result


if __name__ == "__main__":
    result = run_sector_backtest()
    logger.info("Backtest complete")

    # Print summary
    final_value = result.equity_curve.iloc[-1]["portfolio_value"]
    logger.info("Final portfolio value: $%.2f", final_value)

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    result.equity_curve.to_csv(output_dir / "sector_equity_curve.csv", index=False)
    if result.trades:
        trades_df = pd.DataFrame(result.trades)
        trades_df.to_csv(output_dir / "sector_trades.csv", index=False)

    logger.info("Results saved to output/")