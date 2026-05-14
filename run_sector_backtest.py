"""Sector rotation backtest — thin wrapper over run_backtest.py with sector defaults."""
import sys
from run_backtest import main as run_main


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
    start: date = date(2010, 1, 1),
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
        rebalance_freq="weekly",
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
    main()