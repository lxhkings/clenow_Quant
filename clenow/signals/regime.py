"""Market regime detection based on index SMA.

A long-only momentum system should not fight bear markets. This module
determines whether the broad market is in a bull regime (index above
its simple moving average).
"""

import pandas as pd


def is_bull_regime(
    index_close: pd.Series,
    sma_period: int = 200,
) -> bool:
    """Check if the market is in a bull regime.

    Returns True when the last close is above the SMA(sma_period)
    of the index. Defaults to True (bull) when there is insufficient
    data to compute the SMA — a conservative default for a long-only
    system that should not be stuck in cash due to data issues.

    Parameters
    ----------
    index_close : pd.Series
        Daily close prices for a broad market index (e.g. S&P 500).
    sma_period : int
        Simple moving average lookback (default 200).

    Returns
    -------
    bool
        True if bull regime (last close > SMA), False otherwise.
    """
    # Drop NaN values to get a clean series
    clean = index_close.dropna()

    if len(clean) < sma_period:
        # Insufficient data: default to bull (conservative for long-only)
        return True

    # Compute SMA over the last sma_period values
    sma = clean.iloc[-sma_period:].mean()
    last_close = clean.iloc[-1]

    return bool(last_close > sma)
