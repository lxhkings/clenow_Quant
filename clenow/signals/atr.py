"""Average True Range (ATR) using Wilder's Running Moving Average.

ATR measures actual trading price volatility and uses raw (unadjusted)
OHLC data, because it reflects the real price ranges a trader would
experience.
"""

import numpy as np
import pandas as pd


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
    # Need at least period + 1 bars: `period` for the initial SMA plus
    # one prior close for the first TR calculation.
    if len(high) < period + 1:
        return 0.0

    # Align series by index and drop rows with any NaN
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    df = df.dropna()

    if len(df) < period + 1:
        return 0.0

    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # True Range: max(H-L, |H-prevC|, |L-prevC|)
    tr = np.empty(len(df) - 1, dtype=float)
    for i in range(1, len(df)):
        hl = h[i] - l[i]
        hpc = abs(h[i] - c[i - 1])
        lpc = abs(l[i] - c[i - 1])
        tr[i - 1] = max(hl, hpc, lpc)

    # Need at least `period` TR values for the initial SMA
    if len(tr) < period:
        return 0.0

    # Wilder's RMA:
    #   First value = SMA of first `period` TR values
    #   Subsequent:  RMA[t] = (RMA[t-1] * (period - 1) + TR[t]) / period
    rma = np.empty(len(tr), dtype=float)
    rma[period - 1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        rma[i] = (rma[i - 1] * (period - 1) + tr[i]) / period

    atr_value = float(rma[-1])

    # Defensive: illiquid or stopped stocks may have near-zero ATR
    if atr_value < 0.01:
        return 0.0

    return atr_value
