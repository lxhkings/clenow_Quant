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

    # Wilder's RMA: SMA init then recursive scalar loop
    alpha_k = (period - 1) / period
    rma = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        rma = rma * alpha_k + tr[i] / period

    if rma < 0.01:
        return 0.0

    return rma
