import pandas as pd
from datetime import date
from clenow.backtest.engine import _classify_inactive
from clenow.markets import get_profile


def _make_volume_series(zero_runs: int, after_zeros: int = 5) -> pd.Series:
    """Build a Series of len `zero_runs + after_zeros` with `zero_runs` trailing zeros."""
    dates = pd.date_range("2024-01-01", periods=zero_runs + after_zeros)
    nonzero = [10_000] * after_zeros
    zeros = [0] * zero_runs
    return pd.Series(nonzero + zeros, index=dates)


def test_classify_short_zero_run_is_suspended():
    profile = get_profile("CN")  # suspension=20, delisting=60
    vol = _make_volume_series(zero_runs=5)
    assert _classify_inactive(vol, profile) == "suspended"


def test_classify_long_zero_run_is_delisted():
    profile = get_profile("CN")
    vol = _make_volume_series(zero_runs=70)
    assert _classify_inactive(vol, profile) == "delisted"


def test_classify_no_zeros_is_active():
    profile = get_profile("CN")
    vol = _make_volume_series(zero_runs=0)
    assert _classify_inactive(vol, profile) == "active"


def test_classify_us_profile_suspension_zero_means_immediate_delisting():
    """US has suspension_threshold_days=0, delisting=10: any 11-day zero run = delisted."""
    profile = get_profile("US")
    vol = _make_volume_series(zero_runs=15)
    assert _classify_inactive(vol, profile) == "delisted"


def test_classify_us_short_zero_run_suspended():
    """US suspension=0: 5-day zero is below delisting threshold=10, stays suspended."""
    profile = get_profile("US")
    vol = _make_volume_series(zero_runs=5)
    # zero_runs(5) > suspension_threshold(0) but <= delisting(10) -> suspended
    assert _classify_inactive(vol, profile) == "suspended"


def test_classify_empty_series_is_delisted():
    profile = get_profile("CN")
    vol = pd.Series(dtype=float)
    assert _classify_inactive(vol, profile) == "delisted"


def test_classify_exact_delisting_threshold_is_suspended():
    """Exactly at delisting_threshold_days -> suspended (not delisted)."""
    profile = get_profile("CN")  # delisting_threshold_days=60
    vol = _make_volume_series(zero_runs=60)
    assert _classify_inactive(vol, profile) == "suspended"


def test_classify_one_over_delisting_threshold_is_delisted():
    """One over delisting_threshold_days -> delisted."""
    profile = get_profile("CN")  # delisting_threshold_days=60
    vol = _make_volume_series(zero_runs=61)
    assert _classify_inactive(vol, profile) == "delisted"
