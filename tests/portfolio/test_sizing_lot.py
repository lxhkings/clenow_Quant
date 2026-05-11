from clenow.portfolio.sizing import compute_shares_with_lot


def test_lot_size_1_unchanged():
    """US: lot_size=1 returns raw integer floor."""
    assert compute_shares_with_lot(raw_shares=137.8, lot_size=1) == 137


def test_lot_size_100_floors_to_hundred():
    """CN/HK: lot_size=100 floors to nearest 100 multiple."""
    assert compute_shares_with_lot(raw_shares=137.8, lot_size=100) == 100
    assert compute_shares_with_lot(raw_shares=250.0, lot_size=100) == 200
    assert compute_shares_with_lot(raw_shares=199.99, lot_size=100) == 100
    assert compute_shares_with_lot(raw_shares=300.0, lot_size=100) == 300


def test_lot_size_below_one_lot_returns_zero():
    """Sub-lot size: zero (cannot buy partial lot)."""
    assert compute_shares_with_lot(raw_shares=99.0, lot_size=100) == 0
    assert compute_shares_with_lot(raw_shares=0.5, lot_size=1) == 0
