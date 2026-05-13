"""Sector rotation backtest — thin wrapper over run_backtest.py with sector defaults."""
import sys
from run_backtest import main as run_main


def main() -> None:
    # Inject defaults for sector_rotation, allow user overrides
    args = sys.argv[1:]
    if "--strategy" not in args:
        args = ["--strategy", "sector_rotation"] + args
    if "--risk-factor" not in args:
        args = ["--risk-factor", "0.02"] + args
    if "--max-position-pct" not in args:
        args = ["--max-position-pct", "0.12"] + args
    if "--top-pct" not in args:
        args = ["--top-pct", "1.0"] + args
    run_main(args)


if __name__ == "__main__":
    main()