"""Report output layer — metrics, equity, trades, rolling, main."""

from clenow.report.main import generate_report
from clenow.report.metrics import compute_metrics

__all__ = ["compute_metrics", "generate_report"]
