"""
Unit tests for MetricsTracker.
"""

import os
import csv
import pytest

from training.metrics import MetricsTracker


def test_metrics_tracker_logging_and_csv(tmp_path):
    """Verify metrics logging, CSV export, and history structure."""
    out_dir = str(tmp_path / "outputs")
    tracker = MetricsTracker(output_dir=out_dir)

    tracker.log_epoch(epoch=1, train_loss=3.5, val_loss=3.2, learning_rate=1e-4, epoch_duration=10.5, best_val_loss=3.2)
    tracker.log_epoch(epoch=2, train_loss=2.8, val_loss=2.7, learning_rate=2e-4, epoch_duration=9.8, best_val_loss=2.7)

    assert os.path.exists(tracker.csv_path)

    # Read back CSV
    with open(tracker.csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["epoch"] == "1"
        assert float(reader[0]["train_loss"]) == 3.5
        assert reader[1]["epoch"] == "2"
        assert float(reader[1]["val_loss"]) == 2.7


def test_metrics_loss_plot_generation(tmp_path):
    """Verify loss curve image generation."""
    out_dir = str(tmp_path / "outputs")
    tracker = MetricsTracker(output_dir=out_dir)

    for ep in range(1, 4):
        tracker.log_epoch(
            epoch=ep,
            train_loss=4.0 - ep * 0.5,
            val_loss=4.2 - ep * 0.4,
            learning_rate=3e-4,
            epoch_duration=5.0,
            best_val_loss=4.2 - ep * 0.4
        )

    plot_file = tracker.plot_loss_curve()
    assert os.path.exists(plot_file)
    assert os.path.getsize(plot_file) > 1000  # Non-empty PNG image

