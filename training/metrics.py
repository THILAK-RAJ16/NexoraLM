"""
Training metrics tracking, logging, and plotting for NexoraLM.
"""

import os
import csv
import time
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless file generation
import matplotlib.pyplot as plt


class MetricsTracker:
    """
    Tracks per-epoch training and validation metrics, exports to CSV,
    and plots loss curves saved to disk.
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        """
        Initialize MetricsTracker.
        
        Args:
            output_dir (str): Directory where CSV and plot images will be stored.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "metrics.csv")
        self.plot_path = os.path.join(self.output_dir, "loss_curve.png")
        
        self.history: List[Dict[str, float]] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        epoch_duration: float,
        best_val_loss: float
    ) -> None:
        """
        Log metrics for a completed epoch and write/append to CSV.
        
        Args:
            epoch: 1-indexed epoch number.
            train_loss: Average training loss for epoch.
            val_loss: Average validation loss for epoch.
            learning_rate: Current learning rate.
            epoch_duration: Time in seconds taken by the epoch.
            best_val_loss: Lowest validation loss recorded so far.
        """
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "learning_rate": learning_rate,
            "epoch_duration_sec": round(epoch_duration, 2),
            "best_val_loss": round(best_val_loss, 6),
        }
        self.history.append(entry)
        self._write_csv()

    def _write_csv(self) -> None:
        """Write full history to CSV file."""
        if not self.history:
            return
            
        fieldnames = list(self.history[0].keys())
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def plot_loss_curve(self, save_path: Optional[str] = None) -> str:
        """
        Generate and save training vs. validation loss curve to disk.
        
        Args:
            save_path (Optional[str]): Custom save path. Defaults to output_dir/loss_curve.png.
            
        Returns:
            str: Path to saved plot image.
        """
        if not self.history:
            return ""

        path = save_path or self.plot_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        epochs = [entry["epoch"] for entry in self.history]
        train_losses = [entry["train_loss"] for entry in self.history]
        val_losses = [entry["val_loss"] for entry in self.history]

        plt.figure(figsize=(9, 5), dpi=150)
        plt.plot(epochs, train_losses, "b-o", label="Training Loss", linewidth=2, markersize=5)
        plt.plot(epochs, val_losses, "r--s", label="Validation Loss", linewidth=2, markersize=5)

        plt.title("NexoraLM Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Cross-Entropy Loss", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        plt.savefig(path)
        plt.close()

        return path

