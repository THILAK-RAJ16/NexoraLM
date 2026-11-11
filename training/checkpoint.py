"""
Checkpoint management for NexoraLM.
Supports saving latest, saving best, and loading/resuming training states.
"""

import os
import torch
from typing import Any, Dict, Optional, Tuple

from config.config import ModelConfig, TrainingConfig
from model.gpt_model import GPTLanguageModel


class CheckpointManager:
    """
    Manages saving and loading model checkpoints.
    
    A checkpoint contains:
        - model_state_dict: Model parameters
        - optimizer_state_dict: Optimizer momentum and statistics
        - scheduler_state_dict: Learning rate scheduler state (if provided)
        - epoch: Epoch index
        - step: Global training step
        - train_loss: Training loss at time of checkpoint
        - val_loss: Validation loss at time of checkpoint
        - best_val_loss: Historical best validation loss
        - model_config: ModelConfig dictionary representation
        - training_config: TrainingConfig dictionary representation
    """

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir (str): Directory where checkpoints are saved.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        self.best_path = os.path.join(self.checkpoint_dir, "best.pt")

    def save_checkpoint(
        self,
        model: GPTLanguageModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: Current GPTLanguageModel.
            optimizer: Current Optimizer.
            scheduler: Current Learning Rate Scheduler (optional).
            epoch: Current epoch index.
            step: Global step count.
            train_loss: Current training loss.
            val_loss: Current validation loss.
            best_val_loss: Best validation loss seen so far.
            model_config: ModelConfig dataclass.
            training_config: TrainingConfig dataclass.
            is_best: Whether this checkpoint represents the best validation performance.
            filename: Optional custom filename under checkpoint_dir.
            
        Returns:
            str: Path where checkpoint was saved.
        """
        checkpoint_data: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "model_config": vars(model_config) if hasattr(model_config, "__dict__") else model_config,
            "training_config": vars(training_config) if hasattr(training_config, "__dict__") else training_config,
        }

        # 1. Save specific named file or latest.pt
        save_path = os.path.join(self.checkpoint_dir, filename) if filename else self.latest_path
        torch.save(checkpoint_data, save_path)

        # 2. If this is the best checkpoint, save to best.pt
        if is_best:
            torch.save(checkpoint_data, self.best_path)

        return save_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[GPTLanguageModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore states to provided model/optimizer/scheduler.
        
        Args:
            checkpoint_path (str): Filepath to the checkpoint.
            model (Optional[GPTLanguageModel]): Model to restore weights into.
            optimizer (Optional[Optimizer]): Optimizer to restore state into.
            scheduler (Optional[Any]): Scheduler to restore state into.
            device (str): Device to map tensors onto.
            
        Returns:
            Dict[str, Any]: Checkpoint dictionary.
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

