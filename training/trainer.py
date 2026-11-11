"""
Trainer, optimizer, and learning rate scheduler implementations for NexoraLM.
"""

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import ModelConfig, TrainingConfig
from model.gpt_model import GPTLanguageModel
from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker


class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine decay:
    
        - Warmup phase: lr increases linearly from 0 to base_lr over warmup_steps.
        - Cosine decay phase: lr decays from base_lr down to min_lr over remaining steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-5
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(self.warmup_steps + 1, max_steps)
        self.min_lr = min_lr
        
        # Capture initial base learning rate from optimizer param groups
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> float:
        """Advance scheduler by one step and update optimizer learning rates."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        """Compute learning rate for current step."""
        base_lr = self.base_lrs[0]
        
        # 1. Linear warmup
        if self.current_step < self.warmup_steps:
            return base_lr * (self.current_step / float(self.warmup_steps))
            
        # 2. Beyond max steps, maintain min_lr
        if self.current_step > self.max_steps:
            return self.min_lr
            
        # 3. Cosine decay
        decay_ratio = (self.current_step - self.warmup_steps) / float(self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (base_lr - self.min_lr)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_step = state_dict["current_step"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_steps = state_dict["max_steps"]
        self.min_lr = state_dict["min_lr"]
        self.base_lrs = state_dict["base_lrs"]
        # Immediately sync optimizer with restored step's learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class Trainer:
    """
    Complete Trainer for NexoraLM.
    
    Handles:
        - Epoch training loops
        - Mixed precision training with torch.amp.autocast and GradScaler
        - Evaluation loops
        - Gradient clipping
        - Optimizer and scheduler steps
        - Metric tracking
        - Checkpoint saving (latest and best)
        - Early stopping
    """

    def __init__(
        self,
        model: GPTLanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = training_config
        self.model_config = model_config
        
        # Device placement
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        print(f"[NexoraLM Trainer] Active Device: {self.device}")

        # Mixed Precision (AMP) setup
        # AMP is active only on CUDA and when use_amp is requested
        self.use_amp = bool(self.config.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            print("[NexoraLM Trainer] Automatic Mixed Precision (AMP) ENABLED")

        # Optimizer (AdamW with weight decay)
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # Scheduler (Warmup + Cosine Decay)
        total_steps = len(self.train_loader) * self.config.max_epochs
        if scheduler is None:
            self.scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=self.config.warmup_steps,
                max_steps=total_steps,
                min_lr=self.config.min_lr
            )
        else:
            self.scheduler = scheduler

        self.checkpoint_manager = checkpoint_manager or CheckpointManager(self.config.checkpoint_dir)
        self.metrics_tracker = metrics_tracker or MetricsTracker(self.config.output_dir)

        # Training state
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Train the model for one full epoch over train_loader.
        
        Args:
            epoch (int): Epoch index.
            
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            # Transfer batch to active device
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Clear gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional autocast
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits, loss = self.model(x, targets=y)
                assert loss is not None, "Target was provided but model returned None loss."

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Gradient unscaling and clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer and scheduler steps
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, max_iters: Optional[int] = None) -> float:
        """
        Evaluate model on val_loader.
        
        Args:
            max_iters (Optional[int]): Maximum number of evaluation batches to process.
                If None, processes all batches in val_loader.
                
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(self.val_loader):
            if max_iters is not None and batch_idx >= max_iters:
                break

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                _, loss = self.model(x, targets=y)

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def train(self) -> Dict[str, Any]:
        """
        Run the complete multi-epoch training loop.
        
        Returns:
            Dict[str, Any]: Final training summary with metrics.
        """
        print(f"Starting training: {self.config.max_epochs} epochs on {self.device}")
        
        for epoch in range(self.start_epoch, self.config.max_epochs + 1):
            t0 = time.time()

            # Train one epoch
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.evaluate(max_iters=self.config.eval_iters)
            epoch_time = time.time() - t0

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Check if this epoch produced a new best validation loss
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save latest and (if is_best) best checkpoints
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                step=self.global_step,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=self.best_val_loss,
                model_config=self.model_config,
                training_config=self.config,
                is_best=is_best
            )

            # Log metrics and update CSV
            self.metrics_tracker.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                epoch_duration=epoch_time,
                best_val_loss=self.best_val_loss
            )

            print(
                f"Epoch {epoch:02d}/{self.config.max_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
                f"{' [BEST]' if is_best else ''}"
            )

            # Early stopping check
            if self.config.patience is not None and self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch} epochs (patience: {self.config.patience})")
                break

        # Generate and save loss curve plot
        plot_path = self.metrics_tracker.plot_loss_curve()
        print(f"Loss plot saved to: {plot_path}")

        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": epoch,
            "global_step": self.global_step,
            "metrics_csv": self.metrics_tracker.csv_path,
            "loss_plot": plot_path
        }
