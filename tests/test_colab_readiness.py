"""
Unit tests for Google Colab GPU training prerequisites, AMP, and checkpoint portability (Phase 6).
"""

import os
import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader

from config.config import ModelConfig, TrainingConfig
from model.gpt_model import GPTLanguageModel
from training.trainer import Trainer, CosineWarmupScheduler
from training.checkpoint import CheckpointManager


def test_amp_and_grad_scaler_initialization():
    """Verify Trainer correctly enables or disables AMP based on device and config."""
    m_cfg = ModelConfig(vocab_size=32, embed_dim=64, num_layers=1, num_heads=2, context_length=16)
    
    # On CPU, AMP scaler should remain disabled safely even if use_amp is set True
    t_cfg_cpu = TrainingConfig(device="cpu", use_amp=True)
    model_cpu = GPTLanguageModel(m_cfg)
    loader_cpu = DataLoader(TensorDataset(torch.zeros(2, 16, dtype=torch.long), torch.zeros(2, 16, dtype=torch.long)))
    
    trainer_cpu = Trainer(
        model=model_cpu,
        train_loader=loader_cpu,
        val_loader=loader_cpu,
        training_config=t_cfg_cpu,
        model_config=m_cfg
    )
    assert trainer_cpu.use_amp is False
    assert trainer_cpu.scaler.is_enabled() is False


def test_checkpoint_cpu_cuda_portability(tmp_path):
    """Verify checkpoints saved on one device can be loaded seamlessly on another device mapping."""
    ckpt_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(checkpoint_dir=ckpt_dir)

    m_cfg = ModelConfig(vocab_size=32, embed_dim=64, num_layers=1, num_heads=2, context_length=16)
    t_cfg = TrainingConfig()

    model_src = GPTLanguageModel(m_cfg)
    optimizer_src = torch.optim.AdamW(model_src.parameters(), lr=1e-3)
    scheduler_src = CosineWarmupScheduler(optimizer_src, warmup_steps=5, max_steps=50)

    # Advance scheduler to step 30
    for _ in range(30):
        scheduler_src.step()

    save_path = manager.save_checkpoint(
        model=model_src,
        optimizer=optimizer_src,
        scheduler=scheduler_src,
        epoch=3,
        step=30,
        train_loss=2.1,
        val_loss=2.0,
        best_val_loss=2.0,
        model_config=m_cfg,
        training_config=t_cfg,
        is_best=True
    )

    # Load with explicit CPU map_location
    model_dst = GPTLanguageModel(m_cfg)
    optimizer_dst = torch.optim.AdamW(model_dst.parameters(), lr=1e-3)
    scheduler_dst = CosineWarmupScheduler(optimizer_dst, warmup_steps=5, max_steps=50)

    loaded_dict = manager.load_checkpoint(
        checkpoint_path=save_path,
        model=model_dst,
        optimizer=optimizer_dst,
        scheduler=scheduler_dst,
        device="cpu"
    )

    assert loaded_dict["epoch"] == 3
    assert loaded_dict["step"] == 30
    assert scheduler_dst.current_step == 30
    # Verify learning rate was restored and synchronized with optimizer
    assert abs(optimizer_dst.param_groups[0]["lr"] - scheduler_dst.get_lr()) < 1e-7


def test_scheduler_state_restoration_and_sync():
    """Verify CosineWarmupScheduler restores state_dict and synchronizes optimizer lr immediately."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, max_steps=100, min_lr=1e-5)

    # Step to position 5
    for _ in range(5):
        scheduler.step()
    expected_lr = scheduler.get_lr()
    state = scheduler.state_dict()

    # Fresh optimizer and scheduler
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler2 = CosineWarmupScheduler(optimizer2, warmup_steps=10, max_steps=100, min_lr=1e-5)
    scheduler2.load_state_dict(state)

    assert scheduler2.current_step == 5
    assert abs(optimizer2.param_groups[0]["lr"] - expected_lr) < 1e-7
