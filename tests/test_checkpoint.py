"""
Unit tests for CheckpointManager.
"""

import os
import torch
import pytest

from config.config import ModelConfig, TrainingConfig
from model.gpt_model import GPTLanguageModel
from training.checkpoint import CheckpointManager
from training.trainer import CosineWarmupScheduler


def test_checkpoint_save_and_load(tmp_path):
    """Verify full save and load cycle for model, optimizer, scheduler, and configs."""
    ckpt_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(checkpoint_dir=ckpt_dir)

    m_cfg = ModelConfig(vocab_size=69, embed_dim=64, num_layers=1, num_heads=2, context_length=32)
    t_cfg = TrainingConfig(learning_rate=1e-3, max_epochs=5)

    model1 = GPTLanguageModel(m_cfg)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    scheduler1 = CosineWarmupScheduler(optimizer1, warmup_steps=10, max_steps=100)

    # Perform a dummy update so optimizer has state
    x = torch.randint(0, 69, (2, 8))
    _, loss = model1(x, targets=x)
    loss.backward()
    optimizer1.step()
    scheduler1.step()

    # Save latest and best
    save_path = manager.save_checkpoint(
        model=model1,
        optimizer=optimizer1,
        scheduler=scheduler1,
        epoch=2,
        step=50,
        train_loss=2.5,
        val_loss=2.4,
        best_val_loss=2.4,
        model_config=m_cfg,
        training_config=t_cfg,
        is_best=True
    )
    assert os.path.exists(save_path)
    assert os.path.exists(manager.best_path)

    # Restore into fresh instances
    model2 = GPTLanguageModel(m_cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = CosineWarmupScheduler(optimizer2, warmup_steps=10, max_steps=100)

    ckpt = manager.load_checkpoint(
        checkpoint_path=manager.latest_path,
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2
    )

    assert ckpt["epoch"] == 2
    assert ckpt["step"] == 50
    assert ckpt["best_val_loss"] == 2.4

    # Verify weights restored identically
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_best_checkpoint_selection(tmp_path):
    """Verify best.pt is preserved and not overwritten by an inferior validation result."""
    ckpt_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(checkpoint_dir=ckpt_dir)

    m_cfg = ModelConfig(vocab_size=69, embed_dim=64, num_layers=1, num_heads=2, context_length=32)
    t_cfg = TrainingConfig()

    model = GPTLanguageModel(m_cfg)
    optimizer = torch.optim.AdamW(model.parameters())

    # Epoch 1: val_loss 3.0 (best so far)
    manager.save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=1, step=10, train_loss=3.2, val_loss=3.0, best_val_loss=3.0,
        model_config=m_cfg, training_config=t_cfg, is_best=True
    )

    # Epoch 2: val_loss 3.5 (worse, is_best=False)
    manager.save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=2, step=20, train_loss=2.8, val_loss=3.5, best_val_loss=3.0,
        model_config=m_cfg, training_config=t_cfg, is_best=False
    )

    # best.pt should still contain epoch 1
    best_ckpt = manager.load_checkpoint(manager.best_path)
    assert best_ckpt["epoch"] == 1
    assert best_ckpt["val_loss"] == 3.0

    # latest.pt should contain epoch 2
    latest_ckpt = manager.load_checkpoint(manager.latest_path)
    assert latest_ckpt["epoch"] == 2
    assert latest_ckpt["val_loss"] == 3.5

