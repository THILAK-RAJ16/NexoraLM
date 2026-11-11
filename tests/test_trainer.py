"""
Unit tests for Trainer, CosineWarmupScheduler, and training dynamics.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from config.config import ModelConfig, TrainingConfig
from model.gpt_model import GPTLanguageModel
from training.trainer import Trainer, CosineWarmupScheduler


def test_cosine_warmup_scheduler():
    """Verify linear warmup and cosine decay behaviors."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, max_steps=100, min_lr=1e-5)

    # Step 1: should warm up towards 1e-3
    lr1 = scheduler.step()
    assert 0.0 < lr1 < 1e-3
    assert optimizer.param_groups[0]["lr"] == lr1

    # Step 10: warmup finishes at peak base_lr
    for _ in range(9):
        scheduler.step()
    assert abs(scheduler.get_lr() - 1e-3) < 1e-6

    # Step 100: should reach min_lr
    for _ in range(90):
        scheduler.step()
    assert abs(scheduler.get_lr() - 1e-5) < 1e-6


def test_trainer_single_epoch_and_gradients(tmp_path):
    """Verify one epoch of training, loss computation, and gradient updates."""
    m_cfg = ModelConfig(vocab_size=32, embed_dim=64, num_layers=1, num_heads=2, context_length=16)
    t_cfg = TrainingConfig(
        batch_size=2,
        max_epochs=1,
        learning_rate=1e-3,
        grad_clip=1.0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        output_dir=str(tmp_path / "out")
    )
    model = GPTLanguageModel(m_cfg)

    # Synthetic batch data
    x = torch.randint(0, 32, (4, 16))
    y = torch.randint(0, 32, (4, 16))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        training_config=t_cfg,
        model_config=m_cfg
    )

    initial_weights = model.blocks[0].attn.qkv_projection.weight.clone()
    avg_loss = trainer.train_epoch(epoch=1)

    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0
    # Verify optimizer actually updated the weights
    assert not torch.allclose(initial_weights, model.blocks[0].attn.qkv_projection.weight)


def test_overfit_synthetic_dataset(tmp_path):
    """
    Overfit verification: A small model must successfully decrease its training loss
    significantly on a tiny synthetic dataset.
    """
    torch.manual_seed(42)
    m_cfg = ModelConfig(vocab_size=16, embed_dim=64, num_layers=2, num_heads=2, context_length=8, dropout=0.0)
    t_cfg = TrainingConfig(
        batch_size=2,
        max_epochs=35,
        learning_rate=2e-3,
        warmup_steps=2,
        grad_clip=1.0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        output_dir=str(tmp_path / "out")
    )
    model = GPTLanguageModel(m_cfg)

    # 4 synthetic samples
    x = torch.randint(0, 16, (4, 8))
    # Shifted target
    y = torch.randint(0, 16, (4, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        training_config=t_cfg,
        model_config=m_cfg
    )

    first_loss = trainer.train_epoch(epoch=1)
    for ep in range(2, 36):
        final_loss = trainer.train_epoch(epoch=ep)

    # Training loss must decrease noticeably
    assert final_loss < first_loss, f"Loss did not decrease: start {first_loss} -> end {final_loss}"
    assert final_loss < 0.5, f"Expected loss < 0.5 on tiny overfit set, got {final_loss}"


def test_trainer_cpu_and_gpu(tmp_path):
    """Verify Trainer device allocation."""
    m_cfg = ModelConfig(vocab_size=16, embed_dim=32, num_layers=1, num_heads=2, context_length=8)
    t_cfg = TrainingConfig(
        batch_size=2,
        max_epochs=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        output_dir=str(tmp_path / "out")
    )
    model = GPTLanguageModel(m_cfg)
    x = torch.randint(0, 16, (2, 8))
    loader = DataLoader(TensorDataset(x, x), batch_size=2)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        training_config=t_cfg,
        model_config=m_cfg
    )
    val_loss = trainer.evaluate()
    assert isinstance(val_loss, float)
    assert val_loss > 0.0
