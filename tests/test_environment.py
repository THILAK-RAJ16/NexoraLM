"""
Environment and structure verification tests for Phase 1.
"""

import sys
import os
import torch
import numpy as np

# Ensure root of NexoraLM is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import ModelConfig, TrainingConfig, InferenceConfig


def test_model_config():
    """Verify default ModelConfig values and calculations."""
    cfg = ModelConfig()
    assert cfg.vocab_size == 69
    assert cfg.embed_dim == 256
    assert cfg.num_layers == 4
    assert cfg.num_heads == 4
    assert cfg.context_length == 256
    assert cfg.dropout == 0.1
    assert cfg.head_dim == 64, f"Expected head_dim=64, got {cfg.head_dim}"


def test_training_config():
    """Verify default TrainingConfig and device resolution."""
    cfg = TrainingConfig()
    assert cfg.batch_size == 16
    assert cfg.learning_rate == 3e-4
    assert cfg.device in ("cuda", "cpu")


def test_inference_config():
    """Verify default InferenceConfig."""
    cfg = InferenceConfig()
    assert cfg.max_new_tokens == 100
    assert cfg.temperature == 0.8
    assert cfg.top_k == 40
    assert cfg.seed == 42


def test_pytorch_tensor_operations():
    """Verify PyTorch and NumPy basic tensor operations work in this environment."""
    t = torch.randn(2, 4)
    assert t.shape == (2, 4)
    arr = np.ones((3, 3))
    assert arr.sum() == 9.0


if __name__ == "__main__":
    test_model_config()
    test_training_config()
    test_inference_config()
    test_pytorch_tensor_operations()
    print("All Phase 1 verification tests passed successfully!")

