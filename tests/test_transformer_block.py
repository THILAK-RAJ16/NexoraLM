"""
Unit tests for FeedForwardNetwork and TransformerDecoderBlock.
"""

import pytest
import torch

from config.config import ModelConfig
from model.transformer_block import FeedForwardNetwork, TransformerDecoderBlock


def test_feed_forward_dimensions():
    """Verify FeedForwardNetwork expands to hidden_dim and projects back to embed_dim."""
    cfg = ModelConfig(embed_dim=128, ffn_mult=4, dropout=0.0)
    ffn = FeedForwardNetwork(cfg)
    
    assert ffn.fc1.out_features == 512
    assert ffn.fc2.out_features == 128
    
    x = torch.randn(2, 16, 128)
    out = ffn(x)
    assert out.shape == (2, 16, 128)


def test_transformer_decoder_block_dimensions_and_residuals():
    """Verify TransformerDecoderBlock shape preservation and residual addition."""
    cfg = ModelConfig(vocab_size=69, embed_dim=256, num_heads=4, context_length=128)
    block = TransformerDecoderBlock(cfg)
    
    batch_size = 2
    seq_len = 24
    x = torch.randn(batch_size, seq_len, 256)
    
    out, weights = block(x, return_attention_weights=True)
    assert out.shape == (batch_size, seq_len, 256)
    assert weights.shape == (batch_size, cfg.num_heads, seq_len, seq_len)


def test_transformer_block_gradient_propagation():
    """Verify gradients propagate through both attention and FFN sub-layers in block."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_heads=2, context_length=32)
    block = TransformerDecoderBlock(cfg)
    
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = block(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert block.attn.qkv_projection.weight.grad is not None
    assert block.ffn.fc1.weight.grad is not None
    assert block.ln_1.weight.grad is not None
    assert block.ln_2.weight.grad is not None


def test_cpu_and_gpu_compatibility():
    """Verify execution on CPU and GPU (if CUDA is available)."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_heads=2, context_length=32)
    block = TransformerDecoderBlock(cfg)
    
    # 1. CPU test
    x_cpu = torch.randn(1, 4, 64, device="cpu")
    out_cpu, _ = block(x_cpu)
    assert out_cpu.device.type == "cpu"
    
    # 2. CUDA test if available
    if torch.cuda.is_available():
        block_cuda = block.to("cuda")
        x_cuda = torch.randn(1, 4, 64, device="cuda")
        out_cuda, _ = block_cuda(x_cuda)
        assert out_cuda.device.type == "cuda"

