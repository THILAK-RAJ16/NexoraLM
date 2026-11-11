"""
Unit tests for Scaled Dot-Product Attention and Causal Multi-Head Self-Attention.
"""

import pytest
import torch

from config.config import ModelConfig
from model.attention import ScaledDotProductAttention, CausalMultiHeadAttention


def test_scaled_dot_product_attention_shapes_and_normalization():
    """Verify ScaledDotProductAttention shapes and attention weight sum to 1."""
    attn = ScaledDotProductAttention(dropout=0.0)
    batch_size = 2
    num_heads = 4
    seq_len = 10
    head_dim = 16
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    out, weights = attn(q, k, v, return_attention_weights=True)
    assert out.shape == (batch_size, num_heads, seq_len, head_dim)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Softmax rows must sum to 1.0 along the key dimension
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_masking_correctness():
    """Verify that causal attention weights for future tokens (j > i) are strictly 0."""
    attn = ScaledDotProductAttention(dropout=0.0)
    seq_len = 5
    head_dim = 8
    
    q = torch.randn(1, 1, seq_len, head_dim)
    k = torch.randn(1, 1, seq_len, head_dim)
    v = torch.randn(1, 1, seq_len, head_dim)
    
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
    _, weights = attn(q, k, v, mask=mask, return_attention_weights=True)
    
    weights = weights.squeeze()  # (seq_len, seq_len)
    
    # Check strictly upper triangular entries are 0.0
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert weights[i, j].item() == 0.0, f"Future leakage at ({i}, {j}): {weights[i, j].item()}"
            else:
                assert weights[i, j].item() > 0.0, f"Valid position ({i}, {j}) has zero weight"


def test_no_future_token_information_leakage():
    """
    Direct gradient and numerical proof of no future information leakage:
    Changing future tokens in an input sequence must NOT affect the output representation
    or loss computed at earlier positions.
    """
    cfg = ModelConfig(vocab_size=69, embed_dim=32, num_heads=2, context_length=16, dropout=0.0)
    mha = CausalMultiHeadAttention(cfg)
    mha.eval()
    
    # Sequence A: [t0, t1, t2, t3]
    x_a = torch.randn(1, 4, 32)
    
    # Sequence B: identical for positions 0..2, but position 3 is completely modified
    x_b = x_a.clone()
    x_b[:, 3, :] = torch.randn(1, 32) * 50.0
    
    out_a, _ = mha(x_a)
    out_b, _ = mha(x_b)
    
    # Representations for tokens 0, 1, 2 must be mathematically identical
    assert torch.allclose(out_a[:, :3, :], out_b[:, :3, :], atol=1e-6), (
        "Causal leakage detected: modifying future token altered past token representations!"
    )
    # Representation for token 3 should be different
    assert not torch.allclose(out_a[:, 3, :], out_b[:, 3, :], atol=1e-3)


def test_multi_head_attention_dimensions():
    """Verify CausalMultiHeadAttention input and output shapes."""
    cfg = ModelConfig(vocab_size=69, embed_dim=256, num_heads=4, context_length=128)
    mha = CausalMultiHeadAttention(cfg)
    
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 256)
    
    out, weights = mha(x, return_attention_weights=True)
    assert out.shape == (batch_size, seq_len, 256)
    assert weights.shape == (batch_size, cfg.num_heads, seq_len, seq_len)


def test_invalid_head_configurations():
    """Verify exception is raised when embed_dim is not divisible by num_heads."""
    with pytest.raises(ValueError):
        _ = ModelConfig(embed_dim=250, num_heads=4)


def test_mha_gradient_propagation():
    """Verify gradients propagate backwards through attention projections."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_heads=2, context_length=32)
    mha = CausalMultiHeadAttention(cfg)
    
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = mha(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert mha.qkv_projection.weight.grad is not None
    assert mha.out_projection.weight.grad is not None


def test_attention_float32():
    """Verify ScaledDotProductAttention and CausalMultiHeadAttention under standard float32."""
    attn = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 8, 16, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 4, 8, 16, dtype=torch.float32, requires_grad=True)
    mask = torch.tril(torch.ones(8, 8, dtype=torch.bool)).view(1, 1, 8, 8)

    out, weights = attn(q, k, v, mask=mask, return_attention_weights=True)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()

    # Verify backward pass produces finite gradients
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


def test_attention_float16_no_overflow():
    """
    Verify ScaledDotProductAttention and CausalMultiHeadAttention under torch.float16 (Half).
    Prevents 'RuntimeError: value cannot be converted to type c10::Half without overflow'.
    """
    attn = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 4, 8, 16, dtype=torch.float16, requires_grad=True)
    v = torch.randn(2, 4, 8, 16, dtype=torch.float16, requires_grad=True)
    mask = torch.tril(torch.ones(8, 8, dtype=torch.bool)).view(1, 1, 8, 8)

    # Must execute without overflow error
    out, weights = attn(q, k, v, mask=mask, return_attention_weights=True)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()

    # Verify strictly 0 attention weights for masked future positions (j > i)
    for i in range(8):
        for j in range(8):
            if j > i:
                assert weights[:, :, i, j].max().item() == 0.0

    # Verify finite gradients in float16
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()

    # Test full CausalMultiHeadAttention module with float16
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_heads=4, context_length=32, dropout=0.0)
    mha = CausalMultiHeadAttention(cfg).to(dtype=torch.float16)
    x = torch.randn(2, 12, 64, dtype=torch.float16, requires_grad=True)
    mha_out, mha_weights = mha(x, return_attention_weights=True)
    assert mha_out.dtype == torch.float16
    assert torch.isfinite(mha_out).all()
    mha_loss = mha_out.sum()
    mha_loss.backward()
    assert torch.isfinite(x.grad).all()


def test_attention_cuda_autocast_when_available():
    """
    Verify ScaledDotProductAttention and CausalMultiHeadAttention under torch.autocast on CUDA.
    If CUDA is not available, verify the test reports skip safely.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available in current environment; skipping CUDA autocast test.")

    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_heads=4, context_length=32, dropout=0.0)
    mha = CausalMultiHeadAttention(cfg).cuda()
    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out, weights = mha(x, return_attention_weights=True)
        assert torch.isfinite(out).all()
        assert torch.isfinite(weights).all()
        loss = out.sum()

    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(loss).backward()
    scaler.step(torch.optim.AdamW(mha.parameters()))
    scaler.update()
    assert torch.isfinite(x.grad).all()

