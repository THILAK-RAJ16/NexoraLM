"""
Unit tests for GPTLanguageModel (Phase 4).
"""

import math
import pytest
import torch

from config.config import ModelConfig
from model.gpt_model import GPTLanguageModel
from tokenizer.tokenizer import CharacterTokenizer


def test_gpt_model_initialization():
    """Verify model initialization, sub-modules, and configurations."""
    cfg = ModelConfig(vocab_size=69, embed_dim=128, num_layers=2, num_heads=4, context_length=64)
    model = GPTLanguageModel(cfg)
    
    assert len(model.blocks) == 2
    assert model.lm_head.out_features == 69
    assert model.lm_head.in_features == 128
    assert model.ln_f.normalized_shape[0] == 128


def test_forward_pass_output_dimensions():
    """Verify forward pass output logits shape with various batch and sequence lengths."""
    cfg = ModelConfig(vocab_size=69, embed_dim=128, num_layers=2, num_heads=4, context_length=64)
    model = GPTLanguageModel(cfg)
    model.eval()
    
    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids)
    assert loss is None
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)


def test_correct_vocabulary_dimension_in_logits():
    """Verify logits vocabulary dimension matches vocab_size exactly."""
    for vocab_sz in [50, 69, 100]:
        cfg = ModelConfig(vocab_size=vocab_sz, embed_dim=64, num_layers=1, num_heads=2, context_length=32)
        model = GPTLanguageModel(cfg)
        model.eval()
        
        x = torch.randint(0, vocab_sz, (2, 8))
        logits, _ = model(x)
        assert logits.size(-1) == vocab_sz


def test_loss_calculation_with_targets():
    """Verify cross-entropy loss calculation when aligned targets are provided."""
    cfg = ModelConfig(vocab_size=69, embed_dim=128, num_layers=2, num_heads=4, context_length=64)
    model = GPTLanguageModel(cfg)
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids, targets=targets)
    assert loss is not None
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_forward_pass_without_targets():
    """Verify forward pass without targets returns None for loss."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_layers=1, num_heads=2, context_length=32)
    model = GPTLanguageModel(cfg)
    
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    logits, loss = model(input_ids)
    assert loss is None
    assert logits is not None


def test_loss_ignoring_padding_tokens():
    """Verify padding tokens (ignore_index) are completely excluded from loss calculation."""
    cfg = ModelConfig(
        vocab_size=69,
        embed_dim=64,
        num_layers=1,
        num_heads=2,
        context_length=32,
        pad_token_id=0,
        ignore_index=0
    )
    model = GPTLanguageModel(cfg)
    model.eval()
    
    input_ids = torch.tensor([[10, 20, 30, 40]])
    targets_with_pad = torch.tensor([[11, 21, 0, 0]])  # Last 2 are pad tokens
    
    # 1. Loss computed with padding targets
    _, loss_with_pad = model(input_ids, targets=targets_with_pad)
    
    # 2. Loss computed only on non-padded slice
    _, loss_unpadded = model(input_ids[:, :2], targets=targets_with_pad[:, :2])
    
    # Because causal attention ensures positions 0 and 1 cannot see 2 and 3,
    # the unpadded loss on positions 0 and 1 must be identical to the loss
    # calculated over the whole sequence with ignore_index=0!
    assert torch.allclose(loss_with_pad, loss_unpadded, atol=1e-5), (
        f"Loss with padding ({loss_with_pad.item()}) did not match unpadded slice ({loss_unpadded.item()})"
    )


def test_context_length_validation():
    """Verify exception raised when input exceeds context_length."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_layers=1, num_heads=2, context_length=16)
    model = GPTLanguageModel(cfg)
    
    too_long = torch.randint(0, cfg.vocab_size, (1, 17))
    with pytest.raises(ValueError):
        _ = model(too_long)


def test_dynamic_vocabulary_size_using_tokenizer():
    """Verify GPT model dynamically adopts vocabulary size from tokenizer."""
    tok = CharacterTokenizer()
    tok.build_vocab("The quick brown fox jumps over the lazy dog.")
    
    cfg = ModelConfig.from_tokenizer(tok, embed_dim=128, num_layers=2, num_heads=4, context_length=64)
    assert cfg.vocab_size == tok.vocab_size
    assert cfg.pad_token_id == tok.pad_token_id
    assert cfg.ignore_index == tok.pad_token_id
    
    model = GPTLanguageModel(cfg)
    assert model.lm_head.out_features == tok.vocab_size
    
    # Verify forward pass with encoded text
    sample_ids = torch.tensor([tok.encode("The quick fox")])
    logits, _ = model(sample_ids)
    assert logits.shape == (1, sample_ids.shape[1], tok.vocab_size)


def test_gradient_propagation_through_all_blocks():
    """Verify gradients flow end-to-end through LM head, all blocks, and embeddings."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_layers=3, num_heads=2, context_length=32)
    model = GPTLanguageModel(cfg)
    
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    
    logits, loss = model(input_ids, targets=targets)
    loss.backward()
    
    # Check embeddings
    assert model.embeddings.token_embeddings.weight.grad is not None
    assert model.embeddings.positional_embeddings.weight.grad is not None
    
    # Check each transformer decoder block
    for i, block in enumerate(model.blocks):
        assert block.attn.qkv_projection.weight.grad is not None, f"Block {i} attn grad missing"
        assert block.attn.out_projection.weight.grad is not None, f"Block {i} attn out grad missing"
        assert block.ffn.fc1.weight.grad is not None, f"Block {i} ffn fc1 grad missing"
        assert block.ffn.fc2.weight.grad is not None, f"Block {i} ffn fc2 grad missing"
        assert block.ln_1.weight.grad is not None, f"Block {i} ln_1 grad missing"
        assert block.ln_2.weight.grad is not None, f"Block {i} ln_2 grad missing"
        
    # Check final LayerNorm
    assert model.ln_f.weight.grad is not None


def test_parameter_count_calculation():
    """Verify get_num_params calculation matches manual summation."""
    cfg = ModelConfig(vocab_size=69, embed_dim=128, num_layers=2, num_heads=4, context_length=64, tie_weights=True)
    model = GPTLanguageModel(cfg)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.get_num_params() == total_params
    
    non_emb_params = total_params - model.embeddings.positional_embeddings.weight.numel()
    assert model.get_num_params(non_embedding=True) == non_emb_params


def test_weight_tying_consistency():
    """Verify weight tying shares memory buffer between token embeddings and lm_head."""
    # 1. Tied weights
    cfg_tied = ModelConfig(vocab_size=69, embed_dim=128, tie_weights=True)
    model_tied = GPTLanguageModel(cfg_tied)
    assert model_tied.lm_head.weight is model_tied.embeddings.token_embeddings.weight
    
    # 2. Untied weights
    cfg_untied = ModelConfig(vocab_size=69, embed_dim=128, tie_weights=False)
    model_untied = GPTLanguageModel(cfg_untied)
    assert model_untied.lm_head.weight is not model_untied.embeddings.token_embeddings.weight


def test_no_future_leakage_in_full_model():
    """Verify changing future tokens in input does not alter past token logits in eval mode."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_layers=2, num_heads=2, context_length=32, dropout=0.0)
    model = GPTLanguageModel(cfg)
    model.eval()
    
    # Sequence A: [10, 20, 30, 40]
    seq_a = torch.tensor([[10, 20, 30, 40]])
    # Sequence B: [10, 20, 30, 65] (only last token is altered)
    seq_b = torch.tensor([[10, 20, 30, 65]])
    
    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)
        
    # Logits at positions 0, 1, 2 must be mathematically identical!
    assert torch.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-5), (
        "Causal leakage detected in complete GPT model: future token altered past logits!"
    )
    # Logits at position 3 (the changed position) should be different
    assert not torch.allclose(logits_a[:, 3, :], logits_b[:, 3, :], atol=1e-3)


def test_cpu_and_gpu_compatibility():
    """Verify execution on CPU and CUDA (if available)."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, num_layers=2, num_heads=2, context_length=32)
    model = GPTLanguageModel(cfg)
    
    # 1. CPU
    x_cpu = torch.randint(0, cfg.vocab_size, (2, 8), device="cpu")
    y_cpu = torch.randint(0, cfg.vocab_size, (2, 8), device="cpu")
    logits_cpu, loss_cpu = model(x_cpu, targets=y_cpu)
    assert logits_cpu.device.type == "cpu"
    assert loss_cpu.device.type == "cpu"
    
    # 2. CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        x_cuda = x_cpu.to("cuda")
        y_cuda = y_cpu.to("cuda")
        logits_cuda, loss_cuda = model_cuda(x_cuda, targets=y_cuda)
        assert logits_cuda.device.type == "cuda"
        assert loss_cuda.device.type == "cuda"

