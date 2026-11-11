"""
Unit tests for Token and Positional Embeddings.
"""

import pytest
import torch

from config.config import ModelConfig
from model.embeddings import TransformerEmbeddings
from tokenizer.tokenizer import CharacterTokenizer


def test_embedding_output_dimensions():
    """Verify combined embeddings output tensor shapes."""
    cfg = ModelConfig(vocab_size=69, embed_dim=256, context_length=128)
    emb = TransformerEmbeddings(cfg)
    
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    out = emb(input_ids)
    assert out.shape == (batch_size, seq_len, cfg.embed_dim)


def test_positional_embeddings_dimensions():
    """Verify positional embedding table shape and explicit position_ids support."""
    cfg = ModelConfig(vocab_size=69, embed_dim=128, context_length=64)
    emb = TransformerEmbeddings(cfg)
    
    assert emb.positional_embeddings.weight.shape == (64, 128)
    
    # Explicit position IDs
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    pos_ids = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    out = emb(input_ids, position_ids=pos_ids)
    assert out.shape == (batch_size, seq_len, 128)


def test_context_length_exceeded_error():
    """Verify ValueError is raised if sequence exceeds context_length."""
    cfg = ModelConfig(vocab_size=69, embed_dim=128, context_length=32)
    emb = TransformerEmbeddings(cfg)
    
    too_long_ids = torch.randint(0, cfg.vocab_size, (1, 33))
    with pytest.raises(ValueError):
        _ = emb(too_long_ids)


def test_embeddings_from_tokenizer():
    """Verify ModelConfig.from_tokenizer builds matching embedding dimensions."""
    tok = CharacterTokenizer()
    tok.build_vocab("Hello world")
    
    cfg = ModelConfig.from_tokenizer(tok, embed_dim=128, context_length=64)
    assert cfg.vocab_size == tok.vocab_size
    assert cfg.pad_token_id == tok.pad_token_id
    
    emb = TransformerEmbeddings(cfg)
    assert emb.token_embeddings.num_embeddings == tok.vocab_size
    assert emb.token_embeddings.padding_idx == tok.pad_token_id


def test_embeddings_gradient_flow():
    """Verify gradients propagate through both token and positional embeddings."""
    cfg = ModelConfig(vocab_size=69, embed_dim=64, context_length=32, dropout=0.0)
    emb = TransformerEmbeddings(cfg)
    
    input_ids = torch.tensor([[1, 5, 10, 15]])
    out = emb(input_ids)
    loss = out.sum()
    loss.backward()
    
    assert emb.token_embeddings.weight.grad is not None
    assert emb.positional_embeddings.weight.grad is not None
    # Verify non-zero gradient for accessed indices
    assert emb.token_embeddings.weight.grad[1].abs().sum() > 0
    assert emb.positional_embeddings.weight.grad[0].abs().sum() > 0

