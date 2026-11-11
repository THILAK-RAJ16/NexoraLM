"""
Comprehensive test suite for Phase 2: Custom Tokenizer and Dataset Preparation.
"""

import os
import json
import pytest
import torch

from tokenizer.tokenizer import CharacterTokenizer
from data.preprocessing import clean_text, split_data, load_text_file, save_text_file
from data.dataset import TextDataset, create_dataloader


# ==============================================================================
# Tokenizer Tests
# ==============================================================================

def test_tokenizer_initialization():
    """Verify tokenizer initializes with standard special tokens."""
    tok = CharacterTokenizer()
    assert tok.vocab_size == 4
    assert tok.token_to_id[CharacterTokenizer.PAD_TOKEN] == 0
    assert tok.token_to_id[CharacterTokenizer.UNK_TOKEN] == 1
    assert tok.token_to_id[CharacterTokenizer.BOS_TOKEN] == 2
    assert tok.token_to_id[CharacterTokenizer.EOS_TOKEN] == 3
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3


def test_vocabulary_generation():
    """Verify deterministic and dynamic vocabulary construction."""
    tok = CharacterTokenizer()
    sample_text = "banana split"
    tok.build_vocab(sample_text)
    
    # Unique characters: ' ', 'a', 'b', 'i', 'l', 'n', 'p', 's', 't' (9 chars)
    # Special tokens: 4
    # Total: 13
    assert tok.vocab_size == 13
    
    # Check that each character has a bi-directional mapping
    for ch in sample_text:
        tid = tok.token_to_id[ch]
        assert tok.id_to_token[tid] == ch


def test_encoding_and_decoding():
    """Verify round-trip encoding and decoding without loss."""
    tok = CharacterTokenizer()
    sample_text = "To be, or not to be: that is the question."
    tok.build_vocab(sample_text)
    
    encoded = tok.encode(sample_text)
    assert len(encoded) == len(sample_text)
    assert isinstance(encoded[0], int)
    
    decoded = tok.decode(encoded)
    assert decoded == sample_text


def test_unknown_token_handling():
    """Verify out-of-vocabulary characters map to UNK token."""
    tok = CharacterTokenizer()
    tok.build_vocab("abc")
    
    # 'z' and '9' are not in vocab
    encoded = tok.encode("a z 9 c")
    assert encoded[0] == tok.token_to_id["a"]
    assert encoded[1] == tok.unk_token_id  # space was not in vocab
    assert encoded[2] == tok.unk_token_id  # 'z'
    assert encoded[-1] == tok.token_to_id["c"]
    
    decoded = tok.decode(encoded)
    assert tok.UNK_TOKEN in decoded


def test_special_tokens_handling():
    """Verify BOS and EOS flag addition and skip_special_tokens decode."""
    tok = CharacterTokenizer()
    tok.build_vocab("hello")
    
    encoded = tok.encode("hello", add_bos=True, add_eos=True)
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id
    assert len(encoded) == 7  # 1 (bos) + 5 (hello) + 1 (eos)
    
    # Decode with special tokens kept
    decoded_with_special = tok.decode(encoded, skip_special_tokens=False)
    assert decoded_with_special.startswith(CharacterTokenizer.BOS_TOKEN)
    assert decoded_with_special.endswith(CharacterTokenizer.EOS_TOKEN)
    
    # Decode with special tokens skipped
    decoded_clean = tok.decode(encoded, skip_special_tokens=True)
    assert decoded_clean == "hello"


def test_vocabulary_save_and_load(tmp_path):
    """Verify vocabulary persistence and identical roundtrip behavior."""
    tok1 = CharacterTokenizer()
    tok1.build_vocab("The quick brown fox jumps over the lazy dog.")
    
    vocab_file = os.path.join(tmp_path, "vocab_test.json")
    tok1.save_vocabulary(vocab_file)
    assert os.path.exists(vocab_file)
    
    tok2 = CharacterTokenizer()
    tok2.load_vocabulary(vocab_file)
    
    assert tok1.vocab_size == tok2.vocab_size
    assert tok1.token_to_id == tok2.token_to_id
    assert tok1.id_to_token == tok2.id_to_token
    
    # Encode with tok1, decode with tok2
    text = "quick dog"
    assert tok2.decode(tok1.encode(text)) == text


# ==============================================================================
# Preprocessing Tests
# ==============================================================================

def test_clean_text_normalization():
    """Verify whitespace collapsing, newline standardizing, and unicode cleanup."""
    raw = "Line 1\r\nLine   2 with    multiple spaces.\n\n\n\nLine 3\t"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_split_data():
    """Verify split_data preserves lengths without leakage."""
    text = "A" * 90 + "B" * 10
    train_text, val_text = split_data(text, train_ratio=0.9)
    assert len(train_text) == 90
    assert len(val_text) == 10
    assert train_text == "A" * 90
    assert val_text == "B" * 10


def test_split_data_invalid_ratio():
    """Verify split_data raises ValueError on invalid ratio."""
    with pytest.raises(ValueError):
        split_data("test", train_ratio=1.5)
    with pytest.raises(ValueError):
        split_data("test", train_ratio=0.0)


# ==============================================================================
# Dataset and DataLoader Tests
# ==============================================================================

def test_dataset_sequence_creation():
    """Verify input-target next-token shifting behavior."""
    # Tokens: [10, 11, 12, 13, 14, 15]
    # With context_length=4:
    # Sample 0: x=[10, 11, 12, 13], y=[11, 12, 13, 14]
    # Sample 1: x=[11, 12, 13, 14], y=[12, 13, 14, 15] (if stride=1)
    tokens = [10, 11, 12, 13, 14, 15]
    dataset = TextDataset(tokens, context_length=4, stride=1)
    
    assert len(dataset) == 2
    x0, y0 = dataset[0]
    assert x0.tolist() == [10, 11, 12, 13]
    assert y0.tolist() == [11, 12, 13, 14]
    
    x1, y1 = dataset[1]
    assert x1.tolist() == [11, 12, 13, 14]
    assert y1.tolist() == [12, 13, 14, 15]


def test_dataloader_batch_shapes():
    """Verify DataLoader yields tensors of shape (batch_size, context_length)."""
    # 100 tokens, context_length = 8, batch_size = 4
    tokens = list(range(100))
    context_length = 8
    batch_size = 4
    
    dataset = TextDataset(tokens, context_length=context_length, stride=context_length)
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    
    batch_x, batch_y = next(iter(dataloader))
    assert batch_x.shape == (batch_size, context_length)
    assert batch_y.shape == (batch_size, context_length)
    assert batch_x.dtype == torch.long
    assert batch_y.dtype == torch.long
    
    # Verify shift across entire batch
    assert torch.equal(batch_x[:, 1:], batch_y[:, :-1])


def test_empty_and_short_dataset_handling():
    """Verify dataset behavior when tokens are fewer than context_length + 1."""
    # Only 5 tokens, but context_length requires 8 + 1 = 9 tokens
    short_tokens = [1, 2, 3, 4, 5]
    dataset = TextDataset(short_tokens, context_length=8)
    assert len(dataset) == 0
    
    with pytest.raises(IndexError):
        _ = dataset[0]
        
    empty_dataset = TextDataset([], context_length=8)
    assert len(empty_dataset) == 0

