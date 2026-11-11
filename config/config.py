"""
Configuration module for NexoraLM.
Defines model hyperparameters, training configurations, and runtime settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import torch


@dataclass
class ModelConfig:
    """Hyperparameter configuration for the NexoraLM Decoder-only Transformer."""
    vocab_size: int = 69           # Vocabulary size (dynamically settable from tokenizer)
    embed_dim: int = 256          # Embedding dimension (d_model)
    num_layers: int = 4           # Number of Transformer decoder blocks
    num_heads: int = 4            # Number of attention heads
    context_length: int = 256     # Context window / maximum sequence length (block_size)
    dropout: float = 0.1          # Dropout rate for regularization
    bias: bool = True             # Whether to use bias in linear projections
    pad_token_id: int = 0         # Padding token id (ignored in cross-entropy loss)
    ignore_index: int = 0         # Target index ignored in cross entropy loss (corresponds to pad_token_id)
    ffn_mult: int = 4             # Hidden dimension multiplier for FFN (d_ff = ffn_mult * embed_dim)
    tie_weights: bool = True      # Whether to tie token embeddings and LM head projection weights

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.embed_dim // self.num_heads

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> "ModelConfig":
        """
        Create a ModelConfig instance deriving vocab_size, pad_token_id, and
        ignore_index dynamically from an initialized tokenizer instance.
        """
        vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0
        kwargs.setdefault("ignore_index", pad_token_id)
        return cls(vocab_size=vocab_size, pad_token_id=pad_token_id, **kwargs)


@dataclass
class TrainingConfig:
    """Configuration for model training and optimization."""
    batch_size: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5          # Minimum learning rate for cosine decay scheduler
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_epochs: int = 10
    warmup_steps: int = 100       # Number of linear warmup steps
    grad_clip: float = 1.0        # Gradient clipping threshold
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"   # Directory for metrics CSV and loss plots
    save_interval: int = 500       # Step interval for saving periodic checkpoints
    eval_interval: int = 100       # Step interval for computing validation loss
    eval_iters: Optional[int] = 50 # Number of batches used during evaluation (None = evaluate entire val set)
    num_workers: int = 0          # DataLoader subprocess workers
    patience: Optional[int] = 5   # Early stopping patience in epochs (None to disable)
    seed: int = 42                # Random seed for reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()  # Automatic Mixed Precision
    ignore_index: int = 0          # Target index ignored in cross entropy loss (corresponds to pad_token_id)


@dataclass
class InferenceConfig:
    """Configuration for autoregressive text generation."""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = 40
    seed: Optional[int] = 42
