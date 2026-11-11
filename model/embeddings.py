"""
Embedding components for NexoraLM.
Implements learnable token embeddings and learnable positional embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional

from config.config import ModelConfig


class TransformerEmbeddings(nn.Module):
    """
    Combined Token and Positional Embeddings for Decoder-Only Transformer.
    
    Given an input tensor of token IDs with shape (batch_size, sequence_length),
    computes:
        Output = Dropout(TokenEmbedding(tokens) + PositionalEmbedding(positions))
    
    Shapes:
        Input: (B, T) where B is batch size and T <= context_length
        Output: (B, T, embed_dim)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize Token and Positional embeddings.
        
        Args:
            config (ModelConfig): Model hyperparameter configuration.
        """
        super().__init__()
        self.context_length = config.context_length
        self.embed_dim = config.embed_dim
        
        # Token embedding lookup: maps vocabulary IDs to embed_dim vectors
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id
        )
        
        # Learnable positional embedding table: positions from 0 to context_length - 1
        self.positional_embeddings = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.embed_dim
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Zero out padding token embedding row if pad_token_id is configured
        if config.pad_token_id is not None and 0 <= config.pad_token_id < config.vocab_size:
            with torch.no_grad():
                self.token_embeddings.weight[config.pad_token_id].fill_(0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for TransformerEmbeddings.
        
        Args:
            input_ids (torch.Tensor): LongTensor of token IDs, shape (B, T).
            position_ids (Optional[torch.Tensor]): Optional explicit position IDs, shape (B, T) or (1, T).
                If None, defaults to [0, 1, ..., T - 1].
                
        Returns:
            torch.Tensor: Combined embeddings with dropout, shape (B, T, embed_dim).
        """
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.context_length:
            raise ValueError(
                f"Cannot process sequence of length {seq_len}; exceeds model context_length {self.context_length}"
            )
            
        # 1. Look up token embeddings: (B, T) -> (B, T, embed_dim)
        tok_emb = self.token_embeddings(input_ids)
        
        # 2. Compute or look up positional embeddings: (T,) or (B, T) -> (B, T, embed_dim)
        if position_ids is None:
            # Generate deterministic positions: [0, 1, ..., seq_len - 1] on same device
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            # (T,) -> (1, T, embed_dim)
            pos_emb = self.positional_embeddings(position_ids).unsqueeze(0)
        else:
            pos_emb = self.positional_embeddings(position_ids)
            
        # 3. Sum token and position embeddings, apply dropout
        x = self.dropout(tok_emb + pos_emb)
        return x

