"""
Self-Attention mechanisms for NexoraLM.
Implements Scaled Dot-Product Attention with causal masking and Multi-Head Self-Attention.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Self-Attention with Causal Masking:
    
        Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M ) * V
        
    Where:
        Q: Query tensor of shape (B, num_heads, T, head_dim)
        K: Key tensor of shape (B, num_heads, T, head_dim)
        V: Value tensor of shape (B, num_heads, T, head_dim)
        M: Causal mask where future positions (j > i) are set to -inf (or 0 after exp/softmax)
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query (torch.Tensor): (B, H, T_q, d_k)
            key (torch.Tensor): (B, H, T_k, d_k)
            value (torch.Tensor): (B, H, T_k, d_v) where d_v == d_k == head_dim
            mask (Optional[torch.Tensor]): Mask tensor broadcastable to (B, H, T_q, T_k).
                True/1 indicates valid attention, False/0 indicates masked out positions.
            return_attention_weights (bool): Whether to return attention probability matrix.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output tensor: (B, H, T_q, d_k)
                - Attention weights: (B, H, T_q, T_k) if return_attention_weights=True else None
        """
        head_dim = query.size(-1)
        # Scaled raw attention scores: (B, H, T_q, d_k) @ (B, H, d_k, T_k) -> (B, H, T_q, T_k)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if mask is not None:
            # Mask out positions where mask is False/0 by setting score to -inf (-1e9 for numerical stability)
            scores = scores.masked_fill(mask == 0, float("-1e9"))

        # Compute attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_weights)

        # Weighted sum of values: (B, H, T_q, T_k) @ (B, H, T_k, d_v) -> (B, H, T_q, d_v)
        output = torch.matmul(attn_probs, value)

        if return_attention_weights:
            return output, attn_weights
        return output, None


class CausalMultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention.
    
    Projects input into Queries, Keys, and Values across multiple attention heads,
    computes autoregressive causal self-attention ensuring token t can only attend
    to tokens <= t, concatenates heads, and applies an output linear projection.
    
    Shapes:
        Input: (B, T, embed_dim)
        Output: (B, T, embed_dim)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        
        # Combined Q, K, V projection for maximum computational efficiency
        # Input: (B, T, embed_dim) -> Output: (B, T, 3 * embed_dim)
        self.qkv_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.bias)
        
        # Final output projection
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=config.bias)
        
        # Dropouts
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.attention = ScaledDotProductAttention(dropout=config.dropout)
        
        # Register lower-triangular causal mask buffer (1 = attend, 0 = masked)
        # Shape: (1, 1, context_length, context_length)
        causal_mask = torch.tril(torch.ones(self.context_length, self.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, self.context_length, self.context_length))

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for CausalMultiHeadAttention.
        
        Args:
            x (torch.Tensor): Input activations of shape (B, T, embed_dim).
            return_attention_weights (bool): Whether to return the attention weights.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output tensor of shape (B, T, embed_dim)
                - Optional attention weights of shape (B, H, T, T)
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Expected input embed_dim={self.embed_dim}, got {embed_dim}"
        assert seq_len <= self.context_length, f"Sequence length {seq_len} exceeds max context {self.context_length}"

        # 1. Project to Q, K, V in a single matrix multiply: (B, T, 3 * embed_dim)
        qkv = self.qkv_projection(x)
        # Split into Q, K, V: each is (B, T, embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape and transpose for multi-head attention:
        # (B, T, embed_dim) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Slice causal mask for current sequence length: (1, 1, seq_len, seq_len)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # 4. Scaled dot-product attention across all heads
        attn_out, attn_weights = self.attention(
            query=q,
            key=k,
            value=v,
            mask=mask,
            return_attention_weights=return_attention_weights
        )

        # 5. Concatenate heads back together:
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, embed_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 6. Apply output projection and residual dropout
        out = self.resid_dropout(self.out_projection(attn_out))

        return out, attn_weights

