"""
Transformer Decoder Block and Feed-Forward Network for NexoraLM.
Implements Pre-Layer Normalization architecture with residual connections.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from config.config import ModelConfig
from model.attention import CausalMultiHeadAttention


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network:
    
        FFN(x) = Dropout(Linear_2(GELU(Linear_1(x))))
        
    Linear_1 expands embed_dim to hidden_dim (default 4 * embed_dim).
    Linear_2 projects hidden_dim back to embed_dim.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.ffn_mult * config.embed_dim
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FeedForwardNetwork.
        
        Args:
            x (torch.Tensor): Input activations of shape (B, T, embed_dim).
            
        Returns:
            torch.Tensor: Transformed activations of shape (B, T, embed_dim).
        """
        # (B, T, embed_dim) -> (B, T, hidden_dim)
        x = self.fc1(x)
        x = self.act(x)
        # (B, T, hidden_dim) -> (B, T, embed_dim)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Pre-Layer-Normalization Transformer Decoder Block:
    
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
        
    In Pre-LN Transformers (modern GPT style), normalization is applied
    before the sub-layers, which stabilizes gradient propagation during deep training.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForwardNetwork(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for TransformerDecoderBlock.
        
        Args:
            x (torch.Tensor): Input activations of shape (B, T, embed_dim).
            return_attention_weights (bool): Whether to return attention maps.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output activations of shape (B, T, embed_dim)
                - Optional attention weights of shape (B, H, T, T)
        """
        # Pre-LN Causal Multi-Head Self-Attention with Residual Connection
        norm_x = self.ln_1(x)
        attn_out, attn_weights = self.attn(norm_x, return_attention_weights=return_attention_weights)
        x = x + attn_out

        # Pre-LN Feed-Forward Network with Residual Connection
        x = x + self.ffn(self.ln_2(x))

        return x, attn_weights

