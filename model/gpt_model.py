"""
Complete Decoder-Only GPT-Style Language Model for NexoraLM.
Integrates Token & Positional Embeddings, TransformerDecoderBlocks, Final LayerNorm,
and Language-Model Output Head with Next-Token Loss calculation.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig
from model.embeddings import TransformerEmbeddings
from model.transformer_block import TransformerDecoderBlock


class GPTLanguageModel(nn.Module):
    """
    Decoder-Only Generative Pretrained Transformer (GPT) Language Model.
    
    Architecture:
        Input Token IDs (B, T)
                │
        [TransformerEmbeddings: Token + Positional]
                │  Activations: (B, T, embed_dim)
        [Stack of N x TransformerDecoderBlock]
                │  Activations: (B, T, embed_dim)
        [Final LayerNorm]
                │
        [LM Head (Linear projection to vocab_size)]
                │
        Logits: (B, T, vocab_size)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the complete GPT language model.
        
        Args:
            config (ModelConfig): Model hyperparameter configuration.
        """
        super().__init__()
        self.config = config
        
        # 1. Combined Token and Positional Embeddings
        self.embeddings = TransformerEmbeddings(config)
        
        # 2. Stack of N Pre-LN Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.num_layers)
        ])
        
        # 3. Final Layer Normalization
        self.ln_f = nn.LayerNorm(config.embed_dim)
        
        # 4. Language Modeling Output Head
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # 5. Weight Tying (Weight Sharing between token embeddings and LM head)
        if config.tie_weights:
            self.lm_head.weight = self.embeddings.token_embeddings.weight
            
        # 6. Apply GPT-style weight initialization
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to residual projections (GPT-2 standard)
        # Scaled by 1 / sqrt(2 * num_layers) to prevent variance growth through deep residual stacks
        for pn, p in self.named_parameters():
            if pn.endswith("out_projection.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using normal distribution with mean 0.0 and std 0.02,
        biases to zero, and LayerNorm weights to 1.0.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "padding_idx") and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Compute total number of trainable parameters in the model.
        
        Args:
            non_embedding (bool): If True, subtract positional embeddings
                (and token embeddings if untied) to measure core Transformer parameters.
                
        Returns:
            int: Number of trainable parameters.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.embeddings.positional_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for next-token prediction language modeling.
        
        Args:
            input_ids (torch.Tensor): Tensor of token IDs of shape (batch_size, sequence_length).
            targets (Optional[torch.Tensor]): Target token IDs of shape (batch_size, sequence_length).
                Note: The dataset pipeline already supplies aligned next-token sequences
                (e.g., target[i] is the target token following input_ids[i]).
                
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - logits: Tensor of shape (batch_size, sequence_length, vocab_size)
                - loss: Scalar cross-entropy loss tensor if targets are provided, else None.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum context length {self.config.context_length}"
            )
            
        # 1. Embeddings: (B, T) -> (B, T, embed_dim)
        x = self.embeddings(input_ids)
        
        # 2. Sequential pass through all Transformer decoder blocks: (B, T, embed_dim)
        for block in self.blocks:
            x, _ = block(x)
            
        # 3. Final LayerNorm: (B, T, embed_dim)
        x = self.ln_f(x)
        
        # 4. Language modeling projection head: (B, T, embed_dim) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        
        # 5. Loss calculation (if targets supplied)
        loss = None
        if targets is not None:
            # Flatten across batch and sequence dimensions for CrossEntropyLoss:
            # Logits: (B * T, vocab_size)
            # Targets: (B * T)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.ignore_index
            )
            
        return logits, loss

