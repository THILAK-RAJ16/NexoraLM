"""
PyTorch Dataset and DataLoader pipeline for NexoraLM.
Creates input sequences (x) and shifted target sequences (y) for next-token prediction.
"""

from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.tokenizer import CharacterTokenizer


class TextDataset(Dataset):
    """
    Autoregressive Next-Token Prediction Dataset.
    
    Given a sequence of token IDs of length L, this dataset creates chunks of
    length `context_length + 1` and produces pairs:
        x = sequence[i : i + context_length]
        y = sequence[i + 1 : i + context_length + 1]
        
    Example:
        Tokens: [10, 11, 12, 13, 14], context_length = 4
        x: [10, 11, 12, 13]
        y: [11, 12, 13, 14]
    """

    def __init__(
        self,
        token_ids: Union[List[int], torch.Tensor],
        context_length: int = 256,
        stride: Optional[int] = None
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            token_ids (Union[List[int], torch.Tensor]): List or 1D Tensor of token IDs.
            context_length (int): Fixed sequence length fed into the Transformer.
            stride (Optional[int]): Stride step between successive chunks. Defaults to context_length (non-overlapping).
        """
        if context_length < 1:
            raise ValueError(f"context_length must be positive, got {context_length}")
            
        if isinstance(token_ids, list):
            self.data = torch.tensor(token_ids, dtype=torch.long)
        elif isinstance(token_ids, torch.Tensor):
            self.data = token_ids.to(dtype=torch.long).contiguous().flatten()
        else:
            raise TypeError(f"Unsupported data type for token_ids: {type(token_ids)}")
            
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        
        # Calculate number of valid samples
        # Each sample requires context_length + 1 tokens
        seq_len = len(self.data)
        if seq_len < self.context_length + 1:
            self.num_samples = 0
        else:
            self.num_samples = (seq_len - self.context_length - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (x, y) input and shifted target pair.
        
        Args:
            idx (int): Sample index.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (x, y) where x is input and y is target.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples.")
            
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length
        
        # x is tokens from start_idx to end_idx - 1 (length = context_length)
        x = self.data[start_idx:end_idx]
        # y is shifted by 1: tokens from start_idx + 1 to end_idx (length = context_length)
        y = self.data[start_idx + 1 : end_idx + 1]
        
        return x, y


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Convenience wrapper to build a PyTorch DataLoader.
    
    Args:
        dataset (Dataset): Instance of TextDataset.
        batch_size (int): Batch size. Default 16.
        shuffle (bool): Whether to shuffle the data. Default True.
        drop_last (bool): Whether to drop incomplete batches. Default True (if dataset has enough samples).
        num_workers (int): Subprocesses for data loading. Default 0.
        
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    # If dataset has fewer samples than batch_size, avoid dropping all samples if drop_last=True
    if len(dataset) < batch_size and drop_last:
        drop_last = False
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

