"""
Text preprocessing and cleaning utilities for NexoraLM.
Handles cleaning, normalization, file I/O, and dataset splitting.
"""

import os
import re
import unicodedata
from typing import Tuple


def clean_text(text: str) -> str:
    """
    Clean and normalize text for character-level language modeling.
    
    Operations:
    1. Normalize unicode characters to NFKC format.
    2. Normalize line breaks to standard Unix format (\n).
    3. Replace unusual / non-printable unicode control characters (preserving tab and newline).
    4. Collapse non-newline whitespace sequences into a single space while keeping paragraph breaks.
    
    Args:
        text (str): Raw input text string.
        
    Returns:
        str: Cleaned and normalized text.
    """
    if not text:
        return ""
    
    # 1. Unicode normalization (NFKC decomposes compatibility characters and composes canonical ones)
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Normalize Windows and legacy Mac newlines to '\n'
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 3. Filter out non-printable control characters, preserving '\n' and '\t'
    cleaned_chars = []
    for ch in text:
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C"):
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)
    
    # 4. Collapse multiple spaces / horizontal whitespace (excluding newline) into a single space
    text = re.sub(r"[^\S\n]+", " ", text)
    
    # 5. Limit excessive consecutive blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def load_text_file(filepath: str, encoding: str = "utf-8") -> str:
    """
    Load raw text from a file.
    
    Args:
        filepath (str): Absolute or relative path to the text file.
        encoding (str): Text encoding. Default is 'utf-8'.
        
    Returns:
        str: Raw text content.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source text file not found at: {filepath}")
    
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def save_text_file(content: str, filepath: str, encoding: str = "utf-8") -> None:
    """
    Save processed text to a destination file.
    
    Args:
        content (str): Text content to write.
        filepath (str): Destination file path.
        encoding (str): Text encoding. Default is 'utf-8'.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding=encoding) as f:
        f.write(content)


def split_data(text: str, train_ratio: float = 0.9) -> Tuple[str, str]:
    """
    Split text sequentially into training and validation sets without data leakage.
    For autoregressive text generation, sequential splitting preserves temporal/narrative
    coherence without shuffling across sequence boundaries.
    
    Args:
        text (str): Complete preprocessed text corpus.
        train_ratio (float): Fraction of data to assign to the training set. Default 0.9.
        
    Returns:
        Tuple[str, str]: (train_text, val_text)
        
    Raises:
        ValueError: If train_ratio is not strictly between 0 and 1.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be strictly between 0 and 1, got {train_ratio}")
    
    if not text:
        return "", ""
        
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    return train_text, val_text


def prepare_and_split_corpus(
    raw_filepath: str,
    processed_dir: str = "data/processed",
    train_ratio: float = 0.9
) -> Tuple[str, str, str]:
    """
    End-to-end dataset preprocessing pipeline:
    1. Loads raw file.
    2. Cleans text.
    3. Splits into train and validation sets sequentially.
    4. Persists processed datasets into processed_dir.
    
    Returns:
        Tuple[str, str, str]: (train_text, val_text, train_path, val_path) paths and content
    """
    raw_text = load_text_file(raw_filepath)
    cleaned = clean_text(raw_text)
    train_text, val_text = split_data(cleaned, train_ratio=train_ratio)
    
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.txt")
    val_path = os.path.join(processed_dir, "val.txt")
    
    save_text_file(train_text, train_path)
    save_text_file(val_text, val_path)
    
    return train_text, val_text, train_path, val_path

