"""
Character-Level Tokenizer for NexoraLM.
Built entirely from scratch without external tokenization libraries.
"""

import json
import os
from typing import Dict, List, Optional, Union


class CharacterTokenizer:
    """
    A character-level tokenizer with support for special tokens,
    dynamic vocabulary construction, serialization, encoding, and decoding.
    
    Special Tokens:
        <pad>: Padding token (index 0)
        <unk>: Unknown token (index 1)
        <bos>: Beginning of sequence token (index 2)
        <eos>: End of sequence token (index 3)
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    DEFAULT_SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        """
        Initialize the tokenizer.
        
        Args:
            special_tokens (Optional[List[str]]): List of special token strings.
                Defaults to ['<pad>', '<unk>', '<bos>', '<eos>'].
        """
        self.special_tokens = list(special_tokens or self.DEFAULT_SPECIAL_TOKENS)
        
        # Mappings: token (str) <-> id (int)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        self.pad_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        
        self._initialize_special_tokens()

    def _initialize_special_tokens(self) -> None:
        """Register special tokens at the beginning of the vocabulary."""
        self.token_to_id.clear()
        self.id_to_token.clear()
        
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
        self.pad_token_id = self.token_to_id.get(self.PAD_TOKEN)
        self.unk_token_id = self.token_to_id.get(self.UNK_TOKEN)
        self.bos_token_id = self.token_to_id.get(self.BOS_TOKEN)
        self.eos_token_id = self.token_to_id.get(self.EOS_TOKEN)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.token_to_id)

    def __len__(self) -> int:
        return self.vocab_size

    def build_vocab(self, text: str) -> None:
        """
        Dynamically build vocabulary from training text corpus.
        Collects all unique characters, sorts them deterministically,
        and assigns incremental IDs starting after the special tokens.
        
        Args:
            text (str): Source training text.
        """
        self._initialize_special_tokens()
        
        # Find unique characters in text that are not already special tokens
        unique_chars = sorted(list(set(text)))
        next_id = len(self.token_to_id)
        
        for char in unique_chars:
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Convert a string of text into a sequence of token IDs.
        Unrecognized characters are mapped to the <unk> token ID.
        
        Args:
            text (str): Input text string.
            add_bos (bool): Whether to prepend the <bos> token. Default False.
            add_eos (bool): Whether to append the <eos> token. Default False.
            
        Returns:
            List[int]: List of integer token IDs.
        """
        token_ids: List[int] = []
        
        if add_bos and self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)
            
        unk_id = self.unk_token_id if self.unk_token_id is not None else 1
        
        for char in text:
            token_ids.append(self.token_to_id.get(char, unk_id))
            
        if add_eos and self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)
            
        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False
    ) -> str:
        """
        Convert a sequence of token IDs back into a string.
        
        Args:
            token_ids (List[int]): List of integer token IDs.
            skip_special_tokens (bool): Whether to omit special tokens (<pad>, <unk>, etc.). Default False.
            
        Returns:
            str: Decoded text string.
        """
        chars: List[str] = []
        special_set = set(self.special_tokens) if skip_special_tokens else set()
        
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if skip_special_tokens and token in special_set:
                    continue
                chars.append(token)
            else:
                # Fallback for out-of-vocab IDs
                if not skip_special_tokens:
                    chars.append(self.UNK_TOKEN)
                    
        return "".join(chars)

    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary mappings and metadata to a JSON file.
        
        Args:
            filepath (str): Path to write the JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary mappings and metadata from a JSON file.
        
        Args:
            filepath (str): Path to existing vocabulary JSON file.
            
        Raises:
            FileNotFoundError: If the vocabulary file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at: {filepath}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.special_tokens = data.get("special_tokens", self.DEFAULT_SPECIAL_TOKENS)
        self.token_to_id = data["token_to_id"]
        # Convert JSON string keys back to int
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        
        self.pad_token_id = self.token_to_id.get(self.PAD_TOKEN)
        self.unk_token_id = self.token_to_id.get(self.UNK_TOKEN)
        self.bos_token_id = self.token_to_id.get(self.BOS_TOKEN)
        self.eos_token_id = self.token_to_id.get(self.EOS_TOKEN)

