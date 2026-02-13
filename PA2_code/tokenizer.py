"""Tokenizer utilities for the assignment."""

from __future__ import annotations

from collections.abc import Iterable

from nltk.tokenize import wordpunct_tokenize


class SimpleTokenizer:
    """A simple word-level tokenizer with a static vocabulary."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, text: str) -> None:
        """Initialize and build a vocabulary from text."""
        self.vocab: set[str] = set()
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.vocab_size: int = 0
        self.build_vocab(text)

    def build_vocab(self, text: str) -> None:
        """Build vocabulary lookup tables from text."""
        tokens = wordpunct_tokenize(text)
        self.vocab = set(tokens)
        sorted_vocab = sorted(self.vocab)
        self.stoi = {word: i for i, word in enumerate(sorted_vocab, start=2)}
        self.stoi[self.PAD_TOKEN] = 0
        self.stoi[self.UNK_TOKEN] = 1
        self.itos = {index: token for token, index in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids."""
        tokens = wordpunct_tokenize(text)
        return [self.stoi.get(token, self.stoi[self.UNK_TOKEN]) for token in tokens]

    def decode(self, indices: Iterable[int]) -> str:
        """Convert token ids back to a whitespace-joined string."""
        return " ".join(self.itos.get(index, self.UNK_TOKEN) for index in indices)
