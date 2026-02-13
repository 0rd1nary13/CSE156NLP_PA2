"""Dataset definitions for classification and language modeling tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from tokenizer import SimpleTokenizer


class SpeechesClassificationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Classification dataset for politician prediction."""

    def __init__(self, tokenizer: "SimpleTokenizer", file_path: str) -> None:
        """Load label-text pairs from a TSV file."""
        self.tokenizer = tokenizer
        self.samples: list[tuple[int, str]] = []

        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with target_path.open("r", encoding="utf-8") as file:
            for line in file:
                label, text = line.strip().split("\t")
                if label not in ("0", "1", "2"):
                    raise ValueError(f"Invalid label: {label}")
                if len(text.strip()) == 0:
                    continue
                self.samples.append((int(label), text))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return token ids and label tensor for one sample."""
        label, text = self.samples[index]
        input_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_ids, label_tensor


class LanguageModelingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Autoregressive language modeling dataset."""

    def __init__(self, tokenizer: "SimpleTokenizer", text: str, block_size: int) -> None:
        """Encode text and create fixed-length next-token prediction windows."""
        self.tokenizer = tokenizer
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        """Return total number of windows."""
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return one input-target window pair."""
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y