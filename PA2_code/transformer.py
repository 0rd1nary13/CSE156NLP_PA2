"""Transformer encoder/decoder components implemented from scratch."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfAttentionHead(nn.Module):
    """A single self-attention head."""

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        causal: bool,
    ) -> None:
        """Initialize projection layers and optional causal mask."""
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.causal = causal
        self.scale = head_size**-0.5
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute attention outputs and attention probabilities."""
        _, seq_len, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = self.tril[:seq_len, :seq_len] == 0
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = attn_probs @ v
        return out, attn_probs


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with output projection."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        causal: bool,
    ) -> None:
        """Create all heads and projection layers."""
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                    causal=causal,
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply all attention heads and return averaged attention map."""
        head_outputs: List[Tensor] = []
        attn_maps: List[Tensor] = []
        for head in self.heads:
            out, attn_probs = head(x)
            head_outputs.append(out)
            attn_maps.append(attn_probs)

        merged = torch.cat(head_outputs, dim=-1)
        projected = self.proj(merged)
        stacked_maps = torch.stack(attn_maps, dim=1)  # [B, n_head, T, T]
        avg_map = stacked_maps.mean(dim=1)  # [B, T, T]
        return projected, avg_map


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, n_embd: int, hidden_dim: int) -> None:
        """Initialize feedforward layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embd),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply feedforward transformation."""
        return self.net(x)


class EncoderBlock(nn.Module):
    """A single transformer encoder block."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        ff_hidden_dim: int,
    ) -> None:
        """Initialize attention, feedforward, and normalization modules."""
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            causal=False,
        )
        self.ffwd = FeedForward(n_embd=n_embd, hidden_dim=ff_hidden_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Run one encoder block and return block output plus attention map."""
        attn_out, attn_map = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, attn_map


class DecoderBlock(nn.Module):
    """A single transformer decoder block with masked self-attention."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        ff_hidden_dim: int,
    ) -> None:
        """Initialize masked attention, feedforward, and layer norms."""
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            causal=True,
        )
        self.ffwd = FeedForward(n_embd=n_embd, hidden_dim=ff_hidden_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Run one decoder block and return output plus attention map."""
        attn_out, attn_map = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, attn_map


class TransformerEncoder(nn.Module):
    """Transformer encoder for text classification."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        ff_hidden_dim: int = 100,
    ) -> None:
        """Initialize token/position embeddings and encoder blocks."""
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Encode token ids and return contextual embeddings with attention maps."""
        _, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)

        attn_maps: List[Tensor] = []
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map)
        x = self.ln_f(x)
        return x, attn_maps


class SpeechClassifier(nn.Module):
    """Feedforward classifier on top of encoder outputs."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        n_input: int,
        n_hidden: int,
        n_output: int,
    ) -> None:
        """Initialize the classifier head and hold encoder reference."""
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, idx: Tensor) -> Tensor:
        """Compute class logits for input token ids."""
        encoded, _ = self.encoder(idx)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


class TransformerDecoderLM(nn.Module):
    """GPT-like decoder for autoregressive language modeling."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        ff_hidden_dim: int = 100,
    ) -> None:
        """Initialize embeddings, decoder blocks, and LM head."""
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor | None] | Tuple[Tensor, Tensor | None, List[Tensor]]:
        """Return logits and optional loss, optionally including attention maps."""
        batch_size, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("Input sequence length exceeds decoder block size.")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)

        attn_maps: List[Tensor] = []
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, -1),
                targets.view(batch_size * seq_len),
            )

        if return_attn:
            return logits, loss, attn_maps
        return logits, loss