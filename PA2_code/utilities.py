"""Utility helpers for attention sanity checks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor


class Utilities:
    """Helper utilities for sanity-checking transformer attention."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        """Store tokenizer and model references."""
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence: str, block_size: int) -> None:
        """Visualize per-layer attention and check row normalization."""
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        attn_maps = self._extract_attention_maps(input_tensor)

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap="hot", interpolation="nearest")
            ax.xaxis.tick_top()
            fig.colorbar(cax, ax=ax)
            plt.title(f"Attention Map {j + 1}")

            # Save the plot
            plt.savefig(f"attention_map_{j + 1}.png")

            # Show the plot
            plt.show()

    def _extract_attention_maps(self, input_tensor: Tensor) -> list[Tensor]:
        """Extract attention maps from encoder or decoder-style forward APIs."""
        try:
            model_output = self.model(input_tensor, return_attn=True)
        except TypeError:
            model_output = self.model(input_tensor)

        if (
            isinstance(model_output, tuple)
            and len(model_output) == 3
            and isinstance(model_output[2], list)
        ):
            return model_output[2]
        if (
            isinstance(model_output, tuple)
            and len(model_output) == 2
            and isinstance(model_output[1], list)
        ):
            return model_output[1]
        raise ValueError("Model forward output does not include attention maps.")
