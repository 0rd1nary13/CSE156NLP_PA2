# CSE 156 PA2 Report: Transformer Blocks

## 1. Overview

This project implements two transformer-based models from scratch in PyTorch on a political speech dataset:

1. **Part 1 (Encoder + Classifier):** Train a transformer encoder jointly with a feedforward classifier to predict which politician delivered a speech segment (3-way classification).
2. **Part 2 (Decoder LM):** Train a GPT-like masked transformer decoder for autoregressive language modeling and report perplexity on held-out politician-specific test sets.

The implementation follows the assignment's default hyperparameters and uses the provided data splits.

## 2. Dataset and Task Setup

The dataset contains speeches from three politicians:

- `0`: Barack Obama
- `1`: George W. Bush
- `2`: George H. Bush

### Classification task (Part 1)

- Train: `train_CLS.tsv`
- Test: `test_CLS.tsv`

### Language modeling task (Part 2)

- Train: `train_LM.txt`
- Test: `test_LM_obama.txt`, `test_LM_wbush.txt`, `test_LM_hbush.txt`

## 3. Model Architecture

### 3.1 Encoder for Classification

The encoder uses:

- Token embeddings + absolute positional embeddings
- `n_layer=4` transformer encoder blocks
- Multi-head self-attention (`n_head=2`)
- Residual connections + LayerNorm + feedforward sublayers

For classification, token-level encoder outputs are mean-pooled across the sequence dimension. The pooled vector is passed to a 1-hidden-layer feedforward classifier:

- Input: `64`
- Hidden: `100` with ReLU
- Output: `3` logits

### 3.2 Decoder for Language Modeling

The decoder uses:

- Token embeddings + absolute positional embeddings
- `n_layer=4` masked transformer decoder blocks
- Causal self-attention (`n_head=2`) to prevent future-token leakage
- Feedforward hidden size `100` with ReLU
- Linear LM head to vocabulary logits

Training objective is token-level cross-entropy for next-word prediction.

## 4. Hyperparameters

The following base settings were used:

- `batch_size = 16`
- `block_size = 32`
- `learning_rate = 1e-3`
- `n_embd = 64`
- `n_head = 2`
- `n_layer = 4`
- Classification epochs: `15`
- LM iterations: `500`
- Perplexity checkpoints every `100` iterations

## 5. Experimental Results

All results below come from the recorded runs:

- `python3 main.py --part part1`
- `python3 main.py --part part2`

### 5.1 Part 1: Test Accuracy by Epoch

| Epoch | Test Accuracy (%) |
| ---: | ---: |
| 1 | 44.80 |
| 2 | 45.60 |
| 3 | 58.40 |
| 4 | 60.13 |
| 5 | 73.73 |
| 6 | 76.40 |
| 7 | 81.20 |
| 8 | 82.53 |
| 9 | 82.93 |
| 10 | 80.67 |
| 11 | 86.13 |
| 12 | 86.00 |
| 13 | 84.00 |
| 14 | 86.27 |
| 15 | **87.60** |

Final Part 1 metrics:

- **Final test accuracy:** **87.60%**
- **Encoder parameter count:** **477,584**

The classification accuracy rises steadily after early epochs and reaches the expected 80s range, indicating successful joint training of encoder and classifier.

### 5.2 Part 2: Decoder Perplexity

Training perplexity checkpoints:

| Iteration | Train Perplexity |
| ---: | ---: |
| 100 | 562.61 |
| 200 | 409.33 |
| 300 | 290.71 |
| 400 | 218.84 |
| 500 | 164.37 |

Final Part 2 metrics:

- **Final train perplexity:** **165.10**
- **Decoder parameter count:** **839,894**

Perplexity on held-out politician test sets:

| Test Set | Perplexity |
| --- | ---: |
| Obama | 356.64 |
| W. Bush | 465.40 |
| H. Bush | 376.79 |

The decoder shows consistent learning over 500 iterations (monotonic reduction in train perplexity checkpoints). Test perplexities are higher than train perplexity, which is expected due to distribution shift and limited pretraining scale.

## 6. Discussion

1. **Encoder-classifier performance:** The final accuracy of 87.60% is within the expected range, showing that the learned contextual representations are effective for speaker classification.
2. **Decoder behavior:** The decoder reaches train perplexity in the high-100 range and test perplexities in the 300s to 400s, consistent with assignment expectations for this data and training budget.
3. **Differences across politician test sets:** W. Bush has the highest perplexity, while Obama is lowest among the three test sets. This likely reflects differences in lexical style, topic distribution, and overlap with the training LM corpus.
4. **Scale limitations:** With only 500 update steps and a relatively small model/context, the LM remains far from modern large-scale language model performance, but clearly captures useful structure.

## 7. Attention Sanity Check Note

The codebase includes attention-map utilities (`utilities.py`) for row-normalization checks and visualization. These plots can be generated with:

- `python3 main.py --part part1 --sanity-check`
- `python3 main.py --part part2 --sanity-check`

In a final PDF submission, include 1-2 attention heatmaps and short observations (for example, normalization validity and causal masking behavior).

## 8. Conclusion

This project successfully implements transformer encoder and decoder blocks from scratch and evaluates them on both classification and language modeling objectives. The final results match expected ranges:

- Strong classification accuracy in the high 80s
- Decoder perplexities in expected train/test bands

Overall, the experiments demonstrate effective training dynamics and sensible generalization behavior under the assignment's constrained data and compute settings.
