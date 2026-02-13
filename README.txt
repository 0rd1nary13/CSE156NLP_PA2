Project layout:
- `PA2_code/` contains the implementation.
- `PA2_code/speechesdataset/` contains all data files for both tasks.

Requirements:
- Python 3.9+
- PyTorch
- NLTK
- Matplotlib

Install dependencies (example):
- `python3 -m pip install torch nltk matplotlib`

Run from `PA2_code/`:
- Both parts: `python3 main.py`
- Part 1 only (encoder + classifier): `python3 main.py --part part1`
- Part 2 only (decoder LM): `python3 main.py --part part2`
- Optional attention sanity plots: add `--sanity-check`

Notes:
- Hyperparameters match the base settings from `main.py`.
- Part 1 prints test accuracy after each of 15 epochs and final encoder parameter count.
- Part 2 prints train perplexity every 100 iterations, final train perplexity, test perplexities by politician, and decoder parameter count.
