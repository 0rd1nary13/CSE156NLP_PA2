"""Train and evaluate encoder/classifier and decoder language model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset import LanguageModelingDataset, SpeechesClassificationDataset
from tokenizer import SimpleTokenizer
from transformer import SpeechClassifier, TransformerDecoderLM, TransformerEncoder
from utilities import Utilities

seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 100 and output size of 3.
n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def set_seed(seed_value: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def load_texts(directory: Path) -> list[str]:
    """Load training texts for tokenizer vocabulary construction."""
    texts: list[str] = []
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue
        if "test" in file_path.name:  ## don't "read test files"
            continue
        with file_path.open("r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts


def collate_batch(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Collate a classification batch into padded tensors."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(
        padded_sequences,
        (0, max(0, block_size - padded_sequences.shape[1])),
        "constant",
        0,
    )
    label_tensor = torch.stack(labels)
    return padded_sequences, label_tensor


def compute_classifier_accuracy(
    classifier: SpeechClassifier,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
) -> float:
    """Compute classification accuracy."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = classifier(xb)
            predicted = outputs.argmax(dim=1)
            total_correct += int((predicted == yb).sum().item())
            total_samples += yb.size(0)
    classifier.train()
    return 100.0 * total_correct / max(1, total_samples)


def compute_perplexity(
    decoder_model: TransformerDecoderLM,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    max_eval_iters: int = 100,
) -> float:
    """Compute decoder perplexity from cross-entropy loss."""
    decoder_model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = decoder_model(xb, yb)
            if loss is None:
                raise ValueError("Decoder loss is None during perplexity computation.")
            losses.append(loss.item())
            if len(losses) >= max_eval_iters:
                break
    decoder_model.train()
    mean_loss = torch.tensor(losses).mean()
    return torch.exp(mean_loss).item()


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def read_text(file_path: Path) -> str:
    """Read an entire UTF-8 text file."""
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()


def train_part1(tokenizer: SimpleTokenizer, data_dir: Path, run_sanity_check: bool) -> None:
    """Train encoder + classifier and report per-epoch test accuracy."""
    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        ff_hidden_dim=n_hidden,
    ).to(device)
    classifier = SpeechClassifier(
        encoder=encoder,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataset = SpeechesClassificationDataset(tokenizer, str(data_dir / "train_CLS.tsv"))
    test_dataset = SpeechesClassificationDataset(tokenizer, str(data_dir / "test_CLS.tsv"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=False,
    )

    print("\n=== Part 1: Encoder + Classifier ===")
    for epoch in range(epochs_CLS):
        classifier.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        test_accuracy = compute_classifier_accuracy(classifier, test_loader)
        print(
            f"Epoch {epoch + 1:02d}/{epochs_CLS}: "
            f"train_loss={avg_loss:.4f}, test_accuracy={test_accuracy:.2f}%"
        )

    final_accuracy = compute_classifier_accuracy(classifier, test_loader)
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Encoder parameter count: {count_parameters(encoder)}")

    if run_sanity_check:
        print("Running encoder attention sanity check...")
        Utilities(tokenizer, encoder).sanity_check(
            sentence="We must act, knowing that our work will be imperfect.",
            block_size=block_size,
        )


def build_lm_loader(tokenizer: SimpleTokenizer, text_file: Path, shuffle: bool) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create a language-modeling dataloader from a text file."""
    dataset = LanguageModelingDataset(tokenizer, read_text(text_file), block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_part2(tokenizer: SimpleTokenizer, data_dir: Path, run_sanity_check: bool) -> None:
    """Train decoder LM and report perplexities during and after training."""
    decoder = TransformerDecoderLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        ff_hidden_dim=n_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    train_loader = build_lm_loader(tokenizer, data_dir / "train_LM.txt", shuffle=True)
    print("\n=== Part 2: Decoder Language Model ===")
    print("Perplexity checkpoints:")

    for iteration, (xb, yb) in enumerate(train_loader):
        if iteration >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        _, loss = decoder(xb, yb)
        if loss is None:
            raise ValueError("Decoder returned None loss during training.")
        loss.backward()
        optimizer.step()

        if (iteration + 1) % eval_interval == 0:
            train_ppl = compute_perplexity(decoder, train_loader, max_eval_iters=eval_iters)
            print(f"Iteration {iteration + 1:03d}/{max_iters}: train_perplexity={train_ppl:.2f}")

    final_train_ppl = compute_perplexity(decoder, train_loader, max_eval_iters=eval_iters)
    print(f"Final train perplexity: {final_train_ppl:.2f}")
    print(f"Decoder parameter count: {count_parameters(decoder)}")

    obama_loader = build_lm_loader(tokenizer, data_dir / "test_LM_obama.txt", shuffle=False)
    wbush_loader = build_lm_loader(tokenizer, data_dir / "test_LM_wbush.txt", shuffle=False)
    hbush_loader = build_lm_loader(tokenizer, data_dir / "test_LM_hbush.txt", shuffle=False)
    obama_ppl = compute_perplexity(decoder, obama_loader, max_eval_iters=eval_iters)
    wbush_ppl = compute_perplexity(decoder, wbush_loader, max_eval_iters=eval_iters)
    hbush_ppl = compute_perplexity(decoder, hbush_loader, max_eval_iters=eval_iters)
    print(f"Test perplexity (Obama): {obama_ppl:.2f}")
    print(f"Test perplexity (W. Bush): {wbush_ppl:.2f}")
    print(f"Test perplexity (H. Bush): {hbush_ppl:.2f}")

    if run_sanity_check:
        print("Running decoder attention sanity check...")
        Utilities(tokenizer, decoder).sanity_check(
            sentence="Freedom and fear are at war.",
            block_size=block_size,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CSE156 PA2 transformer experiments.")
    parser.add_argument(
        "--part",
        choices=["all", "part1", "part2"],
        default="all",
        help="Select which assignment part to run.",
    )
    parser.add_argument(
        "--data-dir",
        default="speechesdataset",
        help="Path to dataset directory, relative to this file by default.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Generate attention map visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for training and evaluation."""
    args = parse_args()
    set_seed(seed)

    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    print(f"Using device: {device}")
    print("Loading data and creating tokenizer ...")
    tokenizer = SimpleTokenizer(" ".join(load_texts(data_dir)))  # create a tokenizer from the data
    print(f"Vocabulary size is {tokenizer.vocab_size}")

    if args.part in ("all", "part1"):
        train_part1(tokenizer, data_dir, run_sanity_check=args.sanity_check)

    if args.part in ("all", "part2"):
        train_part2(tokenizer, data_dir, run_sanity_check=args.sanity_check)


if __name__ == "__main__":
    main()
