"""Training loop for StoryLSTM on ROCStories."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset import load_rocstories, insert_sentence_tokens
from tokenizer import BPETokenizer
from data_utils import StoryDataset, collate_stories
from model import StoryLSTM


def get_device(device_arg: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(
    model: StoryLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pad_idx: int,
) -> float:
    """Run one epoch. If optimizer is None, runs in eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    mode = "train" if is_train else "val"

    bar = tqdm(loader, desc=mode)
    for inputs, targets in bar:
        inputs, targets = inputs.to(device), targets.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train StoryLSTM on ROCStories")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to train on (default: cpu)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    print("Loading dataset...")
    train_texts, test_texts = load_rocstories()

    print("Wrapping stories with <SOS>/<EOS>...")
    train_wrapped = [insert_sentence_tokens(s) for s in train_texts]
    test_wrapped = [insert_sentence_tokens(s) for s in test_texts]

    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(train_wrapped)
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    print(f"  Vocab size: {tokenizer.vocab_size_actual}")

    # Datasets
    train_ds = StoryDataset(train_wrapped, tokenizer, max_len=args.max_len)
    val_ds = StoryDataset(test_wrapped, tokenizer, max_len=args.max_len)

    # DataLoaders — bump num_workers on CUDA
    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_stories(b, pad_id=tokenizer.PAD_ID),
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_stories(b, pad_id=tokenizer.PAD_ID),
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = StoryLSTM(
        vocab_size=tokenizer.vocab_size_actual,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=tokenizer.PAD_ID,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, tokenizer.PAD_ID)
        val_loss = train_epoch(model, val_loader, None, device, tokenizer.PAD_ID)
        scheduler.step()

        print(f"Epoch {epoch+1:2d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(checkpoint_dir / "best_model.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), str(checkpoint_dir / f"model_epoch{epoch+1}.pt"))

    print(f"Done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
