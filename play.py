"""Interactive story generation REPL."""

import argparse
import torch

from generate import generate, get_device
from model import StoryLSTM
from tokenizer import BPETokenizer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Interactive story generation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--mode", type=str, default="sentence",
                        choices=["sentence", "paragraph"],
                        help="Wrapping mode (must match training mode)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (overrides --mode/--checkpoint-dir)")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer (overrides --mode/--checkpoint-dir)")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Resolve checkpoint and tokenizer paths
    base = Path(args.checkpoint_dir) / args.mode
    ckpt_path = args.checkpoint or str(base / "best_model.pt")
    tok_path = args.tokenizer_path or str(base / "tokenizer.json")

    tok = BPETokenizer.load(tok_path)
    model = StoryLSTM(
        vocab_size=tok.vocab_size_actual,
        embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        pad_idx=tok.PAD_ID,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params/1e6:.1f}M params) — mode: {args.mode}")
    print("Type a prompt (or 'quit' to exit):\n")

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        story = generate(
            model, tok, prompt,
            mode=args.mode,
            temperature=args.temperature,
            max_new_tokens=150,
            num_sentences=5,
            device=device,
        )
        print(f"\n{story}\n")


if __name__ == "__main__":
    main()
