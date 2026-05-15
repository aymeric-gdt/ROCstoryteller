"""Interactive story generation REPL."""

import argparse
import torch

from generate import generate, get_device
from model import StoryLSTM
from tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Interactive story generation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/tokenizer.json")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    tok = BPETokenizer.load(args.tokenizer_path)
    model = StoryLSTM(
        vocab_size=tok.vocab_size_actual,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        pad_idx=tok.PAD_ID,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params/1e6:.1f}M params)")
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
            temperature=args.temperature,
            max_new_tokens=150,
            num_sentences=5,
            device=device,
        )
        print(f"\n{story}\n")


if __name__ == "__main__":
    main()
