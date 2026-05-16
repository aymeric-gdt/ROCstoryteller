"""Export a trained StoryLSTM to ONNX for the web demo.

Usage:
    # Export the sentence model (default)
    python web/export_onnx.py

    # Export the paragraph model
    python web/export_onnx.py --mode paragraph
"""

import argparse
import json
import shutil
import torch
from pathlib import Path

from model import StoryLSTM
from tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Export StoryLSTM to ONNX for web demo")
    parser.add_argument("--mode", type=str, default="sentence",
                        choices=["sentence", "paragraph"],
                        help="Which trained model to export")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--web-dir", type=str, default="web")
    args = parser.parse_args()

    base = Path(args.checkpoint_dir) / args.mode
    ckpt_path = base / "best_model.pt"
    tok_path = base / "tokenizer.json"
    web_dir = Path(args.web_dir)
    web_dir.mkdir(exist_ok=True)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train first with: python train.py --mode {args.mode}")
        return
    if not tok_path.exists():
        print(f"Tokenizer not found: {tok_path}")
        return

    print(f"Loading model from {ckpt_path}...")
    tok = BPETokenizer.load(str(tok_path))
    vocab_size = tok.vocab_size_actual
    print(f"  Vocab size: {vocab_size}")

    model = StoryLSTM(
        vocab_size=vocab_size, embed_dim=256, hidden_dim=512,
        num_layers=2, pad_idx=tok.PAD_ID,
    )
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    # Export ONNX
    onnx_path = web_dir / "storylstm.onnx"
    print(f"Exporting ONNX to {onnx_path}...")
    dummy_input = torch.zeros(1, 1, dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits", "h_n", "c_n"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "logits": {1: "seq_len"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    size_mb = onnx_model.ByteSize() / 1024 / 1024
    print(f"  ONNX valid — {size_mb:.1f} MB")

    # Export vocab as JSON list (index → token)
    vocab = tok._tokenizer.get_vocab()
    vocab_list = [""] * len(vocab)
    for token, idx in vocab.items():
        vocab_list[idx] = token

    vocab_path = web_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab_list, f, ensure_ascii=False)
    print(f"Vocab saved: {len(vocab_list)} tokens -> {vocab_path}")

    # Export special token config
    config = {
        "PAD": tok.PAD_ID,
        "UNK": tok.UNK_ID,
        "SOS": tok.SOS_ID,
        "EOS": tok.EOS_ID,
    }
    config_path = web_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    print(f"Config saved: {config} -> {config_path}")

    print(f"\nDone. Web demo ready for mode='{args.mode}'.")
    print(f"  cd web && python serve.py")


if __name__ == "__main__":
    main()
