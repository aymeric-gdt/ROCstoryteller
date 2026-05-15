"""Story generation from a prompt using temperature sampling."""

import argparse
import torch
import torch.nn.functional as F

from model import StoryLSTM
from tokenizer import BPETokenizer


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def generate(
    model: StoryLSTM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    num_sentences: int = 5,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate a story continuation from a prompt.

    Generation stops after `num_sentences` <EOS> tokens are produced,
    or max_new_tokens is reached.
    """
    model.eval()
    model = model.to(device)

    full_prompt = f"<SOS> {prompt.strip()}"
    prompt_ids = tokenizer.encode(full_prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Feed prompt to prime hidden state
    _, (h, c) = model(input_tensor)

    current_id = prompt_ids[-1]
    generated_ids = list(prompt_ids)
    eos_count = 0

    for _ in range(max_new_tokens):
        emb = model.embedding(torch.tensor([[current_id]], device=device))
        lstm_out, (h, c) = model.lstm(emb, (h, c))
        logits = model.head(lstm_out)

        logits = logits[0, 0] / temperature
        probs = F.softmax(logits, dim=-1)
        current_id = torch.multinomial(probs, 1).item()

        generated_ids.append(current_id)

        if current_id == tokenizer.EOS_ID:
            eos_count += 1
            if eos_count >= num_sentences:
                break

    return tokenizer.decode(generated_ids, skip_special=True)


def main():
    parser = argparse.ArgumentParser(description="Generate a story from a prompt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--num-sentences", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
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

    story = generate(
        model, tok, args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        num_sentences=args.num_sentences,
        device=device,
    )
    print(story)


if __name__ == "__main__":
    main()
