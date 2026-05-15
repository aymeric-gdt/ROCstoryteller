# ROCstoryteller — LSTM Story Generation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Train a BPE-based LSTM on ROCStories. Input = 1-2 prompt sentences, output = the rest of the 5-sentence story.

**Architecture:** BPE Embedding(256) → 2×LSTM(512) → Linear(vocab_size). Trained with teacher forcing.

**Tech Stack:** PyTorch, HuggingFace `datasets` + `tokenizers`, `uv`. CPU training. Local checkpoints.

**Decisions locked:**
- Tokenization: BPE, 8000 tokens, special tokens `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
- Story format: `<SOS> sent1. <EOS> <SOS> sent2. <EOS> ... <SOS> sent5. <EOS>`
- Generation: temperature sampling (T=0.8 default), prompt mode only
- HW: CPU, batch_size=32, hidden=512, embed=256, 2 layers
- Checkpoints: local `checkpoints/` dir
- Model: ~10M params

---

## Implementation Tasks

### Task 1: Project scaffold + dependencies

**Objective:** Set up `pyproject.toml` with all dependencies.

**Files:**
- Create: `pyproject.toml`
- Create: `checkpoints/.gitkeep`

**Step 1: Write pyproject.toml**

```toml
[project]
name = "rocstoryteller"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "datasets>=2.14",
    "tokenizers>=0.14",
    "tqdm>=4.65",
]

[tool.uv]
dev-dependencies = []
```

**Step 2: Verify**

```bash
cd /home/aymeric/Documents/projets/ROCstoryteller
uv sync
python -c "import torch; import datasets; import tokenizers; print('OK')"
```

Expected: `OK`

---

### Task 2: BPETokenizer — train + encode/decode

**Objective:** Train a BPE tokenizer on ROCStories, save it, expose encode/decode with special tokens.

**Files:**
- Create: `tokenizer.py`

**Step 1: Write the BPETokenizer class**

```python
"""BPE tokenizer for ROCStories using HuggingFace tokenizers."""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import NFKC


class BPETokenizer:
    """Wraps a HuggingFace BPE tokenizer with special tokens for story generation."""

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    PAD_ID = 0
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self._tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self._tokenizer.normalizer = NFKC()
        self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Trainer: special tokens + vocab size
        self._trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=2,
        )

        # Post-processing: adds <SOS> at start, <EOS> at end
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single="<SOS> $A <EOS>",
            special_tokens=[
                ("<SOS>", self.SOS_ID),
                ("<EOS>", self.EOS_ID),
            ],
        )

    def train(self, texts: list[str]):
        """Train BPE on a list of raw stories."""
        self._tokenizer.train_from_iterator(texts, self._trainer)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (no <SOS>/<EOS> added)."""
        return self._tokenizer.encode(text).ids

    def encode_story(self, story: str) -> list[int]:
        """Encode a full story WITH <SOS>/<EOS> wrapping."""
        encoded = self._tokenizer.encode(story)
        return encoded.ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special)

    @property
    def vocab_size_actual(self) -> int:
        return self._tokenizer.get_vocab_size()

    def save(self, path: str):
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        inst = cls.__new__(cls)
        inst._tokenizer = Tokenizer.from_file(str(path))
        inst.vocab_size = inst._tokenizer.get_vocab_size()
        return inst
```

**Step 2: Verify tokenizer trains and round-trips**

```bash
cd /home/aymeric/Documents/projets/ROCstoryteller
uv run python -c "
from tokenizer import BPETokenizer
tok = BPETokenizer(8000)
tok.train(['The boy went to the store. He bought milk.'])
print('Vocab:', tok.vocab_size_actual)
ids = tok.encode_story('The boy went to the store. He bought milk.')
print('IDs:', ids)
print('Decoded:', tok.decode(ids))
"
```

Expected: `Vocab: > 4` (small vocab from 1 example), decoded text looks like the input with `<SOS>`/`<EOS>`.

---

### Task 3: Dataset loading + preprocessing

**Objective:** Load ROCStories via HuggingFace, add sentence-boundary tokens, prepare train/val splits.

**Files:**
- Modify: `dataset.py` (replace existing one-liner)

**Step 1: Write dataset.py**

```python
"""ROCStories dataset loading and preprocessing."""

from datasets import load_dataset


def load_rocstories() -> tuple[list[str], list[str]]:
    """
    Load ROCStories from HuggingFace.
    Returns (train_texts, test_texts) — raw story strings.
    """
    ds = load_dataset("mintujupally/ROCStories")
    train_texts = [row["text"] for row in ds["train"]]
    test_texts = [row["text"] for row in ds["test"]]
    return train_texts, test_texts


def insert_sentence_tokens(story: str) -> str:
    """
    Wrap each sentence with <SOS>/<EOS> and join.
    Input:  "Tom went to the store. He bought milk. He went home."
    Output: "<SOS> Tom went to the store. <EOS> <SOS> He bought milk. <EOS> <SOS> He went home. <EOS>"
    """
    sentences = [s.strip() for s in story.split(".") if s.strip()]
    parts = []
    for sent in sentences:
        parts.append(f"<SOS> {sent}. <EOS>")
    return " ".join(parts)
```

**Step 2: Verify**

```bash
uv run python -c "
from dataset import load_rocstories, insert_sentence_tokens
train, test = load_rocstories()
print(f'Train: {len(train)} stories, Test: {len(test)} stories')
story = train[0]
print('Raw:', story[:100], '...')
print('With tokens:', insert_sentence_tokens(story)[:150], '...')
"
```

Expected: `Train: 78528 stories, Test: 19633 stories`, raw and token-wrapped versions shown.

---

### Task 4: PyTorch Dataset + collation

**Objective:** Create a PyTorch Dataset that tokenizes stories and returns (input, target) pairs. Create a collate function for padding.

**Files:**
- Create: `data_utils.py`

**Step 1: Write data_utils.py**

```python
"""PyTorch Dataset and collation for ROCStories."""

import torch
from torch.utils.data import Dataset


class StoryDataset(Dataset):
    """Tokenizes stories into (input_ids, target_ids) pairs for teacher forcing."""

    def __init__(self, stories: list[str], tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        for story in stories:
            ids = tokenizer.encode_story(story)
            if len(ids) > max_len:
                ids = ids[:max_len]
            # input = all tokens except last, target = all tokens except first
            self.data.append((ids[:-1], ids[1:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, tgt = self.data[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_stories(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    """
    Pad a batch of (input, target) pairs to the same length.
    Returns (inputs_padded, targets_padded) — both [batch, max_seq_len].
    """
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)

    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_id)])
            tgt = torch.cat([tgt, torch.full((pad_len,), pad_id)])
        padded_inputs.append(inp.unsqueeze(0))
        padded_targets.append(tgt.unsqueeze(0))

    return torch.cat(padded_inputs), torch.cat(padded_targets)
```

**Step 2: Verify**

```bash
uv run python -c "
from dataset import load_rocstories, insert_sentence_tokens
from tokenizer import BPETokenizer
from data_utils import StoryDataset, collate_stories
import torch

train_texts, test_texts = load_rocstories()
# Wrap with sentence tokens, then train tokenizer
wrapped = [insert_sentence_tokens(s) for s in train_texts[:5000]]

tok = BPETokenizer(8000)
tok.train(wrapped)
print(f'Vocab size: {tok.vocab_size_actual}')

ds = StoryDataset(wrapped[:100], tok, max_len=128)
inp, tgt = ds[0]
print(f'Input shape: {inp.shape}, Target shape: {tgt.shape}')
print('Decoded input:', tok.decode(inp.tolist())[:120])

# Test collation
batch = [ds[i] for i in range(4)]
padded_in, padded_tgt = collate_stories(batch, pad_id=tok.PAD_ID)
print(f'Batch shape: {padded_in.shape}')
"
```

Expected: Vocab near 8000, shapes look reasonable, decoded text contains `<SOS>`/`<EOS>`.

---

### Task 5: LSTM Model

**Objective:** Define the StoryLSTM model.

**Files:**
- Create: `model.py`

**Step 1: Write model.py**

```python
"""StoryLSTM: BPE-level LSTM for story generation."""

import torch
import torch.nn as nn


class StoryLSTM(nn.Module):
    """Embedding → stacked LSTM → Linear → logits over vocabulary."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x: [batch, seq_len] token IDs
        Returns: logits [batch, seq_len, vocab_size], (h, c)
        """
        emb = self.embedding(x)                 # [B, T, E]
        lstm_out, (h, c) = self.lstm(emb, hidden)  # [B, T, H]
        lstm_out = self.dropout(lstm_out)
        logits = self.head(lstm_out)            # [B, T, V]
        return logits, (h, c)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize zero hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
```

**Step 2: Verify forward pass**

```bash
uv run python -c "
import torch
from model import StoryLSTM

model = StoryLSTM(vocab_size=8000, embed_dim=256, hidden_dim=512, num_layers=2)
x = torch.randint(0, 8000, (4, 50))  # batch=4, seq=50
logits, (h, c) = model(x)
print(f'Logits: {logits.shape}')  # [4, 50, 8000]
print(f'Hidden: {h.shape}, {c.shape}')  # [2, 4, 512]
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
"
```

Expected: `Logits: [4, 50, 8000]`, `Hidden: [2, 4, 512]`, `~10M params`.

---

### Task 6: Training loop

**Objective:** Full training loop with teacher forcing, loss tracking, checkpointing.

**Files:**
- Create: `train.py`

**Step 1: Write train.py**

```python
"""Training loop for StoryLSTM on ROCStories."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset import load_rocstories, insert_sentence_tokens
from tokenizer import BPETokenizer
from data_utils import StoryDataset, collate_stories
from model import StoryLSTM


def train(
    model: StoryLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: str = "checkpoints",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)

    best_val_loss = float("inf")
    Path(checkpoint_dir).mkdir(exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            # logits: [B, T, V], targets: [B, T]
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]")
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
                val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.3f}")

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Save periodic
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch{epoch+1}.pt")


def main():
    device = torch.device("cpu")
    print("Loading dataset...")
    train_texts, test_texts = load_rocstories()

    # Wrap with sentence tokens
    train_wrapped = [insert_sentence_tokens(s) for s in train_texts]
    test_wrapped = [insert_sentence_tokens(s) for s in test_texts]

    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=8000)
    tokenizer.train(train_wrapped)
    tokenizer.save("checkpoints/tokenizer.json")
    print(f"Tokenizer saved. Vocab size: {tokenizer.vocab_size_actual}")

    # Create datasets
    train_ds = StoryDataset(train_wrapped, tokenizer, max_len=128)
    val_ds = StoryDataset(test_wrapped, tokenizer, max_len=128)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=lambda b: collate_stories(b, pad_id=tokenizer.PAD_ID),
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        collate_fn=lambda b: collate_stories(b, pad_id=tokenizer.PAD_ID),
    )

    # Model
    model = StoryLSTM(
        vocab_size=tokenizer.vocab_size_actual,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        pad_idx=tokenizer.PAD_ID,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    train(model, train_loader, val_loader, epochs=20, device=device)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it starts (quick test with 1 epoch)**

```bash
# Quick sanity: train on small subset for 1 epoch
uv run python -c "
from dataset import load_rocstories, insert_sentence_tokens
from tokenizer import BPETokenizer
from data_utils import StoryDataset, collate_stories
from model import StoryLSTM
from train import train
from torch.utils.data import DataLoader
import torch

# Tiny test
texts, _ = load_rocstories()
tiny = [insert_sentence_tokens(s) for s in texts[:200]]
tok = BPETokenizer(8000)
tok.train(tiny)
ds = StoryDataset(tiny, tok, max_len=128)
loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_stories(b, tok.PAD_ID))
model = StoryLSTM(tok.vocab_size_actual, embed_dim=64, hidden_dim=128, num_layers=1, dropout=0.0, pad_idx=tok.PAD_ID)
train(model, loader, loader, epochs=1, lr=1e-3)
print('1-epoch sanity check PASSED')
"
```

Expected: Loss decreases, no crashes.

---

### Task 7: Story generation

**Objective:** Generate story continuations from a prompt using temperature sampling.

**Files:**
- Create: `generate.py`

**Step 1: Write generate.py**

```python
"""Story generation from a prompt using temperature sampling."""

import torch
import torch.nn.functional as F
from model import StoryLSTM
from tokenizer import BPETokenizer


@torch.no_grad()
def generate(
    model: StoryLSTM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    stop_at_eos: bool = True,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate a story continuation from a prompt.

    Args:
        model: Trained StoryLSTM
        tokenizer: Trained BPETokenizer
        prompt: Starting text (1-2 sentences), e.g. "Tom went to the store."
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature (0.7-1.0 recommended)
        stop_at_eos: Stop generation when <EOS> is produced
        device: torch device

    Returns:
        Full generated story (prompt + continuation).
    """
    model.eval()
    model = model.to(device)

    # Wrap prompt with <SOS> but NOT <EOS> — we'll generate that
    full_prompt = f"<SOS> {prompt.strip()}"
    prompt_ids = tokenizer.encode(full_prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Feed prompt through LSTM to get hidden state
    logits, (h, c) = model(input_tensor)

    # Start from last token of prompt
    current_id = prompt_ids[-1]
    generated_ids = list(prompt_ids)  # includes <SOS> + prompt tokens

    for _ in range(max_new_tokens):
        # Forward one token
        emb = model.embedding(torch.tensor([[current_id]], device=device))
        lstm_out, (h, c) = model.lstm(emb, (h, c))
        logits = model.head(lstm_out)  # [1, 1, V]

        # Temperature sampling
        logits = logits[0, 0] / temperature
        probs = F.softmax(logits, dim=-1)
        current_id = torch.multinomial(probs, 1).item()

        generated_ids.append(current_id)

        if stop_at_eos and current_id == tokenizer.EOS_ID:
            break

    # Decode
    full_text = tokenizer.decode(generated_ids, skip_special=True)
    return full_text


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Starting sentence(s)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    args = parser.parse_args()

    device = torch.device("cpu")

    tok = BPETokenizer.load(args.tokenizer)
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
        device=device,
    )
    print(story)


if __name__ == "__main__":
    main()
```

**Step 2: Verify generation works (after model is trained)**

```bash
uv run python generate.py --prompt "Tom went to the store" --temperature 0.8
```

Expected: Prints a 5-sentence story continuation. Quality depends on training.

---

### Task 8: Full training run

**Objective:** Train the full model on all 78k stories.

**Step 1: Run training**

```bash
cd /home/aymeric/Documents/projets/ROCstoryteller
uv run python train.py
```

**Step 2: Monitor loss**

Train loss should decrease steadily. Val loss should plateau around epoch 10-15. If val loss keeps decreasing, increase epochs to 30.

---

### Task 9: Interactive generation script

**Objective:** Small REPL to type prompts and get stories.

**Files:**
- Create: `play.py`

```python
"""Interactive story generation REPL."""

import torch
from generate import generate
from model import StoryLSTM
from tokenizer import BPETokenizer


def main():
    device = torch.device("cpu")
    tok = BPETokenizer.load("checkpoints/tokenizer.json")
    model = StoryLSTM(
        vocab_size=tok.vocab_size_actual,
        embed_dim=256, hidden_dim=512, num_layers=2, pad_idx=tok.PAD_ID,
    )
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print("Type a prompt (or 'quit' to exit):")

    while True:
        prompt = input("\n> ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        story = generate(model, tok, prompt, temperature=0.8, max_new_tokens=100, device=device)
        print(f"\n{story}")


if __name__ == "__main__":
    main()
```

---

## Pitfalls

1. **BPE tokenizer must be trained on wrapped stories** (with `<SOS>`/`<EOS>` already in the text) — otherwise the tokenizer won't have those tokens in its vocab.
2. **Loss masking**: `CrossEntropyLoss(ignore_index=PAD_ID)` is critical — without it, the model learns to predict padding tokens.
3. **Generation stops at `<EOS>` by default** — if stories are too short, increase `max_new_tokens`.
4. **CPU training** is slow but doable for ~10M params. Expect 2-5 min per epoch with batch=32.
5. **Temperature too low** (< 0.5) → repetitive loops. **Too high** (> 1.2) → gibberish. Stick to 0.7-0.9.
