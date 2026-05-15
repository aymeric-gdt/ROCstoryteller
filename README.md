# ROCstoryteller

LSTM-based story generator trained on [ROCStories](https://cs.rochester.edu/nlp/rocstories/) — 98k five-sentence everyday stories. Give it a prompt, it writes the rest.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
# CPU (default)
python train.py

# GPU
python train.py --device cuda --batch-size 128 --epochs 30

# Full custom
python train.py --device cuda --vocab-size 16000 --hidden-dim 768 --num-layers 3 --epochs 40
```

Checkpoints saved to `checkpoints/best_model.pt` and `checkpoints/tokenizer.json`.

## Generate

```bash
# One-shot
python generate.py --prompt "Tom went to the store" --temperature 0.8

# Interactive REPL
python play.py --temperature 0.7 --device cuda
```

## Architecture

```
Prompt: "Tom went to the store"
              │
              ▼
    ┌─────────────────────┐
    │    BPETokenizer     │   BPE 8k vocab
    │  <SOS> Tom went...  │   + <PAD> <UNK> <SOS> <EOS>
    └─────────┬───────────┘
              │ token IDs
              ▼
    ┌─────────────────────┐
    │     Embedding       │   256 dim
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │   LSTM × 2 layers   │   512 hidden, dropout 0.3
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Linear → Softmax  │   over vocab (8k)
    └─────────┬───────────┘
              │
              ▼
    Temperature sampling (T=0.8)
              │
              ▼
    "Tom went to the store. He bought milk.
     The cashier smiled at him. Tom felt happy.
     He decided to come back tomorrow."
```

**~10M parameters.** Trained with teacher forcing, cross-entropy loss (padding ignored), AdamW + cosine schedule, gradient clipping 1.0.

## Story format

Each story is wrapped with sentence-boundary tokens before tokenization:

```
<SOS> Tom went to the store. <EOS> <SOS> He bought milk. <EOS> ...
```

This lets the model learn explicit sentence boundaries and generate exactly 5 sentences.

## Project structure

```
ROCstoryteller/
├── requirements.txt
├── dataset.py        # HF datasets loader + <SOS>/<EOS> wrapping
├── tokenizer.py      # BPETokenizer (HuggingFace tokenizers)
├── data_utils.py     # StoryDataset + collate (padding)
├── model.py          # StoryLSTM (Embedding → LSTM → Linear)
├── train.py          # Training loop
├── generate.py       # CLI generation
├── play.py           # Interactive REPL
└── checkpoints/      # Saved models + tokenizer
```

## CLI reference

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--epochs` | `20` | Number of epochs |
| `--batch-size` | `32` | Batch size (bump to 128 on GPU) |
| `--lr` | `3e-4` | Learning rate (AdamW) |
| `--vocab-size` | `8000` | BPE vocabulary size |
| `--embed-dim` | `256` | Embedding dimension |
| `--hidden-dim` | `512` | LSTM hidden dimension |
| `--num-layers` | `2` | LSTM layers |
| `--dropout` | `0.3` | Dropout between LSTM layers |
| `--max-len` | `128` | Max token length per story |

### `generate.py`

| Flag | Default | Description |
|---|---|---|
| `--prompt` | (required) | Starting sentence(s) |
| `--temperature` | `0.8` | Sampling temperature (0.6–1.0) |
| `--num-sentences` | `5` | Sentences to generate |
| `--max-tokens` | `150` | Hard token cap |
| `--device` | `cpu` | `cpu` or `cuda` |

### `play.py`

| Flag | Default | Description |
|---|---|---|
| `--temperature` | `0.8` | Sampling temperature |
| `--device` | `cpu` | `cpu` or `cuda` |
