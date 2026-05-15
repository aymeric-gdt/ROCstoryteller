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
        emb = self.embedding(x)                        # [B, T, E]
        lstm_out, (h, c) = self.lstm(emb, hidden)      # [B, T, H]
        lstm_out = self.dropout(lstm_out)
        logits = self.head(lstm_out)                   # [B, T, V]
        return logits, (h, c)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize zero hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
