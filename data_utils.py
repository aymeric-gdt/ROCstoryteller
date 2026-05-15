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
            ids = tokenizer.encode(story)
            if len(ids) > max_len:
                ids = ids[:max_len]
            # input = all except last, target = all except first
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
