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
    Wrap each sentence with <SOS>/<EOS> markers.

    Input:  "Tom went to the store. He bought milk. He went home."
    Output: "<SOS> Tom went to the store. <EOS> <SOS> He bought milk. <EOS> <SOS> He went home. <EOS>"
    """
    sentences = [s.strip() for s in story.split(".") if s.strip()]
    parts = []
    for sent in sentences:
        parts.append(f"<SOS> {sent}. <EOS>")
    return " ".join(parts)


def wrap_paragraph(story: str) -> str:
    """
    Wrap the whole story with a single <SOS>...<EOS> pair.
    Sentences are kept as-is, separated by periods.

    Input:  "Tom went to the store. He bought milk. He went home."
    Output: "<SOS> Tom went to the store. He bought milk. He went home. <EOS>"
    """
    return f"<SOS> {story.strip()} <EOS>"
