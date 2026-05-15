"""BPE tokenizer for ROCStories using HuggingFace tokenizers."""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import NFKC


class BPETokenizer:
    """
    BPE tokenizer for ROCStories.

    Special tokens (<PAD>, <UNK>, <SOS>, <EOS>) are added to vocab during
    training but NOT auto-injected via post-processing. The dataset layer
    (insert_sentence_tokens) handles <SOS>/<EOS> placement between sentences.
    """

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

        self._trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=2,
        )

    def train(self, texts: list[str]):
        """Train BPE on a list of raw stories (should already include <SOS>/<EOS> markers)."""
        self._tokenizer.train_from_iterator(texts, self._trainer)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs. No auto-wrapping — text must already contain <SOS>/<EOS> if needed."""
        return self._tokenizer.encode(text).ids

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
