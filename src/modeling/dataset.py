# ============================================================
# src/modeling/dataset.py
# PyTorch Dataset class untuk IndoBERT dan LSTM
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import INDOBERT_MAX_LENGTH, LABEL2ID


class ReviewDataset(Dataset):
    """
    Dataset untuk IndoBERT fine-tuning.
    Menghasilkan: input_ids, attention_mask, labels
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = INDOBERT_MAX_LENGTH):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


class LSTMDataset(Dataset):
    """
    Dataset untuk LSTM training.
    Menghasilkan: token_ids (tensor), label
    """

    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int = 128):
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len
        self.vocab   = vocab
        self.data    = [self._encode(t) for t in texts]

    def _encode(self, text: str) -> torch.Tensor:
        tokens = text.split()[:self.max_len]
        ids    = [self.vocab.get(t, self.vocab.get("<UNK>", 1)) for t in tokens]
        # Padding
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def build_vocab(texts: list[str], max_vocab: int = 30_000) -> dict[str, int]:
    """Bangun vocabulary dari daftar teks untuk LSTM."""
    from collections import Counter
    counter: Counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    """Load CSV split dan kembalikan (texts, labels)."""
    df     = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts  = df["review_text_clean"].astype(str).tolist()
    labels = df["sentiment_label"].map(LABEL2ID).tolist()
    return texts, labels
