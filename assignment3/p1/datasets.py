import re
import math
from typing import List, Tuple, Dict, Iterable, Optional


def clean_text(text: str, keep_periods: bool = True) -> str:
    pattern = r"[^a-zA-Z0-9 \.]" if keep_periods else r"[^a-zA-Z0-9 ]"
    text = text.lower()
    text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_to_sentences(text: str) -> List[List[str]]:
    # Split on periods to get sentences; keep non-empty tokens
    sentences = []
    for sent in text.split('.'):
        tokens = [t for t in sent.strip().split() if t]
        if tokens:
            sentences.append(tokens)
    return sentences


def build_vocab(sentences: Iterable[Iterable[str]], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    freq: Dict[str, int] = {}
    for sent in sentences:
        for w in sent:
            freq[w] = freq.get(w, 0) + 1

    words = [w for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])) if c >= min_freq]
    stoi: Dict[str, int] = {"<pad>": 0, "<unk>": 1, ".": 2}
    for i, w in enumerate(words, start=len(stoi)):
        if w != '.':
            stoi[w] = i
    itos: Dict[int, str] = {i: s for s, i in stoi.items()}
    return stoi, itos


def encode_sentence(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    unk = stoi.get("<unk>", 1)
    period = stoi.get('.', 2)
    ids = [stoi.get(t, unk) for t in tokens] + [period]
    return ids


def make_xy(seqs: List[List[int]], block_size: int) -> Tuple[List[List[int]], List[int]]:
    X: List[List[int]] = []
    Y: List[int] = []
    for ids in seqs:
        context = [0] * block_size
        for ix in ids:
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]
    return X, Y


def train_val_split(X, Y, val_ratio: float = 0.1, seed: int = 42):
    import random
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_val = int(math.floor(len(X) * val_ratio))
    val_idx = set(idx[:n_val])
    Xtr, Ytr, Xval, Yval = [], [], [], []
    for i in range(len(X)):
        if i in val_idx:
            Xval.append(X[i])
            Yval.append(Y[i])
        else:
            Xtr.append(X[i])
            Ytr.append(Y[i])
    return Xtr, Ytr, Xval, Yval


def build_dataset_from_text(text: str, block_size: int, min_freq: int = 1):
    cleaned = clean_text(text, keep_periods=True)
    sents = split_to_sentences(cleaned)
    stoi, itos = build_vocab(sents, min_freq=min_freq)
    seqs = [encode_sentence(s, stoi) for s in sents]
    X, Y = make_xy(seqs, block_size)
    return X, Y, stoi, itos


def top_k_words(freq: Dict[str, int], k: int = 10):
    most = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]
    least = sorted(freq.items(), key=lambda x: (x[1], x[0]))[:k]
    return most, least


def word_frequencies(sentences: Iterable[Iterable[str]]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for sent in sentences:
        for w in sent:
            freq[w] = freq.get(w, 0) + 1
    return freq


