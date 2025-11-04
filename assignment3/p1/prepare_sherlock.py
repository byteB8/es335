import os
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

from datasets import (
    clean_text,
    split_to_sentences,
    build_vocab,
    encode_sentence,
    make_xy,
    word_frequencies,
    top_k_words,
)

# ----------------------------
# CONFIG
# ----------------------------
TEXT_FILE = "./data/sherlock_holmes.txt"
OUTPUT_DIR = "./data/sherlock_data"
CONTEXT_LEN = 5  # number of previous words used for prediction
MIN_FREQ = 1  # minimum word frequency to include in vocabulary

# Verify input file exists
if not os.path.exists(TEXT_FILE):
    raise FileNotFoundError(f"Text file not found: {TEXT_FILE}")

# ----------------------------
# 1. READ TEXT FILE
# ----------------------------
print(f"📖 Reading text from: {TEXT_FILE}")
with open(TEXT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print(f"✅ Loaded {len(text):,} characters")

# ----------------------------
# 2. CLEAN AND PREPROCESS TEXT
# ----------------------------
print("\n🧹 Cleaning text...")
cleaned = clean_text(text, keep_periods=True)
print(f"✅ Cleaned text: {len(cleaned):,} characters")

# ----------------------------
# 3. SPLIT INTO SENTENCES
# ----------------------------
print("\n📝 Splitting into sentences...")
sentences = split_to_sentences(cleaned)
print(f"✅ Found {len(sentences):,} sentences")

# ----------------------------
# 4. BUILD VOCABULARY
# ----------------------------
print("\n📚 Building vocabulary...")
stoi, itos = build_vocab(sentences, min_freq=MIN_FREQ)

# Get word frequencies for reporting
word_freq = word_frequencies(sentences)
most_frequent, least_frequent = top_k_words(word_freq, k=10)

print(f"✅ Vocabulary size: {len(stoi):,}")
print("\n📊 Top 10 most frequent words:")
for word, count in most_frequent:
    print(f"   '{word}': {count:,}")
print("\n📊 Top 10 least frequent words:")
for word, count in least_frequent:
    print(f"   '{word}': {count}")

# ----------------------------
# 5. ENCODE SENTENCES
# ----------------------------
print("\n🔢 Encoding sentences...")
seqs = []
for sent in tqdm(sentences, desc="Encoding"):
    ids = encode_sentence(sent, stoi)
    seqs.append(ids)

total_words = sum(len(s) for s in seqs)
print(f"✅ Encoded {len(seqs):,} sentences ({total_words:,} total word tokens)")

# ----------------------------
# 6. CREATE TRAINING DATA (X, y pairs)
# ----------------------------
print("\n🔨 Creating training data...")
X, y = make_xy(seqs, block_size=CONTEXT_LEN)
print(f"✅ Created {len(X):,} training samples")

# Convert to numpy arrays
X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

# ----------------------------
# 7. SAVE DATA
# ----------------------------
print("\n💾 Saving dataset...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_path = os.path.join(OUTPUT_DIR, "train_X.npy")
y_path = os.path.join(OUTPUT_DIR, "train_y.npy")
vocab_path = os.path.join(OUTPUT_DIR, "vocab.pkl")

np.save(X_path, X)
np.save(y_path, y)

vocab_data = {
    "word2idx": stoi,
    "idx2word": {int(k): v for k, v in itos.items()},
    "context_len": CONTEXT_LEN,
    "word_frequencies": word_freq,
}

with open(vocab_path, "wb") as f:
    pickle.dump(vocab_data, f)

print("\n" + "=" * 60)
print("✅ Dataset ready!")
print("=" * 60)
print(f"📊 Training samples: {X.shape[0]:,}")
print(f"📏 Context length: {CONTEXT_LEN}")
print(f"📦 Features shape: {X.shape}")
print(f"🎯 Labels shape: {y.shape}")
print(f"📚 Vocabulary size: {len(stoi):,}")
print("\n💾 Files saved:")
print(f"   - {X_path}")
print(f"   - {y_path}")
print(f"   - {vocab_path}")
print("=" * 60)
