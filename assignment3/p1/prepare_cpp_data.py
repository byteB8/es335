import os
import re
import numpy as np
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "./data/cpp_repo"
CONTEXT_LEN = 5  # number of previous tokens used for prediction

# Verify directory exists
if not os.path.exists(DATA_DIR):
    raise ValueError(f"Data directory not found: {DATA_DIR}")

# ----------------------------
# 1. COLLECT C++ FILES (recursively from all subdirectories)
# ----------------------------
cpp_files = []
folder_stats = defaultdict(int)

for root, dirs, files in os.walk(DATA_DIR):
    # Skip hidden directories
    dirs[:] = [d for d in dirs if not d.startswith('.')]

    for f in files:
        if f.endswith((".cpp", ".h", ".hpp")):
            full_path = os.path.join(root, f)
            cpp_files.append(full_path)
            # Track stats per folder
            rel_folder = os.path.relpath(root, DATA_DIR)
            folder_stats[rel_folder] += 1

print(f"📁 Directory structure: {DATA_DIR}")
print("📊 Files found per folder:")
for folder, count in sorted(folder_stats.items()):
    if folder == '.':
        print(f"  root: {count} files")
    else:
        print(f"  {folder}: {count} files")
print(f"\n✅ Collected {len(cpp_files)} C++ files total")

# ----------------------------
# 2. READ AND CLEAN CODE
# ----------------------------


def clean_line(line):
    # remove comments and preprocess
    line = re.sub(r'//.*', '', line)
    line = re.sub(r'/\*.*?\*/', '', line)
    line = re.sub(r'\s+', ' ', line.strip())
    return line


all_lines = []
failed_files = []

for file in tqdm(cpp_files, desc="Reading files"):
    try:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            file_lines = 0
            for line in f:
                line = clean_line(line)
                if line:
                    all_lines.append(line)
                    file_lines += 1
    except Exception as e:
        failed_files.append((file, str(e)))
        continue

print(
    f"✅ Processed {len(cpp_files) - len(failed_files)}/{len(cpp_files)} files successfully")
if failed_files:
    print(f"⚠️  Failed to read {len(failed_files)} files")
print(f"📝 Total lines of code: {len(all_lines):,}")

# ----------------------------
# 3. TOKENIZATION
# ----------------------------


def tokenize(line):
    line = re.sub(r'([(){};,+\-*/<>=\[\]])', r' \1 ', line)
    return line.split()


print("\n🔤 Tokenizing...")
tokens = []
for line in tqdm(all_lines, desc="Tokenizing lines"):
    tokens.extend(tokenize(line))

print(f"✅ Total tokens: {len(tokens):,}")

# ----------------------------
# 4. BUILD VOCAB
# ----------------------------
print("\n📚 Building vocabulary...")
vocab = sorted(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

print(f"✅ Vocabulary size: {len(vocab):,}")
token_counts = Counter(tokens)
top_tokens = [w for w, _ in token_counts.most_common(10)]
print(f"   Top 10 most frequent tokens: {top_tokens}")

# ----------------------------
# 5. CREATE TRAINING DATA
# ----------------------------
print("\n🔨 Creating training data...")
X, y = [], []
num_samples = len(tokens) - CONTEXT_LEN

for i in tqdm(range(num_samples), desc="Creating context-target pairs"):
    X.append([word2idx[w] for w in tokens[i:i+CONTEXT_LEN]])
    y.append(word2idx[tokens[i+CONTEXT_LEN]])

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

print(f"✅ Created {len(X):,} training samples")

# ----------------------------
# 6. SAVE DATA
# ----------------------------
print("\n💾 Saving dataset...")
os.makedirs(DATA_DIR, exist_ok=True)

X_path = os.path.join(DATA_DIR, "train_X.npy")
y_path = os.path.join(DATA_DIR, "train_y.npy")
vocab_path = os.path.join(DATA_DIR, "vocab.pkl")

np.save(X_path, X)
np.save(y_path, y)

with open(vocab_path, "wb") as f:
    pickle.dump({"word2idx": word2idx, "idx2word": idx2word,
                "context_len": CONTEXT_LEN}, f)

print("\n" + "="*60)
print("✅ Dataset ready!")
print("="*60)
print(f"📊 Training samples: {X.shape[0]:,}")
print(f"📏 Context length: {CONTEXT_LEN}")
print(f"📦 Features shape: {X.shape}")
print(f"🎯 Labels shape: {y.shape}")
print(f"📚 Vocabulary size: {len(vocab):,}")
print("\n💾 Files saved:")
print(f"   - {X_path}")
print(f"   - {y_path}")
print(f"   - {vocab_path}")
print("="*60)
