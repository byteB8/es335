import argparse
import os
import pickle
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import MLPNextWord
from training import train


def load_cpp_data(data_dir: str):
    """Load preprocessed C++ dataset."""
    X_path = os.path.join(data_dir, "train_X.npy")
    y_path = os.path.join(data_dir, "train_y.npy")
    vocab_path = os.path.join(data_dir, "vocab.pkl")

    if not all(os.path.exists(p) for p in [X_path, y_path, vocab_path]):
        raise FileNotFoundError(
            f"Missing files in {data_dir}. Run prepare_cpp_data.py first."
        )

    X = np.load(X_path)
    y = np.load(y_path)
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    word2idx = vocab_data["word2idx"]
    idx2word = vocab_data["idx2word"]
    context_len = vocab_data.get("context_len", X.shape[1])

    print(f"✅ Loaded dataset from {data_dir}")
    print(f"   Samples: {len(X):,}")
    print(f"   Context length: {context_len}")
    print(f"   Vocabulary size: {len(word2idx):,}")

    return X, y, word2idx, idx2word, context_len


def train_val_split_np(X, y, val_ratio: float = 0.1, seed: int = 42):
    """Split numpy arrays into train/val."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_val = int(len(X) * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def create_loaders_np(Xtr, ytr, Xval, yval, batch_size: int, device):
    """Create DataLoaders from numpy arrays."""
    dtr = TensorDataset(
        torch.tensor(Xtr, dtype=torch.long, device=device),
        torch.tensor(ytr, dtype=torch.long, device=device),
    )
    dval = TensorDataset(
        torch.tensor(Xval, dtype=torch.long, device=device),
        torch.tensor(yval, dtype=torch.long, device=device),
    )
    return (
        DataLoader(dtr, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(dval, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def main():
    p = argparse.ArgumentParser(description="Train MLP on C++ code dataset")
    p.add_argument(
        "--data_dir",
        type=str,
        default="./data/cpp_repo",
        help="Directory containing train_X.npy, train_y.npy, vocab.pkl",
    )
    p.add_argument("--emb_dim", type=int, default=64,
                   help="Embedding dimension")
    p.add_argument("--hidden", type=int, default=1024,
                   help="Hidden layer size")
    p.add_argument("--layers", type=int, default=4,
                   help="Number of hidden layers")
    p.add_argument("--activation", type=str,
                   default="relu", choices=["relu", "tanh"])
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="Validation split ratio")
    p.add_argument(
        "--ckpt_dir", type=str, default="checkpoints_cpp", help="Checkpoint directory"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    X, y, word2idx, idx2word, context_len = load_cpp_data(args.data_dir)

    # Train/val split
    Xtr, ytr, Xval, yval = train_val_split_np(
        X, y, val_ratio=args.val_ratio, seed=args.seed)
    print(f"\n📊 Split: {len(Xtr):,} train, {len(Xval):,} val")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Create loaders
    train_loader, val_loader = create_loaders_np(
        Xtr, ytr, Xval, yval, batch_size=args.batch_size, device=device
    )

    # Create model
    model = MLPNextWord(
        vocab_size=len(word2idx),
        block_size=context_len,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(
        f"\n📐 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Train
    print("\n🚀 Starting training...\n")
    history, best_path = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        ckpt_dir=args.ckpt_dir,
        print_every=max(1, args.epochs // 20),
    )

    # Save vocab for inference
    vocab_json_path = os.path.join(args.ckpt_dir, "vocab.json")
    with open(vocab_json_path, "w") as f:
        json.dump(
            {
                "word2idx": word2idx,
                "idx2word": {int(k): v for k, v in idx2word.items()},
                "context_len": context_len,
            },
            f,
        )

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"📁 Checkpoint: {best_path}")
    print(f"📊 Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"📊 Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"📚 Vocab saved: {vocab_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
