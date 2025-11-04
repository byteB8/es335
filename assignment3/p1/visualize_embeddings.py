import argparse
import json
import os
import torch
import matplotlib.pyplot as plt

from models import MLPNextWord, RNNNextWord, GRUNextWord, LSTMNextWord
from visualize import compare_embeddings, plot_embedding_2d


def load_vocab(checkpoint_dir):
    """Load vocabulary from checkpoint directory."""
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    stoi = vocab_data["word2idx"]
    itos = {int(k): v for k, v in vocab_data["idx2word"].items()}
    context_len = vocab_data.get("context_len", 5)

    return stoi, itos, context_len


def create_random_model(model_type, vocab_size, block_size, emb_dim, hidden_size, num_layers):
    """Create a model with random initialization."""
    model_classes = {
        "MLP": MLPNextWord,
        "RNN": RNNNextWord,
        "GRU": GRUNextWord,
        "LSTM": LSTMNextWord,
    }

    model_class = model_classes.get(model_type, MLPNextWord)

    model = model_class(
        vocab_size=vocab_size,
        block_size=block_size,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        activation="relu",
        dropout=0.0,
    )

    return model


def load_trained_model(checkpoint_dir, model_type):
    """Load trained model from checkpoint."""
    ckpt_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)

    stoi, itos, context_len = load_vocab(checkpoint_dir)

    # Determine model config based on checkpoint directory
    if checkpoint_dir == "checkpoints_sherlock":
        emb_dim = 64
        hidden_size = 1024
        num_layers = 4
    elif checkpoint_dir == "checkpoints_cpp":
        emb_dim = 64
        hidden_size = 512
        num_layers = 4
    else:
        # Default fallback
        emb_dim = 64
        hidden_size = 1024
        num_layers = 1

    model = create_random_model(model_type, len(
        stoi), context_len, emb_dim, hidden_size, num_layers)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)

    return model, stoi, itos


def main():
    p = argparse.ArgumentParser(
        description="Visualize word embeddings before and after training")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        choices=["checkpoints_sherlock", "checkpoints_cpp"],
        required=True,
        help="Checkpoint directory to visualize"
    )
    p.add_argument(
        "--model_type",
        type=str,
        choices=["MLP", "RNN", "GRU", "LSTM"],
        default="MLP",
        help="Model architecture"
    )
    p.add_argument(
        "--max_words",
        type=int,
        default=100,
        help="Maximum number of words to visualize (default: 100)"
    )
    p.add_argument(
        "--select_words",
        type=str,
        nargs="+",
        default=None,
        help="Specific words to visualize (e.g., --select_words the quick brown)"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save figure (e.g., embeddings.png)"
    )
    args = p.parse_args()

    print(f"📁 Loading checkpoint from: {args.checkpoint_dir}")
    print(f"🏗️  Model type: {args.model_type}")

    # Load vocabulary
    stoi, itos, context_len = load_vocab(args.checkpoint_dir)
    print(f"✅ Vocabulary size: {len(stoi):,}")

    # Determine model config
    if args.checkpoint_dir == "checkpoints_sherlock":
        emb_dim = 64
        hidden_size = 1024
        num_layers = 4
    else:  # checkpoints_cpp
        emb_dim = 64
        hidden_size = 512
        num_layers = 4

    # Create random model (before training)
    print("\n🎲 Creating random model (before training)...")
    model_before = create_random_model(
        args.model_type, len(
            stoi), context_len, emb_dim, hidden_size, num_layers
    )
    emb_before = model_before.emb.weight

    # Load trained model (after training)
    print("📚 Loading trained model (after training)...")
    model_after, _, _ = load_trained_model(
        args.checkpoint_dir, args.model_type)
    emb_after = model_after.emb.weight

    print(f"📊 Embedding dimensions: {emb_before.shape}")

    # Visualize
    print("\n🎨 Creating visualization...")
    fig = compare_embeddings(
        emb_before,
        emb_after,
        itos,
        title_before=f"Before Training (Random) - {args.checkpoint_dir}",
        title_after=f"After Training - {args.checkpoint_dir}",
        select_words=args.select_words,
        max_words=args.max_words,
    )

    # Save or show
    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"✅ Saved visualization to: {args.output}")
    else:
        plt.show()

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
