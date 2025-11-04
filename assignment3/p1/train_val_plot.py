import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_train_val_loss(checkpoint_dir, title=None, save_path=None, show_plot=True):
    """
    Plot training vs validation loss from history.json in checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory containing history.json
        title: Optional title for the plot (default: uses checkpoint directory name)
        save_path: Optional path to save the figure (e.g., 'loss_plot.png')
        show_plot: Whether to display the plot (default: True)
    """
    history_path = os.path.join(checkpoint_dir, "history.json")

    if not os.path.exists(history_path):
        raise FileNotFoundError(f"history.json not found in {checkpoint_dir}")

    # Load history
    with open(history_path, "r") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss or not val_loss:
        raise ValueError("history.json missing train_loss or val_loss")

    epochs = range(1, len(train_loss) + 1)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss",
             color="blue", linewidth=2, alpha=0.7)
    plt.plot(epochs, val_loss, label="Validation Loss",
             color="red", linewidth=2, alpha=0.7)

    # Formatting
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(
        title or f"Training vs Validation Loss - {os.path.basename(checkpoint_dir)}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add statistics
    min_train = min(train_loss)
    min_val = min(val_loss)
    min_train_epoch = epochs[np.argmin(train_loss)]
    min_val_epoch = epochs[np.argmin(val_loss)]

    stats_text = (
        f"Min Train Loss: {min_train:.4f} @ Epoch {min_train_epoch}\n"
        f"Min Val Loss: {min_val:.4f} @ Epoch {min_val_epoch}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    return plt.gca()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training vs validation loss from checkpoint directory")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory containing history.json"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the plot (default: uses checkpoint directory name)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save plot (e.g., loss_plot.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save if --output is provided)"
    )
    args = parser.parse_args()

    print(f"📁 Loading history from: {args.checkpoint_dir}")

    plot_train_val_loss(
        checkpoint_dir=args.checkpoint_dir,
        title=args.title,
        save_path=args.output,
        show_plot=not args.no_show,
    )

    print("✅ Plotting complete!")


if __name__ == "__main__":
    main()
