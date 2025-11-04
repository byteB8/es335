from typing import Dict, Optional, List
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_embedding_2d(
    emb_weight: torch.Tensor,
    itos: Dict[int, str],
    select_indices: Optional[List[int]] = None,
    title: str = "Embedding visualization (2D/ t-SNE)",
    figsize: tuple = (12, 8),
    fontsize: int = 8,
    max_words: Optional[int] = None,
):
    """Plot embeddings in 2D using t-SNE if needed."""
    W = emb_weight.detach().cpu().numpy()

    # Select subset if specified
    if select_indices is not None:
        W = W[select_indices]
        itos_local = {i: itos[idx] for i, idx in enumerate(select_indices)}
    else:
        itos_local = itos

    # Limit number of words for visualization if too many
    if max_words is not None and len(W) > max_words:
        # Select most frequent or random subset
        rng = np.random.default_rng(42)
        indices = rng.choice(len(W), max_words, replace=False)
        W = W[indices]
        itos_local = {i: itos_local[idx] for i, idx in enumerate(indices)}

    # Reduce to 2D if needed
    if W.shape[1] == 2:
        xs, ys = W[:, 0], W[:, 1]
    else:
        tsne = TSNE(n_components=2, init="random",
                    learning_rate="auto", random_state=42)
        XY = tsne.fit_transform(W)
        xs, ys = XY[:, 0], XY[:, 1]

    plt.figure(figsize=figsize)
    plt.scatter(xs, ys, s=20, c='k', alpha=0.6)

    # Annotate with words
    for i in range(len(xs)):
        plt.text(xs[i] + 0.02, ys[i] + 0.02, itos_local[i], fontsize=fontsize)

    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.tight_layout()
    return plt.gca()


def compare_embeddings(
    emb_before: torch.Tensor,
    emb_after: torch.Tensor,
    itos: Dict[int, str],
    title_before: str = "Before Training (Random)",
    title_after: str = "After Training",
    select_words: Optional[List[str]] = None,
    max_words: Optional[int] = None,
):
    """Compare embeddings before and after training side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    W_before = emb_before.detach().cpu().numpy()
    W_after = emb_after.detach().cpu().numpy()

    # Select specific words if provided
    if select_words is not None:
        select_indices = []
        for word in select_words:
            for idx, w in itos.items():
                if w == word:
                    select_indices.append(idx)
                    break
        if select_indices:
            W_before = W_before[select_indices]
            W_after = W_after[select_indices]
            itos_local = {i: itos[idx] for i, idx in enumerate(select_indices)}
        else:
            itos_local = itos
    else:
        itos_local = itos

    # Limit words for visualization
    if max_words is not None and len(W_before) > max_words:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(W_before), max_words, replace=False)
        W_before = W_before[indices]
        W_after = W_after[indices]
        itos_local = {i: itos_local[idx] for i, idx in enumerate(indices)}

    # Reduce to 2D using t-SNE
    print("Reducing before-training embeddings to 2D...")
    tsne_before = TSNE(n_components=2, init="random",
                       learning_rate="auto", random_state=42)
    XY_before = tsne_before.fit_transform(W_before)

    print("Reducing after-training embeddings to 2D...")
    tsne_after = TSNE(n_components=2, init="random",
                      learning_rate="auto", random_state=42)
    XY_after = tsne_after.fit_transform(W_after)

    # Plot before training
    ax1.scatter(XY_before[:, 0], XY_before[:, 1], s=20, c='blue', alpha=0.6)
    for i in range(len(XY_before)):
        ax1.text(XY_before[i, 0] + 0.02, XY_before[i, 1] + 0.02,
                 itos_local[i], fontsize=8)
    ax1.set_title(title_before, fontsize=14)
    ax1.set_xlabel("Dimension 1", fontsize=12)
    ax1.set_ylabel("Dimension 2", fontsize=12)

    # Plot after training
    ax2.scatter(XY_after[:, 0], XY_after[:, 1], s=20, c='red', alpha=0.6)
    for i in range(len(XY_after)):
        ax2.text(XY_after[i, 0] + 0.02, XY_after[i, 1] + 0.02,
                 itos_local[i], fontsize=8)
    ax2.set_title(title_after, fontsize=14)
    ax2.set_xlabel("Dimension 1", fontsize=12)
    ax2.set_ylabel("Dimension 2", fontsize=12)

    plt.tight_layout()
    return fig
