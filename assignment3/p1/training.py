import os
import json
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def to_tensor2d(X, device):
    return torch.tensor(X, dtype=torch.long, device=device)


def to_tensor1d(y, device):
    return torch.tensor(y, dtype=torch.long, device=device)


def create_loaders(Xtr, Ytr, Xval, Yval, batch_size: int, device) -> Tuple[DataLoader, DataLoader]:
    dtr = TensorDataset(to_tensor2d(Xtr, device), to_tensor1d(Ytr, device))
    dval = TensorDataset(to_tensor2d(Xval, device), to_tensor1d(Yval, device))
    return (
        DataLoader(dtr, batch_size=batch_size, shuffle=True),
        DataLoader(dval, batch_size=batch_size, shuffle=False),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    ckpt_dir: str = "checkpoints",
    print_every: int = 50,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        train_loss = total / max(1, n)

        model.eval()
        vtotal = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vtotal += float(loss.item())
                vn += 1
        val_loss = vtotal / max(1, vn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, best_path)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print(f"epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}")

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f)

    return history, best_path


