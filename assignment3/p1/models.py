from typing import Literal
import torch
from torch import nn


class MLPNextWord(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 64,
        hidden_size: int = 1024,
        num_hidden_layers: int = 1,
        activation: Literal["relu", "tanh"] = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, emb_dim)

        act = nn.ReLU() if activation == "relu" else nn.Tanh()
        layers = [
            nn.Linear(block_size * emb_dim, hidden_size),
            act,
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, num_hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_size, hidden_size),
                          act, nn.Dropout(dropout)])

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, block_size)
        x = self.emb(x)  # (B, block_size, emb_dim)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        logits = self.out(x)
        return logits


class RNNNextWord(nn.Module):
    """RNN-based next-word prediction model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 64,
        hidden_size: int = 1024,
        num_hidden_layers: int = 1,
        activation: Literal["relu", "tanh"] = "tanh",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.bidirectional = bidirectional

        self.emb = nn.Embedding(vocab_size, emb_dim)

        # RNN layer
        self.rnn = nn.RNN(
            emb_dim,
            hidden_size,
            num_layers=num_hidden_layers,
            nonlinearity=activation,
            dropout=dropout if num_hidden_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Output projection
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.out = nn.Linear(out_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, block_size)
        x = self.emb(x)  # (B, block_size, emb_dim)

        # RNN forward pass
        # out: (B, block_size, hidden_size * num_directions)
        out, _ = self.rnn(x)

        # Use last timestep's hidden state
        last_hidden = out[:, -1, :]  # (B, hidden_size * num_directions)
        last_hidden = self.dropout(last_hidden)

        logits = self.out(last_hidden)  # (B, vocab_size)
        return logits


class GRUNextWord(nn.Module):
    """GRU-based next-word prediction model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 64,
        hidden_size: int = 1024,
        num_hidden_layers: int = 1,
        activation: Literal["relu", "tanh"] = "relu",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.bidirectional = bidirectional

        self.emb = nn.Embedding(vocab_size, emb_dim)

        # GRU layer
        self.gru = nn.GRU(
            emb_dim,
            hidden_size,
            num_layers=num_hidden_layers,
            dropout=dropout if num_hidden_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Output projection
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.out = nn.Linear(out_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, block_size)
        x = self.emb(x)  # (B, block_size, emb_dim)

        # GRU forward pass
        # out: (B, block_size, hidden_size * num_directions)
        out, _ = self.gru(x)

        # Use last timestep's hidden state
        last_hidden = out[:, -1, :]  # (B, hidden_size * num_directions)
        last_hidden = self.dropout(last_hidden)

        logits = self.out(last_hidden)  # (B, vocab_size)
        return logits


class LSTMNextWord(nn.Module):
    """LSTM-based next-word prediction model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 64,
        hidden_size: int = 1024,
        num_hidden_layers: int = 1,
        activation: Literal["relu", "tanh"] = "relu",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.bidirectional = bidirectional

        self.emb = nn.Embedding(vocab_size, emb_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_size,
            num_layers=num_hidden_layers,
            dropout=dropout if num_hidden_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Output projection
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.out = nn.Linear(out_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, block_size)
        x = self.emb(x)  # (B, block_size, emb_dim)

        # LSTM forward pass
        # out: (B, block_size, hidden_size * num_directions)
        out, _ = self.lstm(x)

        # Use last timestep's hidden state
        last_hidden = out[:, -1, :]  # (B, hidden_size * num_directions)
        last_hidden = self.dropout(last_hidden)

        logits = self.out(last_hidden)  # (B, vocab_size)
        return logits
