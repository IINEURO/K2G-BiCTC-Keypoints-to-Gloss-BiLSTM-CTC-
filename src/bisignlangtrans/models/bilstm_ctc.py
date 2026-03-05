from __future__ import annotations

import torch
from torch import nn


class BiLSTMCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        proj_dim: int = 248,
        hidden_size: int = 248,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.encoder = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,T,D], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dim mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        x = self.input_proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            input_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(out)
        return logits
