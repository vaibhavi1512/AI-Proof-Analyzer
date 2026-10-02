"""LSTM over 16 Phase 9 feature vectors. The logit is the log-odds of FAKE."""

from __future__ import annotations

import torch
from torch import nn

from .constants import (
    LSTM_DROPOUT as DROPOUT,
    LSTM_HIDDEN_SIZE as HIDDEN_SIZE,
    LSTM_INPUT_SIZE as INPUT_SIZE,
    LSTM_NUM_LAYERS as NUM_LAYERS,
    LSTM_SEQUENCE_LENGTH as SEQUENCE_LENGTH,
)
from .exceptions import LstmModelError as LstmV2Error


class VideoLstmV2(nn.Module):
    """One FAKE logit from the final LSTM timestep. Sigmoid is not applied here."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or sequence_length < 1:
            raise LstmV2Error("LSTM sizes must be positive integers.")
        if dropout < 0 or dropout >= 1:
            raise LstmV2Error("dropout must be in the range [0, 1).")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.sequence_length = int(sequence_length)
        # Dropout applies between LSTM layers. A one-layer network has no inner dropout.
        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(batch, 1)`` from the last temporal step."""
        if features.ndim != 3:
            raise LstmV2Error(f"Expected (batch, sequence, features), got {tuple(features.shape)}.")
        if features.shape[1] != self.sequence_length or features.shape[2] != self.input_size:
            raise LstmV2Error(
                f"Expected sequence {(self.sequence_length, self.input_size)}, got {tuple(features.shape[1:])}."
            )
        if features.dtype != torch.float32:
            raise LstmV2Error(f"Expected float32 features, got {features.dtype}.")
        sequence_output, _state = self.lstm(features)
        return self.classifier(sequence_output[:, -1, :])
