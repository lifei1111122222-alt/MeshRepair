from __future__ import annotations

import torch
from torch import nn


class CueConditionedCyclicBoundaryEncoder(nn.Module):
    def __init__(self, token_dim: int = 12, scalar_dim: int = 5, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(token_dim, hidden, 5, padding=2, padding_mode="circular"), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, 5, padding=2, padding_mode="circular"), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, 3, padding=1, padding_mode="circular"), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2 + scalar_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(nn.Linear(scalar_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def loop_logits(self, tokens: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        # tokens [B,Q,D], scalars [B,S]
        x = self.conv(tokens.transpose(1, 2))
        h = torch.cat([x.mean(-1), x.amax(-1), scalars], dim=1)
        return self.head(h).squeeze(-1)

    def forward(self, tokens: torch.Tensor, scalars: torch.Tensor, null_scalars: torch.Tensor | None = None):
        logits = self.loop_logits(tokens, scalars)
        if null_scalars is None:
            null_scalars = scalars.mean(dim=0, keepdim=True).expand(tokens.shape[0], -1)
        null_logit = self.null_head(null_scalars).squeeze(-1)
        return torch.cat([null_logit[:1], logits], dim=0)

    @staticmethod
    def probabilities(logits: torch.Tensor):
        return torch.softmax(logits, dim=0)
