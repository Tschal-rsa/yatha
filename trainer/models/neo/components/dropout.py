from typing import Callable

import torch
from torch import Tensor, nn


class Dropout(nn.Module):
    __call__: Callable[..., Tensor]

    def __init__(self, p: float) -> None:
        super().__init__()
        self.bernoulli = torch.distributions.Bernoulli(probs=p)

    def forward(self, x: Tensor) -> Tensor:
        mask = self.bernoulli.sample(x.size()).to(x.device)
        masked_x = torch.where(mask.bool(), 1.0 - x, x)
        return masked_x if self.training else x
