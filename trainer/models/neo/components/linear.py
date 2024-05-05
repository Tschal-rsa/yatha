from collections import defaultdict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from config import NeoConfig
from .interface import ModuleInterface


class Linear(ModuleInterface):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.rid2dim: dict[int, int] = {}
        self.rule2weights: list[tuple[int, dict[int, float]]] = []
        self.bl = np.zeros(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

    def bool_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def edge_count(self) -> Tensor:
        return torch.tensor(0.0)

    def l1_norm(self) -> Tensor:
        return self.fc.weight.abs().sum()

    def l2_norm(self) -> Tensor:
        return torch.sum(self.fc.weight**2)

    def clip(self) -> None:
        for param in self.fc.parameters():
            param.data.clamp_(-1.0, 1.0)

    def get_rule2weights(self, prev_layer: ModuleInterface) -> None:
        always_act_pos = prev_layer.node_activation_cnt == prev_layer.forward_tot
        prev_dim2id = prev_layer.dim2id
        Wl, bl = self.fc.parameters()
        bl_data = torch.sum(Wl.T[always_act_pos], dim=0) + bl.data
        Wl_data: NDArray = Wl.numpy(force=True)
        self.bl = bl_data.numpy(force=True)

        marked: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.rid2dim = {}
        for label_id, wl in enumerate(Wl_data):
            for i, w in enumerate(wl):
                rid = prev_dim2id[i]
                if rid == -1:
                    continue
                marked[rid][label_id] += w
                self.rid2dim[rid] = i

        self.rule2weights = sorted(
            marked.items(),
            key=lambda x: max(map(lambda y: abs(y), list(x[1].values()))),
            reverse=True,
        )
