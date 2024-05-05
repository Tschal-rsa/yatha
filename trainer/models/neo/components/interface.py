from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor, nn

# class LayerType(Enum):
#     BINARIZATION = auto()
#     LINEAR = auto()
#     LOGIC = auto()
#     NET = auto()
#     CONJUNCTION = auto()
#     DISJUNCTION = auto()
#     UNION = auto()
#     DROPOUT = auto()


class ModuleInterface(nn.Module, ABC):
    __call__: Callable[..., Tensor]
    node_activation_cnt: Tensor
    forward_tot: Tensor

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        node_activation_cnt = torch.zeros(output_dim, dtype=torch.float)
        self.register_buffer('node_activation_cnt', node_activation_cnt)
        forward_tot = torch.tensor(0.0)
        self.register_buffer('forward_tot', forward_tot)
        self.dim2id: list[int] = []
        self.rule_names: list[str] = []

    def clear_activation(self) -> None:
        self.node_activation_cnt.zero_()
        self.forward_tot.zero_()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def bool_forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def bool_forward_with_count(self, x: Tensor) -> Tensor:
        x = self.bool_forward(x)
        self.node_activation_cnt += torch.sum(x, dim=0)
        self.forward_tot += x.size(0)
        return x

    def get_activation_ratio(self) -> Tensor:
        return self.node_activation_cnt / self.forward_tot

    def debug(self) -> None:
        pass

    @abstractmethod
    def edge_count(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def l1_norm(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def l2_norm(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def clip(self) -> None:
        raise NotImplementedError
