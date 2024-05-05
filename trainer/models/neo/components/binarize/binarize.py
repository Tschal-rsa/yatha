import torch
from torch import Tensor

from dataset import Normalizer

from ..functions import Binarize
from ..interface import ModuleInterface


class BinarizeLayer(ModuleInterface):
    cl: Tensor

    def __init__(
        self, input_dim: int, num_binarization: int, normalizer: Normalizer
    ) -> None:
        output_dim = num_binarization * input_dim * 2
        super().__init__(input_dim, output_dim)
        self.num_binarization = num_binarization
        cl = torch.randn(num_binarization, input_dim)
        self.register_buffer('cl', cl)
        self.normalizer = normalizer
        self.dim2id = [i for i in range(output_dim)]

    def forward(self, x: Tensor) -> Tensor:
        x.unsqueeze_(-1)
        binarize_res = Binarize.apply(x - self.cl.t()).reshape(x.size(0), -1)
        return torch.cat((binarize_res, 1 - binarize_res), dim=1)

    def bool_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def edge_count(self) -> Tensor:
        return torch.tensor(0.0)

    def l1_norm(self) -> Tensor:
        return torch.tensor(0.0)

    def l2_norm(self) -> Tensor:
        return torch.tensor(0.0)

    def clip(self) -> None:
        pass

    def get_bound_name(self, feature_names: list[str]) -> None:
        self.rule_names = []
        cl = self.cl.numpy(force=True)
        cl_normalized = self.normalizer.inverse_transform(cl)
        for op in ['>=', '<']:
            for i, ci in enumerate(cl_normalized.T):
                fi_name = feature_names[i]
                self.rule_names += [f'{fi_name} {op} {c:.3f}' for c in ci]
