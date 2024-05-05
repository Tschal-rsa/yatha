import torch
from torch import Tensor

from dataset import Normalizer

from ..functions import BinarizeNFRL
from .binarize import BinarizeLayer


class BinarizeLayerNFRL(BinarizeLayer):
    def __init__(self, input_dim: int, num_binarization: int, normalizer: Normalizer) -> None:
        super().__init__(input_dim, num_binarization, normalizer)

    def forward(self, x: Tensor) -> Tensor:
        x.unsqueeze_(-1)
        binarize_res = BinarizeNFRL.apply(x - self.cl.t()).reshape(x.size(0), -1)
        return torch.cat((binarize_res, -binarize_res), dim=1)
    
    # override
    def bool_forward_with_count(self, x: Tensor) -> Tensor:
        x = self.bool_forward(x)
        x_clip = x.clamp(0.0, 1.0)
        self.node_activation_cnt += torch.sum(x_clip, dim=0)
        self.forward_tot += x_clip.size(0)
        return x
