import torch
from torch import Tensor, nn

from utils import logger

from config import NeoConfig
from ..functions import Binarize, mask_conjunction, mask_disjunction
from .interface import LogicInterface, RuleType

class LogicLayerNFRL(LogicInterface):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(cfg, input_dim, output_dim)
        self.W = nn.Parameter(torch.rand(input_dim, output_dim) * 0.25 + 0.5)
        # self.r = nn.Parameter(0.5 * torch.randn(output_dim))
        # self.andness = nn.Parameter(torch.rand(output_dim))
        self.andness = nn.Parameter(torch.randn(output_dim) * 0.25 + 0.5)
    
    @property
    def Wb(self) -> Tensor:
        return Binarize.apply(self.W - 0.5)
    
    @property
    def rule_types(self) -> list[RuleType]:
        return [
            # RuleType.AND if is_and else RuleType.OR for is_and in self.r >= 0
            RuleType.AND if is_and else RuleType.OR for is_and in self.andness >= 0.5
        ]
    
    # override
    def forward(self, x: Tensor) -> Tensor:
        Wb = self.Wb
        # rb = Binarize.apply(self.r)
        rb = Binarize.apply(self.andness - 0.5)
        con_x = mask_conjunction(x, Wb)
        dis_x = mask_disjunction(x, Wb)
        return con_x * rb + dis_x * (1 - rb)
    
    def bool_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    # not used
    def fuzzy_forward(self, x: Tensor) -> Tensor:
        return self.bool_forward(x)
    
    def l1_norm(self) -> Tensor:
        return self.W.sum()

    def l2_norm(self) -> Tensor:
        return (self.W**2).sum()

    def clip(self) -> None:
        self.W.data.clamp_(0.0, 1.0)
        # self.r.data.clamp_(-1.0, 1.0)
        self.andness.data.clamp_(0.0, 1.0)
    
    # override
    def bool_forward_with_count(self, x: Tensor) -> Tensor:
        x = self.bool_forward(x)
        x_clip = x.clamp(0.0, 1.0)
        self.node_activation_cnt += torch.sum(x_clip, dim=0)
        self.forward_tot += x_clip.size(0)
        return x
    
    # override
    def debug(self) -> None:
        # activation ratio
        activation_ratio = self.get_activation_ratio()
        logger.debug(activation_ratio)
        # r average
        # r_average = self.r.mean().item()
        # logger.debug(f'AVG r:        {r_average}')
        # andness average
        andness_average = self.andness.mean().item()
        logger.debug(f'AVG Andness:  {andness_average}')
        # r ratio
        # r_ratio = (self.r >= 0).type_as(self.r).mean().item()
        # logger.debug(f'RAT r:        {r_ratio}')
        # andness ratio
        andness_ratio = (self.andness >= 0.5).type_as(self.andness).mean().item()
        logger.debug(f'RAT Andness:  {andness_ratio}')
        # activation rate
        # and_act_rate = activation_ratio[self.r >= 0].mean().item()
        # or_act_rate = activation_ratio[self.r < 0].mean().item()
        and_act_rate = activation_ratio[self.andness >= 0.5].mean().item()
        or_act_rate = activation_ratio[self.andness < 0.5].mean().item()
        logger.debug(f'AND Act Rate: {and_act_rate}')
        logger.debug(f'OR Act Rate:  {or_act_rate}')