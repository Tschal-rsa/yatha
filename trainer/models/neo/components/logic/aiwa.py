import torch
from torch import Tensor, nn

from config import NeoConfig
from utils import logger

from ..functions import AIWA, Binarize, conjunction, disjunction
from .interface import LogicInterface, RuleType


class LogicLayerAIWA(LogicInterface):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(cfg, input_dim, output_dim)
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))
        self.bound, self.eps = cfg.bound, cfg.eps
        self.AIWA = AIWA(self.bound, self.eps)
        self.andness = nn.Parameter(
            torch.rand(output_dim) * (1 - 2 * self.bound) + self.bound
        )

    def fuzzy_forward(self, x: Tensor) -> Tensor:
        orand_x = self.AIWA(x, self.W, self.andness)
        andor_x = 1.0 - self.AIWA(1.0 - x, self.W, 1.0 - self.andness)
        return torch.where(self.andness <= 0.5, orand_x, andor_x)

    @property
    def Wb(self) -> Tensor:
        threshold = torch.where(
            self.andness <= 0.5, 0.5 + self.andness, 1.5 - self.andness
        )
        return Binarize.apply(self.W - threshold)

    @property
    def rule_types(self) -> list[RuleType]:
        return [
            RuleType.AND if is_and else RuleType.OR for is_and in self.andness > 0.5
        ]

    def bool_forward(self, x: Tensor) -> Tensor:
        Wb = self.Wb
        con_x = conjunction(x, Wb)
        dis_x = disjunction(x, Wb)
        return torch.where(self.andness <= 0.5, dis_x, con_x)

    def l1_norm(self) -> Tensor:
        return self.W.sum()

    def l2_norm(self) -> Tensor:
        # [ ] L2 norm
        return torch.tensor(0.0)
        # return (self.W**2).sum()

    def clip(self) -> None:
        self.W.data /= self.W.data.amax(dim=0)
        self.W.data.clamp_(self.eps, 1.0)
        self.andness.data.clamp_(self.bound, 1.0 - self.bound)

    # override
    def debug(self) -> None:
        # activation ratio
        activation_ratio = self.get_activation_ratio()
        logger.debug(activation_ratio)
        # andness average
        andness_average = self.andness.mean().item()
        logger.debug(f'AVG Andness:  {andness_average}')
        # andness ratio
        andness_ratio = (self.andness > 0.5).type_as(self.andness).mean().item()
        logger.debug(f'RAT Andness:  {andness_ratio}')
        # activation rate
        and_act_rate = activation_ratio[self.andness > 0.5].mean().item()
        or_act_rate = activation_ratio[self.andness <= 0.5].mean().item()
        logger.debug(f'AND Act Rate: {and_act_rate}')
        logger.debug(f'OR Act Rate:  {or_act_rate}')
