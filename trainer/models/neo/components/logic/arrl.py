import torch
from torch import Tensor, nn

from config import NeoConfig
from utils import logger

from .interface import LogicInterface, RuleType
from .rrl import LogicInterfaceRRL


class UnionLayer(LogicInterfaceRRL):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(cfg, input_dim, output_dim)
        # self.andness = nn.Parameter(torch.rand(output_dim))
        self.andness = nn.Parameter(torch.randn(output_dim) * 0.25 + 0.5)
    
    def forward(self, x: Tensor) -> Tensor:
        con_x = self.fuzzy_conjunction(x)
        dis_x = self.fuzzy_disjunction(x)
        return self.andness * con_x + (1 - self.andness) * dis_x
    
    def bool_forward(self, x: Tensor) -> Tensor:
        con_x = self.bool_conjunction(x)
        dis_x = self.bool_disjunction(x)
        return torch.where(self.andness >= 0.5, con_x, dis_x)
    
    @property
    def rule_types(self) -> list[RuleType]:
        return [RuleType.AND if is_and else RuleType.OR for is_and in self.andness >= 0.5]
    
    # override
    def clip(self) -> None:
        super().clip()
        self.andness.data.clamp_(0.0, 1.0)
    
    # override
    def debug(self) -> None:
        # activation ratio
        activation_ratio = self.get_activation_ratio()
        logger.debug(activation_ratio)
        # andness average
        andness_average = self.andness.mean().item()
        logger.debug(f'AVG Andness:  {andness_average}')
        # andness ratio
        andness_ratio = (self.andness >= 0.5).type_as(self.andness).mean().item()
        logger.debug(f'RAT Andness:  {andness_ratio}')
        # activation rate
        and_act_rate = activation_ratio[self.andness >= 0.5].mean().item()
        or_act_rate = activation_ratio[self.andness < 0.5].mean().item()
        logger.debug(f'AND Act Rate: {and_act_rate}')
        logger.debug(f'OR Act Rate:  {or_act_rate}')


class LogicLayerAndnessRRL(LogicInterface):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(cfg, input_dim, output_dim)
        self.union_layer = UnionLayer(cfg, input_dim, output_dim)

    # override
    def clear_activation(self) -> None:
        super().clear_activation()
        self.union_layer.clear_activation()
    
    def fuzzy_forward(self, x: Tensor) -> Tensor:
        return self.union_layer(x)
    
    def bool_forward(self, x: Tensor) -> Tensor:
        return self.union_layer.bool_forward_with_count(x)

    @property
    def Wb(self) -> Tensor:
        return self.union_layer.Wb

    @property
    def rule_types(self) -> list[RuleType]:
        return self.union_layer.rule_types

    def l1_norm(self) -> Tensor:
        return self.union_layer.l1_norm()

    def l2_norm(self) -> Tensor:
        return self.union_layer.l2_norm()

    def clip(self) -> None:
        self.union_layer.clip()
    
    # override
    def debug(self) -> None:
        self.union_layer.debug()
