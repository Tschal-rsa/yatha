import torch
from torch import Tensor, nn

from config import NeoConfig
from utils import logger

from ..functions import RRL_G, RRL_P, Binarize, conjunction, disjunction
from ..interface import ModuleInterface
from .interface import LogicInterface, RuleType


class LogicInterfaceRRL(ModuleInterface):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        self.W = nn.Parameter(0.5 * torch.rand(input_dim, output_dim))
        self.G = RRL_G(cfg.alpha, cfg.beta)
        self.P = RRL_P(cfg.gamma)

    def fuzzy_conjunction(self, x: Tensor) -> Tensor:
        return self.P(-self.G(1.0 - x) @ self.G(self.W))

    def fuzzy_disjunction(self, x: Tensor) -> Tensor:
        return 1.0 - self.P(-self.G(x) @ self.G(self.W))

    @property
    def Wb(self) -> Tensor:
        return Binarize.apply(self.W - 0.5)

    def bool_conjunction(self, x: Tensor) -> Tensor:
        return conjunction(x, self.Wb)

    def bool_disjunction(self, x: Tensor) -> Tensor:
        return disjunction(x, self.Wb)

    def edge_count(self) -> Tensor:
        return self.Wb.sum()

    def l1_norm(self) -> Tensor:
        return self.W.sum()

    def l2_norm(self) -> Tensor:
        return (self.W**2).sum()

    def clip(self) -> None:
        self.W.data.clamp_(0.0, 1.0)


class ConjunctionLayer(LogicInterfaceRRL):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(cfg, input_dim, output_dim)
        self.rule_types = [RuleType.AND for _ in range(output_dim)]

    def forward(self, x: Tensor) -> Tensor:
        return self.fuzzy_conjunction(x)

    def bool_forward(self, x: Tensor) -> Tensor:
        return self.bool_conjunction(x)


class DisjunctionLayer(LogicInterfaceRRL):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(cfg, input_dim, output_dim)
        self.rule_types = [RuleType.OR for _ in range(output_dim)]

    def forward(self, x: Tensor) -> Tensor:
        return self.fuzzy_disjunction(x)

    def bool_forward(self, x: Tensor) -> Tensor:
        return self.bool_disjunction(x)


class LogicLayerRRL(LogicInterface):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(cfg, input_dim, output_dim)
        self.logic_dim = output_dim // 2
        self.con_layer = ConjunctionLayer(cfg, input_dim, self.logic_dim)
        self.dis_layer = DisjunctionLayer(cfg, input_dim, self.logic_dim)

    # override
    def clear_activation(self) -> None:
        super().clear_activation()
        self.con_layer.clear_activation()
        self.dis_layer.clear_activation()

    def fuzzy_forward(self, x: Tensor) -> Tensor:
        return torch.cat((self.con_layer(x), self.dis_layer(x)), dim=1)

    def bool_forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            (
                self.con_layer.bool_forward_with_count(x),
                self.dis_layer.bool_forward_with_count(x),
            ),
            dim=1,
        )

    @property
    def Wb(self) -> Tensor:
        return torch.cat((self.con_layer.Wb, self.dis_layer.Wb), dim=1)

    @property
    def rule_types(self) -> list[RuleType]:
        return self.con_layer.rule_types + self.dis_layer.rule_types

    def l1_norm(self) -> Tensor:
        return self.con_layer.l1_norm() + self.dis_layer.l1_norm()

    def l2_norm(self) -> Tensor:
        return self.con_layer.l2_norm() + self.dis_layer.l2_norm()

    def clip(self) -> None:
        self.con_layer.clip()
        self.dis_layer.clip()

    # override
    def debug(self) -> None:
        # activation ratio
        activation_ratio = self.get_activation_ratio()
        logger.debug(activation_ratio)
        # activation rate
        and_act_rate = self.con_layer.get_activation_ratio().mean().item()
        or_act_rate = self.dis_layer.get_activation_ratio().mean().item()
        logger.debug(f'AND Act Rate: {and_act_rate}')
        logger.debug(f'OR Act Rate:  {or_act_rate}')
