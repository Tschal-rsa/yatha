from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor
from numpy.typing import NDArray
from config import NeoConfig
from ..binarize import BinarizeLayer
from ..dropout import Dropout
from ..functions import GradGraft
from ..interface import ModuleInterface


@dataclass(order=True, frozen=True)
class RuleUnit:
    rid: int
    neg: bool


class RuleType(Enum):
    AND = '&'
    OR = '|'


@dataclass(frozen=True)
class Rule:
    runits: tuple[RuleUnit, ...]
    rtype: RuleType

    def get_description(self, input_rule_names: list[str], wrap: bool = False) -> str:
        name = ''
        for i, ru in enumerate(self.runits):
            op_str = f' {self.rtype.value} ' if i > 0 else ''
            not_str = '~' if ru.neg else ''
            var_str = input_rule_names[ru.rid]
            if wrap or ru.neg:
                var_str = f'({var_str})'
            name += f'{op_str}{not_str}{var_str}'
        return name


class LogicInterface(ModuleInterface):
    def __init__(self, cfg: NeoConfig, input_dim: int, output_dim: int, use_not: bool) -> None:
        if use_not:
            input_dim *= 2
        super().__init__(input_dim, output_dim)
        self.use_not = use_not
        self.dropout = Dropout(cfg.dropout_p)
        self.rule_list: list[Rule] = []

    @abstractmethod
    def fuzzy_forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        res_tilde = self.fuzzy_forward(x)
        with torch.no_grad():
            res_bar = self.bool_forward(x)
            res_bar = self.dropout(res_bar)
        return GradGraft.apply(res_bar, res_tilde)
    
    # override
    def bool_forward_with_count(self, x: Tensor) -> Tensor:
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return super().bool_forward_with_count(x)

    @property
    @abstractmethod
    def Wb(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def rule_types(self) -> list[RuleType]:
        raise NotImplementedError

    def edge_count(self) -> Tensor:
        return self.Wb.sum()

    def get_rules(self, prev_layer: ModuleInterface, wrap: bool = False) -> None:
        self.dim2id = []
        prev_dim2id = prev_layer.dim2id
        rule2id: dict[Rule, int] = {}
        tmp_id: int = 0
        self.rule_list = []

        bounds: NDArray | None = None
        if isinstance(prev_layer, BinarizeLayer):
            bounds = torch.cat(
                (prev_layer.cl.t().reshape(-1), prev_layer.cl.t().reshape(-1))
            ).numpy(force=True)

        # ri: range(output_dim), index of a logic neuron
        Wb: NDArray = self.Wb.t().type(torch.int).numpy(force=True)
        for ri, (row, op) in enumerate(zip(Wb, self.rule_types)):
            if (
                self.node_activation_cnt[ri] == 0
                or self.node_activation_cnt[ri] == self.forward_tot
            ):
                self.dim2id.append(-1)
                continue
            runits: set[RuleUnit] = set()
            # feature id -> input id / rule id (identical in the binarize layer)
            feature2id: dict[int, int] = {}
            # ci: range(input_dim), index of a input neuron
            for ci, w in enumerate(row):
                neg = False
                if self.use_not and ci >= (half_input_dim := self.input_dim // 2):
                    neg = True
                    ci -= half_input_dim
                if w > 0 and prev_dim2id[ci] != -1:
                    if isinstance(prev_layer, BinarizeLayer):
                        # merge the bounds for one feature
                        # input_dim = 2 * num_features * num_binarization
                        # ci is also a rule id
                        assert bounds is not None
                        fi = ci // prev_layer.num_binarization
                        if fi not in feature2id.keys():
                            feature2id[fi] = ci
                            runits.add(RuleUnit(ci, False))
                        else:
                            if (ci < bounds.shape[0] // 2 and op == RuleType.AND) or (
                                ci >= bounds.shape[0] // 2 and op == RuleType.OR
                            ):
                                func = max
                            else:
                                func = min
                            new_bound = func(bounds[feature2id[fi]], bounds[ci])
                            if new_bound == bounds[ci]:
                                runits.remove(RuleUnit(feature2id[fi], False))
                                runits.add(RuleUnit(ci, False))
                                feature2id[fi] = ci
                    else:
                        runits.add(RuleUnit(prev_dim2id[ci], neg))

            rule = Rule(tuple(sorted(runits)), op)
            if rule not in rule2id.keys():
                rule2id[rule] = tmp_id
                self.rule_list.append(rule)
                self.dim2id.append(tmp_id)
                tmp_id += 1
            else:
                self.dim2id.append(rule2id[rule])

        self.rule_names = [
            rule.get_description(prev_layer.rule_names, wrap) for rule in self.rule_list
        ]
