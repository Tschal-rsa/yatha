from collections import defaultdict
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from config import NeoConfig
from dataset import Normalizer

from .components import (
    BinarizeLayer,
    get_binarize_layer,
    Linear,
    LogicInterface,
    ModuleInterface,
    get_logic_layer,
)


class Net(ModuleInterface):
    def __init__(self, cfg: NeoConfig, dims: list[int], normalizer: Normalizer) -> None:
        super().__init__(dims[0], dims[-1])
        self.cfg = cfg
        self.dims = dims
        self.t = nn.Parameter(torch.log(torch.tensor(cfg.temp)))

        layer_list: list[ModuleInterface] = []
        prev_dim = dims[0]
        for i in range(1, len(dims)):
            layer: ModuleInterface
            if i == 1:
                layer = get_binarize_layer(cfg, prev_dim, dims[i], normalizer)
            elif i == len(dims) - 1:
                layer = Linear(cfg, prev_dim, dims[i])
            else:
                # FIXME: layer_use_not = True if i != 2 else False
                layer = get_logic_layer(cfg, prev_dim, dims[i])
            prev_dim = layer.output_dim
            layer_list.append(layer)
        self.layer_list = cast(list[ModuleInterface], nn.ModuleList(layer_list))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # override
    def clear_activation(self) -> None:
        # HACK: no need to call super().clear_activation()
        super().clear_activation()
        for layer in self.layer_list:
            layer.clear_activation()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layer_list:
            x = layer(x)
        # HACK: should exp, because t = log(temp)
        return x / torch.exp(self.t)

    @torch.no_grad()
    def bool_forward(self, x: Tensor) -> Tensor:
        for layer in self.layer_list:
            x = layer.bool_forward_with_count(x)
        return x

    @torch.no_grad()
    def edge_count(self) -> Tensor:
        count = torch.tensor(0.0, device=self.device)
        for layer in self.layer_list:
            count += layer.edge_count()
        return count

    def l1_norm(self) -> Tensor:
        norm = torch.tensor(0.0, device=self.device)
        for layer in self.layer_list:
            norm += layer.l1_norm()
        return norm

    def l2_norm(self) -> Tensor:
        norm = torch.tensor(0.0, device=self.device)
        for layer in self.layer_list:
            norm += layer.l2_norm()
        return norm

    def clip(self) -> None:
        for layer in self.layer_list:
            layer.clip()

    @torch.no_grad()
    def print_rules(self, feature_names: list[str], label_names: list[str]) -> None:
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            if isinstance(layer, BinarizeLayer):
                layer.get_bound_name(feature_names)
            elif isinstance(layer, LogicInterface):
                layer.get_rules(self.layer_list[i - 1], i > 1)

        linear = cast(Linear, self.layer_list[-1])
        prev_layer = self.layer_list[-2]
        linear.get_rule2weights(prev_layer)

        rules: list[list[str]] = []
        head: list[str] = ['RID']
        for i, ln in enumerate(label_names):
            head.append(f'{ln}(b={linear.bl[i]:.4f})')
        head += ['Support', 'Rule']
        rules.append(head)

        for rid, w in linear.rule2weights:
            rule: list[str] = [f'{rid}']
            for li in range(len(label_names)):
                rule.append(f'{w[li]:.4f}')
            act_rate = (
                prev_layer.node_activation_cnt[linear.rid2dim[rid]]
                / prev_layer.forward_tot
            ).item()
            rule += [f'{act_rate:.4f}', prev_layer.rule_names[rid]]
            rules.append(rule)

        rules.append(['#' * 60])
        rule_str = '\n'.join('\t'.join(r) for r in rules)
        with open(self.cfg.rrl_file, 'w', encoding='utf-8') as f:
            f.write(rule_str)

    def get_complexity(self) -> float:
        edge_cnt = 0
        connected_rid: dict[int, set[int]] = defaultdict(set)
        for ln in range(len(self.layer_list) - 1, 0, -1):
            layer = self.layer_list[ln]
            if isinstance(layer, Linear):
                for rid, _ in layer.rule2weights:
                    connected_rid[ln - 1].add(rid)
            elif isinstance(layer, LogicInterface):
                for rid in connected_rid[ln]:
                    rule = layer.rule_list[rid].rids
                    edge_cnt += len(rule)
                    for r in rule:
                        connected_rid[ln - 1].add(r)
        return np.log(edge_cnt).item()

    # override
    def debug(self) -> None:
        for layer in self.layer_list:
            layer.debug()
