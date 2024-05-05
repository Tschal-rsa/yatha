from typing import Any, Callable, cast

import os
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import trange

from config import NeoConfig
from dataset import Normalizer, Samples

from .net import Net
from .recorder import Recorder


class Neo:
    def __init__(self, cfg: NeoConfig, dims: list[int], normalizer: Normalizer) -> None:
        self.cfg = cfg
        self.net = Net(cfg, dims, normalizer)

    @staticmethod
    def set_eval(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: 'Neo', *args, **kw) -> Any:
            self.net.eval()
            ret = func(self, *args, **kw)
            self.net.train()
            return ret

        return wrapper

    def get_dataloader(self, *arrays: NDArray, shuffle: bool = False) -> DataLoader:
        dataset = TensorDataset(
            *(torch.tensor(array, dtype=torch.float) for array in arrays)
        )
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, pin_memory=True
        )

    @set_eval
    def count_activation(self, dataloader: DataLoader) -> None:
        self.net.clear_activation()
        for batch in dataloader:
            X = cast(tuple[Tensor, ...], batch)[0]
            X = X.to(self.net.device)
            self.net.bool_forward(X)

    def save_state_dict(self, dataloader: DataLoader) -> None:
        self.count_activation(dataloader)
        torch.save(self.net.state_dict(), self.cfg.model)

    def load_state_dict(self) -> None:
        self.net.load_state_dict(torch.load(self.cfg.model))
    
    def del_state_dict(self) -> None:
        if os.path.isfile(self.cfg.model):
            os.remove(self.cfg.model)

    def linear_scheduler_with_warmup(self, epoch: int) -> float:
        return 1.0 + (
            (self.cfg.max_lr / self.cfg.init_lr - 1.0) * (epoch / self.cfg.warmup_epoch)
            if epoch < self.cfg.warmup_epoch
            else (self.cfg.max_lr / self.cfg.init_lr - 1.0)
            * (self.cfg.epoch - epoch)
            / (self.cfg.epoch - self.cfg.warmup_epoch)
        )

    def get_lr_scheduler(self, optimizer: optim.Optimizer) -> LambdaLR:
        return LambdaLR(
            optimizer, lambda epoch: self.linear_scheduler_with_warmup(epoch)
        )

    def train(self, train_samples: Samples, val_samples: Samples | None) -> None:
        dataloader = self.get_dataloader(
            train_samples.data, train_samples.label, shuffle=True
        )
        criterion = nn.CrossEntropyLoss().to(self.net.device)
        # [ ] different learning rates
        # optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.init_lr)
        optimizer = optim.Adam(
            [
                {
                    'params': [
                        param
                        for name, param in self.net.named_parameters()
                        if 'andness' not in name
                    ],
                    'lr': self.cfg.init_lr,
                },
                {
                    'params': [
                        param
                        for name, param in self.net.named_parameters()
                        if 'andness' in name
                    ],
                    'lr': self.cfg.andness_init_lr,
                },
            ]
        )
        scheduler = self.get_lr_scheduler(optimizer)
        recorder = Recorder(self.cfg)
        self.net.clip()
        for epo in trange(self.cfg.epoch, desc='Epoch'):
            recorder.reset()
            for batch in dataloader:
                X, y = cast(tuple[Tensor, Tensor], batch)
                X = X.to(self.net.device, non_blocking=True)
                y = y.to(self.net.device, torch.long, non_blocking=True)
                optimizer.zero_grad()
                y_pred = self.net(X)
                loss = criterion(y_pred, y) + self.cfg.weight_decay * self.net.l2_norm()
                if torch.isnan(loss).any():
                    print('Loss NaN')
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 10)

                recorder.loss_and_grad_step(loss, self.net)
                if recorder.should_eval():
                    edge_count = self.net.edge_count()
                    recorder.record_avg_batch_loss_and_edge_count(edge_count)
                    if val_samples is None:
                        y_score = self.test(train_samples.data)
                        recorder.record_metrics(train_samples.label, y_score, 'Train')
                    else:
                        y_score = self.test(val_samples.data)
                        recorder.record_metrics(val_samples.label, y_score, 'Val')
                        if recorder.is_best():
                            self.save_state_dict(dataloader)

                optimizer.step()
                self.net.clip()
            recorder.record_loss_and_grad()
            scheduler.step()
        if val_samples is None:
            self.save_state_dict(dataloader)

    @torch.no_grad()
    @set_eval
    def test(self, data: NDArray) -> NDArray:
        dataloader = self.get_dataloader(data, shuffle=False)
        y_pred_list = []
        for batch in dataloader:
            (X,) = cast(tuple[Tensor], batch)
            X = X.to(self.net.device, non_blocking=True)
            y_pred = self.net(X)
            y_pred_list.append(y_pred)
        y_pred_all = torch.cat(y_pred_list)
        y_score = torch.softmax(y_pred_all, dim=1)
        return y_score.numpy(force=True)

    def print_rule(self, feature_names: list[str], label_names: list[str]) -> None:
        self.net.print_rules(feature_names, label_names)

    def get_complexity(self) -> float:
        return self.net.get_complexity()

    def debug(self) -> None:
        self.net.debug()
