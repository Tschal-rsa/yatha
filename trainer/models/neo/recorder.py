from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
import torch
from config import NeoConfig
from utils import logger


class Recorder:
    def __init__(self, cfg: NeoConfig) -> None:
        self.cfg = cfg
        self.writer = SummaryWriter(cfg.folder_path)
        self.epo = -1
        self.cnt = -1
        self.best_auc = -1.0
        self.auc = -1.0
        self.avg_batch_loss = 0.0
        self.epoch_loss = 0.0
        self.abs_gradient_max = 0.0
        self.abs_gradient_avg = 0.0
        self.ba_cnt = 0

    def loss_step(self, loss: Tensor) -> None:
        self.cnt += 1
        self.ba_cnt += 1
        loss_item = loss.item()
        self.epoch_loss += loss_item
        self.avg_batch_loss += loss_item

    def grad_step(self, net: nn.Module) -> None:
        for param in net.parameters():
            assert param.grad is not None
            if torch.isnan(param.grad).any():
                print('Param Grad NaN!')
                exit(1)
            elif torch.isinf(param.grad).any():
                print('Param Grad Inf!')
                exit(1)
            elif param.grad.abs().max() > 10:
                print('Param Grad To High:', param.grad.abs().max())
                exit(1)
            self.abs_gradient_max = max(
                self.abs_gradient_max, param.grad.abs().max().item()
            )
            self.abs_gradient_avg += param.grad.abs().sum().item() / param.grad.numel()

    def loss_and_grad_step(self, loss: Tensor, net: nn.Module) -> None:
        self.loss_step(loss)
        self.grad_step(net)

    def should_eval(self) -> bool:
        return self.cnt % self.cfg.log_iter == 0 and self.cnt != 0

    def record_avg_batch_loss_and_edge_count(self, edge_count: Tensor) -> None:
        self.writer.add_scalar(
            'Avg_Batch_Loss_GradGrafting',
            self.avg_batch_loss / self.cfg.log_iter,
            self.cnt,
        )
        self.avg_batch_loss = 0.0
        self.writer.add_scalar('Edge_penalty/Log', edge_count.log().item(), self.cnt)
        self.writer.add_scalar('Edge_penalty/Origin', edge_count.item(), self.cnt)

    def record_metrics(self, y_true: NDArray, y_score: NDArray, split: str) -> None:
        y_pred = y_score.argmax(axis=1)
        y_pos_score = y_score[:, 1]
        auc = roc_auc_score(
            y_true,
            y_pos_score if y_score.shape[1] <= 2 else y_score,
            average='macro',
            multi_class='ovr',
        )
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        self.writer.add_scalar(f'{split} AUC', auc, self.cnt)
        self.writer.add_scalar(f'{split} Accuracy', acc, self.cnt)
        self.writer.add_scalar(f'{split} F1_Score', f1, self.cnt)
        self.auc = auc
        logger.info(f'{split} AUC: {self.auc * 100:.2f}%')

    def is_best(self) -> bool:
        if self.auc > self.best_auc:
            self.best_auc = self.auc
            logger.warning(f'New best AUC: {self.auc * 100:.2f}%')
            return True
        return False

    def record_loss_and_grad(self) -> None:
        logger.debug(f'Epoch: {self.epo}\tloss: {self.epoch_loss:.4f}')
        self.writer.add_scalar('Training_Loss', self.epoch_loss, self.epo)
        self.writer.add_scalar('Abs_Gradient_Max', self.abs_gradient_max, self.epo)
        self.writer.add_scalar(
            'Abs_Gradient_Avg', self.abs_gradient_avg / self.ba_cnt, self.epo
        )

    def reset(self) -> None:
        self.epoch_loss = 0.0
        self.abs_gradient_max = 0.0
        self.abs_gradient_avg = 0.0
        self.ba_cnt = 0
        self.epo += 1
