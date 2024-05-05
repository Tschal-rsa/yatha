from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import Config
from utils import logger

T = TypeVar('T', list[float], float)


@dataclass
class Record(Generic[T]):
    auc: T
    acc: T
    precision: T
    recall: T
    f1: T
    complexity: T
    cm: NDArray | None = None


class Recorder:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.result = Record[list[float]]([], [], [], [], [], [], None)

    def record(self, y_true: NDArray, y_score: NDArray, complexity: float) -> None:
        y_pred = y_score.argmax(axis=1)
        y_pos_score = y_score[:, 1]
        self.result.auc.append(
            roc_auc_score(
                y_true,
                y_pos_score if self.cfg.dataset.num_class <= 2 else y_score,
                average='macro',
                multi_class='ovr',
            )
        )
        self.result.acc.append(accuracy_score(y_true, y_pred))
        self.result.precision.append(precision_score(y_true, y_pred, average='macro'))
        self.result.recall.append(recall_score(y_true, y_pred, average='macro'))
        self.result.f1.append(f1_score(y_true, y_pred, average='macro'))
        self.result.complexity.append(complexity)
        if self.result.cm is None:
            self.result.cm = confusion_matrix(y_true, y_pred)
        else:
            self.result.cm += confusion_matrix(y_true, y_pred)
        logger.info(f'AUC: {self.result.auc[-1] * 100:.2f}%')
        logger.info(f'ACC: {self.result.acc[-1] * 100:.2f}%')
        logger.info(f'P:   {self.result.precision[-1] * 100:.2f}%')
        logger.info(f'R:   {self.result.recall[-1] * 100:.2f}%')
        logger.info(f'F1:  {self.result.f1[-1] * 100:.2f}%')
        logger.info(f'COM: {self.result.complexity[-1]:.4f}')

    def finish(self) -> Record[float]:
        final = Record(
            np.mean(self.result.auc).item(),
            np.mean(self.result.acc).item(),
            np.mean(self.result.precision).item(),
            np.mean(self.result.recall).item(),
            np.mean(self.result.f1).item(),
            np.mean(self.result.complexity).item(),
            self.result.cm,
        )
        logger.info(f'AVG AUC: {final.auc * 100:.2f}%')
        logger.info(f'AVG ACC: {final.acc * 100:.2f}%')
        logger.info(f'AVG P:   {final.precision * 100:.2f}%')
        logger.info(f'AVG R:   {final.recall * 100:.2f}%')
        logger.info(f'AVG F1:  {final.f1 * 100:.2f}%')
        logger.info(f'AVG COM: {final.complexity:.4f}')
        logger.info(f'Confusion matrix:\n{final.cm}')
        return final
