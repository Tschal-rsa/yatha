from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
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


class RecordROC:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tprs: list[NDArray] = []
        self.aucs: list[float] = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.fig, ax = plt.subplots(figsize=(6, 6))
        self.ax = cast(Axes, ax)

    def record(self, y_true: NDArray, y_pos_score: NDArray, fold: int) -> float:
        viz = RocCurveDisplay.from_predictions(
            y_true,
            y_pos_score,
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=self.ax,
            plot_chance_level=(fold == self.cfg.train.n_splits),
        )
        interp_tpr = np.interp(self.mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(viz.roc_auc)
        return viz.roc_auc

    def finish(self) -> tuple[float, float]:
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        self.ax.plot(
            self.mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        self.ax.fill_between(
            self.mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        self.ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Mean ROC curve with cross validation",
        )
        self.ax.legend(loc="lower right")
        self.fig.tight_layout()
        self.fig.savefig("roc.png", dpi=1000)

        return mean_auc, std_auc.item()


class Recorder:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.result = Record[list[float]]([], [], [], [], [], [], None)
        self.roc = RecordROC(cfg)

    def record(self, y_true: NDArray, y_score: NDArray, complexity: float) -> None:
        y_pred = y_score.argmax(axis=1)
        y_pos_score = y_score[:, 1]
        self.result.auc.append(
            self.roc.record(y_true, y_pos_score, len(self.result.auc) + 1)
            if self.cfg.train.plot_roc
            else roc_auc_score(
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
        if self.cfg.train.plot_roc:
            mean_auc, std_auc = self.roc.finish()
        else:
            mean_auc, std_auc = (
                np.mean(self.result.auc).item(),
                np.std(self.result.auc).item(),
            )
        mean = Record(
            mean_auc,
            np.mean(self.result.acc).item(),
            np.mean(self.result.precision).item(),
            np.mean(self.result.recall).item(),
            np.mean(self.result.f1).item(),
            np.mean(self.result.complexity).item(),
            self.result.cm,
        )
        std = Record(
            std_auc,
            np.std(self.result.acc).item(),
            np.std(self.result.precision).item(),
            np.std(self.result.recall).item(),
            np.std(self.result.f1).item(),
            np.std(self.result.complexity).item(),
            None,
        )
        logger.info(f'AVG AUC: {mean.auc * 100:.2f}% ({std.auc * 100:.2f}%)')
        logger.info(f'AVG ACC: {mean.acc * 100:.2f}% ({std.acc * 100:.2f}%)')
        logger.info(f'AVG P:   {mean.precision * 100:.2f}% ({std.precision * 100:.2f}%)')
        logger.info(f'AVG R:   {mean.recall * 100:.2f}% ({std.recall * 100:.2f}%)')
        logger.info(f'AVG F1:  {mean.f1 * 100:.2f}% ({std.f1 * 100:.2f}%)')
        logger.info(f'AVG COM: {mean.complexity:.4f} ({std.complexity:.4f})')
        logger.info(f'Confusion matrix:\n{mean.cm}')
        return mean
