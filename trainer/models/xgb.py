from typing import Any, cast

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from omegaconf import OmegaConf

from config import Config, XGBoostConfig
from dataset import Samples
from utils import logger

from .estimator import Estimator


class XGBLogging(xgb.callback.TrainingCallback):
    def __init__(self):
        pass

    def after_iteration(self, model, epoch, evals_log):
        log_list = [f'[{epoch}]']
        for data, metric in evals_log.items():
            for m_key, m_value in metric.items():
                log_list.append(f'{data}-{m_key}:{m_value[-1]:.5f}')
        logger.debug('\t'.join(log_list))
        return False


class XGBoostEstimator(Estimator):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.num_class = cfg.dataset.num_class
        self.xgb_cfg = cast(XGBoostConfig, cfg.model)
        self.param = cast(dict[str, Any], OmegaConf.to_object(self.xgb_cfg.param))
        self.update_param()

    def update_param(self) -> None:
        if self.num_class <= 2:
            self.param.update(
                {
                    'objective': 'binary:logistic',
                    'eval_metric': ['auc'],
                }
            )
        else:
            self.param.update(
                {
                    'objective': 'multi:softprob',
                    'num_class': self.num_class,
                }
            )

    def fit(
        self, fold: int, train_samples: Samples, val_samples: Samples | None
    ) -> None:
        dtrain = xgb.DMatrix(train_samples.data, train_samples.label)
        evals = [(dtrain, 'train')]
        if val_samples is not None:
            dtest = xgb.DMatrix(val_samples.data, val_samples.label)
            evals.append((dtest, 'eval'))
        callbacks = [XGBLogging()]
        self.bst = xgb.train(
            self.param,
            dtrain,
            self.xgb_cfg.num_round,
            evals=evals,
            verbose_eval=False,
            callbacks=callbacks,
        )

    def predict(self, samples: Samples) -> tuple[NDArray, float]:
        dtest = xgb.DMatrix(samples.data, samples.label)
        y_score = self.bst.predict(dtest)
        if self.num_class <= 2:
            y_score = np.stack([1 - y_score, y_score], axis=-1)
        else:
            y_score = y_score.reshape(samples.data.shape[0], -1)
        return y_score, 0.0
