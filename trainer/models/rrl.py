from typing import cast

from numpy.typing import NDArray
from omegaconf import OmegaConf

from config import Config, RRLConfig
from dataset import Samples
from repos.rrl import test_model, train_main

from .estimator import Estimator


class RRLEstimator(Estimator):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.rrl_cfg = cast(RRLConfig, cfg.model)

    def fit(
        self, fold: int, train_samples: Samples, val_samples: Samples | None
    ) -> None:
        self.rrl_cfg.ith_kfold = fold
        train_main(cast(RRLConfig, OmegaConf.to_object(self.rrl_cfg)))

    def predict(self, samples: Samples) -> tuple[NDArray, float]:
        y_pred, complexity = test_model(cast(RRLConfig, OmegaConf.to_object(self.rrl_cfg)))
        return y_pred, complexity
