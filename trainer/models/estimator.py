from abc import ABC, abstractmethod

from numpy.typing import NDArray

from config import Config
from dataset import Samples


class Estimator(ABC):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def fit(
        self, fold: int, train_samples: Samples, val_samples: Samples | None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, samples: Samples) -> tuple[NDArray, float]:
        raise NotImplementedError
