from abc import ABC, abstractmethod

import mne
from numpy.typing import NDArray

from config import Config


class Feature(ABC):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def calculate(self, epochs: mne.Epochs) -> NDArray:
        raise NotImplementedError
