from abc import ABC, abstractmethod

from config import Config


class Preprocessor(ABC):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def preprocess(self) -> None:
        raise NotImplementedError
