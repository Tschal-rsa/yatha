from abc import ABC, abstractmethod

from config import Config


class Extractor(ABC):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def extract(self) -> None:
        raise NotImplementedError
