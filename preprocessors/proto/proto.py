import os
from abc import abstractmethod

from config import Config
from preprocessors.preprocessor import Preprocessor
from repos.proto import transform


class ProtoPreprocessor(Preprocessor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    @abstractmethod
    def get_source_output_pairs(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def preprocess(self) -> None:
        for source_base, output_base in self.get_source_output_pairs():
            os.makedirs(output_base, exist_ok=True)
            transform(source_base, output_base)
