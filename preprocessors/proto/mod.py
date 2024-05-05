from config import Config
from utils import logger

from .proto import ProtoPreprocessor
from .pumc import ProtoPUMC


def get_proto_preprocessor(cfg: Config) -> ProtoPreprocessor:
    match cfg.dataset.name:
        # TODO: implement Proto preprocessor for AHEPA.
        case x if x.startswith('PUMC'):
            return ProtoPUMC(cfg)
        case _dataset:
            logger.critical(f'Unknown dataset: {_dataset}!')
            exit(1)
