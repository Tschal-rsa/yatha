from config import Config
from utils import logger

from .ahepa import ArchaicAHEPA
from .archaic import ArchaicPreprocessor
from .pumc import ArchaicPUMC


def get_archaic_preprocessor(cfg: Config) -> ArchaicPreprocessor:
    match cfg.dataset.name:
        case 'AHEPA':
            return ArchaicAHEPA(cfg)
        case x if x.startswith('PUMC'):
            return ArchaicPUMC(cfg)
        case _dataset:
            logger.critical(f'Unknown dataset: {_dataset}!')
            exit(1)
