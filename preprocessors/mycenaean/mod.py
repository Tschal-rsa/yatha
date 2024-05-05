from config import Config
from utils import logger

from .ahepa import MycenaeanAHEPA
from .mycenaean import MycenaeanPreprocessor
from .pumc import MycenaeanPUMC


def get_mycenaean_preprocessor(cfg: Config) -> MycenaeanPreprocessor:
    match cfg.dataset.name:
        case 'AHEPA':
            return MycenaeanAHEPA(cfg)
        case x if x.startswith('PUMC'):
            return MycenaeanPUMC(cfg)
        case _dataset:
            logger.critical(f'Unknown dataset: {_dataset}!')
            exit(1)
