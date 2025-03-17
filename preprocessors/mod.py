from config import Config
from utils import logger

from .archaic import get_archaic_preprocessor
from .mycenaean import get_mycenaean_preprocessor
from .preprocessor import Preprocessor
from .proto import get_proto_preprocessor


def get_preprocessor(cfg: Config) -> Preprocessor:
    logger.warning(
        f'Initializing {cfg.dataset.preprocess_name} preprocessor for {cfg.dataset.name}.'
    )
    match cfg.dataset.preprocess_name:
        case 'Proto':
            return get_proto_preprocessor(cfg)
        case 'Mycenaean':
            return get_mycenaean_preprocessor(cfg)
        case x if x.startswith('Archaic'):
            return get_archaic_preprocessor(cfg)
        case _name:
            logger.critical(f'Unknown preprocess name: {_name}!')
            exit(1)
