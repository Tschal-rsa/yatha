from config import Config
from utils import logger

from .archaic import ArchaicExtractor
from .extractor import Extractor
from .mycenaean import MycenaeanExtractor
from .proto import ProtoExtractor

def get_extractor(cfg: Config) -> Extractor:
    logger.warning(
        f'Initializing {cfg.dataset.preprocess_name} extractor for {cfg.dataset.name}.'
    )
    match cfg.dataset.preprocess_name:
        case 'Proto':
            return ProtoExtractor(cfg)
        case 'Mycenaean':
            return MycenaeanExtractor(cfg)
        case 'Archaic':
            return ArchaicExtractor(cfg)
        case _name:
            logger.critical(f'Unknown preprocess name: {_name}!')
            exit(1)
