from config import Config
from utils import logger

from .rrl import RRLSpliter
from .spliter import Spliter


def get_spliter(cfg: Config) -> Spliter:
    match cfg.model.name:
        case 'RRL':
            return RRLSpliter(cfg)
        case _name:
            logger.warning(f'Getting common spliter for model: {_name}.')
            return Spliter(cfg)
