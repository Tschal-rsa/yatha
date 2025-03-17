from config import NeoConfig
from utils import logger

from .aiwa import LogicLayerAIWA
from .awa import LogicLayerAWA
from .interface import LogicInterface
from .nfrl import LogicLayerNFRL
from .rrl import LogicLayerRRL
from .arrl import LogicLayerAndnessRRL


def get_logic_layer(cfg: NeoConfig, input_dim: int, output_dim: int, use_not: bool) -> LogicInterface:
    logger.warning(f'Initializing {cfg.logic_name} logic layer.')
    match cfg.logic_name:
        case 'RRL':
            return LogicLayerRRL(cfg, input_dim, output_dim, use_not)
        case 'AndnessRRL':
            return LogicLayerAndnessRRL(cfg, input_dim, output_dim, use_not)
        case 'NFRL':
            return LogicLayerNFRL(cfg, input_dim, output_dim, use_not)
        case 'AIWA':
            return LogicLayerAIWA(cfg, input_dim, output_dim, use_not)
        case 'AWA':
            return LogicLayerAWA(cfg, input_dim, output_dim, use_not)
        case _name:
            logger.critical(f'Unknown logic layer name: {_name}!')
            exit(1)
