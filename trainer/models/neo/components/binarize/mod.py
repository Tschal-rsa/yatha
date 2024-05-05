from config import NeoConfig
from dataset import Normalizer
from utils import logger

from .binarize import BinarizeLayer
from .nfrl import BinarizeLayerNFRL


def get_binarize_layer(
    cfg: NeoConfig, input_dim: int, num_binarization: int, normalizer: Normalizer
) -> BinarizeLayer:
    match cfg.logic_name:
        case 'NFRL':
            logger.warning(f'Initializing {cfg.logic_name} binarize layer.')
            return BinarizeLayerNFRL(input_dim, num_binarization, normalizer)
        case _name:
            logger.warning(f'Getting common binarize layer for model: {_name}.')
            return BinarizeLayer(input_dim, num_binarization, normalizer)
