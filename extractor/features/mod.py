from config import Config
from utils import logger

from .connectivity import ConnectivityFeature
from .feature import Feature
from .frequency import FrequencyFeature
from .morlet import MorletFeature
from .omnis import OmnisFeature


def get_calculator(cfg: Config, feature: str) -> Feature:
    match feature:
        case 'frequency':
            return FrequencyFeature(cfg)
        case 'morlet':
            return MorletFeature(cfg)
        case 'connectivity':
            return ConnectivityFeature(cfg)
        case 'connectivity_whole':
            return ConnectivityFeature(cfg, 'whole')
        case 'mne':
            return OmnisFeature(cfg)
        case _feature:
            logger.critical(f'Unknown feature: {_feature}!')
            exit(1)
