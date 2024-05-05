from config import Config
from utils import logger

from .estimator import Estimator
from .neo_rrl import NeoEstimator
from .rrl import RRLEstimator
from .xgb import XGBoostEstimator
from trainer.spliters import Spliter


def get_estimator(cfg: Config, spliter: Spliter) -> Estimator:
    match cfg.model.name:
        case 'XGBoost':
            return XGBoostEstimator(cfg)
        case 'RRL':
            return RRLEstimator(cfg)
        case 'Neo-RRL':
            return NeoEstimator(cfg, spliter)
        case _name:
            logger.critical(f'Unknown estimator: {_name}!')
            exit(1)
