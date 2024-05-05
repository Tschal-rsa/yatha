from tqdm.contrib.logging import logging_redirect_tqdm

from config import Config
from extractor import get_extractor
from preprocessors import get_preprocessor
from trainer import Trainer
from utils import logger, set_cuda, set_seed, set_verbosity, get_working_dir


class Interface:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.objective: float | None = None
        set_cuda(cfg.train.device_ids)
        set_seed(cfg.train.seed)
        set_verbosity()

    def preprocess(self) -> None:
        preprocessor = get_preprocessor(self.cfg)
        preprocessor.preprocess()

    def extract(self) -> None:
        extractor = get_extractor(self.cfg)
        extractor.extract()

    def train(self) -> None:
        trainer = Trainer(self.cfg)
        self.objective = trainer.train()

    def run(self) -> None:
        with logging_redirect_tqdm():
            match self.cfg.run:
                case 'preprocess':
                    self.preprocess()
                case 'extract':
                    self.extract()
                case 'train':
                    self.train()
                case _run:
                    logger.error(f'Unknown command: {_run}!')
            logger.warning(get_working_dir())
