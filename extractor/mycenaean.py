import os

import mne
import numpy as np
from tqdm.autonotebook import tqdm

from config import Config
from utils import logger

from .extractor import Extractor
from .features import get_calculator


class MycenaeanExtractor(Extractor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def get_file_label_map(self) -> dict[str, tuple[str, str]]:
        file_label_map = {}
        for name in os.listdir(self.cfg.dataset.preprocess_path):
            pid, label, *_ = name.split('_')
            file = os.path.join(self.cfg.dataset.preprocess_path, name)
            file_label_map[pid] = (file, label)
        return file_label_map

    def extract(self) -> None:
        os.makedirs(self.cfg.dataset.feature_path, exist_ok=True)
        for feature in self.cfg.preprocess.features:
            feature_file = os.path.join(self.cfg.dataset.feature_path, f'{feature}.npz')
            if os.path.exists(feature_file):
                logger.warning(f'Skipping {feature} feature extraction.')
                continue
            samples_dict = {}
            calculator = get_calculator(self.cfg, feature)
            for pid, (file, label) in tqdm(
                self.get_file_label_map().items(), desc=feature
            ):
                epochs = mne.read_epochs(file)
                data = calculator.calculate(epochs)
                samples_dict[f'{pid}_{label}'] = data
            np.savez_compressed(feature_file, **samples_dict)
