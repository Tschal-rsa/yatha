import os
from abc import abstractmethod

import mne
from tqdm.autonotebook import tqdm

from config import Config
from preprocessors.preprocessor import Preprocessor
from utils import logger


class MycenaeanPreprocessor(Preprocessor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    @abstractmethod
    def get_file_label_map(self) -> dict[str, tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def preprocess_single_file(self, file: str) -> mne.io.Raw | None:
        raise NotImplementedError

    def get_decim(self, sfreq: float) -> int:
        return round(sfreq / self.cfg.preprocess.ideal_sfreq)

    def preprocess(self) -> None:
        os.makedirs(self.cfg.dataset.preprocess_path, exist_ok=True)
        for pid, (file, label) in tqdm(self.get_file_label_map().items(), desc='File'):
            logger.debug(f'{pid} {label}')
            raw = self.preprocess_single_file(file)
            if raw is None:
                logger.warning(f'{pid} is empty!')
                continue
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self.cfg.preprocess.crop_duration,
                overlap=self.cfg.preprocess.crop_overlap,
            )
            decim = self.get_decim(epochs.info['sfreq'])
            epochs.decimate(decim)
            epochs_file = os.path.join(
                self.cfg.dataset.preprocess_path, f'{pid}_{label}_epo.fif'
            )
            epochs.save(epochs_file, overwrite=True)
