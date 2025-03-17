import os
from abc import abstractmethod

import mne
from tqdm.autonotebook import tqdm

from config import Config
from preprocessors.preprocessor import Preprocessor
from utils import logger


class ArchaicPreprocessor(Preprocessor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    @abstractmethod
    def get_file_label_map(self) -> dict[str, tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def preprocess_single_file(self, file: str) -> mne.io.Raw | mne.Epochs | None:
        raise NotImplementedError

    def get_decim(self, sfreq: float) -> int:
        return round(sfreq / self.cfg.preprocess.ideal_sfreq)

    def preprocess(self) -> None:
        os.makedirs(self.cfg.dataset.preprocess_path, exist_ok=True)
        for pid, (file, label) in tqdm(self.get_file_label_map().items(), desc='File'):
            logger.debug(f'{pid} {label}')
            signal = self.preprocess_single_file(file)
            if signal is None:
                logger.warning(f'{pid} is empty!')
                continue
            signal_dir = os.path.join(self.cfg.dataset.preprocess_path, f'{pid}_{label}')
            os.makedirs(signal_dir, exist_ok=True)
            if isinstance(signal, mne.io.Raw):
                max_time = (signal.n_times - 1) / signal.info['sfreq']
                tmin, tmax = 0.0, self.cfg.preprocess.crop_duration
                idx = 0
                while tmax <= max_time:
                    segment_file = os.path.join(signal_dir, f'{idx}_raw.fif')
                    signal.save(segment_file, tmin=tmin, tmax=tmax, overwrite=True)
                    tmin, tmax = tmax, tmax + self.cfg.preprocess.crop_duration
                    idx += 1
            elif isinstance(signal, mne.Epochs):
                for idx, epochs_data in enumerate(signal):
                    raw_epoch = mne.io.RawArray(epochs_data, signal.info)
                    segment_file = os.path.join(signal_dir, f'{idx}_raw.fif')
                    raw_epoch.save(segment_file, overwrite=True)
