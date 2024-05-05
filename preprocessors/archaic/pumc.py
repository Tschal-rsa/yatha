import os

import mne

import const
from config import Config

from .archaic import ArchaicPreprocessor


class ArchaicPUMC(ArchaicPreprocessor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def get_file_label_map(self) -> dict[str, tuple[str, str]]:
        file_label_map = {}
        for root, dirs, files in os.walk(self.cfg.dataset.root):
            for name in files:
                pid, ext = os.path.splitext(name)
                if ext != '.edf':
                    continue
                path = os.path.join(root, name)
                group = os.path.basename(root)
                file_label_map[pid] = (path, group)
        return file_label_map

    @staticmethod
    def get_mapping(ch_name: str) -> str:
        return ch_name[4:-4]

    def preprocess_single_file(self, file: str) -> mne.io.Raw | None:
        raw = mne.io.read_raw_edf(file, preload=True, encoding='latin1')
        if 'EEG Fp1-Ref' in raw.info['ch_names']:
            raw.pick(const.Preprocess.ch_names_0)
        elif 'EEG Fp1-REF' in raw.info['ch_names']:
            raw.pick(const.Preprocess.ch_names_1)
        else:
            return None
        raw.rename_channels(self.get_mapping)
        raw.set_montage('standard_alphabetic')

        try:
            raw.crop(tmin=self.cfg.preprocess.crop_prefix)
        except ValueError:
            # discard signals not long enough to crop
            return None
        raw.filter(0.1, 50, method='fir', n_jobs='cuda')
        # FIXME: change this into Epochs.decimate() later
        raw.resample(self.cfg.preprocess.ideal_sfreq, n_jobs='cuda')
        raw.notch_filter(50, method='spectrum_fit', filter_length='10s')
        raw.set_eeg_reference(['A1', 'A2'])
        raw.drop_channels(['A1', 'A2'])

        return raw
