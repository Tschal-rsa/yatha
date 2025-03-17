import os

import mne
import pandas as pd

from config import Config

from .archaic import ArchaicPreprocessor


class ArchaicAHEPA(ArchaicPreprocessor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def get_file_label_map(self) -> dict[str, tuple[str, str]]:
        participants = pd.read_csv(
            os.path.join(self.cfg.dataset.root, 'participants.tsv'),
            sep='\t',
        )
        return {
            pid: (
                os.path.join(
                    self.cfg.dataset.root,
                    'derivatives',
                    pid,
                    'eeg',
                    f'{pid}_task-eyesclosed_eeg.set',
                ),
                group,
            )
            for pid, group in zip(participants['participant_id'], participants['Group'])
        }

    def preprocess_single_file(self, file: str) -> mne.io.Raw | mne.Epochs | None:
        raw = mne.io.read_raw_eeglab(
            file,
            preload=True,
            uint16_codec='latin1',
        )
        # FIXME: change this into Epochs.decimate() later
        raw.resample(self.cfg.preprocess.ideal_sfreq, n_jobs='cuda')
        return raw
