import os

import mne
import pandas as pd

from config import Config

from .mycenaean import MycenaeanPreprocessor


class MycenaeanAHEPA(MycenaeanPreprocessor):
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

    def preprocess_single_file(self, file: str) -> mne.io.Raw | None:
        raw = mne.io.read_raw_eeglab(
            file,
            preload=True,
            uint16_codec='latin1',
        )
        return raw
