from typing import cast

import mne
from mne_features.feature_extraction import extract_features
from numpy.typing import NDArray
from omegaconf import OmegaConf

from config import Config

from .feature import Feature


class OmnisFeature(Feature):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.selected = cast(
            list[str], OmegaConf.to_object(cfg.preprocess.mne_features['selected'])
        )
        self.params = cast(
            dict[str, str], OmegaConf.to_object(cfg.preprocess.mne_features['params'])
        )

    def calculate(self, epochs: mne.Epochs) -> NDArray:
        return extract_features(
            epochs.get_data(copy=False),
            epochs.info['sfreq'],
            self.selected,
            self.params,
        )
