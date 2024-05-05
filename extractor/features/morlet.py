import mne
import numpy as np
from mne.time_frequency import tfr_morlet
from numpy.typing import NDArray

import const
from config import Config

from .feature import Feature


class MorletFeature(Feature):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def calculate(self, epochs: mne.Epochs) -> NDArray:
        n_cycles = const.Feature.morlet_freqs / 2.0
        decim = int(
            self.cfg.preprocess.crop_duration
            * self.cfg.preprocess.ideal_sfreq
            / self.cfg.preprocess.ideal_morlet_time
        )
        power = tfr_morlet(
            epochs,
            freqs=const.Feature.morlet_freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            decim=decim,
            average=False,
        )
        return power.data
