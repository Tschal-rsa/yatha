from typing import cast

import mne
import numpy as np
from numpy.typing import NDArray

import const
from config import Config

from .feature import Feature


class FrequencyFeature(Feature):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def calculate(self, epochs: mne.Epochs) -> NDArray:
        # [E, C, T]
        spectrum = epochs.compute_psd(method="welch", fmin=0.5, fmax=45)
        data, freqs = cast(
            tuple[NDArray, NDArray], spectrum.get_data(return_freqs=True)
        )
        # [E, C, len(0.5:45:0.5)=90]
        band_list = []
        for lfreq, rfreq in const.Feature.frequency_bands:
            band_list.append(data[..., (lfreq <= freqs) & (freqs < rfreq)].sum(-1))
        # [B, E, C]
        bands = np.stack(band_list, 1)
        # [E, B, C]
        relative_bands = bands / bands.sum(1, keepdims=True)
        return relative_bands
