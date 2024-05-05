from typing import Literal

import mne
import mne_connectivity
import numpy as np
from mne_connectivity import spectral_connectivity_time
from numpy.typing import NDArray

from config import Config
from utils import logger

from .feature import Feature

MODE = Literal['average', 'whole']


class ConnectivityFeature(Feature):
    def __init__(self, cfg: Config, mode: MODE = 'average') -> None:
        super().__init__(cfg)
        self.mode = mode

    def calculate(self, epochs: mne.Epochs) -> NDArray:
        # [E, C, T]
        con: mne_connectivity.EpochSpectralConnectivity = spectral_connectivity_time(
            epochs,
            np.array([2, 6, 10, 18, 35]),
            method='coh',
            mode='cwt_morlet',
            n_cycles=np.array([2, 4, 7, 7, 7]),
        )
        # [E, C, C, B]
        con_data: NDArray = con.get_data('dense').transpose(0, 3, 1, 2)
        # [E, B, C, C]
        match self.mode:
            case 'average':
                con_average = (con_data.sum(-2) + con_data.sum(-1)) / (
                    con_data.shape[-1] - 1
                )
                # [E, B, C] 
                return con_average
            case 'whole':
                indices = np.tril_indices_from(con_data[0, 0, ...], -1)
                con_indexed = con_data[..., indices[0], indices[1]]
                # [E, B, (C - 1)C / 2]
                return con_indexed
            case _mode:
                logger.critical(f'Unknown connectivity mode: {_mode}!')
                exit(1)
