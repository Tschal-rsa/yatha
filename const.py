import numpy as np
from numpy.typing import NDArray


class Preprocess:
    ch_names_0 = [
        'EEG Fp1-Ref',
        'EEG Fp2-Ref',
        'EEG F3-Ref',
        'EEG F4-Ref',
        'EEG C3-Ref',
        'EEG C4-Ref',
        'EEG P3-Ref',
        'EEG P4-Ref',
        'EEG O1-Ref',
        'EEG O2-Ref',
        'EEG F7-Ref',
        'EEG F8-Ref',
        'EEG T3-Ref',
        'EEG T4-Ref',
        'EEG T5-Ref',
        'EEG T6-Ref',
        'EEG Fz-Ref',
        'EEG Cz-Ref',
        'EEG Pz-Ref',
        'EEG A1-Ref',
        'EEG A2-Ref',
    ]
    ch_names_1 = [
        'EEG Fp1-REF',
        'EEG Fp2-REF',
        'EEG F3-REF',
        'EEG F4-REF',
        'EEG C3-REF',
        'EEG C4-REF',
        'EEG P3-REF',
        'EEG P4-REF',
        'EEG O1-REF',
        'EEG O2-REF',
        'EEG F7-REF',
        'EEG F8-REF',
        'EEG T3-REF',
        'EEG T4-REF',
        'EEG T5-REF',
        'EEG T6-REF',
        'EEG Fz-REF',
        'EEG Cz-REF',
        'EEG Pz-REF',
        'EEG A1-REF',
        'EEG A2-REF',
    ]


class Feature:
    frequency_bands: list[tuple[float, float]] = [
        (0.5, 4),
        (4, 8),
        (8, 13),
        (13, 25),
        (25, 45),
    ]
    morlet_freqs: NDArray = np.logspace(*np.log10([6, 35]), num=8)  # type: ignore
