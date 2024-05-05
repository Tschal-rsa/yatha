import json
import os
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from tqdm.autonotebook import trange

from config import Config
from repos.brainfeatures.data_set.edf_abnormal import EDFAbnormal
from repos.brainfeatures.feature_generation.generate_features import (
    default_feature_generation_params,
    generate_features_of_one_file,
)
from utils import logger

from .extractor import Extractor


class ArchaicExtractor(Extractor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def process_one_file(
        self, data_set: EDFAbnormal, file_id: int, feature: str, **kw
    ) -> tuple[str, str, NDArray] | None:
        file_root, sid_raw = os.path.split(data_set.file_names[file_id])
        pid_label = os.path.basename(file_root)
        signals, sfreq, _pathological = data_set[file_id]
        feature_df = generate_features_of_one_file(
            signals, sfreq, domains=[feature], **kw
        )
        if feature_df is None:
            logger.error(f'Feature generation failed for {pid_label}_{sid_raw}.')
            return None
        feature_json = os.path.join(self.cfg.dataset.feature_path, f'{feature}.json')
        if not os.path.exists(feature_json):
            with open(feature_json, 'w', encoding='utf-8') as f:
                json.dump(feature_df.columns.values.tolist(), f, indent=2)
        return pid_label, sid_raw, feature_df.values

    def extract(self) -> None:
        os.makedirs(self.cfg.dataset.feature_path, exist_ok=True)
        for feature in self.cfg.preprocess.features:
            feature_file = os.path.join(self.cfg.dataset.feature_path, f'{feature}.npz')
            if os.path.exists(feature_file):
                logger.warning(f'Skipping {feature} feature extraction.')
                continue
            samples_buffer = defaultdict(list)
            data_dir = self.cfg.dataset.preprocess_path + '/'
            # HACK: deceive brainfeatures with a pseudo subset
            edf_abnormal = EDFAbnormal(
                data_dir,
                '.fif',
                subset=self.cfg.dataset.preprocess_name,
                ch_name_pattern=None,
            )
            edf_abnormal.load()
            default_feature_generation_params.update({'agg_mode': 'median'})

            for file_id in trange(len(edf_abnormal), desc=f'{feature}'):
                bundle = self.process_one_file(
                    data_set=edf_abnormal,
                    file_id=file_id,
                    feature=feature,
                    **default_feature_generation_params,
                )
                if bundle is not None:
                    pid_label, _sid_raw, data = bundle
                    samples_buffer[pid_label].append(data)

            samples_dict = {k: np.concatenate(v, axis=0) for k, v in samples_buffer.items()}
            np.savez_compressed(feature_file, **samples_dict)
