import os
from collections import defaultdict

import numpy as np
from tqdm.autonotebook import trange

from config import Config
from repos.brainfeatures.data_set.edf_abnormal import EDFAbnormal
from repos.brainfeatures.feature_generation.generate_features import (
    default_feature_generation_params,
)
from repos.proto import process_one_file

from .extractor import Extractor


class ProtoExtractor(Extractor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def extract(self) -> None:
        metric = 'cwt'
        splits = ['train', 'val']
        os.makedirs(self.cfg.dataset.feature_path, exist_ok=True)
        feature_file = os.path.join(self.cfg.dataset.feature_path, f'{metric}.npz')
        samples_buffer = defaultdict(list)
        for split in splits:
            data_root = os.path.join(self.cfg.dataset.preprocess_path, split)
            output_root = os.path.join(self.cfg.dataset.feature_path, split)
            for category in os.listdir(data_root):
                data_dir = os.path.join(data_root, category) + '/'
                # HACK: not used
                output_dir = os.path.join(output_root, category) + '/'

                edf_abnormal = EDFAbnormal(data_dir, '.edf', subset=split)
                edf_abnormal.load()

                default_feature_generation_params.update({
                    'agg_mode': 'median', # 'median' / 'mean'...
                })

                for file_id in trange(len(edf_abnormal), desc=f'{split}-{category}'):
                    bundle = process_one_file(
                        data_set=edf_abnormal,
                        file_id=file_id,
                        out_dir=output_dir,
                        domains=[metric],
                        **default_feature_generation_params,
                    )
                    if bundle is not None:
                        pid, label, data = bundle
                        samples_buffer[f'{pid}_{label}'].append(data)

        samples_dict = {k: np.stack(v) for k, v in samples_buffer.items()}
        np.savez_compressed(feature_file, **samples_dict)

        # ["cwt", "dwt", "dft", "phase", "time"]
