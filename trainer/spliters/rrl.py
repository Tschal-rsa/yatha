import json
import os
from typing import cast

from tqdm.autonotebook import trange

import const
from config import Config, RRLConfig
from utils import get_modified_time, logger

from .spliter import Spliter


class RRLSpliter(Spliter):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        rrl_cfg = cast(RRLConfig, cfg.model)
        if rrl_cfg.reconstruct:
            os.makedirs(rrl_cfg.data_dir, exist_ok=True)
            data_file = os.path.join(rrl_cfg.data_dir, f'{rrl_cfg.data_set}.data')
            info_file = os.path.join(rrl_cfg.data_dir, f'{rrl_cfg.data_set}.info')
            # HACK: cancel data existence detection
            self.reconstruct(data_file, info_file)
            # if os.path.exists(data_file) and os.path.exists(info_file):
            #     data_mtime = get_modified_time(data_file)
            #     info_mtime = get_modified_time(info_file)
            #     skip_reconstruct = True
            #     for feature in cfg.preprocess.features:
            #         feature_file = os.path.join(cfg.dataset.feature_path, f'{feature}.npz')
            #         if os.path.exists(feature_file):
            #             feature_mtime = get_modified_time(feature_file)
            #             if feature_mtime > min(data_mtime, info_mtime):
            #                 skip_reconstruct = False
            #                 self.reconstruct(data_file, info_file)
            #                 break
            #         else:
            #             logger.critical(f'Feature {feature} has not been extracted!')
            #             exit(1)
            #     if skip_reconstruct:
            #         logger.warning('Skipping dataset reconstruction for RRL.')
            # else:
            #     self.reconstruct(data_file, info_file)

    def reconstruct(self, data_file: str, info_file: str) -> None:
        logger.warning('Reconstructing dataset for RRL.')
        with open(data_file, 'w', encoding='utf-8') as f:
            for index in trange(self.samples.data.shape[0], desc='Sample'):
                f.write(','.join(map(str, self.samples.data[index].tolist())))
                f.write(
                    f',{self.samples.pid[index]},{self.samples.label[index].item()}\n'
                )
        with open(info_file, 'w', encoding='utf-8') as f:
            for feature in self.cfg.preprocess.features:
                # HACK: split them into small functions or classes
                match feature:
                    # FIXME: figure out what do these features mean
                    case x if x in ['cwt', 'dwt', 'dft', 'phase', 'time']:
                        feature_json = os.path.join(
                            self.cfg.dataset.feature_path, f'{feature}.json'
                        )
                        with open(feature_json, 'r', encoding='utf-8') as fp:
                            feature_list: list[str] = json.load(fp)
                            for feature_name in feature_list:
                                f.write(f'{feature_name} continuous\n')
                    case 'morlet':
                        for channel in range(self.cfg.dataset.n_channels):
                            for band in range(len(const.Feature.morlet_freqs)):
                                for time in range(
                                    self.cfg.preprocess.ideal_morlet_time
                                ):
                                    f.write(
                                        f'morl_{channel}_{band}_{time} continuous\n'
                                    )
                    case _name:
                        for epoch in range(self.cfg.preprocess.sample_epochs):
                            for band in range(len(const.Feature.frequency_bands)):
                                for channel in range(self.cfg.dataset.n_channels):
                                    f.write(
                                        # FIXME: ignore epoch since sample_epochs = 1
                                        # f'{feature[:4]}_{epoch}_{band}_{channel} continuous\n'
                                        f'{_name[:4]}_{band}_{channel} continuous\n'
                                    )
            f.write('group discrete\nclass discrete\nGROUP_POS -2\nLABEL_POS -1\n')
