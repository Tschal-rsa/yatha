import json
import os
import random
from typing import Iterator, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.autonotebook import tqdm

from config import Config
from dataset import (
    Data,
    SampleBuffer,
    Samples,
    custom_collate,
    index_data,
    index_samples,
)


class Spliter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.feature_names: list[str] = []
        self.label_names: list[str] = []
        self.samples = self.get_dataset()
        self.sgkf = StratifiedGroupKFold(n_splits=cfg.train.n_splits)

    def split_samples(self, data: NDArray) -> list[NDArray]:
        if self.cfg.preprocess.sample_epochs <= 1:
            return list(data)
        else:
            # FIXME: unreasonable to save some temporal signals for RRL to reason
            return np.split(
                data,
                np.arange(
                    self.cfg.preprocess.sample_epochs,
                    data.shape[0] + 1,
                    self.cfg.preprocess.sample_epochs,
                ),
            )[:-1]

    def get_dataset_labels(self) -> None:
        mapping = cast(dict[str, int | None], self.cfg.dataset.task.mapping)
        label_names: list[list[str]] = [[] for _ in range(self.cfg.dataset.num_class)]
        for name, label in mapping.items():
            if label is not None:
                label_names[label].append(name)
        self.label_names = ['+'.join(names) for names in label_names]

    def get_dataset(self) -> Data[list[str], NDArray, NDArray]:
        self.get_dataset_labels()
        sample_dict: dict[str, SampleBuffer] = {}
        for feature in self.cfg.preprocess.features:
            with open(
                os.path.join(self.cfg.dataset.feature_path, f'{feature}.json')
            ) as f:
                feature_names = cast(list[str], json.load(f))
                self.feature_names += feature_names
            with np.load(
                os.path.join(self.cfg.dataset.feature_path, f'{feature}.npz')
            ) as data:
                for pid_label in tqdm(data.files, desc=feature):
                    pid, label = cast(tuple[str, str], pid_label.split('_'))
                    if (new_label := self.cfg.dataset.task.mapping[label]) is not None:
                        for sid, sample_data in enumerate(
                            self.split_samples(data[pid_label])
                        ):
                            buffer = sample_dict.setdefault(
                                f'{pid}_{sid}', SampleBuffer(pid, sid, new_label)
                            )
                            buffer.append(sample_data)
        sample_list = [buffer.to_sample() for buffer in sample_dict.values()]
        random.shuffle(sample_list)
        return custom_collate(sample_list)

    def get_split(self) -> Iterator[tuple[Samples, Samples | None, Samples]]:
        for train_index, test_index in self.sgkf.split(
            self.samples.data, self.samples.label, self.samples.pid
        ):
            train_data = index_data(self.samples, train_index)
            val_samples = None
            if self.cfg.train.val_size > 0:
                val_sgkf = StratifiedGroupKFold(
                    n_splits=int(1 / self.cfg.train.val_size)
                )
                train_index_new, val_index = next(
                    val_sgkf.split(train_data.data, train_data.label, train_data.pid)
                )
                train_samples = index_samples(train_data, train_index_new)
                val_samples = index_samples(train_data, val_index)
            else:
                train_samples = Samples(train_data.data, train_data.label)
            test_samples = index_samples(self.samples, test_index)
            yield train_samples, val_samples, test_samples
