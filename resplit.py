import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import cast

import fire  # type: ignore
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm


class Spliter(ABC):
    def __init__(self, split: str) -> None:
        self.root = f'/data/lfz/datasets/EEG/PUMC_{split}'
        self.original_root = '/data/lfz/datasets/EEG/PUMC_sev/original'
        self.feature_root = f'{self.root}/Archaic-10/feature'
        self.sev_feature_root = '/data/lfz/datasets/EEG/PUMC_sev/Archaic-10/feature'
        self.main_feature = 'cwt.npz'

    def get_original_label_map(self) -> dict[str, str]:
        label_map: dict[str, str] = {}
        for root, dirs, files in os.walk(self.original_root):
            for name in files:
                pid, ext = os.path.splitext(name)
                if ext != '.edf':
                    continue
                label = os.path.basename(root)
                label_map[pid.replace('-', '')] = label
        return label_map

    def get_old_label_map(self, file: str) -> dict[str, str]:
        label_map: dict[str, str] = {}
        with np.load(file) as data:
            for pid_label in data.files:
                pid = cast(str, pid_label).split('_')[0]
                label_map[pid.replace('-', '')] = pid_label
        return label_map

    @abstractmethod
    def get_label_map(self) -> dict[str, str]:
        raise NotImplementedError

    def resplit(self) -> None:
        label_map = self.get_label_map()
        os.makedirs(self.feature_root, exist_ok=True)
        for filename in os.listdir(self.sev_feature_root):
            prefix, ext = os.path.splitext(filename)
            sev_file = os.path.join(self.sev_feature_root, filename)
            match ext:
                case '.npz':
                    old_label_map = self.get_old_label_map(sev_file)
                    data = {}
                    with np.load(sev_file) as old_data:
                        for pid, label in tqdm(label_map.items(), desc=prefix):
                            if pid in old_label_map.keys():
                                data[f'{pid}_{label}'] = old_data[old_label_map[pid]]
                    file = os.path.join(self.feature_root, filename)
                    np.savez_compressed(file, **data)
                case '.json' if self.feature_root != self.sev_feature_root:
                    shutil.copy2(sev_file, self.feature_root)

    def check(self) -> None:
        label_map = self.get_label_map()
        file = os.path.join(self.feature_root, self.main_feature)
        original_label_map = self.get_original_label_map()
        old_label_map = self.get_old_label_map(file)
        for pid, label in label_map.items():
            if pid not in old_label_map.keys():
                if pid not in original_label_map.keys():
                    print(f'{pid} {label} origin')
                else:
                    print(f'{pid} {label} preprocess')
        for pid, label in old_label_map.items():
            if pid not in label_map.keys():
                print(f'{label} redundant')

    def stat(self) -> None:
        file = os.path.join(self.feature_root, self.main_feature)
        with np.load(file) as data:
            stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for pid_label in data.files:
                _, label = cast(str, pid_label).split('_')
                stat[label][0] += 1
                stat[label][1] += data[pid_label].shape[0]
            for label, (num_patients, num_samples) in stat.items():
                print(f'{label}: ({num_patients}, {num_samples})')


class AgeSpliter(Spliter):
    def __init__(self) -> None:
        super().__init__('bio_age')

    def get_label_map(self) -> dict[str, str]:
        label_map: dict[str, str] = {}
        excel_path_template = '/data/lfz/datasets/EEG/excel/BIO{}.xlsx'
        excel_dict = pd.read_excel(
            excel_path_template.format(0), sheet_name=['确定AD', 'FTD'], index_col=0
        )
        for pid in excel_dict['确定AD'].index.values:
            age = excel_dict['确定AD'].loc[pid, '年龄']
            label_map[pid] = 'Ae' if age > 65 else 'Ay'
        # FTD 没有分年龄
        for pid in excel_dict['FTD'].index.values:
            label_map[pid] = 'F'
        excel = pd.read_excel(
            excel_path_template.format(1), sheet_name='正常', index_col=0
        )
        for pid in excel.index.values:
            age = excel.loc[pid, '年龄']
            label_map[pid] = 'Ne' if age > 65 else 'Ny'
        return label_map


class SevSpliter(Spliter):
    def __init__(self) -> None:
        super().__init__('sev')

    def get_label_map(self) -> dict[str, str]:
        label_map: dict[str, str] = {}
        excel_path_template = '/data/lfz/datasets/EEG/excel/BIO{}.xlsx'
        excel = pd.read_excel(
            excel_path_template.format(2), sheet_name='正常', index_col=0
        )
        for pid in excel.index.values:
            label_map[pid] = 'N'
        excel = pd.read_excel(
            excel_path_template.format(3), sheet_name='轻度', index_col=0
        )
        for pid in excel.index.values:
            label_map[pid] = 'Mi'
        excel = pd.read_excel(
            excel_path_template.format(4), sheet_name='中度', index_col=0
        )
        for pid in excel.index.values:
            label_map[pid] = 'Mo'
        excel = pd.read_excel(
            excel_path_template.format(5), sheet_name='重度', index_col=0
        )
        for pid in excel.index.values:
            label_map[pid] = 'S'
        return label_map


class ZSpliter(Spliter):
    def __init__(self) -> None:
        super().__init__('Z')
        self.feature_root = f'{self.root}/Archaic-Z/feature'
        self.main_feature = 'all.npz'
    
    def get_label_map(self) -> dict[str, str]:
        return {}


def get_spliter(split: str) -> Spliter:
    match split:
        case 'age':
            return AgeSpliter()
        case 'sev':
            return SevSpliter()
        case 'z':
            return ZSpliter()
        case _split:
            print(f'Unknown split: {_split}!')
            exit(0)


class Interface:
    def __init__(self, split: str) -> None:
        self.spliter = get_spliter(split)

    def resplit(self) -> None:
        self.spliter.resplit()

    def check(self) -> None:
        self.spliter.check()

    def stat(self) -> None:
        self.spliter.stat()


if __name__ == '__main__':
    # python resplit.py --split age/sev resplit/check/stat
    fire.Fire(Interface)
