import os
from typing import cast

from omegaconf import DictConfig, OmegaConf


def get_preprocess_path(preprocess_root: str, preprocess_name: str) -> str:
    return os.path.join(preprocess_root, preprocess_name, 'preprocess')


def get_feature_path(preprocess_root: str, preprocess_name: str) -> str:
    return os.path.join(preprocess_root, preprocess_name, 'feature')


def get_rrl_path(preprocess_root: str, preprocess_name: str) -> str:
    return os.path.join(preprocess_root, preprocess_name, 'rrl')


def get_epoch_overlap(epoch_duration: float) -> float:
    return epoch_duration / 2


def get_device_ids(device_ids: str) -> list[int]:
    # HACK: set device ids to [0]
    # return list(map(int, device_ids.split(',')))
    return [0]


def get_gpus(device_ids: list[int]) -> int:
    # HACK: set number of GPUs to 1
    # return len(device_ids)
    return 1


def get_world_size(gpus: int, nodes: int) -> int:
    return gpus * nodes


def get_num_class(mapping_cfg: DictConfig) -> int:
    mapping = cast(dict[str, int | None], OmegaConf.to_object(mapping_cfg))
    return len(set(i for i in mapping.values() if i is not None))


def get_reduced_n_channels(original_n_channels: int) -> int:
    return original_n_channels - 2
