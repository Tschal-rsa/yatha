from dataclasses import dataclass
from inspect import getmembers, isfunction
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING, SI, OmegaConf

from . import resolvers


@dataclass
class TaskConfig:
    mapping: dict[str, int | None] = MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    preprocess_name: str = MISSING
    root: str = MISSING
    preprocess_root: str = MISSING
    preprocess_path: str = II(
        'get_preprocess_path:${.preprocess_root},${.preprocess_name}'
    )
    feature_root: str = MISSING
    feature_path: str = II('get_feature_path:${.preprocess_root},${.preprocess_name}')
    original_n_channels: int = MISSING
    n_channels: int = MISSING
    sfreq: float = MISSING
    task: TaskConfig = MISSING
    num_class: int = II('get_num_class:${.task.mapping}')


@dataclass
class PreprocessConfig:
    crop_prefix: float = 20
    ideal_sfreq: float = 250
    crop_duration: float = 30
    epoch_duration: float = 2
    epoch_overlap: float = II('get_epoch_overlap:${.epoch_duration}')
    features: list[str] = MISSING
    mne_features: dict[str, Any] = MISSING
    ideal_morlet_time: int = 30
    # HACK: below are deprecated
    crop_overlap: float = II('get_epoch_overlap:${.crop_duration}')
    sample_epochs: int = 1


@dataclass
class ModelConfig:
    name: str = MISSING


@dataclass
class XGBoostConfig(ModelConfig):
    name: str = 'XGBoost'
    param: dict[str, Any] = MISSING
    num_round: int = MISSING


@dataclass
class NeoConfig(ModelConfig):
    name: str = 'Neo-RRL'
    epoch: int = 41
    batch_size: int = 64
    max_lr: float = 0.01
    init_lr: float = 0.0001
    andness_init_lr: float = 0.001
    warmup_epoch: int = 10
    weight_decay: float = 0
    dropout_p: float = 0.2
    ith_kfold: int = 0
    log_iter: int = 500
    temp: float = 1.0
    structure: str = '5@64'
    folder_path: str = '.'
    model: str = SI('model_${.ith_kfold}.pth')
    rrl_file: str = SI('rrl_${.ith_kfold}.txt')
    logic_name: str = 'AIWA'

    # NLAF
    alpha: float = 0.999
    beta: int = 8
    gamma: int = 1

    # AIWA
    bound: float = 0.1
    eps: float = 1e-8


@dataclass
class RRLConfig(NeoConfig):
    name: str = 'RRL'
    data_dir: str = II(
        'get_rrl_path:${dataset.preprocess_root},${dataset.preprocess_name}'
    )
    data_set: str = II('dataset.name')
    reconstruct: bool = True
    nr: int = 0
    learning_rate: float = II('.max_lr')
    lr_decay_rate: float = 0.75
    lr_decay_epoch: int = 10
    master_address: str = '127.0.0.1'
    master_port: str = II('train.master_port')
    nlaf: bool = False
    use_not: bool = False
    save_best: bool = False
    skip: bool = False
    estimated_grad: bool = False
    weighted: bool = False
    print_rule: bool = True
    device_ids: list[int] = II('get_device_ids:${train.device_ids}')
    gpus: int = II('get_gpus:${.device_ids}')
    nodes: int = 1
    world_size: int = II('get_world_size:${.gpus},${.nodes}')
    n_splits: int = II('train.n_splits')
    seed: int = II('train.seed')
    # HACK: below are not used
    log: str = 'log.txt'
    test_res: str = 'test_res.txt'


@dataclass
class TrainConfig:
    seed: int = 42
    device_ids: str = MISSING
    master_port: str = MISSING
    n_splits: int = 5
    val_size: float = 0
    plot_roc: bool = False


@dataclass
class Config:
    run: str = MISSING
    dataset: DatasetConfig = MISSING
    preprocess: PreprocessConfig = MISSING
    model: ModelConfig = MISSING
    train: TrainConfig = MISSING


def register_resolvers() -> None:
    for name, func in getmembers(resolvers, isfunction):
        OmegaConf.register_new_resolver(name, func)


def store_config() -> None:
    register_resolvers()
    cs = ConfigStore.instance()
    cs.store(name='base', node=Config)
    cs.store(group='dataset', name='base', node=DatasetConfig)
    cs.store(group='dataset/task', name='base', node=TaskConfig)
    cs.store(group='preprocess', name='base', node=PreprocessConfig)
    cs.store(group='model', name='base_xgboost', node=XGBoostConfig)
    cs.store(group='model', name='base_neo', node=NeoConfig)
    cs.store(group='model', name='base_rrl', node=RRLConfig)
    cs.store(group='train', name='base', node=TrainConfig)
