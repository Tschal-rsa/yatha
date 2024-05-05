import os
import random

import mne
import numpy as np
import torch


def set_cuda(device_ids: str) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    mne.utils.set_config('MNE_USE_CUDA', 'true')


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
