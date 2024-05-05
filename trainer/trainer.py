from tqdm.autonotebook import tqdm

from config import Config

from .models import get_estimator
from .recorder import Recorder
from .spliters import get_spliter


class Trainer:
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    def train(self) -> float:
        spliter = get_spliter(self.cfg)
        n_splits = self.cfg.train.n_splits
        recorder = Recorder(self.cfg)
        for fold, (train_samples, val_samples, test_samples) in tqdm(
            enumerate(spliter.get_split()), 'Fold', n_splits
        ):
            estimator = get_estimator(self.cfg, spliter)
            estimator.fit(fold, train_samples, val_samples)
            y_score, complexity = estimator.predict(test_samples)
            y_true = test_samples.label
            recorder.record(y_true, y_score, complexity)
        result = recorder.finish()
        return -result.auc
