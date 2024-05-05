from typing import cast

from numpy.typing import NDArray

from config import Config, NeoConfig
from dataset import Normalizer, Samples
from trainer.spliters import Spliter

from .estimator import Estimator
from .neo import Neo


class NeoEstimator(Estimator):
    def __init__(self, cfg: Config, spliter: Spliter) -> None:
        super().__init__(cfg)
        self.spliter = spliter
        self.neo_cfg = cast(NeoConfig, cfg.model)
        self.normalizer = Normalizer()
        self.num_class = cfg.dataset.num_class

    def get_dims(self, data: NDArray) -> list[int]:
        return (
            [data.shape[1]]
            + list(map(int, self.neo_cfg.structure.split('@')))
            + [self.num_class]
        )

    def fit(
        self, fold: int, train_samples: Samples, val_samples: Samples | None
    ) -> None:
        self.neo_cfg.ith_kfold = fold
        train_data = self.normalizer.fit_transform(train_samples.data)
        train_samples_processed = Samples(train_data, train_samples.label)
        val_samples_processed = None
        if val_samples is not None:
            val_data = self.normalizer.transform(val_samples.data)
            val_samples_processed = Samples(val_data, val_samples.label)
        neo = Neo(self.neo_cfg, self.get_dims(train_data), self.normalizer)
        neo.train(train_samples_processed, val_samples_processed)

    def predict(self, samples: Samples) -> tuple[NDArray, float]:
        data = self.normalizer.transform(samples.data)
        neo = Neo(self.neo_cfg, self.get_dims(data), self.normalizer)
        neo.load_state_dict()
        y_score = neo.test(data)
        neo.print_rule(self.spliter.feature_names, self.spliter.label_names)
        complexity = neo.get_complexity()
        neo.debug()
        neo.del_state_dict()
        return y_score, complexity
