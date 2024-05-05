from numpy.typing import NDArray


class Normalizer:
    def __init__(self) -> None:
        self.mean: NDArray | None = None
        self.std: NDArray | None = None

    def fit(self, data: NDArray) -> None:
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data: NDArray) -> NDArray:
        assert self.mean is not None and self.std is not None
        return (data - self.mean) / (self.std + 1e-5)
    
    def inverse_transform(self, data: NDArray) -> NDArray:
        assert self.mean is not None and self.std is not None
        return data * self.std + self.mean

    def fit_transform(self, data: NDArray) -> NDArray:
        self.fit(data)
        return self.transform(data)
