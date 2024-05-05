import dataclasses
from dataclasses import asdict, dataclass, fields
from typing import Any, Generic, TypeVar, overload

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import default_collate

T = TypeVar('T', str, list[str])
U = TypeVar('U', int, NDArray)
V = TypeVar('V', list[NDArray], NDArray)


@dataclass
class Data(Generic[T, U, V]):
    pid: T
    sid: U
    data: V
    label: U


@dataclass
class Samples:
    data: NDArray
    label: NDArray


@overload
def custom_collate(
    batch: list[Data[str, int, NDArray]]
) -> Data[list[str], NDArray, NDArray]: ...


@overload
def custom_collate(batch: list[Any]) -> Any: ...


def custom_collate(batch: list) -> Any:
    elem = batch[0]
    if isinstance(elem, Data):
        collated: dict[str, Any] = default_collate([asdict(s) for s in batch])
        transformed = {
            k: v if isinstance(v, list) else np.asarray(v) for k, v in collated.items()
        }
        return Data(**transformed)
    else:
        return default_collate(batch)


def index_samples(samples: Data | Samples, index: Any) -> Samples:
    return Samples(
        **{field.name: getattr(samples, field.name)[index] for field in fields(Samples)}
    )


def index_data(data: Data[T, U, V], index: Any) -> Data[T, U, V]:
    indexed_data: dict[str, Any] = {}
    for field in fields(Data):
        obj = getattr(data, field.name)
        if isinstance(obj, list):
            indexed_data[field.name] = [obj[idx] for idx in index]
        else:
            indexed_data[field.name] = obj[index]
    return dataclasses.replace(data, **indexed_data)


class SampleBuffer:
    def __init__(self, pid: str, sid: int, label: int) -> None:
        self.buffer = Data[str, int, list[NDArray]](pid, sid, [], label)

    def append(self, data: NDArray) -> None:
        self.buffer.data.append(data)

    def to_sample(self) -> Data[str, int, NDArray]:
        return Data(
            self.buffer.pid,
            self.buffer.sid,
            np.concatenate(self.buffer.data, axis=None),
            self.buffer.label,
        )
