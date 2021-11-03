import numpy as np
from .tensor import Tensor
from .layer import *
from typing import Union, List


def power(x: Tensor, p: Union[int, float]) -> Tensor:
    return Power()(x, p)


def sumup(x: Tensor, y: Union[int, float, Tensor, np.ndarray, List[int], List[float]]) -> Tensor:
    if type(y) == Tensor:
        return TensorSum()(x, y)
    else:
        return NumSum()(x, y)


def prod(x: Tensor, y: Union[int, float, Tensor, np.ndarray, List[int], List[float]]) -> Tensor:
    if type(y) == Tensor:
        return TensorProd()(x, y)
    else:
        return NumProd()(x, y)


def inv(x: Tensor) -> Tensor:
    return TensorInv()(x)


def div(x: Union[Tensor, int, float, Tensor, np.ndarray, List[int], List[float]], y: Union[int, float, Tensor, np.ndarray, List[int], List[float]]) -> Tensor:
    if type(y) == Tensor:
        if type(x) == Tensor:
            return prod(x, inv(y))
        else:
            return x * inv(y)
    else:
        assert type(x) == Tensor
        return prod(x, 1. / y)


def neg(x: Tensor) -> Tensor:
    return prod(x, -1)


def minus(x: Union[Tensor, int, float, Tensor, np.ndarray, List[int], List[float]], y: Union[int, float, Tensor, np.ndarray, List[int], List[float]]) -> Tensor:
    if type(y) == Tensor:
        if type(x) == Tensor:
            return sumup(x, -y)
        else:
            return sumup(-y, x)
    else:
        assert type(x) == Tensor
        return sumup(x, -y)


def tensor(x):
    return Tensor(x)