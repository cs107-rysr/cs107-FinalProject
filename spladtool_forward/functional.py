import numpy as np
from .tensor import Tensor
from .layer import *
from typing import Union, List


def power(x, p):
    return Power()(x, p)


def sumup(x: Tensor, y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
    if type(y) == Tensor:
        return TensorSum()(x, y)
    else:
        return NumSum()(x, y)


def prod(x: Tensor, y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
    if type(y) == Tensor:
        return TensorProd()(x, y)
    else:
        return NumProd()(x, y)


def inv(x: Tensor) -> Tensor:
    return TensorInv()(x)


def div(x: Union[Tensor, int, float, Tensor, np.ndarray, list], y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
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


def minus(x: Union[Tensor, int, float, Tensor, np.ndarray, list], y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
    if type(y) == Tensor:
        if type(x) == Tensor:
            return sumup(x, -y)
        else:
            return sumup(-y, x)
    else:
        assert type(x) == Tensor
        return sumup(x, -y)


def exp(x: Tensor):
    return Exp()(x)

def log(x: Tensor):
    return Log()(x)

def sin(x: Tensor):
    return Sin()(x)

def cos(x: Tensor):
    return Cos()(x)

def tan(x: Tensor):
    return Tan()(x)

def arcsin(x: Tensor):
    return ArcSin()(x)

def arccos(x: Tensor):
    return ArcCos()(x)

def arctan(x: Tensor):
    return ArcTan()(x)

def sinh(x: Tensor):
    return Sinh()(x)

def cosh(x: Tensor):
    return Cosh()(x)

def tanh(x: Tensor):
    return Tanh()(x)

def logistic(x: Tensor):
    return Logistic()(x)

def sqrt(x: Tensor):
    return SquareRoot()(x)

def equal(x: Tensor, y):
    return Equal()(x, y)

def less(x: Tensor, y):
    return Less()(x,y)

def not_equal(x:Tensor, y):
    return NotEqual()(x,y)
    
def greater(x: Tensor, y):
    return Greater()(x,y)

def less_equal(x: Tensor, y):
    return LessEqual()(x,y)

def greater_equal(x: Tensor, y):
    return GreaterEqual()(x,y)
     
def tensor(x):
    return Tensor(x)