import numpy as np
from .tensor import Tensor
from .layer import Square


def square(x):
    square_operator = Square()
    y = square_operator(x)
    return y


def tensor(x):
    return Tensor(x)