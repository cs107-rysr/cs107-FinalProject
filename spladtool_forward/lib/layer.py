import numpy as np  
from .tensor import Tensor
from typing import Union, List


class Layer():
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer'
    
    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    def __str__(self):
        return self.desc

    def __repr__(self):
        return self.desc


class Power(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Power'

    def forward(self, x: Tensor, p: Union[float, Tensor]) -> Tensor:
        if type(p) == float:
            y_data = np.power(x.data.copy(), p)
            y_grad = p * np.power(x.data.copy(), p - 1) * x.grad
        else: 
            y_data = np.power(x.data.copy(), p.data.copy())
            y_grad = p.data.copy() * np.power(x.data.copy(), p.data.copy() - 1) * x.grad
        y = Tensor(y_data, grad=y_grad)
        return y


class TensorSum(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.TensorSum'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        s_data = x.data + y.data
        s_grad = x.grad + y.grad
        s = Tensor(s_data, s_grad)
        return s


class TensorProd(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.TensorProd'
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        p_data = x.data * y.data
        p_grad = x.grad * y.data + x.data * y.grad
        p = Tensor(p_data, p_grad)
        return p


class TensorInv(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.TensorInv'

    def forward(self, x: Tensor) -> Tensor:
        i_data = 1. / x.data
        i_grad = -1. / (x.data ** 2) * x.grad 
        i = Tensor(i_data, i_grad)
        return i


class NumProd(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.NumProd'

    def forward(self, x: Tensor, y: Union[int, float, List[float], List[int], np.ndarray]) -> Tensor:
        if type(y) == List:
            y = np.array(y)
        s_data = x.data * y
        s_grad = x.grad * y
        s = Tensor(s_data, s_grad)
        return s


class NumSum(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.NumSum'

    def forward(self, x: Tensor, y: Union[int, float, List[float], List[int], np.ndarray]) -> Tensor:
        if type(y) == List:
            y = np.array(y)
        s_data = x.data + y
        s_grad = x.grad
        s = Tensor(s_data, s_grad)
        return s






