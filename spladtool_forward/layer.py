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

    def forward(self, x: Tensor, p: float) -> Tensor:
        y_data = np.power(x.data.copy(), p)
        y_grad = p * np.power(x.data.copy(), p - 1) * x.grad
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

    def forward(self, x: Tensor, y: Union[int, float, list, np.ndarray]) -> Tensor:
        if type(y) == list:
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
        if type(y) == list:
            y = np.array(y)
        s_data = x.data + y
        s_grad = x.grad
        s = Tensor(s_data, s_grad)
        return s


class Exp(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Exp'
    
    def forward(self, x: Tensor) -> Tensor:
        s_data = np.exp(x.data)
        s_grad = np.exp(x.data) * x.grad
        s = Tensor(s_data, s_grad)
        return s
    

class Sin(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Sin'
    
    def forward(self, x: Tensor) -> Tensor:
        s_data = np.sin(x.data)
        s_grad = np.cos(x.data) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class Cos(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Cos'
    
    def forward(self, x: Tensor) -> Tensor:
        s_data = np.cos(x.data)
        s_grad = -np.sin(x.data) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class Tan(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Tan'
    
    def forward(self, x: Tensor) -> Tensor:
        s_data = np.tan(x.data)
        s_grad = 1 / (np.cos(x.data) * np.cos(x.data)) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class Log(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Log'
    
    def forward(self, x: Tensor) -> Tensor:
        s_data = np.log(x.data)
        s_grad = 1. / x.data * x.grad
        s = Tensor(s_data, s_grad)
        return s


class ArcSin(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.ArcSin'

    def forward(self, x: Tensor):
        s_data = np.arcsin(x.data)
        s_grad = (1. / np.sqrt(1 - x.data**2)) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class ArcCos(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.ArcCos'

    def forward(self, x: Tensor):
        s_data = np.arccos(x.data)
        s_grad = (-1. / np.sqrt(1 - x.data**2)) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class ArcTan(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.ArcTan'

    def forward(self, x: Tensor):
        s_data = np.arctan(x.data)
        s_grad = (1. / (1 + x.data**2)) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class Sinh(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Sinh'

    def forward(self, x: Tensor):
        s_data = np.sinh(x.data)
        s_grad = np.cosh(x.data) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class Cosh(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Cosh'

    def forward(self, x: Tensor):
        s_data = np.cosh(x.data)
        s_grad = np.sinh(x.data) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Tanh'

    def forward(self, x: Tensor):
        s_data = np.tanh(x.data)
        s_grad = (1. / np.cosh(x.data)**2) * x.grad
        s = Tensor(s_data, s_grad)
        return s


class Logistic(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Logistic'

    def forward(self, x: Tensor):
        s_data = np.exp(x.data) / (np.exp(x.data) + 1)
        s_grad = (np.exp(x.data) / (np.exp(x.data) + 1)**2) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class SquareRoot(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.SquareRoot'

    def forward(self, x: Tensor):
        s_data = np.sqrt(x.data)
        s_grad = (1. / 2 * np.sqrt(x.data)) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class Comparator(Layer):
    def __init__(self, cmp):
        super().__init__()
        self.cmp = cmp
        self.desc = 'spladtool.Layer.Comparator'
    
    def forward(self, x: Tensor, y: Union[float, int, np.ndarray, list, Tensor]) -> Tensor:
        if type(y) == int or type(y) == float:
            s_data = (self.cmp(x.data, y))
            s_grad = np.nan
            return Tensor(s_data, s_grad)
        elif type(y) == list:
            y = np.array(y)
        if (y.shape != x.shape):
            raise TypeError(f'param1{type(x)} and param2{type(y)} does not have the same shape')
        else:
            if type(y) == np.ndarray:
                s_data = (self.cmp(x.data, y))
            else:
                s_data = (self.cmp(x.data, y.data))
            s_grad = np.nan
            return Tensor(s_data, s_grad)


class Equal(Comparator):
    def __init__(self):
        super().__init__(np.equal)
        self.desc = 'spladtool.Layer.Equal'


class NotEqual(Comparator):
    def __init__(self):
        super().__init__(np.not_equal)
        self.desc = 'spladtool.Layer.NotEqual'
    

class Less(Comparator):
    def __init__(self):
        super().__init__(np.less)
        self.desc = 'spladtool.Layer.Less'


class Greater(Comparator):
    def __init__(self):
        super().__init__(np.greater)
        self.desc = 'spladtool.Layer.Greater'


class LessEqual(Comparator):
    def __init__(self):
        super().__init__(np.less_equal)
        self.desc = 'spladtool.Layer.LessEqual'


class GreaterEqual(Comparator):
    def __init__(self):
        super().__init__(np.greater_equal)
        self.desc = 'spladtool.Layer.GreaterEqual'
        
            






