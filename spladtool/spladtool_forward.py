import numpy as np
from typing import Union, List

# TENSOR SCRIPT ========================================
class Tensor():
    def __init__(self, x=None, grad=None, seed=None):
        super().__init__()
        if x is None:
            self.data = None
        else:
            assert type(x) in [np.ndarray, list, int, float, np.float_]
            if type(x) != np.ndarray:
                x = np.array(x)
        self.data = x
        self._shape = x.shape
        if grad is None:
            grad = np.ones_like(self.data)
        if seed is None:
            seed = np.ones_like((x.shape[0],))
        self.grad = grad * seed

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'spladtool.Tensor(\n%s\n)' % str(self.data)

    def __add__(self, y):
        return sumup(self, y)

    def __radd__(self, y):
        return self.__add__(y)

    def __mul__(self, y):
        return prod(self, y)

    def __rmul__(self, y):
        return self.__mul__(y)

    def __truediv__(self, y):
        return div(self, y)

    def __rtruediv__(self, y):
        return div(y, self)

    def __pow__(self, y):
        return power(self, y)

    def __rpow__(self, *args):
        raise NotImplementedError

    def __neg__(self):
        return neg(self)

    def __sub__(self, y):
        return minus(self, y)

    def __rsub__(self, y):
        return minus(y, self)

    def __eq__(self, y):
        return equal(self, y)

    def __lt__(self, y):
        return less(self, y)

    def __gt__(self, y):
        return greater(self, y)

    def __ne__(self, y):
        return not_equal(self, y)

    def __le__(self, y):
        return less_equal(self, y)

    def __ge__(self, y):
        return greater_equal(self, y)

    @property
    def shape(self):
        return self._shape
# TENSOR SCRIPT ========================================

# LAYER SCRIPT ========================================
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

def div(x: Union[Tensor, int, float, Tensor, np.ndarray, list],
        y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
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

def minus(x: Union[Tensor, int, float, Tensor, np.ndarray, list],
          y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
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

def exp_base(x: Tensor, base: float):
    return Exp_Base()(x, base)

def log(x: Tensor):
    return Log()(x)

def log_base(x: Tensor, base: float):
    return Log_Base()(x, base)

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

def abs(x: Tensor):
    return Abs()(x)

def equal(x: Tensor, y):
    return Equal()(x, y)

def less(x: Tensor, y):
    return Less()(x, y)

def not_equal(x: Tensor, y):
    return NotEqual()(x, y)

def greater(x: Tensor, y):
    return Greater()(x, y)

def less_equal(x: Tensor, y):
    return LessEqual()(x, y)

def greater_equal(x: Tensor, y):
    return GreaterEqual()(x, y)

def tensor(x, seed=None):
    return Tensor(x, seed)
# FUNCTIONAL SCRIPT ====================================


# LAYER SCRIPT ====================================
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

class Exp_Base(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.ExpBase'

    def forward(self, x: Tensor, base: float) -> Tensor:
        s_data = base ** x.data
        s_grad = x.data * np.power(base, x.data - 1) * x.grad
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

class Abs(Layer):
    def __init__(self):
        super().__init__()
        self.desc = "spladtool.Layer.Abs"

    def forward(self, x: Tensor) -> Tensor:
        s_data = np.abs(x.data)
        s_grad = x.data / np.abs(x.data) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class Log(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.LogBase'

    def forward(self, x: Tensor) -> Tensor:
        if (x.data <= 0).any():
            raise ValueError('Cannot take the log of something less than or equal to 0.')
        s_data = np.log(x.data)
        s_grad = 1. / x.data * x.grad
        s = Tensor(s_data, s_grad)
        return s

class Log_Base(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.LogBase'

    def forward(self, x: Tensor, base: float) -> Tensor:
        if (x.data <= 0).any():
            raise ValueError('Cannot take the log of something less than or equal to 0.')
        s_data = np.log(x.data) / np.log(base)
        s_grad = (1. / (x.data * np.log(base))) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class ArcSin(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.ArcSin'

    def forward(self, x: Tensor):
        if (x.data < -1).any() or (x.data > 1).any():
            raise ValueError('Cannot perform ArcSin on something outside the range of [-1,1].')
        s_data = np.arcsin(x.data)
        s_grad = (1. / np.sqrt(1 - x.data ** 2)) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class ArcCos(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.ArcCos'

    def forward(self, x: Tensor):
        if (x.data < -1).any() or (x.data > 1).any():
            raise ValueError('Cannot perform ArcCos on something outside the range of [-1,1].')
        s_data = np.arccos(x.data)
        s_grad = (-1. / np.sqrt(1 - x.data ** 2)) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class ArcTan(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.ArcTan'

    def forward(self, x: Tensor):
        s_data = np.arctan(x.data)
        s_grad = (1. / (1 + x.data ** 2)) * x.grad
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
        s_grad = (1. / np.cosh(x.data) ** 2) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class Logistic(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.Logistic'

    def forward(self, x: Tensor):
        s_data = np.exp(x.data) / (np.exp(x.data) + 1)
        s_grad = (np.exp(x.data) / (np.exp(x.data) + 1) ** 2) * x.grad
        s = Tensor(s_data, s_grad)
        return s

class SquareRoot(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool.Layer.SquareRoot'

    def forward(self, x: Tensor):
        if (x.data <= 0).any():
            raise ValueError('Cannot take the square root of something less than 0.')
        s_data = np.sqrt(x.data)
        s_grad = (1. / (2 * np.sqrt(x.data))) * x.grad
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








