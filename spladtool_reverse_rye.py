import numpy as np
from typing import Union, List


# TENSOR SCRIPT ========================================
class Tensor():
    '''
    A class for lazily-computed variables with auto-differentiation
    '''

    def __init__(self, x=None):
        super().__init__()
        if x is None:
            self.data = None
        else:
            assert type(x) in [np.ndarray, list, int, float]
            if type(x) != np.ndarray:
                x = np.array(x)
        self.data = x
        self._grad = np.zeros_like(self.data)
        self.dependency = None
        self.layer = None
        self._shape = x.shape

    def backward(self, g=None):
        # print('output: ', self, 'input: ', self.dependency, 'layer: ', self.layer, 'grad: ', g)
        if g is None:
            g = np.ones_like(self.data)
        assert g.shape == self.data.shape
        # had to change this to make it work, was originally self._grad += g
        self._grad = self._grad + g
        if self.dependency is not None:
            self.layer.backward(*self.dependency, g)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'spladtool_reverse.Tensor(%s)' % str(self.data)

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

    @property
    def shape(self):
        return self._shape

    @property
    def grad(self):
        return self._grad
# TENSOR SCRIPT ========================================


# FUNCTIONAL SCRIPT ====================================
def power(x, p):
    return Power(p)(x)


def sumup(x: Tensor, y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
    if type(y) == Tensor:
        return TensorSum()(x, y)
    else:
        return NumSum(y)(x)


def prod(x: Tensor, y: Union[int, float, Tensor, np.ndarray, list]) -> Tensor:
    if type(y) == Tensor:
        return TensorProd()(x, y)
    else:
        return NumProd(y)(x)


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

def exp(x: Tensor) -> Tensor:
    return Exp()(x)

def log(x: Tensor) -> Tensor:
    return Log()(x)

def sin(x: Tensor) -> Tensor:
    return Sin()(x)

def cos(x: Tensor) -> Tensor:
    return Cos()(x)

def arcsin(x: Tensor):
    return ArcSin()(x)

def arccos(x: Tensor):
    return ArcCos()(x)

def arctan(x: Tensor):
    pass
    # return ArcTan()(x)

def sinh(x: Tensor):
    pass
    # return Sinh()(x)

def cosh(x: Tensor):
    pass
    # return Cosh()(x)

def tanh(x: Tensor):
    pass
    # return Tanh()(x)

# FUNCTIONAL SCRIPT ====================================


# LAYER SCRIPT ====================================
class Layer():
    def __init__(self):
        super().__init__()
        self.name = 'spladtool_reverse.Layer'

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __call__(self, *args):
        return self.forward(*args)


class Power(Layer):
    def __init__(self, p):
        super().__init__()
        self.desc = 'spladtool_reverse.Layer.Power'
        self.p = p

    def forward(self, x):
        y = Tensor(np.power(x.data.copy(), self.p))
        y.dependency = [x]
        y.layer = self
        return y

    def backward(self, x, g):
        grad = self.p * np.power(x.data.copy(), self.p - 1) * g
        x.backward(grad)


class TensorSum(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool_reverse.Layer.TensorSum'

    def forward(self, x, y):
        assert x.shape == y.shape
        s_data = x.data + y.data
        s = Tensor(s_data)
        s.dependency = [x, y]
        s.layer = self
        return s

    def backward(self, x, y, g):
        x_grad = g.copy()
        y_grad = g.copy()
        x.backward(x_grad)
        y.backward(y_grad)


class TensorProd(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool_reverse.Layer.TensorProd'

    def forward(self, x, y):
        assert x.shape == y.shape
        p_data = x.data * y.data
        p = Tensor(p_data)
        p.dependency = [x, y]
        p.layer = self
        return p

    def backward(self, x, y, g):
        x_grad = y.data * g
        y_grad = x.data * g
        x.backward(x_grad)
        y.backward(y_grad)


class TensorInv(Layer):
    def __init__(self):
        super().__init__()
        self.desc = 'spladtool_reverse.Layer.TensorInv'

    def forward(self, x):
        i_data = 1. / x.data
        i = Tensor(i_data)
        i.dependency = [x]
        i.layer = self
        return i

    def backward(self, x, g):
        grad = -1. / (x.data ** 2) * g
        x.backward(grad)


class NumProd(Layer):
    def __init__(self, num):
        super().__init__()
        self.desc = 'spladtool_reverse.Layer.NumProd'
        if type(num) == list:
            self.num = np.array(num)
        else:
            self.num = num

    def forward(self, x):
        s_data = x.data * self.num
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        grad = self.num * g
        x.backward(grad)


class NumSum(Layer):
    def __init__(self, num):
        super().__init__()
        self.desc = 'spladtool_reverse.Layer.NumSum'
        if type(num) == list:
            self.num = np.array(num)
        else:
            self.num = num

    def forward(self, x):
        s_data = x.data + self.num
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        x.backward(g)

class Exp(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'spladtool_reverse.Layer.Exp'

    def forward(self, x):
        s_data = np.exp(x.data)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        grad = g * np.exp(x.data)
        x.backward(grad)

class Log(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'spladtool_reverse.Layer.Log'

    def forward(self, x):
        s_data = np.log(x.data)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        grad = g * (1. / x.data)
        x.backward(grad)

class Sin(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'spladtool_reverse.Layer.Sin'

    def forward(self, x):
        s_data = np.sin(x.data)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        grad = g * np.cos(x.data)
        x.backward(grad)

class Cos(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'spladtool_reverse.Layer.Cos'

    def forward(self, x):
        s_data = np.cos(x.data)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        grad = g * np.sin(x.data)
        x.backward(grad)

# need to raise error?? derivative is undefined outside of [-1,1]
# both ArcSin and ArcCos work with 1 input, but not with 2 inputs (e.g. z = arcsin(x*y)
class ArcSin(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'spladtool_reverse.Layer.ArcSin'

    def forward(self, x):
        s_data = np.arcsin(x.data)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        grad = g * (1. / np.sqrt(1 - x.data**2))
        x.backward(grad)

class ArcCos(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'spladtool_reverse.Layer.ArcCos'

    def forward(self, x):
        s_data = np.arccos(x.data)
        s = Tensor(s_data)
        s.dependency = [x]
        s.layer = self
        return s

    def backward(self, x, g):
        grad = g * (-1. / np.sqrt(1 - x.data**2))
        x.backward(grad)

# def arctan(x: Tensor):
# def sinh(x: Tensor):
# def cosh(x: Tensor):
# def tanh(x: Tensor):


def tensor(x):
    return Tensor(x)

# LAYER SCRIPT ====================================