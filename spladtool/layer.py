import numpy as np  
from .tensor import Tensor

class Layer():
    def __init__(self):
        super().__init__()
    
    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, x, y):
        raise NotImplementedError
    

class Square(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = Tensor(x.data.copy() ** 2)
        y.dependency = x
        y.layer = self
        return y

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x):
        x.grad = Tensor()
        x.grad.data = x.data.copy()
        x.grad.data = x.grad.data * 2
        return x.grad

    def __str__(self):
        return 'spladtool.Layer.Square'
