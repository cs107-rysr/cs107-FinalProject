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
        new_tensor = Tensor()
        new_tensor.data = x.data.copy()
        new_tensor.data = new_tensor.data ** 2
        new_tensor.dependency = x
        new_data.Layer = self

    def __call__(self, x):
        return self.forward(x)

    def backward(self, y):
        x.grad = Tensor()
        x.grad.data = x.data.copy()
        x.grad.data = x.grad.data * 2
        return x.grad

    def __str__(self):
        return 'spladtool.Layer.Square'