import numpy as np 

class Tensor():
    def __init__(self, x=None):
        super().__init__()
        self.data = x
        self.grad = None
        self.dependency = None
        self.layer = None

    def backward():
        self.grad = 1.
        self.layer.backward(self.dependency)
