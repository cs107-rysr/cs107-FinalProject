import numpy as np 

class Tensor():
    def __init__(self, x):
        super().__init__()
        assert type(x) in [np.array, list]
        if type(x) == list:
            x = np.array(x)
        self.data = x
        self.grad = None
        self.dependency = None
        self.layer = None

    def backward():
        self.grad = 1.
        self.layer.backward(self.dependency)
