import numpy as np 

class Tensor():
    def __init__(self, x=None):
        super().__init__()
        if x is None:
            self.data = None
        else:
            assert type(x) in [np.ndarray, list]
            if type(x) == list:
                x = np.array(x)
        self.data = x
        self.grad = None
        self.dependency = None
        self.layer = None

    def backward(self):
        self.grad = 1.
        self.layer.backward(self.dependency)
        
    def __repr__(self):
    	return str(self)
    	
    def __str__(self):
    	return 'spladtool.Tensor(%s)' % str(self.data)
