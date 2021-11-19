import numpy as np 
import spladtool_forward.functional as F


class Tensor():
    def __init__(self, x=None, grad=None):
        super().__init__()
        if x is None:
            self.data = None
        else:
            assert type(x) in [np.ndarray, list, int, float]
            if type(x) != np.ndarray:
                x = np.array(x)
        self.data = x
        self.shape = x.shape
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad
        
    def __repr__(self):
    	return str(self)
    	
    def __str__(self):
    	return 'spladtool.Tensor(\n%s\n)' % str(self.data)

    def __add__(self, y):
        return F.sumup(self, y)
    
    def __radd__(self, y):
        return F.self.__add__(y)
    
    def __mul__(self, y):
        return F.prod(self, y)

    def __rmul__(self, y):
        return self.__mul__(y)

    def __truediv__(self, y):
        return F.div(self, y)

    def __rtruediv__(self, y):
        return F.div(y, self)

    def __pow__(self, y):
        return F.power(self, y)

    def __rpow__(self, y):
        return F.power(y, self)

    def __neg__(self):
        return F.neg(self)

    def __sub__(self, y):
        return F.minus(self, y)

    def __rsub__(self, y):
        return F.minus(y, self)  

    def __eq__(self, y):
        return (self.data == y.data).all()
