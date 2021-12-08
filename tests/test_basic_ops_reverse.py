import unittest
import numpy as np 
from spladtool.spladtool_reverse import spladtool_r as sr


class TestBasicReverse(unittest.TestCase):
    def near(self, x, y):
        return np.abs(x - y).sum() < 1e-7

    def test_define_layer(self):
        class DefinedLayer(sr.Layer):
            def __init__(self):
                super().__init__()
                self.name = 'my_defined_layer'

            def forward(self, x):
                y = 2 * x + 1
                z = - y / (x ** 3)
                return z

        defined_layer = DefinedLayer()
        x = sr.tensor([[1., 2.], [3., 4.]])
        z = defined_layer(x)
        z.backward()
        # x.grad
        print('------------------In reverse mode(defined layer)------------------')
        print('x : ', x)
        print('z: ', z)
        print('x.grad: ', x.grad)
        self.assertTrue((z.data == np.array([[-3., -5. / 8], [-7. / 27, -9. / 64]])).all())
        self.assertTrue((self.near(x.grad, np.array([[7., 11./16], [15. / 81, 19. / 256]]))).all())

 
if __name__ == '__main__':
    unittest.main()