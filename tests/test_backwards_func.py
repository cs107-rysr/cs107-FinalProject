import unittest
import numpy as np 
import torch
import spladtool.spladtool_reverse as str


class TestBackwardBasicOps(unittest.TestCase):
    def near(self, x, y):
        return np.abs(x - y).sum() < 1e-5

    def test_define_layer(self):
        class DefinedLayer(str.Layer):
            def __init__(self):
                super().__init__()
                self.name = 'my_defined_layer'

            def forward(self, x):
                y = 2 * x + 1
                w = str.sin(x) + str.cos(x)
                z = - (y * w )/ (x ** 3) 
                return z

        class TorchLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = 2 * x + 1
                w = torch.sin(x) + torch.cos(x)
                z = - (y * w )/ (x ** 3)
                return z

        defined_layer = DefinedLayer()
        torch_layer = TorchLayer()

        x = str.tensor([[1., 2.], [3., 4.]])
        tx = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

        z = defined_layer(x)
        z.backward()

        external_grad = torch.tensor([[1., 1.], [1.,1.]])
        tz = torch_layer(tx)
        tz.backward(gradient=external_grad)
        
        print('------------------In reverse mode(defined layer)------------------')
        print('x : ', x)
        print('z: ', z)
        print('x.grad: ', x.grad)
        self.assertTrue((self.near(z.data, tz.detach().numpy())).all())
        self.assertTrue((self.near(x.grad, np.array(tx.grad))).all())


 
if __name__ == '__main__':
    unittest.main()