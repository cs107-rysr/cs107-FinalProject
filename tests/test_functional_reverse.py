import unittest
import numpy as np 
import torch
import spladtool_reverse.spladtool_reverse as str


class TestBackwardFunctionalOps(unittest.TestCase):
    def near(self, x, y):
        return np.abs(x - y).sum() < 1e-5

    def test_trig(self):
        x = str.tensor([[1., 2.], [3., 4.]])
        tx = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        z = str.sin(x) + str.cos(x) + str.tan(x)
        tz = torch.sin(tx) + torch.cos(tx) + torch.tan(tx)

        z.backward()
        external_grad = torch.tensor([[1., 1.], [1.,1.]])
        tz.backward(gradient=external_grad)

        print('x : ', x)
        print('z: ', z)
        print('x.grad: ', x.grad)

        self.assertTrue((self.near(z.data, tz.detach().numpy())).all())
        self.assertTrue((self.near(x.grad, np.array(tx.grad))).all())

    def test_inv_trig(self):
        x = str.tensor([[0.5, 0.5], [0.5, 0.5]])
        tx = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
        z = str.arcsin(x) * str.arccos(x) + str.arctan(x)
        tz = torch.arcsin(tx) * torch.arccos(tx) + torch.arctan(tx)

        z.backward()
        external_grad = torch.tensor([[1., 1.], [1.,1.]])
        tz.backward(gradient=external_grad)

        print('x : ', x)
        print('z: ', z)
        print('x.grad: ', x.grad)

        self.assertTrue((self.near(z.data, tz.detach().numpy())).all())
        self.assertTrue((self.near(x.grad, np.array(tx.grad))).all())

    def test_hyp(self):
        x = str.tensor([[0.5, 0.5], [0.5, 0.5]])
        tx = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
        z = str.sinh(x) * str.cosh(x) / str.tanh(x)
        tz = torch.sinh(tx) * torch.cosh(tx) / torch.tanh(tx)

        z.backward()
        external_grad = torch.tensor([[1., 1.], [1.,1.]])
        tz.backward(gradient=external_grad)

        print('x : ', x)
        print('z: ', z)
        print('x.grad: ', x.grad)

        self.assertTrue((self.near(z.data, tz.detach().numpy())).all())
        self.assertTrue((self.near(x.grad, np.array(tx.grad))).all())

    def test_exp_log_sqrt(self):
        x = str.tensor([[1., 2.], [3., 4.]])
        tx = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        z = str.sqrt(x) + str.exp(x) + str.log(x)
        tz = torch.sqrt(tx) + torch.exp(tx) + torch.log(tx)

        z.backward()
        external_grad = torch.tensor([[1., 1.], [1.,1.]])
        tz.backward(gradient=external_grad)

        print(tx.grad)
        print('x : ', x)
        print('z: ', z)
        print('x.grad: ', x.grad)

        self.assertTrue((self.near(z.data, tz.detach().numpy())).all())
        self.assertTrue((self.near(x.grad, np.array(tx.grad))).all())

    

if __name__ == '__main__':
    unittest.main()