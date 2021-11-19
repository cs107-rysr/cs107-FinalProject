import unittest
import numpy as np 
import spladtool_forward as st
import spladtool_forward.functional as F


class TestAnalytical(unittest.TestCase):
    def near(self, x, y):
        return np.abs(x - y).sum() < 1e-7

    def test_synthesis(self):
        x_data = np.array([[1., 2.], [3., 4.]])
        x = st.tensor([[1., 2.], [3., 4.]])

        y_data = (np.exp(x_data) + np.exp(-x_data)) / 2
        y_grad = (np.exp(x_data) - np.exp(-x_data)) / 2
        y = (F.exp(x) + F.exp(-x)) / 2 # this is the famous hyperbolic sin function

        z_data = np.sin(y_data)
        z_grad = y_grad * np.cos(y_data)
        z = F.sin(y)

        w_data = np.cos(y_data)
        w_grad = -y_grad * np.sin(y_data)
        w = F.cos(y)

        u_data = np.tan(y_data)
        u_grad = y_grad * 1 / (np.cos(y_data) ** 2)
        u = F.tan(y)

        v_data = np.log(y_data)
        v_grad = y_grad * 1 / y_data
        v = F.log(y)

        self.assertTrue((y.data == y_data).all())
        self.assertTrue((z.data == z_data).all())
        self.assertTrue((w.data == w_data).all())
        self.assertTrue((u.data == u_data).all())
        self.assertTrue((v.data == v_data).all())
        self.assertTrue((y.grad == y_grad).all())
        self.assertTrue((z.grad == z_grad).all())
        self.assertTrue((w.grad == w_grad).all())
        self.assertTrue((self.near(u.grad, u_grad)).all())
        self.assertTrue((self.near(v.grad, v_grad)).all())
 
 
if __name__ == '__main__':
    unittest.main()