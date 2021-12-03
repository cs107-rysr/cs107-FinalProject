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

        a_data = np.array([0.1, 0.3])
        a = st.tensor([0.1, 0.3])

        b_data = 3 * a_data
        b_grad = 3
        b = 3 * a

        c_data = np.sinh(b_data)
        c_grad = b_grad * np.cosh(b_data)
        c = F.sinh(b)

        d_data = np.cosh(b_data)
        d_grad = b_grad * np.sinh(b_data)
        d = F.cosh(b)

        e_data = np.tanh(b_data)
        e_grad = b_grad * (1. / np.cosh(b_data)**2)
        e = F.tanh(b)

        f_data = np.exp(b_data) / (np.exp(b_data) + 1)
        f_grad = b_grad * (np.exp(b_data) / (np.exp(b_data) + 1)**2)
        f = F.logistic(b)

        g_data = np.sqrt(b_data)
        g_grad = b_grad * (1 / (2 * np.sqrt(b_data)))
        g = F.sqrt(b)

        h_data = np.arcsin(b_data)
        h_grad = b_grad * (1 / (np.sqrt(1 - b_data**2)))
        h = F.arcsin(b)

        i_data = np.arccos(b_data)
        i_grad = b_grad * (-1 / (np.sqrt(1 - b_data**2)))
        i = F.arccos(b)

        j_data = np.arctan(b_data)
        j_grad = b_grad * (1 / (1 + b_data**2))
        j = F.arctan(b)

        self.assertTrue((b.data == b_data).all())
        self.assertTrue((b.grad == b_grad).all())
        self.assertTrue((c.data == c_data).all())
        self.assertTrue((c.grad == c_grad).all())
        self.assertTrue((d.data == d_data).all())
        self.assertTrue((d.grad == d_grad).all())
        self.assertTrue((e.data == e_data).all())
        self.assertTrue((e.grad == e_grad).all())
        self.assertTrue((f.data == f_data).all())
        self.assertTrue((f.grad == f_grad).all())
        self.assertTrue((g.data == g_data).all())
        self.assertTrue((g.grad == g_grad).all())
        self.assertTrue((h.data == h_data).all())
        self.assertTrue((h.grad == h_grad).all())
        self.assertTrue((i.data == i_data).all())
        self.assertTrue((i.grad == i_grad).all())
        self.assertTrue((j.data == j_data).all())
        self.assertTrue((j.grad == j_grad).all())

        # still need to test forward mode comparison operators

if __name__ == '__main__':
    unittest.main()