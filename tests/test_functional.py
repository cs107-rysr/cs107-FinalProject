import unittest
import numpy as np 
import spladtool_forward as st
import spladtool_forward.functional as F


class TestFunctionals(unittest.TestCase):
    def near(self, x, y):
        return np.abs(x - y).sum() < 1e-7

    def test_functionals(self):
        """Test coverage for functionals not covered in test synthesis"""
        x_data = np.array([[1., 2.], [3., 4.]])
        x_grad = np.array([[1., 1.], [1., 1.]])
        x = st.tensor([[1., 2.], [3., 4.]])

        m_data = np.cosh(x_data)
        m_grad = x_grad * np.sinh(x_data)
        m = F.cosh(x)

        n_data = np.tanh(x_data)
        n_grad = x_grad * (1 / np.cosh(x_data)) ** 2
        n = F.tanh(x)

        o_data = np.exp(x.data) / (np.exp(-x.data) + 1)
        o_grad = x_grad * np.exp(x_data) / (1 + np.exp(x_data))**2
        o = F.logistic(x)

        p_data = np.sqrt(x_data)
        p_grad = x_grad * (1. / 2 * np.sqrt(x_data))
        p = F.sqrt(x)

        q_data = np.abs(x_data)
        q_grad = x_grad * ( x_data / np.abs(x_data)) 
        q = F.abs(x)

        # forward mode test for functionals 
        self.assertTrue((x.data == x_data).all())
        self.assertTrue((m.data == m_data).all())
        self.assertTrue((n.data == n_data).all())
        self.assertTrue((o.data == o_data).all())
        self.assertTrue((p.data == p_data).all())
        self.assertTrue((q.data == q_data).all())


        # forward mode test for gradients
        self.assertTrue((self.near(m.grad, m_grad)).all())
        self.assertTrue((self.near(n.grad, n_grad)).all())
        self.assertTrue((self.near(o.grad, o_grad)).all())
        self.assertTrue((self.near(p.grad, p_grad)).all())
        self.assertTrue((self.near(q.grad, q_grad)).all())

    def test_inv_functionals(self):
        x_data = np.array([[1, 0.5]])
        x_grad = np.array([[1., 1.], [1., 1.]])
        x = st.tensor([[1, 0.5]])

        s_data = np.arcsin(x_data)
        s_grad = x_grad * (1. / np.sqrt(1 - x_data**2))
        s = F.arcsin(x)

        t_data = np.arccos(x_data)
        t_grad = x_grad * (-1. / np.sqrt(1 - x_data**2))
        t = F.arccos(x)

        v_data = np.arctan(x_data)
        v_grad = x_grad * (1. / (1 + x.data**2))
        v = F.arctan(x)

        self.assertTrue((x.data == x_data).all())
        self.assertTrue((s.data == s_data).all())
        self.assertTrue((t.data == t_data).all())
        self.assertTrue((v.data == v_data).all())
        
        self.assertTrue((s.grad == s_grad).all())
        self.assertTrue((t.grad == t_grad).all())
        self.assertTrue((v.grad == v_grad).all())
        
if __name__ == '__main__':
    unittest.main()