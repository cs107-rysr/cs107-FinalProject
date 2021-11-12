import unittest
import numpy as np 
from lib import tensor as t

class TestBasic(unittest.TestCase):

    def test_add(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x + st_y, t(x + y))
        self.assertEqual(st_y + st_x, t(y + x))

    def test_sub(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x - st_y, t(x - y))
        self.assertEqual(st_y - st_x, t(y - x))

    def test_mult(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x * st_y, t(x * y))
        self.assertEqual(st_y * st_x, t(y * x))

    def test_div(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x / st_y, t(x / y))
        self.assertEqual(st_y / st_x, t(y / x))

    def test_pow_tensor(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x ** st_y, t(x ** y))
        self.assertEqual(st_y ** st_x, t(y ** x))

    def test_pow_const(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = 3
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x ** st_y, t(x ** y))
        self.assertEqual(st_y ** st_x, t(y ** x))

    def test_neg(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(-st_x, -t(x))
        self.assertEqual(-st_y, -t(y))
 
if __name__ == '__main__':
    unittest.main()