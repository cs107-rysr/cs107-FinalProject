import unittest
import spladtool_forward as sp_f

class TestBasic(unittest.TestCase):

    def test_add(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = sp_f.tensor(x)
        st_y = sp_f.tensor(y)
        self.assertEqual(st_x + st_y, sp_f.tensor(x + y))
        self.assertEqual(st_y + st_x, sp_f.tensor(y + x))

    def test_sub(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = sp_f.tensor(x)
        st_y = sp_f.tensor(y)
        self.assertEqual(st_x - st_y, sp_f.tensor(x - y))
        self.assertEqual(st_y - st_x, sp_f.tensor(y - x))

    def test_mult(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = sp_f.tensor(x)
        st_y = sp_f.tensor(y)
        self.assertEqual(st_x * st_y, sp_f.tensor(x * y))
        self.assertEqual(st_y * st_x, sp_f.tensor(y * x))

    def test_div(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = sp_f.tensor(x)
        st_y = sp_f.tensor(y)
        self.assertEqual(st_x / st_y, sp_f.tensor(x / y))
        self.assertEqual(st_y / st_x, sp_f.tensor(y / x))

    def test_pow_tensor(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = sp_f.tensor(x)
        st_y = sp_f.tensor(y)
        self.assertEqual(st_x ** st_y, sp_f.tensor(x ** y))
        self.assertEqual(st_y ** st_x, sp_f.tensor(y ** x))

    def test_pow_const(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = 3
        st_x = sp_f.tensor(x)
        st_y = sp_f.tensor(y)
        print(st_y ** st_x)
        self.assertEqual(st_x ** st_y, sp_f.tensor(x ** y))
        self.assertEqual(st_y ** st_x, sp_f.tensor(y ** x))

    def test_neg(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = sp_f.tensor(x)
        st_y = sp_f.tensor(y)
        self.assertEqual(-st_x, -sp_f.tensor(x))
        self.assertEqual(-st_y, -sp_f.tensor(y))
 
if __name__ == '__main__':
    import numpy as np
    unittest.main()