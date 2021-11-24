import unittest
import numpy as np 
import spladtool_forward as st
import spladtool_forward.functional as F


class TestCompare(unittest.TestCase):
    def near(self, x, y):
        return np.abs(x - y).sum() < 1e-7

    def test_synthesis(self):
        x = st.tensor([[1., 2.], [3., 4.]])
        y = [[1, 2], [3, 4]]
        z = np.array([[3, 4], [1, 2]])
        w = 4
        xy = (x == y)
        xz1= (x < z)
        xz2 = (x > z)
        xz3 = (x <= z)
        xz4 = (x >= z)
        xw = (x != w)
        self.assertTrue((xy.data == (x.data == y)).all())
        self.assertTrue((xz1.data == (x.data < z)).all())
        self.assertTrue((xz2.data == (x.data > z)).all())
        self.assertTrue((xz3.data == (x.data <= z)).all())
        self.assertTrue((xz4.data == (x.data >= z)).all())
        self.assertTrue((xw.data == (x.data != w)).all())
 
 
if __name__ == '__main__':
    unittest.main()