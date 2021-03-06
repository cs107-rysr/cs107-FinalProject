import unittest
import numpy as np 
import spladtool.spladtool_reverse as str



class TestCompareBackward(unittest.TestCase):
    def test_synthesis_reverse(self):
        x = str.tensor([[1., 2.], [3., 4.]])
        y = [[1, 2], [3, 4]]
        z = np.array([[3, 4], [1, 2]])
        w = 4
        xy = (x == y)
        xz1= (x < z)
        xz2 = (x > z)
        xz3 = (x <= z)
        xz4 = (x >= z)
        xw = (x != w)
        print(xy)
        self.assertTrue((xy.data == (x.data == y)).all())
        self.assertTrue((xz1.data == (x.data < z)).all())
        self.assertTrue((xz2.data == (x.data > z)).all())
        self.assertTrue((xz3.data == (x.data <= z)).all())
        self.assertTrue((xz4.data == (x.data >= z)).all())
        self.assertTrue((xw.data == (x.data != w)).all())
 
 
if __name__ == '__main__':
    unittest.main()