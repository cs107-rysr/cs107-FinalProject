import unittest
import numpy as np
from lib import tensor as t

class CompBasic(unittest.TestCase):

    def test_eq(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x == st_y, (x == y).all())
        self.assertEqual(st_y == st_x, (y == x).all())

    def test_lt(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x < st_y, (x < y).all())
        self.assertEqual(st_y < st_x, (y < x).all())

    def test_gt(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x > st_y, (x > y).all())
        self.assertEqual(st_y > st_x, (y > x).all())

    def test_le(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x <= st_y, (x <= y).all())
        self.assertEqual(st_y <= st_x, (y <= x).all())

    def test_ge(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x >= st_y, (x >= y).all())
        self.assertEqual(st_y >= st_x, (y >= x).all())

    def test_ne(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = t(x)
        st_y = t(y)
        self.assertEqual(st_x != st_y, (x != y).all())
        self.assertEqual(st_y != st_x, (y != x).all())


if __name__ == '__main__':
    unittest.main()