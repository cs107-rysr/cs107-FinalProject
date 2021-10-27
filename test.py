import unittest
import spladtool_forward as st
import numpy as np


class TestArithmeticForward(unittest.TestCase):
    def test_arithmetic1(self):
        x = st.tensor([[1., 2.], [3., 4.]])
        y = 2 * x + 1
        z = - y / (x ** 3)
        print('x : ', x)
        print('y : ', y)
        print('y.grad : ', x.grad)
        print('z: ', z)
        print('z.grad: ', z.grad)
        self.assertTrue((y.data == np.array([[3., 5.], [7., 9.]])).all())
        self.assertTrue((y.grad == np.array([[2., 2.], [2., 2.]])).all())
        self.assertTrue((z.data == np.array([[-3., -5. / 8], [-7. / 27, -9. / 64]])).all())
        self.assertTrue((z.grad == np.array([[7., 11./16], [15. / 81, 19. / 256]])).all())


if __name__ == '__main__':
    unittest.main()
