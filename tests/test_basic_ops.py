import unittest
import numpy as np 
import spladtool_forward as st


class TestBasic(unittest.TestCase):
    def test_add(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = st.tensor(x)
        st_y = st.tensor(y)
        self.assertEqual(st_x + st_y, st.tensor(x + y))
        self.assertEqual(st_y + st_x, st.tensor(y + x))

    def test_sub(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = st.tensor(x)
        st_y = st.tensor(y)
        self.assertEqual(st_x - st_y, st.tensor(x - y))
        self.assertEqual(st_y - st_x, st.tensor(y - x))

    def test_mult(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = st.tensor(x)
        st_y = st.tensor(y)
        self.assertEqual(st_x * st_y, st.tensor(x * y))
        self.assertEqual(st_y * st_x, st.tensor(y * x))

    def test_div(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = st.tensor(x)
        st_y = st.tensor(y)
        self.assertEqual(st_x / st_y, st.tensor(x / y))
        self.assertEqual(st_y / st_x, st.tensor(y / x))

    def test_pow_const(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = 3
        st_x = st.tensor(x)
        self.assertEqual(st_x ** y, st.tensor(x ** y))

    def test_neg(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[4.0], [5.0], [6.0]])
        st_x = st.tensor(x)
        st_y = st.tensor(y)
        self.assertEqual(-st_x, -st.tensor(x))
        self.assertEqual(-st_y, -st.tensor(y))
    
    def test_synthesis(self):
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