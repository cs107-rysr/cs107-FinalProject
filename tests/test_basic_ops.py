import unittest
import numpy as np 
import spladtool.spladtool_forward as stf


class TestBasic(unittest.TestCase):
    
    def test_seed(self):
        seed = [[1.], [0], [0]]
        sf_x = stf.tensor([[1.0], [2.0], [3.0]], seed=seed)
        sf_z = sf_x + 4
        print('x : ', sf_x)
        print('z : ', sf_z)
        print('z.grad: ', sf_z.grad)
        self.assertTrue((sf_z.grad == np.array([[1.],[0],[0]])).all())
        
    def test_add(self):
        x = np.array([[1.0], [2.0], [3.0]])
        z = x + 4
        sf_x = stf.tensor([[1.0], [2.0], [3.0]])
        sf_z = sf_x + 4
        print('x : ', sf_x)
        print('z : ', sf_z)
        print('z.grad: ', sf_z.grad)
        self.assertTrue((sf_x.data + 4 == z).all())
        self.assertTrue((4 + sf_x.data == 4 + x).all())
        self.assertTrue((sf_z.grad == np.array([[1.],[1.],[1.]])).all())

    def test_sub(self):
        x = np.array([[1.0], [2.0], [3.0]])
        z = x - 4
        sf_x = stf.tensor([[1.0], [2.0], [3.0]])
        sf_z = sf_x - 4
        print('x : ', sf_x)
        print('z : ', sf_z)
        print('z.grad: ', sf_z.grad)
        self.assertTrue((sf_x.data - 4 == z).all())
        self.assertTrue((4 - sf_x.data == 4 - x).all())
        self.assertTrue((sf_z.grad == np.array([[1.],[1.],[1.]])).all())

    def test_mult(self):
        x = np.array([[1.0], [2.0], [3.0]])
        z = 3 * x
        sf_x = stf.tensor([[1.0], [2.0], [3.0]])
        sf_z = 3 * sf_x
        print('x : ', sf_x)
        print('z : ', sf_z)
        print('z.grad: ', sf_z.grad)
        self.assertTrue((sf_z.data == z).all())
        self.assertTrue((sf_x.data * 3 == x * 3).all())
        self.assertTrue((sf_z.grad == np.array([[3.],[3.],[3.]])).all())

    def test_div(self):
        x = np.array([[1.0], [2.0], [3.0]])
        z = x / 4
        sf_x = stf.tensor([[1.0], [2.0], [3.0]])
        sf_z = sf_x / 4
        print('x : ', sf_x)
        print('z : ', sf_z)
        print('z.grad: ', sf_z.grad)
        self.assertTrue((sf_z.data == z).all())
        self.assertTrue((4 / sf_x.data == 4 / x).all())
        self.assertTrue((sf_z.grad == np.array([[0.25],[0.25],[0.25]])).all())

    def test_pow_consf(self):
        x = np.array([[1.0], [2.0], [3.0]])
        z = x ** 3
        sf_x = stf.tensor(x)
        sf_z = sf_x ** 3
        print('x : ', sf_x)
        print('z : ', sf_z)
        print('z.grad: ', sf_z.grad)
        self.assertTrue((sf_z.data == z).all())
        self.assertTrue((sf_z.grad == (3 * x ** 2)).all())

    def test_neg(self):
        x = np.array([[1.0], [2.0], [3.0]])
        z = -x 
        sf_x = stf.tensor(x)
        sf_z = -sf_x
        print('x : ', sf_x)
        print('z : ', sf_z)
        print('z.grad: ', sf_z.grad)
        self.assertTrue((sf_z.data == z).all())
        self.assertTrue(((-sf_z).data == -z).all())
        self.assertTrue((sf_z.grad == np.array([[-1.],[-1.],[-1.]])).all())
    
    def test_synthesis(self):
        x = stf.tensor([[1., 2.], [3., 4.]])
        y = 2 * x + 1
        z = - y / (x ** 3)
        print('x : ', x)
        print('y : ', y)
        print('y.grad : ', y.grad)
        print('z: ', z)
        print('z.grad: ', z.grad)
        self.assertTrue((y.data == np.array([[3., 5.], [7., 9.]])).all())
        self.assertTrue((y.grad == np.array([[2., 2.], [2., 2.]])).all())
        self.assertTrue((z.data == np.array([[-3., -5. / 8], [-7. / 27, -9. / 64]])).all())
        self.assertTrue((z.grad == np.array([[7., 11./16], [15. / 81, 19. / 256]])).all())

    def test_repr(self):
        x = stf.tensor([[1., 2.], [3., 4.]])
        self.assertTrue(repr(x) == 'spladtool.Tensor(\n[[1. 2.]\n [3. 4.]]\n)')

 
if __name__ == '__main__':
    unittest.main()