import unittest
import spladtool as st


class TestSquare(unittest.TestCase):
    def test_square(self):
        x = st.tensor([4.0])
        y = st.square(x)
        y.backward()
        self.assertEqual(y.data, 16.0)
        self.assertEqual(x.grad.data, 8.0)


if __name__ == '__main__':
    unittest.main()
