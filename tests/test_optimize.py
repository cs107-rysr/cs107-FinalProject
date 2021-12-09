import unittest
import numpy as np 
import spladtool.spladtool_reverse as str
from spladtool.utils import SGD


class TestOptimize(unittest.TestCase):
    def near(self, x, y):
        return np.abs(x - y).sum() < 1e-5

    def test_classification(self):
        np.random.seed(42)
        x = np.random.randn(200, 2)
        y = ((4 * x[:, 0] - 3 * x[:, 1]) > 1).astype(int)

        class MyModel(str.Module):
            def __init__(self):
                super().__init__()
                self.register_param(w1=str.tensor(np.random.randn()))
                self.register_param(w2=str.tensor(np.random.randn()))
                self.register_param(b=str.tensor(np.random.randn()))
            
            def forward(self, x):
                w1 = self.params['w1'].repeat(x.shape[0])
                w2 = self.params['w2'].repeat(x.shape[0])
                b = self.params['b'].repeat(x.shape[0])
                y = w1 * str.tensor(x[:, 0]) + w2 * str.tensor(x[:, 1]) + b
                return y

        # define loss function and optimizer
        model = MyModel()
        criterion = str.BCELoss()
        opt = SGD(model.parameters(), lr=0.1, momentum=0.9)

        # training using SGD with momentum
        losses = []
        for epoch in range(100):
            outputs = model(x)
            targets = str.tensor(y.astype(float))
            loss = criterion(targets, outputs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.data))
        self.assertLess(losses[-1], 0.1)

    def test_regression(self):
        np.random.seed(42)
        x = np.random.randn(500)
        x_sq = (x ** 2)
        x_cb = (x ** 3)
        y = 4 * x - 5 * x_sq + 3 * x_cb + np.random.randn(500) * 3 - 2

        class PolyModel(str.Module):
            def __init__(self):
                super().__init__()
                self.register_param(w1=str.tensor(np.random.randn()))
                self.register_param(w2=str.tensor(np.random.randn()))
                self.register_param(w3=str.tensor(np.random.randn()))
                self.register_param(b=str.tensor(np.random.randn()))
            
            def forward(self, x):
                w1 = self.params['w1'].repeat(x.shape[0])
                w2 = self.params['w2'].repeat(x.shape[0])
                w3 = self.params['w3'].repeat(x.shape[0])
                b = self.params['b'].repeat(x.shape[0])
                y = w1 * str.tensor(x) + w2 * str.tensor(x ** 2) + w3 * str.tensor(x ** 3) + b
                return y
            
            
        model = PolyModel()
        criterion = str.MSELoss()
        opt = SGD(model.parameters(), lr=0.001, momentum=0.9)

        # training using SGD with momentum
        losses = []
        for epoch in range(100):
            outputs = model(x)
            targets = str.tensor(y.astype(float))
            loss = criterion(targets, outputs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.data))

        self.assertLess(losses[-1], 10)
 
if __name__ == '__main__':
    unittest.main()