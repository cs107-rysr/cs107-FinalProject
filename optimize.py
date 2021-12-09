import spladtool.spladtool_reverse as str
from spladtool.utils import SGD
import numpy as np


# We chose a simple classification model with decision boundary being 4x1 - 3x2 > 0
x = np.random.randn(200, 2)
y = ((x[:, 0] - 3 * x[:, 1]) > 0).astype(float)

# define a linear regression module

np.random.seed(42)

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

# training
for epoch in range(100):
    outputs = model(x)
    targets = str.tensor(y)
    loss = criterion(targets, outputs)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.data)
