import spladtool.spladtool_reverse as str
from spladtool.utils import SGD
from sklearn import datasets
import numpy as np


# We chose the sklearn Iris toy dataset as an example
iris = datasets.load_iris() 
x = np.array(iris.data)
y = np.array(iris.target)

# define a linear regression module

class Linear(str.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_param(w=str.tensor(np.random.randn(in_features)))
        self.register_param(b=str.tensor(np.random.randn(1)))
    
    def forward(self, x):
        y = str.matvecmul(x, self.params['w'])
        return y

# define loss function and optimizer
model = Linear(4, 1)
criterion = str.BCELoss()
opt = SGD(model.parameters(), lr=1e-2)

# training
for epoch in range(10):
    inputs = str.tensor(x)
    targets = str.tensor(y)
    outputs = model(inputs)
    loss = criterion(targets, outputs)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss)
