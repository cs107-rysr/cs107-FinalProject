# Documentation for SPLADTool



## Introduction

With the rapid development of deep learning, auto differentiation has become an indispensable part of multiple optimization algorithms like gradient descent. Numerical means such as Newton's Method and finite-difference method is useful in some situations, we desire to compute the analytical solutions by applying chain rules with our automatic differentiation SPLADTool (**S**imple **P**ytorch-**L**ike **A**uto **D**ifferentiation **Tool**kit), which will be faster and more accurate than numerical methods.

## Background

To help users better understand how automatic differentiation works, we will briefly explain some crucial background concepts applied in the calculations of automatic differentiation.

### Jacobian Matrix

Jacobian Matrix is an important concept in vector calculus, and it also helps us to understand the calculations in the automatic differentiation. Given a mapping $h: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian matrix of h is as follows:

$$
J  =  {\begin{bmatrix}{\dfrac {\partial h_1}{\partial x_1}}&\cdots &{\dfrac {\partial h_{1}}{\partial x_{n}}}\\\vdots &\ddots &\vdots \\{\dfrac {\partial h_{m}}{\partial x_{1}}}&\cdots &{\dfrac {\partial h_{m}}{\partial x_{n}}}\end{bmatrix}}
$$

### Chain Rule

Chain rule is the most important concepts in Automatic Differentiation.

Consider compound function $h(y(\mathbf x))$, where $\mathbf x \in \mathbb{R}^m$, where $y(\mathbf x) = [y_1(\mathbf x), y_2(\mathbf x),\cdots y_n(\mathbf x)]^T$, The gradient of $h$ w.r.t $\mathbf x$ can be computed as follows:

$$
\nabla_\mathbf x h = \sum_{i=1}^n\dfrac{\partial h}{\partial u}\nabla_\mathbf x u + \dfrac{\partial h}{\partial v}\nabla_\mathbf x v
$$

### Gradient Computational Graph

#### Forward Mode

In forward mode, the gradients and the evaluations of basic nodes are computed all together along the forward process. We will use a simple function, $f(x) = \sin(x) + x^2 + 1$ (evaluate at $x = 1$), as an example to illustrate the graph structure of calculations:

We break down the function into several elementary functions. At each node, i.e. $x_i$, we only calculate one elementary operation, its corresponding derivative and their corresponding values. For the initial nodes, the gradients are assigned to be all $1$ vector.

<img src="ForwardExample.png">

The corresponding trace table is:

| trace | elementary operation | current value | elementary derivative | $\nabla_x$  current value |
| ----- | -------------------- | ------------- | --------------------- | ------------------------- |
|$x_1$  | $x$          | $1$ | $\dot{x}$          | $1$ |
|$x_2$| $\sin(x)$ | $\sin(1)$ | $\cos(x)\dot{x}$| $\cos(1)$|
|$x_3$| $x^2$ | $1$ | $2x\dot{x}$ | $2$ |
|$x_4$| $x_2 + x_3$ | $\sin(1) + 1$ | $\cos(x)\dot{x} + \dot{x}$ | $\cos(1) + 1$|
|$x_5$| $x_4 + 1$ | $\sin(1) + 2$ | $\cos(x)\dot{x} + \dot{x}$ | $\cos(1) + 2$|
|$f$| $x_4 + 1$ | $\sin(1) + 2$ | $\cos(x)\dot{x} + \dot{x}$ | $\cos(1) + 2$|

#### Reverse Mode (Optional)

In the reversed mode, the final node's gradient is set to 1. Along the forward process, only function evaluations will be done. Only during the backward process the gradients with respect to the nodes will be computed.


## Implementation

### Data Structures

#### Tensor

The core data structure here is the `spladtool.Tensor`, which contains the value vector (that will be represented by a `numpy.ndarray`) and corresponding gradient. 

In the reverse mode, we need two more attributes or member variables to keep record of the graph dependency: `Tensor.dependency` tracks the dependent tensor and `Tensor.layer` will store the layer or the operation used to attain this tensor. We will explain further how they are used. In the reverse mode, we also add a member function called `Tensor.reverse()`, which will automatically call the `reverse` method of `Tensor.layer` with arguments being `Tensor.dependency` to achieve reverse propagation.

#### Layer

A layer is defined as a basic operations, i.e. sum, product, division, sine function, etc.

All layer classes inherit from a base class called `Layer`. For the forward mode, the member function `Layer.forward()` computes the evaluation and gradients altogether. In the reverse mode, `Layer.forward()` will only handle the evaluation, while `Layer.reverse()` will handle the gradients computation.

### Functional APIs

We wrap up our implementations of operations in functional APIs which can be found in `spladtool_forward/functional.py` or `spladtool/functional.py`.

We also add dunders or magic functions to `Tensor` class so that basic operators can be used on them.

### Python Typing

To make sure the type is  correct, we add python typing to each of the operation classes and functional APIs to make sure the library will raise proper exceptions when encountered with unsupported operations.

### Testing, CI & Coverage Report

We adopt `unittest` as our testing framework. The up-to-now dev-only test script can be found in `./test.py`. As of continuous integration(CI), we are using Travis CI. For coverage report, we use the `coverage` package and upload the result to CodeCov. Find the results of our CI & coverage report by clicking on the badge in `README.md`



## Usage

1. Install dependencies:
   - NumPy

2. Try out an example from `test.py` on arithmetic functions:

   ```python
   import spladtool_forward as st
   x = st.tensor([[1., 2.], [3., 4.]])
           
   # Define output functions y(x) and z(x)
   y = 2 * x + 1
   z = - y / (x ** 3)
   
   # Print out the values calculated by our forward mode automatic differentiation SPLADTool
   print('x : ', x)
   print('y : ', y)
   print('y.grad : ', y.grad)
   print('z: ', z)
   print('z.grad: ', z.grad)
   ```



## Software Organization

```
cs107-FinalProject/
├── README.md
├── LICENSE
├── requirements.txt
├── .travis.yml
├── docs
│   ├── documentation
│   └── ...
├── spladtool
│   ├── __init__.py
│   ├── functional.py
│   ├── layer.py
│   └── tensor.py
└── tests
   ├── tests_basic.py
   ├── tests_comp.py
   └── tests_elem.py
```

Our team plans to include the numpy module as the dependency of the auto-differentiation module and the torch, coverage, and codecov modules to perform tests. 

- The numpy package will be used to work with matrices and perform matrix operations
- The torch package will be used to check our package against PyTorch's automatic differentiation engine, and the coverage and codecov modules will be used to produce coverage reports on the tests. (**This will only be needed in Dev mode**)

The test suite will live in TravisCI and provide coverage reports to Codecov, where the reports are stored.

Our package will be distributed on the Python Package Index (PyPI). This will be done using the following steps:

   1. Add a pyproject.toml file to the project and insert necessary specifications
   2. Install build : python -m pip install build 
   3. Build using : python -m build .
   4. Upload to PyPI using : twine upload dist/* 

## Licensing

This project will be licensed using the traditional MIT license due to several factors. 

- We will be using code from the NumPy library which the MIT license coincides with. 
- As of now, we do not foresee having to deal with any patents or any other dependencies. 
- Since this project won’t contain an abundance of novel code (and, therefore, could be duplicated quite easily), we don’t mind letting others use it as they please. 
- Due to the small scale of the project, we are hoping to use a license which is similarly simple. The MIT license is the best match for our interests outlined above. 

## Feedback

### Milestone 1

   1) Couldn't read the mathematical equations as they didn't render
	- Changed file to .ipynb for easier rendering
   2) Referred to reverse mode incorrectly as backward mode
	- Modified text to correct usage
   3) Didn't include how users should install/download package
	- Included more information about dependencies and included commands for users
   4) Didn't discuss packaging of software
	- Included basic process we will follow regarding building and uploading package









