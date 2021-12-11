# cs107-FinalProject

# SPLADTool: **S**imple **P**yTorch-**L**ike **A**utomatic **D**ifferentiation Tools

[![Build Status](https://app.travis-ci.com/cs107-rysr/cs107-FinalProject.svg?token=e6J2xSxxbBcpWz7pkoA3&branch=main)](https://app.travis-ci.com/cs107-rysr/cs107-FinalProject)
[![codecov](https://codecov.io/gh/cs107-rysr/cs107-FinalProject/branch/main/graph/badge.svg?token=49FWUPH6X1)](https://codecov.io/gh/cs107-rysr/cs107-FinalProject)

Fall 2021 CS107 Final Project Automatic Differentiation

# Info
Group Number: #31
Group Members:
Shihan Lin
Yuanbiao Wang
Raymond Jow
Rye Julson

## Usage

1. Create a virtual environment: Conda
    ```bash
    conda create --name spladtool_env python
    ```
    
   Activate the environment;
   ```bash
   conda activate spladtool_env
   ```
   
   Deactivate the envrionment after use:
   ```bash
   conda deactivate
   ```

2. Download the package to local:
    ```bash
    git clone https://github.com/cs107-rysr/cs107-FinalProject.git
    ```
    Install our dependencies by `requirement.txt` with anaconda or python
    ```bash
    conda install --yes --file requirements.txt
    # or
    pip install -r requirements.txt
    ```

3. Try out an example from `test.py` on arithmetic functions:

   ```python
   import spladtool_forward as st
   import splatool_forward.functional as F

   x = st.tensor([[1., 2.], [3., 4.]])
           
   # Define output functions y(x) and z(x)
   y = 2 * x + 1
   z = - y / (x ** 3)
   w = F.cos((F.exp(z) + F.exp(-z)) / 2)
   
   # Print out the values calculated by our forward mode automatic differentiation SPLADTool
   print('x : ', x)
   print('y : ', y)
   print('y.grad : ', y.grad)
   print('z: ', z)
   print('z.grad: ', z.grad)
   print('w: ', w)
   print('w.grad: ', w.grad)
   ```

4. To run given tests, under UNIX environment, use
```
sh test.sh
```

Under Windows environment, run
```
coverage run test.py
coverage report
```

## Implementation

### Data Structures

#### Tensor

The core data structure here is the `spladtool.Tensor`, which contains the value vector (that will be represented by a `numpy.ndarray`) and corresponding gradient. 

In the reverse mode, we need two more attributes or member variables to keep record of the graph dependency: `Tensor.dependency` tracks the dependent tensor and `Tensor.layer` will store the layer or the operation used to attain this tensor. We will explain further how they are used. In the reverse mode, we also add a member function called `Tensor.backward()`, which will automatically call the `backward` method of `Tensor.layer` with arguments being `Tensor.dependency` to achieve reverse propagation.

#### Layer

A layer is defined as a basic operations, i.e. sum, product, division, sine function, etc.

All layer classes inherit from a base class called `Layer`. For the forward mode, the member function `Layer.forward()` computes the evaluation and gradients altogether. In the reverse mode, `Layer.forward()` will only handle the evaluation, while `Layer.reverse()` will handle the gradients computation.

### Functional APIs

We wrap up our implementations of operations in functional APIs which can be found in `spladtool_forward/spladtool_forward.py`.

We also add dunders or magic functions to `Tensor` class so that basic operators can be used on them.

### Supported Operations(**New**)
- Basic Operations: Add, Substract, Power, Negation, Product, Division
- Analytical functions: trignomical, exponential, logarithm

### Python Typing

To make sure the type is  correct, we add python typing to each of the operation classes and functional APIs to make sure the library will raise proper exceptions when encountered with unsupported operations.

### Testing, CI & Coverage Report

We adopt `unittest` as our testing framework. The up-to-now dev-only test script can be found in `./test.py`. As of continuous integration(CI), we are using Travis CI. For coverage report, we use the `coverage` package and upload the result to CodeCov. Find the results of our CI & coverage report by clicking on the badge in `README.md`


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
├── tests
│   ├── tests_basic.py
│   ├── tests_comp.py
│   └── tests_elem.py
├── test.py
└── test.sh
```

Our team plans to include the numpy module as the dependency of the auto-differentiation module and the torch, coverage, and codecov modules to perform tests. 

- The numpy package will be used to work with matrices and perform matrix operations
- The torch package will be used to check our package against PyTorch's automatic differentiation engine, and the coverage and codecov modules will be used to produce coverage reports on the tests. (**This will only be needed in Dev mode**)

The test suite will live in TravisCI and provide coverage reports to Codecov, where the reports are stored.

In details, we need to do the following things:

1. Add a licence to our software. See the **Licensing** section

2. Create an conda virtual environment.

3. Install all the dependencies.

4. Register an account using an organization email

5. Following PEP517, install `setuptools` ,`twine` and `build` by

   ```shell
   python3 -m pip install --upgrade setuptools build twine
   ```

6. Add a `setup.cfg` or `setup.py` configuration file.

7. Execute build and upload by

   ```shell
   python3 -m build
   python3 -m twine upload --repository testpypi dist/*
   ```
   
## Broader Impact and Inclusivity Statement

### Broader Impact

- Our implementation of automatic differentiation provides a fast and accurate way of calculating derivatives. Our SPLAD Tool package is handy and straightforward to apply in many fields. When handling large-scale computation, utilizing our package will relieve calculation workload and avoid computational errors. Besides, the package is also helpful in dealing with a wide range of mathematical problems. For example, by adding the implementation of loss functions, we were able to apply our spladtool_reverse to construct a simple data classifier, which is demonstrated in detailed under the exmaple directory. Furthermore, our package can also be used to construct root-finding algorithms based on Newton's method. 


- While our automatic differentiation package provides many conveniences and can be applied widely in many fields, it might also be misused in some conditions. As a convenient tool for calculating derivatives automatically, our package might hinder students or beginners from thoroughly learning and understanding the basic theory behind the mechanism. This misuse contradicts our original intentions of helping people study and work more efficiently.


### Software Inclusivity

- In order to make our package to be as inclusive as possible, we intend to publish our package as an open-source resource online. By distributing over Github and PyPI, we allow people from all kinds of backgrounds to be able to download, use and coordinate with us. Furthermore, we also encourage other developers from all communities to contribute to our codebase by enabling people to create new pull requests, leave comments in our repository on Github. All of our group members will continue monitoring new comments and pull requests and schedule meetings at any time to discuss further improvement and optimization if needed.


- Furthermore, to eliminate the barrier to underrepresented groups, we expect to implement new features in the future concerning different communities respectively. For example, to help eliminate the language barrier to non-native English speakers, we expect to provide detailed instructions in multiple languages other than English. Besides, if possible, we may build a GUI that can visualize the trace of automatic differentiation to help users better understand the working flow of automatic differentiation.


## Future Features

For our possible future features, we intend to add data structures and new methods in our implmentation to support matrix multiplication. Besides, we may apply our package to solve more complex problems, like neural network with mutiple layers.


## Licensing

This project will be licensed using the traditional MIT license due to several factors. 

- We will be using code from the NumPy library which the MIT license coincides with. 
- As of now, we do not foresee having to deal with any patents or any other dependencies. 
- Since this project won’t contain an abundance of novel code (and, therefore, could be duplicated quite easily), we don’t mind letting others use it as they please. 
- Due to the small scale of the project, we are hoping to use a license which is similarly simple. The MIT license is the best match for our interests outlined above. 
