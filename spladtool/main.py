import unittest
import numpy as np
from spladtool_forward import spladtool_f as sf
from spladtool_reverse import spladtool_r as sr

print('------------------------In forward mode------------------------')
print('Example 1: z = x**2')
print('Inputs:')
x = sf.Tensor([3])
print('x : ', x)
z = x**2
print('Results:')
print('z.data: ', z)
print('z.grad:', z.grad)
print('x.grad', x.grad) #dz/dx

print('------------------------In reverse mode------------------------')
print('Example 1: z = -y / x^3')
print('Inputs:')
x = sr.Tensor([1., 2.])
y = 2 * x + 1
print('x : ', x)
print('y : ', y)
z = - y / (x ** 3)
z.backward() # execute backward lazily
print('Results:')
print('z: ', z)
print('x.grad', x.grad) #dz/dx
print('y.grad', y.grad) #dz/dy

print('------------------------In reverse mode------------------------')
print('Example: z = tanh(2xy)')
x = sr.Tensor([1,0.5])
y = sr.Tensor([0.7, 1])
print('Input:')
print('x : ', x.data)
print('y : ', y.data)
print('Results:')
z = sr.tanh(2*x*y)
print('z: ', z)
z.backward() # execute backward lazily
print('x.grad', x.grad) #dz/dx
print('y.grad', y.grad) #dz/dx

print('------------------------In reverse mode------------------------')
print('Example: x >= y')
x = sr.Tensor([2,-2])
y = sr.Tensor([-2,0])
print('Input:')
print('x : ', x.data)
print('y : ', y.data)
print('Results:')
z = (x > y)
print('z.data', z.data)
print('z.grad', x.grad)

print('------------------------In reverse mode------------------------')
print('Example: z = log(x)')
print('**************Should raise an error if we input negative number**************')
x = sr.Tensor([2])
# y = str.Tensor([0.7, 1])
print('Input:')
print('x : ', x.data)
# print('y : ', y.data)
print('Results:')
z = sr.sqrt(x)
print('z: ', z)
z.backward() # execute backward lazily
print('x.grad', x.grad) #dz/dx
# print('y.grad', y.grad) #dz/dx