import unittest
import numpy as np 
import spladtool_reverse as str

print('------------------------In reverse mode------------------------')
print('Example 1: z = -y / x^3')
print('Inputs:')
x = str.Tensor([1., 2.])
y = 2 * x + 1
print('x : ', x)
print('y : ', y)
# x
# a = x
# b = x
# c = 2 * a
# d = c + 1
# e = b ** 3
# f = 1 / e
# g = -y
# z = g * f
# z.backward()
z = - y / (x ** 3)
z.backward() # execute backward lazily
print('Results:')
print('z: ', z)
print('x.grad', x.grad) #dz/dx
print('y.grad', y.grad) #dz/dy

print('------------------------In reverse mode------------------------')
print('Example: z = tanh(2xy)')
x = str.Tensor([1,0.5])
y = str.Tensor([0.7, 1])
print('Input:')
print('x : ', x.data)
print('y : ', y.data)
print('Results:')
z = str.tanh(2*x*y)
print('z: ', z)
z.backward() # execute backward lazily
print('x.grad', x.grad) #dz/dx
print('y.grad', y.grad) #dz/dx