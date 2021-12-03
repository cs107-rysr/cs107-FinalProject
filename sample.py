import unittest
import numpy as np 
import spladtool_reverse_rye as str

print('------------------------In reverse mode------------------------')
print('------------------------Two inputs-----------------------------')
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
print('------------------------One input-----------------------------')
x = str.Tensor([2])
y = str.Tensor([2])
print('Input:')
print('x : ', x)
# w1 = x
# w2 = w1^3
# z = w2
# z.backward()
print('Results:')
z = str.exp(x*y)
print('z: ', z)
z.backward() # execute backward lazily
print('x.grad', x.grad) #dz/dx
print('y.grad', y.grad) #dz/dx