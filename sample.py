import unittest
import numpy as np 
import spladtool_reverse as str


x = str.tensor([1., 2.])
y = 2 * x + 1
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
print('------------------------In reverse mode------------------------')
print('x : ', x)
print('y : ', y)
print('z: ', z)
z.backward() # execute backward lazily
print('x.grad', x.grad) #dz/dx