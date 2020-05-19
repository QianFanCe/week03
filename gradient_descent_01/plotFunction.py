#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

#指数函数

def indexFun(a, x):
    return a**x


#对数函数
def logFun(a, x):
    return np.log(x) / np.log(a)

#sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return max(0, x)

def leakRelu(x):
    return max(0.1*x, x)

#变化更平滑
def moreLeakRelu(a,x):
    return x if x >= 0 else a*(np.exp(x) - 1)



x = np.arange(-2, 2, 0.1)
y1 = [indexFun(0.1, i) for i in x]
y2 = [indexFun(10, i) for i in x]


x = np.arange(0.1, 2, 0.1)
y1 = [logFun(0.1, i) for i in x]
y2 = [logFun(10, i) for i in x]
#

x = np.arange(-10, 10, 0.1)
y1 = [sigmoid(i) for i in x]
y2 = [tanh(i) for i in x]


x = np.arange(-10, 10, 0.1)
y1 = [moreLeakRelu(1 ,i) for i in x]
y2 = [leakRelu(i) for i in x]


plt.plot(x, np.zeros_like(x))
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()








