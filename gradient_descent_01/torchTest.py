#coding=utf-8


import torch

x = torch.tensor([2.], requires_grad=True)
b = torch.tensor([2.], requires_grad=True)
y = x**3 + b

#对y求导 y可能有多个变量
# y.backward()
# #求出对x的导数
# print(x.grad)
# print(b.grad)

print(torch.autograd.grad(y, [x, b]))