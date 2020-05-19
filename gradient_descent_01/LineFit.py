#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
import random as rd



#拟合方程
#输出 w b
#为何会产生震荡？
def fitLine(x, y, w, b, learnRate = 0.1,epoch=30):

    #拟合
    '''
    1.求出输出h
    2.求出loss
    3.求出dw db
    4.更新w b
    '''
    for e in range(epoch):
        for xItem,yItem in zip(x, y):
            out = w * xItem + b
            h = out - yItem

            loss = h**2
            #针对每个点求导 由于有波动干扰 数值会变得很大
            print(-2 * h * xItem)

            # dw = -2*h*xItem
            # db = -2*h
            #对梯度做出约束可以解决x 取值大问题
            dw = np.clip(-2 * h * xItem, -2, 2)
            db = np.clip(-2*h, -2, 2)

            w = w + learnRate*dw
            b = b + learnRate*db

    return w,b



#拟合曲线
x = np.array([i for i in range(100)])
y = np.array([3*i + 4 + rd.random() for i in x])

w = rd.random()
b = rd.random()




#作图
w, b = fitLine(x, y, w, b, epoch=100)
# w, b = fitLine(x, y, w, b, epoch=50)

y2 = np.array([w*i+b for i in x])
plt.scatter(x, y)
plt.plot(x, y2)
plt.show()
print(w ,b)

















