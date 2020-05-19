#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
import random as rd


#导数为 sg*(1 - sg)

def sigmiod(x):
    return 1 / (1 + np.exp(-x))


#分类函数
def fitLine2(x, y, labels, w, b, learnRate = 0.1,epoch=30):

    #拟合
    '''
    1.求出输出h
    2.求出loss
    3.求出dw db
    4.更新w b
    '''
    for e in range(epoch):
        for xItem,yItem,label in zip(x, y, labels):
            out = w * xItem + b
            #不连续
            #寻找连续 且 输出为0 1 的函数
            # h = (0 if (out - yItem) > 0 else 1) - label

            h = out  - label
            loss = h**2

            dw = -2*h*xItem
            db = -2*h

            dw = np.clip(dw, -2, 2)
            db = np.clip(db, -2, 2)

            w = w + learnRate*dw
            b = b + learnRate*db

    return w,b



#分类 在直线之上分类为1  在直线之下分类为0
x = np.array([i  for i in range(100) if (0<=i<=20 or 80<=i<=100)])
y = np.array([3*i + 4 + rd.random() if n <= 50 else 3*i + 500 + rd.random()  for n,i in enumerate(x)])
labels = np.array([0 if i <= 50 else 1  for i in x])

w = rd.random()
b = rd.random()


#作图
w, b = fitLine2(x, y,labels, w, b, epoch=30)
# w, b = fitLine(x, y, w, b, epoch=50)

y2 = np.array([w*i+b for i in x])
plt.scatter(x, labels)
plt.plot(x, y2)
plt.show()
print(w ,b)

















