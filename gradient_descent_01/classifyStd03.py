#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
import random as rd


#导数为 sg*(1 - sg)

def sigmiod(x):
    return 1 / (1 + np.exp(-x))
#导数为 1- tanh**2
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + 0.000001)



#分类函数
def fitLine2(x, y, labels, w, b, learnRate = 0.2,epoch=30):

    #拟合
    '''
    1.求出输出h
    2.求出loss
    3.求出dw db
    4.更新w b
    '''
    #indx = np.arange(0,100)
    indx = np.arange(0,40)


    for e in range(epoch):
        np.random.shuffle(indx)
        for idx in indx:
            xItem = x[idx]
            yItem = y[idx]
            label = labels[idx]

        #for xItem,yItem,label in zip(x, y, labels):
            out = w * xItem + b
            #规约到一定范围内
            #如果太远导致 有偏差loss，但是没有梯度dh
            trans = (out - yItem) / 1000
            h = np.tanh(trans) - label
            loss = h**2

            #if np.tanh(out - yItem)>0:
            #    print( out - yItem, np.tanh(out - yItem), h, loss)

            dh = 1 - np.tanh(trans) ** 2
            dw = -2*h*dh*xItem/1000
            db = -2*h*dh/1000

            print("x, y :", xItem, yItem)
            print( 'out - yItem:{} ,  np.tanh(out - yItem):{}, h:{}, dh:{}, loss:{} '.format(out - yItem, np.tanh(out - yItem), h, dh, loss))

            #print(dw, dh)
            dw = np.clip(dw, -2, 2)
            db = np.clip(db, -2, 2)

            # dw = np.clip(dw, -10, 10) if label == 1 else np.clip(dw, -5, 5)
            # db = np.clip(db, -10, 10) if label == 1 else np.clip(db, -5, 5)

            w = w + learnRate*dw
            b = b + learnRate*db

    return w,b



#分类 在直线之上分类为1  在直线之下分类为0
x = np.array([i for i in range(100)])
x = np.array([i  for i in range(100) if (0<=i<=20 or 80<=i<=100)])
y = np.array([3*i + 4 + rd.random() if n <= 50 else 3*i + 500 + rd.random()  for n,i in enumerate(x)])
labels = np.array([-1 if i <= 50 else 1  for i in x])

w = rd.random()
b = rd.random()


#作图
w, b = fitLine2(x, y,labels, w, b, epoch=30)
# w, b = fitLine(x, y, w, b, epoch=50)

y2 = np.array([w*i+b for i in x])
plt.scatter(x, y)
plt.plot(x, y2)
plt.show()
print(w ,b)

















