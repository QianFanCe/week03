#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


#梯度下降法
def gradLine(xs, ys, w, b, rate=0.1, epoch=50):
    '''

    1 前向
    2 loss
    3 链式求导，求导dw db
    4更新
    '''
    for i in range(epoch):
        for x, y in zip(xs, ys):
            out = w*x+b
            h = out - y
            loss = h**2


            dw = np.clip(-2*h*x, -2, 2)
            db = np.clip(-2*h, -2, 2)

            # dw = -2*h*x
            # db = -2*h


            w = w + dw * rate
            b = b + db * rate


            #print("dw={},db={},loss={}".format(dw, db, loss))

    return w,b


import random as rd
xs = np.array([i for i in range(100)])
ys = np.array([3*i+2+rd.random() for i in xs])

w = rd.random()
b = rd.random()

pre_w, pre_b = gradLine(xs, ys, w, b)

#作图
#散点图和曲线图
plt.scatter(xs, ys)
plt.plot(xs, pre_w*xs + pre_b, color='r')
plt.show()




















