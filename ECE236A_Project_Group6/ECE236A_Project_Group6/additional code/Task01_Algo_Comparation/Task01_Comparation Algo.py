#导入所需库函数
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import csv
import os
import random

def prepare_synthetic_data():
    data = dict()

    X = np.loadtxt(r'D:\project_code_236\Code\Data\synthetic\synthetic_X.csv', delimiter=',').reshape(1500, 2)
    Y = np.loadtxt(r'D:\project_code_236\Code\Data\synthetic\synthetic_Y.csv', delimiter=',')

    data['trainX'] = X[:1000]  # 1000 x 2
    data['trainY'] = Y[:1000]  # 1000 x 1
    data['testX'] = X[1000:]   # 500 x 2
    data['testY'] = Y[1000:]   # 500 x 1

    return data

#Task部分
syn = prepare_synthetic_data()
#print(syn)
#print(type(syn))
x1=[]
x2=[]
for n in range(1000):
        x1.append(syn['trainX'][n][0])
        x2.append(syn['trainX'][n][1])

x11=[]
x22=[]
tesY=syn['testY']
for n in range(500):
        x11.append(syn['testX'][n][0])
        x22.append(syn['testX'][n][1])
#plt.scatter(x1,x2)
#plt.scatter(x11,x22)
#plt.show()
#print predicted label
plt.figure(1)
colors = ['r', 'g', 'b']
for lab in range(3):
    for i in range(2000):
        sumloss = 0
        # 随机选择分界线参数w1,w2,b
        w01 = random.uniform(-300, 300)
        w02 = random.uniform(-300, 300)
        b = random.randint(-500, 500)
        for n in range(1000):
            if syn['trainY'][n]==lab:
                s = 1
            else:
                s = -1
            loss = 1-s*(w01*syn['trainX'][n][0]+w02*syn['trainX'][n][1]+b)
            if loss>=0:
                loss = loss
            elif loss<0:
                loss = 0
            sumloss+=loss
        #判断是否小于自定的代价上限
        if sumloss<=2800:
            break
    #print(sumloss)
    #如果小于上限，则确认w,b
    print(w01)
    print(w02)
    print(b)
    """
    #画出分界线
    x0 = np.array([-3,-2,-1,0,1])
    y0 = (w01*x0+b)/(-w02)
    plt.plot(x0,y0,c=colors[lab])
    # 重置坐标轴
    plt.axis('auto')
    """
    #plt.xlim([-10, 9])  # 设置 x 轴范围
    #plt.ylim([-10, 9])  # 设置 y 轴范围
    #plt.scatter(x1,x2)
    #plt.show()
    #进行测试
    #创建矩阵对测试集预测类别数据进行收集
    testclass = np.ones((500,1))
    #将x11,x22代入超平面方程，如果大于等于0，则划分为0类
    for i in range(500):
        if w01*x11[i]+w02*x22[i]+b>=0:
            testclass[i]=0
    #画出所有0类散点图
    #colors0 = '#00CED1' #0类点的颜色
    #colors1 = '#DC143C'
    for n in range(500):
        if testclass[n]==0:
            plt.scatter(x11[n],x22[n],c=colors[lab])
        else:
            continue
            #plt.scatter(x11[n], x22[n], c=colors1)
    x0 = np.array([-2,-1,0,1,2,3])
    y0 = (w01*x0+b)/(-w02)
    plt.plot(x0,y0,c=colors[lab])
plt.ylim(-5,5)
plt.show()


#print true label
plt.figure(2)
for n in range(500):
        if tesY[n]==0:
            plt.scatter(x11[n],x22[n],c=colors[0])
        if tesY[n]==1:
            plt.scatter(x11[n],x22[n],c=colors[1])
        if tesY[n]==2:
            plt.scatter(x11[n],x22[n],c=colors[2])
plt.show()





