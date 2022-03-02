# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:14:16 2021

@author: MaYiming
"""
from adaclass import StumpAdaBoost
import numpy as np
import random
#导入数据
def loadSimpData():
    #数据和标签
    datMat = np.array([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    #进行一下洗牌操作
    Data = []
    data = []
    for i in range(len(datMat)):
        data = []
        for j in datMat[i]:
            data.append(j)
        data.append(classLabels[i])
        Data.append(data)
    #洗牌
    random.shuffle(Data)
    Data = np.array(Data)
    #重划分
    datMat = Data[:,0:2]
    classLabels = Data[:,-1]
    return datMat,classLabels
#导入数据
dataset1,labelset1 = loadSimpData()
NumberOfClass1 = 14
#训练
ada = StumpAdaBoost(dataset1, labelset1, NumberOfClass1)