# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:15:32 2021

@author: MaYiming
"""
from adaclass import StumpAdaBoost
import numpy as np
import matplotlib.pyplot as plt
#导入数据
def loadDataSet(path):
    Dataset = []
    Labels = []
    #分割数据
    with open(path) as p:
        data = []
        for line in p:
            data = line.strip().split('\t')
            Dataset.append(data[:-1])
            Labels.append(data[-1])
    #需要注意的是，这里一定要转化为float型
    return np.mat(Dataset).astype(float),np.array(Labels).astype(float)
#参数设置
params = {}
params["Train_path"] = "horseColicTraining2.txt"
params["Test_path"] = "horseColicTest2.txt"
#数据导入
Train_dataset, Train_labelset = loadDataSet(params["Train_path"])
Test_dataset, Test_labelset = loadDataSet(params["Test_path"])
#误差
err = []
#开始测试
for numberOfClass in range(20):
    ada2 = StumpAdaBoost(Train_dataset, Train_labelset, numberOfClass)
    errRate = ada2.test(Test_dataset, Test_labelset)
    err.append(errRate)
#输出图像
plt.plot(err)
