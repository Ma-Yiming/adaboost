# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:33:58 2021

@author: MaYiming
"""
#np里边的向量是行向量
import numpy as np
import matplotlib.pyplot as plt
import random

def loadSimpData():
    datMat = np.array([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    Data = []
    data = []
    for i in range(len(datMat)):
        data = []
        for j in datMat[i]:
            data.append(j)
        data.append(classLabels[i])
        Data.append(data)
    random.shuffle(Data)
    Data = np.array(Data)
    datMat = Data[:,0:2]
    classLabels = Data[:,-1]
    return datMat,classLabels

def loadDataSet(path):
    Dataset = []
    Labels = []
    with open(path) as p:
        data = []
        for line in p:
            data = line.strip().split('\t')
            Dataset.append(data[:-1])
            Labels.append(data[-1])
    return np.mat(Dataset).astype(float),np.array(Labels).astype(float)

params = {}
params["Train_path"] = "horseColicTraining2.txt"
params["Test_path"] = "horseColicTest2.txt"

def stumpClassify(dataset,featureIndex,threshVal,threshInep):
    res = np.ones((np.shape(dataset)[0],1))
    if threshInep == 'lt':
        res[dataset[:,featureIndex] <= threshVal] = -1.0
    else:
        res[dataset[:,featureIndex] >= threshVal] = -1.0
    #这里的res是竖向量
    return res

def buildStumpFurther(dataset,label,D):
    m, n = np.shape(dataset)
    numSteps = 10.0
    bestClassifier = {}
    bestClassEst = np.zeros((m,1))
    minError = np.inf
    for i in range(n):
        columnMin = dataset[:,i].min()
        columnMax = dataset[:,i].max()
        stepsize = (columnMax-columnMin)/numSteps
        for j in range(-1,int(stepsize+1)):
            for threshInep in ['lt','gt']:
                threshVal = columnMin + j*stepsize
                pre = stumpClassify(dataset, i, threshVal, threshInep)
                err = np.ones((m,1))
                err[pre == label.reshape(-1,1)] = 0
                errweight = D*err
                if errweight.sum() <= 0.5:
                    if errweight.sum() < minError:
                        minError = errweight.sum()
                        bestClassEst = pre.copy()
                        bestClassifier['dim'] = i
                        bestClassifier['thresh'] = threshVal
                        bestClassifier['ineq'] = threshInep
    return bestClassifier,bestClassEst,minError

def AdaBoostTrain(dataset,label,numberOfClass):
    classifierArr = []
    m = np.shape(dataset)[0]
    rand = []
    for i in range(m):
        rand.append(random.random())
    rand = np.array(rand)
    D = np.ones((m,1)) + rand
    D = D/D.sum()
    pre = np.zeros((m,1))
    err = []
    errRate = 100.0
    for i in range(numberOfClass):
        bestClassifier,prelabel,minError = buildStumpFurther(dataset, label, D)
        alpha = float(0.5*np.log((1-minError)/max(minError,1e-16)))
        bestClassifier["alpha"] = alpha
        classifierArr.append(bestClassifier)
        rand = []
        for i in range(m):
            rand.append(random.random())
        rand = np.array(rand)
        D = D*np.exp(-1*alpha*(prelabel*label.reshape(-1,1))) + rand
        D = D/D.sum()
        pre += (alpha*prelabel)
        intergrationError = (np.array(np.sign(pre) != label.reshape(-1,1))).sum()
        errRate = intergrationError/m
        err.append(errRate)
        if errRate == 0.0:
            break
    print(err)
    return classifierArr,pre,err

def test(Test_path,Train_path,numberOfClass):
    Train_dataset,Train_labels = loadDataSet(Train_path)
    classifierArr,pre = AdaBoostTrain(Train_dataset, Train_labels, numberOfClass)
    Test_dataset,Test_labels = loadDataSet(Test_path)
    pre = np.zeros((Test_dataset.shape[0],1))
    for i in range(numberOfClass):
        base_class = stumpClassify(Test_dataset, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        pre += (classifierArr[i]['alpha']*base_class)
    err = np.zeros((Test_dataset.shape[0],1))
    err[np.sign(pre) != Test_labels.reshape(-1,1)] = 1
    errRate = err.sum()/Test_dataset.shape[0]
    return errRate
numberOfClass = 100
dataset,label = loadSimpData()
AdaBoostTrain(dataset, label, numberOfClass)
# err = []
# for numberOfClass in range(10):
#     errRate = test(params["Test_path"], params["Train_path"], numberOfClass)
#     err.append(errRate)
# plt.plot(err)
