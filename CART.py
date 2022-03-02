# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:41:10 2021

@author: MaYiming
"""
import numpy as np
import matplotlib.pyplot as plt
#计算划分出去数据集的基尼指数
def Ginival(dataArr,weights):
    #参数初始化
    length = np.shape(dataArr)[0]
    num = 0.0
    labelClass = {}
    #利用字典键的唯一性，进行数据的分类
    for i in range(length):
        label = dataArr[i][-1]
        if label not in labelClass.keys():
            labelClass[label] = 0
        #数据分类时，乘上训练数据的权重
        labelClass[label]+=weights[i]
        num+=weights[i]
    #初始化
    gini = 1
    #计算基尼系数
    for key in labelClass.keys():
        prop = float(labelClass[key])/num
        gini-= prop*prop
    return gini
#划分数据集
def splitData(dataArr,weights,axis,value,thresh):
    length = np.shape(dataArr)[0]
    dataSet = []
    weightSet = []
    #等于特征值的分到左子树，不等于特征值分到右子树
    if thresh == 'left':
        for i in range(length):
            data = dataArr[i][axis]
            #数据和对应权重append进去
            if data == value:
                dataSet.append(dataArr[i])
                weightSet.append(weights[i])
    else:
        for i in range(length):
            data = dataArr[i][axis]
            #不等于的划分为右数
            if data != value:
                dataSet.append(dataArr[i])
                weightSet.append(weights[i])
    return dataSet,weightSet
#选择最优基尼指数
def chooseBestGini(dataArr,weights):
    m,n = np.shape(dataArr)
    minGini = 1.0;Bestaxis = -1;Bestvalue = ''
    #选特征
    for i in range(n-1):
        #特征所有的值
        feaList = [x[i] for x in dataArr]
        uniFea = set(feaList)
        #选值
        for val in uniFea:
            #分别计算左右子树的基尼指数
            left_dataSet,left_weightSet = splitData(dataArr,weights,i,val,'left')
            left_prob = len(left_dataSet) / m
            left_gini = Ginival(left_dataSet,left_weightSet)
            #计算右子树的基尼系数
            right_dataSet,right_weightSet = splitData(dataArr,weights,i,val,'right')
            right_prob = len(right_dataSet) / m
            right_gini = Ginival(right_dataSet,right_weightSet)
            #总体基尼系数
            gini = left_prob*left_gini+right_prob*right_gini
            #选最优
            if gini < minGini:
                minGini = gini
                Bestaxis = i
                Bestvalue = val
    return Bestaxis,Bestvalue
def createTree(dataArr,weights,axis,value):
    #初始化树
    Tree={}
    Tree[(axis,value)]={}
    classCount0=0
    classCount1=0
    predictArr = []
    m = np.shape(dataArr)[0]
    #我们构建的树，左树为预测类，右树不是预测所需
    for i in range(m):
        if dataArr[i][axis]==value:
            classCount0+=dataArr[i][-1]*weights[i]
        else:
            classCount1+=dataArr[i][-1]*weights[i]
    #Tree[(axis,value)][0]表示数据分到左子树（特征值等于value），Tree[(axis,value)][1]表示分到右子树（特征值不等于value）
    if classCount0>0:
        Tree[(axis,value)][0]=1
    else: 
        Tree[(axis,value)][0] = -1
    if classCount1>0:
        Tree[(axis,value)][1]=1
    else: 
        Tree[(axis,value)][1] =-1
    #进行预测，计算出我们的预测结果
    for i in range(m):
        if dataArr[i][axis] == value:
            predictArr.append(Tree[(axis,value)][0])
        else:
            predictArr.append(Tree[(axis,value)][1])
    return Tree,predictArr
#计算单个
def calEm(dataArr,predictArr,weights):
    #计算错误率
    m = np.shape(dataArr)[0]
    errorMat = np.mat(np.ones(m))
    #计算我们的输出与真实值之间的差别
    errorMat[np.mat(predictArr) == np.mat(dataArr)[:,-1].T] =0
    weightsError = np.multiply(errorMat,weights.T)
    #向量求和
    Em = weightsError.sum()
    return Em
#计算所有
def calEm_all(dataArr,predictArr):
    m = np.shape(dataArr)[0]
    errorMat = np.mat(np.ones(m))
    errorMat[np.mat(predictArr) == np.mat(dataArr)[:, -1].T] = 0
    Em = errorMat.sum()/m
    return Em
#计算alpha权重
def calAlpha(Em):
    alpha = 0.5*np.log((1-Em)/max(Em,1e-16))
    return alpha
#更新分布
def calWeights(dataArr,predictArr,weights,alpha):
    m = np.shape(dataArr)[0]
    errorMat = np.mat(np.ones(m))
    #预测对的数据值为1，预测错的数据值为-1
    errorMat[np.mat(predictArr) != np.mat(dataArr)[:, -1].T] = -1
    weightsError = np.multiply(weights,np.exp(-alpha*errorMat))
    Zm = weightsError.sum()
    newWeights = weightsError/Zm
    return np.array(newWeights)[0]
def trainBoostingTree(dataArr,weights,num):
    #训练
    TreeList = []
    alphaList =[]
    predictList = np.array(np.zeros(np.shape(dataArr)[0]))
    #num个分类器
    for i in range(num):
        #选最好的基尼系数
        axis,value = chooseBestGini(dataArr,weights)
        #对应的决策树
        Tree,predictArr = createTree(dataArr,weights,axis,value)
        #加入ada集成中
        TreeList.append(Tree)
        #计算错误率
        Em = calEm(dataArr,predictArr,weights)
        #计算当前的alpha
        alpha = calAlpha(Em)
        #记录alpha
        alphaList.append(alpha)
        #更新权重分布
        weights = calWeights(dataArr,predictArr,weights,alpha)
        #预测列表
        predictList += np.array(predictArr)*alpha
        predict_all = np.sign(predictList)
        #计算错误率
        Em_all = calEm_all(dataArr,predict_all)
    return TreeList,alphaList,predictList
#参数设置
params = {}
params["Train_path"] = "horseColicTraining2.txt"
params["Test_path"] = "horseColicTest2.txt"
#数据传入
def loadDataSet(path):
    Dataset = []
    with open(path) as p:
        data = []
        for line in p:
            data = line.strip().split('\t')
            eachdata = []
            for dat in data:
                eachdata.append(float(dat))
            Dataset.append(eachdata)
    return Dataset
#测试
def test(dataset,TreeList,alphaList):
    err = 0
    m = np.shape(dataset)[0]
    #进行测试
    for data in dataset:
        pre = 0
        #预测当前决策树对应的类别
        for index,tree in enumerate(TreeList):
            axis, val = list(tree.keys())[0][0],list(tree.keys())[0][1]
            #集成加和
            if data[axis] == val:
                pre += alphaList[index]*list(tree.values())[0][0]
        #计算错误率
        if np.sign(pre) != data[-1]:
            err += 1
    return err/m
#数据导入
Train_dataset = loadDataSet(params["Train_path"])
Test_dataset = loadDataSet(params["Test_path"])
#参数设置
m = np.shape(Train_dataset)[0]
#分布
weights = np.ones((m,1))/m
#错误率
err = []
#测试
for i in range(0,20):
    TreeList,alphaList,predictList = trainBoostingTree(Train_dataset,weights,i)
    err.append(test(Test_dataset,TreeList,alphaList))
plt.plot(err)