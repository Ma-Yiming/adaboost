# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:36:49 2021

@author: MaYiming
"""
import numpy as np
import matplotlib.pyplot as plt
import random
#决策树桩adaboosting类
class StumpAdaBoost:
    def __init__(self,dataset,labelset,NumberOfClass):
        #初始化数据，并进行训练
        self.dataset = dataset
        self.labelset = labelset
        self.NumberOfClass = NumberOfClass
        self.classifierArr,self.pre ,self.err= self.AdaBoostTrain(self.NumberOfClass)
        plt.plot(self.err)
    #决策树桩
    def stumpClassify(self,dataset,featureIndex,threshVal,threshInep):
        res = np.ones((np.shape(dataset)[0],1))
        #进行判断，满足的记为-1
        if threshInep == 'lt':
            res[dataset[:,featureIndex] <= threshVal] = -1.0
        else:
            res[dataset[:,featureIndex] >= threshVal] = -1.0
        #这里的res是竖向量
        return res
    #选择当前D分布下的最好的树桩
    def buildStumpFurther(self,dataset,label,D):
        #初始化参数
        m, n = np.shape(dataset)
        numSteps = 10.0
        #用来记录最好的树桩对应的参数
        bestClassifier = {}
        bestClassEst = np.zeros((m,1))
        minError = np.inf
        #循环特征
        for i in range(n):
            #算出可选择的分类点
            columnMin = dataset[:,i].min()
            columnMax = dataset[:,i].max()
            stepsize = (columnMax-columnMin)/numSteps
            #循环分类点
            for j in range(-1,int(stepsize+1)):
                #循环操作
                for threshInep in ['lt','gt']:
                    #计算当前树桩的错误率
                    threshVal = columnMin + j*stepsize
                    pre = self.stumpClassify(dataset, i, threshVal, threshInep)
                    err = np.ones((m,1))
                    #小trick
                    err[pre == label.reshape(-1,1)] = 0
                    #权重错误
                    errweight = D*err
                    #抛弃不好的分类器
                    if errweight.sum() <= 0.5:
                        if errweight.sum() < minError:
                            #更新最好的树桩对应参数
                            minError = errweight.sum()
                            bestClassEst = pre.copy()
                            bestClassifier['dim'] = i
                            bestClassifier['thresh'] = threshVal
                            bestClassifier['ineq'] = threshInep
        return bestClassifier,bestClassEst,minError
    def AdaBoostTrain(self,NumberOfClass):
        #记录所有树桩和对应参数alpha
        classifierArr = []
        dataset = self.dataset
        labelset = self.labelset
        m = np.shape(dataset)[0]
        # #增加波动性
        # rand = []
        # for i in range(m):
        #     rand.append(random.random())
        # rand = np.array(rand)
        D = np.ones((m,1)) 
        D = D/D.sum()
        #预测和错误率
        pre = np.zeros((m,1))
        err = []
        errRate = 100.0
        #循环分类器
        for i in range(NumberOfClass):
            #找出t时刻最好的分类树桩
            bestClassifier,prelabel,minError = self.buildStumpFurther(dataset, labelset, D)
            #算出alpha
            alpha = float(0.5*np.log((1-minError)/max(minError,1e-16)))
            bestClassifier["alpha"] = alpha
            #记录最好的树桩
            classifierArr.append(bestClassifier)
            rand = []
            for i in range(m):
                rand.append(random.random())
            rand = np.array(rand)
            #更新分布
            D = D*np.exp(-1*alpha*(prelabel*labelset.reshape(-1,1)))
            D = D/D.sum()
            #计算预测
            pre += (alpha*prelabel)
            #计算错误率
            intergrationError = (np.array(np.sign(pre) != labelset.reshape(-1,1))).sum()
            errRate = intergrationError/m
            err.append(errRate)
            if errRate == 0.0:
                break
        return classifierArr,pre,err
    def test(self,Test_dataset,Test_labels):
        #测试数据的导入
        classifierArr,pre_T,NumberOfClass = self.classifierArr,self.pre,self.NumberOfClass
        pre = np.zeros((Test_dataset.shape[0],1))
        #测试分类器数量变化带来的不同
        for i in range(NumberOfClass):
            base_class = self.stumpClassify(Test_dataset, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
            pre += (classifierArr[i]['alpha']*base_class)
        #计算错误率
        err = np.zeros((Test_dataset.shape[0],1))
        err[np.sign(pre) != Test_labels.reshape(-1,1)] = 1
        errRate = err.sum()/Test_dataset.shape[0]
        return errRate

