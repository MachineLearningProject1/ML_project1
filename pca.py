# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:42:42 2018

@author: J
"""
#可以用 np.clip 把数据的范围定在【-1000， 1000】之间 这样可以去除一些很大的值（相当于去除一些噪声）
#降低噪声带来的影响，感觉可以考虑一下 或是做完了试试这样能不能优化。然后再把数据归一化
import numpy as np
def pca(dataMat,topNfeat=9999999):
    #求数据集的平均值，用数据减去其平均值
    meanVals =np.mean(dataMat,axis=0)
    meanRemoved =dataMat - meanVals
    #计算协方差矩阵
    covMat =np.cov(meanRemoved,rowvar =0)
    #得到协方差矩阵的特征值和特征向量
    eigvals,eigVects =np.linalg.eig(np.mat(covMat))
    #对特征值进行排序
    eigValInd =np.argsort(eigvals)
    #得到前N个特征值和对应的特征向量
    eigValInd =eigValInd[:-(topNfeat+1):-1]
    redEigVects =eigVects[:,eigValInd]
    #利用特征向量对去掉平均值的数据进行降维
    lowDDataMat =meanRemoved * redEigVects
    #得到最后的输出低维矩阵
    reconMat =(lowDDataMat * redEigVects.T) +meanVals
    #return lowDDataMat,reconMat
    return lowDDataMat
