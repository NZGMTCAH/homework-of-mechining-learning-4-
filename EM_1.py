# 1. 任选两个以上一维高斯分布，以及权重系数，组成高斯混合模型，采样生成100个样本点，作为数据集。
# coding=utf-8
from __future__ import print_function
import numpy as np


def generateData(k,mu,sigma,dataNum):
    '''
    产生混合高斯模型的数据
    :param k: 比例系数
    :param mu: 均值
    :param sigma: 标准差
    :param dataNum:数据个数
    :return: 生成的数据
    '''
    # 初始化数据
    dataArray = np.zeros(dataNum,dtype=np.float32)
    # 逐个依据概率产生数据
    # 高斯分布个数
    n = len(k)
    for i in range(dataNum):
        rand = np.random.random() # 产生[0,1]之间的随机数
        Sum = 0
        index = 0
        while(index < n):
            Sum += k[index]
            if(rand < Sum):
                dataArray[i] = np.random.normal(mu[index],sigma[index])  #返回随机数
                break
            else:
                index += 1
    return dataArray  #返回采样生成100个样本点

def normPdf(x,mu,sigma):
    '''
    计算均值为mu，标准差为sigma的正态分布函数的密度函数值
    :param x: x值
    :param mu: 均值
    :param sigma: 标准差
    :return: x处的密度函数值
    '''
    return (1./np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*sigma**2))) #正态分布概率密度函数



def em(dataArray,k,mu,sigma,step = 10):
    '''
    em算法估计高斯混合模型
    :param dataNum: 已知数据个数
    :param k: 每个高斯分布的估计系数
    :param mu: 每个高斯分布的估计均值
    :param sigma: 每个高斯分布的估计标准差
    :param step:迭代次数
    :return: em 估计迭代结束估计的参数值[k,mu,sigma]
    '''
    # 高斯分布个数
    n = len(k)
    # 数据个数
    dataNum = dataArray.size
    # 初始化gama数组
    gamaArray = np.zeros((n,dataNum))
    #开始迭代
    for s in range(step): #迭代次数
        for i in range(n):  #高斯分布个数
            for j in range(dataNum): #数据个数
                #计算联合分布的条件概率期望
                Sum = sum([k[t]*normPdf(dataArray[j],mu[t],sigma[t]) for t in range(n)])
                gamaArray[i][j] = k[i]*normPdf(dataArray[j],mu[i],sigma[i])/float(Sum)
        # 更新 mu，模型参数，高斯函数的均值
        for i in range(n):
            mu[i] = np.sum(gamaArray[i]*dataArray)/np.sum(gamaArray[i])
        # 更新 sigma，模型参数，高斯函数的标准差
        for i in range(n):
            sigma[i] = np.sqrt(np.sum(gamaArray[i]*(dataArray - mu[i])**2)/np.sum(gamaArray[i]))
        # 更新系数k，模型参数，高斯函数的系数，决定高斯函数的高度，维度（K）
        for i in range(n):
            k[i] = np.sum(gamaArray[i])/dataNum

    return [k,mu,sigma] #返回更新参数





if __name__ == '__main__':
    # 参数的准确值
    k = [0.3,0.4,0.3]
    mu = [2,4,3]
    sigma = [1,1,4]
    # 样本数
    dataNum = 100
    # 产生数据
    dataArray = generateData(k,mu,sigma,dataNum)
    # 参数的初始值
    # 注意em算法对于参数的初始值是十分敏感的
    k0 = [0.3,0.3,0.4]
    mu0 = [2,2,2]
    sigma0 = [1,1,1]
    step = 10
    # 使用em算法估计参数
    k1,mu1,sigma1 = em(dataArray,k0,mu0,sigma0,step)
    # 输出参数的值
    print("参数实际值:")
    print("k:",k)
    print("mu:",mu)
    print("sigma:",sigma)
    print("参数估计值:")
    print("k1:",k1)
    print("mu1:",mu1)
    print("sigma1:",sigma1)