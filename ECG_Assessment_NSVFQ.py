# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 10:17:19 2020

@author: Administrator
"""
import numpy as np
import os
from numpy.linalg import norm as np_linalg_norm
from scipy.spatial.distance import euclidean
import scipy.stats
from sklearn import preprocessing
import torch

from fastdtw import fastdtw

# 读取所有数据并把它们放到同一个array
x_total=[]
y_total=[]
x_total_G=[]
y_total_G=[]

base_path = 'F:/Data/MIT-BIH-Arrhythmia/ECG_300Hz_NSVFQ/'
base_path_G2000 = 'F:/Data/MIT-BIH-Arrhythmia/DGAN_generated/NFVSQ/num2000/'

N = 'N.npy'
S = 'S.npy'
V = 'V.npy'
F = 'F.npy'
Q = 'Q.npy'

#需要对比的类别,category = [N,L,R,V,A]
#category = [S,V,F]
category = [S]


def data_original(path, category):
        #定义ECG的路径，给定频率下某一类别的心电数据的路径
        for i,j in enumerate(category):

            directory = os.path.join(path, f'{j}')
            #读取新店后放入总的序列中
            ECG_Data = np.load(directory)
            x_total.extend(ECG_Data)

            ECG_label = np.ones(len(ECG_Data))*i
            y_total.extend(ECG_label)

        return x_total,y_total
    
def data_generated(path, category):
        #定义ECG的路径，给定频率下某一类别的心电数据的路径
        for i,j in enumerate(category):

            directory = os.path.join(path, f'{j}')
            #读取新店后放入总的序列中
            ECG_Data = np.load(directory)
            x_total_G.extend(ECG_Data)

            ECG_label = np.ones(len(ECG_Data))*i
            y_total_G.extend(ECG_label)

        return x_total_G,y_total_G
    
x_total,y_total = data_original(base_path,category)
x_total_G,y_total_G = data_generated(base_path_G2000,category)

x_total = np.array(x_total)
x_total_G = np.array(x_total_G)

original_samples = x_total
generated_samples = x_total_G
generated_samples_mean = generated_samples.mean(axis=0)
template_signal = original_samples.mean(axis=0)
template_signal_C = np.repeat([template_signal],len(original_samples),axis=0) 
template_signal_F = np.repeat([template_signal],len(generated_samples),axis=0) 


###########################
#
# 计算欧氏距离  ED
# 单独一类：ED_CIs: 0.7750  ED_FIs: 0.6443
#  四类： ED_CIs: 4.8003 ED_FIs: 2.5772
#
###########################

ED_CIs= np.sqrt(np.sum(np.square(template_signal_C-original_samples))/len(original_samples))
ED_FIs= np.sqrt(np.sum(np.square(template_signal_F-generated_samples))/len(generated_samples))

print('ED_CIs:',ED_CIs,"\n",'ED_FIs:' ,ED_FIs)



###########################
#
# 计算动态时间规整 DTW
# 单独一类：DTW_CIs: 3.1461 DTW_FIs: 2.6162 
#  四类： DTW_CIs: 7.0928  DTW_FIs: 6.4386
#
###########################

DTW_CIs, path = fastdtw(template_signal_C, original_samples, dist=euclidean)
DTW_FIs, path = fastdtw(template_signal_F, generated_samples, dist=euclidean)
DTW_CIs = DTW_CIs/len(template_signal_C)
DTW_FIs = DTW_FIs/len(template_signal_F)

print('DTW_CIs:',DTW_CIs,"\n",'DTW_FIs:' ,DTW_FIs)





###########################
#
# 皮尔逊相关系数 PCC
# 计算平均生成和平均原始数据的PCC
# 有疑问
#
###########################

def cal_pccs(x, y, n):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

PCC_CIs = cal_pccs(template_signal,template_signal,len(original_samples))
PCC_FIs = cal_pccs(template_signal,generated_samples_mean,len(generated_samples))

print('PCC_CIs:',PCC_CIs,"\n",'PCC_FIs:' ,PCC_FIs)





###########################
#
# KL散度 KLD
# 首先要进行归一化，KLD否则失效
#
###########################

def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)

def autoNorm(data): #传入一个矩阵
    mins = data.min(0) #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0) #返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data)) #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0] #返回 data矩阵的行数
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges,(row,1)) #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData

#先进行正则化
original_samples = autoNorm(original_samples)
generated_samples = autoNorm(generated_samples)
template_signal = original_samples.mean(axis=0)

template_signal_C = np.repeat([template_signal],len(original_samples),axis=0) 
template_signal_F = np.repeat([template_signal],len(generated_samples),axis=0) 


#KL散度变为无穷？？？怎么回事？？？
num_C = 0
count_C = 0
for i in range(len(original_samples)):
    KLD_CIs = KL_divergence(template_signal_C[0], original_samples[i])
    if KLD_CIs != float("inf"):
        num_C = num_C + KLD_CIs
        count_C = count_C + 1
#    else :
#        print(i)

KLD_CIs = num_C/count_C

    
num_F = 0
count_F = 0
for i in range(len(generated_samples)):
    KLD_FIs = KL_divergence(template_signal_F[0], generated_samples[i])
    if KLD_FIs != float("inf"):
        num_F = num_F + KLD_FIs
        count_F = count_F + 1
#    else :
#        print(i)

KLD_FIs = num_F/count_F
    
print('KLD_CIs:',KLD_CIs,"\n",'KLD_FIs:' ,KLD_FIs)




