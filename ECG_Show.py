# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:34:58 2020

'''
N中原始数据信息 (74546,)
N中数据筛选后信息 (74517, 256)

L中原始数据信息 (8075,)
L中数据筛选后信息 (8072, 256)

R中原始数据信息 (7259,)
R中数据筛选后信息 (7255, 256)

V中原始数据信息 (6903,)
V中数据筛选后信息 (6902, 256)

A中原始数据信息 (2546, 256)
A中数据筛选后信息 (2546, 256)
'''
@author: intel
"""
import torch
import torchvision
import numpy
import copy, os, datetime
import matplotlib.pyplot as plt

import ECG_config
import ECG_progan_models
import ECG_progan_dataloader


import os
import numpy as np
import matplotlib.pyplot as plt


N = 'N.npy'
L = 'L.npy'
R = 'R.npy'
V = 'V.npy'
A = 'A.npy'
resolution = 32
category = A

#source_directory ="D:/Date/MIT-BIH-Arrhythmia/" 
source_directory ="D:/Date/MIT-BIH-Arrhythmia/ProGAN_generated_nostd/num2000" 


#定义ECG的路径，给定频率下某一类别的心电数据的路径
#directory = os.path.join(source_directory, f'ECG_{resolution}Hz',f'{category}')
directory = os.path.join(source_directory, f'{category}')

#ECG数据读取，读取给定频率下某一类别的所有心电数据
ECG_Data = np.load(directory)
print(np.shape(ECG_Data))


#ECG_Sample = ECG_Data[57].tolist()


def draw_ECG(iters,ECG):
#    plt.xlabel("Sample points")
#    plt.ylabel("Amplitude(mV)")
    plt.plot(iters,ECG) 
    plt.legend()
    plt.draw()

#draw_ECG(X,ECG_Sample)


#循环画出N那张图
for i in range (0,2000,20):
    ECG_Sample = ECG_Data[i].tolist()
    X = range(len(ECG_Sample))
    plt.figure(i)   #这行注释掉之后，是所有曲线都画在一张图上
    draw_ECG(X,ECG_Sample)
    plt.savefig('C:/Users/intel/Desktop/ECG_初稿/类别多样性图例对比/A/' + f'{i}')
    plt.close()


'''
70879
7908
7223
6282
2168

N中原始数据信息 (74546,)
N中数据筛选后信息 (74517, 256)

L中原始数据信息 (8075,)
L中数据筛选后信息 (8072, 256)

R中原始数据信息 (7259,)
R中数据筛选后信息 (7255, 256)

V中原始数据信息 (6903,)
V中数据筛选后信息 (6902, 256)

A中原始数据信息 (2546, 256)
A中数据筛选后信息 (2546, 256)


'''











