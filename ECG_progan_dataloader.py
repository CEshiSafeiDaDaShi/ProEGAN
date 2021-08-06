# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:39:05 2020

N 出现的次数： 74546
L 出现的次数： 8075
R 出现的次数： 7259
A 出现的次数： 6903
V 出现的次数： 2546

@author: intel
"""

import torch
import torchvision
from PIL import Image, ImageOps
import os, random
import numpy as np

class ECG_Dataset(torch.utils.data.Dataset):
    """
    A datset which loads images from a flat directory and scales them
    to a size specified when the dataloader is created. We create a new 
    loader every time we change resolutions in training.
    source_directory：原始图像储存的地址
    resize_directory：为不同尺度图像储存的地址
    resolution：分辨率
    """
    def __init__(self, source_directory, resolution,category):
        # 这个地方首先不能使用totensor,其次这个是对图像做处理的
        # 正则化，这个均值和标准差需要额外写一个东西，暂时拿0.5替代
#        self.transform = torchvision.transforms.Compose([
##            torchvision.transforms.ToTensor(),
#            torchvision.transforms.Normalize(mean=ECG_mean, std=ECG_std)
#        ])
    
        #定义ECG的路径，给定频率下某一类别的心电数据的路径
        self.directory = os.path.join(source_directory, f'ECG_{resolution}Hz',f'{category}')
        #ECG数据读取，读取给定频率下某一类别的所有心电数据
        self.ECG_Data = np.load(self.directory)

    def __len__(self):
        return len(self.ECG_Data)

    def __getitem__(self, index):
        #给一个index,从ECG数据list中读取一个数据，并且transform为tensor。
        ECG = self.ECG_Data[index]
        return ECG
#        return self.transform(ECG)

#求均值和标准差，做标准化。
def ECG_Std(ECG_path_Hz,category):
    """
    求取均值，标准差，标准化。
    ECG_path_Hz = 'work/ECG_324Hz/'
    category = 'A.npy'
    """
    sta_data = np.load(ECG_path_Hz + category)
    mean = sta_data.mean(axis=0)
    std = sta_data.std(axis=0)
    return mean,std

