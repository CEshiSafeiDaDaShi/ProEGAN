#!/usr/bin/env python
# coding: utf-8

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
# 预处理：安心拍顺序排序
#预声明，这里的Hz不代表真正的赫兹，324Hz代表每个心拍324个像素点
import numpy as np

data_path = 'E:/MIT-BIH-Arrhythmia'
# 读取保存的数据
npzfile = np.load(data_path + '/single_beat.npz',allow_pickle=True)

# 按照保存时设定组数key进行访问
beat=npzfile['beat'].tolist()
beatlabel=npzfile['beatlabel'].tolist()
print(np.shape(beat))
print(np.shape(beatlabel))


# 在数据的右侧水平方向上合并标签
beat_label = []
beat = np.array(beat)
beatlabel = np.array(beatlabel)

beat_label = np.stack((beat,beatlabel),axis=1)
print(np.shape(beat_label))
print(type(beat_label))


# In[5]:


#统计疾病类别,把打乱的类型和标签规整
labels = beat_label[:,1]  #取标签第一列
typelist = []
for i in range(5):                   
    typelist.append([])                #建5个空的标签列表 0/1/2/3/4
    
for i,j in enumerate(labels):          #enumerate返回索引i和元素j
        if j == 'N':
            typelist[0].append(beat_label[i,0])
        if j == 'L':
            typelist[1].append(beat_label[i,0])
        if j == 'R':
            typelist[2].append(beat_label[i,0])
        if j == 'V':
            typelist[3].append(beat_label[i,0])
        if j == 'A':
            typelist[4].append(beat_label[i,0])


# In[8]:


#检查每组信号数量
S = ['N','L','R','V','A']
for i,j in enumerate(S):
	num = len(typelist[i])
	print(j,'出现的次数：',num)

#查看各组是否有其他组新号
typelist_N = typelist[0]
typelist_L = typelist[1]
typelist_R = typelist[2]
typelist_V = typelist[3]
typelist_A = typelist[4]

for i,j in enumerate(typelist_N):          #enumerate返回索引i和元素j
        if j == 'LRVA':
            print('typelist_N wrong')
        if j == 'NRVA':
            print('typelist_L wrong')
        if j == 'NLVA':
            print('typelist_R wrong')
        if j == 'NLRA':
            print('typelist_V wrong')
        if j == 'NLRV':
            print('typelist_A wrong')
print('Done!')
print(np.shape(typelist))
print(np.shape(typelist_N))
print(np.shape(typelist_L))
print(np.shape(typelist_R))
print(np.shape(typelist_V))
print(np.shape(typelist_A))


# In[9]:


#检查各组数据是否有不等于324Hz的数据，若不等于则删掉。

# typelist_N = typelist[0]
# typelist_L = typelist[1]
# typelist_R = typelist[2]
# typelist_V = typelist[3]
# typelist_A = typelist[4]


print('N中原始数据信息',np.shape(typelist_N))
for i,j in enumerate(typelist_N.copy()):          #enumerate返回索引i和元素j
        if np.shape(j) != (256,):
            print(np.shape(j))
            typelist_N.remove(j)
print('N中数据筛选后信息',np.shape(typelist_N))  

print('L中原始数据信息',np.shape(typelist_L))
for i,j in enumerate(typelist_L.copy()):          #enumerate返回索引i和元素j
        if np.shape(j) != (256,):
            print(np.shape(j))
            typelist_L.remove(j)
print('L中数据筛选后信息',np.shape(typelist_L))  

print('R中原始数据信息',np.shape(typelist_R))
for i,j in enumerate(typelist_R.copy()):          #enumerate返回索引i和元素j
        if np.shape(j) != (256,):
            print(np.shape(j))
            typelist_R.remove(j)
print('R中数据筛选后信息',np.shape(typelist_R))  


print('V中原始数据信息',np.shape(typelist_V))
for i,j in enumerate(typelist_V.copy()):          #enumerate返回索引i和元素j
        if np.shape(j) != (256,):
            print(np.shape(j))
            typelist_V.remove(j)
print('V中数据筛选后信息',np.shape(typelist_V))



print('A中原始数据信息',np.shape(typelist_A))
for i,j in enumerate(typelist_V.copy()):          #enumerate返回索引i和元素j
        if np.shape(j) != (256,):
            print(np.shape(j))
            typelist_A.remove(j)
print('A中数据筛选后信息',np.shape(typelist_A))


# In[44]:


#将324/162/81/40Hz数据分别按类别保存
ECG_324Hz_path='F:/Data/MIT-BIH-Arrhythmia/ECG_256Hz'
ECG_162Hz_path='F:/Data/MIT-BIH-Arrhythmia/ECG_128Hz'
ECG_81Hz_path='F:/Data/MIT-BIH-Arrhythmia/ECG_64Hz'
ECG_41Hz_path='F:/Data/MIT-BIH-Arrhythmia/ECG_32Hz'

# #重新保存筛选后的324Hz数据
np.save(ECG_324Hz_path + '/N.npy',typelist_N)
np.save(ECG_324Hz_path + '/L.npy',typelist_L)
np.save(ECG_324Hz_path + '/R.npy',typelist_R)
np.save(ECG_324Hz_path + '/V.npy',typelist_V)
np.save(ECG_324Hz_path + '/A.npy',typelist_A)

#############################################################################
#这里注意，不能直接对list的列进行操作，必须转换为array数组后才能进行！
#324数据转换162数据
##############################################################################
print('原始N数据',np.shape(typelist_N))
typearray_N = np.array(typelist_N)
typelist_N_162 = typearray_N[:,::2]  #每个ECG数据，间隔取值
print('降采样后N数据',np.shape(typelist_N_162))

print('原始R数据',np.shape(typelist_R))
typearray_R = np.array(typelist_R)
typelist_R_162 = typearray_R[:,::2]  #每个ECG数据，间隔取值
print('降采样后R数据',np.shape(typelist_R_162))

print('原始L数据',np.shape(typelist_L))
typearray_L = np.array(typelist_L)
typelist_L_162 = typearray_L[:,::2]  #每个ECG数据，间隔取值
print('降采样后A数据',np.shape(typelist_L_162))

print('原始V数据',np.shape(typelist_V))
typearray_V = np.array(typelist_V)
typelist_V_162 = typearray_V[:,::2]  #每个ECG数据，间隔取值
print('降采样后V数据',np.shape(typelist_V_162))

print('原始A数据',np.shape(typelist_A))
typearray_A = np.array(typelist_A)
typelist_A_162 = typearray_A[:,::2]  #每个ECG数据，间隔取值
print('降采样后A数据',np.shape(typelist_A_162))

 # #重新保存筛选后的162Hz数据
np.save(ECG_162Hz_path + '/N.npy',typelist_N_162)
np.save(ECG_162Hz_path + '/L.npy',typelist_L_162)
np.save(ECG_162Hz_path + '/R.npy',typelist_R_162)
np.save(ECG_162Hz_path + '/V.npy',typelist_V_162)
np.save(ECG_162Hz_path + '/A.npy',typelist_A_162)


#############################################################################
#162数据转换81数据
##############################################################################
print('原始N数据',np.shape(typelist_N))
typearray_N = np.array(typelist_N)
typelist_N_81 = typearray_N[:,::4]  #每个ECG数据，间隔取值
print('降采样后N数据',np.shape(typelist_N_81))
print('原始R数据',np.shape(typelist_R))
typearray_R = np.array(typelist_R)
typelist_R_81 = typearray_R[:,::4]  #每个ECG数据，间隔取值
print('降采样后R数据',np.shape(typelist_R_81))
print('原始L数据',np.shape(typelist_L))
typearray_L = np.array(typelist_L)
typelist_L_81 = typearray_L[:,::4]  #每个ECG数据，间隔取值
print('降采样后A数据',np.shape(typelist_L_81))
print('原始V数据',np.shape(typelist_V))
typearray_V = np.array(typelist_V)
typelist_V_81 = typearray_V[:,::4]  #每个ECG数据，间隔取值
print('降采样后V数据',np.shape(typelist_V_81))
print('原始A数据',np.shape(typelist_A))
typearray_A = np.array(typelist_A)
typelist_A_81 = typearray_A[:,::4]  #每个ECG数据，间隔取值
print('降采样后A数据',np.shape(typelist_A_81))

  #重新保存筛选后的162Hz数据
np.save(ECG_81Hz_path + '/N.npy',typelist_N_81)
np.save(ECG_81Hz_path + '/L.npy',typelist_L_81)
np.save(ECG_81Hz_path + '/R.npy',typelist_R_81)
np.save(ECG_81Hz_path + '/V.npy',typelist_V_81)
np.save(ECG_81Hz_path + '/A.npy',typelist_A_81)

#############################################################################
#81数据转换41数据
##############################################################################
print('原始N数据',np.shape(typelist_N))
typearray_N = np.array(typelist_N)
typelist_N_41 = typearray_N[:,::8]  #每个ECG数据，间隔取值
print('降采样后N数据',np.shape(typelist_N_41))

print('原始R数据',np.shape(typelist_R))
typearray_R = np.array(typelist_R)
typelist_R_41 = typearray_R[:,::8]  #每个ECG数据，间隔取值
print('降采样后R数据',np.shape(typelist_R_41))

print('原始L数据',np.shape(typelist_L))
typearray_L = np.array(typelist_L)
typelist_L_41 = typearray_L[:,::8]  #每个ECG数据，间隔取值
print('降采样后A数据',np.shape(typelist_L_41))

print('原始V数据',np.shape(typelist_V))
typearray_V = np.array(typelist_V)
typelist_V_41 = typearray_V[:,::8]  #每个ECG数据，间隔取值
print('降采样后V数据',np.shape(typelist_V_41))

print('原始A数据',np.shape(typelist_A))
typearray_A = np.array(typelist_A)
typelist_A_41 = typearray_A[:,::8]  #每个ECG数据，间隔取值
print('降采样后A数据',np.shape(typelist_A_41))

# #重新保存筛选后的162Hz数据
np.save(ECG_41Hz_path + '/N.npy',typelist_N_41)
np.save(ECG_41Hz_path + '/L.npy',typelist_L_41)
np.save(ECG_41Hz_path + '/R.npy',typelist_R_41)
np.save(ECG_41Hz_path + '/V.npy',typelist_V_41)
np.save(ECG_41Hz_path + '/A.npy',typelist_A_41)


# In[4]:


#import numpy as np
#
#ECG_324Hz_path='work/ECG_324Hz/'
#ECG_162Hz_path='work/ECG_162Hz/'
#ECG_81Hz_path='work/ECG_81Hz/'
#ECG_41Hz_path='work/ECG_41Hz/'
#
#N = 'N.npy'
#L = 'L.npy'
#R = 'R.npy'
#V = 'V.npy'
#A = 'A.npy'
#
##求均值和标准差，做标准化。
#def ECG_Std(ECG_path_Hz,category):
#    """
#    求取均值，标准差，标准化。
#    ECG_path_Hz = 'work/ECG_324Hz/'
#    category = 'A.npy'
#    """
#    sta_data = np.load(ECG_path_Hz + category)
#    mean = sta_data.mean(axis=0)
#    std = sta_data.std(axis=0)
#    return mean,std
#
#
## 对每一类求标准差和均值
#mean,std = ECG_Std(ECG_324Hz_path,A)
#print(np.shape(mean))
#print(np.shape(std))


# In[13]:


#import os
##读取器，拼接路径检查
#source_directory = 'work/ECG_324Hz/'
#resize_directory = 'work'
#resolution = 324
#category = A
#
##路径拼接
#directory = os.path.join(resize_directory, f'ECG_{resolution}Hz',f'{category}')
#
#sta_data = np.load(directory)
#print(np.shape(sta_data))
#print(directory)



