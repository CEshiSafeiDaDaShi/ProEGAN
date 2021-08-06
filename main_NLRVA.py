# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:51:06 2019

RLVA 四分类最佳参数C:16.75 gamma1.25 准确率：0.9719

N : 78879
L : 7988
R : 7223
V : 6282
A : 2168

@author: Yumeng
"""
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pywt 
import os
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV


from read_ecg import read_ecg 
from plot_ecg import plot_ecg 
from extract_data import extract_data_from_train_file,extract_data_from_test_file
from extract_feature import simple_f,wavelets_f


# read all the training data and put them in one array
x_total=[]
y_total=[]
x_total_G=[]
y_total_G=[]

base_path = 'D:/Date/MIT-BIH-Arrhythmia/ECG_256Hz'
base_path_G2000 = 'D:/Date/MIT-BIH-Arrhythmia/ProGAN_generated/num2000'
#base_path_G2000 = 'D:/Date/MIT-BIH-Arrhythmia/ProGAN_generated_nostd/num2000'
#base_path_G2000 = 'F:/Data/MIT-BIH-Arrhythmia/DGAN_generated/num2000'
#base_path_G2000 = 'F:/Data/MIT-BIH-Arrhythmia/DCGAN_generated/num2000'


N = 'N.npy'
L = 'L.npy'
R = 'R.npy'
V = 'V.npy'
A = 'A.npy'

#需要对比的类别,category = [N,L,R,V,A]
category = [L,R,V,A]


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

x_total=np.array(x_total)
y_total=np.array(y_total)
x_total_G=np.array(x_total_G)
y_total_G=np.array(y_total_G)

#使用进行4层小波变换，使用db6作为小波基，用第四次层近似系数形成一个特征向量
trainingdata2 = wavelets_f(x_total)   #小波变换
trainingdata2_G = wavelets_f(x_total_G)   #小波变换

#x_feature = np.hstack((trainingdata1,trainingdata2))
x_feature = trainingdata2
x_feature_G = trainingdata2_G

#打乱顺序
x_feature,y_total=shuffle(x_feature,y_total)
x_feature_G,y_total_G=shuffle(x_feature_G,y_total_G)

###############################################
#             特征选择
###############################################

#选择训练和测试的特征
#idx=int(len(x_feature)*0.7)
#x_train=x_feature[:idx]
#x_valid=x_feature[idx:]
#y_train=y_total[:idx]
#y_valid=y_total[idx:]

#GAN_Train 使用生成数据进行训练，真实数据进行测试，验证多样性
#print('GAN_Train')
#idx=int(len(x_feature))
#x_train=x_feature_G
#x_valid=x_feature
#y_train=y_total_G
#y_valid=y_total

idx=int(len(x_feature)*0.3)
x_train=x_feature_G
x_valid=x_feature[:idx]
y_train=y_total_G
y_valid=y_total[:idx]


###GAN_Test 选择训练和测试的特征
#print('GAN_Test')
#idx=int(len(x_feature))
#x_train=x_feature
#x_valid=x_feature_G
#y_train=y_total
#y_valid=y_total_G

#idx=int(len(x_feature)*0.7)
#x_train=x_feature[:idx]
#x_valid=x_feature_G
#y_train=y_total[:idx]
#y_valid=y_total_G

#RLVA 四分类最佳参数C:16.75 gamma1.25 准确率：
clf_rbf = SVC(kernel='rbf',C = 16.75, gamma = 1.25,class_weight='balanced')

#参数搜索
#parameters = {'kernel':('linear', 'rbf'), 'C':[9, 11, 13, 15, 17, 19], 'gamma':[ 1, 2.5, 5, 10, ]}
###parameters = {'kernel':('linear', 'rbf'), 'C':[16.75], 'gamma':[1.15, 1.25, 1.5]}
##
#clf = GridSearchCV(clf_rbf, parameters, scoring='accuracy')
#clf.fit(x_train, y_train)
#print('The parameters of the best model are: ')
#print(clf.best_params_)


##函数用交叉检验(cross-validation)来估计一个模型的得分
scores_rbf = cross_val_score(clf_rbf, x_feature, y_total, cv=10, scoring='accuracy')
print('SVM得分：',scores_rbf.mean())



# machine learning model(SVM)
# 用训练数据拟合分类器模型
clf_rbf.fit(x_train,y_train)
#输入测试集 x_valid 进行预测
pred_valid=clf_rbf.predict(x_valid)
print (classification_report(y_valid, pred_valid,digits=4))






'''

#3. The assigned 'V' beat info shall be exported to WFDB format (*.test),
# and sent back to Biofourmis.
for index in range(1,3):
    path="C:\\E\\Jobs\\Biofourmis\\ECG\\database\\test\\b"+str(index)
    x_test ,location = extract_data_from_test_file(path)
    n = len(x_test)
    x_test = np.array(x_test)
    testingdata1 = simple_f(x_test,n)
    testingdata2 = wavelets_f(x_test)
    x_feature_test = np.hstack((testingdata1,testingdata2))
    predicted_labels=clf_rbf.predict(x_feature_test)
    ecg_sig, ecg_type, ecg_peak = read_ecg(path)
    for i in range(len(location)):
        if predicted_labels[i]==1:
            ecg_type[location[i]]="V"
    
    name="b"+str(index)
    wfdb.wrann(name, 'test', ecg_peak, ecg_type, write_dir='C:\\E\\Jobs\\Biofourmis\\ECG\\database\\test\\')


'''





