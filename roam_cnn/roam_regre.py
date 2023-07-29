import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Recovery_WZYGraduationProject.utils.data_process import data_process
from Recovery_WZYGraduationProject.classifier.multiRegressor import MultiRegressor

print('collecting data...')
small_data = True
feature_dir = '../vortex/small_feature' if small_data else feature_dir = 'vortex/0823SPECKLE'
walkGen = os.walk(feature_dir)
feature_data = []
for _, __, file_names in walkGen:
    featureImg_array = map(np.array, [plt.imread(feature_dir + '/' + file_name) for file_name in file_names])
    featureImgs = [featureImg for featureImg in featureImg_array]
    feature_data += featureImgs
    del featureImgs, file_names, featureImg_array

label_fileName = '../vortex/small_labels.csv' if small_data else label_fileName = 'vortex/labels.csv'
label_colNames = ['OAM1', 'OAM2']
label_data = pd.read_csv(label_fileName, names=label_colNames).values

print('preprocessing...')
data_process = data_process(
    features=np.array(feature_data), labels=np.array(label_data)
)

"""机器学习"""
data_process.preprocess(mode='linear', need_tensor=False)
train_features, test_features, train_labels, test_labels = data_process.split_data(0.8, 0.2)
losses = ['squared_error', 'absolute_error', 'huber', 'quantile']
# # 自带验证集
# clf = MultiRegressor(estimator='HGBR', max_iter=1000)
# print(clf.train_(train_features, train_labels)['estimator'])
# y_hat, l = clf.predict_(test_features, test_labels)
# print('loss:', l)
# print('accuracy:', data_process.accuracy(y_hat, test_labels))
# del clf
clf = MultiRegressor(estimator='R', max_iter=20000)
print(clf.train_(train_features, train_labels)['estimator'])
y_hat, l = clf.predict_(test_features, test_labels)
print('loss:', l)
print('accuracy:', data_process.accuracy(y_hat, test_labels))
del clf
# clf = MultiRegressor(estimator='R', max_iter=1000)
# print(clf.train_(train_features, train_labels)['estimator'])
# y_hat, l = clf.predict_(test_features, test_labels)
# print('loss:', l)
# print('accuracy:', data_process.accuracy(y_hat, test_labels))
# del clf
# clf = MultiRegressor(estimator='L')
# print(clf.train_(train_features, train_labels)['estimator'])
# y_hat, l = clf.predict_(test_features, test_labels)
# print('loss:', l)
# print('accuracy:', data_process.accuracy(y_hat, test_labels))
# del clf
clf = MultiRegressor(estimator='SGD', max_iter=4000, learning_rate=0.00001, alpha=0.001)
print(clf.train_(train_features, train_labels)['estimator'])
y_hat, l = clf.predict_(test_features, test_labels)
print('loss:', l)
print('accuracy:', data_process.accuracy(y_hat, test_labels))
