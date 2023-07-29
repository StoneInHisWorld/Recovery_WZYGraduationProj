import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.data_process import data_process
from classifier.multiRegressor import MultiRegressor

print('collecting data...')
small_data = True
feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
walkGen = os.walk(feature_dir)
feature_data = []
# 真的发现了文件名对应不上！
for _, __, file_names in walkGen:
    file_names = sorted(file_names, key=lambda name: int(name.split(".")[0]))  # 给文件名排序！
    featureImg_array = map(
        np.array, [plt.imread(feature_dir + '/' + file_name)
                   for file_name in file_names]
    )
    featureImgs = [featureImg for featureImg in featureImg_array]
    feature_data += featureImgs
    del featureImgs, file_names, featureImg_array

label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
label_colNames = ['OAM1', 'OAM2']
label_data = pd.read_csv(label_fileName, names=label_colNames).values

print('preprocessing...')
data_process = data_process(
    features=np.array(feature_data), labels=np.array(label_data)
)

"""机器学习"""
data_process.split_data(0.8, 0.2)
data_process.preprocess(mode='linear', need_tensor=False, need_norm=False)
train_features, train_labels = data_process.train_data
test_features, test_labels = data_process.test_data


print('fitting...')
losses = ['squared_error', 'absolute_error', 'huber', 'quantile']
max_iter = 100
learning_rate = 0.1
alpha = 0.001
validation_fraction = 0.1
random_state=42
# 自带验证集
clf = MultiRegressor(
    estimator='HGBR', max_iter=max_iter, learning_rate=learning_rate,
    alpha=alpha, validation_fraction=validation_fraction
)
print(clf.train_(train_features, train_labels)['estimator'])
y_hat, l = clf.predict_(test_features, test_labels)
print('loss:', l)
print('accuracy:', data_process.accuracy(y_hat, test_labels))
del clf
clf = MultiRegressor(
    estimator='R', max_iter=max_iter, learning_rate=learning_rate,
    alpha=alpha, validation_fraction=validation_fraction
)
print(clf.train_(train_features, train_labels)['estimator'])
y_hat, l = clf.predict_(test_features, test_labels)
print('loss:', l)
print('accuracy:', data_process.accuracy(y_hat, test_labels))
del clf
clf = MultiRegressor(
    estimator='L', max_iter=max_iter, learning_rate=learning_rate,
    alpha=alpha, validation_fraction=validation_fraction
)
print(clf.train_(train_features, train_labels)['estimator'])
y_hat, l = clf.predict_(test_features, test_labels)
print('loss:', l)
print('accuracy:', data_process.accuracy(y_hat, test_labels))
del clf
clf = MultiRegressor(
    estimator='SGD', max_iter=max_iter, learning_rate=learning_rate,
    alpha=alpha, validation_fraction=validation_fraction
)
print(clf.train_(train_features, train_labels)['estimator'])
y_hat, l = clf.predict_(test_features, test_labels)
print('loss:', l)
print('accuracy:', data_process.accuracy(y_hat, test_labels))
