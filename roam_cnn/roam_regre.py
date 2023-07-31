import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.dataprocess import DataProcess
from classifier.multiRegressor import MultiRegressor

print('collecting data...')
# small_data = True
# feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
# walkGen = os.walk(feature_dir)
# feature_data = []
# # 真的发现了文件名对应不上！
# for _, __, file_names in walkGen:
#     file_names = sorted(file_names, key=lambda name: int(name.split(".")[0]))  # 给文件名排序！
#     featureImg_array = map(
#         np.array, [plt.imread(feature_dir + '/' + file_name)
#                    for file_name in file_names]
#     )
#     featureImgs = [featureImg for featureImg in featureImg_array]
#     feature_data += featureImgs
#     del featureImgs, file_names, featureImg_array
#
# label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
# label_colNames = ['OAM1', 'OAM2']
# label_data = pd.read_csv(label_fileName, names=label_colNames).values
small_data = False
feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
data_process = DataProcess(__features__=feature_dir, __labels__=label_fileName)


print('preprocessing...')
data_process.split_data(0.8, 0.2)
data_process.preprocess(need=frozenset(['flatten', 'norm']))
train_features, train_labels = data_process.train_data
test_features, test_labels = data_process.test_data


print('fitting...')
max_iter = 300
learning_rate = 0.1
alpha = 0.001
validation_fraction = 0.1
random_state=42
# 自带验证集
for estimator in ['HGBR', 'L', 'R', 'SGD']:
    clf = MultiRegressor(
        estimator=estimator, max_iter=max_iter, learning_rate=learning_rate,
        alpha=alpha, validation_fraction=validation_fraction
    )
    print(clf.train_(train_features, train_labels))
    y_hat, l = clf.predict_(test_features, test_labels)
    print('loss:', l)
    print('accuracy:', data_process.accuracy(y_hat, test_labels))
    del clf
# estimator = 'R'
# clf = MultiRegressor(
#     estimator=estimator, max_iter=max_iter, learning_rate=learning_rate,
#     alpha=alpha, validation_fraction=validation_fraction
# )
# print(clf.train_(train_features, train_labels)['estimator'])
# y_hat, l = clf.predict_(test_features, test_labels)
# print('loss:', l)
# print('accuracy:', data_process.accuracy(y_hat, test_labels))
# del clf
# estimator = 'L'
# clf = MultiRegressor(
#     estimator='L', max_iter=max_iter, learning_rate=learning_rate,
#     alpha=alpha, validation_fraction=validation_fraction
# )
# print(clf.train_(train_features, train_labels)['estimator'])
# y_hat, l = clf.predict_(test_features, test_labels)
# print('loss:', l)
# print('accuracy:', data_process.accuracy(y_hat, test_labels))
# del clf
# clf = MultiRegressor(
#     estimator='SGD', max_iter=max_iter, learning_rate=learning_rate,
#     alpha=alpha, validation_fraction=validation_fraction
# )
# print(clf.train_(train_features, train_labels)['estimator'])
# y_hat, l = clf.predict_(test_features, test_labels)
# print('loss:', l)
# print('accuracy:', data_process.accuracy(y_hat, test_labels))
