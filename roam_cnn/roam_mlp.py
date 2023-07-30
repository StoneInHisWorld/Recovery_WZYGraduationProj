import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from utils.dataprocess import DataProcess
from network.mlp import MultiLayerPerception


print('collecting data...')
# small_data = False
# feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
# walkGen = os.walk(feature_dir)
# feature_data = []
# for _, __, file_names in walkGen:
#     file_names = sorted(file_names, key=lambda name: int(name.split(".")[0]))  # 给文件名排序！
#     featureImg_array = map(np.array, [plt.imread(feature_dir + '/' + file_name) for file_name in file_names])
#     featureImgs = [featureImg for featureImg in featureImg_array]
#     feature_data += featureImgs
#     del featureImgs, file_names, featureImg_array

# label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
# label_colNames = ['OAM1', 'OAM2']
# label_data = pd.read_csv(label_fileName, names=label_colNames).values
small_data = False
feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
data_process = DataProcess(__features__=feature_dir, __labels__=label_fileName)


print('preprocessing...')
# data_process = DataProcess(
#     features=np.array(feature_data), labels=np.array(label_data)
# )
"""神经网络"""
data_process.split_data(0.8, 0.1, 0.1)
data_process.preprocess(mode='linear', need_tensor=True, need_norm=False)
train_features, train_labels = data_process.train_data
test_features, test_labels = data_process.test_data
valid_features, valid_labels = data_process.valid_data
# del feature_data, label_data, data_process

print('constructing network...')
net = MultiLayerPerception(
    train_features.shape[1], train_labels.shape[1],
    activation='Sigmoid', base=8, para_init='zero'
)
print(net)


print('training...')
num_epochs = 100
batch_size = int(train_features.shape[0]*0.3)
weight_decay = 0.
learning_rate = 0.01
momentum = 0.01
loss = 'mse'
optimizer = 'SGD'
train_ls, valid_ls = net.train_(
    train_features, train_labels, valid_features, valid_labels,
    num_epochs=num_epochs, batch_size=batch_size, weight_decay=weight_decay,
    learning_rate=learning_rate, loss=loss
)
plt.plot(range(num_epochs), train_ls)
plt.xlabel('num_epochs')
plt.ylabel(f'train_loss({loss})')
plt.title(f'optimizer{optimizer}')
plt.savefig(f'loss_plot/num_epochs{num_epochs}batch_size{batch_size}'
            f'weight_decay{weight_decay}learning_rate{learning_rate}'
            f'momentum{momentum}.jpg')
plt.show()
if valid_ls:
    plt.plot(range(num_epochs), valid_ls)
    plt.xlabel('num_epochs')
    plt.ylabel(f'valid_loss({loss})')
    plt.show()


print('testing...')
acc, res = net.test(test_features, test_labels)
print('预测结果为：', res)
print('准确率为：', acc)
