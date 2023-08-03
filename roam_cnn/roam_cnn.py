from network.cnn import CNN
from utils.dataprocess import DataProcess
import utils.tool_func as tools

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

print('collecting data...')
small_data = False
feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
data_process = DataProcess(__features__=feature_dir, __labels__=label_fileName)

print('preprocessing...')
data_process.split_data(0.8, 0.1, 0.1)
data_process.preprocess(need=frozenset(['tensor', 'onehot']))
dummies_columns = data_process.dummies_columns
train_features, train_labels = data_process.train_data
test_features, test_labels = data_process.test_data
valid_features, valid_labels = data_process.valid_data

"""带softmax的感知机"""
print('constructing network...')
activation = 'LeakyReLU'
para_init = 'xavier'
# 获取两端口输出维度
fir_out = len([d for d in dummies_columns if tools.label_names[0] in d])
sec_out = len([d for d in dummies_columns if tools.label_names[1] in d])
net = CNN(
    # train_features.shape[1:],
    (fir_out, sec_out), dummies_columns,
    activation=activation, para_init=para_init, dropout=0.2,
    in_channels=[1, 16, 16, 32, 32, 64, 64, 128],
    out_channels=[16, 16, 32, 32, 64, 64, 128, 128],
    kernel_sizes=[(3, 3) for _ in range(8)],
    strides=[1, 2, 2, 2, 2, 2, 2, 2],
    # bn_shapes=[(127, 127), (64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2)],
    bn_shapes=[128, 64, 32, 16, 8, 4, 2],
    momentums=[0.95 for _ in range(7)],
    paddings=[1 for _ in range(8)]
)
print(net)

print('training...')
num_epochs = 5
# batch_size = int(train_features.shape[0]*0.3)
batch_size = 8
weight_decay = 0
learning_rate = 0.001
momentum = 0e0
loss = 'crossEntro'
optimizer = 'Adam'
train_ls, valid_ls = net.train_(
    train_features, train_labels, valid_features, valid_labels,
    num_epochs=num_epochs, batch_size=batch_size, weight_decay=weight_decay,
    learning_rate=learning_rate, loss=loss
)

print('plotting...')
data_process.plot_data(
    range(num_epochs), train_ls, xlabel='num_epochs', ylabel=f'train_loss({loss})',
    title=f'optimizer{optimizer}',
    # savefig_as=f'loss_plot/num_epochs{num_epochs}batch_size{batch_size}'
    #            f'weight_decay{weight_decay}learning_rate{learning_rate}'
    #            f'momentum{momentum}.jpg'
)
if valid_ls:
    data_process.plot_data(
        range(num_epochs), valid_ls, xlabel='num_epochs', ylabel=f'valid_loss({loss})'
    )

print('testing...')
preds = net.predict(test_features)
acc, compare = data_process.accuracy(preds, test_labels)
print('预测结果为：', compare)
print('准确率为：', acc)
