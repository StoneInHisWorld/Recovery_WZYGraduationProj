from matplotlib import pyplot as plt

from network.softmax_mlp import SoftmaxMLP
from utils.dataprocess import DataProcess

print('collecting data...')
small_data = True
feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
data_process = DataProcess(__features__=feature_dir, __labels__=label_fileName)


print('preprocessing...')
data_process.split_data(0.8, 0.1, 0.1)
data_process.preprocess(need=frozenset(['tensor', 'flatten', 'norm', 'onehot']))
dummies_columns = data_process.dummies_columns
train_features, train_labels = data_process.train_data
test_features, test_labels = data_process.test_data
valid_features, valid_labels = data_process.valid_data


"""带softmax的感知机"""
print('constructing network...')
net = SoftmaxMLP(
    train_features.shape[1], train_labels.shape[1], dummies_columns,
    activation='ReLU', base=10, para_init='xavier'
)
print(net)


print('training...')
num_epochs = 10
# batch_size = int(train_features.shape[0]*0.3)
batch_size = 32
weight_decay = 0.
learning_rate = 0.1
momentum = 0.
loss = 'mse'
optimizer = 'SGD'
train_ls, valid_ls = net.train_(
    train_features, train_labels, valid_features, valid_labels,
    num_epochs=num_epochs, batch_size=batch_size, weight_decay=weight_decay,
    learning_rate=learning_rate, loss= loss
)


print('plotting...')
# plt.plot(range(num_epochs), train_ls)
# plt.xlabel('num_epochs')
# plt.ylabel(f'train_loss({loss})')
# plt.title(f'optimizer{optimizer}')
# plt.savefig(f'loss_plot/num_epochs{num_epochs}batch_size{batch_size}'
#             f'weight_decay{weight_decay}learning_rate{learning_rate}'
#             f'momentum{momentum}.jpg')
# plt.show()
# if valid_ls:
#     plt.plot(range(num_epochs), valid_ls)
#     plt.xlabel('num_epochs')
#     plt.ylabel(f'valid_loss({loss})')
#     plt.show()


print('testing...')
preds = net.predict(test_features)
acc = data_process.accuracy(preds, test_labels)
print('预测结果为：', preds)
print('准确率为：', acc)
