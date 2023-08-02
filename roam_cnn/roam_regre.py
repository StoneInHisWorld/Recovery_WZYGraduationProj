from classifier.multiRegressor import MultiRegressor
from utils.dataprocess import DataProcess

print('collecting data...')
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
num_epochs = 30
learning_rate = 0.01
alpha = 0.
validation_fraction = 0.1
random_state = 42
batch_size = 128
loss = 'squared_error'
# 自带验证集
for estimator in ['L', 'R']:
    clf = MultiRegressor(
        estimator=estimator, max_iter=num_epochs, learning_rate=learning_rate,
        alpha=alpha, validation_fraction=validation_fraction
    )
    train_losses = clf.train_(
        train_features, train_labels, num_epochs=num_epochs, batch_size=batch_size
    )
    data_process.plot_data(
        range(num_epochs), train_losses, f'estimator{estimator}',
        xlabel='epoch', ylabel=f'loss({loss})'
    )
    y_hat, l = clf.predict_(test_features, test_labels)
    print('test loss:', l)
    print('test accuracy:', data_process.accuracy(y_hat, test_labels))
    del clf
