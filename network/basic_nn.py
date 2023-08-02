from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.decorators import unpack_kwargs
from utils import tool_func as tools


class BasicNN(nn.Sequential):

    def __init__(self, *layers):
        super().__init__(*layers)

    train_args = {
        'num_epochs': (100, []),
        'learning_rate': (0.001, []),
        'weight_decay': (0.1, []),
        'batch_size': (4, []),
        'loss': ('mse', ['l1', 'crossEntro', 'mse', 'huber']),
        'momentum': (0, []),
        'shuffle': (True, [True, False]),
        'optimizer': ('SGD', ['SGD', 'Adam'])
    }

    @unpack_kwargs(allow_args=train_args)
    def train_(self, train_features, train_labels, valid_features=None,
               valid_labels=None, parameters=None):
        """
        训练函数
        :param train_features: 进行训练的样本特征集
        :param train_labels: 进行训练的样本标签集
        :param valid_features: 进行验证的特征集（可选）
        :param valid_labels: 进行验证的标签集（可选）
        :param parameters: 额外参数。可选：
            'num_epochs': 迭代次数
            'learning_rate': 学习率
            'weight_decay': 权重衰减
            'batch_size': 批量大小
            'loss': 损失函数，支持['l1', 'crossEntro', 'mse', 'huber']
            'momentum': 动量
            'shuffle': 训练时是否打乱数据集
            'optimizer': 优化器,支持['SGD', 'Adam']
        :return: 训练损失列表（记录每次迭代的损失值），验证损失列表
        """
        num_epochs, learning_rate, weight_decay, batch_size, loss, momentum, shuffle, \
            optimizer = parameters
        loss = tools.get_lossFunc(loss)
        optimizer = tools.get_optimizer(
            self, optim_str=optimizer, learning_rate=learning_rate,
            weight_decay=weight_decay, momentum=momentum
        )
        train_ls, valid_ls = [], []
        train_iter = tools.load_array(
            train_features, train_labels, batch_size, num_epochs, shuffle
        )
        print('优化器参数：', optimizer.state_dict)
        print('使用的损失函数：', loss)
        print(f'训练batch大小：{batch_size}')
        print(f'是否打乱数据集：{shuffle}')
        # 进度条
        with tqdm(total=num_epochs, unit='epoch', desc='Iterating') as pbar:
            # 迭代一个批次
            for X, y in train_iter:
                self.train()
                optimizer.zero_grad()
                # 正向传播并计算损失
                l = loss(self(X), y)
                # 输出所有层的输出值
                with torch.no_grad():
                    print('\nres', self(X))
                # 反向传播
                l.backward()
                optimizer.step()
                # print(list(self.parameters()))
                # 记录损失
                with torch.no_grad():
                    train_ls.append(l.item())
                    if valid_features is not None and valid_labels is not None:
                        self.eval()
                        valid_ls.append(loss(self(valid_features), valid_labels).item())
                # 更新进度条
                pbar.update(1)
            pbar.close()
        return train_ls, valid_ls

    def test(self, features, labels):
        """
        测试函数
        :param features: 测试特征集
        :param labels: 测试标签机
        :return: 预测准确率
        """
        self.eval()
        preds = self(features)
        # preds = torch.tensor([])
        # 用循环分开计算每个样本
        # for f in features:
        #     preds = torch.vstack((preds, self(f))) if preds.shape[0] != 0 else self(f)
        # preds.append(self(f))
        # preds = torch.cat(preds, dim=1)
        # y_hat, y = torch.round(preds, decimals=1), torch.round(labels, decimals=1)
        y_hat, y = preds, labels
        del labels, preds, features
        correct = 0
        for i in range(len(y_hat)):
            if np.array_equal(y_hat[i], y[i]):
                correct += 1
        acc = correct / len(y)
        res = torch.hstack((y, y_hat))
        return acc, res

    def predict(self, features):
        return self(features)

    def __str__(self) -> str:
        return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self.__device__)