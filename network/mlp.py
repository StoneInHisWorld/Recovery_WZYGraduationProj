import math
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.decorators import InitNet, UnpackTrainParameters
from utils.tool_func import get_activation


class MultiLayerPerception(nn.Sequential):

    @InitNet
    def __init__(self, in_features: int, out_features: int, parameters=None):
        """
        建立一个多层感知机。直接将变量名作为方法，输入输入数据集，就可以得到模型的输出数据集。
        :param in_features: 输入特征向量维度
        :param out_features: 输出标签向量维度
        :param parameters: 可选关键字参数：
            `activation`: 激活函数，可选['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
            `para_init`: 初始化函数，可选['normal', 'xavier', 'zero']
            `dropout`: dropout比例。若为0则不加入dropout层
            `base`: 神经网络构建衰减指数。**可能会出现无法进行矩阵运算的错误，请届时更改本参数！**
        """
        activation, dropout, base = parameters
        # 构建模块层
        layers = []
        layer_sizes = np.logspace(
            math.log(in_features, base),
            math.log(out_features, base),
            int(math.log(in_features - out_features, base)),
            base=base
        )
        layer_sizes = list(map(int, layer_sizes))
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            layers.append(nn.Linear(in_size, out_size))
            layers.append(get_activation(activation))
            if dropout > 0.:
                layers.append(nn.Dropout())
        super().__init__(nn.BatchNorm1d(in_features), *layers)

    @UnpackTrainParameters
    def train_(self, train_features, train_labels, valid_features=None,
               valid_labels=None, parameters=None):
        """
        训练函数
        :param train_features: 进行训练的样本特征集
        :param train_labels: 进行训练的样本标签集
        :param valid_features: 进行验证的特征集（可选）
        :param valid_labels: 进行验证的标签集（可选）
        :param parameters: 额外参数。可选：
            `num_epochs`: 迭代次数
            `learning_rate`: 学习率
            `weight_decay`: 权重衰减
            `batch_size`: 批量大小
            `loss`: 损失函数，支持['l1', 'crossEntro', 'mse', 'huber']
            `momentum`: 动量
            `shuffle`: 训练时是否打乱数据集
            `optimizer`: 优化器,支持['SGD', 'Adam']
        :return: 训练损失列表（记录每次迭代的损失值），验证损失列表
        """
        num_epochs, learning_rate, weight_decay, batch_size, loss, momentum, shuffle, \
            optimizer = parameters
        train_ls, valid_ls = [], []
        train_iter = self.load_array(
            train_features, train_labels, batch_size, num_epochs, shuffle
        )
        print('优化器参数：', optimizer.state_dict)
        print('使用的损失函数：', loss)
        # 进度条
        with tqdm(total=num_epochs, unit='epoch', desc='Iterating') as pbar:
            # 迭代一批次
            for X, y in train_iter:
                self.train()
                optimizer.zero_grad()
                # 正向传播并计算损失
                l = loss(self(X), y)
                # 反向传播
                l.backward()
                optimizer.step()
                print(list(self.parameters()))
                # 记录损失
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
        y_hat, y = torch.tensor(preds, device=labels.device), labels
        del labels, preds, features
        correct = 0
        for i in range(len(y_hat)):
            if np.array_equal(y_hat[i], y[i]):
                correct += 1
        acc = correct / len(y)
        res = torch.hstack((y, y_hat))
        return acc, res

    @staticmethod
    def load_array(features, labels, batch_size, num_epochs, shuffle=True):
        """
        加载训练数据
        :param shuffle: 每次输出批次是否打乱数据
        :param num_epochs: 迭代次数
        :param features: 特征集
        :param labels: 标签集
        :param batch_size: 批量大小
        :return 不断产出数据的迭代器
        """
        num_examples = features.shape[0]
        assert batch_size <= num_examples, f'批量大小{batch_size}需要小于样本数量{num_examples}'
        examples_indices = list(range(num_examples))
        # 每次供给一个批次的数据
        for _ in range(num_epochs):
            if shuffle:
                random.shuffle(examples_indices)
            yield features[examples_indices][:batch_size], labels[examples_indices][:batch_size]

    def __log_rmse__(self, features, labels, loss):
        # 为了在取对数时进一步稳定该值，将小于1的值设置为1
        # print(self(features))
        # clipped_preds = torch.clamp(self(features), 1, float('inf'))
        # rmse = torch.sqrt(loss(torch.log(clipped_preds),
        #                        torch.log(labels)))
        pred = self(features)
        pred = torch.log(pred)
        l = loss(pred, torch.log(labels))
        rmse = torch.sqrt(l)
        return rmse.item()

    def __str__(self) -> str:
        return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self.__device__)
