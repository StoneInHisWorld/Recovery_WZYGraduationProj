import torch
from torch import nn
import numpy as np

from utils.decorators import init_net


class SoftmaxMLP(nn.Sequential):

    @init_net
    def __init__(self, in_features: int, out_features: int, parameters=None):
        """
        建立一个多层感知机。直接将变量名作为方法，输入输入数据集，就可以得到模型的输出数据集。
        :param in_features: 输入特征向量维度
        :param out_features: 输出标签向量维度
        :param parameters: 可选关键字参数：
            'activation': 激活函数，可选['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
            'para_init': 初始化函数，可选['normal', 'xavier', 'zero']
            'dropout': dropout比例。若为0则不加入dropout层
            'base': 神经网络构建衰减指数。**可能会出现无法进行矩阵运算的错误，请届时更改本参数！**
        """
        activation, dropout, base = parameters
        print('GPU可用否：', torch.cuda.is_available())
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