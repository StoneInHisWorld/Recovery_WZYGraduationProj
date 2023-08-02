import torch
from torch import nn

import utils.tool_func as tools
from network.basic_nn import BasicNN
from utils.decorators import init_net


class CNN(BasicNN):
    init_args = {
        'para_init': ('zero', ['normal', 'xavier', 'zero']),
        'activation': ('ReLU', ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']),
        'dropout': (0, []),
        'in_channels': ([1 for _ in range(8)], []),
        'out_channels': ([1 for _ in range(8)], []),
        'kernel_sizes': ([(3, 3) for _ in range(8)], []),
        'strides': ([2 for _ in range(8)], []),
        'bn_shapes': (None, []),
        'momentums': ([0.95 for _ in range(7)], [])
    }

    @init_net(init_args)
    def __init__(self, in_features: tuple[int, int], out_features: tuple[int, int],
                 parameters=None):
        activation, dropout, in_channels, out_channels, kernel_sizes, strides, \
            bn_shapes, momentums = parameters
        # 加入卷积层与激活层
        i = 0
        layers = [
            nn.Conv2d(in_channels[0], out_channels[0], kernel_sizes[0], strides[0]),
            tools.get_activation(activation),
            nn.Conv2d(in_channels[1], out_channels[1], kernel_sizes[1], strides[1]),
            tools.get_activation(activation),
        ]
        # 加入附带BN层的卷积层
        for i in range(2, len(in_channels)):
            layers.append(
                nn.BatchNorm2d(bn_shapes[i - 2], momentum=momentums[i - 2])
            )
            layers.append(
                nn.Conv2d(in_channels[i], out_channels[i], kernel_sizes[i], strides[i])
            )
            layers.append(
                tools.get_activation(activation)
            )
        # 附带BN层的最大池化层
        layers.append(
            nn.BatchNorm2d(bn_shapes[-1], momentum=momentums[-1])
        )
        layers.append(nn.MaxPool2d(bn_shapes[-1], stride=2))
        # if dropout > 0:
        #     layers.append(nn.Dropout(dropout))
        # self.output_nn1 = nn.Sequential(
        #     nn.Linear(bn_shapes[-1], out_features[0]),
        #     nn.Dropout(dropout),
        #     nn.Softmax()
        # )
        # self.output_nn2 = nn.Sequential(
        #     nn.Linear(bn_shapes[-1], out_features[1]),
        #     nn.Dropout(dropout),
        #     nn.Softmax()
        # )
        layers.append(
            DualOutputLayer(
                out_channels[-1], out_features[0], out_features[1],
                para_init='xavier', dropout=dropout
            )
        )
        # layers.append(self.output_nn1)
        # layers.append(self.output_nn2)
        super().__init__(*layers)

    # def forward(self, input):
    #     result = self(input)
    #     return self.output_nn1(result), self.output_nn2(result)


class DualOutputLayer(nn.Module):
    init_args = {
        'para_init': ('zero', ['normal', 'xavier', 'zero']),
        'dropout': (0, []),
    }

    @init_net(init_args)
    def __init__(self, in_features, fir_out, sec_out, parameters=None) -> None:
        dropout_rate, = parameters
        super().__init__()
        fir = nn.Sequential(
            nn.Linear(in_features, fir_out),
            nn.Dropout(dropout_rate),
            nn.Softmax()
        )
        sec = nn.Sequential(
            nn.Linear(in_features, sec_out),
            nn.Dropout(dropout_rate),
            nn.Softmax()
        )
        self.add_module('fir', fir)
        self.add_module('sec', sec)
        # self.fir_weight = nn.Parameter(torch.randn((in_features, fir_out)))
        # self.fir_bias = nn.Parameter(torch.randn((fir_out, )))
        # self.sec_weight = nn.Parameter(torch.randn((in_features, sec_out)))
        # self.sec_bias = nn.Parameter(torch.randn((sec_out, )))
        # nn.init.xavier_uniform_(fir.weight)
        # nn.init.xavier_uniform_(fir.bias)
        # nn.init.xavier_uniform_(sec.weight)
        # nn.init.xavier_uniform_(sec.bias)
        # self.__fir__ = fir
        # self.__sec__ = sec

    def forward(self, features):
        return torch.tensor(
            [child(features) for _, child in self.named_children()]
        )
