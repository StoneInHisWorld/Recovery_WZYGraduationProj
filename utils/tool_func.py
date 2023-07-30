"""以下是工具函数"""
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

init_funcs = ['normal', 'xavier', 'zero']
optimizers = ['SGD', 'Adam']
activations = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
losses = ['l1', 'crossEntro', 'mse', 'huber']
train_para = ['num_epochs', 'learning_rate', 'weight_decay', 'batch_size',
              'optimizer', 'loss', 'momentum']
support_files = ['img', 'csv']
label_names = ['OAM1', 'OAM2']


def get_activation(activ_str='ReLU'):
    assert activ_str in activations, \
        f'不支持激活函数{activ_str}, 支持的优化器包括{activations}'
    if activ_str == 'ReLU':
        return nn.ReLU(inplace=True)
    elif activ_str == 'Sigmoid':
        return nn.Sigmoid()
    elif activ_str == 'Tanh':
        return nn.Tanh()
    elif activ_str == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True)


def get_lossFunc(loss_str='mse'):
    assert loss_str in losses, \
        f'不支持激活函数{loss_str}, 支持的优化器包括{losses}'
    if loss_str == 'l1':
        return nn.L1Loss()
    elif loss_str == 'crossEntro':
        return nn.CrossEntropyLoss()
    elif loss_str == 'mse':
        return nn.MSELoss()
    elif loss_str == 'huber':
        return nn.HuberLoss()


def get_Optimizer(net, optim_str, learning_rate, weight_decay, momentum) -> torch.optim:
    assert optim_str in optimizers, f'不支持优化器{optim_str}, 支持的优化器包括{optimizers}'
    if optim_str == 'SGD':
        return torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )
    elif optim_str == 'Adam':
        return torch.optim.Adam(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


def try_gpu(i=0):
    """
    获取一个GPU
    :param i: GPU编号
    :return: 第i号GPU。若GPU不可用，则返回CPU
    """
    print(torch.cuda.is_available())
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def init_netWeight(net, func_str):
    assert func_str in init_funcs, f'不支持的初始化方式{func_str}, 当前支持的初始化方式包括{init_funcs}'
    if func_str == 'normal':
        net.apply(lambda m:
                  nn.init.normal_(m.weight, 0, 1) if type(m) == nn.Linear else m)
        net.apply(lambda m:
                  nn.init.normal_(m.bias, 0, 1) if type(m) == nn.Linear else m)
    if func_str == 'xavier':
        net.apply(lambda m:
                  nn.init.xavier_uniform_(m.weight) if type(m) == nn.Linear else m)
        net.apply(lambda m:
                    nn.init.zeros_(m.bias) if type(m) == nn.Linear else m)
    if func_str == 'zero':
        net.apply(lambda m:
                  nn.init.zeros_(m.weight) if type(m) == nn.Linear else m)
        net.apply(lambda m:
                  nn.init.zeros_(m.bias) if type(m) == nn.Linear else m)


def get_readFunc(func):
    assert func in support_files, f'不支持读取{func}文件，支持的文件包括{support_files}'
    if func == 'img':
        func = read_img
    elif func == 'csv':
        func = read_csv
    return func


def read_data(path: str, file_type: str) -> np.ndarray:
    is_folder = True if file_type.split('/')[0] == 'folder' else False
    data = []
    if is_folder:
        read_func = get_readFunc(file_type.split('/')[1])
        walk_gen = os.walk(path)
        for _, __, file_names in walk_gen:
            file_names = sorted(
                file_names, key=lambda name: int(name.split(".")[0])
            )  # 给文件名排序！
            for fn in file_names:
                data.append(read_func(path + '/' + fn))
    else:
        read_func = get_readFunc(file_type)
        data = read_func(path)
    return np.array(data)


def read_img(path: str) -> np.ndarray:
    img = plt.imread(path)
    return np.array(img)


def read_csv(path: str) -> np.ndarray:
    return pd.read_csv(path, names=label_names).values
