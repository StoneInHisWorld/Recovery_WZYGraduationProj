"""以下是工具函数"""
import torch
from torch import nn

init_funcs = ['normal', 'xavier', 'zero']
optimizers = ['SGD', 'Adam']
activations = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
losses = ['l1', 'crossEntro', 'mse', 'huber']
train_para = ['num_epochs', 'learning_rate', 'weight_decay', 'batch_size',
              'optimizer', 'loss', 'momentum']


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
