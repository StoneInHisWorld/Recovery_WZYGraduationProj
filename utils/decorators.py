from functools import wraps

from utils import tool_func


def UnpackTrainParameters(train_func):
    @wraps(train_func)
    def wrapper(*args, **kwargs):
        parameters = \
            kwargs['num_epochs'] if 'num_epochs' in kwargs.keys() else 100, \
            kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001, \
            kwargs['weight_decay'] if 'weight_decay' in kwargs.keys() else 0.1, \
            kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 4, \
            tool_func.get_lossFunc(kwargs['loss']) if 'loss' in kwargs.keys() else tool_func.get_lossFunc(), \
            kwargs['momentum'] if 'momentum' in kwargs.keys() else 0, \
            kwargs['shuffle'] if 'shuffle' in kwargs.keys() else True
        parameters = *parameters, tool_func.get_Optimizer(
            args[0], optim_str=kwargs['optimizer'] if 'optimizer' in kwargs.keys() else 'SGD',
            learning_rate=parameters[1], weight_decay=parameters[2], momentum=parameters[6]
        )
        return train_func(*args, parameters=parameters)

    return wrapper


def InitNet(init_func):
    @wraps(init_func)
    def wrapper(*args, **kwargs):
        parameters = \
            kwargs['activation'] if 'activation' in kwargs.keys() else 'ReLU', \
            kwargs['dropout'] if 'dropout' in kwargs.keys() else 0., \
            kwargs['base'] if 'base' in kwargs.keys() else 2
        init_func(*args, parameters=parameters)
        init_method = kwargs['para_init'] if 'para_init' in kwargs.keys() else 'normal'
        # 初始化网络权重参数
        net = args[0]
        tool_func.init_netWeight(net, init_method)
        # 获取设备并转移
        net.__device__ = tool_func.try_gpu(0)
        net.to(net.__device__)

    return wrapper
