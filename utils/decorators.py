from functools import wraps

from utils import tool_func


def unpack_kwargs(allow_args: dict):
    """
    为函数拆解输入的关键字参数
    :param allow_args: 函数允许的关键词参数
    :return: 装饰过的函数
    """
    def unpack_decorator(train_func):
        """
        拆解装饰器
        :param train_func: 需要参数的函数
        :return 装饰过的函数
        """
        @wraps(train_func)
        def wrapper(*args, **kwargs):
            # 按允许输入参数排列顺序拆解输入参数，或者赋值为其默认值
            parameters = ()
            for k in allow_args.keys():
                default = allow_args[k][0]
                allow_range = allow_args[k][1]
                input_arg = kwargs.pop(k, default)
                if isinstance(default, str):
                    assert input_arg in allow_range, \
                        f'输入参数{k}:{input_arg}不在允许范围内，允许的值为{allow_range}'
                parameters = *parameters, input_arg

            # parameters = \
            #     kwargs['num_epochs'] if 'num_epochs' in kwargs.keys() else 100, \
            #     kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001, \
            #     kwargs['weight_decay'] if 'weight_decay' in kwargs.keys() else 0.1, \
            #     kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 4, \
            #     tool_func.get_lossFunc(kwargs['loss']) if 'loss' in kwargs.keys() else tool_func.get_lossFunc(), \
            #     kwargs['momentum'] if 'momentum' in kwargs.keys() else 0, \
            #     kwargs['shuffle'] if 'shuffle' in kwargs.keys() else True

            # parameters = \
            #     kwargs.pop('num_epochs', 100), kwargs.pop('learning_rate', 0.001), \
            #     kwargs.pop('weight_decay', 0.1), kwargs.pop('batch_size', 4), \
            #     tool_func.get_lossFunc(kwargs.pop('loss', 'mse')), \
            #     kwargs.pop('momentum', 0), kwargs.pop('shuffle', True)

            # parameters = *parameters, tool_func.get_Optimizer(
            #     args[0], optim_str=kwargs['optimizer'] if 'optimizer' in kwargs.keys() else 'SGD',
            #     learning_rate=parameters[1], weight_decay=parameters[2], momentum=parameters[6]
            # )
            # parameters = *parameters, tool_func.get_Optimizer(
            #     args[0], optim_str=kwargs.pop('optimizer', 'SGD'),
            #     learning_rate=parameters[1], weight_decay=parameters[2], momentum=parameters[6]
            # )
            return train_func(*args, parameters=parameters)
        return wrapper
    return unpack_decorator


init_args = {
    'activation': ('ReLU', ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']),
    'para_init': ('zero', ['normal', 'xavier', 'zero']),
    'dropout': (0, []),
    'base': (2, [])
}


def init_net(init_func):
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
