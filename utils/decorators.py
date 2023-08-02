from functools import wraps

from utils import tool_func as tools


def unpack_kwargs(allow_args: dict[str, tuple]):
    """
    为函数拆解输入的关键字参数。
    若调用时对某字符串参数进行赋值，则会检查该参数是否在提供的参数范围内。
    若未检测到输入值，则会指定默认值。
    会忽略不识别的参数值。
    需要被装饰函数含有parameters参数，设定其值为None，用于接受拆解过后的关键词参数。
    所有关键字参数均被视为用于拆解的参数，被装饰函数不得拥有关键字参数。
    :param allow_args: 函数允许的关键词参数，字典key为参数关键字，value为二元组。
        二元组0号位为默认值，1号位为允许范围。1号位仅当参数类型为字符串时有效，否则为空列表。
    :return: 装饰过的函数
    """
    def unpack_decorator(train_func):
        """
        装饰器
        :param train_func: 需要参数的函数
        :return 装饰过的函数
        """
        @wraps(train_func)
        def wrapper(*args, **kwargs):
            """
            按允许输入参数排列顺序拆解输入参数，或者赋值为其默认值
            :param args: train_func的位置参数
            :param kwargs: train_func的关键字参数
            :return: 包装函数
            """
            parameters = ()
            for k in allow_args.keys():
                default = allow_args[k][0]
                allow_range = allow_args[k][1]
                input_arg = kwargs.pop(k, default)
                if isinstance(default, str):
                    assert input_arg in allow_range, \
                        f'输入参数{k}:{input_arg}不在允许范围内，允许的值为{allow_range}'
                parameters = *parameters, input_arg
            return train_func(*args, parameters=parameters)

        return wrapper

    return unpack_decorator


def init_net(allow_args: dict[str: tuple]):
    """
    对网络进行初始化的装饰器。
    负责拆解参数，搭建网络后对网络权重参数进行初始化，并进行设备迁移。
    需要被装饰函数含有parameters参数，设定其值为None，用于接受拆解过后的关键词参数。
    所有关键字参数均被视为用于拆解的参数，被装饰函数不得拥有关键字参数。
    :param allow_args: 允许输入的关键字参数。0号位参数为指定初始化权重参数方式，其后参数可任意指定。
    :return: 装饰过的__init__函数
    """
    # def init_net(init_func):
    def decorator(init_func):
        """
        装饰器
        :param init_func: 被修饰的函数
        :return: 包装过的函数
        """
        @wraps(init_func)
        def wrapper(*args, **kwargs):
            """
            init_func的包装函数。
            :param args: init_func的位置参数
            :param kwargs: init_func的关键字参数
            :return: None
            """
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
            #     kwargs.pop('activation', 'ReLU'), \
            #     kwargs.pop('dropout', 0.), \
            #     kwargs.pop('base', 2)
            init_method, parameters = parameters[0], parameters[1:]
            init_func(*args, parameters=parameters)
            # 初始化网络权重参数
            net = args[0]
            # tools.init_netWeight(net, kwargs.pop('para_init', 'normal'))
            tools.init_netWeight(net, init_method)
            # 获取设备并转移
            setattr(net, '__device__', tools.try_gpu(0))
            net.to(net.__device__)

        return wrapper
    return decorator


def read_data(allowed_files: dict[str, str]):
    """
    文件读取装饰器，被修饰函数将获得读取文件的功能。
    被修饰函数需要含有files关键字参数，用于接受装饰器所读取的用户输入的文件。
    所有关键字参数均被视为用于需要读取的文件，被装饰函数不得拥有关键字参数。
    :param allowed_files: 指定需要识别的文件、
        :key: 用途字符串，其值将会被作为对象的变量名，保存该文件数据。
        :var: 文件对应类型。若读取文件为文件夹，则格式为'folder/...'，否则为不带句点的文件类型名如'img'
    :return: 被装饰过的函数
    """
    def rd_decorators(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            init_func的包装函数。
            :param args: init_func的位置参数
            :param kwargs: init_func的关键字参数
            :return: None
            """
            files = {}
            for usage in allowed_files.keys():
                assert usage in kwargs.keys(), f'缺少{usage}文件！'
                path = kwargs[usage]
                file_type = allowed_files[usage]
                file = tools.read_data(path, file_type)
                files[usage] = file
            func(*args, files=files)
        return wrapper

    return rd_decorators
