import numpy as np
import pandas as pd
import torch

import utils.tool_func as tools
from utils.decorators import read_data

input_files = {
    '__features__': 'folder/img',
    '__labels__': 'csv'
}


class DataProcess:

    @read_data(input_files)
    def __init__(self, files: dict = None):
        self.__test_data__ = []
        self.__valid_data__ = []
        self.__train_data__ = []
        for k, v in files.items():
            setattr(self, k, v)
        self.__is_splitted__ = False
        self.__mode__ = None
        self.__prepared__ = False
        self.__valid__ = None
        assert self.__features__.shape[0] == self.__labels__.shape[0], '特征集和标签集长度须一致'

    def split_data(self, train, test, valid=.0):
        """
        分割数据集为训练集、测试集、验证集（可选）
        :param train: 训练集比例
        :param test: 测试集比例
        :param valid: 验证集比例
        :return: None
        """
        assert train + test + valid == 1.0, '训练集、测试集、验证集比例之和须为1'
        assert self.__is_splitted__ is False, '已经进行了数据集分割！'
        # assert self.__prepared__ is True, '请先对数据进行预处理，再进行数据集分割'
        data_len = self.__features__.shape[0]
        train_len = int(data_len * train)
        valid_len = int(data_len * valid)
        test_len = int(data_len * test)
        # 训练集分割
        train_start, train_end = 0, int(train_len)
        self.__train_data__ = [
            self.__features__[train_start:train_end],
            self.__labels__[train_start:train_end]
        ]
        if valid > 0:
            # 如需要分割验证集
            self.__valid__ = True
            valid_start, valid_end = train_end, int(train_end + valid_len)
            test_start, test_end = valid_end, int(valid_end + test_len)
            self.__valid_data__ = [
                self.__features__[valid_start:valid_end], self.__labels__[valid_start:valid_end]
            ]
            self.__test_data__ = [
                self.__features__[test_start:test_end], self.__labels__[test_start:test_end]
            ]
        else:
            test_start, test_end = train_end, int(train_end + test_len)
            self.__test_data__ = [
                self.__features__[test_start:test_end], self.__labels__[test_start:test_end]
            ]
        self.__is_splitted__ = True

    def preprocess(self, need: frozenset= frozenset()):
        """
        对对象中数据进行预处理，将对象的__prepared__标记为True
        :param need: 需求集合。将需要进行的预处理步骤对应字符串填入该集合，即可进行对应的预处理。
        :return: None
        """
        assert self.__prepared__ is False, '已经进行了预处理！'
        func = frozenset(['flatten', 'norm', 'tensor', 'onehot'])
        assert need.issubset(func), f'无法满足的需求{need}, 提供的功能包括{func}'
        if 'flatten' in need:
            self.__flatten_preprocess__()
        if 'onehot' in need:
            self.__onehot_preprocess__()
        if 'norm' in need:
            self.__norm_preprocess__()
        if 'tensor' in need:
            self.__tensor_preprocess__()
        self.__prepared__ = True

    def __flatten_preprocess__(self):
        # 预处理训练集
        for i, d in enumerate(self.__train_data__):
            d = self.__flatten__(d)
            self.__train_data__[i] = d
        # 预处理验证集
        if self.__valid__:
            for i, d in enumerate(self.__valid_data__):
                d = self.__flatten__(d)
                self.__valid_data__[i] = d
        # 预处理测试集
        for i, d in enumerate(self.__test_data__):
            d = self.__flatten__(d)
            self.__test_data__[i] = d

    def __norm_preprocess__(self):
        # 预处理训练集
        for i, d in enumerate(self.__train_data__):
            d = self.__normalize__(d)
            self.__train_data__[i] = d
        # 预处理验证集
        if self.__valid__:
            for i, d in enumerate(self.__valid_data__):
                d = self.__normalize__(d)
                self.__valid_data__[i] = d

    def __tensor_preprocess__(self):
        # 预处理训练集
        for i, d in enumerate(self.__train_data__):
            d = self.__to_Tensor__(d)
            self.__train_data__[i] = d
        # 预处理验证集
        if self.__valid__:
            for i, d in enumerate(self.__valid_data__):
                d = self.__to_Tensor__(d)
                self.__valid_data__[i] = d
        # 预处理测试集
        for i, d in enumerate(self.__test_data__):
            d = self.__to_Tensor__(d, requires_grad=False)
            self.__test_data__[i] = d

    def __onehot_preprocess__(self):
        # 合并标签集
        labels = (self.__train_data__[1], self.__valid_data__[1], self.__test_data__[1]) \
            if self.__valid__ else (self.__train_data__[1], self.__test_data__[1])
        labels = np.vstack(labels)
        # 计算独热编码
        labels = self.__get_dummies__(labels)
        # 赋值给类成员
        train_len, valid_len, test_len = len(self.__train_data__[1]), \
            len(self.__valid_data__[1]), len(self.__test_data__[1])
        res = np.split(
            labels, (train_len, train_len + valid_len)
        )
        train_labels, valid_labels, test_labels = res
        self.__train_data__ = [self.__train_data__[0], train_labels]
        self.__valid_data__ = [self.__valid_data__[0], valid_labels]
        self.__test_data__ = [self.__test_data__[0], test_labels]

    @staticmethod
    def __get_dummies__(data):
        data = pd.DataFrame(data, columns=tools.label_names)
        dummies = pd.get_dummies(data, columns=tools.label_names)
        return np.array(dummies)

    @staticmethod
    def __to_Tensor__(data, requires_grad=True):
        device = tools.try_gpu(0)
        return torch.tensor(data, dtype=torch.float32,
                            device=device, requires_grad=requires_grad)

    @staticmethod
    def __normalize__(data):
        data = np.array(data)
        return np.apply_along_axis(
            lambda x: (x - x.mean()) / x.std()
            if x.std() != 0 else x - x.mean(),
            1, data
        )

    @staticmethod
    def __flatten__(data):
        oriShape = data.shape
        return data.reshape((oriShape[0], -1))

    @staticmethod
    def accuracy(y_hat, y):
        if y_hat is torch.Tensor or y is torch.Tensor:
            equal_func = torch.equal
        else:
            equal_func = np.array_equal
        # y_hat, y = np.array(y_hat), np.array(y)
        correct = 0
        for i in range(len(y)):
            # if np.array_equal(y_hat[i], y[i]):
            if equal_func(y_hat[i], y[i]):
                correct += 1
        acc = correct / len(y)
        return acc

    @property
    def train_data(self):
        return self.__train_data__

    @property
    def test_data(self):
        return self.__test_data__

    @property
    def valid_data(self):
        return self.__valid_data__
