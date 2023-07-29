import numpy as np
import torch


class data_process:

    def __init__(self, **kwargs):
        self.__test_data__ = []
        self.__valid_data__ = []
        self.__train_data__ = []
        self.__features__ = kwargs['features']
        self.__labels__ = kwargs['labels']
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

    def preprocess(self, mode='linear', need_tensor=False, need_norm=True):
        """
        对对象中数据进行预处理，将对象的__prepared__标记为True
        :param need_norm: 是否需要标准化
        :param need_tensor: 是否需要转换成张量
        :param mode: 预处理模式，包括'linear'
        :return: None
        """
        mode_list = ['linear', 'none']
        assert self.__prepared__ is False, '已经进行了预处理！'
        assert mode in mode_list, f'mode参数{mode}须为{mode_list}其中一个'
        if mode == 'linear':
            self.__mode__ = 'linear'
            self.__linear_preprocess__(need_norm=need_norm, need_tensor=need_tensor)
        self.__prepared__ = True

    def __linear_preprocess__(self, need_norm=True, need_tensor=True):
        assert self.__mode__ == 'linear'
        assert self.__is_splitted__, '请先进行数据集分割，再进行数据预处理！'
        # 预处理训练集
        for i, d in enumerate(self.__train_data__):
            d = self.__flatten__(d)
            d = self.__normalize__(d) if need_norm else d
            d = self.__to_Tensor__(d) if need_tensor else d
            self.__train_data__[i] = d
        # 预处理验证集
        if self.__valid__:
            for i, d in enumerate(self.__valid_data__):
                d = self.__flatten__(d)
                d = self.__normalize__(d) if need_norm else d
                d = self.__to_Tensor__(d) if need_tensor else d
                self.__valid_data__[i] = d
        # 预处理测试集
        for i, d in enumerate(self.__test_data__):
            d = self.__flatten__(d)
            d = self.__to_Tensor__(d, requires_grad=False) if need_tensor else d
            self.__test_data__[i] = d


    @staticmethod
    def __to_Tensor__(data, requires_grad=True):
        device = data_process.__try_gpu__(0)
        return torch.tensor(data,  dtype=torch.float32,
                            device=device, requires_grad=requires_grad)

    @staticmethod
    def __normalize__(data):
        data = np.array(data)
        # mean = data.mean(axis=1)
        # std = data.std(axis=1)
        # print(data.std(axis=1))
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
    def __try_gpu__(i=0):
        """
        获取一个GPU
        :param i: GPU编号
        :return: 获取成功，则返回GPU，否则返回CPU
        """
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    @staticmethod
    def accuracy(y_hat, y):
        y_hat, y = np.array(y_hat), np.array(y)
        correct = 0
        for i in range(len(y)):
            if np.array_equal(y_hat[i], y[i]):
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

