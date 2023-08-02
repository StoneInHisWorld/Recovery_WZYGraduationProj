from time import time

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from tqdm import tqdm
from utils import tool_func as tools

from utils.decorators import unpack_kwargs


class MultiRegressor(MultiOutputRegressor):
    init_args = {
        'estimator': ('L', ['HGBR', 'L', 'R', 'SGD']),
        'learning_rate': (0.1, []),
        'loss': ('squared_error', ['squared_error', 'absolute_error']),
        # 'max_iter': (100, []),
        'validation_fraction': (0.2, []),
        'alpha': (0.1, []),
        'random_state': (int(time()), [])
    }

    @unpack_kwargs(init_args)
    def __init__(self, parameters=None):
        """
        构建一个多输出回归器。
        :param parameters: 可选关键词参数。包括如下：
            `estimator`: (`L`, [`HGBR`, `L`, `R`, `SGD`]),
            `learning_rate`: (0.1, []),
            `loss`: (`squared_error`, [`squared_error`, `absolute_error`]),
            `validation_fraction`: (0.2, []),
            `alpha`: (0.1, []),
            `random_state`: (int(time()), [])
        """
        estimator, learning_rate, self.__loss_func__, validation_fraction, \
            alpha, random_state = parameters
        verbose = 1
        # assert estimator in estimators, f'不支持的分类器{estimator}，支持的分类器有{estimators}'
        # if estimator == 'HGBR':
        #     super().__init__(HistGradientBoostingRegressor(
        #         learning_rate=learning_rate, loss=self.__loss_func__, max_iter=max_iter,
        #         validation_fraction=validation_fraction, random_state=random_state,
        #         verbose=verbose
        #     ))
        # elif estimator == 'L':
        #     super().__init__(LinearRegression())
        # elif estimator == 'R':
        #     super().__init__(Ridge(
        #         max_iter=max_iter, alpha=alpha, random_state=random_state
        #     ))
        # elif estimator == 'SGD':
        #     super().__init__(SGDRegressor(
        #         max_iter=max_iter, alpha=alpha, eta0=learning_rate, loss=self.__loss_func__,
        #         validation_fraction=validation_fraction, random_state=random_state, verbose=verbose
        #     ))
        if estimator == 'HGBR':
            super().__init__(HistGradientBoostingRegressor(
                learning_rate=learning_rate, loss=self.__loss_func__,
                validation_fraction=validation_fraction, random_state=random_state,
                verbose=verbose, max_iter=1
            ))
        elif estimator == 'L':
            super().__init__(LinearRegression())
        elif estimator == 'R':
            super().__init__(Ridge(
                alpha=alpha, random_state=random_state, max_iter=1
            ))
        elif estimator == 'SGD':
            super().__init__(SGDRegressor(
                alpha=alpha, eta0=learning_rate, loss=self.__loss_func__,
                validation_fraction=validation_fraction, random_state=random_state,
                verbose=verbose, max_iter=1
            ))
        # self.parameters = self.estimator.get_params()

    train_args = {
        'num_epochs': (100, []),
        'batch_size': (4, []),
        'shuffle': (True, [True, False]),
    }

    @unpack_kwargs(allow_args=train_args)
    def train_(self, X, y, parameters=None) -> list[float]:
        """
        使用partial_fit()函数对模型进行训练。
        :param X: 训练数据特征集
        :param y: 训练数据标签集
        :param parameters: 可选关键词参数，包括：
            `num_epochs`: (100, []),
            `batch_size`: (4, []),
            `shuffle`: (True, [True, False])
        :return: 每次迭代所得训练损失值
        """
        num_epochs, batch_size, shuffle = parameters
        losses = []
        with tqdm(total=num_epochs, desc='Iterating...', unit='epoch') as pbar:
            for X, y in tools.load_array(X, y, batch_size, num_epochs, shuffle):
                self.fit(X, y)
                l = self.__loss__(self.predict(X), y)
                losses.append(l)
                for i, estimator in enumerate(self.estimators_):
                    print(f'\nestimator {i} coef_:', estimator.coef_)
                pbar.update(1)
        return losses

    def predict_(self, X, y):
        preds = np.around(self.predict(X), decimals=1)
        y = np.array(y)
        l = self.__loss__(preds, y)
        return preds, l

    def __loss__(self, preds, y):
        if self.__loss_func__ == 'squared_error':
            return metrics.mean_squared_error(y, preds)
        elif self.__loss_func__ == 'absolute_error':
            return metrics.mean_absolute_error(y, preds)

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def parameters(self):
        return self.estimator.get_params()
