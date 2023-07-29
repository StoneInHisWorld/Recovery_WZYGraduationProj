from functools import wraps
from time import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

estimators = ['HGBR', 'L', 'R', 'SGD']


# def get_estimator(estimator='L', parameters=None):
#     learning_rate, loss, max_iter, validation_fraction, alpha, random_state = \
#         unpack_parameters()
#     assert estimator in estimators, f'不支持的分类器{estimator}，支持的分类器有{estimators}'
#     if estimator == 'HGBR':
#         return HistGradientBoostingRegressor(
#             learning_rate=learning_rate, loss=loss, max_iter=max_iter,
#             validation_fraction=validation_fraction, random_state=random_state,
#             verbose=1
#         )
#     elif estimator == 'L':
#         return LinearRegression()
#     elif estimator == 'R':
#         return Ridge(
#             max_iter=max_iter, alpha=alpha, random_state=random_state
#         )
#     elif estimator == 'SGD':
#         return SGDRegressor(
#             max_iter=max_iter, alpha=alpha, eta0=learning_rate, loss=loss,
#             validation_fraction=validation_fraction, random_state=random_state, verbose=1
#         )
#
#
# def InitRegres(init_fnc):
#     @wraps(init_fnc)
#     def wrapper(*args, **kwargs):
#         estimator_str = kwargs['estimator'] if 'estimator' in kwargs.keys() else 'L'
#         kwargs.pop('estimator')
#         parameters = \
#             get_estimator(kwargs['estimator'])
#
#     return wrapper


class MultiRegressor(MultiOutputRegressor):

    def __init__(self, estimator='L', **kwargs):
        learning_rate, self.__loss_func__, max_iter, validation_fraction, alpha, random_state = \
            self.unpack_parameters(kwargs)
        assert estimator in estimators, f'不支持的分类器{estimator}，支持的分类器有{estimators}'
        if estimator == 'HGBR':
            super().__init__(HistGradientBoostingRegressor(
                learning_rate=learning_rate, loss=self.__loss_func__, max_iter=max_iter,
                validation_fraction=validation_fraction, random_state=random_state,
                verbose=1
            ))
        elif estimator == 'L':
            super().__init__(LinearRegression())
        elif estimator == 'R':
            super().__init__(Ridge(
                max_iter=max_iter, alpha=alpha, random_state=random_state
            ))
        elif estimator == 'SGD':
            super().__init__(SGDRegressor(
                max_iter=max_iter, alpha=alpha, eta0=learning_rate, loss=self.__loss_func__,
                validation_fraction=validation_fraction, random_state=random_state, verbose=1
            ))

    def train_(self, X, y):
        self.fit(X, y)
        return self.get_params()

    def predict_(self, X, y):
        preds = np.around(self.predict(X), decimals=1)
        y = np.array(y)
        l = self.__loss__(preds, y)
        return preds, l

    @staticmethod
    def unpack_parameters(pars):
        """
        获取超参数
        :param pars: 用户输入的超参数
        :return: 提取后的超参数
        """
        parameters = \
            pars['learning_rate'] if 'learning_rate' in pars.keys() else 0.1, \
            pars['loss'] if 'loss' in pars.keys() else 'squared_error', \
            pars['max_iter'] if 'max_iter' in pars.keys() else 100, \
            pars['validation_fraction'] if 'validation_fraction' in pars.keys() else 0.2, \
            pars['alpha'] if 'alpha' in pars.keys() else 0.1, \
            pars['random_state'] if 'random_state' in pars.keys() else int(time())
        return parameters

    def __loss__(self, preds, y):
        losses = ['squared_error', 'absolute_error', 'huber', 'quantile']
        assert self.__loss_func__ in losses, f'不支持的损失函数{self.__loss_func__}，支持的损失函数有{losses}'
        if self.__loss_func__ == 'squared_error':
            return metrics.mean_squared_error(y, preds)
