from time import time

import numpy as np
import sklearn.multioutput
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

from utils.decorators import unpack_kwargs


class MultiRegressor(sklearn.multioutput.MultiOutputRegressor):
    init_args = {
        'estimator': ('L', ['HGBR', 'L', 'R', 'SGD']),
        'learning_rate': (0.1, []),
        'loss': ('squared_error', ['squared_error', 'absolute_error']),
        'max_iter': (100, []),
        'validation_fraction': (0.2, []),
        'alpha': (0.1, []),
        'random_state': (int(time()), [])
    }

    @unpack_kwargs(init_args)
    def __init__(self, parameters=None):
        estimator, learning_rate, self.__loss_func__, max_iter, validation_fraction, alpha, random_state = \
            parameters
        verbose = 1
        # assert estimator in estimators, f'不支持的分类器{estimator}，支持的分类器有{estimators}'
        if estimator == 'HGBR':
            super().__init__(HistGradientBoostingRegressor(
                learning_rate=learning_rate, loss=self.__loss_func__, max_iter=max_iter,
                validation_fraction=validation_fraction, random_state=random_state,
                verbose=verbose
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
                validation_fraction=validation_fraction, random_state=random_state, verbose=verbose
            ))
        self.parameters = self.estimator.get_params()

    def train_(self, X, y):
        self.fit(X, y)
        return self.get_params()

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
