import torch
from torch import nn

from network.mlp import MultiLayerPerception
import utils.tool_func as tools


class SoftmaxMLP(MultiLayerPerception):

    def __init__(self, in_features: int, out_features: int, dummies_columns: list,
                 **kwargs):
        """
        建立一个带softmax的多层感知机。
        :param in_features: 输入特征向量维度
        :param out_features: 输出标签向量维度
        :param kwargs: 可选关键字参数：
            'activation': 激活函数，可选['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
            'para_init': 初始化函数，可选['normal', 'xavier', 'zero']
            'dropout': dropout比例。若为0则不加入dropout层
            'base': 神经网络构建衰减指数。
        """
        super().__init__(in_features, out_features, **kwargs)
        self.append(nn.Softmax())
        self.dummies_columns = dummies_columns
    # def test(self, features, labels, dummies_columns):
    #     with torch.no_grad():
    #         preds = self(features)
    #         preds = torch.argmax()

    def predict(self, features):
        with torch.no_grad():
            preds = self(features)
            index_group = []
            # 将预测数据按照dummy列分组
            for label_name in tools.label_names:
                label_dummy_index = [i for i, d in enumerate(self.dummies_columns) if label_name in d]
                index_group.append(label_dummy_index)
            # 将组内每条预测数据中预测值最大的列赋值为1，其余赋值为0
            last_end = 0
            for group in index_group:
                # part = preds[:, group]
                max_index = torch.argmax(preds[:, group], dim=1)
                for i in range(len(preds)):
                    for j in group:
                        preds[i, j] = 1 if j == max_index[i] + last_end else 0
                last_end = group[-1] + 1
                # for i, row in enumerate(preds[:, group]):
                #     for j, col in enumerate(row):
                #         preds[:, group][i, j] = 1 if j == max_index[i] else 0
            return preds

