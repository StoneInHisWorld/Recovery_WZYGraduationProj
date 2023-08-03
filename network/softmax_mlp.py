import torch
from torch import nn

from network.mlp import MultiLayerPerception
import utils.tool_func as tools


class SoftmaxMLP(MultiLayerPerception):

    def __init__(self, in_features: int, out_features: int, dummies_columns: list[str],
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

    def predict(self, features):
        with torch.no_grad():
            preds = self(features)
            index_group = []
            # 将预测数据按照dummy列分组
            for label_name in tools.label_names:
                label_dummy_index = [i for i, d in enumerate(self.dummies_columns) if label_name in d]
                index_group.append(label_dummy_index)
            fact = []
            # 将组内预测值最大的列，赋其列名为当组预测值
            for pred in preds:
                row_fact = []
                last_end = 0
                for i, group in enumerate(index_group):
                    group_name = tools.label_names[i]
                    max_col = torch.argmax(pred[group], dim=0).item()
                    fact_value = self.dummies_columns[last_end + max_col].\
                        replace(group_name, "", 1).replace(tools.label_perfix, "", 1)
                    row_fact.append(float(fact_value))
                    last_end = group[-1] + 1
                fact.append(row_fact)
            preds = torch.tensor(fact, device=self.__device__)
            return preds

