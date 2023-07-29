# import random
# from collections import Iterable
# from functools import wraps
#
# import numpy as np
# import torch
# from torch import nn
# from tqdm import tqdm
#
# from Recovery_WZYGraduationProject.decorators_and_tools import UnpackTrainParameters
# from Recovery_WZYGraduationProject.network.mlp import MultiLayerPerception
#
# init_funcs = ['normal', 'xavier', 'zero']
# optimizers = ['SGD', 'Adam']
# activations = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
# losses = ['l1', 'crossEntro', 'mse', 'huber']
# train_para = ['num_epochs', 'learning_rate', 'weight_decay', 'batch_size',
#               'optimizer', 'loss', 'momentum']
#
# # """理论合理性存疑"""
# # def RationalityCheck(func):
# #     def wrapper():
# #         print('警告：该类的理论合理性存疑！')
# #     func()
#
#
# # def doubtful():
# #
#
#
# """理论合理性存疑"""
#
#
# class MLPVector:
#
#     def __init__(self, in_features, out_features, activation_s=None,
#                  para_init_s=None, dropout_s=None, base_s=None):
#         """
#         建立一个多层感知机向量，其中每个多层感知机只输出标量，再将标量拼接成单个标签。
#         直接将变量名作为方法，输入输入数据集，就可以得到模型的输出数据集。
#         :param in_features: 输入特征向量维度
#         :param out_features: 输出标签向量维度
#         :param activation_s: 激活函数列表，可选['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
#         :param para_init_s: 初始化函数列表，可选['normal', 'xavier', 'zero']
#         :param dropout_s: dropout比例列表，若为0则不加入dropout层
#         :param base_s: 神经网络构建衰减指数列表。**可能会出现无法进行矩阵运算的错误，请届时更改本参数！**
#         """
#         assert isinstance(activation_s, Iterable) and isinstance(para_init_s, Iterable) and \
#                isinstance(dropout_s, Iterable) and isinstance(base_s, Iterable), \
#                '激活函数、初始化函数、dropout比例、神经网络衰减指数均需要为可迭代对象，为每个感知机指定参数'
#         # 初始化参数
#         activation_s = ['ReLU' for _ in range(out_features)] if not activation_s else activation_s
#         para_init_s = ['normal' for _ in range(out_features)] if not para_init_s else para_init_s
#         dropout_s = [0. for _ in range(out_features)] if not dropout_s else dropout_s
#         base_s = [2 for _ in range(out_features)] if not base_s else base_s
#         # 建立感知机向量
#         self.__mlp_s__ = [
#             MultiLayerPerception(in_features, 1, activation=activation_s[i],
#                                  para_init=para_init_s[i], dropout=dropout_s[i],
#                                  base=base_s[i])
#             for i in range(out_features)
#         ]
#
#     @UnpackTrainParameters
#     def train_(self, train_features, train_labels, valid_features=None,
#                valid_labels=None, parameters=None) -> (list, list):
#         """
#         训练函数
#         :param train_features: 进行训练的样本特征集
#         :param train_labels: 进行训练的样本标签集
#         :param valid_features: 进行验证的特征集（可选）
#         :param valid_labels: 进行验证的标签集（可选）
#         :param kwarg: 额外参数。可选：
#             'num_epochs': 迭代次数；'learning_rate_s': 学习率；'weight_decay_ies': 权重衰减；
#             'batch_size_s': 批量大小；'momentum_s': 动量；'shuffle_s': 训练时是否打乱数据集；
#             'optimizer_s': 优化器,支持['SGD', 'Adam']；
#             'loss_es': 损失函数，支持['l1', 'crossEntro', 'mse', 'huber']
#         :return:
#         """
#         assert len(self.__mlp_s__) == train_labels.shape[1], '训练标签集样本列数与输出向量长度不符！'
#         num_epochs, learning_rate_s, weight_decay_ies, batch_size_s, optimizer_s, loss_es, \
#             momentum_s, shuffle_s =
#         assert isinstance(learning_rate_s, Iterable) and \
#                isinstance(weight_decay_ies, Iterable) and isinstance(batch_size_s, Iterable) and \
#                isinstance(optimizer_s, Iterable) and isinstance(loss_es, Iterable) and \
#                isinstance(momentum_s, Iterable) and isinstance(shuffle_s, Iterable), \
#             '某个训练函数并非可迭代对象。训练参数均需要为可迭代对象，该对象为每个感知机的训练赋予参数'
#         train_ls, valid_ls = [], []
#         # 将训练标签集按列拆分
#         trainLabel_vector = [train_labels[:][i] for i in range(train_labels.shape[1])]
#         validLabel_vector = [valid_labels[:][i] for i in range(valid_labels.shape[1])]
#         # 进度条
#         # with tqdm(total=len(trainLabel_vector), unit='mlp', desc='Iterating') as mlp_pbar:
#         for _ in range(num_epochs):
#             cur_Ep_tls, cur_Ep_vls = 0, 0
#             for i, mlp in enumerate(self.__mlp_s__):
#                 t_ls, v_ls = mlp.train_(
#                     train_features, trainLabel_vector[i], valid_features, validLabel_vector[i],
#                     num_epochs=1, learning_rate=learning_rate_s[i], weight_decay=weight_decay_ies[i],
#                     batch_size=batch_size_s[i], optimizer=optimizer_s[i], loss=loss_es[i],
#                     momentum=momentum_s[i], shuffle=shuffle_s[i]
#                 )
#                 cur_Ep_tls += t_ls
#                 cur_Ep_vls += v_ls
#             train_ls.append(cur_Ep_tls / len(self.__mlp_s__))
#             valid_ls.append(cur_Ep_vls / len(self.__mlp_s__))
#         # with tqdm(self.__mlp_s__, desc='Iterating', unit='mlp') as mlp_pbar:
#         #     for i, mlp in mlp_pbar:
#         #         mlp.train_(
#         #             train_features, trainLabel_vector[i], valid_features, validLabel_vector[i],
#         #             num_epochs[i], learning_rate_s[i], weight_decay_ies[i], batch_size_s[i],
#         #             optimizer_s[i], loss_es[i], momentum_s[i], shuffle_s[i]
#         #         )
#         #     mlp_pbar.close()
#         return train_ls, valid_ls
#         # if valid_features is not None and valid_labels is not None:
#         #     return train_ls, valid_ls
#         # else:
#         #     return train_ls
#
#     def test(self, features, labels):
#         """
#         测试函数
#         :param features: 测试特征集
#         :param labels: 测试标签机
#         :return: 预测准确率
#         """
#         # features, labels = features.type(torch.float32), labels.type(torch.float32)
#         preds = self(features)
#         # y_hat, y = torch.round(preds, decimals=1), torch.round(labels, decimals=1)
#         y_hat, y = preds, labels
#         del labels, preds, features
#         correct = 0
#         for i in range(len(y_hat)):
#             if np.array_equal(y_hat[i], y[i]):
#                 correct += 1
#         acc = correct / len(y)
#         res = torch.hstack((y, y_hat))
#         return acc, res
#
#     @staticmethod
#     def load_array(features, labels, batch_size, num_epochs, shuffle=True):
#         """
#         加载训练数据
#         :param num_epochs: 迭代次数
#         :param features: 特征集
#         :param labels: 标签集
#         :param batch_size: 批量大小
#         :return 不断产出数据的迭代器
#         """
#         num_examples = features.shape[0]
#         assert batch_size <= num_examples, f'批量大小{batch_size}需要小于样本数量{num_examples}'
#         examples_indices = list(range(num_examples))
#         # random.shuffle(examples_indices)
#         # for i in range(0, num_examples, batch_size):
#         #     # 选取batch_size大小（若样本不足，则取完尾部）个样本输出
#         #     batch_indices = torch.tensor(
#         #         examples_indices[i: min(i + batch_size, num_examples)]
#         #     )
#         #     yield features[batch_indices], labels[batch_indices]
#         for _ in range(num_epochs):
#             if shuffle:
#                 random.shuffle(examples_indices)
#             # shuffled_features = features[examples_indices]
#             # shuffled_labels = labels[examples_indices]
#             # yield shuffled_features[:batch_size], shuffled_labels[:batch_size]
#             yield features[examples_indices][:batch_size], labels[examples_indices][:batch_size]
#             # del shuffled_features, shuffled_labels
#
#     # @staticmethod
#     # def unpack_parameters(pars):
#     #     """
#     #     获取超参数
#     #     :param pars: 用户输入的超参数
#     #     :return: 提取后的超参数
#     #     """
#     #     parameters = \
#     #         pars['num_epochs'] if 'num_epochs' in pars.keys() else 100, \
#     #         pars['learning_rate'] if 'learning_rate' in pars.keys() else 0.001, \
#     #         pars['weight_decay'] if 'weight_decay' in pars.keys() else 0.1, \
#     #         pars['batch_size'] if 'batch_size' in pars.keys() else 4, \
#     #         pars['optimizer'] if 'optimizer' in pars.keys() else 'SGD', \
#     #         MLPVector.__getLossFunc__(pars['loss']) if 'loss' in pars.keys() \
#     #         else MLPVector.__getLossFunc__(), \
#     #         pars['momentum'] if 'momentum' in pars.keys() else 0, \
#     #         pars['shuffle'] if 'shuffle' in pars.keys() else True
#     #     return parameters
#
#     # def __log_rmse__(self, features, labels, loss):
#     #     # 为了在取对数时进一步稳定该值，将小于1的值设置为1
#     #     # print(self(features))
#     #     # clipped_preds = torch.clamp(self(features), 1, float('inf'))
#     #     # rmse = torch.sqrt(loss(torch.log(clipped_preds),
#     #     #                        torch.log(labels)))
#     #     pred = self(features)
#     #     pred = torch.log(pred)
#     #     l = loss(pred, torch.log(labels))
#     #     rmse = torch.sqrt(l)
#     #     return rmse.item()
#
#     # @staticmethod
#     def __getOptim__(self, optim_str, learning_rate, weight_decay, momentum):
#         assert optim_str in optimizers, f'不支持优化器{optim_str}, 支持的优化器包括{optimizers}'
#         if optim_str == 'SGD':
#             return torch.optim.SGD(
#                 self.parameters(),
#                 lr=learning_rate,
#                 weight_decay=weight_decay,
#                 momentum=momentum
#             )
#         elif optim_str == 'Adam':
#             return torch.optim.Adam(
#                 self.parameters(),
#                 lr=learning_rate,
#                 weight_decay=weight_decay
#             )
#
#     @staticmethod
#     def __getActivation__(activ_str='ReLU'):
#         assert activ_str in activations, \
#             f'不支持激活函数{activ_str}, 支持的优化器包括{activations}'
#         if activ_str == 'ReLU':
#             return nn.ReLU(inplace=True)
#         elif activ_str == 'Sigmoid':
#             return nn.Sigmoid()
#         elif activ_str == 'Tanh':
#             return nn.Tanh()
#         elif activ_str == 'LeakyReLU':
#             return nn.LeakyReLU(inplace=True)
#
#     @staticmethod
#     def __getLossFunc__(loss_str='mse'):
#         assert loss_str in losses, \
#             f'不支持激活函数{loss_str}, 支持的优化器包括{losses}'
#         if loss_str == 'l1':
#             return nn.L1Loss()
#         elif loss_str == 'crossEntro':
#             return nn.CrossEntropyLoss()
#         elif loss_str == 'mse':
#             return nn.MSELoss()
#         elif loss_str == 'huber':
#             return nn.HuberLoss()
#
#     def __str__(self) -> str:
#         return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self.__device__)
#
#     def __call__(self, *args, **kwargs):
#         return None
