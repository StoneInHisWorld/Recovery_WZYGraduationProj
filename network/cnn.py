import torch
from torch import nn
from tqdm import tqdm

import utils.tool_func as tools
from network.basic_nn import BasicNN
from utils.decorators import init_net, unpack_kwargs


class CNN(BasicNN):
    init_args = {
        'para_init': ('zero', ['normal', 'xavier', 'zero']),
        'activation': ('ReLU', ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']),
        'dropout': (0, []),
        'in_channels': ([1 for _ in range(8)], []),
        'out_channels': ([1 for _ in range(8)], []),
        'kernel_sizes': ([(3, 3) for _ in range(8)], []),
        'strides': ([2 for _ in range(8)], []),
        'bn_shapes': (None, []),
        'momentums': ([0.95 for _ in range(7)], []),
        'paddings': ([0 for _ in range(8)], [])
    }

    @init_net(init_args)
    def __init__(self, out_features: tuple[int, int], dummies_columns, parameters=None):
        activation, dropout, in_channels, out_channels, kernel_sizes, strides, \
            bn_shapes, momentums, paddings = parameters
        self.dummies_columns = dummies_columns
        # 加入卷积层与激活层
        i = 0
        layers = [
            nn.Conv2d(in_channels[0], out_channels[0], kernel_sizes[0], strides[0],
                      paddings[0]),
            tools.get_activation(activation),
            # nn.Conv2d(in_channels[1], out_channels[1], kernel_sizes[1], strides[1]),
            # tools.get_activation(activation),
        ]
        # layers = []
        # 加入附带BN层的卷积层
        start = 1
        for i in range(start, len(in_channels)):
            layers.append(
                nn.Conv2d(in_channels[i], out_channels[i], kernel_sizes[i], strides[i],
                          paddings[i])
            )
            layers.append(
                tools.get_activation(activation)
            )
            layers.append(
                nn.BatchNorm1d(bn_shapes[i - start], momentum=momentums[i - start])
            )
        # 附带BN层的最大池化层
        layers.append(
            nn.BatchNorm1d(bn_shapes[-1], momentum=momentums[-1])
        )
        layers.append(nn.MaxPool2d(bn_shapes[-1]))
        # if dropout > 0:
        #     layers.append(nn.Dropout(dropout))
        # self.output_nn1 = nn.Sequential(
        #     nn.Linear(bn_shapes[-1], out_features[0]),
        #     nn.Dropout(dropout),
        #     nn.Softmax()
        # )
        # self.output_nn2 = nn.Sequential(
        #     nn.Linear(bn_shapes[-1], out_features[1]),
        #     nn.Dropout(dropout),
        #     nn.Softmax()
        # )
        layers.append(
            DualOutputLayer(
                out_channels[-1], out_features[0], out_features[1],
                # 1, out_features[0], out_features[1],
                para_init='xavier', dropout=dropout
            )
        )
        # layers.append(self.output_nn1)
        # layers.append(self.output_nn2)
        super().__init__(*layers)

    train_args = {
        'num_epochs': (100, []),
        'learning_rate': (0.001, []),
        'weight_decay': (0.1, []),
        'batch_size': (4, []),
        'loss': ('mse', ['l1', 'crossEntro', 'mse', 'huber']),
        'momentum': (0, []),
        'shuffle': (True, [True, False]),
        'optimizer': ('SGD', ['SGD', 'Adam'])
    }

    @unpack_kwargs(allow_args=train_args)
    def train_(self, train_features: torch.Tensor, train_labels: torch.Tensor,
               valid_features: torch.Tensor = None,
               valid_labels: torch.Tensor = None, parameters=None):
        """
        训练函数
        :param train_features: 进行训练的样本特征集
        :param train_labels: 进行训练的样本标签集
        :param valid_features: 进行验证的特征集（可选）
        :param valid_labels: 进行验证的标签集（可选）
        :param parameters: 额外参数。可选：
            'num_epochs': 迭代次数
            'learning_rate': 学习率
            'weight_decay': 权重衰减
            'batch_size': 批量大小
            'loss': 损失函数，支持['l1', 'crossEntro', 'mse', 'huber']
            'momentum': 动量
            'shuffle': 训练时是否打乱数据集
            'optimizer': 优化器,支持['SGD', 'Adam']
        :return: 训练损失列表（记录每次迭代的损失值），验证损失列表
        """
        num_epochs, learning_rate, weight_decay, batch_size, loss, momentum, shuffle, \
            optimizer = parameters
        loss = tools.get_lossFunc(loss)
        optimizer = tools.get_optimizer(
            self, optim_str=optimizer, learning_rate=learning_rate,
            weight_decay=weight_decay, momentum=momentum
        )
        train_ls, valid_ls = [], []
        train_iter = tools.load_array(
            train_features, train_labels, batch_size, num_epochs, shuffle
        )
        print('优化器参数：', optimizer.state_dict)
        print('使用的损失函数：', loss)
        print(f'训练batch大小：{batch_size}')
        print(f'是否打乱数据集：{shuffle}')
        # 进度条
        with tqdm(total=num_epochs, unit='epoch', desc='Iterating') as pbar:
            for X_batch, y_batch in train_iter:
                train_bls = torch.zeros(1)
                valid_bls = torch.zeros(1)
                # 对batch中的数据逐条计算，并计算损失
                for X, y in zip(X_batch, y_batch):
                    self.train()
                    optimizer.zero_grad()
                    # 正向传播并计算损失
                    X = X.reshape((1, *X.shape))  # 将X变为三维数据
                    l = loss(self(X), y)
                    # 输出所有层的输出值
                    # with torch.no_grad():
                    #     print('\nres', self(X))
                    # 反向传播
                    l.backward(retain_graph=True)
                    optimizer.step()
                    # 记录批量损失
                    with torch.no_grad():
                        train_bls += l.item()
                        if valid_features is not None and valid_labels is not None:
                            valid_bls += self.valid_(
                                valid_features, valid_labels, loss
                            )
                        # if valid_features is not None and valid_labels is not None:
                        #     self.eval()
                        #     valid_ls.append(
                        #         loss(self(valid_features), valid_labels).item()
                        #     )
                # 记录损失
                train_ls.append(train_bls / batch_size)
                if valid_features is not None and valid_labels is not None:
                    valid_ls.append(valid_bls / batch_size)
                # 更新进度条
                pbar.update(1)
            pbar.close()
        return train_ls, valid_ls

    def valid_(self, features, labels, loss):
        self.eval()
        l = torch.zeros(1, device=self.__device__)
        with torch.no_grad():
            for feature, label in zip(features, labels):
                feature = feature.reshape((1, *feature.shape))
                l += loss(self(feature), label)
        return (l / features.shape[0]).item()

    def predict(self, features):
        with torch.no_grad():
            # 得到预测值
            preds = []
            for f in features:
                f = f.reshape((1, *f.shape))
                preds.append(self(f))
            preds = torch.vstack(preds)
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


class DualOutputLayer(nn.Module):
    init_args = {
        'para_init': ('zero', ['normal', 'xavier', 'zero']),
        'dropout': (0, []),
    }

    @init_net(init_args)
    def __init__(self, in_features, fir_out, sec_out, parameters=None) -> None:
        dropout_rate, = parameters
        super().__init__()
        fir = nn.Sequential(
            nn.Linear(in_features, fir_out),
            nn.Dropout(dropout_rate),
            nn.Softmax(dim=1)
        )
        sec = nn.Sequential(
            nn.Linear(in_features, sec_out),
            nn.Dropout(dropout_rate),
            nn.Softmax(dim=1)
        )
        self.add_module('fir', fir)
        self.add_module('sec', sec)
        # self.fir_weight = nn.Parameter(torch.randn((in_features, fir_out)))
        # self.fir_bias = nn.Parameter(torch.randn((fir_out, )))
        # self.sec_weight = nn.Parameter(torch.randn((in_features, sec_out)))
        # self.sec_bias = nn.Parameter(torch.randn((sec_out, )))
        # nn.init.xavier_uniform_(fir.weight)
        # nn.init.xavier_uniform_(fir.bias)
        # nn.init.xavier_uniform_(sec.weight)
        # nn.init.xavier_uniform_(sec.bias)
        # self.__fir__ = fir
        # self.__sec__ = sec

    def forward(self, features):
        in_features_es = [
            next(child.children()).in_features
            for _, child in self
        ]
        features_es = [
            features.reshape((1, shape))
            for shape in in_features_es
        ]
        fir_in, sec_in = features_es
        # children = self.named_children()
        fir_out, sec_out = self[0](fir_in).T, self[1](sec_in).T
        # res = [
        #     child(features_es[i])
        #     for i, (_, child) in enumerate(self.named_children())
        # ]
        # return torch.hstack(res).T
        vector = torch.vstack((fir_out, sec_out))
        return vector.reshape((fir_out.shape[0] + sec_out.shape[0], ))

    def __iter__(self):
        return self.named_children()

    def __getitem__(self, item: int):
        children = self.named_children()
        for _ in range(item):
            next(children)
        return next(children)[1]
