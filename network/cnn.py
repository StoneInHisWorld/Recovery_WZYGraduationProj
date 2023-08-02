from mlp import MultiLayerPerception
from utils.decorators import init_net
from torch import nn


class CNN(nn.Sequential):

    @init_net
    def __init__(self, in_features: int, out_features: int, parameters=None):
        activation, dropout, base = parameters
        kernel_size = (3, 3)
        layers = [
            nn.Conv2d(in_features, 16, kernel_size=kernel_size, ),
            nn.Conv2d(16, 16)
        ]
        super().__init__()
