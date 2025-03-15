import torch.nn as nn
from collections import  OrderedDict
from configs import config


class CnnBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0,drop_out=0.2):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop_out = nn.Dropout(drop_out)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop_out(x)
        return x

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, (hidden_dim,kernal_size) in enumerate(config.model_info):
            lazy_layers[f'cnnblock_{i}'] = CnnBlock(last_hidden_dim, hidden_dim, kernal_size)
            last_hidden_dim = hidden_dim
        self.cnn_layers = nn.Sequential(lazy_layers)
        self.fc = nn.Linear(last_hidden_dim, config.output_dim)
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, (hidden_dim,kernal_size) in enumerate(config.model_info):
            lazy_layers[f'cnnblock_{i}'] = CnnBlock(last_hidden_dim, hidden_dim, kernal_size)
            last_hidden_dim = hidden_dim
        self.cnn_layers = nn.Sequential(lazy_layers)
        self.fc = nn.Linear(last_hidden_dim, 1)
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x