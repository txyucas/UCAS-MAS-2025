import torch.nn as nn
from collections import  OrderedDict
from configs import config

class FnnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_cls=nn.ReLU, dropout=0.1):
        super(FnnBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = activation_cls()
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, data):
        data = self.fc(data)
        data = self.relu(data)
        data = self.dropout(data)
        return data

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.model_info):
            lazy_layers[f'fnnblock_{i}'] = FnnBlock(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
        self.fnn_layers = nn.Sequential(lazy_layers)
        self.fc = nn.Linear(last_hidden_dim, config.output_dim)
    def forward(self, x):
        x = self.fnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.model_info):
            lazy_layers[f'fnnblock_{i}'] = FnnBlock(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
        self.fnn_layers = nn.Sequential(lazy_layers)
        self.fc = nn.Linear(last_hidden_dim, 1)
    def forward(self, x):
        x = self.fnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x