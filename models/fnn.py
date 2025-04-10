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

class fnnActor(nn.Module):
    def __init__(self, config):
        super(fnnActor, self).__init__()
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.model_info):
            lazy_layers[f'fnnblock_{i}'] = FnnBlock(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
        self.fnn_layers = nn.Sequential(lazy_layers)
        self.fc = nn.Linear(last_hidden_dim, 2)
        self.output_activation = nn.Sigmoid()  
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        x = self.fnn_layers(x)
        x = self.fc(x)
        x = self.output_activation(x)
        x[0]=x[0]*30
        x[1]=x[1]*150+50
        return x

class fnnCritic(nn.Module):
    def __init__(self, config):
        super(fnnCritic, self).__init__()
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.model_info):
            lazy_layers[f'fnnblock_{i}'] = FnnBlock(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
        self.fnn_layers = nn.Sequential(lazy_layers)
        self.fc = nn.Linear(last_hidden_dim, 1)
        
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        x = self.fnn_layers(x)
        x = self.fc(x)
        return x