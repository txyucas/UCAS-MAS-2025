import torch.nn as nn
from collections import  OrderedDict
from configs import config
import torch


class FnnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_cls=nn.ReLU, dropout=0.1):
        super(FnnBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = activation_cls()
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, data):
        
        data = self.fc(data)
        data = self.bn(data)
        data = self.relu(data)
        data = self.dropout(data)
        return data

class fnnActor(nn.Module):
    def __init__(self, config):
        super(fnnActor, self).__init__()
        self.config=config
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.model_info):
            lazy_layers[f'fnnblock_{i}'] = FnnBlock(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
        self.fnn_layers = nn.Sequential(lazy_layers)

        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.input_dim)
            dummy_output = self.fnn_layers(dummy_input).view(dummy_input.size(0), -1)  # 展平特征图
            flattened_size = dummy_output.size(1)  # feature_dim
        self.train()  # 恢复到训练模式

        # 添加 LSTM 层或rnn
        if config.rnn_or_lstm == 'rnn':
            self.rnn = nn.RNN(input_size=flattened_size, hidden_size=config.lstm_hidden_size, batch_first=True)
            self.linersize=config.lstm_hidden_size
        elif config.rnn_or_lstm == 'lstm':
            self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=config.lstm_hidden_size, batch_first=True)
            self.linersize=config.lstm_hidden_size
        else:
            self.linersize=flattened_size           

        # 全连接层
        self.liner1=nn.Linear(self.linersize, self.linersize)
        self.liner2=nn.Linear(self.linersize, self.linersize)
        self.fc_mu = nn.Linear(self.linersize, 2)
        self.fc_std = nn.Linear(self.linersize, 2)
        self.output_activation = nn.Tanh()
        self.relu= nn.ReLU()

    def forward(self, x, h_state=None, c_state=None,istest=True):
        
        x = x.view(x.size(0), -1)
        x = self.fnn_layers(x)
        x = x.unsqueeze(1)

        if self.config.rnn_or_lstm == 'rnn':
            x,h_state=self.rnn(x,h_state)
            mu, std = self._get_forward(x)
            return mu, std, h_state
            
        elif self.config.rnn_or_lstm == 'lstm':
            x,(h_state, c_state) = self.lstm(x, (h_state, c_state))
            mu, std = self._get_forward(x)
            return mu, std, (h_state, c_state)
        else:
            mu, std = self._get_forward(x)
            return mu, std

    def _get_forward(self, x):
        x= self.liner1(x)
        x= self.relu(x)
        x= self.liner2(x)+x
        x= self.relu(x)
        mu_raw= self.output_activation(self.fc_mu(x))
        
        #去掉第一个维度
        mu_raw = mu_raw.squeeze(1)
        mu = mu_raw * torch.tensor([150, 30],device='cuda') + torch.tensor([50, 0],device='cuda') 
        std = nn.functional.softplus(self.fc_std(x))
        std=std.squeeze(1)
        #std=torch.tensor([5,1],dtype=torch.float32).to(mu.device)
        return mu, std

class fnnCritic(nn.Module):
    def __init__(self, config):
        super(fnnCritic, self).__init__()
        self.config=config
        lazy_layers = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.model_info):
            lazy_layers[f'fnnblock_{i}'] = FnnBlock(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
        self.fnn_layers = nn.Sequential(lazy_layers)

        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.input_dim)
            dummy_output = self.fnn_layers(dummy_input).view(dummy_input.size(0), -1)
            flattened_size = dummy_output.size(1)  # feature_dim
        self.train()  # 恢复到训练模式

        # 添加 LSTM 层或rnn
        if config.rnn_or_lstm == 'rnn':
            self.rnn = nn.RNN(input_size=flattened_size, hidden_size=config.lstm_hidden_size, batch_first=True)
            self.linersize=config.lstm_hidden_size
        elif config.rnn_or_lstm == 'lstm':
            self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=config.lstm_hidden_size, batch_first=True)
            self.linersize=config.lstm_hidden_size
        else:
            self.linersize=flattened_size           

        # 全连接层
        self.liner=nn.Linear(self.linersize, self.linersize)
        self.relu= nn.ReLU()
        self.fc = nn.Linear(self.linersize, 1)
        
    def forward(self, x, h_state=None, c_state=None):
        
        x = x.view(x.size(0), -1)
        x = self.fnn_layers(x)
        x = x.unsqueeze(1)
        if self.config.rnn_or_lstm == 'rnn':
            x, h_state = self.rnn(x, h_state)
            values=self._get_values(x)
            return values, h_state
        elif self.config.rnn_or_lstm == 'lstm':
            x, (h_state, c_state) = self.lstm(x, (h_state, c_state))
            values = self._get_values(x)
            return values, (h_state, c_state)
        else:
            values = self._get_values(x)
            return values

    def _get_values(self, x):
        x = self.liner(x)
        x= self.relu(x)
        values = self.fc(x)
        values = values.squeeze(-1)
        return values