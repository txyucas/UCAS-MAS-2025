import torch.nn as nn
from collections import OrderedDict
import torch


class CnnBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, drop_out=0.2):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if x.shape[0] > 1 else x
        x = self.relu(x)
        x = self.pool(x) if x.shape[2] > 1 and x.shape[3] > 1 else x
        x = self.drop_out(x)
        return x


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.config=config
        lazy_layers = OrderedDict()
        last_hidden_channel = config.input_channel
        for i, (hidden_channel, kernel_size) in enumerate(config.model_info):
            lazy_layers[f'cnnblock_{i}'] = CnnBlock(last_hidden_channel, hidden_channel, kernel_size)
            last_hidden_channel = hidden_channel
        self.cnn_layers = nn.Sequential(lazy_layers)

        # 假设输入图像的空间维度为 config.input_height 和 config.input_width
        self.input_height = config.input_height
        self.input_width = config.input_width

        # 切换到评估模式以避免 BatchNorm 的统计计算问题
        self.eval()
        with torch.no_grad():
            # 创建 dummy_input，形状为 (1, config.input_channel, self.input_height, self.input_width)
            dummy_input = torch.zeros(1, config.input_channel, self.input_height, self.input_width)
            # 经过卷积层
            dummy_output = self.cnn_layers(dummy_input).view(dummy_input.size(0), -1)  # 展平特征图
            # 获取展平后的特征大小
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
        # 确保输入维度为 (batch_size,  channels, height, width)
        assert x.dim() == 4, "Input tensor must have 4 dimensions (batch, channel, height, width)"
        x=x/10
        x = self.cnn_layers(x).view(x.size(0), -1)  # 展平特征图
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
        x= self.liner2(x)
        x= self.relu(x)
        mu_raw= self.output_activation(self.fc_mu(x))
        
        #去掉第一个维度
        mu_raw = mu_raw.squeeze(1)
        mu = mu_raw * torch.tensor([150, 30],device='cuda') + torch.tensor([50, 0],device='cuda') 
        std = nn.functional.softplus(self.fc_std(x))+self.config.min_std
        std=std*(0.99**self.config.total_step)
        std=std.squeeze(1)
        #std=torch.tensor([5,1],dtype=torch.float32).to(mu.device)
        return mu, std


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        lazy_layers = OrderedDict()
        self.config=config
        last_hidden_channel = config.input_channel
        for i, (hidden_channel, kernel_size) in enumerate(config.model_info):
            lazy_layers[f'cnnblock_{i}'] = CnnBlock(last_hidden_channel, hidden_channel, kernel_size)
            last_hidden_channel = hidden_channel
        self.cnn_layers = nn.Sequential(lazy_layers)

        # 假设输入图像的空间维度为 config.input_height 和 config.input_width
        self.input_height = config.input_height
        self.input_width = config.input_width

        # 计算经过卷积层后的特征图的空间维度
        self.eval()
        with torch.no_grad():
            # 创建 dummy_input，形状为 (1, config.input_channel, self.input_height, self.input_width)
            dummy_input = torch.zeros(1,config.input_channel, self.input_height, self.input_width)
            # 经过卷积层
            dummy_output = self.cnn_layers(dummy_input).view(dummy_input.size(0), -1)
            # 获取展平后的特征大小
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
        x=x/10
        x = self.cnn_layers(x).view(x.size(0), -1)  # 展平特征图
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
