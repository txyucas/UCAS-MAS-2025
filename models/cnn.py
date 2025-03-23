import torch.nn as nn
from collections import  OrderedDict
from configs import config
import torch


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
        last_hidden_channel = config.input_channel
        for i, (hidden_channel, kernel_size) in enumerate(config.model_info):
            lazy_layers[f'cnnblock_{i}'] = CnnBlock(last_hidden_channel, hidden_channel, kernel_size)
            last_hidden_channel = hidden_channel
        self.cnn_layers = nn.Sequential(lazy_layers)
        
        # 假设输入图像的空间维度为 config.input_height 和 config.input_width
        self.input_height = config.input_height
        self.input_width = config.input_width
        
        # 计算经过卷积层后的特征图的空间维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.input_channel, self.input_height, self.input_width)
            dummy_output = self.cnn_layers(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc = nn.Linear(flattened_size, 2)
        self.output_activation = nn.Sigmoid()  
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.output_activation(x)
        x[0]=x[0]*30
        x[1]=x[1]*150+50
        return x
    
class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        lazy_layers = OrderedDict()
        last_hidden_channel = config.input_channel
        for i, (hidden_channel, kernel_size) in enumerate(config.model_info):
            lazy_layers[f'cnnblock_{i}'] = CnnBlock(last_hidden_channel, hidden_channel, kernel_size)
            last_hidden_channel = hidden_channel
        self.cnn_layers = nn.Sequential(lazy_layers)
        
        # 假设输入图像的空间维度为 config.input_height 和 config.input_width
        self.input_height = config.input_height
        self.input_width = config.input_width
        
        # 计算经过卷积层后的特征图的空间维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.input_channel, self.input_height, self.input_width)
            dummy_output = self.cnn_layers(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc = nn.Linear(flattened_size, 1)
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x