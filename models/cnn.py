import torch.nn as nn
from collections import OrderedDict
import torch


class CnnBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, drop_out=0.2):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        #self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x) if x.shape[0] > 1 else x
        x = self.relu(x)
        x = self.pool(x) if x.shape[2] > 1 and x.shape[3] > 1 else x
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

        # 切换到评估模式以避免 BatchNorm 的统计计算问题
        # 切换到评估模式以避免 BatchNorm 的统计计算问题
        self.eval()
        with torch.no_grad():
            # 创建 dummy_input，形状为 ( config.max_sqe_len, config.input_channel, self.input_height, self.input_width)
            dummy_input = torch.zeros(1,config.max_sqe_len, config.input_channel, self.input_height, self.input_width)

            # 展平时间维度，形状变为 (batch_size * seq_len, channels, height, width)
            dummy_input = dummy_input.view(-1, config.input_channel, self.input_height, self.input_width)

            # 经过卷积层
            dummy_output = self.cnn_layers(dummy_input)

            # 恢复时间维度，形状变为 (batch_size, seq_len, feature_dim)
            dummy_output = dummy_output.view(1, config.max_sqe_len, -1)

            # 获取展平后的特征大小
            flattened_size = dummy_output.size(2)  # feature_dim
        self.train()  # 恢复到训练模式

        # 添加 LSTM 层
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=self.lstm_hidden_size, batch_first=True)

        # 全连接层
        self.fc_mu = nn.Linear(self.lstm_hidden_size, 2)
        self.fc_std = nn.Linear(self.lstm_hidden_size, 2)
        self.output_activation = nn.Sigmoid()

    def forward(self, x, h_state=None, c_state=None,istest=True):
        # 确保输入维度为 (batch_size, seq_len, channels, height, width)
        assert x.dim() == 5, "Input tensor must have 5 dimensions (batch, seq, channel, height, width)"
        
        x=torch.nan_to_num(x) if x is not None else x
        x = x.float()
        noise = torch.empty_like(x).uniform_(-1e-4, 1e-4)
        x = x + noise
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # 展平时间维度
        x = self.cnn_layers(x)
        x = x.view(batch_size, seq_len, -1)  # 恢复时间维度
        if torch.isnan(x).any():
            x = torch.zeros_like(x)
        noise = torch.empty_like(x).uniform_(-1e-4, 1e-4)
        x = x + noise

        # 初始化 h 和 c 为纯零
        if h_state is None or c_state is None:
            if istest:
                h_state = torch.zeros(1, 1, self.lstm_hidden_size, device=x.device)
                c_state = torch.zeros(1, 1, self.lstm_hidden_size, device=x.device)
            else:
                h_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
                c_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
                
        if torch.isnan(c_state).any():
            pass
        h_state = torch.nan_to_num(h_state) if h_state is not None else h_state
        c_state = torch.nan_to_num(c_state) if c_state is not None else c_state
        noise= torch.empty_like(h_state).uniform_(-1e-5, 1e-5)
        h_state = h_state + noise
        noise= torch.empty_like(c_state).uniform_(-1e-5, 1e-5)
        c_state = c_state + noise
        # LSTM 处理
        lstm_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        if torch.isnan(lstm_out).any():
            h_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
            c_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)

        # 输出均值和标准差
        mu_raw = self.output_activation(self.fc_mu(lstm_out))
        # 避免原地修改，创建新张量
        mu = torch.zeros_like(mu_raw)
        mu[:, 0] = mu_raw[:, 0] * 150 + 50
        mu[:, 1] = mu_raw[:, 1] * 30
        
        std = nn.functional.softplus(self.fc_std(lstm_out))
        return mu, std, (h_state, c_state)


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
        self.eval()
        with torch.no_grad():
            # 创建 dummy_input，形状为 ( config.max_sqe_len, config.input_channel, self.input_height, self.input_width)
            dummy_input = torch.zeros(1,config.max_sqe_len, config.input_channel, self.input_height, self.input_width)

            # 展平时间维度，形状变为 (batch_size * seq_len, channels, height, width)
            dummy_input = dummy_input.view(-1, config.input_channel, self.input_height, self.input_width)

            # 经过卷积层
            dummy_output = self.cnn_layers(dummy_input)

            # 恢复时间维度，形状变为 (batch_size, seq_len, feature_dim)
            dummy_output = dummy_output.view(1, config.max_sqe_len, -1)

            # 获取展平后的特征大小
            flattened_size = dummy_output.size(2)  # feature_dim
            
            # 获取展平后的特征大小
            flattened_size = dummy_output.size(2)  # feature_dim
        self.train()  # 恢复到训练模式

        # 添加 LSTM 层
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=self.lstm_hidden_size, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(self.lstm_hidden_size, 1)

    # def forward(self, x, h_state=None, c_state=None):
    #     # 确保输入维度为 (batch_size, seq_len, channels, height, width)
        
    #     batch_size, seq_len, c, h, w = x.size()
    #     x = x.view(batch_size * seq_len, c, h, w)  # 展平时间维度
    #     x = self.cnn_layers(x)
    #     x = x.view(batch_size, seq_len, -1)  # 恢复时间维度

    #     # 初始化 h 和 c 为纯零
    #     if h_state is None or c_state is None:

    #         h_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
    #         c_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
    #     h_state = torch.nan_to_num(h_state) if h_state is not None else h_state
    #     c_state = torch.nan_to_num(c_state) if c_state is not None else c_state

    #     # LSTM 处理
    #     lstm_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))

    #     # 取最后一个时间步的输出
    #     lstm_out = lstm_out[:, -1, :]

    #     # 输出值
    #     x = self.fc(lstm_out)
    #     return x[:, 0], (h_state, c_state)  # 返回一维张量和新的隐藏状态
    def forward(self, x, h_state=None, c_state=None):
        # 确保输入维度为 (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # 展平时间维度
        x = self.cnn_layers(x)
        x = x.view(batch_size, seq_len, -1)  # 恢复为 (batch, seq, features)

        # 初始化 LSTM 隐藏状态
        if h_state is None or c_state is None:
            h_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
            c_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)

        # LSTM 处理所有时间步
        lstm_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))  # lstm_out 形状: (batch, seq, hidden_size)

        # 为每个时间步生成值函数
        values = self.fc(lstm_out)          # 形状: (batch, seq, 1)
        values = values.squeeze(-1)        # 形状: (batch, seq)

        return values, (h_state, c_state)  # 返回所有时间步的值和隐藏状态
    

# 在原有Actor基础上新增 HighLevelManager
class HighLevelManager(nn.Module):
    """高层策略网络，生成子目标（连续向量）"""
    def __init__(self, config):
        super().__init__()
        # 共享底层CNN特征提取器（与Actor相同结构）
        self.cnn_layers = Actor(config).cnn_layers  # 直接复用Actor的CNN部分
        
        # LSTM参数
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden_size,  # 与Actor的LSTM hidden size一致
            hidden_size=config.manager_lstm_size,
            batch_first=True
        )
        
        # 子目标生成层
        self.goal_fc = nn.Sequential(
            nn.Linear(config.manager_lstm_size, 64),
            nn.Tanh(),
            nn.Linear(64, config.subgoal_dim)  # subgoal_dim=2 (示例维度)
        )

    def forward(self, x, h_state=None, c_state=None):
        # 输入x形状: (batch, seq_len, c, h, w)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # CNN特征提取（与Actor共享）
        x = x.view(batch_size * seq_len, *x.shape[2:])  # (batch*seq, c, h, w)
        cnn_features = self.cnn_layers(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # (batch, seq, features)
        
        # LSTM处理
        if h_state is None:
            h_state = torch.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            c_state = torch.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
        lstm_out, (h_state, c_state) = self.lstm(cnn_features, (h_state, c_state))
        
        # 生成子目标
        subgoal = self.goal_fc(lstm_out[:, -1, :])  # 取最后一个时间步
        return subgoal, (h_state, c_state)

# 修改原有Actor为LowLevelWorker
class LowLevelWorker(nn.Module):
    """底层策略网络，接收子目标生成动作"""
    def __init__(self, config):
        super().__init__()
        # 保持原有Actor结构，但修改输入维度
        self.cnn_layers = Actor(config).cnn_layers
        
        # 扩展LSTM输入：原特征 + 子目标
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden_size + config.subgoal_dim,  # 新增子目标维度
            hidden_size=config.lstm_hidden_size,
            batch_first=True
        )
        
        # 保持原有输出层
        self.fc_mu = nn.Linear(config.lstm_hidden_size, 2)
        self.fc_std = nn.Linear(config.lstm_hidden_size, 2)