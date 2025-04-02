import torch.nn as nn
from collections import OrderedDict
import torch
from collections import deque
from typing import List, Dict, Tuple
import random


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
        self.config = config
        self.device = config.device

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

    def forward(self, x, h_state=None, c_state=None):
        # 确保输入维度为 (batch_size, seq_len, channels, height, width)
        
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # 展平时间维度
        x = self.cnn_layers(x)
        x = x.view(batch_size, seq_len, -1)  # 恢复时间维度

        # 初始化 h 和 c 为纯零
        if h_state is None or c_state is None:

            h_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
            c_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=x.device)
        h_state = torch.nan_to_num(h_state) if h_state is not None else h_state
        c_state = torch.nan_to_num(c_state) if c_state is not None else c_state

        # LSTM 处理
        lstm_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 输出值
        x = self.fc(lstm_out)
        return x[:, 0], (h_state, c_state)  # 返回一维张量和新的隐藏状态
    
class SelfPlayManager:
    def __init__(
        self, 
        env,                   # 环境实例
        config,                # 配置参数对象
        device: str = "cuda",  # 计算设备
        pool_capacity: int = 5,# 对手池容量
        opponent_prob: float = 0.7  # 选择历史对手的概率
    ):
        self.env = env
        self.device = device
        self.config = config
        self.pool_capacity = pool_capacity
        self.opponent_prob = opponent_prob
        
        # 初始化对手池
        self.opponent_pool = deque(maxlen=pool_capacity)
        initial_policy = self._create_policy()
        self.opponent_pool.append(initial_policy)
        
    def _create_policy(self) -> Actor:
        """创建新策略实例并移至指定设备"""
        policy = Actor(self.config).to(self.device)
        return policy
    
    def _clone_policy(self, source: Actor) -> Actor:
        """克隆策略参数"""
        cloned = self._create_policy()
        cloned.load_state_dict(source.state_dict())
        return cloned
    
    def update_pool(self, new_policy: Actor):
        """更新对手池"""
        cloned = self._clone_policy(new_policy)
        self.opponent_pool.append(cloned)
        
    def select_opponent(self) -> Actor:
        """选择对手策略：有概率选择当前策略"""
        if random.random() < self.opponent_prob and len(self.opponent_pool) > 1:
            return random.choice(list(self.opponent_pool)[:-1])  # 排除最新策略
        else:
            return self.opponent_pool[-1]  # 最新策略即当前策略
        
    def run_episode(
        self, 
        current_policy: Actor, 
        max_steps: int = 200,
        render: bool = False
    ) -> List[Dict]:
        """运行单次自博弈对局"""
        opponent = self.select_opponent()
        state = self.env.reset()
        trajectory = []
        h_state, c_state = None, None  # LSTM初始状态
        
        for _ in range(max_steps):
            # 当前策略动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                mu, std, (h_next, c_next) = current_policy(
                    state_tensor, h_state, c_state
                )
                action = torch.normal(mu, std).squeeze(0).cpu().numpy()
                
            # 环境执行动作
            next_state, reward, done, _ = self.env.step(action)
            
            # 对手策略动作
            with torch.no_grad():
                opp_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
                opp_mu, opp_std, _ = opponent(opp_state_tensor, None, None)
                opp_action = torch.normal(opp_mu, opp_std).squeeze(0).cpu().numpy()
                
            # 环境二次交互
            next_state, opp_reward, done, _ = self.env.step(opp_action)
            
            # 存储轨迹
            trajectory.append({
                "state": state.copy(),
                "action": action,
                "opp_action": opp_action,
                "reward": reward - opp_reward,  # 差分奖励
                "h_state": h_state.clone() if h_state is not None else None,
                "c_state": c_state.clone() if c_state is not None else None,
                "done": done
            })
            
            # 更新状态
            state = next_state.copy()
            h_state, c_state = h_next.detach(), c_next.detach()
            
            if render:
                self.env.render()
                
            if done:
                break
                
        return trajectory
    
    def evaluate(
        self, 
        current_policy: Actor, 
        num_episodes: int = 10
    ) -> float:
        """评估当前策略的胜率"""
        win_count = 0
        
        for _ in range(num_episodes):
            trajectory = self.run_episode(current_policy, render=False)
            final_reward = sum(t["reward"] for t in trajectory)
            if final_reward > 0:  # 根据环境定义调整胜利条件
                win_count += 1
                
        return win_count / num_episodes
    
    def save_pool(self, path: str):
        """保存对手池"""
        torch.save({
            i: policy.state_dict() for i, policy in enumerate(self.opponent_pool)
        }, path)
        
    def load_pool(self, path: str):
        """加载对手池"""
        checkpoint = torch.load(path, map_location=self.device)
        self.opponent_pool = deque(maxlen=self.pool_capacity)
        for i in checkpoint.keys():
            policy = self._create_policy()
            policy.load_state_dict(checkpoint[i])
            self.opponent_pool.append(policy)