import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from models.cnn import Actor, Critic
from torch.distributions import Categorical


import random
from collections import deque
class ReplayBufferQue:
    '''DQN的经验回放池，每次采样batch_size个样本'''
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

class PGReplay(ReplayBufferQue):
    '''PG的经验回放池，每次采样所有样本，因此只需要继承ReplayBufferQue，重写sample方法即可
    '''
    def __init__(self):
        self.buffer = deque()
    def sample(self):
        ''' sample all the transitions
        '''
        batch = list(self.buffer)
        return zip(*batch)

class SequentialPGReplay:
    '''专为序列数据设计的经验回放池，保持时间连续性'''
    def __init__(self, sequence_length=8):
        self.buffer = []
        self.sequence_length = sequence_length
        self.current_episode = []
        self.episodes = []
    
    def push(self, transition):
        '''存储一个转换，并构建完整回合
        
        Args:
            transition: (state, action, log_prob, reward, done) 元组
        '''
        # 对None值进行处理
        state, action, log_prob, reward, done = transition
        # 如果reward为None，设为0.0
        reward = 0.0 if reward is None else reward
        # 如果done为None，设为False
        done = False if done is None else done
        
        # 保存修正后的转换
        self.current_episode.append((state, action, log_prob, reward, done))
        
        # 如果回合结束，保存当前回合
        if done:  # 现在done已经确保是布尔值
            if len(self.current_episode) >= self.sequence_length:
                self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def sample(self):
        '''采样时序连续的完整序列'''
        if not self.episodes:
            return [], [], [], [], []
            
        # 生成用于训练的序列批次
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []
        
        # 从每个回合中提取连续序列
        for episode in self.episodes:
            # 如果回合长度足够，可以提取多个重叠序列
            if len(episode) >= self.sequence_length:
                # 从回合中提取连续序列（可重叠）
                max_start_idx = len(episode) - self.sequence_length
                start_idx = random.randint(0, max_start_idx)
                sequence = episode[start_idx:start_idx + self.sequence_length]
                
                states, actions, log_probs, rewards, dones = zip(*sequence)
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_log_probs.extend(log_probs)
                batch_rewards.extend(rewards)
                batch_dones.extend(dones)
        
        return batch_states, batch_actions, batch_log_probs, batch_rewards, batch_dones
    
    def clear(self):
        '''清空缓冲区'''
        self.episodes = []
        self.current_episode = []
        
    def __len__(self):
        return sum(len(episode) for episode in self.episodes)

class PPO_Agent:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, config, istest=True, actor_path=None, critic_path=None):
        self.actor = Actor(config).to(config.device)
        self.critic = Critic(config).to(config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=config.critic_lr)
        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.k_epochs = config.k_epochs  # 一条序列的数据用来训练轮数
        self.device = config.device
        
        self.eps_clip = config.eps_clip
        self.entropy_coef = config.entropy_coef # entropy coefficient
        self.update_freq = config.update_freq
        self.sample_count = 0

        # 历史状态缓存，用于构造序列输入
        self.max_sqe_len = config.max_sqe_len  # 最大序列长度
        self.history = torch.zeros( (1,self.max_sqe_len, 1, config.input_height, config.input_width), device=config.device)if istest==False else torch.zeros((1, self.max_sqe_len, 1, config.input_height, config.input_width), device=config.device)  # 初始化历史状态缓存
        self.memory = SequentialPGReplay(sequence_length=config.max_sqe_len)
        # 初始化 Actor 和 Critic 的 LSTM 隐藏状态和细胞状态
        self.actor_h = None
        self.actor_c = None
        self.critic_h = None
        self.critic_c = None
        
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            print(f"Actor model loaded from {actor_path}")
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"Critic model loaded from {critic_path}")

    def reset(self):
        """重置历史状态和 LSTM 隐藏状态"""
        self.history = torch.zeros_like(self.history)
        self.actor_h = None
        self.actor_c = None
        self.critic_h = None
        self.critic_c = None

    def sample_action(self, obs, reward=None, done=False):
        self.isnan=False
        self.sample_count += 1
        state = torch.tensor(obs['agent_obs'], dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)  # 添加 batch 维度
        sequence = self.update_history_sequence(state)
        
        # 生成动作分布
        mu, sigma, (self.actor_h, self.actor_c) = self.actor(sequence, self.actor_h, self.actor_c, istest=False)

            
        # 对分布参数进行约束
        sigma = torch.clamp(sigma, min=1e-3, max=1.0)

        # 检查分布参数是否异常
        if torch.isnan(mu).any() or torch.isnan(sigma).any() or torch.isinf(mu).any() or torch.isinf(sigma).any():
            self.isnan=True
            print("警告: 动作分布包含异常值，重置为默认值")
            mu = torch.zeros_like(mu)
            sigma = torch.ones_like(sigma) * 0.1

        # 根据分布采样动作
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=1)

        action_np = action.cpu().numpy()
        
        # 添加截断逻辑
        action_np[:, 1] = np.clip(action_np[:, 1], -30, 30)
        action_np[:, 0] = np.clip(action_np[:, 0], -100, 200)
        
        # 存储转换数据到序列缓冲区
        # 使用提供的奖励和完成标志，如果未提供则使用默认值
        reward = 0.0 if reward is None else reward
        self.memory.push((obs, action.tolist()[0], log_prob, reward, done))
        
        # 保存log_prob供外部使用
        self.log_probs = log_prob
        
        return action_np.tolist()

    @torch.no_grad()
    def act(self, obs):
        state = torch.tensor(obs['agent_obs'], dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)  # 添加 batch 维度
        sequence = self.update_history_sequence(state)
        mu, sigma, (self.actor_h, self.actor_c) = self.actor(sequence, self.actor_h, self.actor_c)
        self.isnan=False
        if torch.isnan(mu).any() or torch.isnan(sigma).any() or torch.isinf(mu).any() or torch.isinf(sigma).any():
            self.isnan=True
            print("警告: 动作分布包含异常值，重置为默认值")
            mu = torch.zeros_like(mu)
            sigma = torch.ones_like(sigma) * 0.1
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = action.cpu().numpy()
        
        
        # 添加截断逻辑
        action[0, 1] = np.clip(action[0, 1], -30, 30)  # 第一维范围限制为 [-30, 30]
        action[0, 0] = np.clip(action[0,0], -100, 200)  # 第二维范围限制为 [-100, 200]
        
        return action[0].tolist()

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update_history_sequence(self, states):
        """
        更新 self.history 并返回包含历史序列的张量。
        
        参数:
            states (torch.Tensor): 当前的状态序列，形状为 (batch_size, height, width) 或 (batch_size, channels, height, width)。
        
        返回:
            torch.Tensor: 包含历史序列的张量，形状为 (batch_size, seq_len, channels, height, width)。
        """
        if  states==None:
            return self.history  # 如果状态为None，直接返回历史
        elif  len(states.shape) == 1 or len(states.shape) == 2:
            return self.history  # 如果状态是单一维度，直接返回历史
        elif len(states.shape) == 3:
            states = states.unsqueeze(1)  # 添加 channels 维度，形状变为 (batch_size, 1, height, width)

        batch_size, channels, height, width = states.shape
        
        # 检查批次大小是否匹配
        if batch_size != self.history.size(0):
            # 创建新的历史记录，匹配当前批次大小
            new_history = torch.zeros((batch_size, self.max_sqe_len, channels, height, width), 
                                    device=self.device)
                                    
            # 区分不同场景的历史处理
            if self.history.size(0) == 1 and batch_size > 1:
                # 从单样本交互转为批处理训练：复制历史给每个批次样本
                for i in range(batch_size):
                    new_history[i] = self.history[0]
            elif self.history.size(0) > 1 and batch_size == 1:
                # 从批处理训练转回单样本交互：使用第一个批次的历史
                new_history[0] = self.history[0]
                
            self.history = new_history
        
        # 更新历史序列
        self.history = torch.roll(self.history, shifts=-1, dims=1)  # 将历史序列向左滚动
        self.history[:, -1] = states  # 更新最新的状态到历史序列的最后一位

        return self.history

    def update(self):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return

        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()

        if not old_states:  # 如果没有足够的序列数据
            return

        # 转换状态为张量
        old_states = [state['agent_obs'] for state in old_states]
        old_states_array = np.array(old_states)

        # 计算序列长度和批次大小
        seq_len = self.max_sqe_len
        batch_size = len(old_states) // seq_len

        if batch_size == 0:
            print("警告: 数据不足以形成完整批次")
            return

        # 重塑状态张量
        if len(old_states_array.shape) == 3:  # (batch*seq, height, width)
            h, w = old_states_array.shape[1:]
            old_states = torch.tensor(old_states_array, device=self.device, dtype=torch.float)
            old_states = old_states.view(batch_size, seq_len, 1, h, w)
        elif len(old_states_array.shape) == 4:  # (batch*seq, channels, height, width)
            c, h, w = old_states_array.shape[1:]
            old_states = torch.tensor(old_states_array, device=self.device, dtype=torch.float)
            old_states = old_states.view(batch_size, seq_len, c, h, w)

        # 处理其他张量
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float)
        old_actions = old_actions.view(batch_size, seq_len, -1)
        old_rewards = torch.tensor(old_rewards, device=self.device, dtype=torch.float)
        old_dones = torch.tensor(old_dones, device=self.device, dtype=torch.float)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float)
        old_log_probs = old_log_probs.view(batch_size, seq_len)

        # 梯度累积参数
        accumulation_steps = self.k_epochs  # 将一个大批次分成多个小批次
        mini_batch_size = max(1, batch_size // accumulation_steps)

        for e in range(self.k_epochs):
            # 重置梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            for i in range(accumulation_steps):
                # 获取小批次数据
                start_idx = i * mini_batch_size
                end_idx = start_idx + mini_batch_size
                mini_states = old_states[start_idx:end_idx]
                mini_actions = old_actions[start_idx:end_idx]
                mini_rewards = old_rewards[start_idx:end_idx]
                mini_log_probs = old_log_probs[start_idx:end_idx]

                # 跳过空批次
                if mini_states.size(0) == 0:
                    continue

                # 前向传播 Critic
                values, _ = self.critic(mini_states)
                values = values.view(-1)
                flat_rewards = mini_rewards.view(-1)
                advantage = flat_rewards - values.detach()

                # 前向传播 Actor
                mu, sigma, _ = self.actor(mini_states, istest=False)
                self.isnan=False
                if torch.isnan(mu).any() or torch.isnan(sigma).any() or torch.isinf(mu).any() or torch.isinf(sigma).any():
                    self.isnan=True
                    print("警告: 动作分布包含异常值，重置为默认值")
                    mu = torch.zeros_like(mu)
                    sigma = torch.ones_like(sigma) * 0.1
                mu = mu.view(-1, mu.size(-1))
                sigma = sigma.view(-1, sigma.size(-1))
                flat_actions = mini_actions.view(-1, mini_actions.size(-1))

                dist = torch.distributions.Normal(mu, sigma)
                new_probs = dist.log_prob(flat_actions).sum(dim=1)
                flat_old_log_probs = mini_log_probs.view(-1)

                # 计算损失
                ratio = torch.exp(new_probs - flat_old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
                critic_loss = (flat_rewards - values).pow(2).mean()

                # 累积梯度
                total_loss = actor_loss + critic_loss
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

            # 更新参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 清空记忆缓冲区
        self.memory.clear()

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Model saved to {actor_path} and {critic_path}")
