import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from models.cnn import Actor, Critic
from torch.distributions import Categorical
import os
import random
from collections import deque


class ReplayBufferQue:
    '''DQN的经验回放池，每次采样batch_size个样本'''
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity  
        self.buffer = deque(maxlen=self.capacity)
        # buffer是一个队列，里面的元素是一个五元组(obs,action,log_prob,reward,done)
        # 其中log_prob:动作的对数概率（重要性采样关键）
        # done:回合是否终止
    
    # 添加单条经验
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)
    
    def sample(self, batch_size: int, sequential: bool = False):
        #保证采样个数不大于回放池中的样本数
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling  顺序采样，连采batchsize个
            rand = random.randint(0, len(self.buffer) - batch_size) # 这是通过限制采样的起始点来确保采样终点的索引在范围内
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
            # 返回一个元组，元组的每一个元素是一种属性的列表
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

# 专门策略梯度算法设计的回放池，将采样全部经验
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
        self.sequence_length = sequence_length    #每个训练序列的长度（如8帧）
        self.current_episode = []    #当前正在收集的回合，临时存储当前回合的连续经验。
        self.episodes = []     # 已完成的完整回合列表，只保存长度 ≥ sequence_length 的完整回合。他是一个列表，元素是一个回合的数据是一个列表，列表的每个元素是五元组。
    
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
        # 如果self.episodes为空，没法采样，直接返回空列表
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
                # 从回合中提取连续序列（可重叠），这里就是顺序采样了，因为要的是连续的数据
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
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
            eps=1e-5,  # 增加数值稳定性
            weight_decay=1e-5  # 添加权重衰减，L2正则化系数，防止过拟合（真的要防止吗？）
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
            eps=1e-5,  # 增加数值稳定性
            weight_decay=1e-5  # 添加权重衰减
        )
        self.gamma = config.gamma   #折扣因子
        self.lmbda = config.lmbda   #GAE参数
        self.k_epochs = config.k_epochs  # 一条序列的数据用来训练轮数
        self.device = config.device
        
        self.eps_clip = config.eps_clip     #PPO截断阈值
        self.entropy_coef = config.entropy_coef # entropy coefficient， 熵奖励系数
        self.update_freq = config.update_freq #策略更新频率
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
        # 这说明了如果一个模型很烂，得删掉重新训
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            print(f"Actor model loaded from {actor_path}")
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"Critic model loaded from {critic_path}")

    def reset(self):
        """增强型重置，彻底清除历史状态"""
        self.history = torch.zeros_like(self.history)
        self.actor_h = None
        self.actor_c = None
        self.critic_h = None
        self.critic_c = None
        # 清空内存缓冲区
        self.memory.clear()
        # 重置优化器状态
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_optimizer.param_groups[0]['lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_optimizer.param_groups[0]['lr'])

    def sample_action(self, obs, reward=None, done=False):
        self.isnan=False
        self.sample_count += 1    # 统计采样次数，用于控制策略更新频率（如每N步更新一次）。
        state = torch.tensor(obs['agent_obs'], dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)  # 添加 batch 维度
        sequence = self.update_history_sequence(state) # 更新历史状态序列（用于时序模型如LSTM），返回包含最近 max_sqe_len 个状态的序列。
        
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

    # @torch.no_grad()
    # def act(self, obs):
    #     state = torch.tensor(obs['agent_obs'], dtype=torch.float).to(self.device)
    #     state = state.unsqueeze(0)  # 添加 batch 维度
    #     sequence = self.update_history_sequence(state)
    #     mu, sigma, (self.actor_h, self.actor_c) = self.actor(sequence, self.actor_h, self.actor_c)
    #     self.isnan=False
    #     if torch.isnan(mu).any() or torch.isnan(sigma).any() or torch.isinf(mu).any() or torch.isinf(sigma).any():
    #         self.isnan=True
    #         print("警告: 动作分布包含异常值，重置为默认值")
    #         mu = torch.zeros_like(mu)
    #         sigma = torch.ones_like(sigma) * 0.1
    #     action_dist = torch.distributions.Normal(mu, sigma)
    #     action = action_dist.sample()
    #     action = action.cpu().numpy()
        
        
        # # 添加截断逻辑
        # action[0, 1] = np.clip(action[0, 1], -30, 30)  # 第一维范围限制为 [-30, 30]
        # action[0, 0] = np.clip(action[0,0], -100, 200)  # 第二维范围限制为 [-100, 200]
        
        # return action[0].tolist()
        
    def act(self, obs, test_mode=False, exploration_scale=0.3):
        """
        改进版动作采样函数，针对稀疏奖励和连续动作空间优化
        参数：
            obs: 环境观测字典，必须包含'agent_obs'键
            test_mode: 测试模式开关，True时关闭探索噪声
            exploration_scale: 初始探索强度系数(0.1~0.5)
        返回：
            list: 2D动作 [action_dim1, action_dim2]
        """
        # 状态预处理
        state = torch.tensor(obs['agent_obs'], dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)  # 添加batch维度 [1, ...]
        
        # 获取动作分布
        with torch.no_grad():
            mu, sigma, (self.actor_h, self.actor_c) = self.actor(
                self.update_history_sequence(state),
                self.actor_h, self.actor_c
            )
            
            # 约束标准差范围并检查异常
            sigma = torch.clamp(sigma, min=1e-3, max=1.0)
            if torch.isnan(mu).any() or torch.isnan(sigma).any():
                self.isnan = True
                print("警告: 动作分布异常，使用安全动作")
                mu = torch.zeros_like(mu)
                sigma = torch.ones_like(sigma) * 0.1
            else:
                self.isnan = False

        # 稀疏奖励环境专用探索策略
        if not test_mode:
            # 自适应探索噪声（训练步数越多噪声越小）
            decay_factor = max(0.1, 1 - self.sample_count / 2e6)  # 200万步衰减到10%
            noise = exploration_scale * decay_factor * torch.randn_like(mu) * sigma
            
            # 对关键维度加强探索（示例：第2维度）
            noise[:, 1] *= 1.5  # 第2维度的探索强度增加50%
            
            action = mu + noise
        else:
            action = mu  # 测试模式直接使用均值

        # 动作截断（根据环境物理限制）
        action = action.cpu().numpy()
        action[0, 0] = np.clip(action[0, 0], -100, 200)  # 第一维限制
        action[0, 1] = np.clip(action[0, 1], -30, 30)    # 第二维限制
        
        return action[0].tolist()

    # def compute_advantage(self, gamma, lmbda, td_delta):
    #     td_delta = td_delta.detach().numpy()
    #     advantage_list = []
    #     advantage = 0.0
    #     for delta in td_delta[::-1]:
    #         advantage = gamma * lmbda * advantage + delta
    #         advantage_list.append(advantage)
    #     advantage_list.reverse()
    #     return torch.tensor(advantage_list, dtype=torch.float)
    
    def compute_advantage(self, rewards, values, dones, last_value):
        """实现广义优势估计(GAE)"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = last_value.detach()
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = 0  # 终止状态不bootstrap
            else:
                delta = rewards[t] + self.gamma * last_value - values[t]
                last_value = values[t]
                
            advantages[t] = delta + self.gamma * self.lmbda * last_advantage
            last_advantage = advantages[t]
    
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update_history_sequence(self, states):
        """
        更新 self.history 并返回包含历史序列的张量。
        
        参数:
            states (torch.Tensor): 当前的状态序列，形状为 (batch_size, height, width) 或 (batch_size, channels, height, width)。
        
        返回:
            torch.Tensor: 包含历史序列的张量，形状为 (batch_size, seq_len, channels, height, width)。
        """
        if  states is None:
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
        '''
        torch.roll(..., shifts=-1, dims=1)：
        沿序列维度（dims=1）向左滚动一位，丢弃最旧的状态。
        例如：原序列 [s1, s2, s3] → [s2, s3, s3]（末尾重复，但下一步覆盖）。
        self.history[:, -1] = states：
        将最新状态 states 写入序列最后一个位置。
        最终结果如 [s2, s3, s_new]。
        '''
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
                ratio = torch.exp(torch.clamp(new_probs - flat_old_log_probs, min=-20, max=20))
                
                # 添加数值稳定性检查
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print("警告: ratio包含NaN或Inf，跳过此批次更新")
                    continue
                    
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
                critic_loss = (flat_rewards - values).pow(2).mean()
                
                # 检查损失是否为NaN
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    print("警告: 损失函数包含NaN值，跳过此批次更新")
                    continue

                # 累积梯度
                total_loss = actor_loss + critic_loss
                total_loss.backward()
                
                # 检查梯度是否包含NaN
                has_nan_grad = False
                for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print("警告: 梯度包含NaN值，跳过此批次更新")
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    continue

                # 更严格的梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

            # 更新参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 清空记忆缓冲区
        self.memory.clear()

    # def save_model(self, actor_path, critic_path):
    #     torch.save(self.actor.state_dict(), actor_path)
    #     torch.save(self.critic.state_dict(), critic_path)
    #     print(f"Model saved to {actor_path} and {critic_path}")

    def save_model(self, actor_path, critic_path):
        """保存模型并自动创建缺失目录"""
        def safe_save(path, model):
            dir_name = os.path.dirname(path)
            if dir_name:  # 防止空目录名的情况
                os.makedirs(dir_name, exist_ok=True)  # 关键修改
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")

        safe_save(actor_path, self.actor)
        safe_save(critic_path, self.critic)
