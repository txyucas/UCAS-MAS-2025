import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import random
from collections import deque
from models.get_model import get_model
from classify import Classify

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
    '''PG的经验回放池，支持采样所有样本或序列样本
    '''
    def __init__(self, capacity=1000):
        super().__init__(capacity)  # 调用父类初始化方法
    
    def sample(self, batch_size=None, sequential=False):
        ''' 采样transitions
        Args:
            batch_size (int, optional): 采样大小，默认为None表示采样所有样本
            sequential (bool, optional): 是否进行序列采样，默认为False
        '''
        if batch_size is None:
            # 采样所有样本
            batch = list(self.buffer)
            return zip(*batch)
        else:
            # 使用父类的sample方法进行批量采样或序列采样
            return super().sample(batch_size, sequential)

class PPO_Agent:
    def __init__(self,config,train_config,env_type=None,model_pth=None):
        self.config=config
        self.model_pth=model_pth
        self.env_type=env_type
        self.device = config.device
        self.train_config=train_config
        self.memory = PGReplay(self.config.buffer_size)
        self._get_models()
        self.reset()
        self._get_train_params()     
        self._get_self_play_args()
        
    def __reset_lstm(self):
        self.actor_hidden = torch.zeros(1,self.train_config.batch_size,self.config.lstm_hidden_size).to(self.device)
        self.critic_hidden = torch.zeros(1,self.train_config.batch_size,self.config.lstm_hidden_size).to(self.device)
        self.actor_cell = torch.zeros(1,self.train_config.batch_size,self.config.lstm_hidden_size).to(self.device)
        self.critic_cell = torch.zeros(1,self.train_config.batch_size,self.config.lstm_hidden_size).to(self.device)
        
    def __reset_rnn(self):
        self.actor_hidden = torch.zeros(1,self.train_config.batch_size,self.config.rnn_hidden_size).to(self.device)
        self.critic_hidden = torch.zeros(1,self.train_config.batch_size,self.config.rnn_hidden_size).to(self.device)
    
    def _get_models(self):
        self.actor,self.critic= get_model(self.config)
        self.actor.to(self.device)
        self.critic.to(self.device)
        if self.env_type is not None and self.model_pth is not None:
            checkpoint = torch.load(self.model_pth)
            self.actor.load_state_dict(checkpoint[f'{self.env_type}']['actor'])
            print(f"actor_{self.env_type} load from {self.model_pth}")
            self.critic.load_state_dict(checkpoint[f'{self.env_type}']['critic'])
            print(f"critic_{self.env_type} load from {self.model_pth}")
        elif self.model_pth is not None:
            checkpoint = torch.load(self.model_pth)
            self.actor.load_state_dict(checkpoint['all']['actor'])
            print(f"actor_all load from {self.model_pth}")
            self.critic.load_state_dict(checkpoint['all']['critic'])
            print(f"critic_all load from {self.model_pth}")
    
    def _get_train_params(self):
        if self.config.is_train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr,)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

            
            self.lmbda = self.config.lmbda
            self.k_epochs = self.config.k_epochs
            self.gamma = self.config.gamma
            self.eps_clip = self.config.eps_clip
            self.entropy_coef = self.config.entropy_coef
            self.update_freq = self.config.update_freq
            self.episode_values=[]
            self.episode_rewards = []
            self.batch_size = self.train_config.batch_size if self.train_config.batch_size is not None else None
        else:
            pass
        
    def _reset_memory(self):
        if self.config.rnn_or_lstm is not None:
            self.__reset_lstm()if self.config.rnn_or_lstm == 'lstm' else self.__reset_rnn()
                    
    def reset(self):
        self._reset_memory()
        self.memory.clear()
        self.sample_count = 0
    
    def _get_self_play_args(self):
        if self.train_config.selfplay==True:
            self.opponent_pool=deque(maxlen=self.config.opponent_pool_size)
            self.sp_update_interval = self.config.sp_update_interval
            self.sp_win_threshold = self.config.sp_win_threshold
            self.sp_replay_ratio = self.config.sp_replay_ratio
            self.policy_noise_std = self.config.policy_noise_std
        else:
            pass
        
    def _reset_episode_data(self):
        """每回合开始时清空临时数据"""
        self.episode_values.clear()
        self.episode_rewards.clear()
        
    def act(self,obs)->list: 
        state = torch.tensor([ob['agent_obs'] for ob in obs], dtype=torch.float).to(self.device)
        if state.ndim == 2:
            state = state.unsqueeze(0)
            # add batch dimension
        state= state.unsqueeze(1) #add channel dimension
        
        with torch.no_grad():
            if self.config.rnn_or_lstm=='lstm':
                mu,sigma,(self.actor_hidden,self.actor_cell)=self.actor(state,self.actor_hidden,self.actor_cell)
            elif self.config.rnn_or_lstm=='rnn':
                mu,sigma,(self.actor_hidden)=self.actor(state,self.actor_hidden)
            else:
                mu,sigma=self.actor(state)
        
        #clip
        action=mu
        action = action.cpu().numpy()
        action[:, 0] = np.clip(action[:, 0], -100, 200)  #motor
        action[:, 1] = np.clip(action[:, 1], -30, 30)   #angle
        return action.tolist()
    @torch.no_grad()
    def sample_action(self,obs): 
        '''
        get act while training
        you need to push the transition to memory after this function
        you also need to renew the episode_reward after that
        '''
        state=torch.tensor([ob['agent_obs'] for ob in obs], dtype=torch.float).to(self.device).unsqueeze(1)
        
        self.sample_count += 1
        
        if self.config.rnn_or_lstm=='lstm':
            self.old_actor_hidden,self.old_actor_cell=self.actor_hidden,self.actor_cell
            mu,sigma,(self.actor_hidden,self.actor_cell)=self.actor(state,self.actor_hidden,self.actor_cell)
        elif self.config.rnn_or_lstm=='rnn':
            self.old_actor_hidden=self.actor_hidden
            mu,sigma,(self.actor_hidden)=self.actor(state,self.actor_hidden)
        else:
            mu,sigma=self.actor(state)
        
        # get log_prob
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=1)

        action_np = action.detach().cpu().numpy()
        
        #clip
        action_np[:, 0] = np.clip(action_np[:, 0], -100, 200)
        action_np[:, 1] = np.clip(action_np[:, 1], -30, 30)
        action = action_np.tolist()
        
        #update episode_values
        with torch.no_grad():
            if self.config.rnn_or_lstm=='lstm':
                self.old_critic_hidden,self.old_critic_cell=self.critic_hidden,self.critic_cell
                current_value, (self.critic_hidden,self.critic_cell) = self.critic(state,self.critic_hidden,self.critic_cell)
            elif self.config.rnn_or_lstm=='rnn':
                self.old_critic_hidden=self.critic_hidden
                current_value, (self.critic_hidden) = self.critic(state,self.critic_hidden)
            else:
                current_value = self.critic(state)
            current_value = current_value.mean()
            self.episode_values.append(current_value.item())
        
        return action,log_prob,state
    
    def _compute_advantage(self, rewards, values, dones, last_value):
        seq_len, batch_size = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = last_value.detach() 

        # Process each time step in reverse order
        for t in reversed(range(seq_len)):
            mask = 1.0 - dones[t].float()    # 保持正确的维度 [batch_size]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]  # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            advantages[t] = delta + self.gamma * self.lmbda * mask * last_advantage  # A_t = δ_t + γλ * A_{t+1}
            # 保存当前优势和值用于下一次循环
            last_advantage = advantages[t]
            last_value = values[t]

        # 正确对优势函数进行归一化（基于批次）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def update(self):
        
        if self.sample_count%self.update_freq != 0 or self.sample_count==0:
            return
        
        if self.config.rnn_or_lstm == 'lstm':
            old_states, old_actions, old_log_probs, old_rewards, old_dones, old_actor_hidden, old_actor_cell, old_critic_hidden, old_critic_cell = self.memory.sample(batch_size=self.config.sample_batch_size)

        elif self.config.rnn_or_lstm == 'rnn':
            old_states, old_actions, old_log_probs, old_rewards, old_dones, old_actor_hidden, old_critic_hidden = self.memory.sample(batch_size=self.config.sample_batch_size)
        else:
            old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample(batch_size=self.config.sample_batch_size)
        
        # Convert to tensors and reshape
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float)
        old_rewards = torch.tensor(np.array(old_rewards), device=self.device, dtype=torch.float)
        old_dones = torch.tensor(np.array(old_dones), device=self.device, dtype=torch.float)
        old_states = old_states
        values=[]
        with torch.no_grad():
            if self.config.rnn_or_lstm=='lstm':
                for (old_state,old_critic_hid,old_critic_cel)in zip(old_states,old_critic_hidden,old_critic_cell):
                    value,_= self.critic(old_state,old_critic_hid,old_critic_cel)
                    values.append(value.cpu().numpy())
            elif self.config.rnn_or_lstm=='rnn':
                for (old_state,old_critic_hid)in zip(old_states,old_critic_hidden):
                    value,_= self.critic(old_state,old_critic_hid)
                    values.append(value.cpu().numpy())
            else:
                for old_state in old_states:
                    value= self.critic(old_state)
                    values.append(value.cpu().numpy())
                
        values=torch.tensor(np.array(values), device=self.device, dtype=torch.float)

        values=values.squeeze(-1).detach()
        
        advantages = self._compute_advantage(rewards=old_rewards, values=values, dones=old_dones, last_value=torch.zeros_like(values[0]).to(self.device))
        advantages=advantages.detach()
        
        total_batch_size= min(self.config.sample_batch_size,self.config.update_freq)
        accumulation_steps = min(self.config.sample_batch_size,self.config.update_freq)
        
        for i in range(self.k_epochs):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss=None
            for e in range(accumulation_steps):
                #initialize batch
                start_idx= e * 1
                end_idx= min(start_idx + 1, total_batch_size)
                batch_states = old_states[start_idx]
                batch_actions = old_actions[start_idx]
                batch_log_probs = old_log_probs[start_idx]
                batch_advantages = advantages[start_idx]
                batch_rewards = old_rewards[start_idx]
                batch_dones = old_dones[start_idx]
                
                if self.config.rnn_or_lstm=='lstm':
                    batch_actor_hidden=old_actor_hidden[start_idx]
                    batch_actor_cell=old_actor_cell[start_idx]
                    mu,sigma,_=self.actor(batch_states,batch_actor_hidden,batch_actor_cell)
                elif self.config.rnn_or_lstm=='rnn':
                    batch_actor_hidden=old_actor_hidden[start_idx]
                    mu,sigma,_=self.actor(batch_states[0],batch_actor_hidden)
                else:
                    mu,sigma=self.actor(batch_states[0])
                    
                # Get new log probabilities
                dist = torch.distributions.Normal(mu, sigma)
                new_probs = dist.log_prob(batch_actions).sum(dim=1)
                # Calculate the ratio
                ratio=torch.exp(torch.clamp(new_probs-batch_log_probs, min=-20, max=20))
                
                # Calculate surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()+ self.entropy_coef * dist.entropy().mean()
                
                # Calculate value
                if self.config.rnn_or_lstm=='lstm':
                    batch_critic_hidden=old_critic_hidden[start_idx]
                    batch_critic_cell=old_critic_cell[start_idx]
                    current_values,_ = self.critic(batch_states,batch_critic_hidden,batch_critic_cell)
                elif self.config.rnn_or_lstm=='rnn':
                    batch_critic_hidden=old_critic_hidden[start_idx]
                    current_values,_ = self.critic(batch_states,batch_critic_hidden)
                else:
                    current_values = self.critic(batch_states)
                
                critic_loss= (current_values.squeeze(-1) - (batch_advantages + values.detach())).pow(2).mean()
                
                total_loss= actor_loss + critic_loss if total_loss is None else total_loss + actor_loss + critic_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            
        self.memory.clear()
        
        result = [np.mean(self.episode_values) if self.episode_values else 0.0,
                  np.std(self.episode_values) if self.episode_values else 0.0,
                  np.mean(self.episode_rewards) if self.episode_rewards else 0.0]
        
        self._reset_episode_data()
        return result
    def save_model(self,dir,step,number,env_type='all'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir=os.path.join(dir,f'agent{number}')
        if not os.path.exists(dir):
            os.makedirs(dir)            
        torch.save({f'{env_type}': {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),}
        }, os.path.join(dir, f'ppo_{step}.pth'))
        print(f"model saved to {os.path.join(dir, f'ppo_{step}.pth')}")












