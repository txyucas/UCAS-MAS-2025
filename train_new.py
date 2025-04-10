import torch
from olympics_engine.generator import create_scenario
from olympics_engine.agent import *
from olympics_engine.scenario import Running, table_hockey, football, wrestling, billiard, \
    curling, billiard_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks
from olympics_engine.AI_olympics import AI_Olympics
from configs.config import CnnConfig1, CnnConfig2,train_config
import wandb
from agents.ppo_new import PPO_Agent
from olympics_engine.agent import *
import os
import numpy as np
from classify import get_batch


wandb.init(project="MAS", config=train_config().__dict__, name='MAS_test2')  # 在脚本开始时进行初始化，而不是函数内部


class Trainer:
    def __init__(self,agent1=PPO_Agent(config=CnnConfig1(),train_config=train_config()),agent2=PPO_Agent(config=CnnConfig2(),train_config=train_config()),config=train_config(),dir='self_play_dir'):
        self.agent1=agent1
        self.agent2=agent2
        self.config = config
        self.random_agent=random_agent()
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir=dir
        self.rewardlist_1 = []
        self.rewardlist_2 = []
        self.step_penalty = 1e-2

    def _train_one_step(self):
        self.envs = get_batch(self.config.batch_size)
        # 保存当前模型状态作为备份
        if hasattr(self, 'backup_step') and self.backup_step % 20 == 0:
            self.agent1.save_model(self.dir,step=self.backup_step,number=1)
            self.agent2.save_model(self.dir,step=self.backup_step,number=2)
        self.agent1.reset()
        self.agent2.reset()
        self.agent1.sample_count = 0
        self.agent2.sample_count = 0
        ep_reward_agent1 = 0
        ep_reward_agent2 = 0
        states = [env.reset() for env in self.envs]  # 重置环境
        
        for env in self.envs:
            env.max_step = self.config.max_steps
        torch.cuda.empty_cache()  # 清理 GPU 缓存
        total_steps = 0
        
        
        for _ in range(self.config.max_steps*6):
            total_steps += 1
            old_states_agent1=[state[0] for state in states]
            old_states_agent2=[state[1] for state in states]
            actions_agent1,log_probs_agent1,states_agent1 = self.agent1.sample_action(old_states_agent1)
            actions_agent2,log_probs_agent2,states_agent2= self.agent2.sample_action(old_states_agent2)
            actions = [[action_agent1, action_agent2] for action_agent1, action_agent2 in zip(actions_agent1, actions_agent2)]
            next_states,rewards,dones=[],[],[]
            for env, action in zip(self.envs, actions):
                try:
                    step_result = env.step(action)
                    if isinstance(step_result, tuple) and len(step_result) == 4:
                        next_state, reward, done, _ = step_result
                except:
                    next_state, reward, done, _ = env.step(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
             # 添加步数惩罚（核心修改）
            step_punishment = self.step_penalty * total_steps / self.config.max_steps
            rewards_agent1 = np.array([reward[0] - step_punishment for reward in rewards])
            rewards_agent2 = np.array([reward[1] - step_punishment for reward in rewards])
        
            # reward_agent1, reward_agent2 = reward[0], reward[1]
            if self.agent1.config.rnn_or_lstm=='lstm':
                self.agent1.memory.push((states_agent1, actions_agent1, log_probs_agent1, rewards_agent1, dones,self.agent1.actor_hidden,self.agent1.actor_cell,self.agent1.critic_hidden,self.agent1.critic_cell))
            elif self.agent1.config.rnn_or_lstm=='rnn':
                self.agent1.memory.push((states_agent1, actions_agent1, log_probs_agent1, rewards_agent1, dones,self.agent1.actor_hidden,self.agent1.critic_hidden))
            else:
                self.agent1.memory.push((states_agent1, actions_agent1, log_probs_agent1, rewards_agent1, dones))
            if self.agent2.config.rnn_or_lstm=='lstm':
                self.agent2.memory.push((states_agent2, actions_agent2, log_probs_agent2, rewards_agent2, dones,self.agent2.actor_hidden,self.agent2.actor_cell,self.agent2.critic_hidden,self.agent2.critic_cell))
            elif self.agent2.config.rnn_or_lstm=='rnn':
                self.agent2.memory.push((states_agent2, actions_agent2, log_probs_agent2, rewards_agent2, dones,self.agent2.actor_hidden,self.agent2.critic_hidden))
            else:
                self.agent2.memory.push((states_agent2, actions_agent2, log_probs_agent2, rewards_agent2, dones))
            agent1.episode_rewards.append(rewards_agent1)
            agent2.episode_rewards.append(rewards_agent2)
            states = next_states
            self.rewardlist_1.append(self.agent1.update())
            self.rewardlist_2.append(self.agent2.update())
            ep_reward_agent1 += rewards_agent1.mean()
            ep_reward_agent2 += rewards_agent2.mean()
            
            if all(dones):
                break
        self.agent1.memory.clear()  # 清空缓冲区
        self.agent2.memory.clear()  # 清空缓冲区
        torch.cuda.empty_cache()  # 再次清理 GPU 缓存
        
        wandb.log({
            "train_reward_agent1": ep_reward_agent1,
            "train_reward_agent2": ep_reward_agent2
        })
    
    def _eval_one_batch(self):
        """评估一个批次（两个智能体对抗）"""
        with torch.no_grad():
            agent1.sample_count = 0
            agent2.sample_count = 0
            total_eval_reward_agent1 = 0
            total_eval_reward_agent2 = 0
    
            for _ in range(self.config.eval_eps):
                eval_reward_agent1 = 0
                agent1.reset()
                agent2.reset()
                eval_reward_agent2 = 0
                self.envs=get_batch(self.config.batch_size)  # 随机选择游戏地图和智能体数量
                states = [env.reset() for env in self.envs]  # 重置环境
                
                for env in self.envs:
                    env.max_step = self.config.max_steps
                #self.env.render()
                for _ in range( self.config.max_steps*6):
                    # 两个智能体分别选择动作
                    old_states_agent1=[state[0] for state in states]
                    old_states_agent2=[state[1] for state in states]
                    actions_agent1= self.agent1.act(old_states_agent1)
                    actions_agent2= self.agent2.act(old_states_agent2)
                    actions = [[action_agent1, action_agent2] for action_agent1, action_agent2 in zip(actions_agent1, actions_agent2)]
                    # 执行动作
                    next_states, rewards, dones = [], [], []
                    for env, action in zip(self.envs, actions):
                        try:
                            step_result = env.step(action)
                            if isinstance(step_result, tuple) and len(step_result) == 4:
                                next_state, reward, done, _ = step_result
                        except:
                            next_state, reward, done, _ = env.step(action)
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done)
    
                    # 更新状态
                    states = next_states
                    # 计算奖励
                    rewards_agent1 = np.array([reward[0] for reward in rewards])
                    rewards_agent2 = np.array([reward[1] for reward in rewards])
                    # 累加奖励
                    eval_reward_agent1 += rewards_agent1.mean()
                    eval_reward_agent2 += rewards_agent2.mean()

                    if all(dones):
                        break
                self.agent1.memory.clear()  # 清空缓冲区
                self.agent2.memory.clear()
                torch.cuda.empty_cache()  # 清理 GPU 缓存    
                # 累加每次评估的奖励
                total_eval_reward_agent1 += eval_reward_agent1
                total_eval_reward_agent2 += eval_reward_agent2
    
            # 计算平均评估奖励
            mean_eval_reward_agent1 = total_eval_reward_agent1 / self.config.eval_eps
            mean_eval_reward_agent2 = total_eval_reward_agent2 / self.config.eval_eps
    
            # 记录到 wandb
            wandb.log({
                "eval_mean_reward_agent1": mean_eval_reward_agent1,
                "eval_mean_reward_agent2": mean_eval_reward_agent2
            })
    
            print(f"评估结果 - 智能体1平均奖励: {mean_eval_reward_agent1:.2f}, 智能体2平均奖励: {mean_eval_reward_agent2:.2f}")
            
    def _renew_args(self):
        """随着训练的进行，逐渐降低学习率、eps_clip 和 entropy_coef"""
        # 减小学习率
        self.agent1.actor_optimizer.param_groups[0]['lr'] *= 0.92 if self.agent1.actor_optimizer.param_groups[0]['lr'] > 5e-5 else 5e-5 
        self.agent1.critic_optimizer.param_groups[0]['lr'] *= 0.95 if self.agent1.critic_optimizer.param_groups[0]['lr'] > 5e-5 else 5e-5
        self.agent2.actor_optimizer.param_groups[0]['lr'] *= 0.92 if self.agent2.actor_optimizer.param_groups[0]['lr'] > 5e-5 else 5e-5 
        self.agent2.critic_optimizer.param_groups[0]['lr'] *= 0.95 if self.agent2.critic_optimizer.param_groups[0]['lr'] > 5e-5 else 5e-5

        # 减小 eps_clip
        self.agent1.eps_clip *= 0.92 if self.agent1.eps_clip > 0.05 else 0.05
        self.agent2.eps_clip *= 0.95 if self.agent2.eps_clip > 0.05 else 0.05

        # 减小 entropy_coef
        self.agent1.entropy_coef *= 0.97 if self.agent1.entropy_coef > 0.005 else 0.005
        self.agent2.entropy_coef *= 0.97 if self.agent2.entropy_coef > 0.005 else 0.005 
        self.config.max_steps=int(self.config.max_steps*0.98) if self.config.max_steps > 400 else 400
    
    def _eval_random(self):
        """评估随机智能体"""
        with torch.no_grad():
            agent1.sample_count = 0
            agent2.sample_count = 0
            total_random_eval_reward_agent1 = 0
            total_random_eval_reward_agent2 = 0

            for _ in range(self.config.eval_eps):
                self.envs=get_batch(self.config.batch_size)  # 随机选择游戏地图和智能体数量
                # 智能体1与随机智能体对抗
                agent1.reset()
                agent2.reset()
                eval_reward_agent1 = 0
                eval_reward_random1 = 0
                states= [env.reset() for env in self.envs]  # 重置环境

                for _ in range(self.config.max_steps*6 ):
                    old_states_agent1=[state[0] for state in states]
                    old_states_agent2=[state[1] for state in states]
                    actions_agent1 = self.agent1.act(old_states_agent1)  # 智能体1选择动作
                    actions_random1 = [self.random_agent.act(old_state_agent2) for old_state_agent2 in old_states_agent2] # 随机智能体选择动作
                    actions = [[action_agent1, action_agent2] for action_agent1, action_agent2 in zip(actions_agent1, actions_random1)]
                    next_states, rewards, dones = [], [], []
                    for env, action in zip(self.envs, actions):
                        try:
                            step_result = env.step(action)
                            if isinstance(step_result, tuple) and len(step_result) == 4:
                                next_state, reward, done, _ = step_result
                        except:
                            next_state, reward, done, _ = env.step(action)
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done)
                    states = next_states
                    rewards_agent1 = np.array([reward[0] for reward in rewards])
                    rewards_random1 = np.array([reward[1] for reward in rewards])
                    eval_reward_agent1 += rewards_agent1.mean()
                    eval_reward_random1 += rewards_random1.mean()

                    if all(dones):
                        break

                total_random_eval_reward_agent1 += eval_reward_agent1

                # 智能体2与随机智能体对抗
                eval_reward_agent2 = 0
                eval_reward_random2 = 0
                states= [env.reset() for env in self.envs]  # 重置环境

                for _ in range(self.config.max_steps*6):
                    old_states_agent1=[state[0] for state in states]
                    old_states_agent2=[state[1] for state in states]
                    actions_random2= [self.random_agent.act(old_state_agent1) for old_state_agent1 in old_states_agent1] # 随机智能体选择动作
                    actions_agent2 = self.agent2.act(old_states_agent2)
                    
                    actions = [[action_agent1, action_agent2] for action_agent1, action_agent2 in zip(actions_random2, actions_agent2)]
                    next_states, rewards, dones = [], [], []
                    for env, action in zip(self.envs, actions):
                        try:
                            step_result = env.step(action)
                            if isinstance(step_result, tuple) and len(step_result) == 4:
                                next_state, reward, done, _ = step_result
                        except:
                            next_state, reward, done, _ = env.step(action)
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done)

                    states = next_states
                    rewards_agent2 = np.array([reward[0] for reward in rewards])
                    rewards_random2 = np.array([reward[1] for reward in rewards])
                    
                    eval_reward_agent2 += rewards_agent2.mean()
                    eval_reward_random2 += rewards_random2.mean()

                    if all(dones):
                        break

                total_random_eval_reward_agent2 += eval_reward_agent2

            # 计算平均奖励
            mean_random_eval_reward_agent1 = total_random_eval_reward_agent1 / self.config.eval_eps
            mean_random_eval_reward_agent2 = total_random_eval_reward_agent2 / self.config.eval_eps
            
            self.agent1.memory.clear()  # 清空缓冲区
            self.agent2.memory.clear()
            torch.cuda.empty_cache()
            # 记录到 wandb
            wandb.log({
                "random_eval_mean_reward_agent1": mean_random_eval_reward_agent1,
                "random_eval_mean_reward_agent2": mean_random_eval_reward_agent2
            })

            print(f"随机对抗评估结果 - 智能体1平均奖励: {mean_random_eval_reward_agent1:.2f}, "
                  f"智能体2平均奖励: {mean_random_eval_reward_agent2:.2f}")
    def training(self,save_dir='self_play_dir'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.dir=save_dir
        for i in range(self.config.num_episodes):
            for i in range(self.config.batch_per_epi):
                self._train_one_step()
            self._eval_one_batch()
            self._eval_random()
            self._renew_args()
            
            
class SelfPlay:
    def __init__(
        self,
        base_agent_1,          # 初始智能体对象（必须实现 clone() 方法）
        base_agent_2,     # 可选的第二个初始智能体对象
        pool_size=10,        # 对手池容量
        num_training=20,  # 训练迭代次数
        # save_dir="selfplay_models",  # 模型存储目录
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # 设备（CPU/GPU）
    ):
        """
        自博弈训练管理器
        核心逻辑：
        1. 维护一个历史对手池（包含不同训练阶段的智能体）
        2. 每次训练时从池中随机选择对手
        3. 定期将当前智能体的克隆加入池中
        """
        self.base_agent_1 = base_agent_1
        self.base_agent_2 = base_agent_2
        self.pool_size = pool_size
        # self.save_dir = save_dir
        self.device = device
        self.current_iter = 0  
        self.num_training = num_training
        self.piority_list = [0,0]
        
        # 初始化对手池（至少包含初始智能体）
        self.opponent_pool = [self._clone_agent(base_agent_1), self._clone_agent(base_agent_2)]
        # os.makedirs(save_dir, exist_ok=True)

    def _clone_agent(self, agent):
        """深拷贝智能体（假设agent实现了clone方法）"""
        cloned = agent.clone()
        return cloned


    def update_pool(self, new_agent):
        """更新对手池（先进先出策略）"""
        # 克隆新智能体加入池中
        self.opponent_pool.append(self._clone_agent(new_agent))
        
        # 保持池的大小不超过限制
        if len(self.opponent_pool) > self.pool_size:
            removed_agent = self.opponent_pool.pop(0)
            del removed_agent  # 显式释放内存

    def get_opponent(self, epsilon=0.8):
        """
        获取训练对手（带探索机制）
        Args:
            epsilon: 使用最新智能体的概率（否则随机选历史对手）
        """
        if random.random() < epsilon or len(self.opponent_pool) == 1:
            maxindex = self.piority_list.index(max(self.piority_list))
            return self.opponent_pool[maxindex], maxindex
        else:
            random_index = random.randint(0, len(self.opponent_pool)-1)  # 生成随机索引
            opponent = self.opponent_pool[random_index]          # 通过索引获取元素
            return opponent, random_index  # 从历史中随机选择

    def compute_piority(self, mean_value, value_std, mean_reward):
        """
        计算对手的优先级
        Args:
            mean_value: 平均值
            value_std: 标准差
            mean_reward: 平均奖励
        """
        reward_factor = 1.0 / (abs(mean_reward) + 1e-6)
        stability_factor = 1.0 / (value_std + 1e-6)
        return 0.7 * reward_factor + 0.3 * stability_factor

    def update_piority(self, opponent_id, updatelist):
        piority_1 = self.compute_piority(updatelist[0], updatelist[1], updatelist[2])
        piority_2 = self.compute_piority(updatelist[3], updatelist[4], updatelist[5])
        
        self.piority_list[opponent_id]=piority_2
        self.piority_list.append(piority_1)

    def training(
        self
    ):
        """
        执行一次自博弈训练迭代
        流程：
        1. 选择对手
        2. 创建训练环境
        3. 训练当前智能体
        4. 更新对手池
        """
        for i in range (self.num_training):
            # 选择对手
            opponent, opponent_id = self.get_opponent()
            # 初始化训练器（假设训练器接收当前智能体和环境）
            trainer = Trainer(agent1=self.base_agent_1, agent2=opponent, config=train_config())
            
            # 执行训练
            trainer.training(actor1_path="ckpt_selfplay/actor1.pth", actor2_path=f"ckp_selfplayt/actor2.pth", critic1_path=f"ckpt_selfplay/critic1.pth", critic2_path=f"ckpt_selfplay/critic2.pth")
            
            mean_value_1 = np.mean(trainer.rewardlist_1[:,0])
            value_std_1 = np.mean(trainer.rewardlist_1[:,1])
            mean_reward_1 = np.mean(trainer.rewardlist_1[:,2])
            mean_value_2 = np.mean(trainer.rewardlist_2[:,0])
            value_std_2 = np.mean(trainer.rewardlist_2[:,1])
            mean_reward_2 = np.mean(trainer.rewardlist_2[:,2])
            
            updatelist = np.array([mean_value_1, value_std_1, mean_reward_1, mean_value_2, value_std_2, mean_reward_2])
            
            # 更新对手池（此处假设每次迭代后都更新）
            self.update_pool(self.base_agent_1)
            self.update_piority(opponent_id, updatelist)

            
if __name__ == "__main__":
    # Initialize the game and agents
    agent1 = PPO_Agent(config=CnnConfig1(),train_config=train_config())
    agent2 = PPO_Agent(config=CnnConfig2(),train_config=train_config())
    config=train_config()
    trainer = Trainer(agent1=agent1, agent2=agent2, config=config)
    trainer.training()
    
    
    #selfplay = SelfPlay(base_agent_1=agent1, base_agent_2=agent2)
    #selfplay.training()