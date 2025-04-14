import torch
from olympics_engine.agent import *
from configs.config import CnnConfig1, CnnConfig2,train_config,Fnnconfig
import wandb
from agents.PPO import PPO_Agent,PGReplay
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy' 
import numpy as np
from models.classify import get_batch

class random_agent:
    def __init__(self,config):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.memory=PGReplay()
        self.config=config
        self.batch_size = config.batch_size
        self.episode_values=[]
        self.episode_rewards = []
        self.config.rnn_or_lstm=None
    def update(self,frozen=False):
        result = [np.mean(self.episode_values) if self.episode_values else 0.0,
          np.std(self.episode_values) if self.episode_values else 0.0,
          np.mean(self.episode_rewards) if self.episode_rewards else 0.0]
        return result
    def act(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [force, angle]
    def sample_action(self, obs):
        acts, log_probs, states = [], [], []
        for i in range(self.batch_size):
            action = self.act(obs[i])
            acts.append(action)
            log_probs.append(0)
            states.append(None)
        return acts, log_probs, states
    def reset(self):
        pass
    def save_model(self,dir,step,number):
        pass
    def load_model(self,dir,step,number):
        pass
    def clone(self):
        return random_agent(config=self.config)
    
    



class Trainer:
    def __init__(self,agent1=PPO_Agent(config=CnnConfig1(),train_config=train_config()),agent2=PPO_Agent(config=CnnConfig2(),train_config=train_config(),),config=train_config(),dir='self_play_dir',section=1):
        self.agent1=agent1
        self.agent2=agent2
        self.config = config
        self.section=section
        self.random_agent=random_agent(config=CnnConfig1)
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir=dir
        self.rewardlist_1 = []
        self.rewardlist_2 = []
        self.step_penalty = 0

    def _train_one_step(self):
        self.agent1.config.total_step += 1
        self.agent2.config.total_step += 1
        self.envs = get_batch(self.config.batch_size,section=self.section)
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
        
        
        #for _ in range(self.config.max_steps*6):
        for _ in range(self.config.max_steps):
            total_steps += 1
            old_states_agent1=[state[0] for state in states]
            old_states_agent2=[state[1] for state in states]
            actions_agent1,log_probs_agent1,states_agent1 = self.agent1.sample_action(old_states_agent1)
            actions_agent2,log_probs_agent2,states_agent2= self.agent2.sample_action(old_states_agent2)
            actions = [list(action_pair) for action_pair in zip(actions_agent1, actions_agent2)]
            next_states,rewards,dones=[],[],[]
            step_punishment = self.step_penalty * total_steps / (self.config.max_steps)**2
            for env, action,i in zip(self.envs, actions,range(len(actions))):
                if env.done ==False:
                    #old_name=env.current_game.game_name if hasattr(env, 'current_game') and hasattr(env.current_game, 'game_name') else None
                    next_state, reward, done, _ = env.step(action)
                    reward[0], reward[1] = reward[0]-step_punishment, reward[1]-step_punishment
                    env.done=done
                    #if done:
                    #    reward[0]*=(self.config.max_steps-total_steps+1)*2/self.config.max_steps
                        
                    #new_name=env.current_game.game_name if hasattr(env, 'current_game') and hasattr(env.current_game, 'game_name') else None
                    #if old_name != new_name:
                    #    self.agent1.actor_cell[i]=torch.zeros_like(self.agent1.actor_cell[i])
                    #    self.agent1.critic_cell[i]=torch.zeros_like(self.agent1.critic_cell[i])
                    #    self.agent1.actor_hidden[i]=torch.zeros_like(self.agent1.actor_hidden[i])
                    #    self.agent1.critic_hidden[i]=torch.zeros_like(self.agent1.critic_hidden[i])
                    #if reward !=[0,0]:
                    if reward[0]>0.05:
                        #print('reward',reward)
                        pass                  
                else:
                    next_state=[{'agent_obs':None},{'agent_obs':None}]
                    reward=[0.0,0.0]
                    done=True                    
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done if done is not None else False)
            rewards_agent1 = np.array([reward[0]for reward in rewards])
            rewards_agent2 = np.array([reward[1]for reward in rewards])
        
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
            self.agent1.episode_rewards.append(rewards_agent1)
            self.agent2.episode_rewards.append(rewards_agent2) 
            states = next_states
            if all(dones):
                self.rewardlist_1.append(self.agent1.update())
                self.rewardlist_2.append(self.agent2.update(frozen=not self.config.train_both))
                break
            elif total_steps ==self.config.max_steps:
                self.rewardlist_1.append(self.agent1.update())
                self.rewardlist_2.append(self.agent2.update(frozen=not self.config.train_both)) 
            ep_reward_agent1 += rewards_agent1.mean()
            ep_reward_agent2 += rewards_agent2.mean()
            
            
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
            self.agent1.sample_count = 0
            self.agent2.sample_count = 0
            total_eval_reward_agent1 = 0
            total_eval_reward_agent2 = 0
            self.step_penalty = 0
            std_min=self.agent1.config.min_std
            self.agent1.config.min_std=0.1
            for _ in range(self.config.eval_eps):
                eval_reward_agent1 = 0
                #self.agent1.config.total_step += 1
                #self.agent2.config.total_step += 1
                self.agent1.reset()
                self.agent2.reset()
                eval_reward_agent2 = 0
                self.envs=get_batch(self.config.batch_size,section=self.section)  # 随机选择游戏地图和智能体数量
                states = [env.reset() for env in self.envs]  # 重置环境
                
                for env in self.envs:
                    env.max_step = self.config.max_steps
                donse=[]
                #self.env.render()
                #for _ in range( self.config.max_steps*6):
                for i in range( self.config.max_steps):    
                    # 两个智能体分别选择动作
                    old_states_agent1=[state[0] for state in states]
                    old_states_agent2=[state[1] for state in states]
                    actions_agent1= self.agent1.act(old_states_agent1)
                    if isinstance(self.agent2,random_agent):
                        actions_agent2= [self.agent2.act(old_states_agent2) for _ in range(config.batch_size)]
                    else:
                        actions_agent2= self.agent2.act(old_states_agent2)
                    actions = [[action_agent1, action_agent2] for (action_agent1, action_agent2) in zip(actions_agent1, actions_agent2)]
                    # 执行动作
                    next_states, rewards, dones = [], [], []
                    step_punishment = self.step_penalty * i / (self.config.max_steps)**2
                    for env, action,t in zip(self.envs, actions,range(self.config.batch_size)):
                        if env.done ==False:
                            #old_name=env.current_game.game_name if hasattr(env, 'current_game') and hasattr(env.current_game, 'game_name') else None
                            next_state, reward, done, _ = env.step(action)
                            if reward !=[0,0]:
                                reward[0]=0 if reward[0]<1 else reward[0]
                                reward[1]=0 if reward[1]<1 else reward[1]
                                #print('reward',reward)
                                pass
                            reward[0], reward[1] = reward[0]-step_punishment, reward[1]-step_punishment
                            env.done=done 
                            #new_name=env.current_game.game_name if hasattr(env, 'current_game') and hasattr(env.current_game, 'game_name') else None
                            #if old_name != new_name:
                            #    self.agent1.actor_cell[i]=torch.zeros_like(self.agent1.actor_cell[i])
                            #    self.agent1.critic_cell[i]=torch.zeros_like(self.agent1.critic_cell[i])
                            #    self.agent1.actor_hidden[i]=torch.zeros_like(self.agent1.actor_hidden[i])
                            #    self.agent1.critic_hidden[i]=torch.zeros_like(self.agent1.critic_hidden[i])
                 
                        else:
                            next_state=[{'agent_obs':None},{'agent_obs':None}]
                            reward=[0.0,0.0]
                            done=True                    
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done if done is not None else False)
                    donse.append(dones)
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
            self.agent1.config.min_std=std_min
        return mean_eval_reward_agent1
            
    def _renew_args(self):
        """随着训练的进行，逐渐降低学习率、eps_clip 和 entropy_coef"""
        # 减小学习率
        self.agent1.actor_optimizer.param_groups[0]['lr'] *= 0.984 if self.agent1.actor_optimizer.param_groups[0]['lr'] > 1e-5 else 1e-5 
        self.agent1.critic_optimizer.param_groups[0]['lr'] *= 0.984 if self.agent1.critic_optimizer.param_groups[0]['lr'] > 1e-5 else 1e-5
        try:
            self.agent2.actor_optimizer.param_groups[0]['lr'] *= 0.984 if self.agent2.actor_optimizer.param_groups[0]['lr'] > 1e-5 else 1e-5 
            self.agent2.critic_optimizer.param_groups[0]['lr'] *= 0.984 if self.agent2.critic_optimizer.param_groups[0]['lr'] > 1e-5 else 1e-5
        except:
            pass

        # 减小 eps_clip
        try:
            self.agent1.eps_clip *= 0.995 if self.agent1.eps_clip > 0.05 else 0.05
            self.agent2.eps_clip *= 0.995 if self.agent2.eps_clip > 0.05 else 0.05
        except:
            self.agent1.eps_clip *= 0.995 if self.agent1.eps_clip > 0.05 else 0.05

        # 减小 entropy_coef
        try:
            self.agent1.entropy_coef *= 0.993 if self.agent1.entropy_coef > 0.005 else 0.005
            self.agent2.entropy_coef *= 0.993 if self.agent2.entropy_coef > 0.005 else 0.005 
        except:
            self.agent1.entropy_coef *= 0.993 if self.agent1.entropy_coef > 0.005 else 0.005
        #self.config.max_steps=int(self.config.max_steps*0.98) if self.config.max_steps > 400 else 400
    
    def _eval_random(self):
        """评估随机智能体"""
        with torch.no_grad():
            total_random_eval_reward_agent1 = 0
            total_random_against_agent1 = 0
            min_std=self.agent1.config.min_std
            self.agent1.config.min_std=0.1
            for _ in range(self.config.eval_eps):
                self.envs=get_batch(self.config.batch_size,section=self.section)  # 随机选择游戏地图和智能体数量
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
                    actions = [[action_agent1, action_agent2] for (action_agent1, action_agent2) in zip(actions_agent1, actions_random1)]
                    next_states, rewards, dones = [], [], []
                    for env, action in zip(self.envs, actions):
                        if env.done ==False:
                            next_state, reward, done, _ = env.step(action)
                            reward[0]=0 if reward[0]<1 else reward[0]
                            reward[1]=0 if reward[1]<1 else reward[1]
                        else:
                            next_state=[{'agent_obs':None},{'agent_obs':None}]
                            reward=[0,0]
                            done=True                    
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done if done is not None else False)
                    states = next_states
                    rewards_agent1 = np.array([reward[0] for reward in rewards])
                    rewards_random1 = np.array([reward[1] for reward in rewards])
                    eval_reward_agent1 += rewards_agent1.mean()
                    eval_reward_random1 += rewards_random1.mean()

                    if all(dones):
                        break

                total_random_eval_reward_agent1 += eval_reward_agent1
                total_random_against_agent1 += eval_reward_random1
                agent1.config.min_std=min_std

            # 计算平均奖励
            mean_random_eval_reward_agent1 = total_random_eval_reward_agent1 / self.config.eval_eps
            mean_random_against_agent1 = total_random_against_agent1 / self.config.eval_eps

            self.agent1.memory.clear()  # 清空缓冲区
            torch.cuda.empty_cache()
            # 记录到 wandb

            wandb.log({
                "random_eval_mean_reward_agent1": mean_random_eval_reward_agent1,
                "random_against_mean_reward_agent1": mean_random_against_agent1,
            })
            print(f"随机对抗评估结果 - 智能体1平均奖励: {mean_random_eval_reward_agent1:.2f},")                
            return mean_random_eval_reward_agent1
    def training(self,save_dir='self_play_dir',step=0):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.dir=save_dir
        
        best_reward=self._eval_one_batch()
        for epi in range(self.config.num_episodes):
            agent1_copy=self.agent1.clone()
            best_reward*=0.95
            
            for i in range(self.config.batch_per_epi):
                self._train_one_step()
            current_reward=self._eval_one_batch()
            self._eval_random()
            if current_reward>=best_reward:
                best_reward=current_reward
                self._save_model(dir=self.dir,step=epi)
                print('模型更新')
            else:                
                print('模型未更新')
                self.agent1=agent1_copy
            #self._save_model(dir=self.dir,step=epi)
            self._renew_args()
            step+=1
            wandb.log({
                "step": step,
                "best_reward": best_reward
            })
        return step
    def _save_model(self,dir,step):
        if self.config.train_both:
            self.agent1.save_model(dir,step,number=1)
            self.agent2.save_model(dir,step,number=2)
        else:
            self.agent1.save_model(dir,step,number=1)
            
class SelfPlay:
    def __init__(
        self,
        base_agent_1,          # 初始智能体对象（必须实现 clone() 方法）
        base_agent_2,     # 可选的第二个初始智能体对象
        config=train_config(),  # 训练迭代次数
        # save_dir="selfplay_models",  # 模型存储目录
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), # 设备（CPU/GPU）
        env_type='running'  # 游戏类型
    ):
        """
        自博弈训练管理器
        核心逻辑：
        1. 维护一个历史对手池（包含不同训练阶段的智能体）
        2. 每次训练时从池中随机选择对手
        3. 定期将当前智能体的克隆加入池中
        """
        self.env_type=env_type
        self.section=game_to_section[env_type]
        self.base_agent_1 = base_agent_1
        self.base_agent_2 = base_agent_2
        self.config=config
        self.base_agent_3 = random_agent(config=CnnConfig1)
        self.pool_size = config.pool_size
        # self.save_dir = save_dir
        self.device = device
        self.current_iter = 0  
        self.num_training = config.num_training
        self.piority_list = [0.,0.]
        
        # 初始化对手池（至少包含初始智能体）
        self.opponent_pool = [self._clone_agent(base_agent_2), self._clone_agent(self.base_agent_3)]
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

    def get_opponent(self, temperature=0.5):
        """
        获取训练对手（按照优先级概率分布选取）
        Args:
            temperature: 控制softmax分布的温度参数
        """
        if temperature == 0:
            # 当 temperature 为 0 时，选择优先级最高的对手
            opponent_index = np.argmax(self.piority_list)
        else:
            # 计算softmax概率分布
            priorities = np.array(self.piority_list)
            priorities=np.clip(priorities, -20,20)  
            exp_priorities = np.exp((priorities) / temperature)  # 防止上溢
            probabilities = exp_priorities / np.sum(exp_priorities)
            
            # 按照概率分布随机选择对手
            opponent_index = np.random.choice(len(self.opponent_pool), p=probabilities)
        
        return self.opponent_pool[opponent_index], opponent_index

    def compute_piority(self, mean_value, value_std, mean_reward):
        """
        计算对手的优先级
        Args:
            mean_value: 平均值
            value_std: 标准差
            mean_reward: 平均奖励
        """
        reward_factor = mean_reward
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
        config=self.config
        step=0
        for i in range (self.num_training):
            # 选择对手
            dir=f'{self.env_type}/number_{i}'
            
            opponent, opponent_id = self.get_opponent()
            wandb.log({
                "opponent_id": opponent_id
            })
            print(f"选择对手: {opponent_id}")
            # 初始化训练器（假设训练器接收当前智能体和环境）
            trainer = Trainer(agent1=self.base_agent_1, agent2=opponent,config=config,dir=dir,section=self.section)
            config=trainer.config
            # 执行训练
            step=trainer.training(save_dir=dir,step=step)
            
            mean_value_1 = np.mean(trainer.rewardlist_1[:][0])
            value_std_1 = np.mean(trainer.rewardlist_1[:][1])
            mean_reward_1 = np.mean(trainer.rewardlist_1[:][2])
            mean_value_2 = np.mean(trainer.rewardlist_2[:][0])
            value_std_2 = np.mean(trainer.rewardlist_2[:][1])
            mean_reward_2 = np.mean(trainer.rewardlist_2[:][2])
            
            updatelist = np.array([mean_value_1, value_std_1, mean_reward_1, mean_value_2, value_std_2, mean_reward_2])
            
            # 更新对手池（此处假设每次迭代后都更新）
            self.update_pool(self.base_agent_1)
            self.update_piority(opponent_id, updatelist)

import numpy as np
import os
def all_seed(seed = 1):
    ''' 万能的seed函数
    '''
    if seed == 0:
        return
    np.random.seed(seed+1)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 

Section_to_game={
        
        1:'billiard-competition',
        2:'table-hockey',
        3:'football',
        4:'wrestling',
        5:'curling-IJACA-competition',
        6:'running1',
        7:'running2',
        8:'running3',
        9:'running4',
}
game_to_section={
        'billiard-competition':1,
        'table-hockey':2,
        'football':3,
        'wrestling':4,
        'curling-IJACA-competition':5,
        'running1':6,
        'running2':7,
        'running3':8,
        'running4':9
}         
if __name__ == "__main__":
    all_seed(3407)
    section=2
    game=Section_to_game[section]
    wandb.init(project=f"MAS-final-all", config=train_config().__dict__, name=f'new-{game}-3')  # 在脚本开始时进行初始化，而不是函数内部
    
    # Initialize the game and agents
    agent1 = PPO_Agent(config=CnnConfig1(),train_config=train_config(),env_type=game)
    agent2 = PPO_Agent(config=CnnConfig2(),train_config=train_config(),env_type=game)
    config=train_config()

    selfplay = SelfPlay(base_agent_1=agent1, base_agent_2=agent2,config=config,env_type=game)
    selfplay.training()