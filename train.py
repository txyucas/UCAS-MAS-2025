import torch
from olympics_engine.generator import create_scenario
from olympics_engine.agent import *
from olympics_engine.scenario import Running, table_hockey, football, wrestling, billiard, \
    curling, billiard_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks
from olympics_engine.AI_olympics import AI_Olympics
from configs.config import CnnConfig1, CnnConfig2,train_config
import wandb
from agents.PPO import PPO_Agent
from olympics_engine.agent import *
import os
import numpy as np
# import datetime
# import glob
game_map_dict = {
     'running': Running,
     'table-hockey': table_hockey,
     'football': football,
     'wrestling': wrestling,
    # #'billiard': billiard,
     'curling': curling,
}

wandb.init(project="MAS", config=train_config().__dict__, name='MAS_test2')  # 在脚本开始时进行初始化，而不是函数内部

def initialize_game(map):
    
    Gamemap = create_scenario(map)
    if map in game_map_dict:
        game = game_map_dict[map](Gamemap)
        agent_num = 2
    else:
        raise ValueError(f"Unknown map: {map}")
    return game, agent_num

def get_random_game(game_map):
    """随机选择游戏地图和智能体数量"""
    game_map = random.choice(list(game_map.keys()))
    game, agent_num = initialize_game(game_map)
    return game, agent_num
    



class Trainer:
    def __init__(self,agent1=PPO_Agent(config=CnnConfig1()),agent2=PPO_Agent(config=CnnConfig2()),config=train_config(),actor1_path="backup_actor1.pth", actor2_path="backup_actor2.pth", critic1_path="backup_critic1.pth", critic2_path="backup_critic2.pth"):
        self.agent1=agent1
        self.agent2=agent2
        self.config = config
        self.random_agent=random_agent()
        self.actor1_path=actor1_path
        self.actor2_path=actor2_path
        self.critic1_path=critic1_path
        self.critic2_path=critic2_path
        self.rewardlist_1 = []
        self.rewardlist_2 = []
        # self.backup_step = 0
        # self.batch_size = 64  # 每次更新需要积累的步数
        # self.mini_batch_size = 16  # 每次梯度更新的小批次大小
        self.step_penalty = 1e-2
        # Load the models if paths are provided

    # def _collect_experiences(self):
    #     """跨多个episode收集经验，直到达到batch_size"""
    #     while len(self.agent1.memory) < self.batch_size:
    #         # 单个episode收集
    #         state = self.env.reset()
    #         episode_steps = 0
    #         done = False
            
    #         while not done:
    #             # 动作采样与环境交互
    #             action1 = self.agent1.sample_action(state[0])[0]
    #             action2 = self.agent2.sample_action(state[1])[0]
    #             next_state, reward, done, _ = self.env.step([action1, action2])
                
    #             # 奖励修正（保留原有逻辑）
    #             step_punishment = self.step_penalty * episode_steps / self.config.max_steps
    #             reward_agent1 = reward[0] - step_punishment
    #             reward_agent2 = reward[1] - step_punishment
                
    #             # 存储经验（不立即清空！）
    #             self.agent1.memory.push((state[0], action1, self.agent1.log_probs, reward_agent1, done))
    #             self.agent2.memory.push((state[1], action2, self.agent2.log_probs, reward_agent2, done))
                
    #             state = next_state
    #             episode_steps +=1

    #             # 仅在episode结束时重置LSTM状态
    #             if done:
    #                 self.agent1.reset_lstm()  # 新增方法，仅重置LSTM
    #                 self.agent2.reset_lstm()

    def _train_one_step(self):
        # 保存当前模型状态作为备份
        if hasattr(self, 'backup_step') and self.backup_step % 20 == 0:
            self._save_model_checkpoint(self.actor1_path, self.actor2_path, self.critic1_path, self.critic2_path)
        
        # # 初始化备份计数器
        # if not hasattr(self, 'backup_step'):
        #     self.backup_step = 0  # 在__init__中添加更规范

        # # 每20步生成唯一备份文件名
        # if self.backup_step % 20 == 0:
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     self._save_model_checkpoint(
        #         f"backup_actor1_step{self.backup_step}_{timestamp}.pth",
        #         f"backup_actor2_step{self.backup_step}_{timestamp}.pth",
        #         f"backup_critic1_step{self.backup_step}_{timestamp}.pth",
        #         f"backup_critic2_step{self.backup_step}_{timestamp}.pth"
        #     )
        # self.backup_step +=1
        
        self.agent1.reset()
        self.agent2.reset()
        self.agent1.sample_count = 0
        self.agent2.sample_count = 0
        ep_reward_agent1 = 0
        ep_reward_agent2 = 0
        state = self.env.reset()
        torch.cuda.empty_cache()  # 清理 GPU 缓存
        
         # 添加步数计数器
        total_steps = 0
        
        for _ in range(69 if self.env.game_name == 'curling' else self.config.max_steps):
            total_steps += 1
            action_agent1 = self.agent1.sample_action(state[0])[0]
            action_agent2 = self.agent2.sample_action(state[1])[0]
            action = [action_agent1, action_agent2]
            next_state, reward, done, _ = self.env.step(action)
             # 添加步数惩罚（核心修改）
            step_punishment = self.step_penalty * total_steps / self.config.max_steps
            reward_agent1 = reward[0] - step_punishment 
            reward_agent2 = reward[1] - step_punishment
        
            # reward_agent1, reward_agent2 = reward[0], reward[1]
            reward_agent1 +=0 if agent1.isnan==False else -10
            reward_agent2 +=0 if agent2.isnan==False else -10
            self.agent1.memory.push((state[0], action_agent1, self.agent1.log_probs, reward_agent1, done))
            self.agent2.memory.push((state[1], action_agent2, self.agent2.log_probs, reward_agent2, done))
            state = next_state
            self.rewardlist_1.append(self.agent1.update())
            self.rewardlist_2.append(self.agent2.update())
            ep_reward_agent1 += reward_agent1
            ep_reward_agent2 += reward_agent2
            
            # 检测是否有NaN值出现
            if self.agent1.isnan or self.agent2.isnan:
                print("检测到NaN值，尝试从备份恢复...")
                self._restore_from_checkpoint()
                return  # 跳过当前步骤

            if done:
                break
        self.agent1.memory.clear()  # 清空缓冲区
        self.agent2.memory.clear()  # 清空缓冲区
        torch.cuda.empty_cache()  # 再次清理 GPU 缓存
        
        wandb.log({
            "train_reward_agent1": ep_reward_agent1,
            "train_reward_agent2": ep_reward_agent2
        })
    
    def _save_model_checkpoint(self, actor1_path, actor2_path, critic1_path, critic2_path):
        torch.save(self.agent1.actor.state_dict(), actor1_path)
        torch.save(self.agent2.actor.state_dict(), actor2_path)
        torch.save(self.agent1.critic.state_dict(), critic1_path)
        torch.save(self.agent2.critic.state_dict(), critic2_path)
    
    def _restore_from_checkpoint(self):
        try:
            self.agent1.actor.load_state_dict(torch.load("backup_actor1.pth"))
            self.agent2.actor.load_state_dict(torch.load("backup_actor2.pth"))
            self.agent1.critic.load_state_dict(torch.load("backup_critic1.pth"))
            self.agent2.critic.load_state_dict(torch.load("backup_critic2.pth"))
            print("成功从备份恢复模型")
            
            # 重置优化器状态
            self.agent1.actor_optimizer = torch.optim.Adam(self.agent1.actor.parameters(), lr=self.agent1.actor_optimizer.param_groups[0]['lr'])
            self.agent1.critic_optimizer = torch.optim.Adam(self.agent1.critic.parameters(), lr=self.agent1.critic_optimizer.param_groups[0]['lr'])
            self.agent2.actor_optimizer = torch.optim.Adam(self.agent2.actor.parameters(), lr=self.agent2.actor_optimizer.param_groups[0]['lr'])
            self.agent2.critic_optimizer = torch.optim.Adam(self.agent2.critic.parameters(), lr=self.agent2.critic_optimizer.param_groups[0]['lr'])
        except:
            print("恢复失败，继续使用当前模型")
    
    # def _restore_from_checkpoint(self):
    #     backup_list = glob.glob("backup_actor1_step*.pth")
    #     if not backup_list:
    #         print("无可用备份")
    #         return

    #     # 显示可恢复的备份选项
    #     print("可恢复的备份版本：")
    #     for i, path in enumerate(backup_list):
    #         print(f"{i+1}. {path}")

    #     # 用户选择或自动选最新
    #     choice = int(input("输入要恢复的版本号：")) -1
    #     latest_backup = backup_list[choice]
        
    #     self.agent1.actor.load_state_dict(torch.load(latest_backup))
    #     self.agent2.actor.load_state_dict(torch.load(latest_backup))
    #     self.agent1.critic.load_state_dict(torch.load(latest_backup))
    #     self.agent2.critic.load_state_dict(torch.load(latest_backup))
    #     print("成功从备份恢复模型")
    
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
                self.env,num_agent=get_random_game(game_map_dict)  # 随机选择游戏地图和智能体数量
                state = self.env.reset()  # 重置环境/
                #self.env.render()
                for _ in range(65 if self.env.game_name=='curling'else self.config.max_steps ):
                    # 两个智能体分别选择动作
                    action_agent1 = self.agent1.act(state[0])  # 智能体1选择动作
                    action_agent2 = self.agent2.act(state[1])  # 智能体2选择动作
    
                    # 环境执行动作，返回新的状态和奖励
                    next_state, (reward_agent1, reward_agent2), done, _ = self.env.step([action_agent1, action_agent2])
    
                    # 更新状态
                    state = next_state
    
                    # 累加奖励
                    eval_reward_agent1 += reward_agent1
                    eval_reward_agent2 += reward_agent2
    
                    if done:
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
        self.config.max_steps=int(self.config.max_steps*0.98) if self.config.max_steps > 500 else 500
    
    def _eval_random(self):
        """评估随机智能体"""
        with torch.no_grad():
            agent1.sample_count = 0
            agent2.sample_count = 0
            total_random_eval_reward_agent1 = 0
            total_random_eval_reward_agent2 = 0

            for _ in range(self.config.eval_eps):
                self.env,num_agent=get_random_game(game_map_dict)  # 随机选择游戏地图和智能体数量
                # 智能体1与随机智能体对抗
                agent1.reset()
                agent2.reset()
                eval_reward_agent1 = 0
                eval_reward_random1 = 0
                state = self.env.reset()  # 重置环境

                for _ in range(69 if self.env.game_name=='curling'else self.config.max_steps ):
                    action_agent1 = self.agent1.act(state[0])  # 智能体1选择动作
                    action_random1 = self.random_agent.act(obs=state[1])  # 随机智能体选择动作

                    next_state, (reward_agent1, reward_random1), done, _ = self.env.step([action_agent1, action_random1])
                    state = next_state

                    eval_reward_agent1 += reward_agent1
                    eval_reward_random1 += reward_random1

                    if done:
                        break

                total_random_eval_reward_agent1 += eval_reward_agent1

                # 智能体2与随机智能体对抗
                eval_reward_agent2 = 0
                eval_reward_random2 = 0
                state = self.env.reset()  # 重置环境

                for _ in range(65 if self.env.game_name=='curling' else self.config.max_steps):
                    action_random2 = self.random_agent.act(state[0])  # 随机智能体选择动作
                    action_agent2 = self.agent2.act(state[1])  # 智能体2选择动作

                    next_state, (reward_random2, reward_agent2), done, _ = self.env.step([action_random2, action_agent2])
                    state = next_state

                    eval_reward_agent2 += reward_agent2
                    eval_reward_random2 += reward_random2

                    if done:
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
    def training(self,actor1_path=None, actor2_path=None, critic1_path=None, critic2_path=None):
        for i in range(self.config.num_episodes):
            for i in range(self.config.batch_per_epi):
                self.env, agent_num = get_random_game(game_map_dict)
                self._train_one_step()
            self._eval_one_batch()
            self._eval_random()
            self._renew_args()
            # Save the models
            self.agent1.save_model(actor1_path, critic1_path)
            self.agent2.save_model(actor2_path, critic2_path)
            
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

    # def _save_agent(self, agent, tag):
    #     """保存智能体到文件（假设agent实现了save方法）"""
    #     path_actor = os.path.join(self.save_dir, f"agent_actor_{tag}.pth")
    #     path_critic = os.path.join(self.save_dir, f"agent_critic_{tag}.pt")
    #     agent.save_model(path_actor, path_critic)

    # def _load_agent(self, path_actor, path_critic):
    #     """加载智能体（假设agent类有load方法）"""
    #     return type(self.base_agent_1).load_model(path_actor, path_critic)

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
    agent1 = PPO_Agent(config=CnnConfig1(),istest=False,)
    agent2 = PPO_Agent(config=CnnConfig2(),istest=False,)
    # trainer = Trainer(agent1=agent1, agent2=agent2, config=train_config())
    
    # # Start training
    # trainer.training(actor1_path="ckpt/actor1.pth", actor2_path="ckpt/actor2.pth", critic1_path="ckpt/critic1.pth", critic2_path="ckpt/critic2.pth")
    
    selfplay = SelfPlay(base_agent_1=agent1, base_agent_2=agent2)
    selfplay.training()