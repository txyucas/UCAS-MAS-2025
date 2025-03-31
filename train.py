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
game_map_dict = {
    'running': Running,
    'table-hockey': table_hockey,
    'football': football,
    'wrestling': wrestling,
    #'billiard': billiard,
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
    def __init__(self,agent1=PPO_Agent(config=CnnConfig1()),agent2=PPO_Agent(config=CnnConfig2()),config=train_config(),):
        self.agent1=agent1
        self.agent2=agent2
        self.config = config
        self.random_agent=random_agent()
        # Load the models if paths are provided
    def _train_one_step(self):
        self.agent1.reset()
        self.agent2.reset()
        self.agent1.sample_count = 0
        self.agent2.sample_count = 0
        ep_reward_agent1 = 0
        ep_reward_agent2 = 0
        state = self.env.reset()
        torch.cuda.empty_cache()  # 清理 GPU 缓存
        for _ in range(69 if self.env.game_name == 'curling' else self.config.max_steps):
            action_agent1 = self.agent1.sample_action(state[0])[0]
            action_agent2 = self.agent2.sample_action(state[1])[0]
            action = [action_agent1, action_agent2]
            next_state, reward, done, _ = self.env.step(action)
            reward_agent1, reward_agent2 = reward[0], reward[1]
            reward_agent1 +=0 if agent1.isnan==False else -10
            reward_agent2 +=0 if agent2.isnan==False else -10
            self.agent1.memory.push((state[0], action_agent1, self.agent1.log_probs, reward_agent1, done))
            self.agent2.memory.push((state[1], action_agent2, self.agent2.log_probs, reward_agent2, done))
            state = next_state
            self.agent1.update()
            self.agent2.update()
            ep_reward_agent1 += reward_agent1
            ep_reward_agent2 += reward_agent2
            if done:
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
                self.env,num_agent=get_random_game(game_map_dict)  # 随机选择游戏地图和智能体数量
                state = self.env.reset()  # 重置环境
    
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

if __name__ == "__main__":
    # Initialize the game and agents
    agent1 = PPO_Agent(config=CnnConfig1(),istest=False,)
    agent2 = PPO_Agent(config=CnnConfig2(),istest=False,)
    trainer = Trainer(agent1=agent1, agent2=agent2, config=train_config())
    
    # Start training
    trainer.training(actor1_path="ckpt.actor1.pth", actor2_path="ckpt.actor2.pth", critic1_path="ckpt.critic1.pth", critic2_path="ckpt.critic2.pth")