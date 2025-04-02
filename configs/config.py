from dataclasses import dataclass, field
import torch

@dataclass
class CnnConfig1:
    #model_config
    input_height: int = 40
    input_width: int = 40
    input_channel: int = 1
    model_info: list[tuple[int, int]] = field(default_factory=lambda: [(8, 3), (32, 3), (48, 5)])
    lstm_hidden_size: int = 48
    
    #rl_config
    lmbda: float = 0.9
    k_epochs: int = 10
    gamma: float = 0.99
    max_sqe_len: int = 10
    eps_clip=0.15
    entropy_coef=0.02 # entropy coefficient
    update_freq=20
    
    #train_config
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size: int = 32
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4
    
     # 新增自博弈相关参数
    opponent_pool_size: int = 5           # 对手池容量
    sp_update_interval: int = 50         # 对手池更新间隔（步数）
    sp_win_threshold: float = 0.65        # 胜率阈值（触发课程学习）
    sp_replay_ratio: float = 0.6          # 自博弈数据占比
    policy_noise_std: float = 0.03       # 策略克隆时的噪声标准差

@dataclass
class CnnConfig2:
    #model_config
    input_height: int = 40
    input_width: int = 40
    input_channel: int = 1
    model_info: list[tuple[int, int]] = field(default_factory=lambda: [(8, 3), (32, 3), (48, 5)])
    lstm_hidden_size: int = 48
    
    #rl_config
    lmbda: float = 0.9
    k_epochs: int = 10
    gamma: float = 0.99
    max_sqe_len: int = 10
    eps_clip=0.12
    entropy_coef=0.015 # entropy coefficient
    update_freq=20
    
    #train_config
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size: int = 32
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4
    
    # ... 同样修改（参数值可不同）...
    opponent_pool_size: int = 5
    sp_update_interval: int = 50
    sp_win_threshold: float = 0.6
    sp_replay_ratio: float = 0.55
    policy_noise_std: float = 0.025
    
@dataclass
class train_config:
    #train_config
    
    eval_eps = 5 # 评估的回合数
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    max_steps: int = 3000
    num_episodes: int = 100
    batch_per_epi:int=30 # 每个回合的批次数
    
     # 新增全局训练参数
    sp_curriculum_rate: float = 1.1      # 课程难度提升倍率
    grad_clip: float = 0.5               # 梯度裁剪阈值
    eval_against_pool_eps: int = 3        # 对策略池的评估回合数
    
    