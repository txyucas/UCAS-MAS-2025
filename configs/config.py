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
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

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
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    
@dataclass
class train_config:
    #train_config
    
    eval_eps = 5 # 评估的回合数
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    max_steps: int = 3000
    num_episodes: int = 100
    batch_per_epi:int=30 # 每个回合的批次数
    