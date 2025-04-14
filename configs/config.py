from dataclasses import dataclass, field
import torch

@dataclass
class CnnConfig1:
    #model_config
    input_height: int = 40
    input_width: int = 40
    input_channel: int = 1
    model_info: list[tuple[int, int]] = field(default_factory=lambda:[(32, 3), (48, 3), (64, 5)])
    lstm_hidden_size: int = 128
    model: str = "cnn"
    is_train: bool = True
    
    rnn_or_lstm: str = "lstm" # rnn or lstm or none
    rnn_hidden_size: int = 96 
    
    min_std: float =torch.tensor([20,4],device="cuda") # 最小标准差
    total_step: int = 1 # 总步数
    
    #rl_config
    lmbda: float = 0.9
    k_epochs: int = 4
    gamma: float = 0.999
    eps_clip=0.3
    entropy_coef=0.03 # entropy coefficient
    #train_config
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size: int = 8
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    entropy_decay: float = 0.9997
    
@dataclass
class Fnnconfig:
    input_dim:int = 40*40
    input_channel: int = 1
    model_info: list[tuple[int, int]] = field(default_factory=lambda: [256, 512, 256])
    lstm_hidden_size: int = 128
    model: str = "fnn"
    is_train: bool = True
    
    rnn_or_lstm: str = None # rnn or lstm or none
    rnn_hidden_size: int = 96 
    
    #rl_config
    lmbda: float = 0.9
    k_epochs: int = 12
    gamma: float = 0.999
    eps_clip=0.15
    entropy_coef=0.02 # entropy coefficient
    min_std: float =torch.tensor([40,10],device="cuda") # 最小标准差
    total_step: int = 1 # 总步数

    #train_config
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size: int = 8
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4

@dataclass
class CnnConfig2:
    #model_config
    input_height: int = 40
    input_width: int = 40
    input_channel: int = 1
    model_info: list[tuple[int, int]] = field(default_factory=lambda:  [(16, 3), (48, 3), (48, 5)])
    lstm_hidden_size: int = 96
    model:str = "cnn" # cnn or lstm or none
    
    rnn_or_lstm: str = "lstm" # rnn or lstm or none
    rnn_hidden_size: int = 96 
    is_train: bool = True
    
    min_std: float =torch.tensor([40,10],device="cuda") # 最小标准差
    total_step: int = 1 # 总步数
    #rl_config
    lmbda: float = 0.9
    k_epochs: int = 8
    gamma: float = 0.999
    eps_clip=0.12
    entropy_coef=0.015 # entropy coefficient
    
    
    #train_config
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size: int = 8
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4

@dataclass
class train_config:
    #train_config
    selfplay: bool = True
    eval_eps = 1 # 评估的回合数
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    max_steps: int = 500
    num_episodes: int = 20
    batch_size: int = 8
    batch_per_epi:int=4 # 每个回合的批次数
    
     # 新增全局训练参数
    train_both: bool = False
    pool_size: int = 20
    num_training: int = 1000
    
    