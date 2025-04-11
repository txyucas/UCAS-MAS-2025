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
    model: str = "cnn"
    is_train: bool = False
    
    rnn_or_lstm: str = "lstm" # rnn or lstm or none
    rnn_hidden_size: int = 48 
    
    #rl_config
    lmbda: float = 0.9
    k_epochs: int = 4
    gamma: float = 0.99
    eps_clip=0.05
    entropy_coef=0.02 # entropy coefficient
    update_freq=32
    buffer_size=32
    sample_batch_size=8
    #train_config
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size: int = 1
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    

@dataclass
class CnnConfig2:
    #model_config
    input_height: int = 40
    input_width: int = 40
    input_channel: int = 1
    model_info: list[tuple[int, int]] = field(default_factory=lambda: [(8, 3), (32, 3), (48, 5)])
    lstm_hidden_size: int = 48
    model:str = "cnn" # cnn or lstm or none
    
    rnn_or_lstm: str = "lstm" # rnn or lstm or none
    rnn_hidden_size: int = 48 
    is_train: bool = False
    
    #rl_config
    lmbda: float = 0.9
    k_epochs: int = 8
    gamma: float = 0.99
    max_sqe_len: int = 10
    eps_clip=0.12
    entropy_coef=0.015 # entropy coefficient
    update_freq=32
    
    #train_config
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size: int = 1
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4
    buffer_size=32
    sample_batch_size:int=8
@dataclass
class train_config:
    #train_config
    selfplay: bool = True
    eval_eps = 1 # 评估的回合数
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    max_steps: int = 400
    num_episodes: int = 8
    batch_size: int = 1
    batch_per_epi:int=3 # 每个回合的批次数
    
     # 新增全局训练参数
    train_both: bool = False
    pool_size: int = 20
    num_training: int = 1000
    
    