import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PPO import PPO_Agent
from config import *
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt.pth')
ppo = PPO_Agent(config=CnnConfig1(),train_config=train_config(),model_pth=path)

def my_controller(observation, action_space, is_act_continuous=False):
    act=ppo.act(observation)[0]
    action = np.array(act).astype(np.float32).reshape(2, 1)
    return action
