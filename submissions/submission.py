import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PPO import PPO_Agent
from config import CnnConfig1,train_config
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt.pth')
ppo=PPO_Agent(config=CnnConfig1(),train_config=train_config(),model_pth=model_path)
def my_controller(observation, action_space, is_act_continuous=False):
    act=ppo.act(observation)[0]
    action = np.array(act).astype(np.float32).reshape(2, 1)
    return action
import torch
observation={'obs': torch.zeros(40,40)}
# while 1:
#    action=my_controller(observation, None, False) 
#    print(action)
#    observation={'obs': torch.zeros(40,40)}
    


