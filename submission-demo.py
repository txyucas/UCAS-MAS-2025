# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agent is random agent , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""


def my_controller(observation, action_space, is_act_continuous=True):
    agent_action = []
    for i in range(len(action_space)):
        action_ = sample_single_dim(action_space[i], is_act_continuous)
        agent_action.append(action_)
        
    return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
        elif action_space_list_each.__class__.__name__ == "Discrete_SC2":
            each = action_space_list_each.sample()
        elif action_space_list_each.__class__.__name__ == "Box":
            each = action_space_list_each.sample()
    return each


def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            # each = [0] * action_space_list_each[j]
            # idx = np.random.randint(action_space_list_each[j])
            if action_space_list_each[j].__class__.__name__ == "Discrete":
                each = [0] * action_space_list_each[j].n
                idx = action_space_list_each[j].sample()
                each[idx] = 1
                player.append(each)
            elif (
                action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle"
            ):
                each = []
                nvec = action_space_list_each[j].high
                sample_indexes = action_space_list_each[j].sample()

                for i in range(len(nvec)):
                    dim = nvec[i] + 1
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
                player.append(each)
    return player

'''
def deep_equal(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(deep_equal(x[k], y[k]) for k in x)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(deep_equal(a, b) for a, b in zip(x, y))
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.array_equal(x, y)
    else:
        return x == y

from envs.olympics_billiard_competition import OlympicsBilliardCompetition
import json
import numpy as np
configs = json.load(open("./configs/config.json"))
conf = configs["olympics-billiard-competition"]
Env = OlympicsBilliardCompetition(conf, seed=OlympicsBilliardCompetition.create_seed())
action_space0 = Env.set_action_space()[0]
observation0 = Env.get_all_observes()[0]
action_space1 = Env.set_action_space()[1]
observation1 = Env.get_all_observes()[1]
last_stat = Env.get_all_observes()

while(True):
    action0 = my_controller(observation0, action_space0)
    action1 = my_controller(observation1, action_space1)
    actions = [action0, action1]
    observation, reward, done, info_before, info = Env.step(actions)
    if done:
        print("Game Over")
        break
    print("reward:", reward)
    print("info:", info)
    print("observation:", deep_equal(observation, last_stat))
    last_stat = observation
    
    print("actions:", actions)
    '''