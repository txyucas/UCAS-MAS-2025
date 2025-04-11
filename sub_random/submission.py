import random
import numpy as np
class random_agent:
    def __init__(self):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
    def act(self, obs):

        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        return force, angle
agent=random_agent()


def my_controller(observation, action_space, is_act_continuous=False):
    act=agent.act(observation)
    action = np.array(act).astype(np.float32).reshape(2, 1)
    return action