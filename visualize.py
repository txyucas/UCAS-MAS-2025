import sys
from submissions.submission import ppo
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent)) 
sys.path.append(r"E:\vscode\多智能体\UCAS-MAS-2025\olympics_engine\olympics_engine")
sys.path.append(str(Path(__file__).resolve().parent.parent))
print(sys.path)
from olympics_engine.generator import create_scenario
import argparse
from olympics_engine.agent import *
import time
from olympics_engine.scenario import Running, table_hockey, football, wrestling, billiard, \
    curling, billiard_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks
from agents.PPO import PPO_Agent
from olympics_engine.AI_olympics import AI_Olympics
from configs.config import CnnConfig1
from sub_random.submission import agent
import random
import json


def store(record, name):

    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

checkpoint_path_dict={}
agent_dict={}


def initialize_game(map):
    #if 'all' in map:
    #    game = AI_Olympics(random_selection=False, minimap=False)
    #    agent_num = 2
    game = AI_Olympics(random_selection=False, minimap=False)
    agent_num = 2
    return game , agent_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default="curling", type= str,
                        help = 'running/table-hockey/football/wrestling/billiard/curling/all/all_v2')
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    config=CnnConfig1()
    for i in range(10):
        game, agent_num = initialize_game(args.map)
        
        ### 修改这里
        agent= agent
        rand_agent = random_agent(config=CnnConfig1)

        obs = game.reset()
        done = False
        step = 0
        if RENDER:
            game.render()

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        time_epi_s = time.time()
        while not done:
            step += 1
            if step % 200 == 0:
                pass
            if agent_num == 2:
                #action1, action2 = agent.act(obs[0]), rand_agent.act(obs[1])
                action1=act=ppo.act(obs[0])[0].tolist()
                action2=rand_agent.act(obs[1])
                action = [action1, action2]
            elif agent_num == 1:
                action1 = agent.act(obs)
                action = [action1]
            obs, reward, done, _ = game.step(action)
            print(f'reward = {reward}')if reward!=[0,0] else None
            if RENDER:
                game.render()

        duration_t = time.time() - time_epi_s
        print("episode duration: ", duration_t,
              "step: ", step,
              "time-per-step:",(duration_t)/step)

