import sys
from test.scenario import Running, table_hockey, football, wrestling, billiard, curling, billiard_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks
from pathlib import Path
#sys.path.append(str(Path(__file__).resolve().parent)) 
#sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy' 
from test.generator import create_scenario
import argparse
from olympics_engine.agent import *
import time

from agents.PPO import PPO_Agent
from olympics_engine.AI_olympics import AI_Olympics
from configs.config import *
import random
import json


ppo=PPO_Agent
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
    config.min_std = 0.1
    config.batch_size=1
    config.is_train = False
    for i in range(100):
        #game, agent_num = initialize_game(args.map)
        vis =  200
        vis_clear =  5
        curling_gamemap = create_scenario('curling-IJACA-competition')
        
        #for agent in curling_gamemap['agents']:
        #    agent.visibility = vis
        #    agent.visibility_clear = vis_clear
        #curling_gamemap['env_cfg']['vis']=vis
        #curling_gamemap['env_cfg']['vis_clear'] = vis_clear
        #game = curling_competition(curling_gamemap)
        #football_gamemap = create_scenario('football')
        #for agent in football_gamemap['agents']:
        #    agent.visibility = vis
        #    agent.visibility_clear = vis_clear
        #game = football(football_gamemap)
        #billiard_gamemap = create_scenario("billiard-competition")
        #for agent in billiard_gamemap['agents']:
        #    agent.visibility = vis
        #    agent.visibility_clear = vis_clear
        #game = billiard_competition(billiard_gamemap)
        #
        #tablehockey_gamemap = create_scenario("table-hockey")
        #for agent in tablehockey_gamemap['agents']:
        #    agent.visibility = vis
        #    agent.visibility_clear = vis_clear
        #game = table_hockey(tablehockey_gamemap)
        
        wrestling_gamemap = create_scenario('wrestling')
        for agent in wrestling_gamemap['agents']:
            agent.visibility = vis
            agent.visibility_clear = vis_clear
        game = wrestling(wrestling_gamemap)
        agent_num = 2
        ### 修改这里
        agent1= PPO_Agent(config,config,model_pth='/home/tian/UCAS-MAS-2025/table-hockey/number_0/agent1/ppo_12.pth')
        agent2=PPO_Agent(config,config,model_pth='/home/tian/UCAS-MAS-2025/table-hockey/number_0/agent1/ppo_12.pth')
        rand_agent = random_agent()

        obs = game.reset()
        done = False
        step = 0
        if RENDER:
            game.render()

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        time_epi_s = time.time()
        while not done:
            step += 1
            
            if agent_num == 2:
                #action1, action2 = agent.act(obs[0]), rand_agent.act(obs[1])
                action1=agent1.act([obs[0]])[0]
                action1=rand_agent.act(obs[0])
                action2=rand_agent.act(obs[1])
                #print("action1",action1)
                #action2=agent2.act([obs[1]])[0]
                action = [action1, action2]
            elif agent_num == 1:
                action1 = agent.act(obs)
                action = [action1]
            obs, reward, done, _ = game.step(action)
            if reward[0]>1 or reward[1]>1:
            
                print("reward",reward)
            #old_name=game.current_game.game_name
            #obs, reward, done, _ = game.step(action)
            #name=game.current_game.game_name
            #if name!=old_name:
            #    agent.reset()
            #    print("reset agent")
            ##print(f'reward = {reward}')
            if RENDER:
                game.render()

        duration_t = time.time() - time_epi_s
        print("episode duration: ", duration_t,
              "step: ", step,
              "time-per-step:",(duration_t)/step)

