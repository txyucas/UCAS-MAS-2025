import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
print(sys.path)
from olympics_engine.generator import create_scenario
import argparse
from olympics_engine.agent import *
import time

from olympics_engine.scenario import Running, table_hockey, football, wrestling, billiard, \
    curling, billiard_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks

from olympics_engine.AI_olympics import AI_Olympics

import random
import json

checkpoint_path_dic={}
agent_dic={}

def store(record, name):

    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

def initialize_game(map):
    if 'all' in map:
        game = AI_Olympics(random_selection=False, minimap=False)
        agent_num = 2
    else:
        Gamemap = create_scenario(map)
        game_map_dict = {
            'running': Running,
            'running-competition': lambda gm: Running_competition(meta_map=gm, map_id=random.randint(1, 10)),
            'seeks': Seeks,
            'table-hockey': table_hockey,
            'football': football,
            'wrestling': wrestling,
            'billiard': billiard,
            'billiard-competition': billiard_competition,
            'curling': curling,
            'billiard-joint': billiard_joint,
            'curling-long': curling_long,
            'curling-competition': curling_competition
        }
        if map in game_map_dict:
            game = game_map_dict[map](Gamemap)
            agent_num = 2
        else:
            raise ValueError(f"Unknown map: {map}")
    #     agent_num = 2
    return game, agent_num




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default="running", type= str,
                        help = 'running/table-hockey/football/wrestling/billiard/curling/all/all_v2')
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    for i in range(1):
        game, agent_num = initialize_game(args.map)

        agent = random_agent()
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

            # print('\n Step ', step)

            #action1 = [100,0]#agent.act(obs)
            #action2 = [100,0] #rand_agent.act(obs)
            if agent_num == 2:
                action1, action2 = agent.act(obs[0]), rand_agent.act(obs[1])
                # action1 = [50,0.1]
                # action2 = [100,-0.2]

                # action1 =[50,1]
                # action2 = [50,-1]


                action = [action1, action2]
            elif agent_num == 1:
                action1 = agent.act(obs)
                action = [action1]

            # if step <= 5:
            #     action = [[200,0]]
            # else:
            #     action = [[0,0]]
            # action = [[200,action1[1]]]

            obs, reward, done, _ = game.step(action)
            print(f'reward = {reward}')
            # print('obs = ', obs)
            # plt.imshow(obs[0])
            # plt.show()
            if RENDER:
                game.render()

        duration_t = time.time() - time_epi_s
        print("episode duration: ", duration_t,
              "step: ", step,
              "time-per-step:",(duration_t)/step)
        # if args.map == 'billiard':
        #     print('reward =', game.total_reward)
        # else:
            # print('reward = ', reward)
        # if R:
        #     store(record,'bug1')

