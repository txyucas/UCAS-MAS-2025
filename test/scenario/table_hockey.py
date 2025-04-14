from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
import pygame
import sys
import random
from pathlib import Path
CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
import os
import math

COLOR_TO_IDX = {
    'red': 7,
    'green': 1,
    'sky blue': 2,
    'yellow': 3,
    'grey': 4,
    'purple': 5,
    'black': 6,
    'light green': 0,
    'blue':8

}

IDX_TO_COLOR = {
    0: 'light green',
    1: 'green',
    2: 'sky blue',
    3: 'yellow',
    4: 'grey',
    5: 'purple',
    6: 'black',
    7: 'red',
    8: 'blue'
}

class table_hockey(OlympicsBase):
    def __init__(self, map):
        self.minimap_mode = map['obs_cfg']['minimap']

        super(table_hockey, self).__init__(map)

        self.game_name = 'table-hockey'

        self.agent1_color = self.agent_list[0].color
        self.agent2_color = self.agent_list[1].color

        self.gamma = map['env_cfg']['gamma']
        self.wall_restitution = map['env_cfg']['wall_restitution']
        self.circle_restitution = map['env_cfg']['circle_restitution']
        self.tau = map['env_cfg']['tau']
        self.speed_cap = map['env_cfg']['speed_cap']
        self.max_step = map['env_cfg']['max_step']

        self.print_log = False

        self.draw_obs = True
        self.show_traj = False
        self.beauty_render = False


    def reset(self):
        self.set_seed()
        self.init_state()
        self.step_cnt = 0
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False

        self.ball_pos_init()

        init_obs = self.get_obs()
        if self.minimap_mode:
            self._build_minimap()

        output_init_obs = self._build_from_raw_obs(init_obs)
        return output_init_obs

    def ball_pos_init(self):
        y_min, y_max = 300, 500
        for index, item in enumerate(self.agent_list):
            if item.type == 'ball':
                random_y = random.uniform(y_min, y_max)
                self.agent_init_pos[index][1] = random_y


    def check_overlap(self):
        pass



    def check_action(self, action_list):
        action = []
        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].type == 'agent':
                action.append(action_list[0])
                _ = action_list.pop(0)
            else:
                action.append(None)

        return action

    def step(self, actions_list):

        actions_list = self.check_action(actions_list)

        self.stepPhysics(actions_list)
        self.speed_limit()
        self.step_cnt += 1
        self.cross_detect()



        step_reward = self.get_reward()
        obs_next = self.get_obs()              #need to add agent or ball check in get_obs

        done = self.is_terminal()
        self.done = done
        self.change_inner_state()

        if self.minimap_mode:
            self._build_minimap()

        output_obs_next = self._build_from_raw_obs(obs_next)

        return output_obs_next, step_reward, done, ''

    def _build_from_raw_obs(self, obs):
        if self.minimap_mode:
            image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)
            return [{"agent_obs": obs[0], "minimap":image, "id":"team_0"},
                    {"agent_obs": obs[1], "minimap": image, "id":"team_1"}]
        else:
            return [{"agent_obs":obs[0], "id":"team_0"}, {"agent_obs": obs[1], "id":"team_1"}]

    def _build_minimap(self):
        #need to render first
        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode = True

        self.viewer.draw_background()
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)




    def cross_detect(self, **kwargs):
        """
        check whether the agent has reach the cross(final) line
        :return:
        """
        for agent_idx in range(self.agent_num):

            agent = self.agent_list[agent_idx]

            if agent.type == 'ball':
                for object_idx in range(len(self.map['objects'])):
                    object = self.map['objects'][object_idx]

                    if not object.can_pass():
                        continue
                    else:
                        if object.color == 'red' and object.check_cross(self.agent_pos[agent_idx], agent.r):
                            agent.color = 'red'
                            agent.finished = True  # when agent has crossed the finished line
                            agent.alive = False


    def get_reward(self):
        ball_end_pos = None
        collision_reward = [0., 0.]
        observation_reward = [0., 0.]  # 新增：基于观测的奖励
        
        # 找到球的位置和索引
        ball_idx = None
        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            if agent.type == 'ball':
                ball_idx = agent_idx
                break
                
        if not hasattr(self, 'agent_last_pos'):  # 检查是否初始化了agent_last_pos
                self.agent_last_pos = self.agent_pos.copy()  # 初始化为适当大小的列表
                
        if not hasattr(self,'width') or not hasattr(self,'height'):
            self.width,self.height= self.viewer.background.get_size()
            
        # 获取当前的观测数据
        current_obs = self.get_obs()
        
        # 检查每个智能体的观测中是否包含球
        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            if agent.type == 'agent':
                obs_data = current_obs[agent_idx]  # 40x40的观测矩阵
                
                # 获取球的颜色对应的int值
                ball_color_value = None
                for a in self.agent_list:
                    if a.type == 'ball':
                        ball_color_value = COLOR_TO_IDX[a.color]
                        break
                
                # 检查观测矩阵中是否有球对应的颜色值
                ball_in_view = False
                if ball_color_value is not None:
                    # 遍历观测矩阵寻找球的颜色值
                    for row in obs_data:
                        if ball_color_value in row:
                            ball_in_view = True
                            break
                
                # 如果观测中有球，给予小奖励
                if ball_in_view:
                    observation_reward[agent_idx if agent_idx < 2 else agent_idx-1] = 0.02
        
        # 计算每个智能体与球的当前距离（仅用于调试或参考）
        current_distances = []
        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            if agent.type == 'agent':
                dist = math.sqrt(
                    (self.agent_pos[agent_idx][0] - self.agent_pos[ball_idx][0])**2 +
                    (self.agent_pos[agent_idx][1] - self.agent_pos[ball_idx][1])**2
                )
                current_distances.append(dist)
        
        # 更新智能体速度
        for id in range(self.agent_num):
            self.agent_list[id].velocity=[self.agent_pos[id][0]-self.agent_last_pos[id][0], 
                                         self.agent_pos[id][1]-self.agent_last_pos[id][1]]
        
        # 检测碰撞和球的方向变化
        if hasattr(self, 'last_ball_velocity'):
            current_ball_velocity = self.agent_list[ball_idx].velocity
            velocity_change = [
                current_ball_velocity[0] - self.last_ball_velocity[0],
                current_ball_velocity[1] - self.last_ball_velocity[1]
            ]
            
            # 对于每个智能体，检查是否发生了有效的碰撞
            for agent_idx in range(self.agent_num):
                agent = self.agent_list[agent_idx]
                if agent.type == 'agent':
                    # 检查智能体和球是否足够近（可能发生碰撞）
                    distance = math.sqrt(
                        (self.agent_pos[agent_idx][0] - self.agent_pos[ball_idx][0])**2 +
                        (self.agent_pos[agent_idx][1] - self.agent_pos[ball_idx][1])**2
                    )
                    if distance <= (agent.r + self.agent_list[ball_idx].r + 1):  # +1为容差值
                        # 根据球的速度变化方向判断是否是有效碰撞
                        if self.agent_pos[agent_idx][0] < 400:  # 左侧智能体
                            if velocity_change[0] > 0:  # 球向右移动（正确踢球）
                                collision_reward[0] = 0.3  # 增加奖励值
                                collision_reward[1] = 0
                            elif velocity_change[0] < 0:
                                collision_reward[0] = 0.05
                                collision_reward[1] = 0
                        else:  # 右侧智能体
                            if velocity_change[0] < 0:  # 球向左移动（正确踢球）
                                collision_reward[1] = 0.3  # 增加奖励值
                                collision_reward[0] = 0
                            elif velocity_change[0] > 0:
                                collision_reward[1] = 0.05
                                collision_reward[0] = 0
                                
        # 记录当前球的速度，用于下一步比较
        self.last_ball_velocity = self.agent_list[ball_idx].velocity.copy()
        
        # 更新上一步位置
        self.agent_last_pos = self.agent_pos.copy()
        
        # 处理终点奖励
        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            if agent.type == 'ball' and agent.finished:
                ball_end_pos = self.agent_pos[agent_idx]
                
        if ball_end_pos is not None and ball_end_pos[0] < 400:
            if self.agent_pos[0][0] < 400:
                return [-0.5, 1.]
            else:
                return [1., -0.5]
        elif ball_end_pos is not None and ball_end_pos[0] > 400:
            if self.agent_pos[0][0] < 400:
                return [1., -0.5]
            else:
                return [-0.5, 1.]
        else:
            # 合并碰撞奖励和观测奖励（不再使用距离奖励）
            total_reward = [
                collision_reward[0] + observation_reward[0],
                collision_reward[1] + observation_reward[1]
            ]
        return total_reward


    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            if agent.type == 'ball' and agent.finished:
                return True

        return False

    def check_win(self):
        if self.done:
            self.ball_end_pos = None
            for agent_idx in range(self.agent_num):
                agent = self.agent_list[agent_idx]
                if agent.type == 'ball' and agent.finished:
                    self.ball_end_pos = self.agent_pos[agent_idx]

        if self.ball_end_pos is None:
            return '-1'
        else:
            if self.ball_end_pos[0] < 400:
                if self.agent_pos[0][0] < 400:
                    return '1'
                else:
                    return '0'
            elif self.ball_end_pos[0] > 400:
                if self.agent_pos[0][0] < 400:
                    return '0'
                else:
                    return '1'

    def render(self, info=None):

        if self.minimap_mode:
            pass
        else:

            if not self.display_mode:
                self.viewer.set_mode()
                self.display_mode = True
                if self.beauty_render:
                    self._load_image()

            self.viewer.draw_background()
            if self.beauty_render:
                self._draw_playground()
                self._draw_energy(self.agent_list)

            for w in self.map['objects']:
                self.viewer.draw_map(w)

            if self.beauty_render:
                self._draw_image(self.agent_pos, self.agent_list, self.agent_theta, self.obs_boundary)
            else:
                self.viewer.draw_ball(self.agent_pos, self.agent_list)
                if self.draw_obs:
                    self.viewer.draw_obs(self.obs_boundary, self.agent_list)

        if self.draw_obs:
            if len(self.obs_list) > 0:
                self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=450, upmost_y=10, gap=130,
                                      energy_width=0 if self.beauty_render else 5)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        # self.viewer.draw_map()

        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)

        for event in pygame.event.get():
            # 如果单击关闭窗口，则退出
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()
        # self.viewer.background.fill((255, 255, 255))


    def _load_image(self):
        self.playground_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/table_hockey/playground.png")).convert_alpha()
        self.playground_image = pygame.transform.scale(self.playground_image, size = (860, 565))

        self.player_1_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/table_hockey/player1.png")).convert_alpha()
        self.player_2_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/table_hockey/player2.png")).convert_alpha()
        self.ball_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/table_hockey/ball.png")).convert_alpha()
        self.player_1_view_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/table_hockey/sight1.png")).convert_alpha()
        self.player_2_view_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/table_hockey/sight2.png")).convert_alpha()

        self.wood_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/board.png")).convert_alpha()
        self.wood_image1 = pygame.transform.scale(self.wood_image, size = (300,170))
        self.wood_image2 = pygame.transform.scale(self.wood_image, size = (70,30))

        self.red_energy_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-red.png")).convert_alpha()
        red_energy_size = self.red_energy_image.get_size()
        self.red_energy_image = pygame.transform.scale(self.red_energy_image, size = (110,red_energy_size[1]*110/red_energy_size[0]))

        self.blue_energy_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-blue.png")).convert_alpha()
        blue_energy_size = self.blue_energy_image.get_size()
        self.blue_energy_image = pygame.transform.scale(self.blue_energy_image, size = (110, blue_energy_size[1]*110/blue_energy_size[0]))

        self.red_energy_bar_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-red-bar.png")).convert_alpha()
        red_energy_bar_size = self.red_energy_bar_image.get_size()
        self.red_energy_bar_image = pygame.transform.scale(self.red_energy_bar_image, size=(85, 10))

        self.blue_energy_bar_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-blue-bar.png")).convert_alpha()
        blue_energy_bar_size = self.blue_energy_bar_image.get_size()
        self.blue_energy_bar_image = pygame.transform.scale(self.blue_energy_bar_image, size=(85, 10))


    def _draw_playground(self):
        loc = (-43,125)
        self.viewer.background.blit(self.playground_image, loc)
        self.viewer.background.blit(self.wood_image1, (400, 0))

        self.viewer.background.blit(self.red_energy_image, (425, 130))
        self.viewer.background.blit(self.blue_energy_image, (555, 130))

    def _draw_energy(self, agent_list):

        # red_energy_bar_size = self.red_energy_bar_image.get_size()
        # blue_energy_bar_size = self.blue_energy_bar_image.get_size()
        # red_energy_bar = pygame.transform.scale(self.red_energy_bar_image, size=(85, 10))
        # blue_energy_bar = pygame.transform.scale(self.blue_energy_bar_image, size=(85, 10))


        start_pos = [448, 136]
        # end_pos = [450+100*remain_energy, 130]
        image = self.red_energy_bar_image
        for agent_idx in range(len(agent_list)):
            if agent_list[agent_idx].type == 'ball':
                continue

            remain_energy = agent_list[agent_idx].energy/agent_list[agent_idx].energy_cap


            self.viewer.background.blit(image, start_pos, [0, 0, 85*remain_energy, 10])

            start_pos[0] += 130
            image = self.blue_energy_bar_image


    def _draw_image(self, pos_list, agent_list, direction_list, view_list):
        assert len(pos_list) == len(agent_list)
        for i in range(len(pos_list)):
            agent = self.agent_list[i]

            t = pos_list[i]
            r = agent_list[i].r
            color = agent_list[i].color
            theta = direction_list[i][0]
            vis = agent_list[i].visibility
            view_back = self.VIEW_BACK*vis if vis is not None else 0
            if agent.type == 'agent':
                if color == self.agent1_color:
                    player_image_size = self.player_1_image.get_size()
                    image= pygame.transform.scale(self.player_1_image, size = (r*2, player_image_size[1]*(r*2)/player_image_size[0]))
                    loc = (t[0]-r ,t[1]-r)

                    view_image = pygame.transform.scale(self.player_1_view_image, size = (vis, vis))
                    rotate_view_image = pygame.transform.rotate(view_image, -theta)

                    new_view_center = [t[0]+(vis/2-view_back)*math.cos(theta*math.pi/180), t[1]+(vis/2-view_back)*math.sin(theta*math.pi/180)]
                    new_view_rect = rotate_view_image.get_rect(center=new_view_center)
                    self.viewer.background.blit(rotate_view_image, new_view_rect)

                    #view player image
                    # player_image_view = pygame.transform.rotate(image, 90)
                    self.viewer.background.blit(image, (470, 90))


                elif color == self.agent2_color:
                    player_image_size = self.player_2_image.get_size()
                    image= pygame.transform.scale(self.player_2_image, size = (r*2, player_image_size[1]*(r*2)/player_image_size[0]))
                    loc = (t[0]-r ,t[1]-r)

                    view_image = pygame.transform.scale(self.player_2_view_image, size = (vis, vis))
                    rotate_view_image = pygame.transform.rotate(view_image, -theta)

                    new_view_center = [t[0]+(vis/2-view_back)*math.cos(theta*math.pi/180), t[1]+(vis/2-view_back)*math.sin(theta*math.pi/180)]
                    new_view_rect = rotate_view_image.get_rect(center=new_view_center)
                    self.viewer.background.blit(rotate_view_image, new_view_rect)

                    # player_image_view = pygame.transform.rotate(image, 90)
                    self.viewer.background.blit(image, (600, 90))

                    # self.viewer.background.blit(image_green, loc)
            elif agent.type == 'ball':
                image = pygame.transform.scale(self.ball_image, size = (r*2, r*2))
                loc = (t[0] - r, t[1] - r)

            # rotate_image = pygame.transform.rotate(image, -theta)

            # new_rect = rotate_image.get_rect(center=image.get_rect(center = t).center)


            self.viewer.background.blit(image, loc)