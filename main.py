from PIL import Image
import numpy as np
import numpy.linalg as lin
import time
import networkx as nx
from shapely.geometry import LineString, Point
import pickle

from wheelchair_module.getmap import GetMap
from wheelchair_env import WheelchairEnv
from wheelchair_env import WheelchairWithMovingObstacleEnv
from wheelchair_module.two_wheels_robot import TwoWheelsRobot3Dof
from wheelchair_module.wrap_angle import wrap_angle

from fym import logging

logger = logging.Logger()
start_time = time.time()


#scenario = 'hospital'
#scenario = 'hospital2'
#scenario = 'toy_corner'
#scenario = 'toy_corner_wo_obstacle'
#scenario = 'toy_diagonal_wo_obstacle'
#scenario = 'toy_straight_wo_obstacle'
scenario = 'toy_for_journal'

#speed_design_method = 'maximum_speed'
speed_design_method = 'suggested'

#hrvo_method = 'no constraint'
hrvo_method = 'suggested'

#K_adapt_method = 'constant'
K_adapt_method = 'suggested'

if scenario == 'hospital':
    r_plant = 1
    r_static = 1
    r_moving = 1
    map_size = [200, 200]
    start = 7
    end = 9
    goal_check = 1
    final_attitude = -90 * np.pi / 180
    # moving obstacle setting
    moving_obstacle_on_off = 1

    # # 0 -> 3
    # moving_obstacle_start = [11, 1, 2]
    # moving_obstacle_end = [12, 0, 1]
    # pedestrian_speed = [1, 0.8, 0.6]
    # pedestrian_tau = [0.1, 1, 2]
    # pedestrian_start = [48, 50, 35]
    # pedestrian_attitude = np.array([0, -70, -45]) * np.pi / 180
    
    # # 3 -> 7
    # moving_obstacle_start = [4, 13, 6]
    # moving_obstacle_end = [2, 14, 4]
    # pedestrian_speed = [1, 0.7, 0.6]
    # pedestrian_tau = [0.1, 1, 1]
    # pedestrian_start = [35, 113, 40]
    # pedestrian_attitude = np.array([-135, 0, -135]) * np.pi / 180

    # 7 -> 9
    moving_obstacle_start = [13, 2, 10]
    moving_obstacle_end = [14, 5, 2]
    pedestrian_speed = [0.6, 0.8, 1]
    pedestrian_tau = [1, 1, 0.1]
    pedestrian_start = [55, 125, 160]
    pedestrian_attitude = np.array([0, 60, 0]) * np.pi / 180
        

    n_obstacle = len(moving_obstacle_start)

    # hospital map
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap1, gap2, gap2, gap2, gap3, gap2, gap2, gap2, gap2, gap2]
    gap_waypoint = [gap3]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
    image_waypoint_set = [Image.open('C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/waypoint.png')]
    waypoint_set = get_map.get_waypoint(image_waypoint_set)
    G = nx.Graph()
    # G.add_edges_from([(0, 1), (1, 11), (2, 11), (2, 3), (2, 4), (11, 4), (4, 5), (5, 6), (6, 7), (8, 9), (8, 10), (10, 2)])
    G.add_edges_from([(0, 1), (1, 4), (2, 1), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (8, 9), (8, 10), (10, 2), (11, 12), (13, 14)])
    waypoint_order = nx.shortest_path(G, source=start, target=end)
elif scenario == 'hospital2':
    r_plant = 1
    r_static = 1
    r_moving = 1
    map_size = [100, 200]
    start = 0
    end = 1
    goal_check = 1
    final_attitude = -80 * np.pi / 180
    # moving obstacle setting
    moving_obstacle_on_off = 0

    moving_obstacle_start = [2, 7, 9, 4]
    moving_obstacle_end = [1, 8, 7, 3]
    pedestrian_speed = [1, 0.7, 0.7, 0.7]
    pedestrian_tau = [0.1, 1, 2, 1]
    pedestrian_start = [50, 80, 97, 150]
    pedestrian_attitude = np.array([60, 35, 150, 160]) * np.pi / 180
    n_obstacle = len(moving_obstacle_start)

    # hospital map
    gap1, gap2, gap3 = [50, 50], [120, 120], [80, 80]
    gap_set = [gap1, gap2, gap2, gap2, gap2, gap2]
    gap_waypoint = [gap3]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/obstacle2_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
    image_waypoint_set = [Image.open('C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/waypoint2.png')]
    waypoint_set = get_map.get_waypoint(image_waypoint_set)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (7, 9)])
    waypoint_order = nx.shortest_path(G, source=start, target=end)
elif scenario == 'toy_corner':
    r_plant = 3
    r_static = 3
    r_moving = 1
    # map setting
    map_size = [100, 100]
    start = 0
    end = 2
    final_attitude = 90 * np.pi / 180
    # moving obstacle setting
    moving_obstacle_on_off = 1
    moving_obstacle_start = [2]
    moving_obstacle_end = [0]
    n_obstacle = len(moving_obstacle_start)

    # hospital map
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap3, gap2]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_corner_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
    image_waypoint_set = [Image.open('C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_corner_waypoint.png')]
    waypoint_set = get_map.get_waypoint(image_waypoint_set)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    waypoint_order = nx.shortest_path(G, source=start, target=end)
elif scenario == 'toy_corner_wo_obstacle':
    r_plant = 1
    r_static = 1
    r_moving = 1
    final_attitude = 90 * np.pi / 180
    # map setting
    map_size = [100, 100]
    start = 0
    end = 2
    # moving obstacle setting
    moving_obstacle_on_off = 1
    moving_obstacle_start = [2]
    moving_obstacle_end = [0]
    n_obstacle = len(moving_obstacle_start)

    # hospital map
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap3]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_corner_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
    image_waypoint_set = [Image.open('C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_corner_waypoint.png')]
    waypoint_set = get_map.get_waypoint(image_waypoint_set)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    waypoint_order = nx.shortest_path(G, source=start, target=end)
elif scenario == 'toy_diagonal_wo_obstacle':
    r_plant = 1
    r_static = 1
    r_moving = 1
    final_attitude = 90 * np.pi / 180
    # map setting
    map_size = [100, 100]
    start = 0
    end = 2
    # moving obstacle setting
    moving_obstacle_on_off = 0
    moving_obstacle_start = [1]
    moving_obstacle_end = [0]
    n_obstacle = len(moving_obstacle_start)

    # hospital map
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap3]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_diagonal_wo_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
    image_waypoint_set = [Image.open('C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_diagonal_wo_obstacle_waypoint.png')]
    waypoint_set = get_map.get_waypoint(image_waypoint_set)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    waypoint_order = nx.shortest_path(G, source=start, target=end)
elif scenario == 'toy_straight_wo_obstacle':
    r_plant = 1
    r_static = 1
    r_moving = 1
    map_size = [100, 100]
    start = 0
    end = 2
    goal_check = 0.5
    final_attitude = 90 * np.pi / 180
    # moving obstacle setting
    moving_obstacle_on_off = 1
    moving_obstacle_start = [1]
    moving_obstacle_end = [0]
    n_obstacle = len(moving_obstacle_start)

    # hospital map
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap2]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_straight_wo_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
    image_waypoint_set = [Image.open('C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_straight_wo_obstacle_waypoint.png')]
    waypoint_set = get_map.get_waypoint(image_waypoint_set)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    waypoint_order = nx.shortest_path(G, source=start, target=end)
elif scenario == 'toy_for_journal':
    r_plant = 1
    r_static = 0.5
    r_moving = 0.5
    map_size = [32, 40]
    start = 0
    end = 3
    goal_check = 1
    final_attitude = 90 * np.pi / 180
    # moving obstacle setting
    moving_obstacle_on_off = 1

    moving_obstacle_start = [2, 3, 4]
    moving_obstacle_end = [9, 8, 5]
    pedestrian_speed = [1, 0.5, 0.4]
    pedestrian_tau = [0.1, 2, 3]

    # hrvo
    # pedestrian_start = [8, 0, 17]

    # proposed
    pedestrian_start = [25, 13.5, 34.2]  

    pedestrian_attitude = np.array([0 , -90, 0]) * np.pi / 180
    n_obstacle = len(moving_obstacle_start)

    # hospital map
    gap1, gap2, gap3 = [80, 80], [120, 120], [80, 80]
    gap_set = [gap1, gap2]
    gap_waypoint = [gap3]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/journal_obs_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
    image_waypoint_set = [Image.open('C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/journal_waypoint.png')]
    waypoint_set = get_map.get_waypoint(image_waypoint_set)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (6, 4), (1, 8), (4, 5), (1,9), (2, 9)])
    waypoint_order = nx.shortest_path(G, source=start, target=end)

# setting
setting = {
    'scenario': scenario,
    'map_size': map_size,
    'moving_obstacle_onoff': moving_obstacle_on_off,
    'n_obstacle': n_obstacle,
    'waypoint_set': waypoint_set,
    'env': {
        'final_time': 400,
        'time_step': 0.01,
        'max_v': 1,
        'max_w': 90 * np.pi / 180,
        'K_max': 10,
        'K_min': 0,
        'control_tau': 3,
        'robust_time': 0.4,
        'goal_check': goal_check,
        'attitude_check': 1 * np.pi / 180,
        'speed_design_method': speed_design_method,
        'hrvo_method': hrvo_method,
        'K_adapt_method': K_adapt_method,
        'moving_onoff': moving_obstacle_on_off,
        'driver_mass': 5,
        'driver_damper': 2,
        'driver_spring': 1,
        'friction': {
            'rate': 0.05,
            'saturation': 0.1
        },
        'PID': np.array([10, 0.1, 0.5]),
        'pedestrian_speed': pedestrian_speed,
        'pedestrian_tau': pedestrian_tau
    },
    'wheelchair': {
        'initial_state': [waypoint_set[start][0], waypoint_set[start][1], 90 * np.pi / 180],
        'waypoint_coordinate': [waypoint_set[i] for i in waypoint_order],
        'waypoint_check': [True] * len(waypoint_order),
        'a_x_limit': [-0.1, 0.1],
        'a_y_limit': [-0.1, 0.1],
        'width': 0.5,
        'length': 0.5,
        'cg': 0.2,
        'mass': 60,
        'mu': 0.05,
        'J': 1
    },
    'actuator': {
        'initial_state': np.array([0]).astype('float'),
        'tau': 0.1,
        'saturation': 4
    },
    'moving_obstacle': {
        f'{i}': {
            'initial_state': [waypoint_set[moving_obstacle_start[i]][0], waypoint_set[moving_obstacle_start[i]][1], 0, pedestrian_attitude[i]],
            'waypoint_coordinate': [waypoint_set[j] for j in nx.shortest_path(G, source=moving_obstacle_start[i], target=moving_obstacle_end[i])],
            'waypoint_check': [True] * len(nx.shortest_path(G, source=moving_obstacle_start[i], target=moving_obstacle_end[i]))
        } for i in np.arange(n_obstacle)
    },
    'static_obstacle_set': static_obstacle_set,
    'obstacle_detection': {
        'detection_range': 4,
        'detection_angle': 180 * np.pi / 180,
        'multi_obstacle_point': 'on'
    },
    'hrvo': {
        'r_plant': r_plant,
        'r_static': r_static,
        'r_moving': r_moving,
        'v_step': 10,
        'angle_step': 360,
        'method': 'hrvo',
        'weight': [9, 1],
        'modified': True
    },
    'caster': {
        'J': 1,
        'length': 0.1,
        'mu_T': 0.1,
        'mu_R': 0.05,
        'initial_state': [[0 * np.pi / 180, 0], [0 * np.pi / 180, 0]]
    }
}

angle_converge = 30
angle_converge_count = 0
if moving_obstacle_on_off:
    env = WheelchairWithMovingObstacleEnv(setting)
else:
    env = WheelchairEnv(setting)

# reset
obs = env.reset()
setting['wheelchair']['waypoint_check'][0] = False
for i in np.arange(n_obstacle):
    setting['moving_obstacle'][f'{i}']['waypoint_check'][0] = False

# simulation
while 1:
    # waypoint setting
    if any(setting['wheelchair']['waypoint_check']):
        index = setting['wheelchair']['waypoint_check'].index(True)
        goal = setting['wheelchair']['waypoint_coordinate'][index]
        if index + 1 < len(setting['wheelchair']['waypoint_check']):
            direction = np.array(setting['wheelchair']['waypoint_coordinate'][index + 1]) - np.array(goal)
            attitude = np.arctan2(direction[1], direction[0])
        else:
            '''
            direction = np.array(goal) - np.array(setting['wheelchair']['waypoint_coordinate'][index - 1])
            attitude = np.arctan2(direction[1], direction[0])
            '''
            attitude = final_attitude
            env.destination = True
        wheelchair_action = [goal, attitude]
    else:
        env.distance_condition = True
        env.angle_condition = True
        if env.speed_condition:
            break

    action = {'wheelchair': wheelchair_action}

    if moving_obstacle_on_off:
        for i in np.arange(n_obstacle):
            if any(setting['moving_obstacle'][f'{i}']['waypoint_check']):
                index = setting['moving_obstacle'][f'{i}']['waypoint_check'].index(True)
                if env.time >= pedestrian_start[i]:
                    goal = setting['moving_obstacle'][f'{i}']['waypoint_coordinate'][index]
                    action[f'pedestrian_{i}'] = [goal, 0]
                else:
                    action[f'pedestrian_{i}'] = []
            else:
                action[f'pedestrian_{i}'] = []
            
    next_obs, _, done, info = env.step(action)

    logger.record(**info)

    if done:
        break
    obs = next_obs

    if lin.norm(action['wheelchair'][0] - obs['wheelchair'][0:2]) <= setting['env']['goal_check']:
        env.distance_condition = True
    if abs(wrap_angle(obs['wheelchair'][2] - action['wheelchair'][1])) < setting['env']['attitude_check'] and env.distance_condition:
        env.angle_condition = True
        # angle_converge_count += 1
        # if angle_converge_count >= angle_converge:
        #     env.angle_condition = True
        #     angle_converge_count = 0
    if speed_design_method == 'maximum_speed':
        env.angle_condition = True
    if all([env.distance_condition, env.angle_condition]) and any(setting['wheelchair']['waypoint_check']):
        index = setting['wheelchair']['waypoint_check'].index(True)
        setting['wheelchair']['waypoint_check'][index] = False
        env.distance_condition = False
        env.angle_condition = False
        env.trigger = 0
    

    if moving_obstacle_on_off:
        for i in np.arange(n_obstacle):
            if action[f'pedestrian_{i}']:
                if lin.norm(action[f'pedestrian_{i}'][0] - obs[f'pedestrian_{i}'][0:2]) < setting['env']['goal_check']:
                    index = setting['moving_obstacle'][f'{i}']['waypoint_check'].index(True)
                    setting['moving_obstacle'][f'{i}']['waypoint_check'][index] = False

logger.record(**info)
env.close()
logger.close()

with open('setting.txt', 'wb') as f:
    pickle.dump(setting, f)

simulation_time = time.time() - start_time
print('simulation time = ', simulation_time, '[s]')

