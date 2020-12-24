from PIL import Image
import numpy as np
import numpy.linalg as lin
import time
import networkx as nx
from shapely.geometry import LineString, Point
import pickle

from wheelchair_module.getmap import GetMap
from wheelchair_only_env import WheelchairEnv
from wheelchair_module.two_wheels_robot import TwoWheelsRobot3Dof

from fym import logging

logger = logging.Logger()
start_time = time.time()

# setting
setting = {
    'env': {
        'final_time': 100,
        'time_step': 0.01,
        'max_v': 1,
        'control_gain': 5,
        'control_tau': 0.1074,
        'robust_time': 0.35,
        'goal_check': 0.1,
        'attitude_check': 1 * np.pi / 180,
        'driver_mass': 5,
        'driver_damper': 2,
        'driver_spring': 1,
        'friction': {
            'rate': 0.05,
            'saturation': 0.2
        },
        'PID': np.array([10, 1, 0.1])
    },
    'wheelchair': {
        'initial_state': [0, 0, np.pi / 2],
        'a_x_limit': [-0.1, 0.5],
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
        'saturation': 2
    },
    'obstacle_detection': {
        'detection_range': 5,
        'detection_angle': 360 * np.pi / 180,
        'multi_obstacle_point': 'on'
    },
    'caster': {
        'J': 0.5,
        'length': 0.1,
        'mu_T': 0.1,
        'mu_R': 0.05,
        'initial_state': [[5 * np.pi / 180, 0], [0 * np.pi / 180, 0]]
    }
}

env = WheelchairEnv(setting)

# reset
obs = env.reset()

# simulation
while 1:            
    next_obs, _, done, info = env.step(1)

    logger.record(**info)

    if done:
        break
    obs = next_obs

logger.record(**info)
env.close()
logger.close()

with open('setting.txt', 'wb') as f:
    pickle.dump(setting, f)

simulation_time = time.time() - start_time
print('simulation time = ', simulation_time, '[s]')

