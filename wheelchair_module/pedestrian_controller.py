import numpy as np
import numpy.linalg as lin


def pedestrian_controller(obs, force, setting):
    v, theta = obs[2], obs[3]
    
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    vel_r = np.array([vx, vy]) + force * setting['env']['time_step']
    v_r = lin.norm(vel_r)
    v_r = np.clip(v_r, -setting['env']['max_v'], setting['env']['max_v'])
    theta_r = np.arctan2(vel_r[1], vel_r[0])

    control = np.array([v_r, theta_r])
    return control