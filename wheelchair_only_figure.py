from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
import numpy as np
import numpy.linalg as lin
import os
import fym.logging as logging
from wheelchair_module.getmap import GetMap
import pickle


# data load
loglist = os.listdir(r"C:\Users\Jung\Desktop\GitProjects\wheelchair\log")
path = os.path.join('log', loglist[-1
], 'data.h5')
data = logging.load(path)
with open('setting.txt', 'rb') as f:
    setting = pickle.load(f)


a_x_limit = setting['wheelchair']['a_x_limit']
a_y_limit = setting['wheelchair']['a_y_limit']
# data arrangement
time = data["time"]
actuator_reference = data['actuator_reference']
actuator = data['actuator']
wheel_reference_right = data['wheel_reference_right']
wheel_reference_left = data['wheel_reference_left']
wheel_right = data['wheel_right']
wheel_left = data['wheel_left']
body_acc = data['body_acceleration']
wheelchair_speed = [(x + y) / 2 for x, y in zip(wheel_right, wheel_left)]
wheelchair = data["state"]["wheelchair"]
omega = data['state']['dynamic'][:,2]
state_cg = data['state']['dynamic']
state_driver = data['state']['driver']
wheel_vel_right = data['wheel_vel_right']
wheel_vel_left = data['wheel_vel_left']


# plotting on/off
trajectory_with_edited_map = True
trajectory_with_hrvo_vel = True
state_history = True
state_deriv_his = True
control_history = True
wheel_speed_history = True
state_cg_history = True

# figure setting
r_waypoint = 3 * np.pi
r_obstacle = 1 * np.pi
r_moving = 2 * np.pi
head_len = 0.5
head_wid = 0.5
wheelchair_width = 0.5
wheelchair_length = 0.8
draw_step = 2
arrow_length = 2
time_step = time[1] - time[0]
linewidth = 0.5

plt.figure()
plt.plot(wheelchair[:,0], wheelchair[:,1])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('trajectory')
plt.axis('equal')

if state_history:
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time, wheelchair[:,0])
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('x [m]')
    plt.title('state')
    plt.subplot(4, 1, 2)
    plt.plot(time, wheelchair[:,1])
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('y [m]')
    plt.subplot(4, 1, 3)
    plt.plot(time, wheelchair_speed, color='b', label='true', linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.legend(loc='upper right')
    plt.ylabel('v [m/s]')
    plt.subplot(4, 1, 4)
    plt.plot(time, wheelchair[:,2] * 180 / np.pi, color='b', linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('theta [deg]')
    plt.xlabel('time [s]')

if state_deriv_his:
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, body_acc[:,0], color='b')
    plt.plot(time, [a_x_limit[0]]*len(time), color='r', lw=linewidth)
    plt.plot(time, [a_x_limit[1]]*len(time), color='r', lw=linewidth)
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('a_x [m/s^2]')
    plt.title('acceleration and angular velocity')
    plt.subplot(3, 1, 2)
    plt.plot(time, body_acc[:,1])
    plt.plot(time, [a_y_limit[0]]*len(time), color='r', lw=linewidth)
    plt.plot(time, [a_y_limit[1]]*len(time), color='r', lw=linewidth)
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('a_y [m/s^2]')
    plt.subplot(3, 1, 3)
    plt.plot(time, omega * 180 / np.pi)
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('w [deg/s]')
    plt.xlabel('time [s]')

if control_history:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, actuator_reference[:,0], color='r', label='command')
    plt.plot(time, actuator[:,0], color='b', label='true')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.legend(loc='upper right')
    plt.ylabel('actuator right [N]')
    plt.title('actuator history')
    plt.subplot(2, 1, 2)
    plt.plot(time, actuator_reference[:,1], color='r')
    plt.plot(time, actuator[:,1], color='b')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('actuator left [N]')
    plt.xlabel('time [s]')

if wheel_speed_history:
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, wheel_reference_right, color='r', label='ref')
    plt.plot(time, wheel_right, color='b', label='true')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.legend(loc='upper right')
    plt.ylabel('right wheel speed [m/s]')
    plt.title('right/left wheel speed history')
    plt.subplot(2,1,2)
    plt.plot(time, wheel_reference_left, color='r')
    plt.plot(time, wheel_left, color='b')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('left wheel speed [m/s]')
    plt.xlabel('time [s]')

if state_cg_history:
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, state_cg[:,0], color='b')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('v_x_cg [m/s]')
    plt.title('cg point velocity')
    plt.subplot(3, 1, 2)
    plt.plot(time, state_cg[:,1])
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('v_y_cg [m/s]')
    plt.subplot(3, 1, 3)
    plt.plot(time, state_cg[:,2] * 180 / np.pi)
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('w [deg/s]')
    plt.xlabel('time [s]')

plt.figure()
plt.plot(state_driver[:,0], state_driver[:,1])
plt.grid(axis='both', linestyle='--', color='gray')
plt.ylabel('y [m]')
plt.xlabel('x [m]')
plt.title('driver movement')

plt.figure()
plt.subplot(3,1,1)
plt.plot(time, wheel_vel_right[:,0])
plt.title('right wheel velocity')
plt.subplot(3,1,2)
plt.plot(time, wheel_vel_right[:,1])
plt.subplot(3,1,3)
plt.plot(time, wheel_vel_right[:,2])

plt.figure()
plt.subplot(3,1,1)
plt.plot(time, wheel_vel_left[:,0])
plt.title('left wheel velocity')
plt.subplot(3,1,2)
plt.plot(time, wheel_vel_left[:,1])
plt.subplot(3,1,3)
plt.plot(time, wheel_vel_left[:,2])

f_r = data['f_r']
f_l = data['f_l']
f_mck_x = data['f_mck_x']
f_mck_y = data['f_mck_y']

plt.figure()
plt.subplot(4,1,1)
plt.plot(time, f_r)
plt.ylabel('f_r')
plt.subplot(4,1,2)
plt.plot(time, f_l)
plt.ylabel('f_l')
plt.subplot(4,1,3)
plt.plot(time, f_mck_x)
plt.ylabel('f_mck_x')
plt.subplot(4,1,4)
plt.plot(time, f_mck_y)
plt.ylabel('f_mck_y')

f_cr_x = data['f_cr_x']
f_cl_x = data['f_cl_x']
f_cr_y = data['f_cr_y']
f_cl_y = data['f_cl_y']

plt.figure()
plt.subplot(4,1,1)
plt.plot(time, f_cr_x)
plt.ylabel('f_cr_x')
plt.subplot(4,1,2)
plt.plot(time, f_cl_x)
plt.ylabel('f_cl_x')
plt.subplot(4,1,3)
plt.plot(time, f_cr_y)
plt.ylabel('f_cr_y')
plt.subplot(4,1,4)
plt.plot(time, f_cl_y)
plt.ylabel('f_cl_y')

f_cr_R = data['f_cr_R']
f_cl_R = data['f_cl_R']
f_cr_T = data['f_cr_T']
f_cl_T = data['f_cl_T']

plt.figure()
plt.subplot(4,1,1)
plt.plot(time, f_cr_R)
plt.ylabel('f_cr_R')
plt.subplot(4,1,2)
plt.plot(time, f_cl_R)
plt.ylabel('f_cl_R')
plt.subplot(4,1,3)
plt.plot(time, f_cr_T)
plt.ylabel('f_cr_T')
plt.subplot(4,1,4)
plt.plot(time, f_cl_T)
plt.ylabel('f_cl_T')

alpha_right = data['state']['caster_right']
alpha_left = data['state']['caster_left']
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(time, alpha_right[:,0] * 180 / np.pi)
plt.ylabel('alpha_right [deg]')
plt.title('caster angle')
plt.subplot(4,1,2)
plt.plot(time, alpha_left[:,0] * 180 / np.pi)
plt.ylabel('alpha_left [deg]')
plt.subplot(4, 1, 3)
plt.plot(time, alpha_right[:,1] * 180 / np.pi)
plt.ylabel('alpha_dot_right [deg/s]')
plt.subplot(4,1,4)
plt.plot(time, alpha_left[:,1] * 180 / np.pi)
plt.ylabel('alpha_dot_left [deg/s]')


e_right = data['error_right']
e_left = data['error_left']
plt.figure()
plt.subplot(2,1,1)
plt.plot(time, e_right)
plt.title('controller error')
plt.subplot(2,1,2)
plt.plot(time, e_left)





plt.show()

