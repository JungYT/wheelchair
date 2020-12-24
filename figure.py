from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
import numpy as np
import numpy.linalg as lin
import os
import fym.logging as logging
from wheelchair_module.getmap import GetMap
import pickle
from wheelchair_module.wrap_angle import wrap_angle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# data load
past = -1
loglist = os.listdir(r"C:\Users\Jung\Desktop\GitProjects\wheelchair\log")
path = os.path.join('log', loglist[past
], 'data.h5')
data = logging.load(path)
with open('setting.txt', 'rb') as f:
    setting = pickle.load(f)

scenario = setting['scenario']
map_size = setting['map_size']
moving_obstacle_onoff = setting['moving_obstacle_onoff']
n_obstacle = setting['n_obstacle']
a_x_limit = setting['wheelchair']['a_x_limit']
a_y_limit = setting['wheelchair']['a_y_limit']
waypoint = setting['waypoint_set']
# data arrangement
time = data["time"]
design_speed = data['designed_speed']
design_angle = data['designed_angle']
hrvo_speed = data['v_hrvo']
hrvo_theta = data['theta_hrvo']
ispass = data['ispass']
actuator_reference = data['actuator_reference']
actuator = data['actuator']
wheel_reference_right = data['wheel_reference_right']
wheel_reference_left = data['wheel_reference_left']
wheel_right = data['wheel_right']
wheel_left = data['wheel_left']
body_acc = data['body_acceleration']
v_f = data['final_speed']
wheelchair_speed = [(x + y) / 2 for x, y in zip(wheel_right, wheel_left)]
wheelchair = data["state"]["wheelchair"]
omega = data['state']['dynamic'][:,2]
speed_design_mode = data['speed_design_mode']
state_cg = data['state']['dynamic']
state_driver = data['state']['driver']
wheel_vel_right = data['wheel_vel_right']
wheel_vel_left = data['wheel_vel_left']
detection = data['obstacle_detection']

color_obs = ['b', 'g', 'c', 'm']

# map
if scenario == 'hospital':
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap1, gap2, gap2, gap2, gap3, gap2, gap2, gap2, gap2, gap2]
    gap_waypoint = [gap3]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
elif scenario == 'hospital2':
    gap1, gap2, gap3 = [50, 50], [120, 120], [80, 80]
    gap_set = [gap1, gap2, gap2, gap2, gap2, gap2]
    gap_waypoint = [gap3]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/obstacle2_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)

elif scenario == 'toy_corner':
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap3, gap2]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(len(gap_set)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_corner_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
elif scenario == 'toy_corner_wo_obstacle':
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap3]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_corner_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
elif scenario == 'toy_diagonal_wo_obstacle':
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap3]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_diagonal_wo_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
elif scenario == 'toy_straight_wo_obstacle':
    gap1, gap2, gap3 = [30, 30], [120, 120], [80, 80]
    gap_set = [gap2]
    gap_waypoint = [gap2]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/toy_straight_wo_obstacle_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)
elif scenario == 'toy_for_journal':
    gap1, gap2, gap3 = [80, 80], [120, 120], [80, 80]
    gap_set = [gap1, gap2]
    gap_waypoint = [gap3]
    get_map = GetMap(map_size, gap_set, gap_waypoint)
    image_set = []
    for i in np.arange(np.size(gap_set, 0)):
        image_set.append(Image.open(f'C:/Users/Jung/Desktop/GitProjects/wheelchair/image_for_map/journal_obs_{i}.png'))
    static_obstacle_set = get_map.get_obstacle_set(image_set)

# plotting on/off
trajectory_with_edited_map = True
trajectory_with_hrvo_vel = True
state_history = True
state_deriv_his = True
control_history = True
wheel_speed_history = True
state_cg_history = True
# moving_obstacle_onoff = False

# figure setting
r_waypoint = 2 * np.pi
r_obstacle = 1 * np.pi
r_moving = 2 * np.pi
head_len = 0.5
head_wid = 0.5
wheelchair_width = 0.5
wheelchair_length = 0.5
draw_step = 3
arrow_length = 2
time_step = time[1] - time[0]
linewidth = 1
fontsize = 15
ticksize = 13

if trajectory_with_edited_map:
    # plt.figure(figsize=(6, 5.5))
    plt.figure()
    for i in np.arange(np.size(static_obstacle_set, 0)):
        plt.plot(*static_obstacle_set[i].exterior.xy, color='k')
    for i in np.arange(np.size(waypoint, 0)):
        plt.scatter(waypoint[i][0], waypoint[i][1], s=r_waypoint, color='k')
    # for i in [10]:
    #     plt.scatter(waypoint[i][0], waypoint[i][1], s=r_waypoint, color='k')
    for i in np.arange(0, len(time), int(draw_step / time_step)):
        p = wheelchair[i, 0] - (np.cos(wheelchair[i,2]-np.pi/2)*wheelchair_width/2 - np.sin(wheelchair[i, 2]-np.pi/2)*wheelchair_length/2)
        q = wheelchair[i, 1] - (np.sin(wheelchair[i,2]-np.pi/2)*wheelchair_width/2 + np.cos(wheelchair[i, 2]-np.pi/2)*wheelchair_length/2)
        plt.gca().add_patch(plt.Rectangle((p, q), wheelchair_width, wheelchair_length, angle=wheelchair[i,2]*180/np.pi-90, linewidth=1, fill=False, edgecolor='r'))
        plt.arrow(wheelchair[i,0], wheelchair[i,1], arrow_length*wheelchair_speed[i] * np.cos(wheelchair[i, 2]), arrow_length*wheelchair_speed[i] * np.sin(wheelchair[i,2]), length_includes_head=True, head_width=head_wid, head_length=head_len, color='r')
    p = wheelchair[-1, 0] - (np.cos(wheelchair[-1,2]-np.pi/2)*wheelchair_width/2 - np.sin(wheelchair[-1, 2]-np.pi/2)*wheelchair_length/2)
    q = wheelchair[-1, 1] - (np.sin(wheelchair[-1,2]-np.pi/2)*wheelchair_width/2 + np.cos(wheelchair[-1, 2]-np.pi/2)*wheelchair_length/2)
    plt.gca().add_patch(plt.Rectangle((p, q), wheelchair_width, wheelchair_length, angle=wheelchair[-1,2]*180/np.pi-90, linewidth=1, edgecolor='r', fill=False, label='wheelchair'))
    plt.arrow(wheelchair[-1,0], wheelchair[-1,1], arrow_length*wheelchair_speed[-1] * np.cos(wheelchair[-1, 2]), arrow_length*wheelchair_speed[-1] * np.sin(wheelchair[-1,2]), length_includes_head=True, head_width=head_wid, head_length=head_len, color='r')
    if moving_obstacle_onoff:
        for i in np.arange(0, np.size(time), int(draw_step / time_step)):
            for j in np.arange(n_obstacle):
                p = data['state'][f'pedestrian_{j}'][i][0]
                q = data['state'][f'pedestrian_{j}'][i][1]
                circle = plt.Circle((p, q), 0.3, color=color_obs[j], fill=False)
                plt.gca().add_patch(circle)
                '''
                p = data['state'][f'pedestrian_{j}'][i][0]
                q = data['state'][f'pedestrian_{j}'][i][1]
                plt.scatter(p, q, s=r_moving, edgecolors='g', facecolors='none')
                #plt.gca().add_patch(plt.Circle((p, q), r_pedestrian, linewidth=1, fill=False, edgecolor='g'))
                '''
        for j in np.arange(n_obstacle):
            p = data['state'][f'pedestrian_{j}'][-1][0]
            q = data['state'][f'pedestrian_{j}'][-1][1]
            circle = plt.Circle((p, q), 0.3, color=color_obs[j], fill=False, label=f'pedestrian{j+1}')
            plt.gca().add_patch(circle)
            # plt.scatter(p, q, s=r_moving, edgecolors='g', facecolors='none')
    plt.xlabel('x [m]', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.ylabel('y [m]', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    # plt.title('trajectory with wheelchair & moving obstacle')
    plt.title('Trajectory of the wheelchair', fontsize=fontsize)
    #plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join('log', loglist[past], 'trajectory.png'), bbox_inches='tight')

# if trajectory_with_hrvo_vel:
#     plt.figure()
#     for i in np.arange(np.size(static_obstacle_set, 0)):
#         plt.plot(*static_obstacle_set[i].exterior.xy, color='k')
#     for i in np.arange(np.size(waypoint, 0)):
#         plt.scatter(waypoint[i][0], waypoint[i][1], s=r_waypoint, color='b')
#     plt.plot(wheelchair[:,0], wheelchair[:,1], '--', color='r')
#     for i in np.arange(0, np.size(time), int(draw_step / time_step)):
#         plt.arrow(wheelchair[i,0], wheelchair[i,1], hrvo_speed[i]*np.cos(hrvo_theta[i]), hrvo_speed[i]*np.sin(hrvo_theta[i]), length_includes_head=True, head_width=head_wid, head_length=head_len)
#     plt.arrow(wheelchair[-1,0], wheelchair[-1,1], hrvo_speed[-1]*np.cos(hrvo_theta[-1]), hrvo_speed[-1]*np.sin(hrvo_theta[-1]), length_includes_head=True, head_width=head_wid, head_length=head_len)
#     plt.xlabel('x [m]', fontsize=fontsize)
#     plt.ylabel('y [m]', fontsize=fontsize)
#     plt.title('Trajectory with hrvo vel', fontsize=fontsize)
#     plt.gca().set_aspect('equal', adjustable='box')

# if state_history:
#     # plt.figure(figsize=(7.5, 7))
#     plt.figure()

#     plt.subplot(3, 1, 1)
#     plt.plot(time, wheelchair_speed, color='b', label='true')
#     plt.plot(time, design_speed, color='g', label='desired', linestyle='-.')
#     plt.plot(time, hrvo_speed, color='r', label='mHRVO', linestyle='--')
#     # plt.plot(time, v_f, color='k', label='final speed', linestyle='--')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     # plt.legend(loc='upper right')
#     #plt.legend(loc='upper left')
#     plt.legend(loc='best')
#     plt.ylabel('$v \; [m/s]$', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)
#     plt.tick_params(axis='x', colors=(0,0,0,0))
#     plt.title('Speed, attitude, and obstacle detection', fontsize=fontsize)

#     plt.subplot(3, 1, 2)
#     plt.plot(time, wheelchair[:,2] * 180 / np.pi, color='b')
#     plt.plot(time, np.unwrap(design_angle) * 180 / np.pi, color='g', linestyle='-.')
#     plt.plot(time, np.unwrap(hrvo_theta) * 180 / np.pi, color='r', linestyle='--')
#     # plt.plot(time, hrvo_theta * 180 / np.pi, color='r', linestyle='--')
#     # plt.plot(time, wrap_angle(np.unwrap(hrvo_theta)) * 180 / np.pi, color='r', linestyle='--')
#     # plt.plot(time, wrap_angle(hrvo_theta) * 180 / np.pi, color='r', linestyle='--')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('$ \\theta \; [deg]$', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)
#     plt.tick_params(axis='x', colors=(0,0,0,0))
    
#     plt.subplot(3, 1, 3)
#     plt.plot(time, detection, color='r')
#     plt.grid(axis='x', linestyle='--', color='gray')
#     plt.ylabel('Detection', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)
#     plt.xlabel('time [s]', fontsize=fontsize)
#     plt.xticks(fontsize=ticksize)

#     plt.savefig(os.path.join('log', loglist[past], 'state.png'),          bbox_inches='tight')

if state_history:
    plt.figure(figsize=(7, 6))
    # plt.figure()

    plt.subplot(4, 1, 1)
    plt.plot(time, wheelchair_speed, color='b', label='true')
    plt.plot(time, design_speed, color='g', label='desired', linestyle="-.")
    plt.plot(time, hrvo_speed, color='r', label='mHRVO', linestyle='--')
    # plt.plot(time, v_f, color='k', label='final speed', linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    # plt.legend(loc='upper right')
    #plt.legend(loc='upper left')
    plt.legend(loc='best')
    plt.ylabel('$v \; [m/s]$', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.title('Speed, attitude, and body accelerations', fontsize=fontsize)
    plt.tick_params(axis='x', which='both', right=False, left=False, top=False, bottom=False)
    plt.tick_params(axis='x', colors=(0,0,0,0))

    plt.subplot(4, 1, 2)
    plt.plot(time, wheelchair[:,2] * 180 / np.pi, color='b')
    plt.plot(time, np.unwrap(design_angle) * 180 / np.pi, color='g', linestyle='-.')
    plt.plot(time, np.unwrap(hrvo_theta) * 180 / np.pi, color='r', linestyle='--')
    # plt.plot(time, hrvo_theta * 180 / np.pi, color='r', linestyle='--')
    # plt.plot(time, wrap_angle(np.unwrap(hrvo_theta)) * 180 / np.pi, color='r', linestyle='--')
    # plt.plot(time, wrap_angle(hrvo_theta) * 180 / np.pi, color='r', linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('$ \\theta \; [deg]$', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.tick_params(axis='x', colors=(0,0,0,0))

    plt.subplot(4, 1, 3)
    plt.plot(time, body_acc[:,0], color='b', label='true')
    plt.plot(time, [a_x_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--', label='boundary')
    plt.plot(time, [a_x_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('$a_x \; [m/s^2]$', fontsize=fontsize-1)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='best')
    plt.tick_params(axis='x', colors=(0,0,0,0))

    plt.subplot(4, 1, 4)
    plt.plot(time, body_acc[:,1], color='b')
    plt.plot(time, [a_y_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--')
    plt.plot(time, [a_y_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('$a_y \; [m/s^2]$', fontsize=fontsize-1)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)

    plt.savefig(os.path.join('log', loglist[past], 'state.png'), bbox_inches='tight')


if state_deriv_his:
    
    # plt.figure()
    plt.figure(figsize=(7, 6))

    # plt.subplot(3, 1, 1)
    # plt.plot(time, body_acc[:,0], color='b')
    # plt.plot(time, [a_x_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--')
    # plt.plot(time, [a_x_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
    # plt.grid(axis='both', linestyle='--', color='gray')
    # plt.ylabel('$a_x \; [m/s^2]$', fontsize=fontsize)
    # plt.yticks(fontsize=ticksize)
    # plt.title('Acceleration and angular rate', fontsize=fontsize)
    # plt.subplot(3, 1, 2)
    # plt.plot(time, body_acc[:,1], color='b')
    # plt.plot(time, [a_y_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--')
    # plt.plot(time, [a_y_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
    # plt.grid(axis='both', linestyle='--', color='gray')
    # plt.ylabel('$a_y \; [m/s^2]$', fontsize=fontsize)
    # plt.yticks(fontsize=ticksize)

    plt.subplot(4, 1, 1)
    plt.plot(time, body_acc[:,0], color='b', label='true')
    plt.plot(time, [a_x_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--', label='boundary')
    plt.plot(time, [a_x_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('$a_x \; [m/s^2]$', fontsize=fontsize-1)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='best')
    plt.title('Acceleration, angular rate, and angular rate gain', fontsize=fontsize)
    plt.tick_params(axis='x', colors=(0,0,0,0))

    plt.subplot(4, 1, 2)
    plt.plot(time, body_acc[:,1], color='b')
    plt.plot(time, [a_y_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--')
    plt.plot(time, [a_y_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('$a_y \; [m/s^2]$', fontsize=fontsize-1)
    plt.yticks(fontsize=ticksize)
    plt.tick_params(axis='x', colors=(0,0,0,0))

    plt.subplot(4, 1, 3)
    plt.plot(time, omega * 180 / np.pi, color='r')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.ylabel('$\\omega \; [deg/s]$', fontsize=fontsize-1)
    plt.yticks(fontsize=ticksize)
    # plt.title('Angular rate and angular rate gain', fontsize=fontsize)
    plt.tick_params(axis='x', colors=(0,0,0,0))
    
    plt.subplot(4, 1, 4)
    plt.plot(time, data['K'], color='r')
    plt.ylabel('K', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.grid(axis='both', linestyle='--', color='gray')

    plt.xlabel('time [s]', fontsize=fontsize)

    plt.savefig(os.path.join('log', loglist[past], 'K.png'), bbox_inches='tight')

# if control_history:
#     plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.plot(time, actuator_reference[:,0], color='r', label='command')
#     plt.plot(time, actuator[:,0], color='b', label='true')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.legend(loc='upper right')
#     plt.ylabel('actuator right [N]', fontsize=fontsize)
#     plt.title('Actuator history', fontsize=fontsize)
#     plt.subplot(2, 1, 2)
#     plt.plot(time, actuator_reference[:,1], color='r')
#     plt.plot(time, actuator[:,1], color='b')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('actuator left [N]', fontsize=fontsize)
#     plt.xlabel('time [s]', fontsize=fontsize)

# if wheel_speed_history:
#     plt.figure()
#     plt.subplot(2,1,1)
#     plt.plot(time, wheel_reference_right, color='r', label='ref')
#     plt.plot(time, wheel_right, color='b', label='true')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.legend(loc='upper right')
#     plt.ylabel('right wheel speed [m/s]', fontsize=fontsize)
#     plt.title('Right/Left wheel speed history', fontsize=fontsize)
#     plt.subplot(2,1,2)
#     plt.plot(time, wheel_reference_left, color='r')
#     plt.plot(time, wheel_left, color='b')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('left wheel speed [m/s]', fontsize=fontsize)
#     plt.xlabel('time [s]', fontsize=fontsize)

# if state_cg_history:
#     plt.figure()
#     plt.subplot(3, 1, 1)
#     plt.plot(time, state_cg[:,0], color='b')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('v_x_cg [m/s]')
#     plt.title('cg point velocity', fontsize=fontsize)
#     plt.subplot(3, 1, 2)
#     plt.plot(time, state_cg[:,1])
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('v_y_cg [m/s], fontsize=fontsize')
#     plt.subplot(3, 1, 3)
#     plt.plot(time, state_cg[:,2] * 180 / np.pi)
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('w [deg/s]', fontsize=fontsize)
#     plt.xlabel('time [s]', fontsize=fontsize)

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(time, detection, color='r')
# plt.ylabel('Detection', fontsize=fontsize)
# plt.yticks(fontsize=ticksize)
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.title('Obstacle detection & Angular rate gain', fontsize=fontsize)
# plt.subplot(2, 1, 2)
# plt.plot(time, data['K'])
# plt.ylabel('Angular velocity gain', fontsize=fontsize)
# plt.yticks(fontsize=ticksize)
# plt.xlabel('time [s]', fontsize=fontsize)
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.savefig(os.path.join('log', loglist[-1], 'K.png'), bbox_inches='tight')



# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(time, ispass, color='r', linestyle='--', label='ispass')
# plt.plot(time, detection, color='b', linestyle='--', label='detection')
# plt.ylabel('ispass&detection')
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.legend(loc='upper left')
# plt.subplot(3, 1, 2)
# plt.plot(time, speed_design_mode)
# plt.ylabel('speed design mode')
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.subplot(3, 1, 3)
# plt.plot(time, data['K'])
# plt.ylabel('K')
# plt.xlabel('time [s]')
# plt.grid(axis='both', linestyle='--', color='gray')

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(time, wheel_vel_right[:,0])
# plt.title('right wheel velocity')
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.subplot(3,1,2)
# plt.plot(time, wheel_vel_right[:,1])
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.subplot(3,1,3)
# plt.plot(time, wheel_vel_right[:,2])
# plt.grid(axis='both', linestyle='--', color='gray')

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(time, wheel_vel_left[:,0])
# plt.title('left wheel velocity')
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.subplot(3,1,2)
# plt.plot(time, wheel_vel_left[:,1])
# plt.grid(axis='both', linestyle='--', color='gray')
# plt.subplot(3,1,3)
# plt.plot(time, wheel_vel_left[:,2])
# plt.grid(axis='both', linestyle='--', color='gray')

# f_r = data['f_r']
# f_l = data['f_l']
# f_mck_x = data['f_mck_x']
# f_mck_y = data['f_mck_y']

# plt.figure()
# plt.subplot(4,1,1)
# plt.plot(time, f_r)
# plt.ylabel('f_r')
# plt.subplot(4,1,2)
# plt.plot(time, f_l)
# plt.ylabel('f_l')
# plt.subplot(4,1,3)
# plt.plot(time, f_mck_x)
# plt.ylabel('f_mck_x')
# plt.subplot(4,1,4)
# plt.plot(time, f_mck_y)
# plt.ylabel('f_mck_y')

# f_cr_x = data['f_cr_x']
# f_cl_x = data['f_cl_x']
# f_cr_y = data['f_cr_y']
# f_cl_y = data['f_cl_y']

# plt.figure()
# plt.subplot(4,1,1)
# plt.plot(time, f_cr_x)
# plt.ylabel('f_cr_x')
# plt.subplot(4,1,2)
# plt.plot(time, f_cl_x)
# plt.ylabel('f_cl_x')
# plt.subplot(4,1,3)
# plt.plot(time, f_cr_y)
# plt.ylabel('f_cr_y')
# plt.subplot(4,1,4)
# plt.plot(time, f_cl_y)
# plt.ylabel('f_cl_y')

# f_cr_R = data['f_cr_R']
# f_cl_R = data['f_cl_R']
# f_cr_T = data['f_cr_T']
# f_cl_T = data['f_cl_T']

# plt.figure()
# plt.subplot(4,1,1)
# plt.plot(time, f_cr_R)
# plt.ylabel('f_cr_R')
# plt.subplot(4,1,2)
# plt.plot(time, f_cl_R)
# plt.ylabel('f_cl_R')
# plt.subplot(4,1,3)
# plt.plot(time, f_cr_T)
# plt.ylabel('f_cr_T')
# plt.subplot(4,1,4)
# plt.plot(time, f_cl_T)
# plt.ylabel('f_cl_T')

# alpha_right = data['state']['caster_right']
# alpha_left = data['state']['caster_left']
# plt.figure()
# plt.subplot(4, 1, 1)
# plt.plot(time, alpha_right[:,0] * 180 / np.pi)
# plt.ylabel('alpha_right [deg]')
# plt.title('caster angle')
# plt.subplot(4,1,2)
# plt.plot(time, alpha_left[:,0] * 180 / np.pi)
# plt.ylabel('alpha_left [deg]')
# plt.subplot(4, 1, 3)
# plt.plot(time, alpha_right[:,1] * 180 / np.pi)
# plt.ylabel('alpha_dot_right [deg/s]')
# plt.subplot(4,1,4)
# plt.plot(time, alpha_left[:,1] * 180 / np.pi)
# plt.ylabel('alpha_dot_left [deg/s]')

# e_right = data['error_right']
# e_left = data['error_left']
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(time, e_right)
# plt.title('controller error')
# plt.subplot(2,1,2)
# plt.plot(time, e_left)

if moving_obstacle_onoff:
    plt.figure()
    # ax = plt.subplot(1, 1, 1)
    for j in np.arange(n_obstacle):
        p = data['state'][f'pedestrian_{j}'][:,0]
        q = data['state'][f'pedestrian_{j}'][:,1]
        distance = np.sqrt((wheelchair[:,0] - p)**2 + (wheelchair[:,1] - q)**2)
        plt.plot(time, distance, label=f'pedestrian{j+1}', color=color_obs[j])
        # axins = inset_axes(ax, width=0.25, height=0.25, loc=2)
        # axins.plot(time, distance)
        # axins.set_xlim(0, 10)
        # axins.set_ylim(0, 3)
        # mark_inset(ax, axins, loc1=2, loc2=4)

    plt.legend(loc='best')
    plt.grid(axis='both', linestyle='--', color='gray')
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.ylabel('distance [m]', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.ylim(bottom=0)
    plt.title('Distance between the wheelchair and pedestrians', fontsize=fontsize)
    plt.savefig(os.path.join('log', loglist[past], 'distance.png'), bbox_inches='tight')

# if True:
#     plt.figure(figsize=(7.5, 9.5))
#     plt.subplot(6, 1, 1)
#     plt.plot(time, hrvo_speed, color='r', label='hrvo')
#     plt.plot(time, design_speed, color='g', label='design', linestyle='--')
#     plt.plot(time, wheelchair_speed, color='b', label='true', linestyle='--')
#     # plt.plot(time, v_f, color='k', label='final speed', linestyle='--')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     # plt.legend(loc='upper right')
#     #plt.legend(loc='upper left')
#     plt.legend(loc='best')
#     plt.ylabel('$v \; [m/s]$', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)

#     plt.subplot(6, 1, 2)
#     plt.plot(time, np.unwrap(hrvo_theta) * 180 / np.pi, color='r')
#     plt.plot(time, np.unwrap(design_angle) * 180 / np.pi, color='g', linestyle='--')
#     plt.plot(time, wheelchair[:,2] * 180 / np.pi, color='b', linestyle='--')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('$ \\theta \; [deg]$', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)

#     plt.subplot(6, 1, 3)
#     plt.plot(time, body_acc[:,0], color='b', label='true')
#     plt.plot(time, [a_x_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--', label='boundary')
#     plt.plot(time, [a_x_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('$a_x \; [m/s^2]$', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)
#     plt.legend(loc='best')
#     # plt.title('Acceleration and angular rate', fontsize=fontsize)
#     plt.subplot(6, 1, 4)
#     plt.plot(time, body_acc[:,1], color='b')
#     plt.plot(time, [a_y_limit[0]]*len(time), color='r', lw=linewidth, linestyle='--')
#     plt.plot(time, [a_y_limit[1]]*len(time), color='r', lw=linewidth, linestyle='--')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('$a_y \; [m/s^2]$', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)

#     plt.subplot(6, 1, 5)
#     plt.plot(time, omega * 180 / np.pi, color='r')
#     plt.grid(axis='both', linestyle='--', color='gray')
#     plt.ylabel('$\\omega \; [deg/s]$', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)

#     plt.subplot(6, 1, 6)
#     plt.plot(time, data['K'], color='r')
#     plt.ylabel('K', fontsize=fontsize)
#     plt.yticks(fontsize=ticksize)
#     plt.grid(axis='both', linestyle='--', color='gray')

#     plt.xlabel('time [s]', fontsize=fontsize)

#     plt.savefig(os.path.join('log', loglist[past], 'total.png'), bbox_inches='tight')



plt.show()

