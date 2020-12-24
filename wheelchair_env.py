import numpy as np
import numpy.linalg as lin
import gym
from gym import spaces
from fym.core import BaseEnv, infinite_box
from fym.agents.PID import PID

from wheelchair_module.two_wheels_robot import TwoWheelsRobot3Dof
from wheelchair_module.actuator import Actuator
from wheelchair_module.wheelchair_dynamic import WheelchairDynamic
from wheelchair_module.pedestrian import Pedestrian
from wheelchair_module.hrvo_dynamic import HRVO as HRVO_dynamic
from wheelchair_module.hrvo import HRVO
from wheelchair_module.wrap_angle import wrap_angle
from wheelchair_module.driver import Driver
from wheelchair_module.caster import Caster


class WheelchairEnv(BaseEnv):
    def __init__(self, setting):
        ########################## system setting ##############################
        systems = {
            'wheelchair': TwoWheelsRobot3Dof(
                initial_state=setting['wheelchair']['initial_state'], setting=setting
            ),
            'actuator_right': Actuator(
                initial_state=setting['actuator']['initial_state'],
                tau=setting['actuator']['tau']
            ),
            'actuator_left': Actuator(
                initial_state=setting['actuator']['initial_state'],
                tau=setting['actuator']['tau']
            ),
            'caster_right': Caster(
                initial_state=setting['caster']['initial_state'][0],
                setting=setting
            ),
            'caster_left': Caster(
                initial_state=setting['caster']['initial_state'][1],
                setting=setting
            ),
            'dynamic': WheelchairDynamic(
                initial_state=[0, 0, 0],
                setting=setting
            ),
            'driver': Driver(
                initial_state=[0, 0, 0, 0],
                setting=setting
            )
        }
        
        self.setting = setting
        ###################### Action space, HRVO ##############################
        self.action_space = infinite_box((1, ))
        self.hrvo = HRVO(setting)
        self.hrvo_dynamic = HRVO_dynamic(setting)

        ######################### setting paramters ############################
        self.saturation = setting['actuator']['saturation']

        self.a_x_lb = setting['wheelchair']['a_x_limit'][0]
        self.a_x_ub = setting['wheelchair']['a_x_limit'][1]
        self.a_y_ub = setting['wheelchair']['a_y_limit'][1]
        self.W = setting['wheelchair']['width']
        self.L = setting['wheelchair']['length']
        self.r_cg = setting['wheelchair']['cg']
        self.wheelchair_mass = setting['wheelchair']['mass']
        self.mu = setting['wheelchair']['mu']
        self.J = setting['wheelchair']['J']
        
        self.v_max = setting['env']['max_v']
        self.w_max = setting['env']['max_w']
        self.t_rob = setting['env']['robust_time']
        self.tau_c = setting['env']['control_tau']
        self.K_max = setting['env']['K_max']
        self.K_min = setting['env']['K_min']
        self.speed_design_method = setting['env']['speed_design_method']
        self.K_adapt_method = setting['env']['K_adapt_method']
        self.driver_mass = setting['env']['driver_mass']
        self.friction_rate = setting['env']['friction']['rate']
        self.friction_saturation = setting['env']['friction']['saturation']
        self.hrvo_method = setting['env']['hrvo_method']
        
        self.mu_c_T = setting['caster']['mu_T']
        self.mu_c_R = setting['caster']['mu_R']
        
        self.m = setting['wheelchair']['mass'] + setting['env']['driver_mass']
        
        ######################### Other parameters #############################
        self.distance_condition = False
        self.angle_condition = False
        self.angle_condition_initial = False
        self.convergence_condition = False
        self.convergence_count = 0
        self.initial_count = 0
        self.destination = False
        self.speed_condition = False
        self.n = setting['env']['moving_onoff'] * len(setting['moving_obstacle'].keys())
        self.g = 9.81

        self.GR = np.array([-self.r_cg, -self.W / 2, 0])
        self.GL = np.array([-self.r_cg, self.W / 2, 0])
        self.GCr = np.array([self.L - self.r_cg, -self.W / 2, 0])
        self.GCl = np.array([self.L - self.r_cg, self.W / 2, 0])

        N_right = self.m * self.g * (1 - self.r_cg / self.L) / 2
        N_left = self.m * self.g * (1 - self.r_cg / self.L) / 2
        N_c_right = self.m * self.g * self.r_cg / self.L / 2
        N_c_left = self.m * self.g * self.r_cg / self.L / 2
        self.normal_force = [N_right, N_left, N_c_right, N_c_left]
        self.pedestrian_speed = setting['env']['pedestrian_speed']

        ######################## Initialize for global #########################
        self.v_f = 0
        self.K = self.K_max
        self.v = 0
        self.pos = np.array([0, 0])
        self.speed_design_mode = 0
        self.Ccb_right = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.Ccb_left = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.dynamic_deriv = [0, 0, 0]
        self.control_input = {}
        self.time = 0

        ########################## PID Controller ##############################
        self.PID_gain = setting['env']['PID']
        self.pid_right = PID(self.PID_gain, windup=0.5)
        self.pid_left = PID(self.PID_gain, windup=0.5)
        self.e_theta_integral = PID(np.array([0, 1, 0]), windup=1)

        ######################## super().__init__ ##############################
        super().__init__(
            systems=systems,
            dt=setting['env']['time_step'],
            max_t=setting['env']['final_time']
        )

    def reset(self, noise=0):
        super().reset()
        return self.states

    def derivs(self, t, states, action):
        xdot = self.get_xdot(t, states, action)
        return self.unpack_state(xdot)

    def step(self, action):
        states = self.states
        time = self.clock.get()
        self.time = time
        reward = 0

        ######################### Velocity design ##############################
        speed, direction = self.velocity_design(states, action['wheelchair'])
        
        ############################# Adapt K ##################################
        if self.K_adapt_method == 'suggested':
            _ = self.adapt_K(states, states['wheelchair'][2], direction)

        ############################# HRVO #####################################
        v_hrvo, theta_hrvo, ispass, detection = self.hrvo.hrvo('wheelchair', states, self.control_input, speed, direction, self.K, self.hrvo_method, action['wheelchair'], self.distance_condition)

        ######################## Desired wheel speed ###########################
        v_ref_right, v_ref_left = self.desired_wheel_speed(v_hrvo, theta_hrvo, states['wheelchair'][2])

        ############################# Controller ###############################
        v_Ri_b, v_Li_b, _, _ = self.change_vel(states)
        e_right = -(v_Ri_b[0] - v_ref_right)
        e_left = -(v_Li_b[0] - v_ref_left)
        F_R_ref = self.pid_right.get(e_right)
        F_L_ref = self.pid_left.get(e_left)

        self.control_input['actuator_right'] = F_R_ref
        self.control_input['actuator_left'] = F_L_ref
        ############################## Friction ################################
        f_r, f_l, f_cr_x, f_cr_y, f_cl_x, f_cl_y, f_cr_R, f_cr_T, f_cl_R, f_cl_T = self.friction(states)

        ##################### mass spring damper system ########################
        f_mck_x = -self.driver_mass * self.dynamic_deriv[0]
        f_mck_y = -self.driver_mass * self.dynamic_deriv[1]
        self.control_input['driver'] = [-f_mck_x, -f_mck_y]

        ######################### Reaction force ###############################
        F_R = states['actuator_right']
        F_L = states['actuator_left']
        vx, vy, omega = states['dynamic']
        x_bar, y_bar, _, _ = states['driver']
        M = self.W/2 * (F_R - F_L + f_r - f_l + f_cr_x - f_cl_x) + (self.L - self.r_cg) * (f_cr_y + f_cl_y) + x_bar * f_mck_y - y_bar * f_mck_x

        reaction_force = self.m * self.J / (self.J + self.m * self.r_cg**2) * (vx * omega - (f_cr_y + f_cl_y + f_mck_y) / self.m + self.r_cg / self.J * M)

        ###################### Sum of force and moment #########################
        F_x = F_R + F_L + f_r + f_l + f_cr_x + f_cl_x + f_mck_x
        F_y = f_cr_y + f_cl_y + f_mck_y + reaction_force
        M_total = M - self.r_cg * reaction_force

        self.control_input['dynamic'] = np.hstack([F_x, F_y, M_total])

        ############################# Caster ###################################
        self.control_input['caster_right'] = f_cr_T
        self.control_input['caster_left'] = f_cl_T

        ##################### Two wheel robot dynamic ##########################
        v_Ri_b, v_Li_b, _, _ = self.change_vel(states)

        self.control_input['wheelchair'] = [v_Ri_b[0], v_Li_b[0]]

        ######################### Moving obstacle ##############################
        for i in np.arange(self.n):
            if action[f'pedestrian_{i}']:
                pos_ped = action[f'pedestrian_{i}'][0] - states[f'pedestrian_{i}'][0:2]
                direction_ped = np.arctan2(pos_ped[1], pos_ped[0])
                hrvo_v_ped, hrvo_theta_ped, _, _ = self.hrvo_dynamic.hrvo(f'pedestrian_{i}', states, self.control_input, self.pedestrian_speed[i], direction_ped, self.K, 'no constraint', action[f'pedestrian_{i}'])
                self.control_input[f'pedestrian_{i}'] = [hrvo_v_ped, hrvo_theta_ped]
            else:
                self.control_input[f'pedestrian_{i}'] = self.setting['moving_obstacle'][f'{i}']['initial_state'][2:4]

        ### Actuator, dynamic, Two wheel robot kinematic, Caster dynamic, Moving obstacle ###
        next_states, _ = self.get_next_states(time, states, self.control_input)

        ######################## Actuator saturation ###########################
        next_states = self.actuator_saturation(next_states)

        ############################ Data log ##################################
        body_acc = [self.dynamic_deriv[0] - vy * omega, (v_Ri_b[0] + v_Li_b[0]) / 2 * states['dynamic'][2]]

        info = {
            'time': time,
            'state': states,
            'designed_speed': speed,
            'designed_angle': direction,
            'v_hrvo': v_hrvo,
            'theta_hrvo': theta_hrvo,
            'ispass': ispass,
            'actuator_reference': [F_R_ref, F_L_ref],
            'actuator': [F_R, F_L],
            'wheel_reference_right': v_ref_right,
            'wheel_right': v_Ri_b[0],
            'wheel_reference_left': v_ref_left,
            'wheel_left': v_Li_b[0],
            'body_acceleration': body_acc,
            'final_speed': self.v_f,
            'speed_design_mode': self.speed_design_mode,
            'wheel_vel_right': v_Ri_b,
            'wheel_vel_left': v_Li_b,
            'f_r': f_r,
            'f_l': f_l,
            'f_mck_x': f_mck_x,
            'f_mck_y': f_mck_y,
            'f_cr_x': f_cr_x,
            'f_cr_y': f_cr_y,
            'f_cl_x': f_cl_x,
            'f_cl_y': f_cl_y,
            'f_cr_R': f_cr_R,
            'f_cr_T': f_cr_T,
            'f_cl_R': f_cl_R,
            'f_cl_T': f_cl_T,
            'error_right': e_right,
            'error_left': e_left,
            'step_input': self.control_input,
            'K': self.K,
            'obstacle_detection': detection
        }

        ############################### Update #################################
        self.states = next_states
        self.clock.tick()
        done = self.is_terminal()

        return next_states, reward, done, info

    def is_terminal(self):
        if self.clock.get() > self.clock.max_t:
            return True
        else:
            return False

    def get_xdot(self, t, states, action):
        xdot = {
            'wheelchair': self.systems['wheelchair'].deriv(t, states['wheelchair'], action['wheelchair']),
            'actuator_right': self.systems['actuator_right'].deriv(t, states['actuator_right'], action['actuator_right']),
            'actuator_left': self.systems['actuator_left'].deriv(t, states['actuator_left'], action['actuator_left']),
            'caster_right': self.systems['caster_right'].deriv(t, states['caster_right'], action['caster_right']),
            'caster_left': self.systems['caster_left'].deriv(t, states['caster_left'], action['caster_left']),
            'dynamic': self.systems['dynamic'].deriv(t, states['dynamic'], action['dynamic']),
            'driver': self.systems['driver'].deriv(t, states['driver'], action['driver'])
        }

        self.dynamic_deriv = xdot['dynamic']

        return xdot

    ########################## Velocity design #################################
    def velocity_design(self, states, action):
        x, y, theta = states['wheelchair']
        waypoint = action[0]
        attitude = action[1]
        self.pos = np.array(waypoint) - np.array([x, y])

        v_Ri_b, v_Li_b, _, _ = self.change_vel(states)
        self.v = (v_Ri_b[0] + v_Li_b[0]) / 2
        
        # if self.distance_condition:
        #     direction = attitude
        #     if self.speed_design_method == 'maximum_speed':
        #         speed = self.v_max
        #     elif self.speed_design_method == 'suggested':
        #         speed = self.v_f
        #     else:
        #         speed = 0
        #         print('need to set speed design method')
        # else:
        #     direction = np.arctan2(self.pos[1], self.pos[0])
        #     if self.speed_design_method == 'maximum_speed':
        #         speed = self.v_max
        #     elif self.speed_design_method == 'suggested':
        #         speed = self.speed_design(states, action)
        #     else:
        #         speed = 0
        #         print('need to set speed design method')
        
        if self.distance_condition:
            direction = attitude
        else:
            direction = np.arctan2(self.pos[1], self.pos[0])

        if self.speed_design_method == 'maximum_speed':
            speed = self.v_max
            if all([self.distance_condition, self.angle_condition, self.destination]):
                speed = 0
                if self.v < 1e-2:
                    self.speed_condition = True

        elif self.speed_design_method == 'suggested':
            speed = self.speed_design(states, action)
            if all([self.distance_condition, self.angle_condition, self.destination]):
                speed = max(self.v + self.tau_c * self.a_x_lb, 0)
                if self.v < 1e-2:
                    self.speed_condition = True
        else:
            speed = 0
            print('need to set speed design method')

        if abs(wrap_angle(direction - theta)) < 1 * np.pi / 180 and self.angle_condition_initial == False:
            self.initial_count += 1
            if self.initial_count > 30:
                self.angle_condition_initial = True

        return speed, direction

    def speed_design(self, states, action):
        #if self.speed_design_method != 2 and self.speed_design_method != 6:
        _ = self.final_speed(states, action)

        if self.angle_condition_initial:
            # if self.distance_condition:
            #     designed_speed = self.v_f
            # else:
            distance = lin.norm(self.pos)
            if abs(self.v - self.v_max) < 1e-2:
                if self.v_f == self.v_max:
                    designed_speed = self.v_max
                    self.speed_design_mode = 0
                else:
                    criteria = (self.v_f**2 - self.v_max**2) / (2 * self.a_x_lb)
                    # + self.tau_c * (self.v_max + self.tau_c * self.a_x_lb - self.v_f)
                    
                    if distance > criteria:
                        designed_speed = self.v_max
                        self.speed_design_mode = 1
                    else:
                        designed_speed = max(self.v + self.tau_c * self.a_x_lb, self.v_f)
                        self.speed_design_mode = 2
            else:
                if self.v_f == self.v_max:
                    designed_speed = min(self.v_max, self.v + self.tau_c * self.a_x_ub)
                    self.speed_design_mode = 3
                else:
                    criteria_1 = ((self.v_max**2 - self.v**2) / self.a_x_ub + (self.v_f**2 - self.v_max**2) / self.a_x_lb) / 2
                    criteria_2 = (self.v_f**2 - self.v_max**2) / (2 * self.a_x_lb)
                    if criteria_1 < distance:
                        designed_speed = min(self.v + self.tau_c * self.a_x_ub, self.v_max)
                        self.speed_design_mode = 4
                    elif criteria_2 < distance <= criteria_1:
                        designed_speed = min(self.v + self.tau_c * self.a_x_ub, self.v_max)
                        self.speed_design_mode = 5
                    else:
                        designed_speed = max(self.v + self.tau_c * self.a_x_lb, self.v_f)
                        self.speed_design_mode = 6

            # if self.distance_condition and not self.angle_condition:
            #     designed_speed = self.v_f
        else:
            designed_speed = 0
            self.speed_design_mode = -1
    
        return designed_speed

    def final_speed(self, states, action):
        if not self.destination:
            if not self.distance_condition:
                theta_d = action[1]
                waypoint = action[0]
                pos = states['wheelchair'][0:2]
                theta = np.arctan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
                diff = wrap_angle(theta_d - theta)
                if diff == 0:
                    self.v_f = self.v_max
                else:
                    w_f = self.K_max * abs(diff) * (1 - np.exp(-self.t_rob / self.tau_c))
                    self.v_f = min(self.a_y_ub / w_f, self.v_max)            
        else:
            self.v_f = 0
        
        #self.v_f = self.a_y_ub / self.w_max
        return self.v_f

    ############################### Adapt K ####################################
    def adapt_K(self, states, theta, theta_d):
        v_Ri_b, v_Li_b, _, _ = self.change_vel(states)
        w_0 = (v_Ri_b[0] - v_Li_b[0]) / self.W
        diff = wrap_angle(theta_d - theta)
        v = (v_Ri_b[0] + v_Li_b[0]) / 2

        # theta_dot = w_0 * np.exp(-self.t_rob / self.tau_c) + self.K * diff * (1 - np.exp(-self.t_rob / self.tau_c))

        # if v < 1e-4:
        #     self.K = self.K_max
        # else:
        #     if diff > 0:
        #         self.K = (self.a_y_ub / v - w_0 * np.exp(-self.t_rob / self.tau_c)) / (diff * (1 - np.exp(-self.t_rob / self.tau_c)))
        #     else:
        #         self.K = (-self.a_y_ub / v - w_0 * np.exp(-self.t_rob / self.tau_c)) / (diff * (1 - np.exp(-self.t_rob / self.tau_c)))
        

        if v < 1e-3:
            # self.K = self.w_max / (abs(diff) * (1 - np.exp(-self.t_rob / self.tau_c)))
            self.K = self.K_max
        else:                
            self.K = self.a_y_ub / (abs(diff) * v * (1 - np.exp(-self.t_rob / self.tau_c)))
        self.K = np.clip(self.K, self.K_min, self.K_max)

        return self.K

    ########################### Desired wheel speed ############################
    def desired_wheel_speed(self, v_d, theta_d, theta):
        theta_diff = wrap_angle(theta_d - theta)
        v_right = v_d + self.K * self.W * theta_diff / 2
        v_left = v_d - self.K * self.W * theta_diff / 2
        return v_right, v_left

    ############################ velocity change ###############################
    def change_vel(self, states):
        v_Gi_b = np.array([states['dynamic'][0], states['dynamic'][1], 0])
        omega = np.array([0, 0, states['dynamic'][2]])
        alpha_right = states['caster_right'][0]
        alpha_left = states['caster_left'][0]
        self.Ccb_right = np.array([[np.cos(alpha_right), np.sin(alpha_right), 0], [-np.sin(alpha_right), np.cos(alpha_right), 0], [0, 0, 1]])
        self.Ccb_left = np.array([[np.cos(alpha_left), np.sin(alpha_left), 0], [-np.sin(alpha_left), np.cos(alpha_left), 0], [0, 0, 1]])
        
        v_Ri_b = v_Gi_b + np.cross(omega, self.GR)
        v_Li_b = v_Gi_b + np.cross(omega, self.GL)
        v_Cri_b = v_Gi_b + np.cross(omega, self.GCr)
        v_Cli_b = v_Gi_b + np.cross(omega, self.GCl)
        v_Cri_c = self.Ccb_right.dot(v_Cri_b)
        v_Cli_c = self.Ccb_left.dot(v_Cli_b)
        
        return v_Ri_b, v_Li_b, v_Cri_c, v_Cli_c

    ############################### Friction ###################################
    def friction(self, states):
        v_Ri_b, v_Li_b, v_Cri_c, v_Cli_c = self.change_vel(states)

        f_r = np.clip(-self.friction_rate * self.mu * self.normal_force[0] * v_Ri_b[0], -self.friction_saturation, self.friction_saturation)
        f_l = np.clip(-self.friction_rate * self.mu * self.normal_force[0] * v_Li_b[0], -self.friction_saturation, self.friction_saturation)
        f_cr_R = np.clip(-self.friction_rate * self.mu_c_R * self.normal_force[0] * v_Cri_c[0], -self.friction_saturation, self.friction_saturation)
        f_cr_T = np.clip(-self.friction_rate * self.mu_c_R * self.normal_force[0] * v_Cri_c[1], -self.friction_saturation, self.friction_saturation)
        f_cl_R = np.clip(-self.friction_rate * self.mu_c_R * self.normal_force[0] * v_Cli_c[0], -self.friction_saturation, self.friction_saturation)
        f_cl_T = np.clip(-self.friction_rate * self.mu_c_R * self.normal_force[0] * v_Cli_c[1], -self.friction_saturation, self.friction_saturation)

        f_cr = np.transpose(self.Ccb_right).dot(np.array([f_cr_R, f_cr_T, 0]))
        f_cr_x = f_cr[0]
        f_cr_y = f_cr[1]
        f_cl = np.transpose(self.Ccb_left).dot(np.array([f_cl_R, f_cl_T, 0]))
        f_cl_x = f_cl[0]
        f_cl_y = f_cl[1]

        return f_r, f_l, f_cr_x, f_cr_y, f_cl_x, f_cl_y, f_cr_R, f_cr_T, f_cl_R, f_cl_T

    ########################## Actuator saturation #############################
    def actuator_saturation(self, states):
        if states['actuator_right'] > self.saturation:
            states['actuator_right'] = np.array([self.saturation])
        elif states['actuator_right'] < -self.saturation:
            states['actuator_right'] = -np.array([self.saturation])
        
        if states['actuator_left'] > self.saturation:
            states['actuator_left'] = np.array([self.saturation])
        elif states['actuator_left'] < -self.saturation:
            states['actuator_left'] = -np.array([self.saturation])
        return states
    ############################################################################


class WheelchairWithMovingObstacleEnv(WheelchairEnv):
    def __init__(self, setting):
        super().__init__(setting)

        self.n = len(setting['moving_obstacle'])
        new_systems = {
            f'pedestrian_{i}': Pedestrian(
                initial_state=setting['moving_obstacle'][f'{i}']['initial_state'], tau=setting['env']['pedestrian_tau'][i]) for i in np.arange(self.n)
        }

        for i in np.arange(self.n):
            self.control_input[f'pedestrian_{i}'] = setting['moving_obstacle'][f'{i}']['initial_state'][2:4]
        self.append_systems(new_systems)
        self.action_space = infinite_box((1,))

    def reset(self, noise=0):
        states = super().reset()
        return states

    def derivs(self, t, states, action):
        temp = {'wheelchair', 'actuator_right', 'actuator_left', 'caster_right', 'caster_left', 'dynamic', 'driver'}
        temp2 = {'wheelchair', 'actuator_right', 'actuator_left', 'caster_right', 'caster_left', 'dynamic', 'driver'}
        super_states = {key: states[key] for key in states.keys() & temp}
        super_action = {key: action[key] for key in action.keys() & temp2}
        xdot = self.get_xdot(t, super_states, super_action)
        for i in np.arange(self.n):
            xdot[f'pedestrian_{i}'] = self.systems[f'pedestrian_{i}'].deriv(t, states[f'pedestrian_{i}'], action[f'pedestrian_{i}'])

        return self.unpack_state(xdot)

    def step(self, action):
        next_states, reward, done, info = super().step(action)

        return next_states, reward, done, info

