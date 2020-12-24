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

        ###################### Action space, HRVO ##############################
        self.action_space = infinite_box((1, ))

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
        self.t_rob = setting['env']['robust_time']
        self.tau_c = setting['env']['control_tau']
        self.K = setting['env']['control_gain']
        self.driver_mass = setting['env']['driver_mass']
        self.friction_rate = setting['env']['friction']['rate']
        self.friction_saturation = setting['env']['friction']['saturation']
        
        self.mu_c_T = setting['caster']['mu_T']
        self.mu_c_R = setting['caster']['mu_R']
        
        self.m = setting['wheelchair']['mass'] + setting['env']['driver_mass']
        
        ######################### Other parameters #############################
        self.distance_condition = False
        self.angle_condition = False
        self.control_input = {}
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

        ######################## Initialize for global #########################
        self.v_f = 0
        self.v = 0
        self.pos = np.array([0, 0])
        self.speed_design_mode = 0
        self.Ccb_right = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.Ccb_left = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.dynamic_deriv = [0, 0, 0]
        self.control_input = {}

        ########################## PID Controller ##############################
        self.PID_gain = setting['env']['PID']
        self.pid_right = PID(self.PID_gain, windup=5)
        self.pid_left = PID(self.PID_gain, windup=5)

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
        reward = 0

        ######################## Desired wheel speed ###########################
        v_ref_right = 1
        v_ref_left = 1

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
        F_x = F_R + F_L + f_r + f_l + f_cr_x + f_cl_x + f_mck_x + vy * omega
        F_y = f_cr_y + f_cl_y + f_mck_y + reaction_force - vx * omega
        M_total = M - self.r_cg * reaction_force

        self.control_input['dynamic'] = np.hstack([F_x, F_y, M_total])

        ############################# Caster ###################################
        self.control_input['caster_right'] = f_cr_T
        self.control_input['caster_left'] = f_cl_T

        ##################### Two wheel robot dynamic ##########################
        v_Ri_b, v_Li_b, _, _ = self.change_vel(states)

        self.control_input['wheelchair'] = [v_Ri_b[0], v_Li_b[0]]

        ####### Actuator, dynamic, Two wheel robot kinematic, Caster dynamic, Moving obstacle ######
        next_states, _ = self.get_next_states(time, states, self.control_input)

        ######################## Actuator saturation ###########################
        next_states = self.actuator_saturation(next_states)

        ############################ Data log ##################################
        body_acc = [self.dynamic_deriv[0], self.v * states['dynamic'][2]]

        info = {
            'time': time,
            'state': states,
            'actuator_reference': [F_R_ref, F_L_ref],
            'actuator': [F_R, F_L],
            'wheel_reference_right': v_ref_right,
            'wheel_right': v_Ri_b[0],
            'wheel_reference_left': v_ref_left,
            'wheel_left': v_Li_b[0],
            'body_acceleration': body_acc,
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
            'step_input': self.control_input
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

    ########################### Desired wheel speed ############################
    def desired_wheel_speed(self, v_d, theta_d, theta):
        theta_diff = wrap_angle(theta_d - theta)
        v_right = v_d + self.K * self.W * theta_diff / 2
        v_left = v_d - self.K * self.W * theta_diff / 2
        return v_right, v_left

    ############################################################################

