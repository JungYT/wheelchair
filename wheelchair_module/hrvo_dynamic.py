import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
from wheelchair_module.wrap_angle import wrap_angle
from wheelchair_module.obstacle_detection import ObstacleDetection


class HRVO(ObstacleDetection):
    def __init__(self, setting):
        self.r_plant = setting['hrvo']['r_plant']
        self.r_static_obstacle = setting['hrvo']['r_static']
        self.r_moving = setting['hrvo']['r_moving']
        self.method = setting['hrvo']['method']
        self.v_max = setting['env']['max_v']
        self.d_angle = setting['hrvo']['angle_step']
        self.d_v = setting['hrvo']['v_step']
        self.setting = setting
        self.a_x_ub = self.setting['wheelchair']['a_x_limit'][1]
        self.a_x_lb = self.setting['wheelchair']['a_x_limit'][0]
        self.control_tau = self.setting['env']['control_tau']
        self.a_y_ub = self.setting['wheelchair']['a_y_limit'][1]
        self.L = self.setting['wheelchair']['width']
        self.Len = setting['wheelchair']['length']
        self.robust_time = self.setting['env']['robust_time']
        self.weight = self.setting['hrvo']['weight']
        self.r_cg = setting['wheelchair']['cg']
        self.GR = np.array([-self.r_cg, -self.L / 2, 0])
        self.GL = np.array([-self.r_cg, self.L / 2, 0])
        self.GCr = np.array([self.Len - self.r_cg, -self.L / 2, 0])
        self.GCl = np.array([self.Len - self.r_cg, self.L / 2, 0])

        super().__init__(setting)


    def hrvo(self, key, states, control_input, desired_speed, desired_angle, K, hrvo_method, action):
        self.K = K
        # set state according to key
        moving_obstacle_key = list(states.keys() - {'actuator_right', 'actuator_left', 'caster_right', 'caster_left', 'dynamic', 'driver',key})
        pos = states[key][0:2]
        theta_dot = states['dynamic'][2]
        if key == 'wheelchair':
            theta = states[key][2]
            v_Ri_b, v_Li_b = self.change_vel(states)
            v = (v_Ri_b[0] + v_Li_b[0]) / 2
        else:
            v = states[key][2]
            theta = states[key][3]
        vel = np.array([v * np.cos(theta), v * np.sin(theta)])

        # set static obstacles
        static_obstacles = self.static_obstacle_detection(np.hstack([pos, theta]), action, False)
        
        # make hrvo set
        hrvo_set = []

        # make static obstacle hrvo set
        for i in np.arange(len(static_obstacles)):
            pos_static = static_obstacles[i]
            pos_ba = pos_static - pos
            distance = lin.norm(pos_ba)
            if distance < self.r_plant + self.r_static_obstacle:                    distance = self.r_plant + self.r_static_obstacle

            alpha_1 = np.arctan2(pos_ba[1], pos_ba[0])
            alpha_2 = np.arcsin((self.r_plant + self.r_static_obstacle) / distance)
            right_bound_angle = alpha_1 - alpha_2
            left_bound_angle = alpha_1 + alpha_2
            point = pos

            hrvo_set.append([point, right_bound_angle, left_bound_angle])
        # make moving obstacle hrvo set
        for i in np.arange(len(moving_obstacle_key)):
            pos_moving = states[moving_obstacle_key[i]][0:2]
            if moving_obstacle_key[i] == 'wheelchair':
                theta_moving = states[moving_obstacle_key[i]][2]
                v_Ri_b, v_Li_b = self.change_vel(states)
                v_moving = (v_Ri_b[0] + v_Li_b[0]) / 2
            else:
                v_moving = control_input[moving_obstacle_key[i]][0]
                theta_moving = control_input[moving_obstacle_key[i]][1]
            pos_ba = pos_moving - pos
            distance = lin.norm(pos_ba)
            if distance < self.r_plant + self.r_moving:                             distance = self.r_plant + self.r_moving
            alpha_1 = np.arctan2(pos_ba[1], pos_ba[0])
            alpha_2 = np.arcsin((self.r_plant + self.r_moving) / distance)
            moving_obstacle_detection = False
            diff = wrap_angle(alpha_1 - theta)
            if distance < self.detection_range and abs(diff) < self.detection_angle / 2:
                moving_obstacle_detection = True

            if moving_obstacle_detection:
                if v == 0:
                    whether_close_or_not = wrap_angle(alpha_1 - theta_moving)
                    if whether_close_or_not >= np.pi / 2:
                        right_bound_angle = alpha_1 - alpha_2
                        left_bound_angle = theta_moving
                        point = pos
                    elif whether_close_or_not <= -np.pi / 2:
                        right_bound_angle = theta_moving
                        left_bound_angle = alpha_1 - alpha_2
                        point = pos
                    elif -np.pi / 2 < whether_close_or_not < 0:
                        a1 = np.tan(alpha_1 - alpha_2)
                        a2 = np.tan(alpha_1 + alpha_2)
                        if abs(a1 - a2) < 0.001:
                            left_bound_angle = alpha_1 + alpha_2
                            right_bound_angle = alpha_1 - alpha_2
                        else:
                            vel_moving = v_moving * np.array([ np.cos(theta_moving), np.sin(theta_moving)])
                            vel_vo = pos + vel_moving
                            vel_rvo = pos + (vel + vel_moving) / 2
                            b1 = -vel_vo[0] * a1 + vel_vo[1]
                            b2 = -vel_rvo[0] * a2 + vel_rvo[1]
                            cross = np.array([(b2 - b1) / (a1 - a2), (a1 * b2 - b1 * a2) / (a1 - a2)])                        
                            left_bound_angle = np.arctan2(cross[1] - pos[1], cross[0] - pos[0])
                            right_bound_angle = alpha_1 - alpha_2
                        point = pos
                    elif 0 <= whether_close_or_not < np.pi / 2:
                        a1 = np.tan(alpha_1 - alpha_2)
                        a2 = np.tan(alpha_1 + alpha_2)
                        if abs(a1 -a2) < 0.001:
                            left_bound_angle = alpha_1 + alpha_2
                            right_bound_angle = alpha_1 - alpha_2
                        else:
                            vel_moving = v_moving * np.array([ np.cos(theta_moving), np.sin(theta_moving)])
                            vel_vo = pos + vel_moving
                            vel_rvo = pos + (vel + vel_moving) / 2
                            b1 = -vel_rvo[0] * a1 + vel_rvo[1]
                            b2 = -vel_vo[0] * a2 + vel_vo[1]
                            cross = np.array([(b2 - b1) / (a1 - a2), (a1 * b2 - b1 * a2) / (a1 - a2)])                        
                            left_bound_angle = alpha_1 + alpha_2
                            right_bound_angle = np.arctan2(cross[1] - pos[1], cross[0] - pos[0])
                        point = pos
                else:
                    vel_moving = v_moving * np.array([ np.cos(theta_moving), np.sin(theta_moving)])
                    right_bound_angle = alpha_1 - alpha_2
                    left_bound_angle = alpha_1 + alpha_2
                    
                    if self.method == 'vo':
                        point = pos + vel_moving
                    elif self.method == 'rvo':
                        point = pos + (vel + vel_moving) / 2
                    elif self.method == 'hrvo':
                        vel_vo = pos + vel_moving
                        vel_rvo = pos + (vel + vel_moving) / 2
                        vel_from_rvo = pos + vel - vel_rvo
                        if np.arctan2(vel_from_rvo[1], vel_from_rvo[0]) < alpha_1:
                            a1 = np.tan(right_bound_angle)
                            a2 = np.tan(left_bound_angle)
                            if abs(a1 - a2) < 0.001:
                                point = vel_vo
                            else:
                                a = np.array([[a1, -1], [a2, -1]])
                                b = np.array([vel_rvo[0] * a1 - vel_rvo[1], vel_vo[0] * a2 - vel_vo[1]])
                                point = lin.solve(a, b)
                        else:
                            a1 = np.tan(right_bound_angle)
                            a2 = np.tan(left_bound_angle)
                            if abs(a1 - a2) < 0.001:
                                point = vel_vo
                            else:
                                a = np.array([[a2, -1], [a1, -1]])
                                b = np.array([vel_rvo[0] * a2 - vel_rvo[1], vel_vo[0] * a1 - vel_vo[1]])
                                point = lin.solve(a, b)
                    else:
                        print('set hrvo method')
                hrvo_set.append([point, right_bound_angle, left_bound_angle])
        if hrvo_set:
            detection = 1
            if hrvo_method == 'no constraint':
                v_result, theta_result, ispass = self.hrvo_no_constraint(hrvo_set, desired_speed, desired_angle, pos)
            elif hrvo_method == 'suggested':
                v_result, theta_result, ispass = self.hrvo_suggested(hrvo_set, desired_speed, desired_angle, pos, v, theta, theta_dot)
            else:
                v_result, theta_result, ispass = [0, 0, 0]
                print('need to set hrvo method')
        else:
            v_result = desired_speed
            theta_result = desired_angle
            ispass = 0
            detection = 0
        return v_result, theta_result, ispass, detection

    def hrvo_no_constraint(self, hrvo_set, desired_speed, desired_angle, pos):
        if self.check(hrvo_set, desired_speed, desired_angle, pos):
            v_result = desired_speed
            theta_result = desired_angle
            ispass = 0
        else:
            possible_set = []
            speed_set = np.arange(0, self.v_max, self.v_max / self.d_v)
            angle_set = np.arange(desired_angle, desired_angle + 2 * np.pi, 2 * np.pi / self.d_angle)
            for speed in speed_set:
                for angle in angle_set:
                    test = self.check(hrvo_set, speed, angle, pos)
                    if test:
                        possible_set.append([speed, angle])
            if possible_set:
                desired_vel = desired_speed * np.array([np.cos(desired_angle), np.sin(desired_angle)])
                
                optimize = min(possible_set, key=lambda x: lin.norm(x[0] * np.array([np.cos(x[1]), np.sin(x[1])]) - desired_vel))
                v_result = optimize[0]
                theta_result = optimize[1]
                ispass = 1
            else:
                v_result = 0
                theta_result = desired_angle
                ispass = 2
        return v_result, theta_result, ispass

    def hrvo_suggested(self, hrvo_set, desired_speed, desired_angle, pos, v, theta, theta_dot):
        if self.check(hrvo_set, desired_speed, desired_angle, pos) and self.acc_check(v, theta, desired_speed, desired_angle):
            v_result = desired_speed
            theta_result = desired_angle
            ispass = 0
        else:
            possible_set = []

            if v <= 1e-3:
                angle_lb = theta - np.pi
                angle_ub = theta + np.pi
                angle_set = np.arange(angle_lb, angle_ub, (angle_ub - angle_lb) / self.d_angle)
                for angle in angle_set:
                    test = self.check(hrvo_set, v, angle, pos)
                    if test:
                        possible_set.append([v, angle])
            else:
                speed_lb = max(0, v + self.control_tau * self.a_x_lb)
                speed_ub = min(v + self.control_tau * self.a_x_ub, self.v_max)
                speed_set = np.arange(speed_lb, speed_ub, (speed_ub - speed_lb) / self.d_v)
                for speed in speed_set:
                    angle_lb = theta - self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + speed * (1 - np.exp(-self.robust_time / self.control_tau))))
                    angle_ub = theta + self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + speed * (1 - np.exp(-self.robust_time / self.control_tau))))
                    angle_set = np.arange(angle_lb, angle_ub, (angle_ub - angle_lb) / self.d_angle)
                    for angle in angle_set:
                        test = self.check(hrvo_set, speed, angle, pos)
                        if test:
                            possible_set.append([speed, angle])

            if possible_set:
                optimize = min(possible_set, key=lambda x: self.weight[0] * ((x[0] - desired_speed) / self.v_max)**2 + self.weight[1] * (wrap_angle(x[1] - desired_angle) / np.pi)**2)
                v_result = optimize[0]
                theta_result = optimize[1]
                ispass = 1

                '''
                desired_vel = desired_speed * np.array([np.cos(desired_angle), np.sin(desired_angle)])


                optimize = min(possible_set, key=lambda x: lin.norm(x[0] * np.array([np.cos(x[1]), np.sin(x[1])]) - desired_vel))
                v_result = optimize[0]
                theta_result = optimize[1]
                ispass = 1
                '''
            else:
                v_result = max(0, v + self.control_tau * self.a_x_lb)
                index = np.argmax([x[2] - x[1] for x in hrvo_set])
                closest_hrvo = hrvo_set[index]
                right = abs(wrap_angle(closest_hrvo[1] - theta))
                left = abs(wrap_angle(closest_hrvo[2] - theta))
                if right < left:
                    theta_result = theta - self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + v_result * (1 - np.exp(-self.robust_time / self.control_tau))))
                else:
                    theta_result = theta + self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + v_result * (1 - np.exp(-self.robust_time / self.control_tau))))
                ispass = 2
        return v_result, theta_result, ispass

    def check(self, hrvo_set, speed, angle, pos):
        test_desired = True
        for velocity_obstacle in hrvo_set:
            if speed == 0:
                angle_test = angle
                right_bound_angle = velocity_obstacle[1]
                left_bound_angle = velocity_obstacle[2]
                right_test = wrap_angle(right_bound_angle - angle_test)
                left_test = wrap_angle(left_bound_angle - angle_test)
                if right_test < 0 and left_test > 0:
                    test_desired = False
            else:
                desired_vel = speed * np.array([np.cos(angle), np.sin(angle)])
                vel_test = pos + desired_vel
                vel_test_from_point = vel_test - velocity_obstacle[0]
                if lin.norm(vel_test_from_point) > 1e-6:
                    angle_test = np.arctan2(vel_test_from_point[1], vel_test_from_point[0])
                    right_bound_angle = velocity_obstacle[1]
                    left_bound_angle = velocity_obstacle[2]
                    right_test = wrap_angle(right_bound_angle - angle_test)
                    left_test = wrap_angle(left_bound_angle - angle_test)
                    if right_test < 0 and left_test > 0:
                        test_desired = False
        result = test_desired
        return result

    def acc_check(self, v, theta, speed, angle):
        result = False
        if v == 0 and theta == 0:
            result = True
        else:
            a_x = -v / self.control_tau + speed / self.control_tau
            a_y = (v * np.exp(-self.robust_time / self.control_tau) + speed * (1 - np.exp(-self.robust_time / self.control_tau))) * self.K * (angle - theta) * (1 - np.exp(-self.robust_time / self.control_tau))
            if self.a_x_lb <= a_x <= self.a_x_ub and -self.a_y_ub <= a_y <= self.a_y_ub:
                result = True

        return result        

    def object_function(self, x, *desired_vel):
        y = lin.norm(x[0] * np.array([np.cos(x[1]), np.sin(x[1])]) - desired_vel)
        
        return y

    def change_vel(self, states):
        v_Gi_b = np.array([states['dynamic'][0], states['dynamic'][1], 0])
        omega = np.array([0, 0, states['dynamic'][2]])
                
        v_Ri_b = v_Gi_b + np.cross(omega, self.GR)
        v_Li_b = v_Gi_b + np.cross(omega, self.GL)
                
        return v_Ri_b, v_Li_b

    