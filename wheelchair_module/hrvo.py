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
        self.hrvo_modified = setting['hrvo']['modified']
        self.ispass = 0
        self.moving = False
        self.static = False

        super().__init__(setting)


    def hrvo(self, key, states, control_input, desired_speed, desired_angle, K, hrvo_method, action, destination):
        self.ispass = 0
        self.K = K
        if hrvo_method == 'no constraint':
            self.hrvo_modified = False
        # set state according to key
        moving_obstacle_key = list(states.keys() - {'actuator_right', 'actuator_left', 'caster_right', 'caster_left', 'dynamic', 'driver',key})
        pos = states[key][0:2]
        theta_dot = states['dynamic'][2]
        if key == 'wheelchair':
            theta = states[key][2]
            v_Ri_b, v_Li_b = self.change_vel(states)
            v = (v_Ri_b[0] + v_Li_b[0]) / 2
            
            theta_hrvo = theta
            ## could be changed
            if self.hrvo_modified:
                pos_modified = pos + np.array([ v * np.cos(theta), v * np.sin(theta)]) * np.exp(-self.robust_time / self.control_tau) 
                theta_modified = theta + (v_Ri_b[0] - v_Li_b[0]) / self.L * np.exp(-self.robust_time / self.control_tau)
                theta_hrvo = theta_modified
        else:
            v = states[key][2]
            theta = states[key][3]
            theta_hrvo = theta

        # vel = np.array([v * np.cos(theta_hrvo), v * np.sin(theta_hrvo)])
        vel = np.array([v * np.cos(theta), v * np.sin(theta)])

        # set static obstacles
        static_obstacles = self.static_obstacle_detection(np.hstack([pos, theta]), action, destination)
        
        # make hrvo set
        hrvo_set = []

        if static_obstacles:
            self.static = True
        else:
            self.static = False

        # make static obstacle hrvo set
        for i in np.arange(len(static_obstacles)):
            pos_hrvo = pos
            pos_static = static_obstacles[i]
            pos_ba = pos_static - pos_hrvo
            
            if self.hrvo_modified:
                alpha_1 = np.arctan2(pos_ba[1], pos_ba[0])
                pos_ba_modified = pos_static - pos_modified
                alpha_1_modified = np.arctan2(pos_ba_modified[1], pos_ba_modified[0])
                if abs(wrap_angle(alpha_1 - alpha_1_modified)) < np.pi / 2:
                    pos_hrvo = pos_modified
                    pos_ba = pos_ba_modified
            
            distance = lin.norm(pos_ba)
            
            if distance < self.r_plant + self.r_static_obstacle:                    distance = self.r_plant + self.r_static_obstacle
            alpha_1 = np.arctan2(pos_ba[1], pos_ba[0])
            alpha_2 = np.arcsin((self.r_plant + self.r_static_obstacle) / distance)
            right_bound_angle = alpha_1 - alpha_2
            left_bound_angle = alpha_1 + alpha_2
            point = pos_hrvo
            pos_hrvo = pos

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
                self.moving = True
                pos_hrvo = pos
                
                if self.hrvo_modified:
                    pos_moving_modified = pos_moving + np.array([v_moving * np.cos(theta_moving), v_moving * np.sin(theta_moving)]) * self.control_tau
                    # pos_ba_modified = pos_moving_modified - pos_modified
                    pos_ba_modified = pos_moving - pos_modified
                    alpha_1_modified = np.arctan2(pos_ba_modified[1], pos_ba_modified[0])

                    if abs(wrap_angle(alpha_1 - alpha_1_modified)) < np.pi / 2:
                        pos_hrvo = pos_modified
                        pos_ba = pos_ba_modified
                        pos_moving = pos_moving_modified

                        distance = lin.norm(pos_ba)
                        if distance < self.r_plant + self.r_moving:                 distance = self.r_plant + self.r_moving
                        alpha_1 = np.arctan2(pos_ba[1], pos_ba[0])
                        alpha_2 = np.arcsin((self.r_plant + self.r_moving) / distance)
                        self.ispass = 5
                    else:
                        distance = self.r_plant + self.r_moving

                if v == 0:
                    whether_close_or_not = wrap_angle(alpha_1 - theta_moving)
                    if whether_close_or_not >= np.pi / 2:
                        right_bound_angle = alpha_1 - alpha_2
                        left_bound_angle = theta_moving
                        point = pos_hrvo
                    elif whether_close_or_not <= -np.pi / 2:
                        right_bound_angle = theta_moving
                        left_bound_angle = alpha_1 - alpha_2
                        point = pos_hrvo
                    elif -np.pi / 2 < whether_close_or_not < 0:
                        a1 = np.tan(alpha_1 - alpha_2)
                        a2 = np.tan(alpha_1 + alpha_2)
                        if abs(a1 - a2) < 0.001:
                            left_bound_angle = alpha_1 + alpha_2
                            right_bound_angle = alpha_1 - alpha_2
                        else:
                            vel_moving = v_moving * np.array([ np.cos(theta_moving), np.sin(theta_moving)])
                            vel_vo = pos_hrvo + vel_moving
                            vel_rvo = pos_hrvo + (vel + vel_moving) / 2
                            b1 = -vel_vo[0] * a1 + vel_vo[1]
                            b2 = -vel_rvo[0] * a2 + vel_rvo[1]
                            cross = np.array([(b2 - b1) / (a1 - a2), (a1 * b2 - b1 * a2) / (a1 - a2)])                        
                            left_bound_angle = np.arctan2(cross[1] - pos_hrvo[1], cross[0] - pos_hrvo[0])
                            right_bound_angle = alpha_1 - alpha_2
                        point = pos_hrvo
                    elif 0 <= whether_close_or_not < np.pi / 2:
                        a1 = np.tan(alpha_1 - alpha_2)
                        a2 = np.tan(alpha_1 + alpha_2)
                        if abs(a1 -a2) < 0.001:
                            left_bound_angle = alpha_1 + alpha_2
                            right_bound_angle = alpha_1 - alpha_2
                        else:
                            vel_moving = v_moving * np.array([ np.cos(theta_moving), np.sin(theta_moving)])
                            vel_vo = pos_hrvo + vel_moving
                            vel_rvo = pos_hrvo + (vel + vel_moving) / 2
                            b1 = -vel_rvo[0] * a1 + vel_rvo[1]
                            b2 = -vel_vo[0] * a2 + vel_vo[1]
                            cross = np.array([(b2 - b1) / (a1 - a2), (a1 * b2 - b1 * a2) / (a1 - a2)])                        
                            left_bound_angle = alpha_1 + alpha_2
                            right_bound_angle = np.arctan2(cross[1] - pos_hrvo[1], cross[0] - pos_hrvo[0])
                        point = pos_hrvo
                else:
                    vel_moving = v_moving * np.array([ np.cos(theta_moving), np.sin(theta_moving)])
                    right_bound_angle = alpha_1 - alpha_2
                    left_bound_angle = alpha_1 + alpha_2
                    
                    if self.method == 'vo':
                        point = pos_hrvo + vel_moving
                    elif self.method == 'rvo':
                        point = pos_hrvo + (vel + vel_moving) / 2
                    elif self.method == 'hrvo':
                        vel_vo = pos_hrvo + vel_moving
                        vel_rvo = pos_hrvo + (vel + vel_moving) / 2
                        vel_from_rvo = pos_hrvo + vel - vel_rvo
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
            else:
                self.moving = False
        if hrvo_set:
            detection = 1
            if hrvo_method == 'no constraint':
                v_result, theta_result, ispass = self.hrvo_no_constraint(hrvo_set, desired_speed, desired_angle, pos)
            elif hrvo_method == 'suggested':
                v_result, theta_result, ispass = self.hrvo_suggested(hrvo_set, desired_speed, desired_angle, pos_hrvo, v, theta, theta_dot)
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
            # angle_set = np.arange(desired_angle, desired_angle + 2 * np.pi, 2 * np.pi / self.d_angle)
            # angle_set = np.arange(desired_angle - np.pi, desired_angle + np.pi, 2 * np.pi / self.d_angle)
            angle_set = np.arange(np.pi, -np.pi, -2 * np.pi / self.d_angle)
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
        # if all([self.check(hrvo_set, desired_speed, desired_angle, pos), self.acc_check(v, theta, desired_speed, desired_angle)]):
        if self.check(hrvo_set, desired_speed, desired_angle, pos):
            theta_result = desired_angle

            # #1
            # if hrvo_set:
            #     v_result = min(v, desired_speed)
            # else:
            #     v_result = desired_speed

            #2
            # v_result = desired_speed

            3
            if self.moving:
                if hrvo_set:
                    v_result = min(v, desired_speed)
                else:
                    v_result = desired_speed
            else:
                v_result = desired_speed


            
            self.ispass += 0
        else:
            possible_set = []
            v_result = min(desired_speed, v)
            if v_result < 1e-4 and v < 1e-4:
                angle_lb = - np.pi
                angle_ub = np.pi
                angle_set = np.arange(angle_lb, angle_ub + (angle_ub - angle_lb) / self.d_angle, (angle_ub - angle_lb) / self.d_angle)
            else:
                angle_lb = theta - self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + v_result * (1 - np.exp(-self.robust_time / self.control_tau))))
                # angle_lb = np.clip(angle_lb, -np.pi, np.pi)
                angle_lb = wrap_angle(angle_lb)
                angle_ub = theta + self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + v_result * (1 - np.exp(-self.robust_time / self.control_tau))))
                # angle_ub = np.clip(angle_ub, -np.pi, np.pi)
                angle_ub = wrap_angle(angle_ub)
                angle_set = np.arange(angle_lb, angle_ub + (angle_ub - angle_lb) / self.d_angle, (angle_ub - angle_lb) / self.d_angle)

            for angle in angle_set:
                test = self.check(hrvo_set, v_result, angle, pos)
                if test:
                    possible_set.append(angle)

            if possible_set:
                # theta_result = min(possible_set, key=lambda x: (wrap_angle(x-desired_angle))**2)
                theta_result = min(possible_set, key=lambda x: (x-desired_angle)**2)
                self.ispass += 1
            else:
                v_result = max(0, v + self.control_tau * self.a_x_lb)
                if v_result == 0 and v == 0:
                    angle_lb = - np.pi
                    angle_ub = np.pi
                    angle_set = np.arange(angle_lb, angle_ub, (angle_ub - angle_lb) / self.d_angle)
                else:
                    angle_lb = theta - self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + v_result * (1 - np.exp(-self.robust_time / self.control_tau))))
                    # angle_lb = np.clip(angle_lb, -np.pi, np.pi)
                    angle_lb = wrap_angle(angle_lb)

                    angle_ub = theta + self.a_y_ub / (self.K * (1 - np.exp(-self.robust_time / self.control_tau)) * (v * np.exp(-self.robust_time / self.control_tau) + v_result * (1 - np.exp(-self.robust_time / self.control_tau))))
                    # angle_ub = np.clip(angle_ub, -np.pi, np.pi)
                    angle_ub = wrap_angle(angle_ub)

                    angle_set = np.arange(angle_lb, angle_ub, (angle_ub - angle_lb) / self.d_angle)

                for angle in angle_set:
                    test = self.check(hrvo_set, v_result, angle, pos)
                    if test:
                        possible_set.append(angle)

                if possible_set:
                    # theta_result = min(possible_set, key=lambda x: (wrap_angle(x-desired_angle))**2)
                    theta_result = min(possible_set, key=lambda x: (x-desired_angle)**2)
                    self.ispass = +2
                else:
                    index = np.argmax([x[2] - x[1] for x in hrvo_set])
                    closest_hrvo = hrvo_set[index]
                    right = abs(wrap_angle(closest_hrvo[1] - theta))
                    left = abs(wrap_angle(closest_hrvo[2] - theta))
                    if right < left:
                        #theta_result = wrap_angle(-closest_hrvo[1])
                        theta_result = angle_lb
                    else:
                        #theta_result = wrap_angle(-closest_hrvo[2])
                        theta_result = angle_ub
                    self.ispass += 3
            
        return v_result, theta_result, self.ispass

    def check(self, hrvo_set, speed, angle, pos):
        test_desired = True
        for velocity_obstacle in hrvo_set:
            if speed < 1e-4:
                angle_test = angle
                right_bound_angle = velocity_obstacle[1]
                left_bound_angle = velocity_obstacle[2]
                right_test = wrap_angle(right_bound_angle - angle_test)
                left_test = wrap_angle(left_bound_angle - angle_test)
                if right_test < -1e-6 and left_test > 1e-6:
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
                    if right_test < -1e-6 and left_test > 1e-6:
                        test_desired = False
        result = test_desired
        return result

    def acc_check(self, v, theta, speed, angle):
        result = False
        if v < 1e-4 and abs(wrap_angle(angle - theta)) < 1e-4:
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

    