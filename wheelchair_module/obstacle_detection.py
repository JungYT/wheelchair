"""
Thie module calculates social force.
Thie modeul consists of
    1. checking whether static obstacles is in the detection range or not.
        - input: agent state, static obstacles information, detection range
        - output: coordinate of static obstacles which is in detection range
    2. checking whether moving obstacles is in the detection range or not.
        - input: agent state, moving obstacles state, detection range
        - output: whether moving obstacles are in detection range or not
    3. calculating attractive force of goal
        - input: agent state, goal, static obstalces information, moving obstacles information
        - output: attractive force
    4. calculating repulsive force of static obstacles
        - input: agent state, static obstacles information
        - output: repulsive force of static obstacles
    5. calculating repulsive force of moving obstacles
        - input: agent state, moving obstacles information
        - output: repulsive force of moving obstacles
    6. getting total social force
        - input: agent state, static obstacles information, moving obstacles information, detection range, goal
        - output: social force
"""
import numpy as np
import numpy.linalg as lin
from shapely.geometry import LineString, Point
import numdifftools as nd
from wheelchair_module.wrap_angle import wrap_angle


class ObstacleDetection():
    def __init__(self, setting):
        self.detection_range = setting['obstacle_detection']['detection_range']
        self.detection_angle = setting['obstacle_detection']['detection_angle']
        self.static_obstacle = setting['static_obstacle_set']
        self.multiobstaclepoint = setting['obstacle_detection']['multi_obstacle_point']

    def static_obstacle_detection(self, state, action, destination):
        pos = Point(state[0:2])
        theta = state[2]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        pos_static_obstacle = []
        distance_static_obstacle = []
        n = len(self.static_obstacle)
        goal = action[0]
        distance_to_goal = lin.norm(np.array(np.array(pos) - goal))
        for i in np.arange(n):
            poly = self.static_obstacle[i]
            line = LineString(poly.exterior.coords)
            dis = np.array(line.distance(pos))
            if dis < self.detection_range and dis < distance_to_goal and destination == False:
                closest_point = np.array(line.interpolate(line.project(pos)))
                theta_obstacle = np.arctan2(closest_point[1]-state[1], closest_point[0]-state[0])
                diff = wrap_angle(theta - theta_obstacle)
                if abs(diff) < self.detection_angle / 2:
                    pos_static_obstacle.append(closest_point)
                    if self.multiobstaclepoint == 'off':
                        distance_static_obstacle.append(dis)

        if self.multiobstaclepoint == 'on':
            pos_obstacle = pos_static_obstacle
        else:
            if distance_static_obstacle:
                idx = distance_static_obstacle.index(min(distance_static_obstacle))
                pos_obstacle = [pos_static_obstacle[idx]]
            else:
                pos_obstacle = []
        return pos_obstacle







        



