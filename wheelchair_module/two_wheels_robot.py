'''
refrence: M. Deng, A.Inoue, K. Sekiguchi, L. Jian, 
"Two-wheeled mobile robot motion control in dynamic environments," 2009
'''
import gym
from gym import spaces
import numpy as np
from fym.core import BaseSystem


class TwoWheelsRobot3Dof(BaseSystem):
    def __init__(self, initial_state, setting):
        self.initial_state = initial_state
        self.W = setting['wheelchair']['width']
        super().__init__(initial_state=self.initial_state)
        
    def deriv(self, t, state, control):
        _, _, theta = state
        v_right, v_left = control

        dxdt = (v_right + v_left) * np.cos(theta) / 2
        dydt = (v_right + v_left) * np.sin(theta) / 2
        dthetadt = (v_right - v_left) / self.W
        
        return np.hstack([dxdt, dydt, dthetadt])
        
