"""
References: E. Porter, S. H. Hamdar, and W. Daamen,
“Pedestrian dynamics at transit stations:
an integrated pedestrian flow modeling approach,”
Transp. A Transp. Sci., vol. 14, no. 5, pp. 468–483, 2018.
"""
import gym
from gym import spaces
import numpy as np
from fym.core import BaseSystem


class Pedestrian(BaseSystem):

    def __init__(self, initial_state, tau):
        super().__init__(initial_state=initial_state)
        self.tau = tau

    def deriv(self, t, state, control):
        x, y, v, theta = state
        v_r, theta_r = control

        dxdt = v*np.cos(theta)
        dydt = v*np.sin(theta)
        dvdt = -v / self.tau + v_r / self.tau
        dthetadt = -theta / self.tau + theta_r / self.tau

        '''
        dxdt = v_r*np.cos(theta_r)
        dydt = v_r*np.sin(theta_r)
        '''
        
        return np.hstack([dxdt, dydt, dvdt, dthetadt])
