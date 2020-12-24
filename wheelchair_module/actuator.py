'''
refrence: M. Deng, A.Inoue, K. Sekiguchi, L. Jian, 
"Two-wheeled mobile robot motion control in dynamic environments," 2009
'''
import gym
import numpy as np
from fym.core import BaseSystem


class Actuator(BaseSystem):
    def __init__(self, initial_state, tau):
        self.tau = tau
        super().__init__(initial_state)
        
    def deriv(self, t, x, u):
        dx = -x / self.tau + u / self.tau
        return dx
        
