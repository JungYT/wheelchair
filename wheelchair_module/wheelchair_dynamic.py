import gym
from gym import spaces
import numpy as np
from fym.core import BaseSystem


class WheelchairDynamic(BaseSystem):
    def __init__(self, initial_state, setting):
        self.m = setting['wheelchair']['mass'] + setting['env']['driver_mass']
        self.J = setting['wheelchair']['J']
        super().__init__(initial_state=initial_state)
        
    def deriv(self, t, state, control):
        vx, vy, omega = state
        F_x, F_y, M = control

        dvxdt = F_x / self.m + vy * omega
        dvydt = F_y / self.m - vx * omega
        domegadt = M / self.J

        return np.hstack([dvxdt, dvydt, domegadt])
        
