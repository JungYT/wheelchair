import gym
from gym import spaces
import numpy as np
from fym.core import BaseSystem


class Driver(BaseSystem):
    def __init__(self, initial_state, setting):
        self.m = setting['env']['driver_mass']
        self.c = setting['env']['driver_damper']
        self.k = setting['env']['driver_spring']
        super().__init__(initial_state=initial_state)
        
    def deriv(self, t, state, control):
        x, y, x_dot, y_dot = state
        a_x, a_y = control

        dxdt = x_dot
        dydt = y_dot
        dx_dotdt = -self.c * x_dot / self.m - self.k * x / self.m + a_x
        dy_dotdt = -self.c * y_dot / self.m - self.k * y / self.m + a_y

        return np.hstack([dxdt, dydt, dx_dotdt, dy_dotdt])
        
