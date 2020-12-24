import gym
import numpy as np
from fym.core import BaseSystem


class Caster(BaseSystem):
    def __init__(self, initial_state, setting):
        self.p = setting['caster']['length']
        self.mu = setting['caster']['mu_T']
        self.J = setting['caster']['J']
        super().__init__(initial_state)
        
    def deriv(self, t, x, u):
        _, alpha_dot = x
        F = u

        dalpha = alpha_dot
        dalpha_dot = -self.p * F / self.J

        return np.hstack([dalpha, dalpha_dot])