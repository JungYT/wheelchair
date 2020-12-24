import numpy as np
import gym
from gym import spaces
from wheelchair_module.pedestrian import Pedestrian
from fym.core import BaseEnv, infinite_box


class MovingObstacle(BaseEnv):
    def __init__(self, setting):
        systems = {}
        self.n = len(setting['moving_obstacle'])
        for i in np.arange(self.n):
            systems[f'pedestrian_{i}'] = Pedestrian(
                initial_state=setting['moving_obstacle'][f'{i}']['initial_state']
            )

        self.observation_space = infinite_box((
            len(setting['moving_obstacle']['1']['initial_state'])
            * self.n,
        ))
        self.action_space = infinite_box((
            2 * self.n,
        ))

        super().__init__(
            systems=systems,
            dt=setting['env']['time_step'],
            max_t=setting['env']['final_time']
        )

    def reset(self, noise=0):
        super().reset()
        return self.states

    def derivs(self, t, states, action):
        xdot = {}
        for i in np.arange(self.n):
            xdot[f'pedestrian_{i}'] = self.systems[f'pedestrian_{i}'].deriv(t, states[f'pedestrian_{i}'], action[f'pedestrian_{i}'])

        return self.unpack_state(xdot)

    def step(self, action):
        states = self.states
        time = self.clock.get()

        next_states, full_hist = self.get_next_states(time, states, action)

        reward = 0
        
        done = self.is_terminal()

        info = {'states': states, 'next_states': self.states}

        return next_states, reward, done, info

    def is_terminal(self):
        if self.clock.get() > self.clock.max_t:
            return True
        else:
            return False
