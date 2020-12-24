import numpy as np


def wrap_angle(angle):
        wraped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return wraped_angle