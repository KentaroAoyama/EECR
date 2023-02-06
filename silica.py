"""Calculate the conductivity of silica"""

from copy import deepcopy
import numpy as np


class Silica:
    
    def __init__(self):
        self.m_cond_tensor: np.ndarray = np.array([[1.0e-12, 0., 0.],
                                                   [0., 1.0e-12, 0.],
                                                   [0., 0., 1.0e-12]],
                                                  dtype=np.float64)
    
    def get_cond_tensor(self):
        return deepcopy(self.m_cond_tensor)
