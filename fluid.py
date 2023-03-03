"""Calculate electrical properties of fluid"""

# pylint: disable=import-error
from typing import Dict
from copy import deepcopy
from logging import Logger

import pickle
import numpy as np

import constants as const


ion_props_default = const.ion_props_default.copy()


class Fluid:
    pass


class NaCl(Fluid):
    # TODO: external_propsクラス (or Dict)をメンバ変数とする
    # TODO: ion_propsとactivitiesをメンバ変数に追加する
    # TODO: ion_propsとactivitiesを統合する
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        temperature: float = 298.15,
        pressure: float = 1.0e5,
        ion_props: Dict = deepcopy(ion_props_default),
        conductivity: float = None,
        cond_tensor: np.ndarray = None,
        logger: Logger = None,
    ):
        self.m_temperature = temperature
        self.m_pressure = pressure
        self.m_ion_props = ion_props
        self.m_conductivity = conductivity
        self.m_cond_tensor = cond_tensor
        self.m_logger = logger

    def sen_and_goode_1992(
        self, temperature: float = 298.15, concentration: float = 1.0e-3
    ) -> float:
        """Calculate conductivity of NaCl fluid based on Sen & Goode, 1992 equation.
        The modified equation was in Watanabe et al., 2021.
        Args:
            temperature (float): temperature in Kelvin.
            pressure (float): pressire in Pa
            concentration (float): NaCl concentration in mol/l.
        """
        # convert Kelvin to Celsius
        temperature -= 273.15
        _m = concentration
        left = (5.6 + 0.27 * temperature - 1.5 * 1.0e-4 * temperature**2) * _m
        right = (2.36 + 0.099 * temperature) / (1.0 + 0.214 * _m**0.5) * _m**1.5
        self.m_conductivity = left - right
        return self.m_conductivity

    def calc_cond_tensor_cube_oxyz(self) -> np.ndarray:
        """Calculate conductivity tensor. The T-O-T plane is the xy-plane,
        and perpendicular to it is the z-axis.

        Returns:
            np.ndarray: 3 rows and 3 columns condutivity tensor
        """
        cond_tensor = np.array(
            [
                [self.m_conductivity, 0.0, 0.0],
                [0.0, self.m_conductivity, 0.0],
                [0.0, 0.0, self.m_conductivity],
            ]
        )
        self.m_cond_tensor = cond_tensor
        if self.m_logger is not None:
            self.m_logger.info(f"{__name__} cond tensor: {self.m_cond_tensor}")
        return deepcopy(self.m_cond_tensor)

    def get_cond_tensor(self) -> np.ndarray or None:
        """Getter for the conductivity tensor

        Returns:
            np.ndarray: Conductivity tensor with 3 rows and 3 columns
        """
        if self.m_cond_tensor is not None:
            return deepcopy(self.m_cond_tensor)
        return self.m_cond_tensor

    def save(self, _pth: str) -> None:
        """Save NaCl class as pickle

        Args:
            _pth (str): path to save
        """
        with open(_pth, "wb") as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)
