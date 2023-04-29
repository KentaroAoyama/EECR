"""Calculate electrical properties of fluid"""
# TODO: H+とOH-の移動度の温度依存性を実装する
# pylint: disable=import-error
from typing import Dict
from copy import deepcopy
from logging import Logger

import pickle
import numpy as np
import iapws

from constants import (
    ion_props_default,
    Species,
    IonProp,
    DISSOSIATION_WATER,
    DIELECTRIC_VACUUM,
)
from msa import calc_mobility


ion_props_default = deepcopy(ion_props_default)


class Fluid:
    pass


class NaCl(Fluid):
    """Class of fluid dissolved only in NaCl"""

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        temperature: float = 298.15,
        pressure: float = 2.0e6,
        cnacl: float = 0.001,
        ph: float = 7.0,
        conductivity: float = None,
        cond_tensor: np.ndarray = None,
        logger: Logger = None,
    ):
        """Initialize NaCl instance

        Args:
            temperature (float): Absolute temperature (K)
            pressure (float): Absolute pressure (Pa).
            cnacl (float): Concentration of NaCl sodium (mol/l).
            ph (float, optional): pH.
            conductivity (float): Electrical conductivity of NaCl fluid
            cond_tensor (np.ndarray): Electrical conductivity tensor (3×3)
            logger (Logger): Logger
        """
        self.temperature = temperature
        self.pressure = pressure
        self.conductivity = conductivity
        self.cond_tensor = cond_tensor
        self.logger = logger

        # TODO: 活量を計算する (それにあわせてTLMのパラメータも修正する必要ある)
        # Set ion_props and activities other than mobility
        ion_props: Dict = deepcopy(ion_props_default)
        for _s, _prop in ion_props.items():
            if _s not in (
                Species.Na.name,
                Species.Cl.name,
                Species.H.name,
                Species.OH.name,
            ):
                del ion_props[_s]
                continue
            if _s == Species.H.name:
                _c = 10.0 ** (-1.0 * ph)
                _prop[IonProp.Concentration.name] = _c
                _prop[IonProp.Activity.name] = _c
                continue
            if _s == Species.OH.name:
                _c = DISSOSIATION_WATER / (10.0 ** (-1.0 * ph))
                _prop[IonProp.Concentration.name] = _c
                _prop[IonProp.Activity.name] = _c
                continue
            # Na or Cl
            _prop[IonProp.Concentration.name] = cnacl
            _prop[IonProp.Activity.name] = cnacl

        # Calculate sodium ion mobility by MSA model and empirical findings of
        # Revil et al. (1998)
        tempe_ref: float = 298.15
        msa_props_tref = calc_mobility(ion_props, tempe_ref, self.pressure)
        msa_props_tgiven = calc_mobility(ion_props, self.temperature, self.pressure)
        for _s, _prop in ion_props.items():
            if _s not in msa_props_tref:
                continue
            _m = msa_props_tref[_s]["mobility"]
            # Under a wide range of NaCl concentrations, the mobility of ions in the electric
            # double layer is 1/10, and linear temperature depandence regardless of the species.
            _m *= 0.1 * (1.0 + 0.037 * (temperature - tempe_ref))
            _prop[IonProp.MobilityInfDiffuse.name] = msa_props_tgiven[_s]["mobility"]
            # based on https://doi.org/10.1029/2008JB006114
            _prop[IonProp.MobilityTrunDiffuse.name] = _m
            if _s == Species.H.name:
                _prop[IonProp.MobilityTrunDiffuse.name] = ion_props_default[
                    IonProp.MobilityTrunDiffuse.name
                ]
            # based on https://doi.org/10.1016/j.jcis.2015.03.047
            _prop[IonProp.MobilityStern.name] = _m

        self.ion_props: Dict = ion_props

        # get dielectric constant
        water = iapws.IAPWS97(P=self.pressure * 1.0e-6, T=self.temperature)
        self.dielec_water: float = (
            iapws._iapws._Dielectric(water.rho, self.temperature) * DIELECTRIC_VACUUM
        )

    def sen_and_goode_1992(self) -> float:
        """Calculate conductivity of NaCl fluid based on Sen & Goode, 1992 equation.
        The modified equation was in Watanabe et al., 2021.

        Returens:
            float: Conductivity of NaCl fluid in liquid phase
        """
        # convert Kelvin to Celsius
        temperature = self.temperature - 273.15
        _m = self.ion_props[Species.Na.name]["Concentration"]
        left = (5.6 + 0.27 * temperature - 1.5 * 1.0e-4 * temperature**2) * _m
        right = (2.36 + 0.099 * temperature) / (1.0 + 0.214 * _m**0.5) * _m**1.5
        self.conductivity = left - right
        return self.conductivity

    def set_cond(self, _cond: float) -> None:
        """Set fluid conductivity

        Args:
            _cond (float): Fluid conductivity
        """
        self.conductivity = _cond

    def calc_cond_tensor_cube_oxyz(self) -> np.ndarray:
        """Calculate conductivity tensor. The T-O-T plane is the xy-plane,
        and perpendicular to it is the z-axis.

        Returns:
            np.ndarray: 3 rows and 3 columns condutivity tensor
        """
        cond_tensor = np.array(
            [
                [self.conductivity, 0.0, 0.0],
                [0.0, self.conductivity, 0.0],
                [0.0, 0.0, self.conductivity],
            ]
        )
        self.cond_tensor = cond_tensor
        if self.logger is not None:
            self.logger.info(f"{__name__} cond tensor: {self.cond_tensor}")
        return deepcopy(self.cond_tensor)

    def get_ion_props(self) -> Dict:
        """Getter for the ion_props

        Returns:
            Dict: Ion properties
        """
        return deepcopy(self.ion_props)

    def get_pressure(self) -> float:
        """Getter for the pressure
        Returns:
            float: Absolute pressure
        """
        return self.pressure

    def get_temperature(self) -> float:
        """Getter for the temperature

        Returns:
            float: Absolute temperature
        """
        return self.temperature

    def get_dielec_water(self) -> float:
        """Getter for the permittivity of water

        Returns:
            float: permittivity of water
        """
        return self.dielec_water

    def get_cond(self) -> float:
        """Getter for the electrical conductivity of fluid

        Returns:
            float: electrical conductivity (S/m)
        """
        return self.conductivity

    def get_cond_tensor(self) -> np.ndarray or None:
        """Getter for the conductivity tensor

        Returns:
            np.ndarray: Conductivity tensor with 3 rows and 3 columns
        """
        if self.cond_tensor is not None:
            return deepcopy(self.cond_tensor)
        return self.cond_tensor

    def save(self, _pth: str) -> None:
        """Save NaCl class as pickle

        Args:
            _pth (str): path to save
        """
        with open(_pth, "wb") as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    pass
