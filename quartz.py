"""Calculate electrical propertis of quartz by model of Revil & Glover (1997)

    Reference:
        A.Revil and P.W.J.Glover, Theory of ionic-surface electrical conduction
        in porous media, Phys. Rev. B 55, 1757 – Published 15 January 1997
        DOI:https://doi.org/10.1103/PhysRevB.55.1757
"""
from math import sqrt, exp, log, log10
from logging import Logger
from sys import float_info
from copy import deepcopy

import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad

import constants as const
from constants import Species, IonProp
from fluid import NaCl


class Quartz:
    """Containing electrical properties of quartz"""

    def __init__(
        self,
        nacl: NaCl,
        gamma_o: float = 1.5,
        k_plus: float = None,
        k_minus: float = 5.01187234e-7,
        pzc: float = 3.0,
        potential_stern: float = None,
        logger: Logger = None,
    ):
        """Initialize Quartz class

        Args:
            nacl (NaCl): Instance of NaCl
            gamma_o (float): Surface site density (Unit: sites/nm^2).
            k_plus (float): Equilibrium constants of SiOH+ + H+ ⇔ SiOH2
            k_minus (float): Equilibrium constants of SiOH ⇔ SiO- + H+
            pzc (float): pH at point of zero charge
            potential_stern (float): Stern plane potential
            logger (Logger): Logger
        """
        assert pzc is not None or k_plus is not None, "Either pzc or k_plus must be set"
        self.gamma_o = gamma_o * 1.0e18
        self.k_plus = k_plus
        self.k_minus = k_minus
        self.potential_stern = None
        self.ion_props = nacl.get_ion_props()
        self.temperature = nacl.get_temperature()
        self.logger = logger
        # set pH
        self.ph = -1.0 * log10(
            self.ion_props[Species.H.name][IonProp.Concentration.name]
        )

        # re-set k_plus (eq.91)
        if self.k_plus is None:
            _ch_pzc = 10.0 ** (-1.0 * pzc)
            self.k_plus = self.k_minus / (_ch_pzc**2)

        # set δ
        self.delta = self.k_plus / self.k_minus

        # set η
        self.eta = sqrt(
            8000.0
            * const.calc_dielectric_const_water(self.temperature)
            * const.BOLTZMANN_CONST
            * self.temperature
            * const.AVOGADRO_CONST
        ) / (const.ELEMENTARY_CHARGE * self.gamma_o)

        # pKw
        self.pkw = -1.0 * log10(const.DISSOSIATION_WATER)

        # stern potential
        self.potential_stern = potential_stern
        if self.potential_stern is None:
            _x = newton(self.__calc_eq106, 1.0)
            self.potential_stern = (
                -1.0
                * (
                    2.0
                    * const.BOLTZMANN_CONST
                    * self.temperature
                    / const.ELEMENTARY_CHARGE
                )
                * log(_x)
            )

        # κ (inverted eq.37)
        _if = 0.0  # ionic strength
        for _s, _prop in self.ion_props.items():
            if _s in (Species.Na.name, Species.Cl.name):
                _if += (
                    _prop[IonProp.Valence.name] ** 2 * _prop[IonProp.Concentration.name]
                )
        _if *= 0.5
        _top = 2.0 * const.ELEMENTARY_CHARGE**2 * _if
        _bottom = (
            const.calc_dielectric_const_water(self.temperature)
            * const.BOLTZMANN_CONST
            * self.temperature
        )
        self.kappa = sqrt(_top / _bottom)

        # calculate conductivity tensor
        self.__calc_cond_diffuse()
        self.__calc_cond_tensor()

    def __calc_eq106(self, _x: float) -> float:
        """Calculate eq.(106) in Revil & Glover (1997)

        Args:
            _x (float): Distance from surface (stern plane)

        Returns:
            float: F[X]
        """
        if _x == 0.0:
            return -1 * float_info.max
        _cf = self.ion_props[Species.Na.name][IonProp.Concentration.name]
        _t1 = self.eta / 2.0
        _t2 = sqrt(_cf + 10.0 ** (-1.0 * self.ph) + 10.0 ** (self.ph - self.pkw))
        _t3 = _x - 1.0 / _x
        _t4 = (
            1.0
            + self.delta * 10.0 ** (-2.0 * self.ph) * _x**4
            + 1.0 / self.k_minus * 10.0 ** (-self.ph) * _x**2
        )
        _t5 = self.delta * 10.0 ** (-2.0 * self.ph) * _x**4 - 1.0
        return _t1 * _t2 * _t3 * _t4 + _t5

    def __calc_cond_at_x_inf_diffuse(self, _x: float) -> float:
        """Calculate the conductivity of the infinite diffuse layer.

        Args:
            _x (float): Distance from the surface (stern layer)

        Returns:
            float: Consuctivity at the point _x
        """
        phi_x = self.potential_stern * exp(-1.0 * self.kappa * _x)
        _e = const.ELEMENTARY_CHARGE
        _kb = const.BOLTZMANN_CONST
        _na = const.AVOGADRO_CONST
        _cond = 0.
        for _, _prop in self.ion_props.items():
            _conc = 1000.0 * _prop[IonProp.Concentration.name]
            _v = _prop[IonProp.Valence.name]
            _mobility = _prop[IonProp.MobilityInfDiffuse.name]
            _conc *= exp((-1.0) * _v * _e * phi_x / (_kb * self.temperature))
            _cond += _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_cond_diffuse(self) -> None:
        """Calculate the specific conductivity of diffuse layer"""
        _xdl = 1.0 / self.kappa
        cond_ohmic_diffuse, _ = quad(
            self.__calc_cond_at_x_inf_diffuse, 0.0, 1.0 / self.kappa
        )
        self.cond_diffuse = cond_ohmic_diffuse / _xdl

    def __calc_cond_tensor(self):
        """Calculate the conductivity tensor"""
        cond_silica = 1.0e-12
        cond_tensor = np.array(
            [[cond_silica, 0.0, 0.0], [0.0, cond_silica, 0.0], [0.0, 0.0, cond_silica]],
            dtype=np.float64,
        )
        self.cond_tensor = cond_tensor

    def get_cond_tensor(self) -> np.ndarray or None:
        """Getter for the conductivity tensor

        Returns:
            np.ndarray: Conductivity tensor with 3 rows and 3 columns
        """
        if self.cond_tensor is not None:
            return deepcopy(self.cond_tensor)
        return self.cond_tensor

    def get_potential_stern(self) -> float or None:
        """Getter for the stern potential

        Returns:
            float: Stern plane potential
        """
        return self.potential_stern

    def get_cond_infdiffuse(self) -> float or None:
        """Getter for the conductivity of infinite diffuse layer

        Returns:
            float: Conductivity of the stern layer (Unit: S/m)
        """
        return self.cond_diffuse

    def get_double_layer_length(self) -> float or None:
        """Getter for the double layer length (surface to the end of the diffuse layer)

        Returns:
            float or None: Length of the electrical double layer
        """
        return 1.0 / self.kappa


if __name__ == "__main__":
    pass
