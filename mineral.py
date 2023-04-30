"""Calculate electrical propertis of quartz by model of Revil & Glover (1997)

    Reference:
        A.Revil and P.W.J.Glover, Theory of ionic-surface electrical conduction
            in porous media, Phys. Rev. B 55, 1757 – Published 15 January 1997
            DOI:https://doi.org/10.1103/PhysRevB.55.1757
        A.Revil and P.W.J.Glover, Nature of surface electrical conductivity 
            in natural sands, sandstones, and clays, 1998, https://doi.org/10.1029/98GL00296

"""
from typing import Dict
from math import sqrt, exp, log, log10
from logging import Logger
from sys import float_info
from copy import deepcopy

import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
import iapws

import constants as const
from constants import (
    Species,
    IonProp,
    calc_standard_gibbs_energy,
    calc_equibilium_const,
)
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
        """Initialize Quartz class. pH is assumed to be near the neutral.

        Args:
            nacl (NaCl): Instance of NaCl
            gamma_o (float): Surface site density (Unit: sites/nm^2).
            k_plus (float): Equilibrium constants of SiOH+ + H+ ⇔ SiOH2 at 20℃
            k_minus (float): Equilibrium constants of SiOH ⇔ SiO- + H+ at 20℃
            pzc (float): pH at point of zero charge
            potential_stern (float): Stern plane potential
            logger (Logger): Logger
        """
        assert pzc is not None or k_plus is not None, "Either pzc or k_plus must be set"
        self.gamma_o: float = gamma_o * 1.0e18
        self.k_plus: float = k_plus
        self.k_minus: float = k_minus
        self.potential_stern: float = None
        self.ion_props: Dict = nacl.get_ion_props()
        self.temperature: float = nacl.get_temperature()
        self.dielec: float = nacl.get_dielec_water()
        self.pressure: float = nacl.pressure
        self.logger: Logger = logger

        # set pH
        self.ph = -1.0 * log10(
            self.ion_props[Species.H.name][IonProp.Concentration.name]
        )

        # re-set k_plus (eq.91)
        if self.k_plus is None:
            _ch_pzc = 10.0 ** (-1.0 * pzc)
            self.k_plus = self.k_minus / (_ch_pzc ** 2)

        # consider temperature dependence of equilibrium constant
        dg_plus = calc_standard_gibbs_energy(self.k_plus, 298.15)
        dg_minus = calc_standard_gibbs_energy(self.k_minus, 298.15)
        self.k_plus = calc_equibilium_const(dg_plus, self.temperature)
        self.k_minus = calc_equibilium_const(dg_minus, self.temperature)

        # set δ
        self.delta = self.k_plus / self.k_minus

        # set η
        self.eta = sqrt(
            8000.0
            * nacl.get_dielec_water()
            * const.BOLTZMANN_CONST
            * self.temperature
            * const.AVOGADRO_CONST
        ) / (const.ELEMENTARY_CHARGE * self.gamma_o)

        # pKw
        self.pkw = -1.0 * log10(const.DISSOSIATION_WATER)

        # stern potential
        self.potential_stern = potential_stern
        if self.potential_stern is None:
            _x = newton(self.__calc_eq106, 1.0e1)
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

        # κ (inverted eq.37 of Revil & Glover (1997), modified)
        _if = 0.0  # ionic strength
        for _s, _prop in self.ion_props.items():
            if _s in (Species.Na.name, Species.Cl.name):
                _if += (
                    _prop[IonProp.Valence.name] ** 2 * _prop[IonProp.Concentration.name]
                )
        _if *= 0.5
        _top = 2000.0 * const.ELEMENTARY_CHARGE ** 2 * _if * const.AVOGADRO_CONST
        _bottom = self.dielec * const.BOLTZMANN_CONST * self.temperature
        self.kappa = sqrt(_top / _bottom)

        # water properties
        water = iapws.IAPWS97(P=self.pressure * 1.0e-6, T=self.temperature)
        self.viscosity: float = iapws._iapws._Viscosity(
            water.rho, self.temperature
        )
        
        self.length_edl = None

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
            + self.delta * 10.0 ** (-2.0 * self.ph) * _x ** 4
            + 1.0 / self.k_minus * 10.0 ** (-self.ph) * _x ** 2
        )
        _t5 = self.delta * 10.0 ** (-2.0 * self.ph) * _x ** 4 - 1.0
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
        _cond = 0.0
        for _s, _prop in self.ion_props.items():
            if _s in (Species.H.name, Species.OH.name):
                continue
            _conc = 1000.0 * _prop[IonProp.Concentration.name]
            _v = _prop[IonProp.Valence.name]
            _mobility = _prop[IonProp.MobilityInfDiffuse.name]
            _conc *= exp((-1.0) * _v * _e * phi_x / (_kb * self.temperature))
            _cond += _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_cond_diffuse(self) -> None:
        """Calculate the specific conductivity of diffuse layer by Revil & Glover 1998"""
        # consider temperature dependence (reference temperature is 20~25℃)
        s_edl = self.__calc_edl()
        s_stern = self.__calc_stern()
        s_prot = 2.4e-9
        self.cond_diffuse = (s_edl + s_stern + s_prot) / self.length_edl

    def __calc_edl(self) -> float:
        """Calculate specific conductance of EDL by eq.55 in Revil

        Returns:
            float: Spesicic conductivity of EDL
        """
        xd = sqrt(
            0.5
            * (self.dielec * const.BOLTZMANN_CONST * self.temperature)
            / (
                1000.0
                * const.AVOGADRO_CONST
                * const.ELEMENTARY_CHARGE ** 2
                * (
                    self.ion_props[Species.Na.name][IonProp.Concentration.name]
                    + self.ion_props[Species.H.name][IonProp.Concentration.name]
                )
            )
        )
        coeff = 2000.0 * const.AVOGADRO_CONST * xd * const.ELEMENTARY_CHARGE
        n: float = 0.0
        for _s, _prop in self.ion_props.items():
            # Currently mobility of H+ and OH- are not calculated well
            if _s in (Species.H.name, Species.OH.name):
                continue
            v = _prop[IonProp.Valence.name]
            b = _prop[
                IonProp.MobilityInfDiffuse.name
            ] + 2.0 * self.dielec * const.BOLTZMANN_CONST * self.temperature / (
                self.viscosity * const.ELEMENTARY_CHARGE * v
            )
            n += (
                b
                * _prop[IonProp.Concentration.name]
                * (exp(
                    -v
                    * const.ELEMENTARY_CHARGE
                    * self.potential_stern
                    / (2.0 * const.BOLTZMANN_CONST * self.temperature)
                ))
            )
        self.length_edl = xd
        return coeff * n

    def __calc_stern(self) -> float:
        """Calculate stern layer conductivity by eq.(9) in Revil & Glover, 1998

        Returns:
            float: Specific conductivity of stern layer
        """
        e = const.ELEMENTARY_CHARGE
        bs = 0.4e-8 * (1.0 + 0.037 * (self.temperature - 293.15))
        _dg = calc_standard_gibbs_energy(10.0 ** (-2.77), 293.15)
        km = calc_equibilium_const(_dg, self.temperature)
        gamma_0: float = 10.0e18
        return e * bs * self.__calc_ohm(km) * gamma_0

    def __calc_ohm(self, km: float) -> float:
        """Calculate eq.(10) in Revil & Glover (1998)
        (also refered eq.(84) in Revil & Glover (1997))

        Returns:
            float: Ωm
        """
        # Assuming Cf equals Na+ concentration
        cf = self.ion_props[Species.Na.name][IonProp.Concentration.name]
        ch = self.ion_props[Species.H.name][IonProp.Concentration.name]
        top = km * cf
        bottom = (
            ch
            + self.k_minus
            + top
        )
        return top / bottom

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
        return self.length_edl


from matplotlib import pyplot as plt

if __name__ == "__main__":
    pass
