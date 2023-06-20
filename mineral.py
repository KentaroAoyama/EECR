"""Calculate electrical propertis of quartz by model of Revil & Glover (1997)

    Reference:
        A.Revil and P.W.J.Glover, Theory of ionic-surface electrical conduction
            in porous media, Phys. Rev. B 55, 1757 – Published 15 January 1997
            DOI:https://doi.org/10.1103/PhysRevB.55.1757
        A.Revil and P.W.J.Glover, Nature of surface electrical conductivity 
            in natural sands, sandstones, and clays, 1998, https://doi.org/10.1029/98GL00296
        P.J. Scales, Electrokinetics of the Muscovite Mica-Aqueous Solution Interface,
            1989, https://doi.org/10.1021/la00093a012
"""
from typing import Dict
from math import sqrt, exp, log, log10, sinh
from logging import Logger
from copy import deepcopy
from sys import float_info

import numpy as np
from scipy.optimize import bisect

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
        k_na: float = 0.0031622776601683794,
        pzc: float = 3.0,
        potential_stern: float = None,
        method: str = "eq106",
        logger: Logger = None,
    ):
        """Initialize Quartz class. pH is assumed to be near the neutral.

        Args:
            nacl (NaCl): Instance of NaCl
            gamma_o (float): Surface site density (Unit: sites/nm^2).
            k_plus (float): Equilibrium constants of >SiOH+ + H+ ⇔ >SiOH2 at 25℃.
                Default value is based on Revil & Glover (1997).
            k_minus (float): Equilibrium constants of >SiOH ⇔ >SiO- + H+ at 25℃.
            k_na (float): Equilibrium constants of >SiOH + Na+ ⇔ SiONa + H+ at 25℃.
                Default value is based on Scales (1989).
            pzc (float): pH at point of zero charge. Default value is based on Revil &
                Glover (1997).
            potential_stern (float): Stern plane potential (V)
            method (str): Methods to calculate the potential of the stern surface
                (solve eq.44 or eq.106)
            logger (Logger): Logger
        """
        assert pzc is not None or k_plus is not None, "Either pzc or k_plus must be set"
        self.gamma_o: float = gamma_o * 1.0e18
        self.k_plus: float = k_plus
        self.k_minus: float = k_minus
        self.k_na: float = k_na
        self.potential_stern: float = None
        self.ion_props: Dict = nacl.get_ion_props()
        self.temperature: float = nacl.get_temperature()
        self.dielec_fluid: float = nacl.get_dielec_fluid()
        self.viscosity: float = nacl.get_viscosity()
        self.logger: Logger = logger
        self.ph: float = -1.0 * log10(
            self.ion_props[Species.H.name][IonProp.Activity.name]
        )
        self.method: str = method.lower()

        # parameters in eq.(106)
        self.delta: float = None
        self.eta: float = None
        self.pkw: float = None

        # re-set k_plus (eq.91)
        if self.k_plus is None:
            # implicitly assume that activity coefficient of proton is 1
            _ch_pzc = 10.0 ** (-1.0 * pzc)
            self.k_plus = self.k_minus / (_ch_pzc**2)

        # consider temperature dependence of equilibrium constant
        dg_plus = calc_standard_gibbs_energy(self.k_plus, 298.15)
        dg_minus = calc_standard_gibbs_energy(self.k_minus, 298.15)
        dg_na = calc_standard_gibbs_energy(self.k_na, 298.15)
        self.k_plus = calc_equibilium_const(dg_plus, self.temperature)
        self.k_minus = calc_equibilium_const(dg_minus, self.temperature)
        self.k_na = calc_equibilium_const(dg_na, self.temperature)

        # stern potential
        self.potential_stern = potential_stern
        if self.potential_stern is None:
            if method == "eq44":
                self.potential_stern = bisect(self.__calc_eq_44, -0.5, 1.0)
            if method == "eq106":
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
                self.pkw = -1.0 * log10(nacl.get_kw())

                # initial upper and lower values
                _x0 = exp(self.__calc_phid_tilda(-0.5) * 0.5)
                _x1 = exp(self.__calc_phid_tilda(1.0) * 0.5)
                _x = bisect(self.__calc_eq106, _x0, _x1)
                # convert X to phid
                self.potential_stern = (
                    -2.0
                    * const.BOLTZMANN_CONST
                    * self.temperature
                    / const.ELEMENTARY_CHARGE
                    * log(_x)
                )

        # κ (inverted eq.(37) of Revil & Glover (1997), modified)
        _if = 0.0  # ionic strength
        for _s, _prop in self.ion_props.items():
            if _s in (Species.Na.name, Species.Cl.name):
                _if += _prop[IonProp.Valence.name] ** 2 * _prop[IonProp.Molarity.name]
        _if *= 0.5
        _top = 2000.0 * const.ELEMENTARY_CHARGE**2 * _if * const.AVOGADRO_CONST
        _bottom = self.dielec_fluid * const.BOLTZMANN_CONST * self.temperature
        self.kappa = sqrt(_top / _bottom)
        self.length_edl = 1.0 / self.kappa

        # calculate conductivity tensor
        self.__calc_cond_surface()
        self.__calc_cond_tensor()

    def __calc_eq_44(self, phid: float) -> float:
        """Calculate eq.(44) (Qs+Qs0) in Revil & Glover (1997)

        Args:
            phid (float): Surface (Stern) plane potential

        Returns:
            float: Value of eq.(44)
        """

        # molarities
        Cf = self.ion_props[Species.Na.name][IonProp.Molarity.name]
        Ch = self.ion_props[Species.H.name][IonProp.Molarity.name]
        Coh = self.ion_props[Species.OH.name][IonProp.Molarity.name]

        # activities
        Ah = self.ion_props[Species.H.name][IonProp.Activity.name]
        ANa = self.ion_props[Species.Na.name][IonProp.Activity.name]

        # phid tilda in eqs.(100) and (103)
        phidt = self.__calc_phid_tilda(phid)

        # charge density at diffuse layer (eq. 103)
        Qsd = sqrt(
            8000.0
            * self.dielec_fluid
            * const.BOLTZMANN_CONST
            * self.temperature
            * const.AVOGADRO_CONST
            * (Cf + Ch + Coh)
        ) * sinh(0.5 * phidt)

        # charge density at the surface (eq.100)
        # NOTE: ignore the 5th item in the denominator
        Qs0 = (
            const.ELEMENTARY_CHARGE
            * self.gamma_o
            * (
                (self.k_plus * Ah * exp(phidt) - self.k_minus / Ah * exp(-phidt))
                / (
                    1.0
                    + self.k_plus * Ah * exp(phidt)
                    + self.k_minus / Ah * exp(-phidt)
                    + self.k_na * ANa / Ah
                )
            )
        )
        return Qs0 + Qsd

    def __calc_phid_tilda(self, phid: float) -> float:
        """Calculate phid tilda in eqs.(100) and (103).

        Args:
            phid (float): Surface (Stern) plane potential (V)

        Returns:
            float: Phid tilda (V)
        """
        return (
            -1.0
            * const.ELEMENTARY_CHARGE
            * phid
            / (const.BOLTZMANN_CONST * self.temperature)
        )

    def __calc_eq106(self, _x: float) -> float:
        """Calculate eq.(106) in Revil & Glover (1997)

        Args:
             _x (float): X in eq.(106)

        Returns:
            float: Value of eq.(106)
        """
        if _x == 0.0:
            return -1 * float_info.max
        _cf = self.ion_props[Species.Na.name][IonProp.Molarity.name]
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

    def __calc_cond_surface(self) -> None:
        """Calculate the specific conductivity of diffuse layer by Revil & Glover(1998)"""
        s_diffuse = self.__calc_diffuse()
        s_stern = self.__calc_stern()
        # based on Revil & Glover (1998)
        s_prot = 2.4e-9
        self.cond_surface = (s_diffuse + s_stern + s_prot) / self.length_edl

    def __calc_diffuse(self) -> float:
        """Calculate specific conductance of EDL by eq.(55) in Revil & Glover(1997)

        Returns:
            float: Spesicic conductivity of EDL
        """
        coeff = (
            2000.0 * const.AVOGADRO_CONST * self.length_edl * const.ELEMENTARY_CHARGE
        )
        n: float = 0.0
        for _s, _prop in self.ion_props.items():
            # Currently H+ and OH- are not considered
            if _s in (Species.H.name, Species.OH.name):
                continue
            v = _prop[IonProp.Valence.name]
            b = _prop[
                IonProp.MobilityInfDiffuse.name
            ] + 2.0 * self.dielec_fluid * const.BOLTZMANN_CONST * self.temperature / (
                self.viscosity * const.ELEMENTARY_CHARGE * v
            )
            n += (
                b
                * _prop[IonProp.Molarity.name]
                * (
                    exp(
                        -v
                        * const.ELEMENTARY_CHARGE
                        * self.potential_stern
                        / (2.0 * const.BOLTZMANN_CONST * self.temperature)
                    )
                )
            )
        return coeff * n

    def __calc_stern(self) -> float:
        """Calculate stern layer conductivity by eq.(9) in Revil & Glover (1998)

        Returns:
            float: Specific conductivity of stern layer
        """
        e = const.ELEMENTARY_CHARGE
        bs = self.ion_props[Species.Na.name][IonProp.MobilityStern.name]
        return e * bs * self.__calc_ohm(self.potential_stern) * self.gamma_o

    def __calc_ohm(self, phid: float) -> float:
        """Calculate eq.(84) in Revil & Glover (1997).
        Also stated in Revil & Glover (1998), eq.(10).

        Returns:
            float: Ω0Na
        """
        Ah = self.ion_props[Species.H.name][IonProp.Activity.name]
        ANa = self.ion_props[Species.Na.name][IonProp.Activity.name]
        # calculate A in eq.(84)
        # NOTE: ignore 5th term
        Ah0 = Ah * exp(self.__calc_phid_tilda(phid))
        A = 1.0 + self.k_plus * Ah0 + self.k_minus / Ah0 + self.k_na * ANa / Ah
        return self.k_na * ANa / (Ah * A)

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

    def get_cond_surface(self) -> float or None:
        """Getter for the conductivity of infinite diffuse layer

        Returns:
            float: Conductivity of the stern layer (Unit: S/m)
        """
        return self.cond_surface

    def get_double_layer_length(self) -> float or None:
        """Getter for the double layer length (surface to the end of the diffuse layer)

        Returns:
            float or None: Length of the electrical double layer
        """
        return self.length_edl


if __name__ == "__main__":
    pass
