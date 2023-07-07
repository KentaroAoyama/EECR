"""Calculate electrical propertis of quartz by model of Revil & Glover (1997)

    Reference:
        A.Revil and P.W.J.Glover, Theory of ionic-surface electrical conduction
            in porous media, Phys. Rev. B 55, 1757 – Published 15 January 1997
            DOI:https://doi.org/10.1103/PhysRevB.55.1757
        A.Revil and P.W.J.Glover, Nature of surface electrical conductivity 
            in natural sands, sandstones, and clays, 1998, https://doi.org/10.1029/98GL00296
        P.J. Scales, Electrokinetics of the Muscovite Mica-Aqueous Solution Interface,
            1989, https://doi.org/10.1021/la00093a012
        Leroy P., Devau N., Revil A., Bizi M., Influence of surface conductivity on the apparent
            zeta potential of amorphous silica nanoparticles, 2013, https://doi.org/10.1016/j.jcis.2013.08.012
"""
# TODO: docstring
from typing import Dict
from math import sqrt, exp, log, log10, sinh, cosh
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
        gamma_o: float = 4.2,
        k_plus: float = None,
        k_minus: float = 2.5118864315095823e-07,
        k_na: float = 15.848931924611133,
        c1: float = 3.3,
        pzc: float = 3.0,
        potential_0: float = None,
        potential_stern: float = None,
        potential_zeta: float = None,
        charge_0: float = None,
        charge_stern: float = None,
        charge_diffuse: float = None,
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
            c1 (float): Capacitance of surface used in Leroy et al.(2013) (C/m2)
            pzc (float): pH at point of zero charge. Default value is based on Revil &
                Glover (1997).
            potential_0 (float): Surface plane potential (V)
            potential_stern (float): Stern plane potential (V)
            potential_zeta (float): Zeta plane potential (V)
            charge_0 (float): Charge density at O-layer (C/m2)
            charge_stern (float): Charge density at stern layer (C/m2)
            charge_diffuse (float): Charge density at diffuse layer (C/m2)
            method (str): Methods to calculate the potential of the stern surface
                (solve eq.44 or eq.106)
            logger (Logger): Logger
        """
        assert pzc is not None or k_plus is not None, "Either pzc or k_plus must be set"
        self.gamma_o: float = gamma_o * 1.0e18
        self.k_plus: float = k_plus
        self.k_minus: float = k_minus
        self.k_na: float = k_na
        self.c1 = c1
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

        # κ (inverted eq.(37) of Revil & Glover (1997), modified)
        _if = 0.0  # ionic strength
        for _s, _prop in self.ion_props.items():
            if _s in (Species.Na.name, Species.Cl.name):
                _if += _prop[IonProp.Valence.name] ** 2 * _prop[IonProp.Molarity.name]
        _if *= 0.5
        self.ion_strength = _if
        _top = 2000.0 * const.ELEMENTARY_CHARGE**2 * _if * const.AVOGADRO_CONST
        _bottom = self.dielec_fluid * const.BOLTZMANN_CONST * self.temperature
        self.kappa = sqrt(_top / _bottom)
        self.length_edl = 1.0 / self.kappa

        # electrical properties
        self.potential_0 = potential_0
        self.potential_stern = potential_stern
        self.potential_zeta = potential_zeta
        self.charge_0 = charge_0
        self.charge_stern = charge_stern
        self.charge_diffuse = charge_diffuse
        self.qs_coeff = None
        # stern plane mobility of Na+ (based on Zhang 2011)
        self.mobility_stern = self.ion_props[Species.Na.name][IonProp.Mobility.name] * (
            0.1
            + exp(-7.0 * self.ion_props[Species.Na.name][IonProp.Molarity.name] - 0.105)
        )
        if self.potential_stern is None:
            if method == "eq44":
                self.potential_stern = bisect(self.__calc_eq_44, -0.5, 1.0)
                self.potential_zeta = self.potential_stern
                self.__calc_cond_surface_1997()
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
                self.potential_zeta = self.potential_stern
                self.__calc_cond_surface_1997()
            if method == "leroy2013":
                self.k_minus = 1.0 / self.k_minus
                self.qs_coeff = sqrt(
                    8000.0
                    * self.dielec_fluid
                    * const.BOLTZMANN_CONST
                    * self.temperature
                    * const.AVOGADRO_CONST
                    * self.ion_strength
                )
                self.__calc_cond_surface_2013()

        # calculate conductivity tensor
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

        Qs0 = self.__calc_qs0(Ah, ANa, phidt)

        return Qs0 + Qsd

    def __calc_qs0(self, ah: float, ana: float, phidt: float) -> float:
        """Calculate surface charge density Qs0 (eq.100)
        NOTE: ignore the 5th item in the denominator

        Args:
            ah (float): Activity of H+
            ana (float): Activity of Na+
            phidt (float): φd tilda in eqs.(100) and (103)

        Returns:
            float: Surface charge density (C/m2)
        """
        return (
            const.ELEMENTARY_CHARGE
            * self.gamma_o
            * (
                (self.k_plus * ah * exp(phidt) - self.k_minus / ah * exp(-phidt))
                / (
                    1.0
                    + self.k_plus * ah * exp(phidt)
                    + self.k_minus / ah * exp(-phidt)
                    + self.k_na * ana / ah
                )
            )
        )

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

    def __calc_cond_surface_1997(self) -> None:
        """Calculate the specific conductivity of diffuse layer by Revil & Glover(1998)"""
        s_diffuse = self.__calc_diffuse_1997()
        s_stern = self.__calc_stern_1997()
        # based on Revil & Glover (1998)
        s_prot = 2.4e-9
        self.cond_surface = (s_diffuse + s_stern + s_prot) / self.length_edl

    def __calc_cond_surface_2013(
        self,
        iter_max: int = 1000,
        convergence_condition: float = 1.0e-10,
        oscillation_tol: float = 1.0e-04,
        beta: float = 0.75,
        lamda: float = 2.0,
    ) -> float:
        xn = np.array([-1.0, -0.5, -1.0, 1.0, 1.0], dtype=np.float64).reshape(-1, 1)
        fn = self.__calc_functions(xn)
        norm_fn: float = np.sum(np.sqrt(np.square(fn)), axis=0)[0]
        cou = 0
        while convergence_condition < norm_fn:
            _j = self.__calc_2013_jacobian(xn[0][0], xn[1][0])
            # To avoid overflow when calculating the inverse matrix
            _j = _j + float_info.min
            _j_inv = np.linalg.inv(_j)
            step = np.matmul(_j_inv, fn)
            _cou_damp: int = 0
            _norm_fn_tmp, _rhs = float_info.max, float_info.min
            while _norm_fn_tmp > _rhs:
                # update μ
                _mu = 1.0 / (lamda**_cou_damp)
                # calculate left hand side of eq.(21) of [1]
                xn_tmp: np.ndarray = xn - _mu * step
                fn_tmp = self.__calc_functions(xn_tmp)
                _norm_fn_tmp = np.sum(np.sqrt(np.square(fn_tmp)), axis=0)[0]
                # calculate right hand side of eq.(21) of [1]
                _rhs = (1.0 - (1.0 - beta) * _mu) * norm_fn
                _cou_damp += 1
                if _cou_damp > 10000:
                    break
            xn = xn_tmp
            fn = fn_tmp
            norm_fn = _norm_fn_tmp
            cou += 1
            if cou > iter_max:
                _norm_step = np.sum(np.sqrt(np.square(step)), axis=0)[0]
                if _norm_step > oscillation_tol:
                    raise RuntimeError(
                        f"Loop count exceeded {iter_max} times &"
                        f" exceeds oscillation tolerance: {_norm_step}"
                    )
                else:
                    break
        self.potential_0 = xn[0][0]
        self.potential_stern = xn[1][0]
        self.potential_zeta = xn[1][0]
        self.charge_0 = xn[2][0]
        self.charge_stern = xn[3][0]
        self.charge_diffuse = xn[4][0]

        self.cond_surface = (
            1.53e-9 + self.__calc_diffuse_1997() + self.__calc_stern_1997()
        ) / self.length_edl
        return self.cond_surface

    def __calc_functions(self, xn: np.ndarray) -> np.ndarray:
        f1 = self.__calc_f1(xn[2][0], xn[0][0], xn[1][0], )
        f2 = self.__calc_f2(xn[3][0], xn[0][0], xn[1][0])
        f3 = self.__calc_f3(xn[4][0], xn[1][0])
        f4 = self.__calc_f4(xn[2][0], xn[3][0], xn[4][0])
        f5 = self.__calc_f5(xn[2][0], xn[0][0], xn[1][0])
        fn = np.array([f1, f2, f3, f4, f5]).reshape(-1, 1)
        return fn

    def __calc_f1(self, q0: float, phi0: float, phib: float) -> float:
        # eq.(15)
        A = self.__calc_A(phi0, phib)
        e = const.ELEMENTARY_CHARGE
        gamma_sio = self.gamma_o / A
        gamma_siom = (
            gamma_sio
            * self.k_na
            * self.ion_props[Species.Na.name][IonProp.Activity.name]
            * exp(-e * phib / (const.BOLTZMANN_CONST * self.temperature))
        )
        return q0 + e * (gamma_sio + gamma_siom)

    def __calc_f2(self, qb: float, phi0: float, phib: float) -> float:
        # eq.(16)
        A = self.__calc_A(phi0, phib)
        e = const.ELEMENTARY_CHARGE
        gamma_siom = (
            self.gamma_o
            / A
            * self.k_na
            * self.ion_props[Species.Na.name][IonProp.Activity.name]
            * exp(-e * phib / (const.BOLTZMANN_CONST * self.temperature))
        )
        return qb - e * gamma_siom

    def __calc_f3(self, qs: float, phib: float) -> float:
        # eq.(17)
        return qs - self.qs_coeff * sinh(
            -(
                const.ELEMENTARY_CHARGE
                * phib
                / (2.0 * const.BOLTZMANN_CONST * self.temperature)
            )
        )

    def __calc_f4(self, q0: float, qb: float, qs: float) -> float:
        # eq.(18)
        return q0 + qb + qs

    def __calc_f5(self, q0: float, phi0: float, phib: float) -> float:
        return phi0 - phib - q0 / self.c1

    def __calc_f1_phi0(self, phi0: float, phib: float) -> float:
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29Q+%2B+Divide%5BE*C%2Ca%2Bb*exp%5C%2840%29c*x%5C%2841%29%2Bd*exp%5C%2840%29c*y%5C%2841%29%5D*%5C%2840%291%2Bd*exp%5C%2840%29c*y%5C%2841%29%5C%2841%29%5C%2841%29%2Cx%5D&lang=ja
        e = const.ELEMENTARY_CHARGE
        kbt = const.BOLTZMANN_CONST * self.temperature
        a = 1.0
        b = self.k_minus * self.ion_props[Species.H.name][IonProp.Activity.name]
        c = -e / kbt
        d = self.k_na * self.ion_props[Species.Na.name][IonProp.Activity.name]
        C = self.gamma_o
        return (
            -(e * b * c * C * exp(c * phi0) * (d * exp(c * phib) + 1.0))
            / (a + b * exp(c * phi0) + d * exp(c * phib)) ** 2
        )

    def __calc_f1_phib(self, phi0: float, phib: float) -> float:
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29Q+%2B+Divide%5BA*C%2Ca%2Bb*exp%5C%2840%29c*x%5C%2841%29%2Bd*exp%5C%2840%29c*y%5C%2841%29%5D*%5C%2840%291%2Bd*exp%5C%2840%29c*y%5C%2841%29%5C%2841%29%5C%2841%29%2Cy%5D&lang=ja
        e = const.ELEMENTARY_CHARGE
        kbt = const.BOLTZMANN_CONST * self.temperature
        a = 1.0
        b = self.k_minus * self.ion_props[Species.H.name][IonProp.Activity.name]
        c = -e / kbt
        d = self.k_na * self.ion_props[Species.Na.name][IonProp.Activity.name]
        C = self.gamma_o
        return (e * c * C * d * exp(c * phib) * (a * b * exp(c * phi0) - 1.0)) / (
            a + b * exp(c * phi0) + d * exp(c * phib)
        ) ** 2

    def __calc_f2_phi0(self, phi0: float, phib: float) -> float:
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29Divide%5B-A*C%2Ca%2Bb*exp%5C%2840%29c*x%5C%2841%29%2Bd*exp%5C%2840%29c*y%5C%2841%29%5D*d*exp%5C%2840%29c*y%5C%2841%29%5C%2841%29%2Cx%5D&lang=ja
        e = const.ELEMENTARY_CHARGE
        kbt = const.BOLTZMANN_CONST * self.temperature
        a = 1.0
        b = self.k_minus * self.ion_props[Species.H.name][IonProp.Activity.name]
        c = -e / kbt
        d = self.k_na * self.ion_props[Species.Na.name][IonProp.Activity.name]
        C = self.gamma_o
        return (e * b * c * C * d * exp(c * (phi0 + phib))) / (
            a + b * exp(c * phi0) + d * exp(c * phib)
        ) ** 2

    def __calc_f2_phib(self, phi0: float, phib: float) -> float:
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29Divide%5B-A*C%2Ca%2Bb*exp%5C%2840%29c*x%5C%2841%29%2Bd*exp%5C%2840%29c*y%5C%2841%29%5D*d*exp%5C%2840%29c*y%5C%2841%29%5C%2841%29%2Cy%5D&lang=ja
        e = const.ELEMENTARY_CHARGE
        kbt = const.BOLTZMANN_CONST * self.temperature
        a = 1.0
        b = self.k_minus * self.ion_props[Species.H.name][IonProp.Activity.name]
        c = -e / kbt
        d = self.k_na * self.ion_props[Species.Na.name][IonProp.Activity.name]
        C = self.gamma_o
        return (
            -(e * c * C * d * exp(c * phib) * (a * b * exp(c * phi0)))
            / (a + b * exp(c * phi0) + d * exp(c * phib)) ** 2
        )

    def __calc_f3_phib(self, phib: float) -> float:
        a = -const.ELEMENTARY_CHARGE / (2.0 * const.BOLTZMANN_CONST * self.temperature)
        return -self.qs_coeff * a * cosh(a * phib)

    def __calc_2013_jacobian(self, phi0: float, phib: float) -> np.ndarray:
        jacobian = np.zeros((5, 5), dtype=np.float64)
        jacobian[0][0] = self.__calc_f1_phi0(phi0, phib)
        jacobian[0][1] = self.__calc_f1_phib(phi0, phib)
        jacobian[0][2] = 1.0
        jacobian[1][0] = self.__calc_f2_phi0(phi0, phib)
        jacobian[1][1] = self.__calc_f2_phib(phi0, phib)
        jacobian[1][3] = 1.0
        jacobian[2][1] = self.__calc_f3_phib(phib)
        jacobian[3][2] = 1.0
        jacobian[3][3] = 1.0
        jacobian[3][4] = 1.0
        jacobian[4][0] = 1.0
        jacobian[4][1] = -1.0
        jacobian[4][2] = -1.0 / self.c1
        return jacobian

    def __calc_A(self, phi0, phib) -> float:
        """Calculate eq.(14) in Leroy et al.(2013)

        Returns:
            float: Value of eq.(14)
        """
        ah = self.ion_props[Species.H.name][IonProp.Activity.name]
        ana = self.ion_props[Species.Na.name][IonProp.Activity.name]
        kbt = self.temperature * const.BOLTZMANN_CONST
        e = const.ELEMENTARY_CHARGE
        return (
            1.0
            + self.k_minus * ah * exp(-e * phi0 / kbt)
            + self.k_na * ana * exp(-e * phib / kbt)
        )

    def __calc_diffuse_1997(self) -> float:
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
                IonProp.Mobility.name
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
                        * self.potential_zeta
                        / (2.0 * const.BOLTZMANN_CONST * self.temperature)
                    )
                )
            )
        return coeff * n

    def __calc_stern_1997(self) -> float:
        """Calculate stern layer conductivity by eq.(9) in Revil & Glover (1998)

        Returns:
            float: Specific conductivity of stern layer
        """
        e = const.ELEMENTARY_CHARGE
        return (
            e
            * self.mobility_stern
            * self.__calc_ohm(self.potential_stern)
            * self.gamma_o
        )

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

    def get_surface_charge(self) -> float:
        """Getter for the surface charge density (Qs0 in eq.100)

        Returns:
            float: Surface charge density (C/m2)
        """
        ah = self.ion_props[Species.H.name][IonProp.Activity.name]
        ana = self.ion_props[Species.Na.name][IonProp.Activity.name]
        phidt = self.__calc_phid_tilda(self.potential_stern)
        return self.__calc_qs0(ah, ana, phidt)


if __name__ == "__main__":
    q = Quartz(NaCl(cnacl=0.001), method="leroy2013")
    print(q.get_cond_surface())
    pass
