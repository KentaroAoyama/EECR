"""Calculate electrical properties of phyllosilicate"""
# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=no-member
from typing import Dict, List, Tuple, Union
from logging import Logger
from sys import float_info
from os import path, PathLike
import math
from copy import deepcopy
from functools import partial

import pickle
import numpy as np
from scipy.integrate import quad

import constants as const
from constants import Species, IonProp
from fluid import NaCl

# load initial parameters for Newton-Raphson method
# for smectite, infinite diffuse layer case
smectite_inf_init_pth: PathLike = path.join(
    path.dirname(__file__), "params", "smectite_inf_init.pkl"
)
with open(smectite_inf_init_pth, "rb") as pkf:
    smectite_inf_init_params = pickle.load(pkf)

# for kaolinite
kaolinite_init_pth: PathLike = path.join(
    path.dirname(__file__), "params", "kaolinite_init.pkl"
)
with open(kaolinite_init_pth, "rb") as pkf:
    kaolinite_init_params = pickle.load(pkf)

# for smectite, truncated case
smectite_trun_init_pth: PathLike = path.join(
    path.dirname(__file__), "params", "smectite_trun_init.pkl"
)
with open(smectite_trun_init_pth, "rb") as pkf:
    smectite_trun_init_params = pickle.load(pkf)


class TLMParams:
    """Parameters of triple layer model (TLM)"""

    def __init__(
        self,
        T: float = 298.15,
        gamma_1i: float = 0.0,
        gamma_2i: float = 5.5,
        gamma_3i: float = 5.5,
        qii: float = -1.5,
        k1i: float = 1.0e-10,
        k2i: float = 1.3e-6,
        k3i: float = 1.0e-2,
        k4i: float = 1.122,
        c1i: float = 2.09,
        c2i: float = 5.5,
        gamma_1o: float = 0.0,
        gamma_2o: float = 5.5,
        gamma_3o: float = 5.5,
        qio: float = -1.0,
        k1o: float = 1.0e-10,
        k2o: float = 1.3e-6,
        k3o: float = 1.0e-2,
        k4o: float = 0.95,
        c1o: float = 2.1,
        c2o: float = 0.55,
    ) -> None:
        """
        Args:
            T (float): Absolute temperature (K)
            gamma_1 (float, ※1): Surface site densities of aluminol (sites/nm^2).
            gamma_2 (float, ※1): Surface site densities of sianol (sites/nm^2).
            gamma_3 (float, ※1): Surface site densities of >Si-O-Al-O (sites/nm^2).
            qi (float, ※1): Layer charge density (charge/nm^2).
            k1 (float, ※1): Equilibrium constant of >AlOH2 ⇔ >AlOH + H+
            k2 (float, ※1): Equilibrium constant of >SiOH ⇔ >SiO + H+
            k3 (float, ※1): Equilibrium constant of >XH ⇔ >X + H+ (※2)
            k4 (float, ※1): Equilibrium constant of >XH ⇔ >X + Na+ (※2)
            c1 (float, ※1): Capacitance of stern layer (unit: F/m2).
            c2 (float, ※1): Capacitance of diffuse layer (unit: F/m2).
        ※1 The subscript i indicates that it is used to calculate the properties
            of the interlayer, and o indicates that it is used to calculate the
            properties of the EDLs that develop on the bulk solution
        ※2 >X stands for the surface site >Al-O-Si<
        """
        # Inner part
        self.gamma_1i = gamma_1i
        self.gamma_2i = gamma_2i
        self.gamma_3i = gamma_3i
        self.qii = qii * 1.0e18 * const.ELEMENTARY_CHARGE
        self.k1i = self.calc_K_T(k1i, T)
        self.k2i = self.calc_K_T(k2i, T)
        self.k3i = self.calc_K_T(k3i, T)
        self.k4i = self.calc_K_T(k4i, T)
        self.c1i = c1i
        self.c2i = c2i

        # Outer part
        self.gamma_1o = gamma_1o
        self.gamma_2o = gamma_2o
        self.gamma_3o = gamma_3o
        self.qio = qio * 1.0e18 * const.ELEMENTARY_CHARGE
        self.k1o = self.calc_K_T(k1o, T)
        self.k2o = self.calc_K_T(k2o, T)
        self.k3o = self.calc_K_T(k3o, T)
        self.k4o = self.calc_K_T(k4o, T)
        self.c1o = c1o
        self.c2o = c2o

    def calc_K_T(self, K: float, T: float) -> float:
        """Calculate the equilibrium constant at temperature T.

        Args:
            K (float): Equilibrium constant at 25℃
            T (float): Absolute temperature (K)

        Returns:
            float: Equilibrium constant at T (K)
        NOTE: This equation assumes that the standard Gibbs
            energy of formation is constant
        """
        # NOTE: This equation assumes that the standard Gibbs
        # energy of formation is constant
        return const.calc_equibilium_const(
            const.calc_standard_gibbs_energy(K, 298.15), T
        )


# TODO: modify tensor calculations to be overridden by each inherited class
class Phyllosilicate:
    """
    Phyllosilicate Class
    It has a function to calculate the EDL properties and electrical conductivity
        of inner and outer plane.

    To calculate the surface potential, we use the equations proposed by
    Gonçalvès et al. (2007). Those equations have been modified in the following points:
        1. Eq.(11): Avogadro's number and a factor of 1000 to match concentration
            units were added.
        2. Eq.(16): The Qi unit was modified from "charges nm-2" to "C/m2."
        3. Eq.(23): "phib" was modified to "phi0."
        4. Eq.(16) & (17): 10^18 was multipled by the surface site densities.
        5. Eq.(21): The negative sign was added in sinh.
        6. Eq.(21): The symbol "phiD" was modified to "phid" (uppercase to lowercase).
        7. Eq.(32): Avogadro's number was multiplied by the right hand side.
        8. Eq.(32) & (34): A factor of 1000 was multiplied by the Cf.

    References:
        Gonçalvès J., P. Rousseau-Gueutin, A. Revil, 2007, doi:10.1016/j.jcis.2007.07.023
        Leroy P., A. Revil, 2004, doi:10.1016/j.jcis.2003.08.007
        Leroy P., and A. Revil, 2009, doi:10.1029/2008JB006114
        Leroy P., T. Christophe, B. Olivier, D. Nicolas, A. Mohamed, 2015,
            doi: 10.1016/j.jcis.2015.03.047
        Shirozu, 1998, Introduction to Clay Mineralogy
        Bourg I.C., Sposito G., Molecular dynamics simulations of the electrical
            double layer on smectite surfaces contacting concentrated mixed
            electrolyte (NaCl-CaCl2) solutions, 2011, doi:10.1016/j.jcis.2011.04.063
        Zheng X., Underwood T.R., Bourg I.C., Molecular dynamics simulation of thermal,
            hydraulic, and mechanical properties of bentonite clay at 298 to 373 K,
            2023, https://doi.org/10.1016/j.clay.2023.106964
        Zhang L., Lu X., Liu X., Zhou J., Zhou H., Hydration and Mobility of Interlayer
            Ions of (Nax,Cay)-Montmorillonite: A Molecular Dynamics Study, 2014,
            https://doi.org/10.1021/jp508427c

    NOTE: This class can only treat chemical reactions at the interface with NaCl solution.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        nacl: NaCl,
        layer_width: float = 1.52e-9,
        tlm_params: TLMParams = None,
        potential_0_o: float = None,
        potential_stern_o: float = None,
        potential_zeta_o: float = None,
        charge_0_o: float = None,
        charge_stern_o: float = None,
        charge_diffuse_o: float = None,
        potential_0_i: float = None,
        potential_stern_i: float = None,
        potential_zeta_i: float = None,
        potential_r_i: float = None,
        charge_0_i: float = None,
        charge_stern_i: float = None,
        charge_diffuse_i: float = None,
        xd: float = None,
        cond_intra: float = None,
        cond_infdiffuse: float = None,
        logger: Logger = None,
    ):
        """Initialize phyllosilicate class.

        Args:
            nacl (NaCl): Instance of NaCl class
            layer_width (float): Thickness of the interlayer (m)
            tlm_params (TLMParams): TLM parameter
            potential_0_o (float): Surface potential of the outer surface (V)
            potential_stern_o (float): Stern potential of the outer surface (V)
            potential_zeta_o (float): Zeta potential of the outer surface (V)
            charge_0_o (float): Charge density of the outer surface (C/m^2)
            charge_stern_o (float): Charge density of the Stern layer on the
                outer surface (C/m^2)
            charge_diffuse_o (float): Charge density of the diffuse layer on
                the outer surface (C/m^2)
            potential_0_i (float): Surface potential of the inner surface (V)
            potential_stern_i (float): Stern potential of the inner surface (V)
            potential_zeta_i (float): Zeta potential of the inner surface (V)
            potential_r_i (float): Potential at the midpoint of the interlayer
                (i.e., truncation point) (V)
            charge_0_i (float): Charge density of the inner surface (C/m^2)
            charge_stern_i (float): Charge density of the Stern layer in
                the inner surafce (C/m^2)
            charge_diffuse_i (float): Charge density of the diffuse layer
                in the inner surface (C/m^2)
            xd (float): Distance from the surface (i.e., the center of the oxygen
                in the siloxane surface) to the shear plane (V)
            cond_intra (float): Inter layer conductivity (S/m)
            cond_infdiffuse (float): Conductivity of the outer surface (S/m)
            logger (Logger): Logger for debugging
        """
        # sanity check
        # xd must be smaller than layer width
        if (isinstance(xd, float) or isinstance(xd, int)) and (
            isinstance(layer_width, float) or isinstance(layer_width, int)
        ):
            assert (
                xd < layer_width * 0.5
            ), "xd must not exceed half the length of layer_width"

        ####################################################
        # Parameters related to the external environment
        ####################################################
        self.temperature: float = nacl.get_temperature()
        self.ion_props: Dict = nacl.get_ion_props()
        self.dielec_fluid: float = nacl.get_dielec_fluid()
        self.viscosity: float = nacl.get_viscosity()

        ####################################################
        # Specific parameters of phyllosilicate
        ####################################################
        self.layer_width: float = layer_width
        self.tlm_params: TLMParams = tlm_params
        self.gamma_1: float = None
        self.gamma_2: float = None
        self.gamma_3: float = None
        self.qi: float = None
        self.k1: float = None
        self.k2: float = None
        self.k3: float = None
        self.k4: float = None
        self.c1: float = None
        self.c2: float = None
        self.mobility_stern: float = self.__calc_mobility_stern()

        # electrochemical properties at the outer surface
        self.potential_0_o: float = potential_0_o
        self.potential_stern_o: float = potential_stern_o
        self.potential_zeta_o: float = potential_zeta_o
        self.charge_0_o: float = charge_0_o
        self.charge_stern_o: float = charge_stern_o
        self.charge_diffuse_o: float = charge_diffuse_o
        # electrochemical properties at the inner surface
        self.potential_0_i: float = potential_0_i
        self.potential_stern_i: float = potential_stern_i
        self.potential_zeta_i: float = potential_zeta_i
        self.potential_r_i: float = potential_r_i
        self.charge_0_i: float = charge_0_i
        self.charge_stern_i: float = charge_stern_i
        self.charge_diffuse_i: float = charge_diffuse_i
        self.xd: float = xd
        self.cond_intra: float = cond_intra
        self.cond_infdiffuse: float = cond_infdiffuse
        # auxiliary variables
        self.ionic_strength: float = None
        self.kappa: float = None
        self.kappa_truncated: float = None
        self.qs_coeff1_inf: float = None
        self.qs_coeff2_inf: float = None
        self.cond_tensor: np.ndarray = None
        self.double_layer_length: float = None
        # other properties (TODO:)
        self.partition_coefficient: float = None
        self.swelling_pressure: float = None
        self.osmotic_coefficient: float = None
        self.osmotic_pressure: float = None

        ####################################################
        # LOGGER
        ####################################################
        self.logger = logger

        # start debugging
        if self.logger is not None:
            self.logger.info("=== Initialize Phyllosilicate ===")

        # calculate frequently used parameters.
        self.__init_default()

    def __init_default(self) -> None:
        """Calculate parameters frequently used when computing
        electrochemical properties.
        """
        _e = const.ELEMENTARY_CHARGE
        _kb = const.BOLTZMANN_CONST

        # ionic strength (based on Leroy & Revil, 2004)
        strength = 0.0
        for elem, props in self.ion_props.items():
            if elem in (Species.Na.name, Species.Cl.name):
                strength += (
                    props[IonProp.Molarity.name] * props[IonProp.Valence.name] ** 2
                )
        self.ionic_strength = strength

        # calculate κ (Eq.[11] in Gonçalvès et al., 2007)
        # electrolyte concentration is assumed to be equal to Na+ concentration
        top = (
            2000.0
            * self.ion_props[Species.Na.name][IonProp.Molarity.name]
            * const.AVOGADRO_CONST
            * _e**2
        )
        bottom = self.dielec_fluid * _kb * self.temperature
        self.kappa = np.sqrt(top / bottom)

    def __calc_mobility_stern(self) -> float:
        """Calculate mobility of Na+ at the Stern plane"""
        # NOTE: mobility of the stern layer is assumed to be half of the bulk water
        d_cf = self.ion_props[Species.Na.name][IonProp.Mobility.name] * 0.5
        return d_cf

    def __calc_mobility_diffuse(self, _x: float, _s: str) -> Union[float, None]:
        """Calculate the mobility of Na+ and Cl- in diffuse layer based on
        Bourg & Sposito (2011).

        Args:
            _x (float): Distance from the smectite surface (center of the oxygen
                in the siloxane) (m)
            _s (str): Strings of ion species
        """
        assert _s in (Species.Na.name, Species.Cl.name)
        _m = self.ion_props[_s][IonProp.Mobility.name] * (
            (1.0 - np.exp(-0.14 * _x * 1.0e10))
        )
        if _s == Species.Na.name:
            return _m
        if _s == Species.Cl.name:
            return _m
        return None

    def __calc_f1(self, phi0: float, q0: float) -> float:
        """Calculate Eq.(16) in Gonçalvès et al. (2007).

        Args:
            phi0 (float): Surface potential (V)
            q0 (float): Charge density on the surface (C/m^2)

        Returns:
            float: Left-hand side minus right-hand side of Eq.(16)
        """
        _e = const.ELEMENTARY_CHARGE
        _kb = const.BOLTZMANN_CONST
        ch = self.ion_props[Species.H.name][IonProp.Activity.name]
        x = -_e * phi0 / (_kb * self.temperature)
        a = self.__calc_A(phi0)
        b = self.__calc_B(phi0)
        right1 = self.gamma_1 / a * (ch / self.k1 * np.exp(x) - 1.0)
        right2 = self.gamma_2 / b * (ch / self.k2 * np.exp(x) - 1.0)
        right3 = 2.0 * self.gamma_3
        f1 = q0 - _e * 0.5 * 1.0e18 * (right1 + right2 - right3) - self.qi
        return f1

    def __calc_f2(self, phib: float, qb: float) -> float:
        """Calculate Eq.(17) in Gonçalvès et al. (2007).

        Args:
            phib (float): Stern plane potential (V)
            qb (float): Charge density of Stern layer (C/m^2)

        Returns:
            float: Left-hand side minus right-hand side of Eq.(17)
        """
        _e = const.ELEMENTARY_CHARGE
        _kb = const.BOLTZMANN_CONST
        ch = self.ion_props[Species.H.name][IonProp.Activity.name]
        cna = self.ion_props[Species.Na.name][IonProp.Activity.name]
        x = -_e * phib / (_kb * self.temperature)
        c = self.__calc_C(phib)
        f2 = (
            qb
            - _e
            * self.gamma_3
            / c
            * (ch / self.k3 + cna / self.k4)
            * np.exp(x)
            * 1.0e18
        )
        return f2

    def __calc_f3(self, q0: float, qb: float, qs: float) -> float:
        """Calculate Eq.(18) in Gonçalvès et al. (2007).

        Args:
            q0 (float): Charge density on the surface (C/m^2)
            qb (float): Charge density of the Stern layer (C/m^2)
            qs (float): Charge density of the diffuse layer (C/m^2)
        Returns:
            float: Left-hand side minus right-hand side of Eq.(18)
        """
        return q0 + qb + qs

    def __calc_f4(self, phi0: float, phib: float, q0: float) -> float:
        """Calculate Eq.(19) in Gonçalvès et al. (2007).

        Args:
            phi0 (float): Surface plane potential (V)
            phib (float): Stern plane potential (V)
            q0 (float): Charge density on the surface (C/m^2)
        Returns:
            float: Left-hand side minus right-hand side of Eq.(19)
        """
        return phi0 - phib - q0 / self.c1

    def __calc_f5(self, phib: float, phid: float, qs: float) -> float:
        """Calculate Eq.(20) in Gonçalvès et al. (2007).

        Args:
            phib (float): Stern place potential (V)
            phid (float): Zeta potential (V)
            qs (float): Charge density of the diffuse layer (C/m^2)
        Returns:
            float: Left-hand side minus right-hand side of Eq.(20)
        """
        return phib - phid + qs / self.c2

    def __calc_f6(self, phid: float, qs: float) -> float:
        """Calculate Eq.(21) in Gonçalvès et al. (2007).
        NOTE: When pH is shifted from 7, we may need to take
        into account the contribution of H+ and OH-.

        Args:
            phid (float): Zeta potential (V)
            qs (float): Charge density of the diffuse layer (C/m^2)
        Returns:
            float: Left-hand side minus right-hand side Eq.(21)
        """
        _e = const.ELEMENTARY_CHARGE
        _na = const.AVOGADRO_CONST
        kb = const.BOLTZMANN_CONST
        dielec = self.dielec_fluid
        x = _e * phid / (2.0 * kb * self.temperature)
        coeff = np.sqrt(
            8.0e3 * _na * self.ionic_strength * dielec * kb * self.temperature
        )
        f6 = qs - coeff * np.sinh(-x)
        return f6

    def __calc_f6_truncated(self, phid: float, phir: float, qs: float) -> float:
        """Calculate Eq.(32) in Gonçalvès et al. (2007).
        NOTE: When pH is shifted from 7, we may need to take
        into account the contribution of H+ and OH-.

        Args:
            phid (float): Zeta potential (V)
            phir (float): Truncated plane potential (V)
            qs (float): Charge density of the diffuse layer (C/m^2)
        Returns:
            float: Left-hand side minus right-hand side of Eq.(32)
        """
        dielec = self.dielec_fluid
        kb = const.BOLTZMANN_CONST
        _t = self.temperature
        cf = self.ion_props[Species.Na.name][IonProp.Molarity.name] * 1.0e3
        _e = const.ELEMENTARY_CHARGE
        _na = const.AVOGADRO_CONST
        c = _e / (kb * _t)
        coeff = 2.0 * np.sqrt(_na * cf * kb * _t * dielec)
        right1 = np.cosh(c * phid)
        right2 = np.cosh(c * phir)
        return qs - coeff * np.sqrt(right1 - right2)

    def __calc_f7_truncated(self, phid: float, phir: float) -> float:
        """Calculate Eq.(34) in Gonçalvès et al. (2007).
        NOTE: When pH is shifted from 7, we may need to take
        into account the contribution of H+ and OH-.

        Args:
            phid (float): Zeta potential (V)
            phir (float): Truncated plane potential (V)
        Returns:
            float: Left-hand side minus right-hand side of Eq.(34)
        """
        # TODO: implement sixth derivative?
        # electrolyte concentration is assumed to be equal to Na+
        cf = self.ion_props[Species.Na.name][IonProp.Molarity.name] * 1.0e3
        _e = const.ELEMENTARY_CHARGE
        dielec = self.dielec_fluid
        _t = self.temperature
        _r = self.layer_width / 2.0
        kb = const.BOLTZMANN_CONST
        _na = const.AVOGADRO_CONST
        xd = self.xd
        c = _e / (kb * _t)
        right2_1 = _e * _na * cf / dielec
        right2_2 = np.sinh(c * phir) * (xd - _r) ** 2
        right2 = right2_1 * right2_2
        right3_1 = _e**3 * _na**2 * cf**2 / (12.0 * dielec**2 * kb * _t)
        right3_2 = np.sinh(2.0 * c * phir) * (xd - _r) ** 4
        right3 = right3_1 * right3_2
        return phid - phir - right2 - right3

    def _calc_Goncalves_jacobian(
        self, phi0: float, phib: float, phid: float
    ) -> np.ndarray:
        """Calculate jacobians of f1 to f6.

        Args:
            phi0 (float): Surface potential (V)
            phib (float): Stern plane potential (V)
            phid (float): Zeta potential (V)

        Returns:
            np.ndarray: Jacobians (6×6)
        """
        f1_phi0 = self.__calc_f1_phi0(phi0)
        f1_q0 = 1.0
        f2_phib = self.__calc_f2_phib(phib)
        f2_qb = 1.0
        f3_q0, f3_qb, f3_qs = 1.0, 1.0, 1.0
        f4_phi0, f4_phib, f4_q0 = 1.0, -1.0, -1.0 / self.c1
        f5_phib, f5_phid, f5_qs = 1.0, -1.0, 1.0 / self.c2
        f6_phid = self.__calc_f6_phid(phid)
        f6_qs = 1.0
        jacobian = np.zeros((6, 6), dtype=np.float64)
        jacobian[0][0] = f1_phi0
        jacobian[0][3] = f1_q0
        jacobian[1][1] = f2_phib
        jacobian[1][4] = f2_qb
        jacobian[2][3] = f3_q0
        jacobian[2][4] = f3_qb
        jacobian[2][5] = f3_qs
        jacobian[3][0] = f4_phi0
        jacobian[3][1] = f4_phib
        jacobian[3][3] = f4_q0
        jacobian[4][1] = f5_phib
        jacobian[4][2] = f5_phid
        jacobian[4][5] = f5_qs
        jacobian[5][2] = f6_phid
        jacobian[5][5] = f6_qs
        return jacobian

    def _calc_Goncalves_jacobian_truncated(
        self, phi0: float, phib: float, phid: float, phir: float
    ) -> np.ndarray:
        """Calculate jacobians of f1 to f7 for trucated case.

        Args:
            phi0 (float): Surface potential (V)
            phib (float): Stern plane potential (V)
            phid (float): Zeta potential (V)
            phir (float): Truncated plane potential (V)

        Returns:
            np.ndarray: Jacobians (7×7)
        """
        f1_phi0 = self.__calc_f1_phi0(phi0)
        f1_q0 = 1.0
        f2_phib = self.__calc_f2_phib(phib)
        f2_qb = 1.0
        f3_q0, f3_qb, f3_qs = 1.0, 1.0, 1.0
        f4_phi0, f4_phib, f4_q0 = 1.0, -1.0, -1.0 / self.c1
        f5_phib, f5_phid, f5_qs = 1.0, -1.0, 1.0 / self.c2
        f6_phid = self.__calc_f6_phid_truncated(phid, phir)
        f6_phir = self.__calc_f6_phir_truncated(phid, phir)
        f6_qs = 1.0
        f7_phid = 1.0
        f7_phir = self.__calc_f7_phir_truncated(phir)
        jacobian = np.zeros((7, 7), dtype=np.float64)
        jacobian[0][0] = f1_phi0
        jacobian[0][4] = f1_q0
        jacobian[1][1] = f2_phib
        jacobian[1][5] = f2_qb
        jacobian[2][4] = f3_q0
        jacobian[2][5] = f3_qb
        jacobian[2][6] = f3_qs
        jacobian[3][0] = f4_phi0
        jacobian[3][1] = f4_phib
        jacobian[3][4] = f4_q0
        jacobian[4][1] = f5_phib
        jacobian[4][2] = f5_phid
        jacobian[4][6] = f5_qs
        jacobian[5][2] = f6_phid
        jacobian[5][3] = f6_phir
        jacobian[5][6] = f6_qs
        jacobian[6][2] = f7_phid
        jacobian[6][3] = f7_phir
        return jacobian

    def __calc_f1_phi0(self, phi0: float) -> float:
        """Calculate ∂f1/∂phi0

        Args:
            phi0 (float): Surface potential (V)

        Returns:
            float: ∂f1/∂phi0
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29Divide%5Bc%2C1%2Bb*exp%5C%2840%29-a*x%5C%2841%29%5D*%5C%2840%29b*exp%5C%2840%29-a*x%5C%2841%29-1%5C%2841%29%2BDivide%5Be%2C1%2Bd*exp%5C%2840%29-a*x%5C%2841%29%5D*%5C%2840%29d*exp%5C%2840%29-a*x%5C%2841%29-1%5C%2841%29%5C%2841%29%2Cx%5D&lang=ja
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        ch = self.ion_props[Species.H.name][IonProp.Activity.name]
        a = _e / (kb * self.temperature)
        x = a * phi0
        b = ch / self.k1
        c = self.gamma_1
        d = ch / self.k2
        e = self.gamma_2
        coeff = _e * a * 1.0e18
        top1_1 = c * (np.exp(x) + d) ** 2
        top1_2 = 2.0 * d * e * np.exp(x)
        top1 = b * (top1_1 + top1_2)
        top2 = d * e * np.exp(2.0 * x)
        top3 = b**2 * d * e
        top = top1 + top2 + top3
        bottom1 = 1.0 + b / np.exp(x)
        bottom2 = np.exp(x) + b
        bottom3 = np.square(np.exp(x) + d)
        bottom = bottom1 * bottom2 * bottom3
        return coeff * top / bottom

    def __calc_f2_phib(self, phib: float) -> float:
        """Calculate ∂f2/∂phib

        Args:
            phib (float): Stern plane potential (V)

        Returns:
            float: ∂f2/∂phib
        """
        # https://www.wolframalpha.com/input?key=&i2d=true&i=D%5B%5C%2840%29Divide%5Bd%2C1%2B%5C%2840%29b%2Bc%5C%2841%29*exp%5C%2840%29-a*x%5C%2841%29%5D*%5C%2840%29b%2Bc%5C%2841%29*exp%5C%2840%29-a*x%5C%2841%29%5C%2841%29%2Cx%5D&lang=ja
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        ch = self.ion_props[Species.H.name][IonProp.Activity.name]
        na = self.ion_props[Species.Na.name][IonProp.Activity.name]
        a = _e / (kb * self.temperature)
        exp = np.exp(a * phib)
        b = ch / self.k3
        c = na / self.k4
        d = _e * 1.0e18 * self.gamma_3
        # overflow prevention
        top = a * d * (b + c)
        bottom1 = 1.0 + (b + c) / exp
        bottom2 = exp + b + c
        return top / (bottom1 * bottom2)

    def __calc_f6_phid(self, phid: float) -> float:
        """Calculate ∂f6/∂phid

        Args:
            phid (float): Zeta potential (V)

        Returns:
            float: ∂f6/∂phid
        """
        _e = const.ELEMENTARY_CHARGE
        _na = const.AVOGADRO_CONST
        kb = const.BOLTZMANN_CONST
        _dielec = self.dielec_fluid
        _t = self.temperature
        _i = self.ionic_strength
        a = _e / (2.0 * kb * _t)
        coeff = np.sqrt(8.0e3 * _na * _i * _dielec * kb * _t)
        cosh = a * np.cosh(a * phid)
        return coeff * cosh

    def __calc_f6_phid_truncated(self, phid: float, phir: float) -> float:
        """Calculate ∂f6/∂phid for truncated case

        Args:
            phid (float): Zeta potential (V)
            phir (float): Truncated plane potential (V)

        Returns:
            float: ∂f6/∂phid
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29-b*Sqrt%5Bcosh%5C%2840%29c*x%5C%2841%29-cosh%5C%2840%29c*y%5C%2841%29%5D%5C%2841%29%2Cx%5D&lang=ja
        # (x: phir, y: phid)
        dielec = self.dielec_fluid
        kb = const.BOLTZMANN_CONST
        _t = self.temperature
        _e = const.ELEMENTARY_CHARGE
        # electrolyte concentration is assumed to be equal to Na+ concentration
        cf = self.ion_props[Species.Na.name][IonProp.Molarity.name] * 1.0e3
        _na = const.AVOGADRO_CONST
        b = np.sqrt(_na * cf * kb * _t * dielec)
        c = _e / (kb * _t)
        top = -b * c * np.sinh(c * phid)
        bottom = np.sqrt(np.cosh(c * phid) - np.cosh(c * phir))
        if bottom == 0.0:
            return float_info.max
        return top / bottom

    def __calc_f6_phir_truncated(self, phid: float, phir: float) -> float:
        """Calculate ∂f6/∂phir for truncated case

        Args:
            phid (float): Zeta potential (V)
            phir (float): Truncated plane potential (V)

        Returns:
            float: ∂f6/∂phir
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29-b*Sqrt%5Bcosh%5C%2840%29c*x%5C%2841%29-cosh%5C%2840%29c*y%5C%2841%29%5D%5C%2841%29%2Cy%5D&lang=ja
        dielec = self.dielec_fluid
        kb = const.BOLTZMANN_CONST
        _t = self.temperature
        # electrolyte concentration is assumed to be equal to Na+ concentration
        cf = self.ion_props[Species.Na.name][IonProp.Molarity.name] * 1.0e3
        _e = const.ELEMENTARY_CHARGE
        _na = const.AVOGADRO_CONST
        b = np.sqrt(_na * cf * kb * _t * dielec)
        c = _e / (kb * _t)
        top = b * c * np.sinh(c * phir)
        bottom = np.sqrt(np.cosh(c * phid) - np.cosh(c * phir))
        if bottom == 0.0:
            return float_info.max
        return top / bottom

    def __calc_f7_phir_truncated(self, phir: float) -> float:
        """Calculate ∂f7/∂phir for truncated case

        Args:
            phir (float): Truncated plane potential (V)

        Returns:
            float: ∂f7/∂phir
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29d-x-a*sinh%5C%2840%29b*x%5C%2841%29*%5C%2840%29y-r%5C%2841%29**2-c*sinh%5C%2840%292*b*x%5C%2841%29*%5C%2840%29y-r%5C%2841%29**4%5C%2841%29%2Cx%5D&lang=ja
        # electrolyte concentration is assumed to be equal to Na+ concentration
        cf = self.ion_props[Species.Na.name][IonProp.Molarity.name] * 1.0e3
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.temperature
        _r = self.layer_width * 0.5
        dielec = self.dielec_fluid
        _na = const.AVOGADRO_CONST
        xd = self.xd
        b = _e * _na * cf / dielec
        c = _e / (kb * _t)
        d = _e**3 * _na**2 * cf**2 / (12.0 * dielec**2 * kb * _t)
        _1 = -c * b * np.cosh(c * phir) * (xd - _r) ** 2
        _2 = -2.0 * c * d * np.cosh(2.0 * c * phir) * (xd - _r) ** 4
        return -1.0 + _1 + _2

    def __calc_A(self, phi0: float) -> float:
        """Calculate Eq.(22) in Gonçalvès et al. (2007)

        Args:
            phi0 (float): Surface potential (V)

        Returns:
            float: Value of "A" in Eq.(22)
        """
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        t = self.temperature
        ch = self.ion_props[Species.H.name][IonProp.Activity.name]
        k1 = self.k1
        return 1.0 + ch / k1 * np.exp(-_e * phi0 / (kb * t))

    def __calc_B(self, phi0: float) -> float:
        """Calculate Eq.(23) in Gonçalvès et al. (2007)

        Args:
            phi0 (float): Surface potential (V)

        Returns:
            float: Value of "B" in Eq.(23)
        """
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        t = self.temperature
        ch = self.ion_props[Species.H.name][IonProp.Activity.name]
        k2 = self.k2
        return 1.0 + ch / k2 * np.exp(-_e * phi0 / (kb * t))

    def __calc_C(self, phib: float) -> float:
        """Calculate Eq.(24) in Gonçalvès et al. (2007)

        Args:
            phib (float): Stern plane potential (V)

        Returns:
            float: Value of "C" in Eq.(23)
        """
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        t = self.temperature
        ch = self.ion_props[Species.H.name][IonProp.Activity.name]
        cna = self.ion_props[Species.Na.name][IonProp.Activity.name]
        k3 = self.k3
        k4 = self.k4
        return 1.0 + (ch / k3 + cna / k4) * np.exp(-_e * phib / (kb * t))

    def __calc_functions_inf(self, xn: np.ndarray) -> np.ndarray:
        """Calculate functions 1 to 6 for infinite (i.e., not truncated)
        diffuse layer case

        Args:
            xn (np.ndarray): 2d array of electrochemical properties (column vector).
            xn[0][0]: Potential of the 0-plane
            xn[1][0]: Potential of the Stern plane
            xn[2][0]: Potential of the shear plane
            xn[3][0]: Charge density on the surface
            xn[4][0]: Charge density of the Stern layer
            xn[5][0]: Charge density of the diffuse layer

        Returns:
            np.ndarray: 2d array containing the residuals of each function
        """
        f1 = self.__calc_f1(xn[0][0], xn[3][0])
        f2 = self.__calc_f2(xn[1][0], xn[4][0])
        f3 = self.__calc_f3(xn[3][0], xn[4][0], xn[5][0])
        f4 = self.__calc_f4(xn[0][0], xn[1][0], xn[3][0])
        f5 = self.__calc_f5(xn[1][0], xn[2][0], xn[5][0])
        f6 = self.__calc_f6(xn[2][0], xn[5][0])
        fn = np.array([f1, f2, f3, f4, f5, f6]).reshape(-1, 1)
        return fn

    def __calc_functions_truncated(self, xn: np.ndarray) -> np.ndarray:
        """Calculate functions 1 to 7 for truncated diffuse layer case

        Args:
            xn (np.ndarray): 2d array of electrochemical properties (column vector).
            xn[0][0]: Potential of the 0-plane
            xn[1][0]: Potential of the Stern plane
            xn[2][0]: Potential of the shear plane
            xn[3][0]: Potential of truncated plane
            xn[4][0]: Charge density of the surface
            xn[5][0]: Charge density of the Stern layer
            xn[6][0]: Charge density of the diffuse layer

        Returns:
            np.ndarray: 2d array containing residuals of each function
        """
        f1 = self.__calc_f1(xn[0][0], xn[4][0])
        f2 = self.__calc_f2(xn[1][0], xn[5][0])
        f3 = self.__calc_f3(xn[4][0], xn[5][0], xn[6][0])
        f4 = self.__calc_f4(xn[0][0], xn[1][0], xn[4][0])
        f5 = self.__calc_f5(xn[1][0], xn[2][0], xn[6][0])
        f6 = self.__calc_f6_truncated(xn[2][0], xn[3][0], xn[6][0])
        f7 = self.__calc_f7_truncated(xn[2][0], xn[3][0])
        fn = np.array([f1, f2, f3, f4, f5, f6, f7]).reshape(-1, 1)
        return fn

    def __calc_qs_inf(self, _x: float) -> float:
        # TODO: should remove?
        """Calculate the amount of charge (C/m3) at _x distance from the
        surface in the case of infinite diffuse layer development

        Args:
            _x (float): Distance from the surface to pore

        Returns:
            float: Charge density at _x distance from the surface (C/m3)
        """
        assert self.__check_if_calculated_qs_coeff(), (
            "Before calculating Qs, we should obtain"
            "the coefficients for calculating Qs"
        )
        return self.qs_coeff1_inf * np.sinh(
            self.qs_coeff2_inf * np.exp((-1.0) * self.kappa * _x)
        )

    def __calc_qs_coeff_inf(self) -> None:
        # TODO: should remove?
        """Amount of charge in an infinitely developing diffuse layer."""
        assert self._check_if_calculated_electrical_params_inf(), (
            "Before calculating xd, we should obtain electrical"
            "arameters for infinite diffuse layer case"
        )
        _e = const.ELEMENTARY_CHARGE
        # electrolyte concentration is assumed to be equal to Na+ concentration
        self.qs_coeff1_inf = (
            2000.0
            * _e
            * const.AVOGADRO_CONST
            * self.ion_props[Species.Na.name][IonProp.Molarity.name]
        )
        self.qs_coeff2_inf = (
            -_e * self.potential_zeta_o / (const.BOLTZMANN_CONST * self.temperature)
        )

    def __check_if_calculated_qs_coeff(self) -> bool:
        """Return true if the coefficients for calculating Qs
        have already been calculated.

        Returns:
            bool: True if already calculated
        """
        flag = True
        if self.qs_coeff1_inf is None:
            flag = False
        if self.qs_coeff2_inf is None:
            flag = False
        return flag

    def calc_xd(self) -> float:
        """Calculate the distance from the surface to the shear plane
        based on the MD simulation results by Zhang et al.(2014)

        Returns:
            float: xd (m)
        """
        xd: float = None
        rintp = self.layer_width * 1.0e10
        r_ls = (9.4, 12.3, 15.2, 18.4, 21.6)
        xd_ls = (
            2.0845481049562675,
            2.4187499999999993,
            4.206896551724138,
            4.182509505703424,
            4.220116618075801,
        )
        xdintp: float = None
        if rintp < r_ls[0]:
            xdintp = xd_ls[0]
        elif rintp > r_ls[-1]:
            xdintp = xd_ls[-1]
        else:
            for i, (r, xd) in enumerate(zip(r_ls, xd_ls)):
                if r <= rintp <= r_ls[i + 1]:
                    dr = rintp - r
                    xdintp = xd + dr * ((xd_ls[i + 1] - xd) / (r_ls[i + 1] - r))
                    break
        self.xd = xdintp * 1.0e-10
        return self.xd

    def calc_potentials_and_charges_inf(
        self,
        x_init: List = None,
        iter_max: int = 1000,
        convergence_condition: float = 1.0e-10,
        oscillation_tol: float = 1.0e-04,
        beta: float = 0.75,
        lamda: float = 2.0,
    ) -> List:
        """Calculate the electrochemical properties for the case of infinite
        (i.e., not truncated) diffuse layer.
        Eqs.(16)–(21) in Gonçalvès et al. (2007) are solved by the damped
        Newton-Raphson method

        x_init (List): Initial value of electrochemical properties (length is 6).
            Need to place the electrochemical properties in the same order as
            the return value.
        iter_max (int): Maximum number of iterations of the Newton-Raphson method.
        convergence_condition (float): Convergence conditions (the L2 norm of the
            equations) for the Newton-Raphson method.
        oscillation_tol (float): Oscillation tolerance.
        beta (float): Hyperparameter for damping((0, 1)). The larger this value,
            the smaller the damping effect.
        lamda (float): Hyperparameter for damping([1, ∞]); i.e., the amount
            by which the step-width coefficients are updated in a single iteration.

        Raises:
            RuntimeError: Occurs if the loop count exceeds m_iter_max and
             norm of step exceeds oscillation_tol

        Returns:
            List: Containing the following electrochemical properties:
              [Surface potential (V),
               Stern plane potential (V),
               Zeta potential (V),
               Charge density on surface (C/m^2)
               Charge density of the Stern layer (C/m^2),
               Charge density of the diffuse layer (C/m^2)]
        """
        assert 0.0 < beta < 1.0
        assert lamda > 1.0
        assert self.dielec_fluid is not None
        assert self.viscosity is not None

        # set constants
        self.gamma_1 = self.tlm_params.gamma_1o
        self.gamma_2 = self.tlm_params.gamma_2o
        self.gamma_3 = self.tlm_params.gamma_3o
        self.qi = self.tlm_params.qio
        self.k1 = self.tlm_params.k1o
        self.k2 = self.tlm_params.k2o
        self.k3 = self.tlm_params.k3o
        self.k4 = self.tlm_params.k4o
        self.c1 = self.tlm_params.c1o
        self.c2 = self.tlm_params.c2o

        if x_init is None:
            # Set initial electrical parameters
            if isinstance(self, Smectite):
                # for smectite case
                params = smectite_inf_init_params
            elif isinstance(self, Kaolinite):
                # for kaolinite case
                params = kaolinite_init_params
            else:
                # TODO: Prepare more initial parameters in other minerals.
                params = kaolinite_init_params
            ch = self.ion_props[Species.H.name][IonProp.Molarity.name]
            ch_ls = list(params.keys())
            _idx_ch: int = np.argmin(np.square(np.array(ch_ls, dtype=np.float64) - ch))
            cna = self.ion_props[Species.Na.name][IonProp.Molarity.name]
            cna_dct: Dict = params[ch_ls[_idx_ch]]
            cna_ls = list(cna_dct.keys())
            _idx_cna: int = np.argmin(
                np.square(np.array(cna_ls, dtype=np.float64) - cna)
            )
            x_init = cna_dct[cna_ls[_idx_cna]]

        xn = np.array(x_init, np.float64).reshape(-1, 1)
        fn = self.__calc_functions_inf(xn)
        norm_fn: float = np.sum(np.sqrt(np.square(fn)), axis=0)[0]
        cou = 0
        while convergence_condition < norm_fn:
            _j = self._calc_Goncalves_jacobian(xn[0][0], xn[1][0], xn[2][0])
            # avoid overflow when calculating the inverse matrix
            _j = _j + float_info.min
            _j_inv = np.linalg.inv(_j)
            step = np.matmul(_j_inv, fn)
            # Damping is applied. References:
            # [1] http://www.misojiro.t.u-tokyo.ac.jp/~murota/lect-suchi/newton130805.pdf
            # [2] http://www.ep.sci.hokudai.ac.jp/~gfdlab/comptech/y2016/resume/070_newton/2014_0626-mkuriki.pdf
            _cou_damp: int = 0
            _norm_fn_tmp, _rhs = float_info.max, float_info.min
            # enter the following loop at least once
            while _norm_fn_tmp > _rhs:
                # update μ
                _mu = 1.0 / (lamda**_cou_damp)
                # calculate left hand side of Eq.(21) of [1]
                xn_tmp: np.ndarray = xn - _mu * step
                fn_tmp = self.__calc_functions_inf(xn_tmp)
                _norm_fn_tmp = np.sum(np.sqrt(np.square(fn_tmp)), axis=0)[0]
                # calculate right hand side of Eq.(21) of [1]
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
        xn = xn.T.tolist()[0]
        self.potential_0_o = xn[0]
        self.potential_stern_o = xn[1]
        self.potential_zeta_o = xn[2]
        self.charge_0_o = xn[3]
        self.charge_stern_o = xn[4]
        self.charge_diffuse_o = xn[5]
        # DEBUG
        if self.logger is not None:
            self.logger.info(
                "Finished the calculation of electrical "
                "properties for an infinite diffusion layer"
            )
            self.logger.debug(f"potential_0_o: {self.potential_0_o}")
            self.logger.debug(f"potential_stern_o: {self.potential_stern_o}")
            self.logger.debug(f"potential_zeta_o: {self.potential_zeta_o}")
            self.logger.debug(f"charge_0_o: {self.charge_0_o}")
            self.logger.debug(f"charge_stern_o: {self.charge_stern_o}")
            self.logger.debug(f"charge_diffuse_o: {self.charge_diffuse_o}")

        # calculates the distance from the surface to shear plane
        self.calc_xd()
        return xn

    def calc_potentials_and_charges_truncated(
        self,
        x_init: List = None,
        iter_max: int = 1000,
        convergence_condition: float = 1.0e-6,
        step_tol: float = 1.0e-10,
        oscillation_tol: float = 1.0e-04,
        beta: float = 0.75,
        lamda: float = 2.0,
    ) -> List:
        """Calculate the electrochemical properties for the case of
        truncated diffuse layer development.
        Eqs.(16)–(20), (32), and (34) in Gonçalvès et al. (2007) are solved
        by the damped Newton-Raphson method.

        x_init (List): Initial value of electrochemical properties (length is 7).
            Need to place the electrochemical properties in the same order as
            the return value.
        iter_max (int): Maximum number of iterations of the Newton-Raphson method.
        convergence_condition (float): Convergence conditions (the L2 norm of the
            equations) for the Newton-Raphson method.
        oscillation_tol (float): Oscillation tolerance.
        beta (float): Hyper parameter for damping((0, 1)). The larger this value,
            the smaller the damping effect.
        lamda (float): Hyperparameter for damping([1, ∞]); i.e., the amount
            by which the step-width coefficients are updated in a single iteration.

        Raises:
            RuntimeError: Occurs if the loop count exceeds m_iter_max and
             norm of step exceeds oscillation_tol

        Returns:
            List: Containing the following electrochemical properties:
              [Surface potential (V),
               Stern plane potential (V),
               Zeta potential (V),
               Truncated plane potential (V),
               Charge density on surface (C/m^2)
               Charge density of the Stern layer (C/m^2),
               Charge density of the diffuse layer (C/m^2)]
        """
        # assert self.layer_width >= 0.9e-9, "self.layer_width < 0.9e-9"
        assert 0.0 < beta < 1.0
        assert lamda > 1.0
        assert self.dielec_fluid is not None
        assert self.viscosity is not None

        # obtain init values based on infinity developed diffuse layer
        # phi0, phib, phid, phir, q0, qb, qs
        if not self._check_if_calculated_electrical_params_inf():
            self.calc_potentials_and_charges_inf()
        if self.xd is None:
            self.calc_xd()
        if x_init is None:
            # layer width
            r_ls = list(smectite_trun_init_params.keys())
            _r = self.layer_width
            _idx = np.argmin(np.square((np.array(r_ls, dtype=np.float64) - _r)))
            t_ch_cna_dict: Dict = smectite_trun_init_params[r_ls[_idx]]
            # temperature
            t_ls = list(t_ch_cna_dict.keys())
            _idx = np.argmin(np.square(np.array(t_ls) - self.temperature))
            ch_cna_dct: Dict = t_ch_cna_dict[t_ls[_idx]]
            # pH
            _ch = self.ion_props[Species.H.name][IonProp.Molarity.name]
            _cna = self.ion_props[Species.Na.name][IonProp.Molarity.name]
            ch_ls = list(ch_cna_dct.keys())
            _idx = np.argmin(np.square(np.log10(ch_ls) - np.log10(_ch)))
            cna_dct: Dict = ch_cna_dct[ch_ls[_idx]]
            # sodium concentration
            cna_ls = list(cna_dct.keys())
            _idx = np.argmin(
                np.square((np.log10(cna_ls, dtype=np.float64) - np.log10(_cna)))
            )
            x_init = cna_dct[cna_ls[_idx]]

        # set constants
        self.gamma_1 = self.tlm_params.gamma_1i
        self.gamma_2 = self.tlm_params.gamma_2i
        self.gamma_3 = self.tlm_params.gamma_3i
        self.qi = self.tlm_params.qii
        self.k1 = self.tlm_params.k1i
        self.k2 = self.tlm_params.k2i
        self.k3 = self.tlm_params.k3i
        self.k4 = self.tlm_params.k4i
        self.c1 = self.tlm_params.c1i
        self.c2 = self.tlm_params.c2i

        xn = np.array(x_init, np.float64).reshape(-1, 1)
        fn = self.__calc_functions_truncated(xn)
        norm_fn: float = np.sum(np.sqrt(np.square(fn)), axis=0)[0]
        cou = 0
        is_norm_converged: bool = False
        while convergence_condition < norm_fn:
            _j = self._calc_Goncalves_jacobian_truncated(
                xn[0][0], xn[1][0], xn[2][0], xn[3][0]
            )
            # avoid overflow when calculating the inverse matrix
            _j = _j + float_info.min
            _j_inv = np.linalg.inv(_j)
            step = np.matmul(_j_inv, fn)
            # Damping is applied. References:
            # [1] http://www.misojiro.t.u-tokyo.ac.jp/~murota/lect-suchi/newton130805.pdf
            # [2] http://www.ep.sci.hokudai.ac.jp/~gfdlab/comptech/y2016/resume/070_newton/2014_0626-mkuriki.pdf
            _cou_damp: int = 0
            _norm_fn_tmp, _rhs = float_info.max, float_info.min
            # enter the following loop at least once
            while _norm_fn_tmp > _rhs:
                # update μ
                _mu = 1.0 / (lamda**_cou_damp)
                # calculate left hand side of Eq.(21) of [1]
                xn_tmp: np.ndarray = xn - _mu * step
                fn_tmp = self.__calc_functions_truncated(xn_tmp)
                _norm_fn_tmp = np.sum(np.sqrt(np.square(fn_tmp)), axis=0)[0]
                # calculate right hand side of Eq.(21) of [1]
                _rhs = (1.0 - (1.0 - beta) * _mu) * norm_fn
                _cou_damp += 1
                if norm_fn < convergence_condition:
                    break
                if _cou_damp > 10000:
                    break
            xn = xn_tmp
            fn = fn_tmp
            norm_fn = _norm_fn_tmp
            cou += 1
            if cou > iter_max:
                norm_step = np.sum(np.sqrt(np.square(step)), axis=0)[0]
                if step_tol < norm_step:
                    break
                if norm_step < oscillation_tol:
                    break
                else:
                    _msg: str = (
                        f"Loop count exceeded {iter_max} times &"
                        f" exceeds oscillation tolerance: {norm_step}"
                    )
                    if self.logger is not None:
                        self.logger.error(_msg)
                    raise RuntimeError(_msg)
        if norm_fn < convergence_condition:
            is_norm_converged = True
        xn = xn.T.tolist()[0]
        # assign member variables
        self.potential_0_i = xn[0]
        self.potential_stern_i = xn[1]
        self.potential_zeta_i = xn[2]
        self.potential_r_i = xn[3]
        self.charge_0_i = xn[4]
        self.charge_stern_i = xn[5]
        self.charge_diffuse_i = xn[6]
        # fix minor error
        # zeta potential
        if math.isclose(self.potential_zeta_i, self.potential_r_i):
            self.potential_r_i = self.potential_zeta_i
        # make zeta potential and truncated plane potential the same sign
        elif self.potential_zeta_i <= 0.0:
            if self.potential_r_i > 0.0:
                self.potential_r_i -= 2.0 * self.potential_r_i
        elif self.potential_zeta_i > 0.0:
            if self.potential_r_i < 0.0:
                self.potential_r_i -= 2.0 * self.potential_r_i

        # DEBUG
        if self.logger is not None:
            self.logger.info(
                "Finished the calculation of electrical "
                "properties for an truncated diffusion layer"
            )
            self.logger.debug(f"potential_0_i: {self.potential_0_i}")
            self.logger.debug(f"potential_stern_i: {self.potential_stern_i}")
            self.logger.debug(f"potential_zeta_i: {self.potential_zeta_i}")
            self.logger.debug(f"potential_r_i: {self.potential_r_i}")
            self.logger.debug(f"charge_0_i: {self.charge_0_i}")
            self.logger.debug(f"charge_stern_i: {self.charge_stern_i}")
            self.logger.debug(f"charge_diffuse_i: {self.charge_diffuse_i}")

        return xn, is_norm_converged

    def _check_if_calculated_electrical_params_inf(self) -> bool:
        """Check if the electrochemical properties of the infinite
        (i.e., not truncated) diffuse layer has already been calculated.

        Returns:
            bool: True if already calculated
        """
        flag = True
        if self.potential_0_o is None:
            flag = False
        if self.potential_stern_o is None:
            flag = False
        if self.potential_zeta_o is None:
            flag = False
        if self.charge_0_o is None:
            flag = False
        if self.charge_stern_o is None:
            flag = False
        if self.charge_diffuse_o is None:
            flag = False
        return flag

    def _check_if_calculated_electrical_params_truncated(self) -> bool:
        """Check if the electrochemical properties for the truncated diffuse layer
        has already been calculated.

        Returns:
            bool: True if already calculated the electrochemical properties
                for the truncated diffuse layer
        """
        flag = True
        if self.potential_0_i is None:
            flag = False
        if self.potential_stern_i is None:
            flag = False
        if self.potential_zeta_i is None:
            flag = False
        if self.potential_r_i is None:
            flag = False
        if self.charge_0_i is None:
            flag = False
        if self.charge_stern_i is None:
            flag = False
        if self.charge_diffuse_i is None:
            flag = False
        if self.xd is None:
            flag = False
        return flag

    def __calc_kappa_truncated(self) -> None:
        """In the truncated diffuse layer, calculate the damping parameter
        κ instead of Eq.(11) in Gonçalvès et al. (2007).
        """
        _r = self.layer_width * 0.5
        self.kappa_truncated = (
            1.0 / _r * np.log(self.potential_zeta_i / self.potential_r_i)
        )

    def __calc_cond_diffuse_inf(self, x: float, s: str, coeff: float) -> float:
        """Calculate the electrical conductivity of the infinite
        (i.e., not truncated) diffuse layer normalized by the
        elementary charge.

        Args:
            x (float): Distance from the shear plane (m)
            s (str): Ion species name
            coeff (float): Ratio of dielectric constant to viscosity
                (Eq.[26] in Leroy et al., 2015)

        Returns:
            float: Electrical conductivity normalized by the
                elementary charge (S/(m・C))
        """
        # calculate number density
        potential: float = self.potential_zeta_o * np.exp((-1.0) * self.kappa * x)
        _props: Dict = self.ion_props[s]
        v = _props[IonProp.Valence.name]
        # ion mobility at position x
        bx = _props[IonProp.Mobility.name] + (v / abs(v)) * coeff * (
            potential - self.potential_zeta_o
        )
        # number density of the ion species "s"
        n = (
            np.exp(
                -v
                * const.ELEMENTARY_CHARGE
                * potential
                / (const.BOLTZMANN_CONST * self.temperature)
            )
            * 1000.0
            * const.AVOGADRO_CONST
            * _props[IonProp.Molarity.name]
            * abs(v)
        )
        return bx * n

    def __calc_cond_diffuse_truncated(self, _x: float) -> float:
        """Calculate the electrical conductivity of the truncated diffuse
        layer normalized by the elementary charge.

        Args:
            _x (float): Distance from the shear plane (m)
        Returns:
            float: Electrical conductivity normalized by the
                elementary charge (S/(m・C))
        """
        potential: float = self.potential_zeta_i * np.exp(
            (-1.0) * self.kappa_truncated * _x
        )
        _cond = 0.0
        for _s, _prop in self.ion_props.items():
            if _s in (Species.H.name, Species.OH.name):
                continue
            v = _prop[IonProp.Valence.name]
            n = (
                np.exp(
                    -v
                    * const.ELEMENTARY_CHARGE
                    * potential
                    / (const.BOLTZMANN_CONST * self.temperature)
                )
                * 1000.0
                * const.AVOGADRO_CONST
                * _prop[IonProp.Molarity.name]
                * abs(v)
            )
            beta = self.__calc_mobility_diffuse(_x + self.xd, _s)
            _cond += n * beta
        return _cond

    def __calc_n_stern(self, orientation: str) -> float:
        """Calculate Na+ number density of the Stern layer
        by Eq.(12) in Leroy & Revil (2004).

        Args:
            orientation (str): Strings indicating where the Stern
                layer exists. Set "outer" for the outer surface and
                "inner" for the inner surface, i.e., interlayer.

        Returns:
            float: Number density of Na+ (m^-2)
        """
        phib = None
        if orientation == "outer":
            phib = self.potential_stern_o
        elif orientation == "inner":
            phib = self.potential_stern_i
        assert phib is not None, phib
        n: float = (
            self.gamma_3
            * 1.0e18
            / self.__calc_C(phib)
            * (self.ion_props[Species.Na.name][IonProp.Activity.name])
            / self.k4
            * np.exp(
                -const.ELEMENTARY_CHARGE
                * phib
                / (const.BOLTZMANN_CONST * self.temperature)
            )
        )
        return n

    def calc_cond_interlayer(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate the electrical conductivity of the Stern + difusse layer
        for the interlayer.

        Returns:
            Tuple[float, Tuple[float, float]]: Electrical conductivity of
                the interlayer (S/m), and the tuple containing conductivity of
                the Stern layer and the diffuse layer (both S/m).
        """
        if self.xd is None:
            self.calc_xd()
        if not self._check_if_calculated_electrical_params_truncated():
            self.calc_potentials_and_charges_truncated()
        if self.kappa_truncated is None:
            self.__calc_kappa_truncated()

        # Na+ number density (m^-2) of the Stern layer
        gamma_stern = self.__calc_n_stern("inner")
        _xdl = self.layer_width * 0.5
        # Na+ number density (m^-2) of the diffuse layer
        cond_diffuse = 0.0
        if not math.isclose(self.xd, _xdl):
            cond_diffuse, _ = quad(self.__calc_cond_diffuse_truncated, self.xd, _xdl)

        # total number density
        cond_stern = gamma_stern * self.mobility_stern
        cond_intra: float = (
            const.ELEMENTARY_CHARGE * (cond_stern + cond_diffuse)
        ) / _xdl

        # DEBUG
        if self.logger is not None:
            self.logger.debug(f"cond_intra: {cond_intra}")

        self.cond_intra = cond_intra
        return self.cond_intra, (cond_stern, cond_diffuse)

    def calc_cond_infdiffuse(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate the electrical conductivity of the Stern + diffuse layer
        for the inifinite (i.e., not truncated) diffuse layer.

        Returns:
            Tuple[float, Tuple[float, float]]: Electrical conductivity of
                the EDL (S/m), and the tuple containing conductivity of
                the Stern layer and the diffuse layer (both S/m).
        """
        if not self._check_if_calculated_electrical_params_inf():
            self.calc_potentials_and_charges_inf()
        if self.xd is None:
            self.calc_xd()

        # Na+ number density (m^-2) of the Stern layer
        gamma_stern = self.__calc_n_stern("outer")
        cond_stern = gamma_stern * self.mobility_stern
        # Na+ number density (m^-2) of the diffuse layer
        xdl = 1.0 / self.kappa
        coeff = self.dielec_fluid / self.viscosity
        __callback = partial(
            self.__calc_cond_diffuse_inf,
            s=Species.Na.name,
            coeff=coeff,
        )
        cond_na_diffuse, _ = quad(__callback, 0.0, xdl)

        # Cl- number density (m^-2) of the diffuse layer
        __callback = partial(
            self.__calc_cond_diffuse_inf,
            s=Species.Cl.name,
            coeff=coeff,
        )
        cond_cl_diffuse, _ = quad(__callback, 0.0, xdl)

        # calculate conductivity
        cond_diffuse: float = (
            const.ELEMENTARY_CHARGE * (cond_stern + cond_na_diffuse + cond_cl_diffuse)
        ) / xdl

        # DEBUG
        if self.logger is not None:
            self.logger.debug(f"cond_diffuse: {cond_diffuse}")

        self.cond_infdiffuse = cond_diffuse
        self.double_layer_length = xdl

        return self.cond_infdiffuse, (cond_stern, cond_na_diffuse)

    def calc_cond_tensor(self) -> None:
        """Calculate conductivity tensor."""
        cond_silica = 1.0e-12
        cond_tensor = np.array(
            [[cond_silica, 0.0, 0.0], [0.0, cond_silica, 0.0], [0.0, 0.0, cond_silica]],
            dtype=np.float64,
        )
        if self.logger is not None:
            self.logger.info(f"cond_tensor: {self.cond_tensor}")
        self.cond_tensor = cond_tensor
        return cond_tensor    

    def __calc_na_density_at_x(self, x: float) -> float:
        """Calculate Na+ number density at distance x from the shear plane.

        Args:
            x (float): Distance from the shear plane

        Returns:
            float: Na+ number density (m^-3)
        """
        phix = self.potential_zeta_i * np.exp(-self.kappa_truncated * x)
        return self.ion_props[Species.Na.name][IonProp.Molarity.name] * np.exp(
            -const.ELEMENTARY_CHARGE * phix / (const.BOLTZMANN_CONST * self.temperature)
        )

    def calc_partition_coefficient(self) -> float:
        """Calculate the ratio of the number density of Na+ in the Stern layer
        to the total number of the EDL (i.e., partition coefficient).

        Returns:
            float: Partition coefficient (-)
        """
        if self.xd is None:
            self.calc_xd()
        if self.kappa_truncated is None:
            self.__calc_kappa_truncated()
        _xdl = self.layer_width * 0.5
        gamma_na_diffuse = (
            1000.0
            * const.AVOGADRO_CONST
            * quad(self.__calc_na_density_at_x, self.xd, _xdl)[0]
        )
        gamma_na_stern = self.__calc_n_stern("inner")
        self.partition_coefficient = gamma_na_stern / (
            gamma_na_stern + gamma_na_diffuse
        )
        return self.partition_coefficient

    def calc_osmotic_pressure(self) -> float:
        """Calculate the osmotic pressure of the truncated EDL

        Returns:
            float: Osmotic pressure (Pa)
        """
        assert isinstance(self.potential_r_i, float), self.potential_r_i
        kb = const.BOLTZMANN_CONST
        T = self.temperature
        e = const.ELEMENTARY_CHARGE
        Na = const.AVOGADRO_CONST
        Cf = self.ion_props[Species.Na.name][IonProp.Molarity.name]
        phir = self.potential_r_i
        self.osmotic_pressure = (
            2000.0 * kb * T * Na * Cf * (np.cosh(e * phir / (kb * T)) - 1.0)
        )
        return self.osmotic_pressure

    def set_cond_tensor(self, cond_tensor: np.ndarray) -> None:
        """Setter of the electrical conductivity tensor

        Args:
            cond_tensor (np.ndarray): Electrical conductivity tensor (3×3; S/m)
        """
        self.cond_tensor = cond_tensor

    def set_cond_surface(self, cond_surface: float) -> None:
        """Setter of the electrical conductivity of the EDL faced to
        the bulk liquid

        Args:
            cond_surface (float): Electrical conductivity of the EDL faced
                to the bulk liquid (S/m)
        """
        self.cond_infdiffuse = cond_surface

    def set_double_layer_length(self, double_layer_length: float) -> None:
        """Setter of the user defined length of the EDL face to the bulk liquid

        Args:
            double_layer_length (float): User defined length of the EDL (m)
        """
        self.double_layer_length = double_layer_length

    def get_logger(self) -> Logger:
        """Getter for the Logger

        Returns:
            Logger: Containing debugging information, etc.
        """
        return self.logger

    def get_cond_tensor(self) -> Union[np.ndarray, None]:
        """Getter for the conductivity tensor

        Returns:
            np.ndarray: Electrical conductivity tensor (3×3; S/m)
        """
        if self.cond_tensor is not None:
            return deepcopy(self.cond_tensor)
        return self.cond_tensor

    def get_cond_surface(self) -> Union[float, None]:
        """Getter for the Stern plane potential

        Returns:
            float: Electrical conductivity of the Stern layer (S/m)
        """
        return self.cond_infdiffuse

    def get_double_layer_length(self) -> Union[float, None]:
        """Getter for the EDL thickness (the Debye length by default).

        Returns:
            float or None: EDL thickness (m)
        """
        return self.double_layer_length

    def save(self, _pth: str) -> None:
        """Save phyllosilicate class as pickle object.

        Args:
            _pth (str): path to save
        """
        with open(_pth, "wb") as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)


# pylint: disable=dangerous-default-value
class Smectite(Phyllosilicate):
    """Subclass of Phyllosilicate, with TLM parameters are fixed to the
    smectite's properties.

    Args:
        Phyllosilicate: Phyllosilicate class
    """

    def __init__(
        self,
        nacl: NaCl,
        layer_width: float = 1.52e-9,
        tlm_params: TLMParams = None,
        potential_0_o: float = None,
        potential_stern_o: float = None,
        potential_zeta_o: float = None,
        charge_0_o: float = None,
        charge_stern_o: float = None,
        charge_diffuse_o: float = None,
        potential_0_i: float = None,
        potential_stern_i: float = None,
        potential_zeta_i: float = None,
        potential_r_i: float = None,
        charge_0_i: float = None,
        charge_stern_i: float = None,
        charge_diffuse_i: float = None,
        xd: float = None,
        cond_intra: float = None,
        cond_infdiffuse: float = None,
        logger: Logger = None,
    ):
        """Set TLM parameters for representing the electrochemical properties of smectite.

        Args:
            nacl (NaCl): Instance of NaCl class
            layer_width (float): Thickness of the interlayer.
            tlm_params (TLMParams): TLM parameter.
            potential_0_o (float): Surface potential of the outer surface (V)
            potential_stern_o (float): Stern potential of the outer surface (V)
            potential_zeta_o (float): Zeta potential of the outer surface (V)
            charge_0_o (float): Charge density of the outer surface (C/m^2)
            charge_stern_o (float): CCharge density of the Stern layer on the
                outer surface (C/m^2)
            charge_diffuse_o (float): Charge density of the diffuse layer on
                the outer surface (C/m^2)
            potential_0_i (float): Surface potential of the inner surface (V)
            potential_stern_i (float): Stern potential of the inner surface (V)
            potential_zeta_i (float): Zeta potential of the inner surface (V)
            potential_r_i (float): Potential at the midpoint of the interlayer
                (i.e., truncation point) (V)
            charge_0_i (float): Charge density of the inner surface (C/m^2)
            charge_stern_i (float): Charge density of the Stern layer in
                the inner surafce (C/m^2)
            charge_diffuse_i (float): Charge density of the diffuse layer
                in the innner surface (C/m3)
            xd (float): Distance from the surface (i.e., the center of the oxygen
                in the siloxane surface) to shear plane (V)
            cond_intra (float): Inter layer conductivity (S/m)
            cond_infdiffuse (float): Conductivity of the outer surface (S/m)
            logger (Logger): Logger for debugging
        """
        # default parameters
        if tlm_params is None:
            tlm_params = TLMParams(
                T=nacl.get_temperature(),
                gamma_1i=0.0,
                gamma_2i=5.5,
                gamma_3i=5.5,
                qii=-1.0,
                k1i=1.0e-10,
                k2i=1.3e-6,
                k3i=1.0e-2,
                k4i=0.95,
                c1i=2.1,
                c2i=0.55,
                gamma_1o=0.0,
                gamma_2o=5.5,
                gamma_3o=5.5,
                qio=-2.747271878506929,
                k1o=1.0e-10,
                k2o=1.3e-6,
                k3o=1.0e-2,
                k4o=0.49829061,
                c1o=0.93188509,
                c2o=2.56335737,
            )
        super().__init__(
            nacl=nacl,
            layer_width=layer_width,
            tlm_params=tlm_params,
            potential_0_o=potential_0_o,
            potential_stern_o=potential_stern_o,
            potential_zeta_o=potential_zeta_o,
            charge_0_o=charge_0_o,
            charge_stern_o=charge_stern_o,
            charge_diffuse_o=charge_diffuse_o,
            potential_0_i=potential_0_i,
            potential_stern_i=potential_stern_i,
            potential_zeta_i=potential_zeta_i,
            potential_r_i=potential_r_i,
            charge_0_i=charge_0_i,
            charge_stern_i=charge_stern_i,
            charge_diffuse_i=charge_diffuse_i,
            xd=xd,
            cond_intra=cond_intra,
            cond_infdiffuse=cond_infdiffuse,
            logger=logger,
        )
        
    def calc_cond_tensor(self) -> np.ndarray:
        """Calculate the electrical conductivity tensor of smectite with layers
        aligned parallel to the xy-plane. The TOT plane is assumed to be an
        insulator (e.g., Watanabe [2005]).

        Returns:
            np.ndarray: Electrical conductivity tensor (3×3; S/m)
        """
        assert self._check_if_calculated_electrical_params_truncated, (
            "Before calculating the conductivity tensor of smectite, you should"
            "calculate the electrochemical properties of the truncated diffuse layer"
        )
        if self.cond_intra is None:
            self.calc_cond_interlayer()
        sigma_intra = self.cond_intra
        # conductivity and width of the TOT layer
        ctot, dtot = 1.0e-12, 6.6e-10
        sigma_h = (sigma_intra * self.layer_width + ctot * dtot) / (
            dtot + self.layer_width
        )
        sigma_v = (self.layer_width + dtot) / (
            self.layer_width / sigma_intra + dtot / ctot
        )
        cond_tensor = np.array(
            [[sigma_h, 0.0, 0.0], [0.0, sigma_h, 0.0], [0.0, 0.0, sigma_v]],
            dtype=np.float64,
        )
        self.cond_tensor = cond_tensor
        return cond_tensor


# pylint: disable=dangerous-default-value
class Kaolinite(Phyllosilicate):
    """Set TLM parameters for representing the electrochemical properties of smectite.

    Args:
        Phyllosilicate: Phyllosilicate class
    """

    def __init__(
        self,
        nacl: NaCl,
        tlm_params: TLMParams = None,
        potential_0_o: float = None,
        potential_stern_o: float = None,
        potential_zeta_o: float = None,
        charge_0_o: float = None,
        charge_stern_o: float = None,
        charge_diffuse_o: float = None,
        cond_infdiffuse: float = None,
        logger: Logger = None,
    ):
        """Inherited class from Phyllosilicate. Number density of
            reactors on the surface and fixing the layer charge for
            kaolinite case.

        Args:
            nacl (NaCl): Instance of NaCl class
            tlm_params (TLMParams): TLM parameter
            potential_0_o (float): Surface potential of the outer surface (V)
            potential_stern_o (float): Stern potential of the outer surface (V)
            potential_zeta_o (float): Zeta potential of the outer surface (V)
            charge_0_o (float): Charge density of the outer surface (C/m^2)
            charge_stern_o (float): CCharge density of the Stern layer on the
                outer surface (C/m^2)
            charge_diffuse_o (float): Charge density of the diffuse layer on
                the outer surface (C/m^2)
            cond_infdiffuse (float): Conductivity of the outer surface (S/m)
            logger (Logger): Logger for debugging.
        """

        if tlm_params is None:
            # TODO: validate this parameter
            # Set parameters are based on Leroy & Revil (2004)
            tlm_params = TLMParams(
                T=nacl.get_temperature(),
                gamma_1i=None,
                gamma_2i=None,
                gamma_3i=None,
                qii=None,
                k1i=None,
                k2i=None,
                k3i=None,
                k4i=None,
                c1i=None,
                c2i=None,
                gamma_1o=5.5,
                gamma_2o=5.5,
                gamma_3o=5.5,
                qio=0.0,
                k1o=1.0e-10,
                k2o=4.95e-6,
                k3o=1.0e-2,
                k4o=5.04e-2,
                c1o=1.49,
                c2o=0.2,
            )
        super().__init__(
            nacl=nacl,
            layer_width=1.0e-10,
            tlm_params=tlm_params,
            potential_0_o=potential_0_o,
            potential_stern_o=potential_stern_o,
            potential_zeta_o=potential_zeta_o,
            charge_0_o=charge_0_o,
            charge_stern_o=charge_stern_o,
            charge_diffuse_o=charge_diffuse_o,
            cond_infdiffuse=cond_infdiffuse,
            logger=logger,
        )

    def calc_cond_tensor(self) -> np.ndarray:
        """Calculate conductivity tensor of kaolinite.

        Args:
            edge_length (float): Lengths of the edges of the cube's cells

        Returns:
            np.ndarray: 3 rows and 3 columns condutivity tensor
        """
        cond_silica = 1.0e-12
        cond_tensor = np.array(
            [[cond_silica, 0.0, 0.0], [0.0, cond_silica, 0.0], [0.0, 0.0, cond_silica]],
            dtype=np.float64,
        )
        self.cond_tensor = cond_tensor
        return cond_tensor


if __name__ == "__main__":
    pass
