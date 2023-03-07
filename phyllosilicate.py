"""Calculate electrical properties of phillosillicate"""
# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=no-member
from typing import Dict, List, Tuple
from logging import Logger
from sys import float_info
from os import path, PathLike
import math
from copy import deepcopy
import pickle

import numpy as np
from scipy.integrate import quad

import constants as const
from constants import Species, IonProp
from fluid import NaCl


# load global parameters
# for smectite, infinite diffuse layer case
smectite_inf_init_pth: PathLike = path.join(
    path.dirname(__file__), "params", "smectite_inf_init.pkl"
)
with open(smectite_inf_init_pth, "rb") as pkf:
    smectite_inf_init_params = pickle.load(pkf)

# for smectite, truncated diffuse layer case
smectite_trun_init_pth: PathLike = path.join(
    path.dirname(__file__), "params", "smectite_trun_init.pkl"
)
with open(smectite_trun_init_pth, "rb") as pkf:
    smectite_trun_init_params = pickle.load(pkf)

# for kaolinite
kaolinite_init_pth: PathLike = path.join(
    path.dirname(__file__), "params", "kaolinite_init.pkl"
)
with open(kaolinite_init_pth, "rb") as pkf:
    kaolinite_init_params = pickle.load(pkf)


class Phyllosilicate:
    """
    Phyllosilicate Class
    It has a function to calculate the conductivity of phyllosilicate particles and
    the member variables necessary for the calculation.

    To calculate the surface potential, we use the equation proposed by
    Gonçalvès et al. (2007). The equation has been modified in the following points:
        1. In eq.(11), add Avogadro's number and a factor of 1000 to match units of concentration
        2. The units of Qi in eq.(16) were modified from charges nm-2 to C/m2
        3. The phib in eq.(23) was modified to phi0.
        4. Multiply the first term in eq.(16) and eq.(17) by 1.0e18
        5. Added the negative sign of the molecule of sinh in eq.(21).
        6. Corrected phiD in eq.(21) to phid (uppercase to lowercase).
        7. Multiply the coefficient of eq.(32) by Avogadro's constant
        8. Cf in eq.(32) and (34) are multiplied by 1000 for unit matching.

    References:
        Gonçalvès J., P. Rousseau-Gueutin, A. Revil, 2007, doi:10.1016/j.jcis.2007.07.023
        Leroy P., A. Revil, 2004, doi:10.1016/j.jcis.2003.08.007
        Leroy p., and Revil A., 2009, doi:10.1029/2008JB006114
        Shirozu, 1998, Introduction to Clay Mineralogy
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        nacl: NaCl,
        layer_width: float = 1.3e-9,
        gamma_1: float = 0.0,
        gamma_2: float = 5.5,
        gamma_3: float = 5.5,
        qi: float = -1.0,
        potential_0: float = None,
        potential_stern: float = None,
        potential_zeta: float = None,
        potential_r: float = None,
        charge_0: float = None,
        charge_stern: float = None,
        charge_diffuse: float = None,
        xd: float = None,
        cond_stern_plus_edl: float = None,
        logger: Logger = None,
    ):
        # TODO: m_layer_widthのassertionを電位, 電荷を計算する関数に加える
        # TODO: oからStern層までの長さを計算する関数作り、積分区間を変更する
        # TODO: 中性条件以外の条件だと, f6, f7はH+, OH-の寄与を考慮する必要がでてくるので修正する.
        # TODO: NaCl濃度が約3M以上で, truncatedの場合, 収束が悪い (10^-4)不具合があるので, 原因を特定して修正する
        # TODO: external_propsクラス (or Dict)を引数としてメンバ変数を減らす
        """Initialize Phyllosilicate class.

        Args:
            nacl (NaCl): Instance of NaCl class
            layer_width (float): Distance between sheets of phyllosilicate minerals
                (unit: m). Defaults to 1.3e-9 (When 3 water molecules are trapped).
            gamma_1 (float): Surface site densities of aluminol (unit: sites/nm2).
            gamma_2 (float): Surface site densities of sianol (unit: sites/nm2).
            gamma_3 (float): Surface site densities of >Si-O-Al< (unit: sites/nm2).
            qi (float): Layer charge density (charge/nm2).
            potential_0 (float, optional): Surface potential (unit: V).
            potential_stern (float, optional): Stern plane potential (unit: V).
            potential_zeta (float, optional): Zeta plane potential (unit: V).
            potential_r (float, optional): Potential at the position truncated inside the inter layer (unit: V).
            charge_0 (float, optional): Charges in surface layer (unit: C/m3).
            charge_stern (float, optional): Charges in stern layer (unit: C/m3).
            charge_zeta (float, optional): Charges in zeta layer (unit: C/m3).
            xd (float, optional): Distance from mineral surface to zeta plane (unit: V).
            cond_stern_plus_edl (float, optional): Conductivity of Stern layer + EDL (unit: S/m).
            logger (Logger): Logger for debugging.
        """
        # Check input values
        # xd must be smaller than layer width, so we should check that
        if (isinstance(xd, float) or isinstance(xd, int)) and (
            isinstance(layer_width, float) or isinstance(layer_width, int)
        ):
            assert (
                xd < layer_width * 0.5
            ), "xd must not exceed half the length of layer_width"

        ####################################################
        # Parameters related to the external environment
        ####################################################
        self.m_temperature: float = nacl.get_temperature()
        self.m_ion_props: Dict = nacl.get_ion_props()
        self.m_dielec_water: float = nacl.get_dielec_water()
        ####################################################
        # Parameters held by phyllosilicate
        ####################################################
        self.m_layer_width: float = layer_width
        self.m_gamma_1: float = gamma_1
        self.m_gamma_2: float = gamma_2
        self.m_gamma_3: float = gamma_3
        self.m_qi: float = qi * 1.0e18 * const.ELEMENTARY_CHARGE
        # k1 (float): equilibrium constant of eq.(12) in Gonçalvès et al. (2007).
        # k2 (float): equilibrium constant of eq.(13) in Gonçalvès et al. (2007).
        # k3 (float): equilibrium constant of eq.(14) in Gonçalvès et al. (2007).
        # k4 (float): equilibrium constant of eq.(15) in Gonçalvès et al. (2007).
        # c1 (float): capacitance of stern layer (unit: F/m2).
        # c2 (float): capacitance of diffuse layer (unit: F/m2).
        self.m_k1: float = None
        self.m_k2: float = None
        self.m_k3: float = None
        self.m_k4: float = None
        self.m_c1: float = None
        self.m_c2: float = None
        self.m_potential_0: float = potential_0
        self.m_potential_stern: float = potential_stern
        self.m_potential_zeta: float = potential_zeta
        self.m_potential_r: float = potential_r
        self.m_charge_0: float = charge_0
        self.m_charge_stern: float = charge_stern
        self.m_charge_diffuse: float = charge_diffuse
        self.m_xd = xd
        self.m_cond_stern_plus_edl = cond_stern_plus_edl
        # Parameters subordinate to those required for phyllosilicate initialization,
        # but useful to be obtained in advance
        self.m_ionic_strength = None
        self.m_kappa = None
        self.m_kappa_truncated = None
        self.m_kappa_stern = None
        self.m_qs_coeff1_inf = None
        self.m_qs_coeff2_inf = None
        self.m_cond_tensor = None
        self.m_cond_infdiffuse = None
        self.m_double_layer_length = None

        ####################################################
        # DEBUG LOGGER
        ####################################################
        self.m_logger = logger

        # Calculate frequently used constants and parameters.
        self.init_default()

        # START DEBUGGING
        if self.m_logger is not None:
            self.m_logger.info("Initialize Phyllosilicate")
            for name, value in vars(self).items():
                _msg = f"name: {name}, value: {value}"
                self.m_logger.debug(_msg)

    def init_default(self) -> None:
        """Calculate constants and parameters commonly used when computing
        electrical parameters
        """
        if self.m_qi < 0.0 and self.m_gamma_1 == 0.0:
            self.__set_constant_for_smectite_truncated()
        else:
            self.__set_constant_for_kaolinite()
        _e = const.ELEMENTARY_CHARGE
        _kb = const.BOLTZMANN_CONST

        # Ionic strength (based on Leroy & Revil, 2004)
        strength = 0.0
        for elem, props in self.m_ion_props.items():
            if elem in (Species.Na.name, Species.Cl.name):
                strength += props[IonProp.Concentration.name]
        strength += self.m_ion_props[Species.H.name][IonProp.Concentration.name]
        self.m_ionic_strength = strength

        # calculate kappa (eq.(11) of Gonçalvès et al., 2004)
        # Electrolyte concentration is assumed to be equal to Na+ concentration
        top = (
            2000.0
            * self.m_ion_props[Species.Na.name][IonProp.Concentration.name]
            * const.AVOGADRO_CONST
            * _e**2
        )
        bottom = self.m_dielec_water * _kb * self.m_temperature
        self.m_kappa = np.sqrt(top / bottom)

    def __set_constant_for_kaolinite(self) -> None:
        """Set the constants for the case of kaolinite and infinite diffuse layer"""
        self.m_k1: float = const.calc_equibilium_const(
            const.dg_aloh_kaol, self.m_temperature
        )
        self.m_k2: float = const.calc_equibilium_const(
            const.dg_sioh_kaol, self.m_temperature
        )
        self.m_k3: float = const.calc_equibilium_const(
            const.dg_xh_kaol, self.m_temperature
        )
        self.m_k4: float = const.calc_equibilium_const(
            const.dg_xna_kaol, self.m_temperature
        )
        self.m_c1: float = const.c1_kaol
        self.m_c2: float = const.c2_kaol

    def __set_constant_for_smectite_inf(self) -> None:
        """Set the constants for the case of smectite and infinite diffuse layer"""
        self.m_k1: float = const.calc_equibilium_const(
            const.dg_aloh_smec_inf, self.m_temperature
        )
        self.m_k2: float = const.calc_equibilium_const(
            const.dg_sioh_smec_inf, self.m_temperature
        )
        self.m_k3: float = const.calc_equibilium_const(
            const.dg_xh_smec_inf, self.m_temperature
        )
        self.m_k4: float = const.calc_equibilium_const(
            const.dg_xna_smec_inf, self.m_temperature
        )
        self.m_c1: float = const.c1_smec_inf
        self.m_c2: float = const.c2_smec_inf

    def __set_constant_for_smectite_truncated(self) -> None:
        """Set the constants for the case of smectite and midway truncation of the diffuse layer"""
        self.m_k1: float = const.calc_equibilium_const(
            const.dg_aloh_smec_trun, self.m_temperature
        )
        self.m_k2: float = const.calc_equibilium_const(
            const.dg_sioh_smec_trun, self.m_temperature
        )
        self.m_k3: float = const.calc_equibilium_const(
            const.dg_xh_smec_trun, self.m_temperature
        )
        self.m_k4: float = const.calc_equibilium_const(
            const.dg_xna_smec_trun, self.m_temperature
        )
        self.m_c1: float = const.c1_smec_trun
        self.m_c2: float = const.c2_smec_trun

    def __check_constant_as_smectite_truncated(self) -> bool:
        """Check if it is equal to the value in Table 1 of Gonçalvès et al., 2007.

        Returns:
            bool: Returns True if the above conditions are satisfied.
        """
        flag = True
        if not math.isclose(
            self.m_k1,
            const.calc_equibilium_const(const.dg_aloh_smec_trun, self.m_temperature),
            rel_tol=1.0e-10,
        ):
            flag = False
        if not math.isclose(
            self.m_k2,
            const.calc_equibilium_const(const.dg_sioh_smec_trun, self.m_temperature),
            rel_tol=1.0e-10,
        ):
            flag = False
        if not math.isclose(
            self.m_k3,
            const.calc_equibilium_const(const.dg_xh_smec_trun, self.m_temperature),
            rel_tol=1.0e-10,
        ):
            flag = False
        if not math.isclose(
            self.m_k4,
            const.calc_equibilium_const(const.dg_xna_smec_trun, self.m_temperature),
            rel_tol=1.0e-10,
        ):
            flag = False
        if not math.isclose(self.m_c1, const.c1_smec_trun, rel_tol=1.0e-10):
            flag = False
        if not math.isclose(self.m_c2, const.c2_smec_trun, rel_tol=1.0e-10):
            flag = False
        return flag

    def __calc_f1(self, phi0: float, q0: float) -> float:
        """Calculate eq.(16) of Gonçalvès et al. (2004)

        Args:
            phi0 (float): surface plane potential
            q0 (float): surface layer charge

        Returns:
            float: left side of eq.(16) minus right side
        """
        _e = const.ELEMENTARY_CHARGE
        _kb = const.BOLTZMANN_CONST
        ch = self.m_ion_props[Species.H.name][IonProp.Activity.name]
        x = -_e * phi0 / (_kb * self.m_temperature)
        a = self.__calc_A(phi0)
        b = self.__calc_B(phi0)
        right1 = self.m_gamma_1 / a * (ch / self.m_k1 * np.exp(x) - 1.0)
        right2 = self.m_gamma_2 / b * (ch / self.m_k2 * np.exp(x) - 1.0)
        right3 = 2.0 * self.m_gamma_3
        f1 = q0 - _e * 0.5 * 1.0e18 * (right1 + right2 - right3) - self.m_qi
        return f1

    def __calc_f2(self, phib: float, qb: float) -> float:
        """Calculate eq.(17) of Gonçalvès et al. (2004)

        Args:
            phib (float): stern plane potential
            qb (float): stern layer charge

        Returns:
            float: left side of eq.(17) minus right side
        """
        _e = const.ELEMENTARY_CHARGE
        _kb = const.BOLTZMANN_CONST
        ch = self.m_ion_props[Species.H.name][IonProp.Activity.name]
        cna = self.m_ion_props[Species.Na.name][IonProp.Activity.name]
        x = -_e * phib / (_kb * self.m_temperature)
        c = self.__calc_C(phib)
        f2 = (
            qb
            - _e
            * self.m_gamma_3
            / c
            * (ch / self.m_k3 + cna / self.m_k4)
            * np.exp(x)
            * 1.0e18
        )
        return f2

    def __calc_f3(self, q0: float, qb: float, qs: float) -> float:
        """Calculate eq.(18) of Gonçalvès et al. (2004)

        Args:
            q0 (float): surface layer charge
            qb (float): stern layer charge
            qs (float): diffuse layer charge
        Returns:
            float: left side of eq.(18) minus right side
        """
        return q0 + qb + qs

    def __calc_f4(self, phi0: float, phib: float, q0: float) -> float:
        """Calculate eq.(19) of Gonçalvès et al. (2004)

        Args:
            phi0 (float): surface place potential
            phib (float): zeta plane potential
            q0 (float): surface layer charge
        Returns:
            float: left side of eq.(19) minus right side
        """
        return phi0 - phib - q0 / self.m_c1

    def __calc_f5(self, phib: float, phid: float, qs: float) -> float:
        """Calculate eq.(20) of Gonçalvès et al. (2004)

        Args:
            phib (float): stern place potential
            phid (float): zeta plane potential
            qs (float): diffuse layer charge
        Returns:
            float: left side of eq.(20) minus right side
        """
        return phib - phid + qs / self.m_c2

    def __calc_f6(self, phid: float, qs: float) -> float:
        """Calculate eq.(21) of Gonçalvès et al. (2004)

        Args:
            phid (float): zeta place potential
            qs (float): diffuse layer charge
        Returns:
            float: left side of eq.(21) minus right side
        """
        _e = const.ELEMENTARY_CHARGE
        _na = const.AVOGADRO_CONST
        kb = const.BOLTZMANN_CONST
        dielec = self.m_dielec_water
        x = _e * phid / (2.0 * kb * self.m_temperature)
        coeff = np.sqrt(
            8.0e3 * _na * self.m_ionic_strength * dielec * kb * self.m_temperature
        )
        f6 = qs - coeff * np.sinh(-x)
        return f6

    def __calc_f6_truncated(self, phid: float, phir: float, qs: float) -> float:
        """Calculate eq.(32) of Gonçalvès et al. (2004)

        Args:
            phid (float): zeta place potential
            phir (float): truncated plane potential
            qs (float): diffuse layer charge
        Returns:
            float: left side of eq.(32) minus right side
        """
        dielec = self.m_dielec_water
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        cf = self.m_ion_props[Species.Na.name][IonProp.Concentration.name] * 1.0e3
        _e = const.ELEMENTARY_CHARGE
        _na = const.AVOGADRO_CONST
        c = _e / (kb * _t)
        coeff = 2.0 * np.sqrt(_na * cf * kb * _t * dielec)
        right1 = np.cosh(c * phid)
        right2 = np.cosh(c * phir)
        return qs - coeff * np.sqrt(right1 - right2)

    def __calc_f7_truncated(self, phid: float, phir: float) -> float:
        """Calculate eq.(33) of Gonçalvès et al. (2004)

        Args:
            phid (float): zeta place potential
            phir (float): truncated plane potential
            qs (float): diffuse layer charge
        Returns:
            float: left side of eq.(33) minus right side
        """
        # TODO: ６回微分まで実装する
        # electrolyte concentration assumed equal to cf
        cf = self.m_ion_props[Species.Na.name][IonProp.Concentration.name] * 1.0e3
        _e = const.ELEMENTARY_CHARGE
        dielec = self.m_dielec_water
        _t = self.m_temperature
        _r = self.m_layer_width / 2.0
        kb = const.BOLTZMANN_CONST
        _na = const.AVOGADRO_CONST
        xd = self.m_xd
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
            phi0 (float): surface plane potential
            phib (float): stern plane potential
            phid (float): zeta plane potential

        Returns:
            np.ndarray: 6 rows and 6 columns of Jacobians
        """
        f1_phi0 = self.__calc_f1_phi0(phi0)
        f1_q0 = 1.0
        f2_phib = self.__calc_f2_phib(phib)
        f2_qb = 1.0
        f3_q0, f3_qb, f3_qs = 1.0, 1.0, 1.0
        f4_phi0, f4_phib, f4_q0 = 1.0, -1.0, -1.0 / self.m_c1
        f5_phib, f5_phid, f5_qs = 1.0, -1.0, 1.0 / self.m_c2
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
            phi0 (float): surface plane potential
            phib (float): stern plane potential
            phid (float): zeta plane potential
            phir (float): truncated plane potential

        Returns:
            np.ndarray: 7 rows and 7 columns of Jacobians
        """
        f1_phi0 = self.__calc_f1_phi0(phi0)
        f1_q0 = 1.0
        f2_phib = self.__calc_f2_phib(phib)
        f2_qb = 1.0
        f3_q0, f3_qb, f3_qs = 1.0, 1.0, 1.0
        f4_phi0, f4_phib, f4_q0 = 1.0, -1.0, -1.0 / self.m_c1
        f5_phib, f5_phid, f5_qs = 1.0, -1.0, 1.0 / self.m_c2
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
            phi0 (float): surface plane potential

        Returns:
            float: ∂f1/∂phi0
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29Divide%5Bc%2C1%2Bb*exp%5C%2840%29-a*x%5C%2841%29%5D*%5C%2840%29b*exp%5C%2840%29-a*x%5C%2841%29-1%5C%2841%29%2BDivide%5Be%2C1%2Bd*exp%5C%2840%29-a*x%5C%2841%29%5D*%5C%2840%29d*exp%5C%2840%29-a*x%5C%2841%29-1%5C%2841%29%5C%2841%29%2Cx%5D&lang=ja
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        ch = self.m_ion_props[Species.H.name][IonProp.Activity.name]
        a = _e / (kb * self.m_temperature)
        x = a * phi0
        b = ch / self.m_k1
        c = self.m_gamma_1
        d = ch / self.m_k2
        e = self.m_gamma_2
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
            phib (float): stern plane potential

        Returns:
            float: ∂f2/∂phib
        """
        # https://www.wolframalpha.com/input?key=&i2d=true&i=D%5B%5C%2840%29Divide%5Bd%2C1%2B%5C%2840%29b%2Bc%5C%2841%29*exp%5C%2840%29-a*x%5C%2841%29%5D*%5C%2840%29b%2Bc%5C%2841%29*exp%5C%2840%29-a*x%5C%2841%29%5C%2841%29%2Cx%5D&lang=ja
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        ch = self.m_ion_props[Species.H.name][IonProp.Activity.name]
        na = self.m_ion_props[Species.Na.name][IonProp.Activity.name]
        a = _e / (kb * self.m_temperature)
        exp = np.exp(a * phib)
        b = ch / self.m_k3
        c = na / self.m_k4
        d = _e * 1.0e18 * self.m_gamma_3
        # overflow prevention
        top = a * d * (b + c)
        bottom1 = 1.0 + (b + c) / exp
        bottom2 = exp + b + c
        return top / (bottom1 * bottom2)

    def __calc_f6_phid(self, phid: float) -> float:
        """Calculate ∂f6/∂phid

        Args:
            phid (float): zeta plane potential

        Returns:
            float: ∂f6/∂phid
        """
        _e = const.ELEMENTARY_CHARGE
        _na = const.AVOGADRO_CONST
        kb = const.BOLTZMANN_CONST
        _dielec = self.m_dielec_water
        _t = self.m_temperature
        _i = self.m_ionic_strength
        a = _e / (2.0 * kb * _t)
        coeff = np.sqrt(8.0e3 * _na * _i * _dielec * kb * _t)
        cosh = a * np.cosh(a * phid)
        return coeff * cosh

    def __calc_f6_phid_truncated(self, phid: float, phir: float) -> float:
        """Calculate ∂f6/∂phid for truncated case

        Args:
            phid (float): zeta plane potential
            phir (float): truncated plane potential

        Returns:
            float: ∂f6/∂phid
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29-b*Sqrt%5Bcosh%5C%2840%29c*x%5C%2841%29-cosh%5C%2840%29c*y%5C%2841%29%5D%5C%2841%29%2Cx%5D&lang=ja
        # x: phir, y: phid
        dielec = self.m_dielec_water
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        _e = const.ELEMENTARY_CHARGE
        # electrolyte concentration is assumed to be equal to Na+ concentration
        cf = self.m_ion_props[Species.Na.name][IonProp.Concentration.name] * 1.0e3
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
            phid (float): zeta plane potential
            phir (float): truncated plane potential

        Returns:
            float: ∂f6/∂phir
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29-b*Sqrt%5Bcosh%5C%2840%29c*x%5C%2841%29-cosh%5C%2840%29c*y%5C%2841%29%5D%5C%2841%29%2Cy%5D&lang=ja
        dielec = self.m_dielec_water
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        # electrolyte concentration is assumed to be equal to Na+ concentration
        cf = self.m_ion_props[Species.Na.name][IonProp.Concentration.name] * 1.0e3
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
            phir (float): truncated plane potential

        Returns:
            float: ∂f7/∂phir
        """
        # https://www.wolframalpha.com/input?i2d=true&i=D%5B%5C%2840%29d-x-a*sinh%5C%2840%29b*x%5C%2841%29*%5C%2840%29y-r%5C%2841%29**2-c*sinh%5C%2840%292*b*x%5C%2841%29*%5C%2840%29y-r%5C%2841%29**4%5C%2841%29%2Cx%5D&lang=ja
        # electrolyte concentration is assumed to be equal to Na+ concentration
        cf = self.m_ion_props[Species.Na.name][IonProp.Concentration.name] * 1.0e3
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        _r = self.m_layer_width * 0.5
        dielec = self.m_dielec_water
        _na = const.AVOGADRO_CONST
        xd = self.m_xd
        b = _e * _na * cf / dielec
        c = _e / (kb * _t)
        d = _e**3 * _na**2 * cf**2 / (12.0 * dielec**2 * kb * _t)
        _1 = -c * b * np.cosh(c * phir) * (xd - _r) ** 2
        _2 = -2.0 * c * d * np.cosh(2.0 * c * phir) * (xd - _r) ** 4
        return -1.0 + _1 + _2

    def __calc_A(self, phi0: float) -> float:
        """Calculate eq.(22) of Gonçalvès et al. (2004)

        Args:
            phi0 (float): surface layer potential

        Returns:
            float: value of "A" in eq.(22)
        """
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        t = self.m_temperature
        ch = self.m_ion_props[Species.H.name][IonProp.Activity.name]
        k1 = self.m_k1
        return 1.0 + ch / k1 * np.exp(-_e * phi0 / (kb * t))

    def __calc_B(self, phi0: float) -> float:
        """Calculate eq.(23) of Gonçalvès et al. (2004)

        Args:
            phi0 (float): surface plane potential

        Returns:
            float: value of "B" in eq.(23)
        """
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        t = self.m_temperature
        ch = self.m_ion_props[Species.H.name][IonProp.Activity.name]
        k2 = self.m_k2
        return 1.0 + ch / k2 * np.exp(-_e * phi0 / (kb * t))

    def __calc_C(self, phib: float) -> float:
        """Calculate eq.(24) of Gonçalvès et al. (2004)

        Args:
            phib (float): stern plane potential

        Returns:
            float: value of "C" in eq.(23)
        """
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        t = self.m_temperature
        ch = self.m_ion_props[Species.H.name][IonProp.Activity.name]
        cna = self.m_ion_props[Species.Na.name][IonProp.Activity.name]
        k3 = self.m_k3
        k4 = self.m_k4
        return 1.0 + (ch / k3 + cna / k4) * np.exp(-_e * phib / (kb * t))

    def __calc_functions_inf(self, xn: np.ndarray) -> np.ndarray:
        """Calculate functions 1 to 6 for infinite diffuse layer version

        Args:
            xn (np.ndarray): 2d array of electrical parameters (column vector).
            xn[0][0]: potential of O plane
            xn[1][0]: potential of Stern plane
            xn[2][0]: potential of Zeta plane
            xn[3][0]: charge of O layer
            xn[4][0]: charge of Stern layer
            xn[5][0]: charge of Diffuse layer

        Returns:
            np.ndarray: 2d array containing the value of each function
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
        """Calculate functions 1 to 7 for truncated version

        Args:
            xn (np.ndarray): 2d array of electrical parameters (column vector).
            xn[0][0]: potential of O plane
            xn[1][0]: potential of Stern plane
            xn[2][0]: potential of Zeta plane
            xn[3][0]: potential of Truncated plane
            xn[4][0]: charge of O layer
            xn[5][0]: charge of Stern layer
            xn[6][0]: charge of Diffuse layer

        Returns:
            np.ndarray: 2d array containing the value of each function
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
        """Calculate the amount of charge (C/m3) at _x distance from the
        surface in the case of infinite diffuse layer development

        Args:
            _x (float): Distance from surface to pore

        Returns:
            float: Charge density at _x distance from the surface (C/m3)
        """
        assert self.__check_if_calculated_qs_coeff(), (
            "Before calculating Qs, we should obtain"
            "the coefficients for calculating Qs"
        )
        return self.m_qs_coeff1_inf * np.sinh(
            self.m_qs_coeff2_inf * np.exp((-1.0) * self.m_kappa * _x)
        )

    def __calc_qs_coeff_inf(self) -> None:
        """Amount of charge in an infinitely developing diffuse layer."""
        assert self._check_if_calculated_electrical_params_inf(), (
            "Before calculating xd, we should obtain electrical"
            "arameters for infinite diffuse layer case"
        )
        _e = const.ELEMENTARY_CHARGE
        # electrolyte concentration is assumed to be equal to Na+ concentration
        self.m_qs_coeff1_inf = (
            2000.0
            * _e
            * const.AVOGADRO_CONST
            * self.m_ion_props[Species.Na.name][IonProp.Concentration.name]
        )
        self.m_qs_coeff2_inf = (
            -_e * self.m_potential_zeta / (const.BOLTZMANN_CONST * self.m_temperature)
        )

    def __check_if_calculated_qs_coeff(self) -> bool:
        """Find out if the coefficients for calculating Qs
        have already been calculated.

        Returns:
            bool: if already calculated, True
        """
        flag = True
        if self.m_qs_coeff1_inf is None:
            flag = False
        if self.m_qs_coeff2_inf is None:
            flag = False
        return flag

    def calc_xd(self) -> Tuple[float]:
        """Calculate the distance from the surface to the zeta plane (xd)

        Returns:
            Tuple[float]: xd, integral error
        """
        assert self._check_if_calculated_electrical_params_inf(), (
            "Before calculating xd, we should obtain electrical"
            "arameters for infinite diffuse layer case"
        )
        assert (
            not self._check_if_calculated_electrical_params_truncated()
        )  # TODO?: infとtruncatedでパラメータを場合分けしたほうがいいかも
        if not self.__check_if_calculated_qs_coeff():
            self.__calc_qs_coeff_inf()
        _qs = self.m_charge_diffuse
        # xd_ls: List = [1.0e-12 + float(i) * 1.0e-10 for i in range(100)]
        xd_ls: List = np.logspace(-15, -7, 1000, base=10.0).tolist()
        qs_ls: List = []
        err_ls: List = []
        for _xd_tmp in xd_ls:
            _xd_dl = _xd_tmp + 1.0 / self.m_kappa
            qs_tmp, _err = quad(self.__calc_qs_inf, _xd_tmp, _xd_dl, limit=1000)
            qs_ls.append(qs_tmp)
            err_ls.append(_err)
        qs_diff = np.square(np.array(qs_ls) - _qs)
        _idx = np.argmin(qs_diff)
        _xd = xd_ls[_idx]
        # TODO: 以下加えたほうがよいのか検討する
        # if self.m_layer_width * 0.5 < _xd:
        #     _xd = self.m_layer_width * 0.5
        self.m_xd = _xd
        _err = err_ls[_idx]
        assert _err < abs(_qs), "Integral error exceeds the value of qs"
        return self.m_xd, _err

    def calc_potentials_and_charges_inf(
        self,
        x_init: List = None,
        iter_max: int = 1000,
        convergence_condition: float = 1.0e-10,
        oscillation_tol: float = 1.0e-04,
        beta: float = 0.75,
        lamda: float = 2.0,
    ) -> List:
        """Calculate the potential and charge of each layer
        in the case of infinite diffuse layer development.
        eq.(16)~(21) of Gonçalvès et al. (2007) is used.
        Damped Newton-Raphson method is applied.

        x_init (List): Initial electrical parameters (length is 6)
        iter_max (int): Maximum number of iterations when calculating potential and charge
            using the Newton-Raphson method. Defaults to 1000.
        convergence_condition (float): Convergence conditions for calculating potential
            nd charge using the Newton-Raphson method. Defaults to 1.0e-10.
        oscillation_tol (float): Oscillation tolerance. Defaults to 1.0e-5.
        beta (float): Hyper parameter for damping(0, 1). The larger this value,
            the smaller the damping effect.
        lamda (float): Hyper parameter for damping([1, ∞]). The amount by which
            the step-width coefficients are updated in a single iteration

        Raises:
            RuntimeError: Occurs when loop count exceeds m_iter_max &
             norm of step exceeds oscillation_tol

        Returns:
            List: list containing potentials and charges
              [phi0, phib, phid, q0, qb, qs]
        """
        assert 0.0 < beta < 1.0
        assert lamda > 1.0

        # TODO: May need to consider other cases (e.g., illite, etc.)
        if self.m_qi < 0.0 and self.m_gamma_1 == 0.0:
            self.__set_constant_for_smectite_inf()
        else:
            self.__set_constant_for_kaolinite()

        if x_init is None:
            # Set initial electrical parameters
            if self.m_qi < 0 and self.m_gamma_1 == 0.0:
                # for smectite case
                params = smectite_inf_init_params
            elif self.m_qi == 0 and self.m_gamma_1 > 0.0:
                # for kaolinite case
                params = kaolinite_init_params
            else:
                # TODO: Prepare more initial parameters in other minerals.
                params = kaolinite_init_params
            ch = self.m_ion_props[Species.H.name][IonProp.Concentration.name]
            ch_ls = list(params.keys())
            _idx_ch: int = np.argmin(np.square(np.array(ch_ls, dtype=np.float64) - ch))
            cna = self.m_ion_props[Species.Na.name][IonProp.Concentration.name]
            cna_dct: Dict = params[ch_ls[_idx_ch]]
            cna_ls = list(cna_dct.keys())
            _idx_cna: int = np.argmin(
                np.square(np.array(cna_ls, dtype=np.float64) - cna)
            )
            x_init = cna_dct[cna_ls[_idx_cna]]
        xn = np.array(x_init, np.float64).reshape(-1, 1)
        fn = self.__calc_functions_inf(xn)
        norm_fn: float = np.sum(np.sqrt(np.square(fn)), axis=0)[0]
        # The convergence condition is that the L2 norm in eqs.1~7
        # becomes sufficiently small.
        cou = 0
        while convergence_condition < norm_fn:
            _j = self._calc_Goncalves_jacobian(xn[0][0], xn[1][0], xn[2][0])
            # To avoid overflow when calculating the inverse matrix
            _j = _j + float_info.min
            _j_inv = np.linalg.inv(_j)
            step = np.matmul(_j_inv, fn)
            # Damping is applied. References are listed below:
            # [1] http://www.misojiro.t.u-tokyo.ac.jp/~murota/lect-suchi/newton130805.pdf
            # [2] http://www.ep.sci.hokudai.ac.jp/~gfdlab/comptech/y2016/resume/070_newton/2014_0626-mkuriki.pdf
            _cou_damp: int = 0
            _norm_fn_tmp, _rhs = float_info.max, float_info.min
            # At least once, enter the following loop
            while _norm_fn_tmp > _rhs:
                # update μ
                _mu = 1.0 / (lamda**_cou_damp)
                # calculate left hand side of eq.(21) of [1]
                xn_tmp: np.ndarray = xn - _mu * step
                fn_tmp = self.__calc_functions_inf(xn_tmp)
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
        xn = xn.T.tolist()[0]
        self.m_potential_0 = xn[0]
        self.m_potential_stern = xn[1]
        self.m_potential_zeta = xn[2]
        self.m_charge_0 = xn[3]
        self.m_charge_stern = xn[4]
        self.m_charge_diffuse = xn[5]
        # DEBUG
        if self.m_logger is not None:
            self.m_logger.info(
                "Finished the calculation of electrical "
                "properties for an infinite diffusion layer"
            )
            self.m_logger.debug(f"m_potential_0: {self.m_potential_0}")
            self.m_logger.debug(f"m_potential_stern: {self.m_potential_stern}")
            self.m_logger.debug(f"m_potential_zeta: {self.m_potential_zeta}")
            self.m_logger.debug(f"m_charge_0: {self.m_charge_0}")
            self.m_logger.debug(f"m_charge_stern: {self.m_charge_stern}")
            self.m_logger.debug(f"m_charge_diffuse: {self.m_charge_diffuse}")
        return xn

    def calc_potentials_and_charges_truncated(
        self,
        x_init: List = None,
        iter_max: int = 1000,
        convergence_condition: float = 1.0e-10,
        oscillation_tol: float = 1.0e-04,
        beta: float = 0.75,
        lamda: float = 2.0,
    ) -> List:
        """Calculate the potential and charge of each layer
        in the case of truncated diffuse layer development.
        eq.(16)~(20), (32), (33) of Gonçalvès et al. (2007) is used.
        Damped Newton-Raphson method is applied.

        x_init (List): Initial electrical parameters (length is 7)
        iter_max (int): Maximum number of iterations when calculating potential and charge
            using the Newton-Raphson method. Defaults to 1000.
        convergence_condition (float): Convergence conditions for calculating potential
            nd charge using the Newton-Raphson method. Defaults to 1.0e-10.
        oscillation_tol (float): Oscillation tolerance. Defaults to 1.0e-5.
        beta (float): Hyper parameter for damping. The larger this value,
            the smaller the damping effect.
        lamda (float): Hyper parameter for damping. The amount by which
            the step-width coefficients are updated in a single iteration
        Raises:
            RuntimeError: Occurs when loop count exceeds iter_max &
             norm of step exceeds oscillation_tol

        Returns:
            List: list containing potentials and charges
              [phi0, phib, phid, phir, q0, qb, qs]
        """
        assert 0.0 < beta < 1.0
        assert lamda > 1.0

        # obtain init values based on infinity developed diffuse layer
        # phi0, phib, phid, phir, q0, qb, qs
        if not self._check_if_calculated_electrical_params_inf():
            self.calc_potentials_and_charges_inf()
        # TODO: Currently, it is assumed that this function is called only in the case of
        # smectite. If we want to calculate the electrochemical properties of another clay
        # mineral, such as vermiculite, we should modify this function.
        if not self.__check_constant_as_smectite_truncated():
            self.__set_constant_for_smectite_truncated()
        if self.m_xd is None:
            self.calc_xd()
        if x_init is None:
            r_ls = list(smectite_trun_init_params.keys())
            _r = self.m_layer_width
            _idx = np.argmin(np.square((np.array(r_ls, dtype=np.float64) - _r)))
            ch_cna_dict: Dict = smectite_trun_init_params[r_ls[_idx]]
            _ch = self.m_ion_props[Species.H.name][IonProp.Concentration.name]
            _cna = self.m_ion_props[Species.Na.name][IonProp.Concentration.name]
            ch_ls = list(ch_cna_dict.keys())
            _idx = np.argmin(np.square((np.array(ch_ls, dtype=np.float64) - _ch)))
            cna_dct: Dict = ch_cna_dict[ch_ls[_idx]]
            cna_ls = list(cna_dct.keys())
            _idx = np.argmin(np.square((np.array(cna_ls, dtype=np.float64) - _cna)))
            x_init = cna_dct[cna_ls[_idx]]
        xn = np.array(x_init, np.float64).reshape(-1, 1)
        fn = self.__calc_functions_truncated(xn)
        norm_fn: float = np.sum(np.sqrt(np.square(fn)), axis=0)[0]
        # The convergence condition is that the L2 norm in eqs.1~7
        # becomes sufficiently small.
        cou = 0
        history_ls: List = []  # TODO: currently not used
        while convergence_condition < norm_fn:
            history_ls.append(norm_fn)
            _j = self._calc_Goncalves_jacobian_truncated(
                xn[0][0], xn[1][0], xn[2][0], xn[3][0]
            )
            # To avoid overflow when calculating the inverse matrix
            _j = _j + float_info.min
            _j_inv = np.linalg.inv(_j)
            step = np.matmul(_j_inv, fn)
            # Damping is applied. References are listed below:
            # [1] http://www.misojiro.t.u-tokyo.ac.jp/~murota/lect-suchi/newton130805.pdf
            # [2] http://www.ep.sci.hokudai.ac.jp/~gfdlab/comptech/y2016/resume/070_newton/2014_0626-mkuriki.pdf
            _cou_damp: int = 0
            _norm_fn_tmp, _rhs = float_info.max, float_info.min
            # At least once, enter the following loop
            while _norm_fn_tmp > _rhs:
                # update μ
                _mu = 1.0 / (lamda**_cou_damp)
                # calculate left hand side of eq.(21) of [1]
                xn_tmp: np.ndarray = xn - _mu * step
                fn_tmp = self.__calc_functions_truncated(xn_tmp)
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
                norm_step = np.sum(np.sqrt(np.square(step)), axis=0)[0]
                if norm_step > oscillation_tol:
                    _msg: str = (
                        f"Loop count exceeded {iter_max} times &"
                        f" exceeds oscillation tolerance: {norm_step}"
                    )
                    # log
                    if self.m_logger is not None:
                        self.m_logger.error(_msg)
                    raise RuntimeError(_msg)
                else:
                    break
        xn = xn.T.tolist()[0]
        self.m_potential_0 = xn[0]
        self.m_potential_stern = xn[1]
        self.m_potential_zeta = xn[2]
        self.m_potential_r = xn[3]
        self.m_charge_0 = xn[4]
        self.m_charge_stern = xn[5]
        self.m_charge_diffuse = xn[6]
        # DEBUG
        if self.m_logger is not None:
            self.m_logger.info(
                "Finished the calculation of electrical "
                "properties for an truncated diffusion layer"
            )
            self.m_logger.debug(f"m_potential_0: {self.m_potential_0}")
            self.m_logger.debug(f"m_potential_stern: {self.m_potential_stern}")
            self.m_logger.debug(f"m_potential_zeta: {self.m_potential_zeta}")
            self.m_logger.debug(f"m_potential_r: {self.m_potential_r}")
            self.m_logger.debug(f"m_charge_0: {self.m_charge_0}")
            self.m_logger.debug(f"m_charge_stern: {self.m_charge_stern}")
            self.m_logger.debug(f"m_charge_diffuse: {self.m_charge_diffuse}")
        return xn

    def _check_if_calculated_electrical_params_inf(self) -> bool:
        """Check if the electrical properties for infinite diffuse layer
         has already been calculated

        Returns:
            bool: if already calculated, return True
        """
        flag = True
        if self.m_potential_0 is None:
            flag = False
        if self.m_potential_stern is None:
            flag = False
        if self.m_potential_zeta is None:
            flag = False
        if self.m_charge_0 is None:
            flag = False
        if self.m_charge_stern is None:
            flag = False
        if self.m_charge_diffuse is None:
            flag = False
        return flag

    def _check_if_calculated_electrical_params_truncated(self) -> bool:
        """Check if the electrical properties for truncated diffuse layer
         has already been calculated

        Returns:
            bool: if already calculated, return True
        """
        flag = True
        if self.m_potential_0 is None:
            flag = False
        if self.m_potential_stern is None:
            flag = False
        if self.m_potential_zeta is None:
            flag = False
        if self.m_potential_r is None:
            flag = False
        if self.m_charge_0 is None:
            flag = False
        if self.m_charge_stern is None:
            flag = False
        if self.m_charge_diffuse is None:
            flag = False
        if self.m_xd is None:
            flag = False
        return flag

    def __calc_cond_at_x_inf_diffuse(self, _x: float) -> float:
        """Calculate the conductivity of the infinite diffuse layer.
        The following assumptions are made:
        1. Mobility is assumed to be a constant following Leroy & Revil(2009).

        Args:
            _x (float): Distance from zeta plane

        Returns:
            float: Conductivity at a point _x away from the zeta plane
        """
        assert self.m_kappa is not None, "self.m_kappa is None"
        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        _cond = 0.0
        potential = self.m_potential_zeta * np.exp((-1.0) * self.m_kappa * _x)
        for _, prop in self.m_ion_props.items():
            _conc = 1000.0 * prop[IonProp.Concentration.name]
            _v = prop[IonProp.Valence.name]
            _mobility = prop[IonProp.MobilityInfDiffuse.name]
            _conc = _conc * np.exp((-1.0) * _v * _e * potential / (kb * _t))
            _cond += _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_specific_cond_at_x_inf_diffuse(self, _x: float) -> float:
        """Calculate the specific conductivity of the infinite diffuse layer.
        Specific conductivity is defined as eq.(26) in Leroy and Revil (2004)

        Args:
            _x (float): Distance from zeta plane

        Returns:
            float: Specific conductivity at a point _x away from the zeta plane
        """
        assert self.m_kappa is not None, "self.m_kappa is None"
        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        potential = self.m_potential_zeta * np.exp((-1.0) * self.m_kappa * _x)
        prop_na = self.m_ion_props[Species.Na.name]
        _conc = 1000.0 * prop_na[IonProp.Concentration.name]
        _v = prop_na[IonProp.Valence.name]
        _mobility = prop_na[IonProp.MobilityInfDiffuse.name]
        _conc = _conc * np.exp((-1.0) * _v * _e * potential / (kb * _t))
        _cond = _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_cond_at_x_truncated_diffuse(self, _x: float) -> float:
        """Calculate the conductivity of the inter layer.
            The following assumptions are made:
            1. Mobility is assumed to be a constant following Leroy & Revil(2009).

        Args:
            _x (float): Distance from zeta plane

        Returns:
            float: Conductivity at a point _x away from the zeta plane
        """
        # TODO: Verify that the mobility is a constant (search temperature dependence or etc).
        assert self.m_kappa_truncated is not None, "self.m_kappa_truncated is None"
        assert self.m_xd <= _x, "self.m_xd > _x"
        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        _cond = 0.0
        potential = self.m_potential_zeta * np.exp((-1.0) * self.m_kappa_truncated * _x)
        for _, prop in self.m_ion_props.items():
            _conc = 1000.0 * prop[IonProp.Concentration.name]
            _v = prop[IonProp.Valence.name]
            _mobility = prop[IonProp.MobilityTrunDiffuse.name]
            _conc = _conc * np.exp((-1.0) * _v * _e * potential / (kb * _t))
            _cond += _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_specific_cond_at_x_truncated_diffuse(self, _x: float) -> float:
        """Calculate the specific conductivity of the truncated diffuse layer.

        Args:
            _x (float): Distance from zeta plane

        Returns:
            float: Specific conductivity at a point _x away from the zeta plane
        """
        # TODO: Verify that the mobility is a constant (search temperature dependence or etc).
        assert self.m_kappa_truncated is not None, "self.m_kappa_truncated is None"
        assert self.m_xd <= _x, "self.m_xd > _x"
        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        potential = self.m_potential_zeta * np.exp((-1.0) * self.m_kappa_truncated * _x)
        prop_na = self.m_ion_props[Species.Na.name]
        _conc = 1000.0 * prop_na[IonProp.Concentration.name]
        _v = prop_na[IonProp.Valence.name]
        _mobility = prop_na[IonProp.MobilityTrunDiffuse.name]
        _conc = _conc * (np.exp((-1.0) * _v * _e * potential / (kb * _t)) - 1.0)
        _cond = _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_cond_at_x_stern_inf(self, _x: float) -> float:
        """Calculate the conductivity of the inter layer.
            The following assumptions are made:
            1. Mobility is assumed to be a constant following Leroy & Revil(2009).

        Args:
            _x (float): Distance from zeta plane

        Returns:
            float: Conductivity at a point _x away from the zeta plane
        """
        # TODO: Verify that the mobility is a constant (search temperature dependence or etc)..
        assert self.m_kappa_stern is not None, "self.m_kappa_stern is None"
        assert _x <= self.m_xd, "self.m_xd < _x"

        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        _cond = 0.0
        potential = self.m_potential_stern * np.exp((-1.0) * self.m_kappa_stern * _x)
        for _, prop in self.m_ion_props.items():
            _conc = 1000.0 * prop[IonProp.Concentration.name]
            _v = prop[IonProp.Valence.name]
            # TODO: H+とOH-のStern層における移動度がわからないので, とりあえず拡散層の1/2とする
            # 参考文献：doi:10.1029/2008JB006114.
            # TODO?: __calc_cond_at_x_stern_infと__calc_cond_at_x_stern_truncatedの違いはここだけなので, flagで制御したほうがいいかも？
            _mobility = prop[IonProp.MobilityInfDiffuse.name] * 0.5
            _conc = _conc * np.exp((-1.0) * _v * _e * potential / (kb * _t))
            _cond += _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_specific_cond_at_x_stern_inf(self, _x: float) -> float:
        """Calculate specific conductivity in the stern layer at a point _x away
        from the surface for the infinite diffuse layer case.
        Specific conductity is defined as eq.(26) in Leroy and Revil (2004)

        Args:
            _x (float): Distance from the mineral surface

        Returns:
            float: Specific conductivity
        """
        assert self.m_kappa_stern is not None, "self.m_kappa_stern is None"
        assert _x <= self.m_xd, "self.m_xd < _x"
        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        potential = self.m_potential_stern * np.exp((-1.0) * self.m_kappa_stern * _x)
        prop_na: Dict = self.m_ion_props[Species.Na.name]
        _conc = 1000.0 * prop_na[IonProp.Concentration.name]
        _v = prop_na[IonProp.Valence.name]  # 1
        _mobility = prop_na[IonProp.MobilityStern.name]
        _conc = _conc * np.exp((-1.0) * _v * _e * potential / (kb * _t))
        _cond = _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_cond_at_x_stern_truncated(self, _x: float) -> float:
        """Calculate the conductivity of the inter layer.
            The following assumptions are made:
            1. Mobility is assumed to be a constant following Leroy & Revil(2009).

        Args:
            _x (float): Distance from zeta plane

        Returns:
            float: Conductivity at a point _x away from the zeta plane
        """
        # TODO: Verify that the mobility is a constant (search temperature dependence or etc)..
        assert self.m_kappa_stern is not None, "self.m_kappa_stern is None"
        assert _x <= self.m_xd, "self.m_xd < _x"
        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        _cond = 0.0
        potential = self.m_potential_stern * np.exp((-1.0) * self.m_kappa_stern * _x)
        for _, prop in self.m_ion_props.items():
            _conc = 1000.0 * prop[IonProp.Concentration.name]
            _v = prop[IonProp.Valence.name]
            # TODO?: __calc_cond_at_x_stern_infと__calc_cond_at_x_stern_truncatedの違いはここだけなので, flagで制御したほうがいいかも？
            _mobility = prop[IonProp.MobilityTrunDiffuse.name] * 0.5
            _conc = _conc * np.exp((-1.0) * _v * _e * potential / (kb * _t))
            _cond += _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_specific_cond_at_x_stern_truncated(self, _x: float) -> float:
        """Calculate specific conductivity in the stern layer at a point _x away
        from the surface for the truncated diffuse layer case.
        Specific conductity is defined as eq.(26) in Leroy and Revil (2004)

        Args:
            _x (float): Distance from the mineral surface

        Returns:
            float: Specific conductivity
        """
        assert self.m_kappa_stern is not None, "self.m_kappa_stern is None"
        assert _x <= self.m_xd, "self.m_xd < _x"
        _na = const.AVOGADRO_CONST
        _e = const.ELEMENTARY_CHARGE
        kb = const.BOLTZMANN_CONST
        _t = self.m_temperature
        potential = self.m_potential_stern * np.exp((-1.0) * self.m_kappa_stern * _x)
        prop_na: Dict = self.m_ion_props[Species.Na.name]
        _conc = 1000.0 * prop_na[IonProp.Concentration.name]
        _v = prop_na[IonProp.Valence.name]  # 1
        # TODO: H+とOH-のStern層における移動度がわからないので, とりあえず拡散層の1/2とする. 他に方法がないか調べる
        # 参考文献：doi:10.1029/2008JB006114.
        # TODO?: __calc_specific_cond_at_x_stern_infと__calc_specific_cond_at_x_stern_truncatedの違いはここだけなので, flagで制御したほうがいいかも？
        _mobility = prop_na[IonProp.MobilityTrunDiffuse.name] * 0.5
        _conc = _conc * (np.exp((-1.0) * _v * _e * potential / (kb * _t)) - 1.0)
        _cond = _e * abs(_v) * _mobility * _na * _conc
        return _cond

    def __calc_kappa_truncated(self) -> None:
        """Calculate the kappa of the potential (instead of Eq. 11
        of Gonçalvès et al., 2004) when the diffuse layer is truncated
        """
        _r = self.m_layer_width * 0.5
        self.m_kappa_truncated = (
            1.0 / _r * np.log(self.m_potential_zeta / self.m_potential_r)
        )

    def __calc_kappa_stern(self) -> None:
        """Calculate the rate of decay (kappa) of the potential in
        the stern layer. Assume the thickness of the stern layer is
        equal to xd (O layer thickness is assumed negligibly small)
        """
        self.m_kappa_stern = (
            1.0 / self.m_xd * np.log(self.m_potential_stern / self.m_potential_zeta)
        )

    def calc_cond_interlayer(self) -> Tuple[float]:
        """Calculate the Stern + EDL conductivity of the inter layer

        Returns:
            Tuple[float]: conductivity, integral error
        """
        # When the layer thickness is less than 1 nm,, water molecules
        # cannot pass between the layers of smectite (Shirozu, 1998)
        assert self.m_layer_width >= 1.0e-9, "self.m_layer_width < 1.0e-9"
        assert self._check_if_calculated_electrical_params_truncated(), (
            "Before calculating the conductivity of interlayer, we should"
            "obtain electrical parameters for truncated diffuse layer case"
        )

        # TODO: Currently, it is assumed that this function is called only in the case of
        # smectite. If we want to calculate the electrochemical properties of another clay
        # mineral, such as vermiculite, we should modify this function.
        if not self.__check_constant_as_smectite_truncated():
            self.__set_constant_for_smectite_truncated()

        if self.m_xd is None:
            self.calc_xd()
        assert (
            self.m_xd < self.m_layer_width
        ), f"self.m_xd: {self.m_xd} > self.m_layer_width: {self.m_layer_width}"
        if self.m_kappa_truncated is None:
            self.__calc_kappa_truncated()
        if self.m_kappa_stern is None:
            self.__calc_kappa_stern()
        _xdl = self.m_layer_width * 0.5
        cond_ohmic_diffuse, _err1 = quad(
            self.__calc_cond_at_x_truncated_diffuse, self.m_xd, _xdl
        )
        cond_ohmic_stern, _err2 = quad(
            self.__calc_cond_at_x_stern_truncated, 0.0, self.m_xd
        )
        cond = (cond_ohmic_diffuse + cond_ohmic_stern) / _xdl
        self.m_cond_stern_plus_edl = cond
        if self.m_logger is not None:
            self.m_logger.info("Finished the calculation of interlayer conductivity")
            self.m_logger.debug(f"cond_ohmic_diffuse: {cond_ohmic_diffuse}")
            self.m_logger.debug(f"cond_ohmic_stern: {cond_ohmic_diffuse}")
        return cond, _err1 + _err2

    def calc_cond_infdiffuse(self) -> Tuple[float]:
        """Calculate the Stern + EDL conductivity for the inifinite diffuse
         layer case.

        Returns:
            Tuple[float]: conductivity, integral error
        """
        # TODO: May need to consider other cases (e.g., illite, etc.)
        if self.m_qi < 0.0 and self.m_gamma_1 == 0.0:
            self.__set_constant_for_smectite_inf()
        else:
            self.__set_constant_for_kaolinite()
        if not self._check_if_calculated_electrical_params_inf():
            self.calc_potentials_and_charges_inf()
        if self.m_xd is None:
            self.calc_xd()
        if self.m_kappa_stern is None:
            self.__calc_kappa_stern()
        # End of the diffuse layer
        xdl = self.m_xd + 1.0 / self.m_kappa
        cond_ohmic_diffuse, _err1 = quad(
            self.__calc_cond_at_x_inf_diffuse, self.m_xd, xdl
        )
        cond_ohmic_stern, _err2 = quad(self.__calc_cond_at_x_stern_inf, 0.0, self.m_xd)
        # Based on eq.(26) of Leroy & Revil (2004), we assume that Na
        # ions are densely charged on the surface
        cond = (cond_ohmic_diffuse + cond_ohmic_stern) / xdl
        self.m_cond_stern_plus_edl = cond
        if self.m_logger is not None:
            self.m_logger.info(
                "Finished the calculation of infinite diffuse layer conductivity"
            )
            self.m_logger.debug(f"cond_ohmic_diffuse: {cond_ohmic_diffuse}")
            self.m_logger.debug(f"cond_ohmic_stern: {cond_ohmic_diffuse}")
        self.m_cond_infdiffuse = cond
        self.m_double_layer_length = xdl
        return cond, _err1 + _err2

    def calc_specific_surface_cond_inf(self, cond_fluid: float) -> Tuple[float]:
        """Calculate specific surface conductivity of infinite diffuse layer
            Specific surface conductivity is defined as eq.(26) in Leroy and Revil (2004)
            Prepared for the purpose of comparison with Fig. 9, Fig. 10 (a) of Leroy & Revil, 2004

        Args:
            cond_fluid (float): conductivity of the fluid

        Returns:
            Tuple[float]: Specific surface conductivity, integral error
        """
        # Leroy & Revil, 2004のFig.9, Fig10 (a)と比較するために作成した関数
        if not self._check_if_calculated_electrical_params_inf():
            self.calc_potentials_and_charges_inf()
        if self.m_xd is None:
            self.calc_xd()
        if self.m_kappa_stern is None:
            self.__calc_kappa_stern()
        _xdl = self.m_xd + 1.0 / self.m_kappa
        cond_ohmic_diffuse, _err1 = quad(
            self.__calc_cond_at_x_inf_diffuse,
            self.m_xd,
            _xdl,
        )
        cond_ohmic_stern, _err2 = quad(self.__calc_cond_at_x_stern_inf, 0.0, self.m_xd)
        cond_ohmic_fluid = cond_fluid * _xdl
        cond_specific = cond_ohmic_diffuse + cond_ohmic_stern - cond_ohmic_fluid
        # In the dynamic stern layer assumtion, stern layer has surtain
        # mobility (https://doi.org/10.1016/j.jcis.2015.03.047)
        return cond_specific, _err1 + _err2

    def calc_smec_cond_tensor_cube_oxyz(self) -> np.ndarray:
        """Calculate conductivity tensor in smectite with layers aligned
         perpendicular to the z-plane. The T-O-T plane is assumed to be an
         insulator, following Watanabe (2005). The T-O-T plane is the xy-plane,
         and perpendicular to it is the z-axis.

        Returns:
            np.ndarray: 3 rows and 3 columns condutivity tensor
        """
        assert self._check_if_calculated_electrical_params_truncated, (
            "Before calculating the conductivity of the smectite cell, we should"
            "calculate electrical parameters for truncated diffuse layer case"
        )
        sigma_intra = self.m_cond_stern_plus_edl
        assert sigma_intra is not None, (
            "Before calculating the conductivity"
            "of the smectite cell, we should calculate interlayer conductivity"
        )
        sigma_h = sigma_intra * self.m_layer_width / (6.6e-10 + self.m_layer_width)
        sigma_v = 1.0e-12
        cond_tensor = np.array(
            [[sigma_h, 0.0, 0.0], [0.0, sigma_h, 0.0], [0.0, 0.0, sigma_v]],
            dtype=np.float64,
        )
        return cond_tensor

    def calc_kaol_cond_tensor_cube_oxyz(self) -> np.ndarray:
        """Calculate conductivity tensor in kaolinite.

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
        return cond_tensor

    def calc_cond_tensor(self) -> None:
        """Calculate conductivity tensor. Separate cases by smectite and kaolinite.

        Args:
            edge_length (float): Length of one side of a cube cell (unit: m)
        """
        if self.m_qi < 0.0 and self.m_gamma_1 == 0.0:
            tensor = self.calc_smec_cond_tensor_cube_oxyz()
        else:
            tensor = self.calc_kaol_cond_tensor_cube_oxyz()

        self.m_cond_tensor = tensor

        if self.m_logger is not None:
            self.m_logger.info(f"{__name__} cond tensor: {self.m_cond_tensor}")

    def get_logger(self) -> Logger:
        """Getter for the logging.Logger

        Returns:
            Logger: Logger containing debugging information
        """
        return self.m_logger

    def get_cond_tensor(self) -> np.ndarray or None:
        """Getter for the conductivity tensor

        Returns:
            np.ndarray: Conductivity tensor with 3 rows and 3 columns
        """
        if self.m_cond_tensor is not None:
            return deepcopy(self.m_cond_tensor)
        return self.m_cond_tensor

    def get_cond_infdiffuse(self) -> float or None:
        """Getter for the stern potential

        Returns:
            float: Conductivity of the stern layer (Unit: S/m)
        """
        return self.m_cond_infdiffuse

    def get_double_layer_length(self) -> float or None:
        """Getter for the double layer length (surface to the end of the diffuse layer)

        Returns:
            float or None: Length of the electrical double layer
        """
        return self.m_double_layer_length

    def save(self, _pth: str) -> None:
        """Save Phyllosilicate class as pickle

        Args:
            _pth (str): path to save
        """
        with open(_pth, "wb") as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)


# pylint: disable=dangerous-default-value
class Smectite(Phyllosilicate):
    """Inherited class of Phyllosilicate, with surface adsorption site density and layer
    charge fixed to the physical properties of smectite

    Args:
        Phyllosilicate: Phyllosilicate class
    """

    def __init__(
        self,
        nacl: NaCl,
        layer_width: float = 1.3e-9,
        potential_0: float = None,
        potential_stern: float = None,
        potential_zeta: float = None,
        potential_r: float = None,
        charge_0: float = None,
        charge_stern: float = None,
        charge_diffuse: float = None,
        xd: float = None,
        cond_stern_plus_edl: float = None,
        logger: Logger = None,
    ):
        """Inherited classes from Philosilicate. Number density of
            reactors on the surface and fixing the layer charge for
            smectite case.

        Args:
            nacl (NaCl): Instance of NaCl class
            layer_width (float): Distance between sheets of phyllosilicate minerals
                (unit: m). Defaults to 1.3e-9 (When 3 water molecules are trapped).
            potential_0 (float, optional): surface potential (unit: V).
            potential_stern (float, optional): stern plane potential (unit: V).
            potential_zeta (float, optional): zeta plane potential (unit: V).
            potential_r (float, optional): potential at the position truncated inside the inter layer (unit: V).
            charge_0 (float, optional): charges in surface layer (unit: C/m3).
            charge_stern (float, optional): charges in stern layer (unit: C/m3).
            charge_zeta (float, optional): charges in zeta layer (unit: C/m3).
            xd (float, optional): Distance from mineral surface to zeta plane (unit: V).
            cond_stern_plus_edl (float, optional): Conductivity of Stern layer + EDL (unit: S/m).
            logger (Logger): Logger for debugging.
            flag_truncated (bool): True if truncated parameters will be calculate.
        """
        super().__init__(
            nacl=nacl,
            layer_width=layer_width,
            gamma_1=0.0,
            gamma_2=5.5,
            gamma_3=5.5,
            qi=-1.0,
            potential_0=potential_0,
            potential_stern=potential_stern,
            potential_zeta=potential_zeta,
            potential_r=potential_r,
            charge_0=charge_0,
            charge_stern=charge_stern,
            charge_diffuse=charge_diffuse,
            xd=xd,
            cond_stern_plus_edl=cond_stern_plus_edl,
            logger=logger,
        )


# pylint: disable=dangerous-default-value
class Kaolinite(Phyllosilicate):
    """Inherited class of Phyllosilicate, with surface adsorption site density, layer
    charge, and layer width fixed to the physical properties of kaolinite

    Args:
        Phyllosilicate: Phyllosilicate class
    """

    def __init__(
        self,
        nacl: NaCl,
        layer_width: float = 1.0e-10,
        potential_0: float = None,
        potential_stern: float = None,
        potential_zeta: float = None,
        potential_r: float = None,
        charge_0: float = None,
        charge_stern: float = None,
        charge_diffuse: float = None,
        xd: float = None,
        cond_stern_plus_edl: float = None,
        logger: Logger = None,
    ):
        """Inherited classes from Philosilicate. Number density of
            reactors on the surface and fixing the layer charge for
            kaolinite case.

        Args:
            nacl (NaCl): Instance of NaCl class
            layer_width (float): Distance between sheets of phyllosilicate minerals
                (unit: m). Defaults to 1.3e-9 (When 3 water molecules are trapped).
            potential_0 (float, optional): surface potential (unit: V).
            potential_stern (float, optional): stern plane potential (unit: V).
            potential_zeta (float, optional): zeta plane potential (unit: V).
            potential_r (float, optional): potential at the position truncated inside the inter layer (unit: V).
            charge_0 (float, optional): charges in surface layer (unit: C/m3).
            charge_stern (float, optional): charges in stern layer (unit: C/m3).
            charge_zeta (float, optional): charges in zeta layer (unit: C/m3).
            xd (float, optional): Distance from mineral surface to zeta plane (unit: V).
            cond_stern_plus_edl (float, optional): Conductivity of Stern layer + EDL (unit: S/m).
            logger (Logger): Logger for debugging.
        """
        super().__init__(
            nacl=nacl,
            layer_width=layer_width,
            gamma_1=5.5,
            gamma_2=5.5,
            gamma_3=5.5,
            qi=0.0,
            potential_0=potential_0,
            potential_stern=potential_stern,
            potential_zeta=potential_zeta,
            potential_r=potential_r,
            charge_0=charge_0,
            charge_stern=charge_stern,
            charge_diffuse=charge_diffuse,
            xd=xd,
            cond_stern_plus_edl=cond_stern_plus_edl,
            logger=logger,
        )
